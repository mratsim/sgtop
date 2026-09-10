use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CLine};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;

use crate::args::Args;
use crate::derive::{Derived, WIN_LABELS};
use crate::scrape::Shared;
use crate::theme::{usage_color, Theme, THEMES};

/// Minimum terminal size (rows, cols) for the full layout; below either, the
/// compact layout takes over (subtitles collapse first, then graphs).
const FULL_MIN_ROWS: u16 = 26;
const FULL_MIN_COLS: u16 = 90;

pub struct Ui {
    pub theme_idx: usize,
    /// None = auto from terminal size, Some = forced
    pub compact: Option<bool>,
    pub graphs_on: bool,
    pub window_focus: usize,
    pub paused_since: Option<Instant>,
    /// `--insecure` is active: flagged in the header so unverified TLS is visible
    pub insecure: bool,
    pub overlay: Overlay,
    pub scroll: u16,
    pub interval: f64,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Overlay {
    None,
    Help,
    Explain,
}

impl Ui {
    pub fn new(args: &Args) -> Self {
        let theme_idx = THEMES
            .iter()
            .position(|t| t.name == args.theme)
            .unwrap_or(0);
        Self {
            theme_idx,
            compact: if args.compact { Some(true) } else { None },
            graphs_on: true,
            window_focus: 2,
            paused_since: None,
            insecure: args.insecure,
            overlay: Overlay::None,
            scroll: 0,
            interval: args.interval.clamp(0.5, 10.0),
        }
    }

    pub fn theme(&self) -> &'static Theme {
        &THEMES[self.theme_idx]
    }
}

pub fn run(
    terminal: &mut ratatui::DefaultTerminal,
    shared: Arc<Shared>,
    ui: &mut Ui,
) -> anyhow::Result<()> {
    loop {
        if event_ready()? {
            if let crossterm::event::Event::Key(k) = crossterm::event::read()? {
                if k.kind == crossterm::event::KeyEventKind::Press
                    && handle_key(ui, &shared, k.code)
                {
                    return Ok(());
                }
            }
        }

        terminal.draw(|f| draw(f, ui, &shared))?;
    }
}

fn event_ready() -> anyhow::Result<bool> {
    Ok(crossterm::event::poll(Duration::from_millis(120))?)
}

fn handle_key(ui: &mut Ui, shared: &Arc<Shared>, code: crossterm::event::KeyCode) -> bool {
    use crossterm::event::KeyCode::*;
    match code {
        Char('q') | Esc => true,
        Char(' ') => {
            let now_paused = !shared.paused.load(Ordering::Relaxed);
            shared.paused.store(now_paused, Ordering::Relaxed);
            ui.paused_since = now_paused.then_some(Instant::now());
            false
        }
        Char('c') => {
            let auto = ui.compact.unwrap_or_else(auto_compact);
            ui.compact = Some(!auto);
            false
        }
        Char('g') => {
            ui.graphs_on = !ui.graphs_on;
            false
        }
        Char('t') => {
            ui.theme_idx = (ui.theme_idx + 1) % THEMES.len();
            false
        }
        Char(c @ '1'..='3') => {
            ui.window_focus = c.to_digit(10).unwrap_or(3) as usize - 1;
            false
        }
        Char('e') => {
            ui.overlay = if ui.overlay == Overlay::Explain {
                Overlay::None
            } else {
                Overlay::Explain
            };
            ui.scroll = 0;
            false
        }
        Char('?') => {
            ui.overlay = if ui.overlay == Overlay::Help {
                Overlay::None
            } else {
                Overlay::Help
            };
            false
        }
        Char('+') | Char('=') => {
            ui.interval = (ui.interval * 2.0).min(10.0);
            shared
                .interval_ms
                .store((ui.interval * 1000.0) as u64, Ordering::Relaxed);
            false
        }
        Char('-') => {
            ui.interval = (ui.interval / 2.0).max(0.5);
            shared
                .interval_ms
                .store((ui.interval * 1000.0) as u64, Ordering::Relaxed);
            false
        }
        Down | Char('j') => {
            if ui.overlay == Overlay::Explain {
                ui.scroll = ui.scroll.saturating_add(1);
            }
            false
        }
        Up | Char('k') => {
            ui.scroll = ui.scroll.saturating_sub(1);
            false
        }
        _ => false,
    }
}

fn auto_compact() -> bool {
    match crossterm::terminal::size() {
        Ok((cols, rows)) => rows < FULL_MIN_ROWS || cols < FULL_MIN_COLS,
        Err(_) => false,
    }
}

// ---------------------------------------------------------------- rendering

pub fn draw(f: &mut Frame, ui: &Ui, shared: &Arc<Shared>) {
    let t = ui.theme();
    f.render_widget(Block::new().style(Style::new().bg(t.bg)), f.area());

    let history = shared.history.lock().unwrap();
    let d = crate::derive::derive(&history, ui.window_focus);
    let uptime = history.first_seen.map(|t| t.elapsed());
    drop(history);
    let age = shared.last_ok.lock().unwrap().map(|t| t.elapsed());
    let err = shared.last_error.lock().unwrap().clone();

    let compact = ui.compact.unwrap_or_else(|| {
        let a = f.area();
        a.height < FULL_MIN_ROWS || a.width < FULL_MIN_COLS
    });

    let mut constraints: Vec<Constraint> = vec![
        Constraint::Length(1),
        Constraint::Length(if compact { 3 } else { 5 }),
    ];
    let mut graph_i: Option<usize> = None;
    if ui.graphs_on {
        graph_i = Some(constraints.len());
        constraints.push(if compact {
            Constraint::Min(3)
        } else {
            Constraint::Min(7)
        });
    }
    let lat_i = constraints.len();
    constraints.push(Constraint::Length(if compact { 5 } else { 8 }));
    let det_i = constraints.len();
    constraints.push(if compact {
        Constraint::Length(3)
    } else {
        // detail sizes to its content (capped); all spare rows go to the
        // graphs row above, which is the only Min and absorbs the rest
        Constraint::Length(detail_height(d.as_ref(), ui.window_focus))
    });

    let chunks = Layout::vertical(constraints).split(f.area());

    let peaks = *shared.peaks.lock().unwrap();
    draw_header(f, ui, t, chunks[0], d.as_ref(), uptime, age);
    draw_hero(f, ui, t, chunks[1], d.as_ref(), compact);
    if let Some(gi) = graph_i {
        if let Some(d) = &d {
            if compact {
                draw_rate_graph(f, t, chunks[gi], d, RateGraph::Decode, &peaks);
            } else {
                draw_graphs_row(f, t, chunks[gi], d, &peaks);
            }
        }
    }
    draw_latency(f, ui, t, chunks[lat_i], d.as_ref(), compact);
    draw_detail(
        f,
        ui,
        t,
        chunks[det_i],
        d.as_ref(),
        compact,
        &shared.peaks.lock().unwrap(),
    );

    match ui.overlay {
        Overlay::None => {}
        Overlay::Help => draw_help(f, t, f.area()),
        Overlay::Explain => draw_explain(f, ui, t, f.area()),
    }

    if let Some(paused_at) = ui.paused_since {
        let secs = paused_at.elapsed().as_secs();
        banner(
            f,
            f.area(),
            format!("⏸ paused {secs}s — press space to resume"),
            t.warn,
        );
    } else if err.is_some() && (d.is_none() || age_is_stale(age, ui)) {
        let e = err.unwrap_or_default();
        let a = age.map(|a| a.as_secs()).unwrap_or(0);
        banner(
            f,
            f.area(),
            format!("⚠ connection lost — retrying… last data {a}s ago — {e}"),
            t.bad,
        );
    }
}

fn age_is_stale(age: Option<Duration>, ui: &Ui) -> bool {
    match age {
        None => true,
        Some(a) => a > Duration::from_secs_f64(ui.interval * 3.0 + 1.0),
    }
}

fn banner(f: &mut Frame, area: Rect, msg: String, color: ratatui::style::Color) {
    let area = Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(1),
        width: area.width,
        height: 1,
    };
    let line = Line::from(Span::styled(
        msg,
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    ));
    f.render_widget(Paragraph::new(line), area);
}

// ---- header ---------------------------------------------------------------

fn draw_header(
    f: &mut Frame,
    ui: &Ui,
    t: &Theme,
    area: Rect,
    d: Option<&Derived>,
    uptime: Option<Duration>,
    age: Option<Duration>,
) {
    let mut spans = vec![Span::styled(
        " sgtop",
        Style::new().fg(t.accent).add_modifier(Modifier::BOLD),
    )];
    if let Some(d) = d {
        spans.push(Span::styled(
            format!(" · {} ({})", d.model, d.engine),
            Style::new().fg(t.fg),
        ));
        if let Some(up) = uptime {
            spans.push(Span::styled(
                format!(" · up {}", fmt_dur(up)),
                Style::new().fg(t.dim),
            ));
        }
    }
    spans.push(Span::styled(
        format!(" · poll {:.1}s", ui.interval),
        Style::new().fg(t.dim),
    ));
    if ui.insecure {
        spans.push(Span::styled(
            " · \u{26a0} insecure TLS",
            Style::new().fg(t.warn),
        ));
    }
    if let Some(a) = age {
        if a > Duration::from_secs(2) {
            spans.push(Span::styled(
                format!(" · data {:.0}s old", a.as_secs_f64()),
                Style::new().fg(t.warn),
            ));
        }
    }
    spans.push(Span::styled(
        "  [? help · e explain]",
        Style::new().fg(t.dim),
    ));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---- hero strip ------------------------------------------------------------

fn draw_hero(f: &mut Frame, ui: &Ui, t: &Theme, area: Rect, d: Option<&Derived>, compact: bool) {
    let focus = ui.window_focus;
    let block = block(t, "Engine status", None);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // WORDS OUT gets double width so decode and prefill both fit in full
    let cols = Layout::horizontal([
        Constraint::Ratio(1, 7),
        Constraint::Ratio(1, 7),
        Constraint::Ratio(2, 7),
        Constraint::Ratio(1, 7),
        Constraint::Ratio(1, 7),
        Constraint::Ratio(1, 7),
    ])
    .split(inner);

    let mut cells: Vec<(String, Vec<Line<'static>>)> = Vec::new();
    if let Some(d) = d {
        let kv = |label: &str, value: &str, sub: &str, style: Style| {
            (
                label.to_string(),
                vec![
                    Line::from(Span::styled(format!(" {}", value), style)),
                    Line::from(Span::styled(format!(" {}", sub), Style::new().fg(t.dim))),
                ],
            )
        };
        let f_run = Style::new().fg(t.fg).add_modifier(Modifier::BOLD);
        cells.push(kv(
            "RUNNING",
            &fmt_num(d.running),
            "being answered now",
            f_run,
        ));
        let q_style = match d.queue.unwrap_or(0.0) {
            q if q > 20.0 => Style::new().fg(t.bad).add_modifier(Modifier::BOLD),
            q if q > 5.0 => Style::new().fg(t.warn).add_modifier(Modifier::BOLD),
            _ => Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        };
        cells.push(kv("QUEUE", &fmt_num(d.queue), "waiting to start", q_style));
        let rate_line = |tag: &str, instant: Option<f64>, win: &[Option<f64>; 3], color| {
            Line::from(vec![
                Span::styled(format!(" {tag} "), Style::new().fg(t.dim)),
                Span::styled(
                    format!("{:.0} tok/s", instant.unwrap_or(0.0)),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  ·  {}", triple(win)), Style::new().fg(t.dim)),
            ])
        };
        cells.push((
            "WORDS OUT".into(),
            vec![
                rate_line("prefill", d.prefill_instant, &d.prefill_rate, t.s2),
                rate_line("decode", d.decode_instant, &d.decode_rate, t.s1),
            ],
        ));
        // host tier is excluded from the alarm: it idles near 100% by design (LRU-recycled write-through cache); only KV/mamba fullness blocks requests
        let worst = d
            .pools
            .iter()
            .filter(|p| p.name != "host")
            .map(|p| p.usage)
            .fold(0.0_f64, f64::max);
        cells.push(kv(
            "MEMORY POOLS",
            &fmt_pct(Some(worst)),
            &d.pools
                .iter()
                .map(|p| format!("{} {}", p.name, pool_value(p)))
                .collect::<Vec<_>>()
                .join(" · "),
            Style::new()
                .fg(usage_color(t, worst))
                .add_modifier(Modifier::BOLD),
        ));
        let ttft = d.ttft[focus].p95;
        cells.push(kv(
            "FIRST WORD",
            &fmt_secs(ttft),
            "p95 · first token",
            Style::new()
                .fg(latency_color(t, ttft, &d.ttft[focus]))
                .add_modifier(Modifier::BOLD),
        ));
        let s = d.stalls[focus];
        let stall_style = if s.count > 0 {
            Style::new().fg(t.bad).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(t.good)
        };
        cells.push(kv(
            "STALLS",
            &if s.count > 0 {
                format!("{}×", s.count)
            } else {
                "0".into()
            },
            &format!("{:.1}s frozen in {}", s.seconds, WIN_LABELS[focus]),
            stall_style,
        ));
    }

    for (i, cell) in cells.iter().enumerate() {
        if i >= cols.len() {
            break;
        }
        let area = cols[i];
        let label = if compact {
            cell.0.to_lowercase()
        } else {
            cell.0.clone()
        };
        let mut lines = vec![truncate_line(
            Line::from(Span::styled(label, Style::new().fg(t.dim))),
            area.width as usize,
        )];
        for l in &cell.1 {
            lines.push(truncate_line(l.clone(), area.width as usize));
        }
        f.render_widget(Paragraph::new(lines), area);
    }
}

fn latency_color(
    t: &Theme,
    p95: Option<f64>,
    _q: &crate::derive::Quantiles,
) -> ratatui::style::Color {
    match p95 {
        Some(v) if v > 2.0 => t.bad,
        Some(v) if v > 0.5 => t.warn,
        Some(_) => t.good,
        None => t.dim,
    }
}

// ---- graphs -----------------------------------------------------------------

fn draw_graphs_row(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    d: &Derived,
    peaks: &crate::derive::Peaks,
) {
    let cols = Layout::horizontal([
        Constraint::Percentage(29),
        Constraint::Percentage(29),
        Constraint::Percentage(42),
    ])
    .split(area);
    draw_rate_graph(f, t, cols[0], d, RateGraph::Prefill, peaks);
    draw_rate_graph(f, t, cols[1], d, RateGraph::Decode, peaks);
    draw_right_graphs(f, t, cols[2], d, peaks);
}

/// Which rate series a standalone graph shows.
pub enum RateGraph {
    Prefill,
    Decode,
}

/// Calibration gridlines for an observed peak: every multiple of the
/// round step up to and including the first one above the peak, with the
/// scale topped 10% above the highest gridline.
fn grid_for_scale(scale: f64, t: &Theme) -> (Vec<(f64, ratatui::style::Color)>, f64) {
    if scale >= 10.0 {
        let step = 10.0_f64.powf(scale.log10().floor());
        let top_line = (scale / step).ceil() * step;
        let lines: Vec<(f64, ratatui::style::Color)> = (1..=(top_line / step) as i64)
            .map(|m| (m as f64 * step, t.dim))
            .collect();
        let ymax = (top_line * 1.1).max(scale * 1.15).max(10.0);
        (lines, ymax)
    } else {
        (Vec::new(), (scale * 1.15).max(10.0))
    }
}

fn draw_rate_graph(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    d: &Derived,
    which: RateGraph,
    peaks: &crate::derive::Peaks,
) {
    let g = &d.graphs;
    let (vals, instant, color, title, gloss, sub, with_stalls) = match which {
        RateGraph::Prefill => (
            &g.prefill.vals,
            d.prefill_instant,
            t.s2,
            "Prefill",
            "prompt processing",
            Some(Line::from(Span::styled(
                " bursts freeze streams ".to_string(),
                Style::new().fg(t.dim),
            ))),
            false,
        ),
        RateGraph::Decode => (
            &g.decode.vals,
            d.decode_instant,
            t.s1,
            "Decode",
            "token generation",
            Some(Line::from(Span::styled(
                " red ticks are stalls ".to_string(),
                Style::new().fg(t.dim),
            ))),
            true,
        ),
    };
    // scale is the larger of the in-window max and the session peak;
    // grid_for_scale adds headroom above the top gridline
    let vmax = vals.iter().cloned().fold(0.0_f64, f64::max);
    let peak = match which {
        RateGraph::Prefill => peaks.prefill,
        RateGraph::Decode => peaks.decode,
    };
    let scale = vmax.max(peak.unwrap_or(0.0));
    let (ref_lines, ymax) = grid_for_scale(scale, t);
    let title = Line::from(vec![
        Span::styled(
            format!(" {title} "),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("({gloss}) "), Style::new().fg(t.dim)),
        Span::styled(
            format!(
                "\u{2014} now {:.0} · peak {:.0} ",
                instant.unwrap_or(0.0),
                scale
            ),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "\u{25cf} ",
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]);
    let block = block_titled(t, title, sub);
    let inner = block.inner(area);
    f.render_widget(block, area);

    mini_line_graph(
        f,
        inner,
        vals,
        &g.dt,
        ymax,
        color,
        with_stalls.then_some(&g.stall),
        crate::derive::WINDOWS[2].as_secs_f64(),
        &ref_lines,
    );
}

fn draw_right_graphs(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    d: &Derived,
    peaks: &crate::derive::Peaks,
) {
    let rows =
        Layout::vertical([Constraint::Percentage(52), Constraint::Percentage(48)]).split(area);

    let g = &d.graphs;

    // pools history: one lane (own canvas) per active pool
    let pool_colors = [t.s1, t.warn, t.accent, t.s2];
    let mut title_spans = vec![Span::styled(
        " Memory pools",
        Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
    )];
    for (i, p) in d.pools.iter().enumerate() {
        title_spans.push(Span::styled(" \u{b7} ", Style::new().fg(t.dim)));
        title_spans.push(Span::styled(
            format!("{} {}", p.name, pool_value(p)),
            Style::new()
                .fg(pool_colors[i % pool_colors.len()])
                .add_modifier(Modifier::BOLD),
        ));
    }
    title_spans.push(Span::styled(" ", Style::new().fg(t.fg)));
    let pblock = block_titled(
        t,
        Line::from(title_spans),
        Some(Line::from(Span::styled(
            " KV/mamba full = requests wait · host full is normal (it recycles its own old entries) ".to_string(),
            Style::new().fg(t.dim),
        ))),
    );
    let lane_area = pblock.inner(rows[0]);
    f.render_widget(pblock, rows[0]);
    // never more lanes than the area has rows (each lane needs a graph
    // row plus its axis row), or the axis lines of zero-height lanes
    // would smear onto the block border
    let n = d
        .pools
        .len()
        .clamp(1, (lane_area.height as usize / 2).max(1));
    let lanes = Layout::vertical(vec![Constraint::Ratio(1, n as u32); n])
        .spacing(1)
        .split(lane_area);
    for (i, p) in d.pools.iter().take(n).enumerate() {
        let series: &[f64] = match p.name {
            "KV" => &g.pool_kv,
            "mamba" => &g.pool_mamba,
            "host" => &g.pool_host,
            "SWA" => &g.pool_swa,
            _ => continue,
        };
        let color = pool_colors[i % pool_colors.len()];
        // the axis underline gets its own row; the graph keeps the rows above it
        let lane = lanes[i];
        let graph_rect = Rect {
            height: lane.height.saturating_sub(1),
            ..lane
        };
        mini_line_graph(
            f,
            graph_rect,
            series,
            &g.dt,
            1.05,
            color,
            None,
            crate::derive::WINDOWS[2].as_secs_f64(),
            &[],
        );
        // labeled zero-axis underline: each lane is its own 0-100% scale
        let w = lanes[i].width as usize;
        let axis = Line::from(vec![
            Span::styled(
                format!(" {} ", p.name),
                Style::new().fg(color).add_modifier(Modifier::BOLD),
            ),
            // exactly fill the lane width, or the graph's right-edge
            // column peeks out past the underline
            Span::styled(
                "\u{2500}".repeat(w.saturating_sub(p.name.chars().count() + 2)),
                Style::new().fg(color),
            ),
        ]);
        if lanes[i].height >= 1 {
            let axis_rect = Rect {
                x: lanes[i].x,
                y: lanes[i].y + lanes[i].height - 1,
                width: lanes[i].width,
                height: 1,
            };
            f.render_widget(Paragraph::new(axis), axis_rect);
        }
    }

    // speed per stream: total decode rate divided by running requests (inverse of ITL)
    let smax_view = g
        .per_stream
        .iter()
        .filter_map(|v| *v)
        .fold(0.0_f64, f64::max);
    let sscale = smax_view.max(peaks.decode_single.unwrap_or(0.0));
    let (sgrid, smax) = grid_for_scale(sscale, t);
    let title = format!(
        "Speed per stream — now {} · peak {}",
        g.per_stream
            .last()
            .copied()
            .flatten()
            .map(|v| format!("{v:.0} tok/s"))
            .unwrap_or_else(|| "idle".into()),
        peaks
            .decode_single
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "\u{2014}".into())
    );
    let iblock = block_titled(
        t,
        Line::from(Span::styled(
            format!(" {title} "),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        )),
        Some(Line::from(Span::styled(
            " how fast each answer is being written — dips are stalls ".to_string(),
            Style::new().fg(t.dim),
        ))),
    );
    let inner = iblock.inner(rows[1]);
    f.render_widget(iblock, rows[1]);
    let vals: Vec<f64> = g.per_stream.iter().map(|v| v.unwrap_or(0.0)).collect();
    mini_line_graph(
        f,
        inner,
        &vals,
        &g.dt,
        smax,
        t.warn,
        None,
        crate::derive::WINDOWS[2].as_secs_f64(),
        &sgrid,
    );
}

#[allow(clippy::too_many_arguments)]
fn mini_line_graph(
    f: &mut Frame,
    area: Rect,
    vals: &[f64],
    dt: &[f64],
    ymax: f64,
    color: ratatui::style::Color,
    stall: Option<&[bool]>,
    window_secs: f64,
    ref_lines: &[(f64, ratatui::style::Color)],
) {
    if dt.is_empty() || ymax <= 0.0 || window_secs <= 0.0 || area.width == 0 || area.height == 0 {
        return;
    }
    // coordinates are braille dots (2 per char column, 4 per char row);
    // the newest bin is held out to the last dot column so the graph
    // reaches the right border
    let w_dots = (area.width as f64) * 2.0;
    let h_dots = (area.height as f64) * 4.0;
    // fixed wall-clock axis: the window is always `window_secs` wide,
    // newest sample pinned to the right edge
    let span: f64 = dt.iter().sum::<f64>();
    let axis_x = |cum: f64| {
        ((w_dots - 1.0) - (span - cum) / window_secs * (w_dots - 1.0)).clamp(0.0, w_dots - 1.0)
    };
    // each sample holds for its whole scrape interval; fill columns
    // interpolate between interval ends to form a solid histogram
    let mut cols: Vec<(f64, f64)> = Vec::new();
    let mut tops: Vec<(f64, f64)> = Vec::new();
    let mut ticks: Vec<(f64, f64)> = Vec::new();
    let mut x = 0.0;
    for (i, &v) in vals.iter().enumerate() {
        let cx = axis_x(x);
        let h = ((v / ymax) * h_dots * 0.96).min(h_dots * 0.96);
        tops.push((cx, h));
        if stall.is_some_and(|s| s.get(i).copied().unwrap_or(false)) {
            ticks.push((cx, 0.0));
        }
        x += dt.get(i).copied().unwrap_or(1.0);
    }
    for w in tops.windows(2) {
        let (x0, h0) = w[0];
        let (x1, h1) = w[1];
        let mut cx = x0;
        while cx < x1 {
            let t = (cx - x0) / (x1 - x0).max(1.0);
            let h = h0 + (h1 - h0) * t;
            cols.push((cx, h.max(1.0)));
            cx += 1.0;
        }
    }
    // the newest bin is still in progress: hold its value out to the
    // right edge so the graph always touches the present moment
    if let Some((_, h)) = tops.last() {
        let h = h.max(1.0);
        let start = tops.last().unwrap().0;
        for cx in start as i64..=(w_dots as i64 - 1) {
            cols.push((cx as f64, h));
        }
    }
    // dashed gridlines at each multiple of the step with faded labels;
    // the lowest line carries the unit, the rest are bare numbers
    let grid: Vec<(f64, f64)> = ref_lines
        .iter()
        .map(|(v, _)| (*v, (*v / ymax * h_dots * 0.96).clamp(1.0, h_dots - 2.0)))
        .collect();
    let label_style = ref_lines.first().map(|(_, c)| *c);
    let canvas = Canvas::default()
        .marker(Marker::Braille)
        .x_bounds([0.0, w_dots])
        .y_bounds([0.0, h_dots])
        .paint(move |ctx| {
            for (v, ry) in &grid {
                let mut x = 0.0;
                while x < w_dots {
                    ctx.draw(&CLine {
                        x1: x,
                        y1: *ry,
                        x2: (x + 2.0).min(w_dots - 1.0),
                        y2: *ry,
                        color: label_style.unwrap_or(color),
                    });
                    x += 5.0;
                }
                let n = if *v >= 1.0e6 {
                    format!("{:.0}M", v / 1.0e6)
                } else if *v >= 1.0e3 {
                    format!("{:.0}k", v / 1.0e3)
                } else {
                    format!("{v:.0}")
                };
                let label = if *v == grid.iter().map(|(v, _)| *v).fold(f64::MAX, f64::min) {
                    format!("{n} tok/s")
                } else {
                    n
                };
                ctx.print(
                    2.0,
                    ry + 2.0,
                    Span::styled(label, Style::new().fg(label_style.unwrap_or(color))),
                );
            }
            for (cx, h) in &cols {
                ctx.draw(&CLine {
                    x1: *cx,
                    y1: 0.0,
                    x2: *cx,
                    y2: *h,
                    color,
                });
            }
            for (cx, _) in &ticks {
                ctx.draw(&CLine {
                    x1: *cx,
                    y1: 0.0,
                    x2: *cx,
                    y2: (h_dots * 0.08).max(2.0),
                    color: ratatui::style::Color::Red,
                });
            }
        });
    f.render_widget(canvas, area);
}

// ---- latency panel -----------------------------------------------------------

fn draw_latency(f: &mut Frame, ui: &Ui, t: &Theme, area: Rect, d: Option<&Derived>, compact: bool) {
    let block = block(
        t,
        "Latency — how long users wait to see words",
        Some(&lat_subtitle(d)),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    if let Some(d) = d {
        let w = ui.window_focus;
        if !compact {
            lines.push(lat_header(t));
            for (name, lat) in [
                ("time to first token (TTFT)", &d.ttft),
                ("time between tokens (ITL)", &d.itl),
                ("full answer (end-to-end)", &d.e2e),
                ("queue time", &d.queue_time),
            ] {
                lines.push(lat_row(t, name, &lat[w], lat));
            }
        } else {
            for (name, lat) in [
                ("time to first token (TTFT)", &d.ttft),
                ("time between tokens (ITL)", &d.itl),
            ] {
                lines.push(lat_row_compact(t, name, &lat[w]));
            }
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// Subtitle: the percentile legend, plus prompt-length context when known.
fn lat_subtitle(d: Option<&Derived>) -> String {
    let mut s = String::from(
        "p50 typical · p95 1-in-20 · p99 worst · right side: fresh p95 of each window's requests — bigger windows move slower only because they cover more time",
    );
    if let Some(d) = d {
        match (d.prompt_len_p50, d.prompt_len_p95) {
            (Some(p50), Some(p95)) => s.push_str(&format!(
                " · typical prompt {}–{} tok",
                fmt_num(Some(p50)),
                fmt_num(Some(p95))
            )),
            (Some(p50), None) => {
                s.push_str(&format!(" · typical prompt {} tok", fmt_num(Some(p50))))
            }
            _ => {}
        }
    }
    s
}

fn triple(v: &[Option<f64>; 3]) -> String {
    v.iter()
        .map(|x| x.map(|x| format!("{x:.0}")).unwrap_or_else(|| "—".into()))
        .collect::<Vec<_>>()
        .join("/")
}

/// Header for the combined latency table: focused-window percentiles, then
/// p95 across the three windows.
fn lat_header(t: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(" ".repeat(27), Style::new().fg(t.dim)),
        Span::styled(
            format!("{:>7}  {:>7}  {:>7}", "p50", "p95", "p99"),
            Style::new().fg(t.dim).add_modifier(Modifier::BOLD),
        ),
        Span::styled("   │  ", Style::new().fg(t.dim)),
        Span::styled(
            format!(
                "{:>7} {:>7} {:>7}",
                "p95\u{b7}5s", "p95\u{b7}15s", "p95\u{b7}60s"
            ),
            Style::new().fg(t.dim).add_modifier(Modifier::BOLD),
        ),
    ])
}

fn lat_row(
    t: &Theme,
    name: &str,
    q: &crate::derive::Quantiles,
    lat: &crate::derive::LatencyTriple,
) -> Line<'static> {
    let fmt = |v: Option<f64>| fmt_secs(v);
    let wins = [fmt(lat[0].p95), fmt(lat[1].p95), fmt(lat[2].p95)];
    Line::from(vec![
        Span::styled(format!(" {name:<26}"), Style::new().fg(t.fg)),
        Span::styled(
            format!("{:>7}  ", fmt(q.p50)),
            Style::new().fg(latency_color(t, q.p50, q)),
        ),
        Span::styled(
            format!("{:>7}  ", fmt(q.p95)),
            Style::new().fg(latency_color(t, q.p95, q)),
        ),
        Span::styled(
            format!("{:>7}", fmt(q.p99)),
            Style::new().fg(latency_color(t, q.p99, q)),
        ),
        Span::styled("   │  ", Style::new().fg(t.dim)),
        Span::styled(
            format!("{:>7} {:>7} {:>7}", wins[0], wins[1], wins[2]),
            Style::new().fg(t.fg),
        ),
    ])
}

fn lat_row_compact(t: &Theme, name: &str, q: &crate::derive::Quantiles) -> Line<'static> {
    let fmt = |v: Option<f64>| fmt_secs(v);
    Line::from(vec![
        Span::styled(format!(" {name:<26}"), Style::new().fg(t.fg)),
        Span::styled(
            format!("p50 {:>7}  ", fmt(q.p50)),
            Style::new().fg(latency_color(t, q.p50, q)),
        ),
        Span::styled(
            format!("p95 {:>7}  ", fmt(q.p95)),
            Style::new().fg(latency_color(t, q.p95, q)),
        ),
        Span::styled(
            format!("p99 {:>7}", fmt(q.p99)),
            Style::new().fg(latency_color(t, q.p99, q)),
        ),
    ])
}

// ---- detail ------------------------------------------------------------------

fn queues_col_lines(d: Option<&Derived>, t: &Theme) -> Vec<Line<'static>> {
    let mut l: Vec<Line> = Vec::new();
    if let Some(d) = d {
        let sq = &d.subqueues;
        let val = |i: usize| -> String {
            sq.get(i)
                .map(|(_, v)| fmt_num(*v))
                .unwrap_or_else(|| "\u{2014}".into())
        };
        let group = |label: &str, a: &str, b: &str| {
            vec![
                Line::from(Span::styled(format!(" {label}"), Style::new().fg(t.dim))),
                Line::from(Span::styled(format!("  {a} · {b}"), Style::new().fg(t.fg))),
            ]
        };
        l.extend(group(
            "prefill queues:",
            &format!("bootstrap {}", val(0)),
            &format!("inflight {}", val(1)),
        ));
        l.extend(group(
            "decode queues:",
            &format!("prealloc {}", val(2)),
            &format!("transfer {}", val(3)),
        ));
        if !val(4).is_empty() {
            l.push(Line::from(Span::styled(
                format!(" grammar: {}", val(4)),
                Style::new().fg(t.fg),
            )));
        }
    }
    l
}

fn quality_col_kvs(d: Option<&Derived>) -> Vec<Kv> {
    let mut q: Vec<Kv> = Vec::new();
    if let Some(d) = d {
        q.push(Kv::plain(
            "cache hit",
            fmt_pct(d.cache_hit),
            "how much of each new prompt it already remembers — high = fast starts",
        ));
        q.push(Kv::plain(
            "spec accept",
            fmt_pct(d.spec_accept),
            "how often it guesses its own next words right — high = feels faster",
        ));
        if let Some(l) = d.spec_accept_len {
            q.push(Kv::plain(
                "spec length",
                format!("{l:.2}"),
                "words gained per guess, on average",
            ));
        }
        q.push(Kv::plain(
            "l2 dev\u{b7}host",
            format!("{} · {}", fmt_pct(d.l2_device), fmt_pct(d.l2_host)),
            "prompt portions served from GPU / host RAM",
        ));
        q.push(Kv::plain(
            "l2 wb\u{b7}rb",
            format!("{} · {}", fmt_num(d.l2_wb), fmt_num(d.l2_rb)),
            "tokens/s written to / read back from host tier",
        ));
        q.push(Kv::plain(
            "gen depth",
            format!("{} tok", fmt_num(d.gen_progress)),
            "how far into their answers the current requests are",
        ));
        if let Some(r) = d.new_token_ratio {
            q.push(Kv::plain(
                "new-token ratio",
                format!("{r:.2}"),
                "scheduler policy knob (1.0 = default)",
            ));
        }
    }
    q
}

fn health_col_kvs(d: Option<&Derived>, focus: usize, t: &Theme) -> Vec<Kv> {
    let mut h: Vec<Kv> = Vec::new();
    if let Some(d) = d {
        let s = d.stalls[focus];
        let (sc, scol) = if s.count > 0 {
            (
                format!(
                    "{} × {:.1}s in {}",
                    s.count,
                    s.seconds / s.count.max(1) as f64,
                    WIN_LABELS[focus]
                ),
                t.bad,
            )
        } else {
            ("0".into(), t.good)
        };
        h.push(Kv::colored(
            "stalls",
            sc,
            "moments when prefill work froze every stream",
            scol,
        ));
        h.push(Kv::colored(
            "evictions/s",
            fmt_num(d.evict_rate),
            "device KV cache slots freed to make room",
            trouble_color(t, d.evict_rate),
        ));
        h.push(Kv::colored(
            "retractions/s",
            fmt_num(d.retract_rate),
            "requests restarted mid-answer — the worst kind",
            trouble_color(t, d.retract_rate),
        ));
        h.push(Kv::colored(
            "503/s",
            fmt_num(d.http_503_rate),
            "requests refused outright",
            trouble_color(t, d.http_503_rate),
        ));
        h.push(Kv::colored(
            "l2 drop/s",
            fmt_num(d.l2_drop),
            "device tokens destroyed without a host backup",
            trouble_color(t, d.l2_drop),
        ));
        h.push(Kv::plain(
            "http active",
            fmt_num(d.http_active),
            "connections open right now",
        ));
        h.push(Kv::plain(
            "cpu cores/s",
            format!(
                "{:.1}/{:.1}/{:.1}",
                d.cpu_tokenizer.unwrap_or(0.0),
                d.cpu_detokenizer.unwrap_or(0.0),
                d.cpu_scheduler.unwrap_or(0.0)
            ),
            "",
        ));
        h.push(Kv::gloss_only(
            "tokenizer · detokenizer · scheduler".to_string(),
        ));
    }
    h
}

fn peaks_col_lines(
    d: Option<&Derived>,
    peaks: &crate::derive::Peaks,
    t: &Theme,
) -> Vec<Line<'static>> {
    let pk = |label: &str, v: Option<f64>| {
        Line::from(vec![
            Span::styled(format!(" {label:<9}"), Style::new().fg(t.dim)),
            Span::styled(
                v.map(|v| format!("{v:.0} tok/s"))
                    .unwrap_or_else(|| "\u{2014}".into()),
                Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
            ),
        ])
    };
    let mut l = vec![
        pk("decode", peaks.decode),
        pk("prefill", peaks.prefill),
        pk("single", peaks.decode_single),
    ];
    l.push(Line::from(vec![
        Span::styled(" engine   ".to_string(), Style::new().fg(t.dim)),
        Span::styled(
            d.map(|d| tok_gauge(d.gen_throughput_gauge))
                .unwrap_or_else(|| "\u{2014}".into()),
            Style::new().fg(t.fg),
        ),
    ]));
    l
}

/// Rows the detail row needs (content + borders), bounded to [9, 14].
fn detail_height(d: Option<&Derived>, focus: usize) -> u16 {
    let t = &THEMES[0];
    let n = [
        queues_col_lines(d, t).len(),
        quality_col_kvs(d).len(),
        health_col_kvs(d, focus, t).len(),
        peaks_col_lines(d, &crate::derive::Peaks::default(), t).len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    (n as u16 + 2).clamp(9, 14)
}

fn draw_detail(
    f: &mut Frame,
    ui: &Ui,
    t: &Theme,
    area: Rect,
    d: Option<&Derived>,
    compact: bool,
    peaks: &crate::derive::Peaks,
) {
    let focus = ui.window_focus;
    if compact {
        let mut text = String::new();
        if let Some(d) = d {
            let sq: Vec<String> = d
                .subqueues
                .iter()
                .map(|(n, v)| format!("{} {}", n, fmt_num(*v)))
                .collect();
            text.push_str(&format!(
                "queues: {} · cache hit {} · spec accept {} · gen depth {} tok",
                sq.join(" · "),
                fmt_pct(d.cache_hit),
                fmt_pct(d.spec_accept),
                fmt_num(d.gen_progress)
            ));
            text.push('\n');
            let s = d.stalls[focus];
            text.push_str(&format!(
                "stalls {} × {:.1}s in {} · evict/s {} · retract/s {} · 503/s {} · active {}",
                s.count,
                if s.count > 0 {
                    s.seconds / s.count.max(1) as f64
                } else {
                    0.0
                },
                WIN_LABELS[focus],
                fmt_num(d.evict_rate),
                fmt_num(d.retract_rate),
                fmt_num(d.http_503_rate),
                fmt_num(d.http_active)
            ));
        }
        let b = block(t, "Queues & health", None);
        f.render_widget(
            Paragraph::new(text).style(Style::new().fg(t.fg)),
            b.inner(area),
        );
        f.render_widget(b, area);
        return;
    }

    let cols = Layout::horizontal([
        Constraint::Percentage(24),
        Constraint::Percentage(25),
        Constraint::Percentage(28),
        Constraint::Percentage(23),
    ])
    .split(area);

    let b1 = block(t, "Waiting lines", Some("requests staged before running"));
    let inner1 = b1.inner(cols[0]);
    let l1 = queues_col_lines(d, t)
        .into_iter()
        .map(|l| truncate_line(l, inner1.width as usize))
        .collect::<Vec<Line>>();
    f.render_widget(Paragraph::new(l1), inner1);
    f.render_widget(b1, cols[0]);

    let b2 = block(
        t,
        "Answer quality",
        Some("what makes the model feel fast or smart"),
    );
    let q = quality_col_kvs(d);
    let inner2 = b2.inner(cols[1]);
    f.render_widget(
        Paragraph::new(kv_lines(t, q, inner2.width as usize)),
        inner2,
    );
    f.render_widget(b2, cols[1]);

    let b3 = block(
        t,
        "Health & trouble",
        Some("anything here that is not zero deserves a look"),
    );
    let h = health_col_kvs(d, focus, t);
    let inner3 = b3.inner(cols[2]);
    f.render_widget(
        Paragraph::new(kv_lines(t, h, inner3.width as usize)),
        inner3,
    );
    f.render_widget(b3, cols[2]);

    let b4 = block(t, "Peaks", Some("since sgtop started"));
    let inner4 = b4.inner(cols[3]);
    let l4 = peaks_col_lines(d, peaks, t)
        .into_iter()
        .map(|l| truncate_line(l, inner4.width as usize))
        .collect::<Vec<Line>>();
    f.render_widget(Paragraph::new(l4), inner4);
    f.render_widget(b4, cols[3]);
}

/// Truncate a styled line to `width` chars, appending `…` when cut.
fn truncate_line(line: Line<'static>, width: usize) -> Line<'static> {
    if width == 0 {
        return Line::default();
    }
    let mut spans: Vec<Span> = Vec::new();
    let mut remaining = width;
    for span in line.spans {
        if remaining == 0 {
            break;
        }
        let len = span.content.chars().count();
        if len <= remaining {
            remaining -= len;
            spans.push(span);
        } else {
            let cut: String = span
                .content
                .chars()
                .take(remaining.saturating_sub(1))
                .collect();
            spans.push(Span::styled(format!("{cut}\u{2026}"), span.style));
            remaining = 0;
        }
    }
    Line::from(spans)
}

/// One aligned key-value row spec for [`kv_lines`].
struct Kv {
    k: String,
    v: String,
    gloss: String,
    style: Option<Style>,
}

impl Kv {
    fn plain(k: &str, v: String, gloss: &str) -> Self {
        Self {
            k: k.into(),
            v,
            gloss: gloss.into(),
            style: None,
        }
    }
    fn colored(k: &str, v: String, gloss: &str, style: ratatui::style::Color) -> Self {
        Self {
            k: k.into(),
            v,
            gloss: gloss.into(),
            style: Some(Style::new().fg(style)),
        }
    }
    /// A dim full-width line (used for the CPU column legend).
    fn gloss_only(gloss: String) -> Self {
        Self {
            k: String::new(),
            v: String::new(),
            gloss,
            style: None,
        }
    }
}

/// Render kv entries with the value column sized to the widest value, so
/// narrow panels don't waste columns on padding.
fn kv_lines(t: &Theme, entries: Vec<Kv>, width: usize) -> Vec<Line<'static>> {
    let kpad = entries
        .iter()
        .map(|e| e.k.chars().count())
        .max()
        .unwrap_or(0)
        .max(8);
    let vpad = entries
        .iter()
        .map(|e| e.v.chars().count())
        .max()
        .unwrap_or(1);
    entries
        .into_iter()
        .map(|e| {
            if e.k.is_empty() && e.v.is_empty() {
                return truncate_line(
                    Line::from(Span::styled(
                        format!(" {}", e.gloss),
                        Style::new().fg(t.dim),
                    )),
                    width,
                );
            }
            let value_style = e
                .style
                .unwrap_or(Style::new().fg(t.fg))
                .add_modifier(Modifier::BOLD);
            truncate_line(
                Line::from(vec![
                    Span::styled(format!(" {:<kpad$}  ", e.k), Style::new().fg(t.fg)),
                    Span::styled(format!("{:<vpad$}  ", e.v), value_style),
                    Span::styled(e.gloss, Style::new().fg(t.dim)),
                ]),
                width,
            )
        })
        .collect()
}

fn trouble_color(t: &Theme, v: Option<f64>) -> ratatui::style::Color {
    if v.unwrap_or(0.0) > 0.0 {
        t.bad
    } else {
        t.good
    }
}

// ---- help & explain overlays ---------------------------------------------

fn draw_help(f: &mut Frame, t: &Theme, area: Rect) {
    let w = 52.min(area.width);
    let h = 16.min(area.height);
    let x = area.x + (area.width - w) / 2;
    let y = area.y + (area.height - h) / 2;
    let area = Rect {
        x,
        y,
        width: w,
        height: h,
    };

    let rows = [
        ("q / Esc", "quit"),
        ("space", "pause scraping (for staring & screenshots)"),
        ("c", "toggle compact / full layout"),
        ("g", "toggle graphs"),
        ("1 2 3", "focus the 5s / 15s / 60s window"),
        ("t", "cycle theme"),
        ("e", "plain-language tour of every panel"),
        ("+ / -", "poll interval (faster / slower)"),
        ("↑ ↓", "scroll the explain screen"),
        ("?", "this keymap"),
    ];
    let mut lines: Vec<Line> = Vec::new();
    for (k, v) in rows {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:<9}", k),
                Style::new().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(v.to_string(), Style::new().fg(t.fg)),
        ]));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.accent))
        .style(Style::new().bg(t.bg))
        .title(Span::styled(
            " keymap ",
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_explain(f: &mut Frame, ui: &Ui, t: &Theme, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.accent))
        .style(Style::new().bg(t.bg))
        .title(Span::styled(
            " What am I looking at?  (e or q to close · ↑↓ scroll) ",
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(ratatui::widgets::Clear, area);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    let section = |title: &str, body: &str, lines: &mut Vec<Line>| {
        lines.push(Line::from(Span::styled(
            format!(" {title}"),
            Style::new().fg(t.accent).add_modifier(Modifier::BOLD),
        )));
        for para in body.split("\n\n") {
            for seg in wrap_text(para, (inner.width as usize).saturating_sub(4).max(20)) {
                lines.push(Line::from(Span::styled(
                    format!(" {seg}"),
                    Style::new().fg(t.fg),
                )));
            }
            lines.push(Line::default());
        }
    };
    section("A token", "A token is roughly a word-piece: the model reads your prompt and writes its answer one token at a time, several times per second. \"tok/s\" below is how many of those pieces appear per second.", &mut lines);
    section("Words out / prefill", "Words out (decode) is the model writing its answer — the number users experience as speed. Prefill is the model reading a new prompt; it happens in bursts and can momentarily freeze everyone else's stream (the seesaw in the graph).", &mut lines);
    section("Time to first token (TTFT)", "When you send a message, this is the pause before the first word appears. Small is good. It grows when the model is busy reading long prompts or when there is a line of requests waiting.", &mut lines);
    section("Time between tokens (ITL) & speed per stream", "Once words are flowing, ITL is the rhythm: the time between one word and the next. Steady and small means the text streams smoothly; spikes are why text sometimes freezes then jumps — usually a new request's prompt being processed in the middle of your answer.\n\nThe Speed-per-stream graph shows the same story from the other side: total writing speed divided by how many answers are being written at once. When it dips while the seesaw graph shows a prefill spike, everyone's stream briefly froze.", &mut lines);
    section("p50 / p95 / p99", "p50 is a typical request. p95 is the experience of the unluckiest 1-in-20. p99 is the worst moments. If p50 is fine but p95 is bad, most people are happy but some are having a bad time — that gap is the number to watch.\n\nThe 5s/15s/60s columns are a fresh p95 of just that window's requests — nothing is averaged with the past, so a single slow request shows up in the 5s column immediately. Bigger windows move slower only because they cover more requests.", &mut lines);
    section("Waiting line (queue)", "Requests that arrived but have not started. A short, spiky line is normal. A tall, flat line means the server is overloaded and everyone's wait grows.", &mut lines);
    section("Peaks", "The Peak box holds the highest rates seen since sgtop started: the busiest decode moment, the biggest prefill burst, and single — the fastest one stream has ever moved (total decode speed divided by how many requests were sharing it).", &mut lines);
    section("Memory pools", "The model keeps working memory for every conversation in progress (KV), and optionally for alternative memory types (mamba, SWA). At 100% a pool is full: new requests wait, and the server may throw out or restart old ones (see evictions and retractions).\n\nThe host tier is the exception: it is a large CPU-RAM cache of prompt prefixes that stays pinned near 100% in normal operation and recycles the oldest entries when it needs room. A full host tier is not a problem — only KV or mamba running full is. The evictions/s counter belongs to the device KV cache, not the host tier.", &mut lines);
    section("Stalls", "A stall is a moment when every stream froze because a prompt was being read. Measured as how many times it happened and how many seconds in total over the window.", &mut lines);
    section("Cache hit & the L2 tiers", "How much of each new prompt the server already remembers from earlier turns. High cache hit = fast starts and less work. A sudden drop usually means a restart or very different traffic.\n\nLarge servers often keep a second, bigger cache tier in host RAM (the \"host\" pool). The l2 dev\u{b7}host line shows how much of the recent prefill work was served from the GPU-resident tier vs the host tier; wb\u{b7}rb is how many tokens per second are being written down to, or read back from, that tier. l2 drop/s counts device tokens destroyed without a host backup \u{2014} work the host tier failed to save; it should be zero.", &mut lines);
    section("Speculative accept", "The model tries to guess several words ahead and checks them in one go. Accept rate is how often the guesses are right; accept length is how many words each lucky guess saves. High numbers make everything feel faster.", &mut lines);
    section("Retractions & evictions", "A retraction means a running request was abandoned and restarted — its user watched their answer reset. Evictions free device KV cache slots to make room; the freed prefixes may already be backed up to the host tier, so steady recycling is normal while a spike means memory pressure. l2 drop/s is the harmful one: device tokens destroyed without a host backup, work truly lost.", &mut lines);

    f.render_widget(Paragraph::new(lines).scroll((ui.scroll, 0)), inner);
}

fn wrap_text(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in s.split('\n') {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if !line.is_empty() && line.len() + word.len() + 1 > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        out.push(line);
    }
    out
}

// ---- shared small helpers -------------------------------------------------

fn block<'a>(t: &Theme, title: &str, subtitle: Option<&str>) -> Block<'a> {
    block_titled(
        t,
        Line::from(Span::styled(
            format!(" {title} "),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        )),
        subtitle.map(|s| Line::from(Span::styled(format!(" {s} "), Style::new().fg(t.dim)))),
    )
}

fn block_titled<'a>(t: &Theme, title: Line<'static>, subtitle: Option<Line<'static>>) -> Block<'a> {
    let mut b = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.dim))
        .style(Style::new().bg(t.bg))
        .title(title);
    if let Some(sub) = subtitle {
        b = b.title_bottom(sub);
    }
    b
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

fn fmt_num(v: Option<f64>) -> String {
    v.map(|v| {
        if v.abs() >= 1000.0 {
            format!("{:.1}k", v / 1000.0)
        } else {
            format!("{v:.0}")
        }
    })
    .unwrap_or_else(|| "—".into())
}

fn fmt_pct(v: Option<f64>) -> String {
    v.map(|v| format!("{:.0}%", v * 100.0))
        .unwrap_or_else(|| "—".into())
}

/// 115904 -> "116k", 2843392 -> "2.8M"
pub fn fmt_tokens(v: f64) -> String {
    if v >= 1.0e6 {
        format!("{:.1}M", v / 1.0e6)
    } else if v >= 1.0e3 {
        format!("{:.0}k", v / 1.0e3)
    } else {
        format!("{v:.0}")
    }
}

/// "20% (116k / 1.3M)" when counts are known, else just the percentage.
fn pool_value(p: &crate::derive::Pool) -> String {
    match (p.used, p.total) {
        (Some(u), Some(t)) if t > 0.0 => {
            let counts = format!("{} / {}", fmt_tokens(u), fmt_tokens(t));
            if p.unit == "slots" {
                format!("{} ({} slots)", fmt_pct(Some(p.usage)), counts)
            } else {
                format!("{} ({})", fmt_pct(Some(p.usage)), counts)
            }
        }
        _ => fmt_pct(Some(p.usage)),
    }
}

fn fmt_secs(v: Option<f64>) -> String {
    match v {
        Some(v) if v >= 100.0 => format!("{v:.0}s"),
        Some(v) if v >= 10.0 => format!("{v:.1}s"),
        Some(v) => format!("{:.0}ms", v * 1000.0),
        None => "—".into(),
    }
}

fn tok_gauge(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into())
}
