use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use sgtop::history::History;
use sgtop::metrics::parse;
use sgtop::scrape::Shared;
use sgtop::ui::{draw, Overlay, Ui};

fn shared_with_fixture() -> Arc<Shared> {
    let body = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/live.txt"
    ))
    .unwrap();
    let shared = Shared {
        history: Mutex::new(History::default()),
        last_ok: Mutex::new(Some(Instant::now())),
        last_error: Mutex::new(None),
        paused: AtomicBool::new(false),
        interval_ms: std::sync::atomic::AtomicU64::new(1000),
        peaks: Mutex::new(sgtop::derive::Peaks {
            decode: Some(402.0),
            prefill: Some(812.0),
            decode_single: Some(134.0),
        }),
    };
    let mut h = shared.history.lock().unwrap();
    // two samples one scrape apart, the second with a distinct decode
    // reading, so the graphs draw a real interval instead of a degenerate
    // zero-duration, zero-delta sample
    let bumped = body.replace(
        "mode=\"decode\",model_name=\"glm-5.3-flash\",moe_ep_rank=\"0\",pp_rank=\"0\",tp_rank=\"0\"} 114659.0",
        "mode=\"decode\",model_name=\"glm-5.3-flash\",moe_ep_rank=\"0\",pp_rank=\"0\",tp_rank=\"0\"} 115059.0",
    );
    h.push(Instant::now(), parse(&body).unwrap());
    h.push(
        Instant::now() + std::time::Duration::from_secs(1),
        parse(&bumped).unwrap(),
    );
    drop(h);
    Arc::new(shared)
}

// History seeded from explicit payloads one scrape apart, for tests
// needing a custom latest sample.
fn shared_with_bodies(bodies: &[String]) -> Arc<Shared> {
    let shared = Shared {
        history: Mutex::new(History::default()),
        last_ok: Mutex::new(Some(Instant::now())),
        last_error: Mutex::new(None),
        paused: AtomicBool::new(false),
        interval_ms: std::sync::atomic::AtomicU64::new(1000),
        peaks: Mutex::new(sgtop::derive::Peaks::default()),
    };
    let mut h = shared.history.lock().unwrap();
    let t0 = Instant::now();
    for (i, body) in bodies.iter().enumerate() {
        h.push(
            t0 + std::time::Duration::from_secs(i as u64),
            parse(body).unwrap(),
        );
    }
    drop(h);
    Arc::new(shared)
}

fn render_at(cols: u16, rows: u16, overlay: Overlay) -> String {
    let shared = shared_with_fixture();
    let backend = TestBackend::new(cols, rows);
    let mut terminal = Terminal::new(backend).unwrap();
    let ui = Ui {
        overlay,
        ..ui_default()
    };
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..rows {
        for col in 0..cols {
            let cell = &terminal.backend().buffer()[(col, row)];
            out.push(cell.symbol().chars().next().unwrap_or(' '));
        }
        out.push('\n');
    }
    out
}

fn ui_default() -> Ui {
    Ui::new(&sgtop::args::Args {
        url: "http://localhost:30000".into(),
        api_key: None,
        insecure: false,
        interval: 1.0,
        theme: "gruvbox".into(),
        compact: false,
        once: false,
    })
}

#[test]
fn full_layout_renders_key_panels() {
    // 120x30
    let out = render_at(120, 30, Overlay::None);
    assert!(out.contains("sgtop"), "header missing");
    assert!(out.contains("glm-5.3-flash"), "model missing");
    assert!(out.contains("RUNNING"), "hero missing");
    assert!(out.contains("Prefill"), "prefill graph missing");
    assert!(out.contains("Decode"), "decode graph missing");
    assert!(out.contains("Speed per stream"), "per-stream graph missing");
    assert!(out.contains("Memory pools"), "pools missing");
    assert!(out.contains("KV"), "KV pool missing");
    assert!(out.contains("mamba"), "mamba pool missing");
    assert!(out.contains("Latency"), "latency panel missing");
    assert!(out.contains("p50"), "percentiles missing");
    assert!(out.contains("Health & trouble"), "health panel missing");
    assert!(out.contains("Peaks"), "peaks panel missing");
    assert!(out.contains("402 tok/s"), "peak decode missing");
}

#[test]
fn compact_layout_renders_at_quarter_screen() {
    // 96x14
    let out = render_at(96, 14, Overlay::None);
    assert!(out.contains("sgtop"));
    assert!(out.contains("RUNNING") || out.contains("running"));
    // compact collapses the subtitle lines but keeps hero + graph + latency
    assert!(out.contains("p50"));
}

#[test]
fn overlays_render() {
    let help = render_at(120, 30, Overlay::Help);
    assert!(help.contains("keymap"));
    assert!(help.contains("pause scraping"));
    let explain = render_at(120, 40, Overlay::Explain);
    assert!(explain.contains("What am I looking at?"));
    assert!(explain.contains("Time to first token"));
    assert!(explain.contains("p50 is a typical request"));
}

#[test]
fn tiny_terminal_does_not_panic() {
    let _ = render_at(40, 8, Overlay::None);
    let _ = render_at(20, 4, Overlay::None);
}

#[test]
fn insecure_tls_hint_renders_only_when_enabled() {
    let out = render_at(120, 30, Overlay::None);
    assert!(
        !out.contains("insecure TLS"),
        "hint shown without --insecure"
    );

    let shared = shared_with_fixture();
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    let ui = Ui {
        insecure: true,
        ..ui_default()
    };
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..120 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(
        out.contains("insecure TLS"),
        "hint missing with --insecure:\n{out}"
    );
}

#[test]
fn debug_print_full() {
    let out = render_at(120, 30, Overlay::None);
    println!("=====\n{out}\n=====");
}

#[test]
fn error_banner_renders_when_stale() {
    let shared = shared_with_fixture();
    *shared.last_error.lock().unwrap() = Some("scrape failed: connection refused".into());
    *shared.last_ok.lock().unwrap() = Some(Instant::now() - std::time::Duration::from_secs(30));
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    let ui = ui_default();
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..120 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(out.contains("connection lost"), "banner missing:\n{out}");
    assert!(out.contains("30s ago"));
}

#[test]
fn narrow_full_layout_truncates_with_ellipsis() {
    let shared = shared_with_fixture();
    let backend = TestBackend::new(76, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    // force full layout below the auto threshold to exercise the panels
    let ui = Ui {
        compact: Some(false),
        ..ui_default()
    };
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..76 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(
        out.contains('\u{2026}'),
        "expected truncated lines with ellipsis:\n{out}"
    );
    // no line may overflow its panel border (wrap would have pushed content down)
    assert!(out.contains("Answer quality"));
    assert!(out.contains("Health & trouble"));
}

#[test]
fn debug_print_wide() {
    let out = render_at(230, 45, Overlay::None);
    println!("=====\n{out}\n=====");
}

// A non-finite gauge reading is no data: the UI renders the missing marker,
// never a "NaN" percentage or a maximal-looking bar.
#[test]
fn non_finite_gauge_reading_renders_as_no_data() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/live.txt"
    ))
    .unwrap();
    for reading in ["NaN", "+Inf", "-Inf"] {
        // newest scrape carries a non-finite cache-hit reading
        // where the fixture has a finite one
        let stale = format!(
            "{}\nsglang:cache_hit_rate{{engine_type=\"unified\",model_name=\"glm-5.3-flash\",\
             moe_ep_rank=\"0\",pp_rank=\"0\",tp_rank=\"0\"}} {reading}\n",
            fixture
        );
        let shared = shared_with_bodies(&[fixture.clone(), stale]);
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, &ui_default(), &shared)).unwrap();
        let mut out = String::new();
        for row in 0..30 {
            for col in 0..120 {
                out.push(
                    terminal.backend().buffer()[(col, row)]
                        .symbol()
                        .chars()
                        .next()
                        .unwrap_or(' '),
                );
            }
            out.push('\n');
        }
        assert!(
            !out.contains("NaN"),
            "a {reading} reading must render as no data, got:\n{out}"
        );
    }
}

// The very first frame, before any scrape has landed, shows a stable
// zero state: the panels and their placeholders draw without panic,
// no graph is plotted from an empty ring, and no fabricated numbers
// appear. Two consecutive frames must render identically.
#[test]
fn cold_start_frame_renders_a_stable_zero_state() {
    let shared_for = || {
        Arc::new(Shared {
            history: Mutex::new(History::default()),
            last_ok: Mutex::new(None),
            last_error: Mutex::new(None),
            paused: AtomicBool::new(false),
            interval_ms: std::sync::atomic::AtomicU64::new(1000),
            peaks: Mutex::new(sgtop::derive::Peaks::default()),
        })
    };
    let render = |cols: u16, rows: u16, shared: &Arc<Shared>| {
        let backend = TestBackend::new(cols, rows);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, &ui_default(), shared)).unwrap();
        let mut out = String::new();
        for row in 0..rows {
            for col in 0..cols {
                out.push(
                    terminal.backend().buffer()[(col, row)]
                        .symbol()
                        .chars()
                        .next()
                        .unwrap_or(' '),
                );
            }
            out.push('\n');
        }
        out
    };

    let shared = shared_for();
    let first = render(120, 30, &shared);
    let again = render(120, 30, &shared_for());
    assert_eq!(
        first, again,
        "the zero state must render identically every frame"
    );

    assert!(first.contains("sgtop"), "header missing:\n{first}");
    assert!(first.contains("Engine status"), "hero frame missing");
    assert!(first.contains("Latency"), "latency panel missing");
    assert!(first.contains("Peaks"), "peaks panel missing");
    // the peaks panel shows the missing marker for every zero value
    assert!(
        first.contains("decode   —"),
        "peaks placeholders missing:\n{first}"
    );
    // an empty ring plots no graphs and shows no hero numbers
    assert!(!first.contains("Prefill"), "a graph rendered from no data");
    assert!(!first.contains("Decode"), "a graph rendered from no data");
    assert!(
        !first.contains("RUNNING"),
        "hero numbers rendered from no data"
    );
    assert!(!first.contains("NaN"), "a NaN reached the screen:\n{first}");

    // A terminal too small for the panels' combined minimum height has
    // more than one valid layout split, and which split the solver lands
    // on is not stable across runs
    // (solver variable ids are process global, so parallel tests shift them).
    // Content is therefore pinned
    // at the size above, whose panel minimums exactly fill the screen,
    // while smaller terminals get a no-panic check only.
    let _ = render(40, 8, &shared_for());
}
