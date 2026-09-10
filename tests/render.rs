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
    // two samples so rates/graphs have a base
    h.push(Instant::now(), parse(&body).unwrap());
    h.push(Instant::now(), parse(&body).unwrap());
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
