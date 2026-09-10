use anyhow::Result;

use crate::derive::{self, Derived, WIN_LABELS};
use crate::history::History;

/// One-shot text snapshot for scripts, cron logs, and parser verification.
pub fn print_once(h: &History, peaks: &mut derive::Peaks) -> Result<()> {
    let d = derive::derive(h, 2).ok_or_else(|| anyhow::anyhow!("no data scraped yet"))?;
    derive::update_peaks(peaks, h);
    print_snapshot(&d, peaks);
    Ok(())
}

fn tok(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.0} tok/s"))
        .unwrap_or_else(|| "—".into())
}

fn secs(v: Option<f64>) -> String {
    match v {
        Some(v) if v >= 10.0 => format!("{v:.1}s"),
        Some(v) => format!("{:.0}ms", v * 1000.0),
        None => "—".into(),
    }
}

fn pct(v: Option<f64>) -> String {
    v.map(|v| format!("{:.0}%", v * 100.0))
        .unwrap_or_else(|| "—".into())
}

fn pool_value(p: &derive::Pool) -> String {
    match (p.used, p.total) {
        (Some(u), Some(t)) if t > 0.0 => {
            let counts = format!(
                "{} / {}",
                crate::ui::fmt_tokens(u),
                crate::ui::fmt_tokens(t)
            );
            if p.unit == "slots" {
                format!("{} ({} slots)", pct(Some(p.usage)), counts)
            } else {
                format!("{} ({})", pct(Some(p.usage)), counts)
            }
        }
        _ => pct(Some(p.usage)),
    }
}

fn num(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into())
}

fn print_snapshot(d: &Derived, peaks: &derive::Peaks) {
    let f = d.window_focus;
    println!(
        "sglang {} ({}) — running {} · queue {} · pools: {}",
        d.model,
        d.engine,
        num(d.running),
        num(d.queue),
        d.pools
            .iter()
            .map(|p| format!("{} {}", p.name, pool_value(p)))
            .collect::<Vec<_>>()
            .join(" / ")
    );
    println!(
        "tokens: decode {} · prefill {} (5s/15s/60s: {}/{}/{}) · cache hit {} · spec accept {}",
        tok(d.decode_instant),
        tok(d.prefill_instant),
        tok(d.decode_rate[0]),
        tok(d.decode_rate[1]),
        tok(d.decode_rate[2]),
        pct(d.cache_hit),
        pct(d.spec_accept)
    );
    for (name, lat) in [("ttft", &d.ttft), ("itl", &d.itl), ("e2e", &d.e2e)] {
        let q = &lat[f];
        println!(
            "{name} ({}): p50 {} · p95 {} · p99 {}",
            WIN_LABELS[f],
            secs(q.p50),
            secs(q.p95),
            secs(q.p99)
        );
    }
    println!(
        "queues: {}",
        d.subqueues
            .iter()
            .map(|(n, v)| format!("{n} {}", num(*v)))
            .collect::<Vec<_>>()
            .join(" · ")
    );
    let pk = |v: Option<f64>| v.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into());
    println!(
        "l2: dev {} · host {} · wb {} tok/s · rb {} tok/s · drop {} tok/s",
        pct(d.l2_device),
        pct(d.l2_host),
        num(d.l2_wb),
        num(d.l2_rb),
        num(d.l2_drop)
    );
    println!(
        "peaks (this run): decode {} · prefill {} · single stream {} tok/s",
        pk(peaks.decode),
        pk(peaks.prefill),
        pk(peaks.decode_single)
    );
    let s = d.stalls[f];
    println!(
        "stalls: {} × {:.1}s in {} · evict/s {} · retract/s {} · 503/s {}",
        s.count,
        if s.count > 0 {
            s.seconds / s.count as f64
        } else {
            0.0
        },
        WIN_LABELS[f],
        num(d.evict_rate),
        num(d.retract_rate),
        num(d.http_503_rate)
    );
}
