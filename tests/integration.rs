use std::time::{Duration, Instant};

use sgtop::derive::{derive, WINDOWS};
use sgtop::history::{quantile_from_buckets, History};
use sgtop::metrics::{parse, Sample, SeriesKey};

fn fixture() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/live.txt")).unwrap()
}

fn key(name: &str, labels: &[(&str, &str)]) -> SeriesKey {
    SeriesKey {
        name: name.into(),
        labels: labels.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    }
}

#[test]
fn parses_live_fixture() {
    let s = parse(&fixture());
    let running = key(
        "sglang:num_running_reqs",
        &[
            ("engine_type", "unified"),
            ("model_name", "glm-5.3-flash"),
            ("moe_ep_rank", "0"),
            ("pp_rank", "0"),
            ("tp_rank", "0"),
        ],
    );
    // engine was fully idle at scrape time
    assert_eq!(s.simple.get(&running), Some(&0.0));
    let usage = s.gauge("sglang:token_usage").unwrap();
    assert!((0.0..=1.0).contains(&usage));

    // histogram ladder: ITL — bucket counts must match the emitted _count
    let itl = &s.hist[&key(
        "sglang:inter_token_latency_seconds",
        &[("engine_type", "unified"), ("model_name", "glm-5.3-flash")],
    )];
    let expected_count: u64 = fixture()
        .lines()
        .find(|l| l.starts_with("sglang:inter_token_latency_seconds_count"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| v as u64)
        .expect("fixture must contain an ITL _count line");
    assert_eq!(itl.count, expected_count);
    assert_eq!(itl.le.first().unwrap().to_string(), "0.002");
    assert!(itl.le.last().unwrap().is_infinite());
    assert_eq!(itl.counts.last().unwrap(), &(expected_count as f64));

    // every # HELP family must have produced at least one series
    let help_count = fixture().lines().filter(|l| l.starts_with("# HELP")).count();
    let families: std::collections::BTreeSet<&str> = s
        .simple
        .keys()
        .chain(s.hist.keys())
        .map(|k| k.name.as_str())
        .collect();
    assert_eq!(families.len(), help_count, "parsed families must match HELP count");
}

#[test]
fn parses_labels_with_escapes_and_timestamps() {
    let s = parse(
        "# TYPE m:gauge gauge\nm:gauge 3.5\nm:gauge{a=\"x\",b=\"say \\\"hi\\\"\"} 4.5 1234567890\n",
    );
    assert_eq!(s.gauge("m:gauge"), Some(3.5)); // label-less series
    let k = SeriesKey {
        name: "m:gauge".into(),
        labels: vec![("a".into(), "x".into()), ("b".into(), "say \"hi\"".into())],
    };
    assert_eq!(s.simple.get(&k), Some(&4.5));
}

#[test]
fn quantile_linear_interpolation() {
    // cumulative buckets: <=1s holds 10 obs, <=2s holds 20
    let b = vec![(1.0, 10.0), (2.0, 20.0), (f64::INFINITY, 20.0)];
    let p50 = quantile_from_buckets(&b, 0.5).unwrap(); // rank 10 -> boundary of first bucket
    assert!((p50 - 1.0).abs() < 1e-9);
    let p75 = quantile_from_buckets(&b, 0.75).unwrap(); // rank 15 -> 1.0 + 5/10 = 1.5
    assert!((p75 - 1.5).abs() < 1e-9);
    assert!(quantile_from_buckets(&[], 0.5).is_none());
    assert!(quantile_from_buckets(&[(1.0, 0.0), (2.0, 0.0)], 0.5).is_none());
}

fn one_gauge(name: &str, v: f64) -> Sample {
    parse(&format!("# TYPE {name} gauge\n{name} {v}\n"))
}

#[test]
fn rates_and_reset_clamping() {
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, one_gauge("m:c", 100.0));
    h.push(t0 + Duration::from_secs(10), one_gauge("m:c", 150.0));
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    assert!((rate - 5.0).abs() < 1e-9);

    // counter reset: value drops to 2 -> treated as +2 over the span, not -148
    h.push(t0 + Duration::from_secs(20), one_gauge("m:c", 2.0));
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    // 0->10s: +50; 10->20s: reset clamps to +2 => 52/20 = 2.6
    assert!((rate - 2.6).abs() < 1e-9, "rate was {rate}");
}

#[test]
fn window_quantile_falls_back_when_sparse() {
    let mut h = History::default();
    let t0 = Instant::now();
    let body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 10\nm:lat_bucket{le=\"1\"} 19\nm:lat_bucket{le=\"+Inf\"} 20\n\
        m:lat_count 20\nm:lat_sum 5.0\n";
    h.push(t0, parse(body));
    h.push(t0 + Duration::from_secs(1), parse(body)); // identical -> zero new observations
    // sparse (<5 observations in window) -> snapshot fallback must kick in
    let q = h.hist_quantile(|k| k.name == "m:lat", 0.5, Duration::from_secs(60));
    let snapshot = h.hist_snapshot_quantile(&|k: &SeriesKey| k.name == "m:lat", 0.5);
    assert_eq!(q, snapshot);
    // snapshot p50: rank 10 sits exactly at the le=0.1 boundary
    assert!((snapshot.unwrap() - 0.1).abs() < 1e-9);
}

fn mix(decode: f64, prefill: f64) -> Sample {
    parse(&format!(
        "# TYPE sglang:realtime_tokens_total counter\n\
         sglang:realtime_tokens_total{{mode=\"decode\"}} {decode}\n\
         sglang:realtime_tokens_total{{mode=\"prefill_compute\"}} {prefill}\n\
         # TYPE sglang:num_running_reqs gauge\n\
         sglang:num_running_reqs 4\n"
    ))
}

#[test]
fn stall_signature_detected() {
    let mut h = History::default();
    let t0 = Instant::now();
    // calm decode traffic
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(40.0, 0.0));
    // prefill burst: decode freezes while prefill_compute surges
    h.push(t0 + Duration::from_secs(2), mix(40.0, 100.0));
    h.push(t0 + Duration::from_secs(3), mix(42.0, 200.0));
    let d = derive(&h, 0).unwrap();
    let stalls = d.stalls[0]; // 5s window
    assert!(stalls.count >= 1, "expected stall detection, got {stalls:?}");
    assert!(stalls.seconds > 0.0, "stall seconds must be positive");

    let mut h = History::default();
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(40.0, 0.0));
    h.push(t0 + Duration::from_secs(2), mix(80.0, 0.0));
    h.push(t0 + Duration::from_secs(3), mix(120.0, 0.0));
    let d = derive(&h, 0).unwrap();
    assert_eq!(d.stalls[0].count, 0);
}

#[test]
fn windows_cover_expected_spans() {
    assert_eq!(WINDOWS.len(), 3);
    assert_eq!(WINDOWS[2], Duration::from_secs(60));
}

#[test]
fn session_peaks_accumulate_and_reset_on_restart() {
    use sgtop::derive::{update_peaks, Peaks};

    let mut h = History::default();
    let mut peaks = Peaks::default();
    let t0 = Instant::now();
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(100.0, 500.0));
    update_peaks(&mut peaks, &h);
    h.push(t0 + Duration::from_secs(2), mix(240.0, 900.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, Some(140.0)); // 240-100 over the last 1s
    assert_eq!(peaks.prefill, Some(500.0)); // first interval 0->500 beats 400
    // single = 140 decode / 4 running
    assert_eq!(peaks.decode_single, Some(35.0));

    // lower follow-up intervals must not lower the peaks
    h.push(t0 + Duration::from_secs(3), mix(250.0, 910.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, Some(140.0));
    assert_eq!(peaks.decode_single, Some(35.0));

    // counter reset (server restart) clears the session peaks
    h.push(t0 + Duration::from_secs(4), mix(3.0, 2.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, None);
    assert_eq!(peaks.prefill, None);
    assert_eq!(peaks.decode_single, None);
}
