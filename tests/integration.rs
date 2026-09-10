use std::time::{Duration, Instant};

use sgtop::derive::{derive, WINDOWS};
use sgtop::history::{quantile_from_buckets, History};
use sgtop::metrics::{parse, Sample, SeriesKey, BUCKET_CAP, SERIES_CAP};

fn fixture() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/live.txt"
    ))
    .unwrap()
}

fn key(name: &str, labels: &[(&str, &str)]) -> SeriesKey {
    SeriesKey {
        name: name.into(),
        labels: labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

#[test]
fn parses_live_fixture() {
    let s = parse(&fixture()).unwrap();
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

    // series must exist for every # HELP family carrying readable data.
    // a family whose every record holds a non-finite value is no data:
    // it parses to no series, and the UI renders the missing marker
    // rather than a maximal value
    let mut family_has_finite_record: std::collections::BTreeMap<String, bool> = Default::default();
    for l in fixture().lines() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let name = l
            .split('{')
            .next()
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap();
        let base = name
            .strip_suffix("_bucket")
            .or_else(|| name.strip_suffix("_sum"))
            .or_else(|| name.strip_suffix("_count"))
            .unwrap_or(name);
        let finite = l
            .split_whitespace()
            .last()
            .and_then(|v| v.parse::<f64>().ok())
            .is_some_and(|v| v.is_finite());
        *family_has_finite_record
            .entry(base.to_string())
            .or_insert(false) |= finite;
    }
    let expected: std::collections::BTreeSet<String> = fixture()
        .lines()
        .filter(|l| l.starts_with("# HELP"))
        .filter_map(|l| l.split_whitespace().nth(2))
        .filter(|fam| family_has_finite_record.get(*fam).copied().unwrap_or(true))
        .map(String::from)
        .collect();
    let families: std::collections::BTreeSet<String> = s
        .simple
        .keys()
        .chain(s.hist.keys())
        .map(|k| k.name.clone())
        .collect();
    assert_eq!(
        families, expected,
        "parsed families must match the HELP families carrying readable data"
    );
}

#[test]
fn parses_labels_with_escapes_and_timestamps() {
    let s = parse(
        "# TYPE m:gauge gauge\nm:gauge 3.5\nm:gauge{a=\"x\",b=\"say \\\"hi\\\"\"} 4.5 1234567890\n",
    )
    .unwrap();
    assert_eq!(s.gauge("m:gauge"), Some(3.5)); // label-less series
    let k = SeriesKey {
        name: "m:gauge".into(),
        labels: vec![("a".into(), "x".into()), ("b".into(), "say \"hi\"".into())],
    };
    assert_eq!(s.simple.get(&k), Some(&4.5));
}

#[test]
fn label_values_with_braces_and_escapes_parse_fully() {
    // raw payload lines: `\\` in Rust source is one backslash in the payload
    let body = "# TYPE m gauge\n\
        m{v=\"a{b}\"} 1\n\
        m{v=\"a\\\"{b}\\\"\"} 2\n\
        m{v=\"a\\\\\"} 3\n";
    let s = parse(body).unwrap();
    assert_eq!(s.simple.get(&key("m", &[("v", "a{b}")])), Some(&1.0));
    // escaped quotes survive unescaping: payload value is a"{b}"
    assert_eq!(s.simple.get(&key("m", &[("v", "a\"{b}\"")])), Some(&2.0));
    // escaped backslash: payload value is a\
    assert_eq!(s.simple.get(&key("m", &[("v", "a\\")])), Some(&3.0));
}

// A record whose label quote never closes is dropped like any other
// malformed line: there is no reliable label block to recover, and
// scanning for one must neither panic nor hang.
#[test]
fn record_with_unclosed_quote_is_dropped() {
    let body = "# TYPE m gauge\nm{v=\"open 4\nm{ok=\"x\"} 5\n";
    let s = parse(body).unwrap();
    assert_eq!(s.simple.len(), 1);
    assert_eq!(s.simple.get(&key("m", &[("ok", "x")])), Some(&5.0));
}

#[test]
fn series_cap_admits_payloads_under_the_limit() {
    // 5000 series: well under the cap, exercises the same counting path
    let mut body = String::from("# TYPE m:g gauge\n");
    for i in 0..5000 {
        body.push_str(&format!("m:g{{id=\"{i}\"}} 1\n"));
    }
    let s = parse(&body).unwrap();
    assert_eq!(s.simple.len(), 5000);
}

// One series past the cap rejects the entire payload: the error names
// the cap and the observed count, and no partial sample is returned.
#[test]
fn series_cap_rejects_oversized_payload_whole() {
    let mut body = String::from("# TYPE m:g gauge\n");
    for i in 0..=SERIES_CAP {
        body.push_str(&format!("m:g{{id=\"{i}\"}} 1\n"));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: {} series (cap {SERIES_CAP})",
            SERIES_CAP + 1
        )),
        "error message was: {err}"
    );
}

// Histogram families count as one series regardless of bucket-ladder length:
// SERIES_CAP - 1 gauges plus one 5-record histogram family
// sits exactly at the cap and must be accepted.
#[test]
fn series_cap_counts_histogram_families_not_bucket_lines() {
    let mut body = String::from("# TYPE m:g gauge\n# TYPE m:h histogram\n");
    for i in 0..SERIES_CAP - 1 {
        body.push_str(&format!("m:g{{id=\"{i}\"}} 1\n"));
    }
    body.push_str(
        "m:h_bucket{le=\"0.1\"} 1\n\
         m:h_bucket{le=\"1\"} 2\n\
         m:h_bucket{le=\"+Inf\"} 2\n\
         m:h_count 2\n\
         m:h_sum 1.0\n",
    );
    let s = parse(&body).unwrap();
    assert_eq!(s.simple.len() + s.hist.len(), SERIES_CAP);
}

// One histogram family past the bucket cap rejects the entire payload,
// like the series cap: the error names the cap and the observed count, and
// no partial sample is returned.
#[test]
fn bucket_cap_rejects_family_over_the_limit() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..=BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: m:h has {} buckets (cap {BUCKET_CAP})",
            BUCKET_CAP + 1
        )),
        "error message was: {err}"
    );
}

// The bucket-cap bail renders the full SeriesKey through Display.
// This payload carries a non-`le` label so the labeled branch
// (brace open, separators, closing brace) is exercised,
// not just the bare family name.
#[test]
fn bucket_cap_error_names_the_full_labeled_family() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..=BUCKET_CAP {
        body.push_str(&format!(
            "m:h_bucket{{mode=\"decode\",le=\"{}\"}} 1\n",
            i as f64 * 0.001
        ));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: m:h{{mode=\"decode\"}} has {} buckets (cap {BUCKET_CAP})",
            BUCKET_CAP + 1
        )),
        "error message was: {err}"
    );
}

#[test]
// ladder-only shape is what is admitted at exactly the cap: a real
// exporter's mandatory `_sum`/`_count` completion lines
// after a 256-bucket ladder would reject (see the cap check's comment)
fn bucket_cap_admits_family_at_the_limit() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    let s = parse(&body).unwrap();
    assert_eq!(s.hist[&key("m:h", &[])].le.len(), BUCKET_CAP);
}

#[test]
// the cap check runs on every histogram-family record, not just _bucket:
// a family already at 256 buckets rejects even if the next line is _sum,
// not a 257th bucket. The assertion pins the family name and the cap,
// not the reported bucket count.
fn bucket_cap_fires_on_trailing_sum_after_buckets_at_the_limit() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    body.push_str("m:h_sum 1.0\nm:h_count 2\n");
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains("endpoint too large: m:h") && err.contains("(cap 256)"),
        "error message was: {err}"
    );
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
    parse(&format!("# TYPE {name} gauge\n{name} {v}\n")).unwrap()
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
    h.push(t0, parse(body).unwrap());
    // identical -> zero new observations
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
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
    .unwrap()
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
    assert!(
        stalls.count >= 1,
        "expected stall detection, got {stalls:?}"
    );
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

#[test]
fn histogram_bucket_without_le_is_skipped_not_fatal() {
    let body = "\
# TYPE m:lat histogram
m:lat_count{mode=\"decode\"} 2
m:lat_bucket{mode=\"decode\"} 1
m:lat_bucket{mode=\"decode\",le=\"NaN\"} 5
m:lat_bucket{mode=\"decode\",le=\"10\"} 2
m:other_gauge 7
";
    let sample = parse(body).unwrap();
    let key = key("m:lat", &[("mode", "decode")]);
    let h = sample.hist.get(&key).expect("histogram family present");
    // only the bucket with a usable `le` survives
    assert_eq!(h.le, vec![10.0]);
    assert_eq!(h.counts, vec![2.0]);
    assert_eq!(h.count, 2);
    assert_eq!(sample.simple.len(), 1);
}

#[test]
fn interval_rejects_non_finite() {
    use clap::Parser as _;
    use sgtop::args::Args as _Args;
    for bad in ["nan", "NaN", "inf", "-inf", "infinity"] {
        let res = _Args::try_parse_from(["sgtop", &format!("--interval={bad}")]);
        assert!(res.is_err(), "--interval {bad} must be rejected");
        let msg = format!("{}", res.unwrap_err().render());
        assert!(msg.contains("finite"), "error for {bad}: {msg}");
    }
    for ok in ["0.5", "10", "1.5"] {
        assert!(
            _Args::try_parse_from(["sgtop", "--interval", ok]).is_ok(),
            "--interval {ok} must be accepted"
        );
    }
}

// ---- display hardening: non-finite gauges, pool visibility ----

fn mamba_pool_sample(usage: Option<f64>) -> Sample {
    let usage_line = match usage {
        Some(v) => format!("sglang:mamba_usage {v}\n"),
        None => String::new(),
    };
    parse(&format!(
        "# TYPE sglang:mamba_used_tokens gauge\n\
         sglang:mamba_used_tokens 8.0\n\
         # TYPE sglang:mamba_available_tokens gauge\n\
         sglang:mamba_available_tokens 8.0\n\
         # TYPE sglang:mamba_usage gauge\n\
         {usage_line}"
    ))
    .unwrap()
}

// NaN and ±Inf gauge values are unreadable data, not extreme readings:
// skip the record like any other malformed line, leaving the series
// absent so consumers see no data, never a maximal-looking value.
#[test]
fn non_finite_gauge_values_read_as_no_data() {
    let body = "# TYPE m:g gauge\n\
                m:g NaN\n\
                m:g{r=\"0\"} +Inf\n\
                m:g{r=\"1\"} -Inf\n\
                m:g{r=\"2\"} 7.0\n";
    let s = parse(body).unwrap();
    assert_eq!(
        s.gauge("m:g"),
        Some(7.0),
        "only the finite reading may survive"
    );
    assert_eq!(s.simple.len(), 1);
}

// A NaN reading inside a windowed series must produce the same derived
// shape as the series being absent altogether: same gauge lookup,
// same pool membership, same graph lane.
#[test]
fn nan_reading_in_a_windowed_series_has_the_shape_of_a_missing_sample() {
    let t0 = Instant::now();
    let mut h_nan = History::default();
    h_nan.push(t0, mamba_pool_sample(Some(0.5)));
    h_nan.push(
        t0 + Duration::from_secs(1),
        mamba_pool_sample(Some(f64::NAN)),
    );
    let mut h_missing = History::default();
    h_missing.push(t0, mamba_pool_sample(Some(0.5)));
    h_missing.push(t0 + Duration::from_secs(1), mamba_pool_sample(None));

    let usage = |k: &SeriesKey| k.name == "sglang:mamba_usage";
    assert_eq!(
        h_nan.gauge_pred(usage),
        h_missing.gauge_pred(usage),
        "a non-finite reading must not look like a present value"
    );

    let d_nan = derive(&h_nan, 0).unwrap();
    let d_missing = derive(&h_missing, 0).unwrap();
    assert_eq!(
        d_nan.pools.len(),
        d_missing.pools.len(),
        "pool membership must match the absent-series case"
    );
    assert_eq!(
        d_nan.graphs.pool_mamba, d_missing.graphs.pool_mamba,
        "pool graph lane must match the absent-series case"
    );
}

// A completely full optional pool (zero available slots, all used) has
// data and must keep rendering at exactly 100% usage.
#[test]
fn optional_pool_at_exactly_full_capacity_still_renders() {
    let body = "\
# TYPE sglang:mamba_used_tokens gauge
sglang:mamba_used_tokens 16.0
# TYPE sglang:mamba_available_tokens gauge
sglang:mamba_available_tokens 0.0
# TYPE sglang:mamba_usage gauge
sglang:mamba_usage 1.0
# TYPE sglang:swa_used_tokens gauge
sglang:swa_used_tokens 16.0
# TYPE sglang:swa_available_tokens gauge
sglang:swa_available_tokens 0.0
# TYPE sglang:swa_token_usage gauge
sglang:swa_token_usage 1.0
";
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
    let d = derive(&h, 0).unwrap();
    let mamba = d
        .pools
        .iter()
        .find(|p| p.name == "mamba")
        .expect("full mamba pool must keep rendering");
    assert_eq!(mamba.usage, 1.0);
    let swa = d
        .pools
        .iter()
        .find(|p| p.name == "SWA")
        .expect("full SWA pool must keep rendering");
    assert_eq!(swa.usage, 1.0);
}

// The visibility gate is "ever had data", not "ever existed": a pool
// reporting zero capacity and zero usage stays hidden.
#[test]
fn optional_pool_that_never_had_data_stays_hidden() {
    let body = "\
# TYPE sglang:mamba_used_tokens gauge
sglang:mamba_used_tokens 0.0
# TYPE sglang:mamba_available_tokens gauge
sglang:mamba_available_tokens 0.0
# TYPE sglang:mamba_usage gauge
sglang:mamba_usage 0.0
";
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
    let d = derive(&h, 0).unwrap();
    assert!(
        d.pools.iter().all(|p| p.name != "mamba"),
        "a pool with no data in the window must stay hidden"
    );
}

// A usage-ratio gauge that blinks non-finite mid-window must not hide
// a pool that has reported data: visibility follows the ever-had-data
// rule, and the usage shown falls back to the latest in-window reading.
#[test]
fn pool_stays_visible_when_its_usage_gauge_blinks_non_finite() {
    let t0 = Instant::now();
    let live = "\
# TYPE sglang:mamba_used_tokens gauge
sglang:mamba_used_tokens 4.0
# TYPE sglang:mamba_available_tokens gauge
sglang:mamba_available_tokens 4.0
# TYPE sglang:mamba_usage gauge
sglang:mamba_usage 0.5
";
    let blink = live.replace("sglang:mamba_usage 0.5", "sglang:mamba_usage NaN");

    // blink while the counts stay present in the newest scrape:
    // the usage shown is the computed used/total ratio
    let mut h_blink = History::default();
    h_blink.push(t0, parse(live).unwrap());
    h_blink.push(t0 + Duration::from_secs(1), parse(&blink).unwrap());
    let d_blink = derive(&h_blink, 0).unwrap();
    let mamba = d_blink
        .pools
        .iter()
        .find(|p| p.name == "mamba")
        .expect("a pool with in-window data must keep rendering through a usage-gauge blink");
    assert_eq!(mamba.usage, 0.5, "usage must come from the computed ratio");

    // blink with the newest scrape carrying no mamba gauges at all:
    // the usage shown is the latest in-window usage reading
    let absent = "# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs 1.0\n";
    let mut h_absent = History::default();
    h_absent.push(t0, parse(live).unwrap());
    h_absent.push(t0 + Duration::from_secs(1), parse(absent).unwrap());
    let d_absent = derive(&h_absent, 0).unwrap();
    let mamba_absent =
        d_absent.pools.iter().find(|p| p.name == "mamba").expect(
            "a pool with in-window data must keep rendering when the newest scrape omits it",
        );
    assert_eq!(
        mamba_absent.usage, 0.5,
        "usage must fall back to the prior in-window reading"
    );
}

// ---- windowed-delta semantics: reappearance, label order, ladder pairing ----

fn mix_without_decode(prefill: f64) -> Sample {
    parse(&format!(
        "# TYPE sglang:realtime_tokens_total counter\n\
         sglang:realtime_tokens_total{{mode=\"prefill_compute\"}} {prefill}\n\
         # TYPE sglang:num_running_reqs gauge\n\
         sglang:num_running_reqs 4\n"
    ))
    .unwrap()
}

fn counter_with_gap(values: [Option<f64>; 5]) -> History {
    let mut h = History::default();
    let t0 = Instant::now();
    for (i, v) in values.iter().enumerate() {
        let mut body = String::from("# TYPE m:other gauge\nm:other 1\n");
        if let Some(v) = v {
            body.push_str(&format!("# TYPE m:c counter\nm:c {v}\n"));
        }
        h.push(
            t0 + Duration::from_secs(i as u64 * 10),
            parse(&body).unwrap(),
        );
    }
    h
}

// A counter that vanishes for one or more scrapes and later reappears
// counts as a new series from the reappearance point. Pairing the first
// reappearance sample with a pre-gap sample would fabricate a spike
// (higher value) or a fake reset (lower value).
#[test]
fn reappearance_after_gap_is_not_paired_with_the_pre_gap_sample() {
    let h = counter_with_gap([Some(100.0), Some(150.0), None, Some(160.0), Some(220.0)]);
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    // paired deltas 50 + 60 over 40s (the reappearance interval adds 0)
    assert!((rate - 2.75).abs() < 1e-9, "rate was {rate}");
}

// A reappearance whose value is lower than the pre-gap reading follows
// the same rule: no pairing across the gap, and the next interval
// measures against the reappearance value itself.
#[test]
fn reappearance_after_gap_with_lower_value_starts_fresh() {
    let h = counter_with_gap([Some(100.0), Some(150.0), None, Some(40.0), Some(90.0)]);
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    // paired deltas 50 + 50 over 40s
    assert!((rate - 2.5).abs() < 1e-9, "rate was {rate}");
}

// Flagged intervals must be exactly the intervals whose earlier sample
// lacks the series, and every other interval must keep its true delta.
// The decoded rate vector below pins those positions exactly.
#[test]
fn reappearance_intervals_are_exactly_those_after_an_absent_predecessor() {
    let mut h = History::default();
    let t0 = Instant::now();
    // decode counter absent at entry 2, present everywhere else
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(10.0, 0.0));
    h.push(t0 + Duration::from_secs(2), mix_without_decode(0.0));
    h.push(t0 + Duration::from_secs(3), mix(30.0, 0.0));
    h.push(t0 + Duration::from_secs(4), mix(60.0, 0.0));
    let d = derive(&h, 0).unwrap();
    // interval 2 (reappearance) contributes 0, the vanished interval has
    // nothing to sum, and the others keep their true deltas
    let expected = [10.0, 0.0, 0.0, 30.0];
    assert_eq!(
        d.graphs.decode.len(),
        expected.len(),
        "decode lanes were {:?}",
        d.graphs.decode
    );
    for (i, (got, want)) in d.graphs.decode.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "decode lane {i} was {got}, expected {want} (lanes {:?})",
            d.graphs.decode
        );
    }
}

// Session peaks must not record the fabricated rate of a reappearance
// interval: the interval that brings the series back contributes no rate.
#[test]
fn reappearance_does_not_record_a_peak_spike() {
    use sgtop::derive::{update_peaks, Peaks};

    let mut h = History::default();
    let mut peaks = Peaks::default();
    let t0 = Instant::now();
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(100.0, 0.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, Some(100.0));
    h.push(t0 + Duration::from_secs(2), mix_without_decode(0.0));
    update_peaks(&mut peaks, &h);
    h.push(t0 + Duration::from_secs(3), mix(400.0, 0.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(
        peaks.decode,
        Some(100.0),
        "the reappearance interval must not feed the session peak"
    );
}

// Two scrapes of the same metric whose labels are emitted in different
// orders are one series, not two: label order must not affect identity,
// or the "new" series fabricates a full-value delta on its first pair.
#[test]
fn label_emit_order_does_not_fork_series_identity() {
    let a = parse("# TYPE m:c counter\nm:c{a=\"x\",b=\"y\"} 100\n").unwrap();
    let b = parse("# TYPE m:c counter\nm:c{b=\"y\",a=\"x\"} 150\n").unwrap();
    let key_a = a.simple.keys().next().unwrap().clone();
    assert!(
        b.simple.contains_key(&key_a),
        "reordered labels must resolve to the same series key"
    );
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, a);
    h.push(t0 + Duration::from_secs(10), b);
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    assert!((rate - 5.0).abs() < 1e-9, "rate was {rate}");
}

// Histogram bucket counts must pair on the `le` boundary value, never
// on list position: a bucket inserted mid-ladder between two samples
// shifts every later index pairing, and the shifted deltas can stay
// non-negative, so only boundary-keyed pairing keeps the quantile sane.
#[test]
fn mid_ladder_bucket_insertion_pairs_quantiles_by_boundary() {
    let mut h = History::default();
    let t0 = Instant::now();
    let old_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 2\nm:lat_bucket{le=\"1\"} 19\nm:lat_bucket{le=\"+Inf\"} 20\n\
        m:lat_count 20\nm:lat_sum 9.0\n";
    let new_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 7\nm:lat_bucket{le=\"0.5\"} 19\n\
        m:lat_bucket{le=\"1\"} 29\nm:lat_bucket{le=\"+Inf\"} 30\n\
        m:lat_count 30\nm:lat_sum 14.0\n";
    h.push(t0, parse(old_body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(new_body).unwrap());
    let p50 = h
        .hist_quantile(|k| k.name == "m:lat", 0.5, Duration::from_secs(60))
        .unwrap();
    let p95 = h
        .hist_quantile(|k| k.name == "m:lat", 0.95, Duration::from_secs(60))
        .unwrap();
    // shared-boundary window deltas: +5 at le=0.1, +10 at le=1
    // p50 rank 5 sits exactly on the le=0.1 boundary, p95 rank 9.5
    // interpolates inside the (0.1, 1] bucket
    assert!((p50 - 0.1).abs() < 1e-9, "p50 was {p50}");
    assert!((p95 - 0.91).abs() < 1e-9, "p95 was {p95}");
}

// A bucket with an `le` bound of NaN cannot be ordered, so parse must
// drop it: a retained NaN bound wedges the boundary-keyed delta walk
// and silently truncates deltas at later boundaries. The +Inf bound
// is the standard top bucket and stays valid.
#[test]
fn nan_le_bound_is_dropped_and_quantiles_stay_sane() {
    let old_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 10\nm:lat_bucket{le=\"NaN\"} 15\n\
        m:lat_bucket{le=\"1\"} 19\nm:lat_bucket{le=\"+Inf\"} 20\n\
        m:lat_count 20\nm:lat_sum 9.0\n";
    let new_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 14\nm:lat_bucket{le=\"1\"} 29\n\
        m:lat_bucket{le=\"+Inf\"} 30\nm:lat_count 30\nm:lat_sum 14.0\n";
    let old = parse(old_body).unwrap();
    let key = old.hist.keys().next().unwrap().clone();
    let h = &old.hist[&key];
    // only usable bounds survive: the NaN bound must not be stored
    assert_eq!(h.le, vec![0.1, 1.0, f64::INFINITY]);
    assert_eq!(h.counts, vec![10.0, 19.0, 20.0]);

    let mut hist = History::default();
    let t0 = Instant::now();
    hist.push(t0, old);
    hist.push(t0 + Duration::from_secs(1), parse(new_body).unwrap());
    // shared-boundary deltas: +4 at le=0.1, +10 at le=1
    // p50 rank 5 interpolates inside the (0.1, 1] bucket
    let p50 = hist
        .hist_quantile(|k| k.name == "m:lat", 0.5, Duration::from_secs(60))
        .unwrap();
    assert!((p50 - 0.25).abs() < 1e-9, "p50 was {p50}");
}
