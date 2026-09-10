use std::time::Duration;

use crate::history::{counter_delta, History};
use crate::metrics::{Sample, SeriesKey};

pub const WINDOWS: [Duration; 3] = [
    Duration::from_secs(5),
    Duration::from_secs(15),
    Duration::from_secs(60),
];

pub const WIN_LABELS: [&str; 3] = ["5s", "15s", "60s"];

fn fam(name: &'static str) -> impl Fn(&SeriesKey) -> bool + Copy {
    move |k: &SeriesKey| k.name == name
}

fn fam_labeled(
    name: &'static str,
    lk: &'static str,
    lv: &'static str,
) -> impl Fn(&SeriesKey) -> bool + Copy {
    move |k: &SeriesKey| k.name == name && k.has_label(lk, lv)
}

/// One cache pool (KV / mamba / SWA / host) with its current usage ratio.
#[derive(Clone, Copy)]
pub struct Pool {
    pub name: &'static str,
    pub usage: f64,
    /// absolute counts backing the usage ratio, when the engine
    /// exposes them (engine-wide, summed across ranks)
    pub used: Option<f64>,
    pub total: Option<f64>,
    /// what the counts count: "tokens" for KV/host, "slots" for
    /// state pools (mamba, SWA)
    pub unit: &'static str,
}

/// Rate values indexed by window (5s / 15s / 60s).
pub type Triple = [Option<f64>; 3];

/// p50/p95/p99 for one metric.
#[derive(Clone, Copy, Default)]
pub struct Quantiles {
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

/// Quantiles for each window (5s / 15s / 60s).
pub type LatencyTriple = [Quantiles; 3];

#[derive(Clone, Copy, Default, Debug)]
pub struct Stalls {
    pub count: u32,
    pub seconds: f64,
}

/// One rate lane over the trailing 60s window: one rate per scrape
/// interval, with the interval's endpoint presence flags.
#[derive(Clone, Default)]
pub struct Lane {
    /// Rate in the lane's unit (tokens/s) per scrape interval. An interval
    /// whose series is missing from either endpoint sample pairs nothing
    /// and reads 0: data absence, never a measured collapse.
    pub vals: Vec<f64>,
    /// The interval's earlier sample lacks the series: a first appearance,
    /// a reappearance after an absence, or the return leg of a non-finite
    /// reading dropped at the parse boundary. Recorded for both lanes:
    /// every rate carries its gap provenance, and the stall fold consumes
    /// the decode lane's flags.
    pub absent_old: Vec<bool>,
    /// The interval's later sample lacks the series: the absence starting.
    pub absent_new: Vec<bool>,
}

/// The trailing 60s window in the shape the graphs draw, produced by one
/// pass over the ring (`scan_window`) and cached next to it.
#[derive(Clone, Default)]
pub struct Graphs {
    pub decode: Lane,
    pub prefill: Lane,
    /// flagged intervals (see the stall definition on `scan_window`)
    pub stall: Vec<bool>,
    /// per-pool usage ratios (0–1) per scrape: KV / mamba / SWA / host
    pub pool_kv: Vec<f64>,
    pub pool_mamba: Vec<f64>,
    pub pool_host: Vec<f64>,
    pub pool_swa: Vec<f64>,
    /// Per-stream decode speed per interval: the interval's decode rate
    /// divided by its later sample's running count. None while idle.
    /// Also None when the decode series is missing from either endpoint
    /// sample (missing data never reads as a measured collapse).
    pub per_stream: Vec<Option<f64>>,
    /// interval durations in seconds, parallel to the vectors above
    pub dt: Vec<f64>,
}

/// Everything the UI needs for one draw: rates, quantiles, and pools
/// computed per call, plus the graphs and stall flags from the cached
/// per-push window scan (`History::scan`).
pub struct Derived {
    pub model: String,
    pub engine: String,
    pub running: Option<f64>,
    pub queue: Option<f64>,
    pub subqueues: Vec<(&'static str, Option<f64>)>,
    pub decode_rate: Triple,
    /// instantaneous decode rate (most recent scrape interval)
    pub decode_instant: Option<f64>,
    /// instantaneous prefill-compute rate
    pub prefill_instant: Option<f64>,
    pub prefill_rate: Triple,
    pub pools: Vec<Pool>,
    pub cache_hit: Option<f64>,
    pub spec_accept: Option<f64>,
    pub spec_accept_len: Option<f64>,
    pub ttft: LatencyTriple,
    pub itl: LatencyTriple,
    pub e2e: LatencyTriple,
    pub queue_time: LatencyTriple,
    pub prompt_len_p50: Option<f64>,
    pub prompt_len_p95: Option<f64>,
    pub stalls: [Stalls; 3],
    pub evict_rate: Option<f64>,
    pub retract_rate: Option<f64>,
    pub http_503_rate: Option<f64>,
    pub http_active: Option<f64>,
    pub gen_throughput_gauge: Option<f64>,
    /// L2/HiCache: per-tier prefill hit rates (fraction of prefill tokens
    /// served from device/host tier, windowed)
    pub l2_device: Option<f64>,
    pub l2_host: Option<f64>,
    /// host-tier traffic: tokens/s written back / read back
    pub l2_wb: Option<f64>,
    pub l2_rb: Option<f64>,
    /// device KV tokens/s destroyed without a host backup (should be 0)
    pub l2_drop: Option<f64>,
    pub new_token_ratio: Option<f64>,
    /// average generation length (tokens) of currently running requests
    pub gen_progress: Option<f64>,
    pub cpu_tokenizer: Option<f64>,
    pub cpu_detokenizer: Option<f64>,
    pub cpu_scheduler: Option<f64>,
    pub graphs: Graphs,
    pub window_focus: usize,
}

fn rate_triple(h: &History, pred: impl Fn(&SeriesKey) -> bool + Copy) -> Triple {
    let mut out = [None; 3];
    for (i, w) in WINDOWS.iter().enumerate() {
        out[i] = h.rate_sum(pred, *w);
    }
    out
}

fn latency_triple(h: &History, name: &'static str) -> LatencyTriple {
    let mut out: LatencyTriple = Default::default();
    for (i, w) in WINDOWS.iter().enumerate() {
        out[i] = Quantiles {
            p50: h.hist_quantile(fam(name), 0.50, *w),
            p95: h.hist_quantile(fam(name), 0.95, *w),
            p99: h.hist_quantile(fam(name), 0.99, *w),
        };
    }
    out
}

/// Fold the shared stall flags into the 5s/15s/60s time budgets: each
/// window's trailing span counts the flagged intervals and their frozen
/// seconds (the oldest interval may be cut off mid-way).
fn fold_stalls(flags: &[bool], dt: &[f64]) -> [Stalls; 3] {
    let mut out = [Stalls::default(); 3];
    for (i, w) in WINDOWS.iter().enumerate() {
        let mut budget = w.as_secs_f64();
        let mut count = 0u32;
        let mut seconds = 0.0;
        for j in (0..dt.len()).rev() {
            if budget <= 0.0 {
                break;
            }
            let take = dt[j].min(budget);
            if flags[j] {
                count += 1;
                seconds += take;
            }
            budget -= dt[j];
        }
        out[i] = Stalls { count, seconds };
    }
    out
}

/// Session peak rates, accumulated by the scraper across the whole run.
/// `decode_single` is the fastest per-stream rate seen: an interval's decode
/// rate divided by the requests that were running during it.
#[derive(Default, Clone, Copy)]
pub struct Peaks {
    pub decode: Option<f64>,
    pub prefill: Option<f64>,
    pub decode_single: Option<f64>,
}

/// Fold the most recent scrape interval into the session peaks. A counter
/// reset (engine restart) clears all session peaks, since the old process's
/// peaks are not this engine's.
pub fn update_peaks(peaks: &mut Peaks, h: &History) {
    let n = h.len();
    if n < 2 {
        return;
    }
    let (Some(prev), Some(cur)) = (h.get(n - 2), h.get(n - 1)) else {
        return;
    };
    let dt = (cur.t - prev.t).as_secs_f64();
    if dt <= 0.0 {
        return;
    }
    let had_reset = ["decode", "prefill_compute"].iter().any(|mode| {
        let keys = |s: &Sample| {
            s.simple
                .iter()
                .filter(|(k, _)| {
                    k.name == "sglang:realtime_tokens_total" && k.has_label("mode", mode)
                })
                .map(|(k, v)| (k.clone(), *v))
                .collect::<Vec<_>>()
        };
        keys(&prev.sample)
            .iter()
            .any(|(k, v_old)| cur.sample.simple.get(k).map(|v| v < v_old).unwrap_or(false))
    });
    if had_reset {
        // the engine restarted: the old process's peaks are not this
        // engine's peaks
        *peaks = Peaks::default();
        return;
    }
    let decode = interval_rate(&prev.sample, &cur.sample, "decode") / dt;
    let prefill = interval_rate(&prev.sample, &cur.sample, "prefill_compute") / dt;
    if decode > peaks.decode.unwrap_or(0.0) {
        peaks.decode = Some(decode);
    }
    if prefill > peaks.prefill.unwrap_or(0.0) {
        peaks.prefill = Some(prefill);
    }
    let running = cur.sample.gauge("sglang:num_running_reqs").unwrap_or(0.0);
    if running > 0.0 {
        let single = decode / running;
        if single > peaks.decode_single.unwrap_or(0.0) {
            peaks.decode_single = Some(single);
        }
    }
}

fn interval_rate(old: &crate::metrics::Sample, new: &crate::metrics::Sample, mode: &str) -> f64 {
    let key = |k: &SeriesKey| k.name == "sglang:realtime_tokens_total" && k.has_label("mode", mode);
    let mut delta = 0.0;
    for (k, v_new) in &new.simple {
        if key(k) {
            // a key absent from the earlier sample is a new series
            // (first appearance, or reappearance after a gap): never
            // pair it with a pre-gap sample, so no fabricated spike
            // reaches the peaks or the graph lanes
            let Some(v_old) = old.simple.get(k) else {
                continue;
            };
            delta += counter_delta(*v_old, *v_new);
        }
    }
    delta
}

impl Lane {
    /// Append one scrape interval: the paired counter deltas over `dt`
    /// become the rate, or 0 when the series is missing at one end.
    fn push_interval(&mut self, old: &Sample, new: &Sample, mode: &str, dt: f64) {
        let of_mode =
            |k: &SeriesKey| k.name == "sglang:realtime_tokens_total" && k.has_label("mode", mode);
        self.vals.push(interval_rate(old, new, mode) / dt);
        self.absent_old.push(!old.simple.keys().any(of_mode));
        self.absent_new.push(!new.simple.keys().any(of_mode));
    }
}

/// One pass over the trailing 60s window, rebuilt once per scrape push,
/// cached next to the ring (`History::scan`): per-series interval rates
/// with their presence flags, the stall flags, and the graph lanes
/// UI draws. Banner counters and red graph ticks read this shared
/// pass, so no consumer re-derives stalls.
///
/// # Stall definition
///
/// An interval (a pair of consecutive ring samples) is stalled
/// whenever the engine was working through it without producing
/// decode output:
///
/// - window: the trailing 60s (`WINDOWS[2]`)
/// - median over: the decode rate (tokens/s) of every interval, gaps
///   included. The full-window median is the bar, so the window's busy
///   level sets it
/// - threshold: decode rate below 0.25 × that median, while the interval's
///   later sample reports requests running and prefill was computing
///   (a positive prefill rate)
///
/// Gaps: the parse boundary drops non-finite counter readings, so a NaN
/// blink is a sample whose series is absent. Unreadable data is a gap,
/// intended behavior rather than something to suppress. An interval
/// leading out of a gap pairs nothing (rate 0) and is flagged whenever
/// the series was carried by an earlier window sample: evidence
/// of a real gap (a reappearance, NaN blink, or ongoing absence),
/// never a first appearance. The running-count and prefill
/// conditions still apply, so the red tick lands on the position
/// of the unreadable sample. A series never present in an earlier window
/// sample (a first appearance) is not a stall: no collapse was ever
/// observed, and a leading interval's zero rate is absence.
/// An interval where the absence starts (the later sample lacks the series)
/// is not flagged either: nothing measurable collapsed. The same
/// per-pair absence rule drives the rate sums, so a counter that only
/// moves across gap intervals shows a zero real rate.
pub fn scan_window(h: &History) -> Graphs {
    let entries = h.window_entries(WINDOWS[2]);
    let mut decode = Lane::default();
    let mut prefill = Lane::default();
    let mut running: Vec<f64> = Vec::new();
    let mut pool_kv = Vec::new();
    let mut pool_mamba = Vec::new();
    let mut pool_host = Vec::new();
    let mut pool_swa = Vec::new();
    let mut per_stream: Vec<Option<f64>> = Vec::new();
    let mut dt = Vec::new();
    // gap intervals with evidence: the decode series existed in an earlier
    // sample of the window, so an absent predecessor is a reappearance
    // after a gap rather than a first appearance (never a stall)
    let mut gap_after_seen: Vec<bool> = Vec::new();
    let mut decode_seen = false;

    for pair in entries.windows(2) {
        let d = (pair[1].t - pair[0].t).as_secs_f64();
        if d <= 0.0 {
            continue;
        }
        decode.push_interval(&pair[0].sample, &pair[1].sample, "decode", d);
        prefill.push_interval(&pair[0].sample, &pair[1].sample, "prefill_compute", d);
        let reappearance = decode.absent_old.last() == Some(&true) && decode_seen;
        gap_after_seen.push(reappearance);
        // the pair's earlier sample counts as evidence for the next interval
        decode_seen |= decode.absent_old.last() != Some(&true);
        let running_at = pair[1]
            .sample
            .gauge("sglang:num_running_reqs")
            .unwrap_or(0.0);
        running.push(running_at);
        // an absent decode endpoint leaves the lane's rate at 0. Displayed
        // against a positive running count, that absence reads as a measured
        // collapse: missing data stores no speed instead
        let decode_missing =
            decode.absent_old.last() == Some(&true) || decode.absent_new.last() == Some(&true);
        per_stream.push(if running_at > 0.0 && !decode_missing {
            Some(decode.vals.last().copied().unwrap_or(0.0) / running_at)
        } else {
            None
        });
        // lanes use raw counts, so graph and title values agree
        // (mamba_usage measures pages, not slots)
        let s = &pair[1].sample;
        pool_kv.push(
            s.gauge("sglang:kv_used_tokens")
                .zip(s.gauge("sglang:max_total_num_tokens"))
                .filter(|(_, t)| *t > 0.0)
                .map(|(u, t)| (u / t).clamp(0.0, 1.0))
                .unwrap_or(0.0),
        );
        pool_mamba.push(
            s.gauge("sglang:mamba_used_tokens")
                .map(|u| {
                    let total = u + s.gauge("sglang:mamba_available_tokens").unwrap_or(0.0);
                    if total > 0.0 {
                        (u / total).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0),
        );
        pool_host.push(
            match (
                s.gauge("sglang:hicache_host_total_tokens"),
                s.gauge("sglang:hicache_host_used_tokens"),
            ) {
                (Some(total), Some(used)) if total > 0.0 => used / total,
                _ => 0.0,
            },
        );
        pool_swa.push(
            s.gauge("sglang:swa_used_tokens")
                .map(|u| {
                    let total = u + s.gauge("sglang:swa_available_tokens").unwrap_or(0.0);
                    if total > 0.0 {
                        (u / total).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0),
        );

        dt.push(d);
    }

    // the median bar spans the whole window, so the flags are set after
    // the walk, as an arithmetic fold over the collected lanes
    let median = median_of(&decode.vals);
    let stall: Vec<bool> = (0..decode.vals.len())
        .map(|i| {
            running[i] > 0.0
                && prefill.vals[i] > 0.0
                && (gap_after_seen[i]
                    || (!decode.absent_old[i]
                        && !decode.absent_new[i]
                        && decode.vals[i] < 0.25 * median))
        })
        .collect();

    Graphs {
        decode,
        prefill,
        stall,
        pool_kv,
        pool_mamba,
        pool_host,
        pool_swa,
        per_stream,
        dt,
    }
}

fn median_of(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match s.len() {
        0 => 0.0,
        n => s[n / 2],
    }
}

/// Per-tier cache hit rate per sglang's documented formula:
/// rate(mode=<tier>) / rate(sum of all modes) — windowed.
fn l2_hit_rate(h: &History, tier: &'static str) -> Option<f64> {
    // the engine updates prefill_effective_tokens_total on its log interval
    // (tens of seconds), so these rates are only meaningful over the 60s
    // window regardless of the focused window
    let hit = h.rate_sum(
        fam_labeled("sglang:prefill_effective_tokens_total", "mode", tier),
        WINDOWS[2],
    )?;
    let total = h.rate_sum(fam("sglang:prefill_effective_tokens_total"), WINDOWS[2])?;
    if total <= 0.0 {
        return None;
    }
    Some((hit / total).clamp(0.0, 1.0))
}

/// Host (L2) pool usage; present once the engine reports capacity.
fn host_pool(h: &History) -> Option<(f64, f64, f64)> {
    let total = h.sum_gauge(fam("sglang:hicache_host_total_tokens"))?;
    if total <= 0.0 {
        return None;
    }
    let used = h
        .sum_gauge(fam("sglang:hicache_host_used_tokens"))
        .unwrap_or(0.0);
    Some((used / total, used, total))
}

pub fn derive(h: &History, window_focus: usize) -> Option<Derived> {
    let last = h.last()?;
    let running = h.gauge_pred(fam("sglang:num_running_reqs"));
    let queue = h.gauge_pred(fam("sglang:num_queue_reqs"));

    let subqueues: Vec<(&'static str, Option<f64>)> = vec![
        (
            "prefill bootstrap",
            h.gauge_pred(fam("sglang:num_prefill_bootstrap_queue_reqs")),
        ),
        (
            "prefill inflight",
            h.gauge_pred(fam("sglang:num_prefill_inflight_queue_reqs")),
        ),
        (
            "decode prealloc",
            h.gauge_pred(fam("sglang:num_decode_prealloc_queue_reqs")),
        ),
        (
            "decode transfer",
            h.gauge_pred(fam("sglang:num_decode_transfer_queue_reqs")),
        ),
        (
            "grammar",
            h.gauge_pred(fam("sglang:num_grammar_queue_reqs")),
        ),
    ];

    // pools: KV always, mamba/SWA once the engine has reported pool data
    // inside the 60s window (sticky, so idle lapses within the window
    // don't make rows vanish)
    let mut pools = Vec::new();
    if let Some(u) = h.gauge_pred(fam("sglang:token_usage")) {
        // logical capacity: max_total_num_tokens is exported per rank
        // but every rank reports the same logical pool (KV is sharded
        // by head, so each rank stores a shard of every token).
        // Summing would double-count; take one rank's value.
        let used = h.gauge_pred(fam("sglang:kv_used_tokens"));
        let total = h.gauge_pred(fam("sglang:max_total_num_tokens"));
        // counts are self-consistent per scrape; the usage gauges update on
        // the engine's log interval
        let usage = used
            .zip(total)
            .filter(|(_, t)| *t > 0.0)
            .map(|(u, t)| (u / t).clamp(0.0, 1.0))
            .unwrap_or(u);
        pools.push(Pool {
            name: "KV",
            usage,
            used,
            total,
            unit: "tokens",
        });
    }
    let ever_live =
        |avail: &'static str, used: &'static str, usage: &'static str| -> (bool, Option<f64>) {
            // qualifying data is a nonzero available or used reading:
            // a pool at exactly 100% (zero available) still qualifies.
            let ever = h.window_entries(WINDOWS[2]).iter().any(|e| {
                e.sample.gauge(avail).unwrap_or(0.0) > 0.0
                    || e.sample.gauge(used).unwrap_or(0.0) > 0.0
            });
            // the usage shown is the latest in-window reading of the usage gauge:
            // the parse boundary drops non-finite values, so a NaN
            // blink leaves the prior reading in place
            let u = h
                .window_entries(WINDOWS[2])
                .iter()
                .rev()
                .find_map(|e| e.sample.gauge(usage));
            (ever, u)
        };
    let (ever, u) = ever_live(
        "sglang:mamba_available_tokens",
        "sglang:mamba_used_tokens",
        "sglang:mamba_usage",
    );
    if ever {
        let used = h.sum_gauge(fam("sglang:mamba_used_tokens"));
        let avail = h.sum_gauge(fam("sglang:mamba_available_tokens"));
        let total = used.zip(avail).map(|(u, a)| u + a);
        let usage = total
            .filter(|t| *t > 0.0)
            .map(|t| (used.unwrap_or(0.0) / t).clamp(0.0, 1.0))
            .unwrap_or(u.unwrap_or(0.0));
        pools.push(Pool {
            name: "mamba",
            usage,
            used,
            total,
            unit: "slots",
        });
    }
    let (ever, u) = ever_live(
        "sglang:swa_available_tokens",
        "sglang:swa_used_tokens",
        "sglang:swa_token_usage",
    );
    if ever {
        let used = h.sum_gauge(fam("sglang:swa_used_tokens"));
        let avail = h.sum_gauge(fam("sglang:swa_available_tokens"));
        let total = used.zip(avail).map(|(u, a)| u + a);
        let usage = total
            .filter(|t| *t > 0.0)
            .map(|t| (used.unwrap_or(0.0) / t).clamp(0.0, 1.0))
            .unwrap_or(u.unwrap_or(0.0));
        pools.push(Pool {
            name: "SWA",
            usage,
            used,
            total,
            unit: "slots",
        });
    }
    // host (L2) tier pool — present once the engine allocates host capacity
    if let Some((u, used, total)) = host_pool(h) {
        pools.push(Pool {
            name: "host",
            usage: u,
            used: Some(used),
            total: Some(total),
            unit: "tokens",
        });
    }

    let running_now = running.unwrap_or(0.0);
    let gen_progress = match (h.gauge_pred(fam("sglang:decode_sum_seq_lens")), running_now) {
        (Some(s), r) if r > 0.0 => Some(s / r),
        _ => None,
    };

    let decode_rate = rate_triple(
        h,
        fam_labeled("sglang:realtime_tokens_total", "mode", "decode"),
    );
    let prefill_rate = rate_triple(
        h,
        fam_labeled("sglang:realtime_tokens_total", "mode", "prefill_compute"),
    );
    let evict_rate = h.rate_sum(fam("sglang:evicted_tokens_total"), WINDOWS[2]);
    let retract_rate = h.rate_sum(fam("sglang:num_retracted_reqs"), WINDOWS[2]);
    let http_503_rate = h.rate_sum(
        fam_labeled("sglang:http_responses_total", "status_code", "503"),
        WINDOWS[2],
    );
    let cpu_tokenizer = h.rate_sum(
        fam_labeled("sglang:process_cpu_seconds_total", "component", "tokenizer"),
        WINDOWS[2],
    );
    let cpu_detokenizer = h.rate_sum(
        fam_labeled(
            "sglang:process_cpu_seconds_total",
            "component",
            "detokenizer",
        ),
        WINDOWS[2],
    );
    let cpu_scheduler = h.rate_sum(
        fam("sglang:scheduler_process_cpu_seconds_total"),
        WINDOWS[2],
    );

    let ttft = latency_triple(h, "sglang:time_to_first_token_seconds");
    let itl = latency_triple(h, "sglang:inter_token_latency_seconds");
    let e2e = latency_triple(h, "sglang:e2e_request_latency_seconds");
    let queue_time = latency_triple(h, "sglang:queue_time_seconds");
    let prompt_len_p50 = h.hist_quantile(fam("sglang:prompt_tokens_histogram"), 0.50, WINDOWS[2]);
    let prompt_len_p95 = h.hist_quantile(fam("sglang:prompt_tokens_histogram"), 0.95, WINDOWS[2]);

    let graphs = h.scan().cloned()?;
    let stalls = fold_stalls(&graphs.stall, &graphs.dt);
    let decode_instant = graphs.decode.vals.last().copied();
    let prefill_instant = graphs.prefill.vals.last().copied();

    let mut model = String::from("?");
    let mut engine = String::from("?");
    if let Some(k) = last
        .sample
        .simple
        .keys()
        .find(|k| k.name == "sglang:num_running_reqs")
    {
        for (lk, lv) in &k.labels {
            match lk.as_str() {
                "model_name" => model = lv.clone(),
                "engine_type" => engine = lv.clone(),
                _ => {}
            }
        }
    }

    Some(Derived {
        model,
        engine,
        running,
        queue,
        subqueues,
        decode_rate,
        prefill_rate,
        pools,
        cache_hit: h.gauge_pred(fam("sglang:cache_hit_rate")),
        spec_accept: h.gauge_pred(fam("sglang:spec_accept_rate")),
        spec_accept_len: h.gauge_pred(fam("sglang:spec_accept_length")),
        ttft,
        itl,
        e2e,
        queue_time,
        prompt_len_p50,
        prompt_len_p95,
        stalls,
        evict_rate,
        retract_rate,
        http_503_rate,
        http_active: h.gauge_pred(fam("sglang:http_requests_active")),
        gen_throughput_gauge: h.gauge_pred(fam("sglang:gen_throughput")),
        l2_device: l2_hit_rate(h, "device_hit"),
        l2_host: l2_hit_rate(h, "host_hit"),
        l2_wb: h.rate_sum(fam("sglang:hicache_backup_tokens_total"), WINDOWS[2]),
        l2_rb: h.rate_sum(fam("sglang:load_back_tokens_total"), WINDOWS[2]),
        l2_drop: h.rate_sum(fam("sglang:hicache_dropped_tokens_total"), WINDOWS[2]),
        new_token_ratio: h.gauge_pred(fam("sglang:new_token_ratio")),
        gen_progress,
        cpu_tokenizer,
        cpu_detokenizer,
        cpu_scheduler,
        decode_instant,
        prefill_instant,
        graphs,
        window_focus,
    })
}
