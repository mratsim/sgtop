use std::collections::BTreeMap;

/// Identity of one time series: metric family name + its (sorted) labels.
/// For histograms, the `le` label is stripped from bucket samples and the
/// `_sum`/`_count` suffixes are folded into the parent series key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub name: String,
    pub labels: Vec<(String, String)>,
}

impl SeriesKey {
    pub fn has_label(&self, k: &str, v: &str) -> bool {
        self.labels.iter().any(|(lk, lv)| lk == k && lv == v)
    }
}

/// Cumulative histogram state at one scrape, for one label combination.
#[derive(Debug, Clone, Default)]
pub struct HistSeries {
    /// Upper bounds, ascending; last entry is `f64::INFINITY`.
    pub le: Vec<f64>,
    /// Cumulative counts, parallel to `le`.
    pub counts: Vec<f64>,
    pub count: u64,
}

/// One scrape: simple gauge/counter series plus histogram families.
#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub simple: BTreeMap<SeriesKey, f64>,
    pub hist: BTreeMap<SeriesKey, HistSeries>,
}

impl Sample {
    pub fn gauge(&self, name: &str) -> Option<f64> {
        self.simple.iter().find(|(k, _)| k.name == name).map(|(_, v)| *v)
    }
}

#[derive(Default)]
struct HistBuilder {
    buckets: Vec<(f64, f64)>,
    count: u64,
}

/// Parse a Prometheus text exposition payload into a [`Sample`].
///
/// Handles `# HELP`/`# TYPE` headers, `name{labels} value` records, label
/// escape sequences (`\\`, `\"`, `\n`), optional timestamps, and histogram
/// bucket ladders (`_bucket` with `le`, `_sum`, `_count`).
pub fn parse(body: &str) -> Sample {
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut simple = BTreeMap::new();
    let mut hists: BTreeMap<SeriesKey, HistBuilder> = BTreeMap::new();

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# ") {
            let mut it = rest.split_whitespace();
            if it.next() == Some("TYPE") {
                if let (Some(name), Some(ty)) = (it.next(), it.next()) {
                    types.insert(name.to_string(), ty.to_string());
                }
            }
            continue;
        }

        let (name, label_src, value_src) = match line.find('{') {
            Some(open) => match line[open..].find('}') {
                Some(close_rel) => {
                    let close = open + close_rel;
                    (
                        &line[..open],
                        Some(&line[open + 1..close]),
                        line[close + 1..].trim(),
                    )
                }
                None => continue,
            },
            None => match line.split_once(char::is_whitespace) {
                Some((n, v)) => (n, None, v.trim()),
                None => continue,
            },
        };

        let value = match parse_value(value_src) {
            Some(v) => v,
            None => continue,
        };

        let labels = label_src.map(parse_labels).unwrap_or_default();

        // Histogram families announce themselves via `# TYPE <base> histogram`;
        // their samples are the base name suffixed with _bucket/_sum/_count.
        let (base, part, le) = if let Some(b) = name.strip_suffix("_bucket") {
            let le = labels
                .iter()
                .find(|(k, _)| k == "le")
                .and_then(|(_, v)| parse_value(v));
            (b, 0, le)
        } else if let Some(b) = name.strip_suffix("_sum") {
            (b, 1, None)
        } else if let Some(b) = name.strip_suffix("_count") {
            (b, 2, None)
        } else {
            (name, 3, None)
        };
        let is_hist = part < 3
            && types.get(base).map(String::as_str) == Some("histogram");

        if is_hist {
            let key = SeriesKey {
                name: base.to_string(),
                labels: labels.into_iter().filter(|(k, _)| k != "le").collect(),
            };
            let b = hists.entry(key).or_default();
            match part {
                0 => b.buckets.push((le.unwrap(), value)),
                // _sum is parsed and discarded: mean latency is not displayed
                1 => {}
                _ => b.count = value.max(0.0) as u64,
            }
        } else {
            let key = SeriesKey { name: name.to_string(), labels };
            simple.insert(key, value);
        }
    }

    let hist = hists
        .into_iter()
        .map(|(k, b)| {
            let mut buckets = b.buckets;
            buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (le, counts): (Vec<f64>, Vec<f64>) =
                buckets.into_iter().unzip();
            (
                k,
                HistSeries { le, counts, count: b.count },
            )
        })
        .collect();

    Sample { simple, hist }
}

fn parse_value(s: &str) -> Option<f64> {
    // value may be followed by an optional timestamp
    let tok = s.split_whitespace().next()?;
    match tok {
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        "Nan" | "NaN" | "nan" => Some(f64::NAN),
        v => v.parse().ok(),
    }
}

fn parse_labels(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    loop {
        // skip separators
        while matches!(chars.peek(), Some(',') | Some(' ') | Some('\t')) {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        let mut name = String::new();
        while let Some(&c) = chars.peek() {
            if c == '=' {
                break;
            }
            name.push(c);
            chars.next();
        }
        if chars.next() != Some('=') {
            break;
        }
        if chars.next() != Some('"') {
            break;
        }
        let mut value = String::new();
        loop {
            match chars.next() {
                Some('"') | None => break,
                Some('\\') => match chars.next() {
                    Some('n') => value.push('\n'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some(other) => value.push(other),
                    None => break,
                },
                Some(c) => value.push(c),
            }
        }
        out.push((name, value));
    }
    out
}
