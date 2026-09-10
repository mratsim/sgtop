use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;

use sgtop::{args::Args, metrics, scrape, ui};

fn main() -> Result<()> {
    let args = Args::parse();
    let url = scrape::metrics_url(&args.url);

    if args.once {
        let mut h = sgtop::history::History::default();
        // two scrapes so windows have a base for rates/quantiles
        for (i, wait) in [0.0, args.interval.clamp(0.5, 10.0)]
            .into_iter()
            .enumerate()
        {
            if i > 0 {
                std::thread::sleep(Duration::from_secs_f64(wait));
            }
            let body = scrape::fetch_once(&url, args.api_key.as_deref(), args.insecure)?;
            h.push(Instant::now(), metrics::parse(&body)?);
        }
        let mut peaks = sgtop::derive::Peaks::default();
        sgtop::once::print_once(&h, &mut peaks)?;
        return Ok(());
    }

    let shared = scrape::Shared::new();
    shared.interval_ms.store(
        (args.interval.clamp(0.5, 10.0) * 1000.0) as u64,
        Ordering::Relaxed,
    );
    scrape::spawn_scraper(shared.clone(), url, args.api_key.clone(), args.insecure);

    let mut terminal = ratatui::init();
    let mut ui_state = ui::Ui::new(&args);
    let result = ui::run(&mut terminal, shared, &mut ui_state);
    ratatui::restore();
    result
}
