use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::history::History;
use crate::metrics;

pub struct Shared {
    pub history: Mutex<History>,
    pub last_ok: Mutex<Option<Instant>>,
    pub last_error: Mutex<Option<String>>,
    pub paused: AtomicBool,
    /// scraper poll interval, live-adjustable from the TUI
    pub interval_ms: std::sync::atomic::AtomicU64,
    /// session peak rates (see derive::Peaks)
    pub peaks: Mutex<crate::derive::Peaks>,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            history: Mutex::new(History::default()),
            last_ok: Mutex::new(None),
            last_error: Mutex::new(None),
            paused: AtomicBool::new(false),
            interval_ms: std::sync::atomic::AtomicU64::new(1000),
            peaks: Mutex::new(crate::derive::Peaks::default()),
        })
    }
}

fn agent(insecure: bool, timeout: Duration) -> ureq::Agent {
    let builder = ureq::AgentBuilder::new().timeout(timeout);
    if insecure {
        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            Arc::new(danger::NoVerifier);
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let cfg = rustls::ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(rustls::ALL_VERSIONS)
            .expect("tls versions")
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        builder.tls_config(Arc::new(cfg)).build()
    } else {
        builder.build()
    }
}

mod danger {
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, Error, SignatureScheme};

    /// Accepts every server certificate. Only reachable via `--insecure`,
    /// which exists for homelab endpoints behind self-signed certs.
    #[derive(Debug)]
    pub struct NoVerifier;

    impl ServerCertVerifier for NoVerifier {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            verify(message, cert, dss)
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            verify(message, cert, dss)
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    fn verify(
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        let algs = &rustls::crypto::ring::default_provider().signature_verification_algorithms;
        rustls::crypto::verify_tls12_signature(message, cert, dss, algs)
            .or_else(|_| rustls::crypto::verify_tls13_signature(message, cert, dss, algs))
    }
}

/// Normalize a user-supplied base URL into a full /metrics URL.
pub fn metrics_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/metrics") {
        base.to_string()
    } else {
        format!("{base}/metrics")
    }
}

pub fn fetch_once(url: &str, api_key: Option<&str>, insecure: bool) -> Result<String> {
    let agent = agent(insecure, Duration::from_secs(5));
    fetch(&agent, url, api_key)
}

pub fn spawn_scraper(shared: Arc<Shared>, url: String, api_key: Option<String>, insecure: bool) {
    let agent = agent(insecure, Duration::from_secs(5));
    std::thread::spawn(move || loop {
        if !shared.paused.load(Ordering::Relaxed) {
            match fetch(&agent, &url, api_key.as_deref()) {
                Ok(body) => match metrics::parse(&body) {
                    // a rejected payload (e.g. series cap) leaves history and peaks
                    // untouched so the UI goes stale under the error banner
                    Ok(sample) => {
                        if let Ok(mut h) = shared.history.lock() {
                            h.push(Instant::now(), sample);
                        }
                        if let Ok(mut p) = shared.peaks.lock() {
                            if let Ok(h) = shared.history.lock() {
                                crate::derive::update_peaks(&mut p, &h);
                            }
                        }
                        if let Ok(mut t) = shared.last_ok.lock() {
                            *t = Some(Instant::now());
                        }
                        if let Ok(mut e) = shared.last_error.lock() {
                            *e = None;
                        }
                    }
                    Err(e) => {
                        if let Ok(mut slot) = shared.last_error.lock() {
                            *slot = Some(format!("{e:#}"));
                        }
                    }
                },
                Err(e) => {
                    if let Ok(mut slot) = shared.last_error.lock() {
                        *slot = Some(format!("{e:#}"));
                    }
                }
            }
        }
        let ms = shared
            .interval_ms
            .load(Ordering::Relaxed)
            .clamp(250, 10_000);
        std::thread::sleep(Duration::from_millis(ms));
    });
}

fn fetch(agent: &ureq::Agent, url: &str, api_key: Option<&str>) -> Result<String> {
    let mut req = agent.get(url);
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let resp = req.call().context("scrape failed")?;
    let mut reader = resp.into_reader().take(64 * 1024 * 1024);
    let mut body = String::new();
    reader
        .read_to_string(&mut body)
        .context("reading scrape body")?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_normalization() {
        assert_eq!(
            metrics_url("http://localhost:30000"),
            "http://localhost:30000/metrics"
        );
        assert_eq!(
            metrics_url("http://localhost:30000/"),
            "http://localhost:30000/metrics"
        );
        assert_eq!(
            metrics_url("https://example.org/metrics"),
            "https://example.org/metrics"
        );
    }
}
