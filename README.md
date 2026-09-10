# sgtop

Live terminal dashboard for a sglang inference server: prefill/decode
histograms, memory pool occupancy, latency percentiles. Portable binary,
no storage, no daemon.

<p align="center">
  <img src="media/screenshot.png" alt="sgtop dashboard: prefill and decode
  histograms with tok/s gridlines, red stall ticks, memory-pool lanes, and
  latency percentiles" width="100%">
</p>

<p align="center">
  <img src="media/screenshot-load.png" alt="sgtop under sustained load:
  three stalls in 60s flagged in the hero, evictions/s counter lit, mamba
  pool churning, speed-per-stream graph tracking per-stream decode bursts"
  width="100%">
</p>

<p align="center">
  <img src="media/screenshot-stress.png" alt="sgtop under stress: queue
  non-empty, four stalls in 60s, mamba pool at 100% with the memory-pool
  alarm lit, TTFT p95 in red, prefill bursts against the 1k-10k gridline
  ladder" width="100%">
</p>

## Features

- **Rate histograms.** Prefill and decode tok/s on a fixed 60s wall-clock
  axis, newest sample pinned to the right edge. Dashed gridlines at every
  10/100/1k… multiple through the observed peak, anchored to the session
  peak (re-anchors on engine restart).
- **Stall detection.** Decode-rate collapse below 25% of the running median
  while prefill is active and requests are in flight; red ticks mark the
  intervals on the decode graph. The seesaw — a prefill spike riding a
  decode dip — is the long-prompt-freezes-everyone signature.
- **Memory pools.** KV, mamba, SWA and host (L2) tiers as occupancy lanes,
  with percentages plus absolute counts (`124k / 655k`). Counts are
  aggregated across TP ranks per metric type without double-counting the
  logical KV capacity (`max_total_num_tokens` is exported per rank but is
  one logical pool). KV/mamba full = requests wait; host full is normal
  (LRU prefix cache — watch evictions/s instead).
- **Latency.** TTFT / ITL / e2e / queue time as p50/p95/p99 per window. Each
  window column is a fresh quantile computed from histogram bucket deltas
  over 5s/15s/60s — no smoothing across windows; the cumulative snapshot is
  only a labeled fallback when a window is sparse.
- **Health counters.** Evictions/s, retractions/s, 503/s, retracted
  requests, L2 (host-tier) drops, open HTTP connections, tokenizer/
  detokenizer/scheduler CPU cores.
- **Two phase vocabularies.** Graph titles gloss prefill as *prompt
  processing* and decode as *token generation* (llama.cpp terms); `e` opens
  a full glossary/explain screen.

## Quick start

Install from prebuilt binaries (Linux x86_64/arm64, Windows x86_64/arm64,
macOS universal):

```
curl -LO https://github.com/mratsim/sgtop/releases/latest/download/sgtop-0.1.0-linux-x86_64.tar.gz
# or: go to https://github.com/mratsim/sgtop/releases/latest
tar -xzf sgtop-0.1.0-linux-x86_64.tar.gz
sudo mv sgtop-0.1.0-linux-x86_64/sgtop /usr/local/bin/
```

or build from source:

```
cargo install --path .      # installs `sgtop`
```

then:

```
sgtop                       # watches sglang's default port (30000)
sgtop --url http://localhost:30000 --interval 1.0
sgtop --url https://gpu-box.home.example.org --insecure   # self-signed proxy
sgtop --api-key sk-...                                    # if server uses --api-key
sgtop --once                # one text snapshot, for scripts/cron
```

## Keys

| Key | Action |
|---|---|
| `q` / `Esc` | quit |
| `space` | pause scraping |
| `c` | toggle compact / full layout |
| `g` | toggle graphs |
| `1` `2` `3` | graphs follow the 5s / 15s / 60s window |
| `t` | cycle theme (gruvbox, catppuccin, tokyonight) |
| `e` | explain screen |
| `?` | keymap |
| `+` / `-` | poll interval (clamped 0.5–10s) |
| `↑` `↓` | scroll the explain screen |

Layout auto-fits: full at ≥26 rows × ≥90 cols, compact below (subtitles
collapse first, then graphs). Fits half or quarter of a 1920×1080 screen.
