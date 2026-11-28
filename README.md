# Summon

**Your GPU. Your model. Your API.**  ·  v1.2.0 ([changelog](CHANGELOG.md))

One command turns your NVIDIA GPU into a private AI with an OpenAI-compatible API. Summon picks the model, weights and context that fit your card, verifies the download, keeps the port private and tests tool calling before it hands you the URL. No cloud, no API bill, nobody else's servers.

![Summon: Nanbeige on llama.cpp, Qwen3.8 27B on Ollama](docs/img/summon-hero.png)

## Install

**Linux** (Ubuntu/Debian, rented GPU pods):

```bash
curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash
```

**Windows 10/11** (PowerShell; runs Summon inside WSL 2 with your GPU):

```powershell
irm https://raw.githubusercontent.com/codyvolckmar/summon/main/docs/install.ps1 | iex
```

**Just looking?** The full wizard, nothing installed:

```bash
curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash -s -- --dry-run
SUMMON_GPU="RTX 3090:24576" ./summon.sh --dry-run     # preview a card you don't have
```

Piping a script into a shell gives it root. Read [summon.sh](summon.sh) first; it's one commented file.

## Models

**New:** four new models — Bonsai 2.5 (ternary + MTP), Granite 4.2 30B (IBM reasoning), DaisyRose 27B (RL-tuned agents) and Inkling 2 Mini (the first open looped transformer, 1M context). Plus OBLITERATUS's abliterated models, up to the 748B DeepSeek V4.1 Flash — see [OBLITERATUS.md](OBLITERATUS.md).

| Model | Engine | 24 GB (RTX 3090/4090) | 48 GB (RTX A6000) |
|---|---|---|---|
| **Nanbeige 4.3 33B**: agentic coding, Artificial Analysis index 20 (top of the ≤40B class) | llama.cpp | up to 48K | 128K (Q6_K) · 256K (Q3_K_M) |
| **Qwen3.8 27B pi**: Qwen3.8 27B + a LoRA RL-tuned to drive the [pi](https://github.com/earendil-works/pi) coding agent, thinking off (vendor benchmarks) | llama.cpp | ~48K (Q4_K_M) | 256K, 8-bit weights |
| **Ornith 1.5 35B**: fast MoE, ~3B active (vendor benchmarks) | llama.cpp | 128K with experts in RAM (needs ~30 GB free RAM) | 256K, 8-bit weights |
| **Ornith 1.5 9B**: fast and light (vendor benchmarks) | llama.cpp | 256K, 8-bit weights | 256K, 8-bit weights |
| **ThinkingCap Qwen3.8 27B**: Qwen3.8 27B that thinks ~37% less (vendor: -0.86 pts), built-in MTP head. PolyForm Small Business licence | llama.cpp | 128K, IQ4_XS | 128K, 8-bit weights |
| **Ternary Bonsai 2 27B**: Qwen3.8 27B in ternary weights, 7.6 GB (vendor: 98% of FP16 across 14 benchmarks). Runs on 16 GB cards at 128K | llama.cpp | 256K (q8_0 cache) | 256K (f16 cache) |
| **Ternary Bonsai 2.5 27B**: the agentic release — same ternary weights + an in-file MTP head (vendor: FP16-level tool calling, 1.5–1.8x faster decode). Stock llama.cpp b11556+ | llama.cpp | 128K | 256K |
| **Granite 4.2 30B**: IBM's dense 30B reasoning model (vendor: SWE-bench Verified 57, AIME25 89). Heavy KV — 128K wants a q8_0 cache | llama.cpp | 96K (Q3_K_M, q8_0 cache) | 128K, 8-bit weights |
| **DaisyRose 27B**: Qwen3.8-27B + RL-only tuning on long-horizon agent tasks (vendor: Terminal-Bench 4.1 5.6 → 12.0), in-file MTP layer | llama.cpp | 128K (Q4_K_M, q8_0 cache) | 256K, 8-bit weights |
| **Inkling 2 Mini**: Thinking Machines' looped MoE, 26B / 3.4B active ×3 passes (vendor: SWE-bench Verified 79.2, TB 2.1 71.6 — top of its class), DSpark draft, 1M context. Sandboxed agents only | llama.cpp | 128K, IQ4_NL | up to 1M (q8_0 cache) |
| **Qwen3.8 27B OBLITERATED**: uncensored Qwen3.8 (OBLITERATUS V3 abliteration; vendor MMLU -2.1pp), thinking off, served with Qwen3.8's own template so tool calls work | llama.cpp | 128K, Q4_K_M | 256K, 8-bit weights |
| **Ornith 1.5 9B OBLITERATED**: uncensored Ornith 9B (vendor MMLU -4pp), thinking off | llama.cpp | 256K, 8-bit weights | 256K, 8-bit weights |
| **feyospace-s1 OBLITERATED**: uncensored ~28B tuned for cybersecurity (pentests, CTFs, vuln research; vendor WMDP-Cyber 63.8%). Isolate its tools | llama.cpp | 128K, Q4_K_M | 256K, 8-bit weights |
| **DeepSeek V4.1 Flash OBLITERATED**: uncensored 748B MoE (refusals reduced, not removed; MMLU 83.6%), 2.66-bit GGUF, 237 GB download. Needs a 96 GB card + ~130 GB free RAM, or ~160 GB of VRAM across cards | llama.cpp | not offered | not offered |
| **Ollama models**: Qwen3.8 27B, Qwen3.6 35B-A3B, Gemma 4 31B, K2 Horizon, Muse Glimmer… plus CPU-friendly picks (gpt-oss 20B, Qwen3 30B-A3B, Phi-4 14B, Qwen2.5 14B) for a RAM-only box; and an Ollama build of Inkling 2 Mini (IQ4_XS + vision, ~20 GB at 128K, needs Ollama 0.40.18+) | Ollama | Qwen3.8 27B at 64K | most at 128K, 8-bit |

## What should I rent?

| GPU (VRAM) | Lium price per hour | What Summon runs on it |
|---|---|---|
| RTX 4060 Ti / 5060 Ti / 5070 Ti (16 GB) | $0.12–0.30 | Ternary Bonsai 2 27B at 128K (the only 27B that fits). 12 GB cards are not offered |
| RTX 3090 / 4090 (24 GB) | $0.15–0.64 (3090), $0.35–0.55 (4090) | Ornith 1.5 9B at 128K, 8-bit. Ternary Bonsai 2 27B at 256K, Bonsai 2.5 at 128K. ThinkingCap / Qwen3.8 / DaisyRose at 128K, 4-bit. Inkling 2 Mini at 128K, 4-bit. Ornith 35B with experts in RAM |
| **L40 / L40S / RTX A6000 (48 GB)** | **$0.33 (L40), $0.38 (L40S)**, $0.42–0.67 (A6000) | **Best value.** Nanbeige 33B at 128K, 6-bit; ThinkingCap, Qwen3.8, Ornith 35B at 128K, 8-bit (~42 tok/s measured on an L40); Granite 4.2 30B, DaisyRose and Inkling 2 Mini at 128K–256K, 8-bit |
| RTX PRO 6000 Blackwell (96 GB) | $1.60–1.99 | Every 27–35B model at 256K, 8-bit; Nanbeige up to 384K at 8-bit (512K at 3-bit). DeepSeek V4.1 Flash 748B too, if the pod has ~130 GB free RAM (experts in RAM) |

Prices are [Lium](https://lium.io/pricing)'s live asks per GPU on 2026-11-20 and change with supply. Check any card first: `SUMMON_GPU="L40:46068" ./summon.sh --dry-run`.

## Usage

```bash
./summon.sh [PORT]                          # the wizard: one menu with every model, sized to your card
./summon.sh --auto [PORT]                   # just go: the recommended model, no questions (or GO=1)
./summon.sh status                          # what's running, health, last speed, GPU memory
./summon.sh nanbeige|qwen38pi|ornith35b|ornith9b|thinkingcap|bonsai27b|bonsai25|granite42|daisyrose|inklingmini|qwen38obl|ornith9bobl|feyospace|deepseek41obl [PORT]   # skip the menu
ONLY=1,3 ./summon.sh [PORT]                 # pull several Ollama models at once (rows in list order)
./summon.sh --dry-run [PORT]                # plan only, installs nothing
./summon.sh --connect                       # chat page, API key and client setup of the running server
TUNNEL=1 ./summon.sh [PORT]                 # also get a public HTTPS URL (Cursor, phones)
./summon.sh --lock [PORT]                   # firewall the port to your current IP, on top of the key
./summon.sh --uninstall [llamacpp|ollama]   # remove an engine and its model files
./summon.sh --help                          # every option (CTX, QUANT, ALLOW_IP, SPEC, ...)
```

When it's done, Summon prints:
- the **chat page** (llama-server's built-in web chat: open it in a browser, paste the key once);
- the **base URL, model name, API key and context window**;
- **copy-paste setup for Hermes Agent, pi, opencode, OpenClaw and Cursor**, with your values filled in.

`./summon.sh --connect` prints it all again.

**Access:** llama.cpp models need an **API key**, generated once and kept across reruns (`API_KEY=...` to set your own, `--new-key` to rotate). On your own PC the server only listens on `127.0.0.1`. On a rented pod the port is open and the key is the lock; `TUNNEL=1` adds a free HTTPS URL through a Cloudflare quick tunnel. Cursor needs that, because it calls models from its own servers. `ALLOW_IP=1.2.3.4` adds a firewall on top. Ollama has no API keys, so Ollama models keep the IP lock: firewalled to your SSH IP on rented pods, `127.0.0.1` on your own PC.

**Reboots:** llama.cpp runs in Docker and comes back on its own. Ollama runs as the `summon-ollama` systemd service, so it starts on boot too and re-applies its IP lock (iptables rules don't survive a reboot). Machines without systemd, like many pod containers, fall back to a `screen` session: rerun Summon after a reboot there.

## How it decides

- **Only what fits.** The menu lists only models that fit this machine's VRAM (summed over all cards) and, for MoE models with experts in RAM, its free RAM. The rest are named in one line and left out.
- **Context first, then precision.** Every token reads all the weights plus the part of the cache in use, so reserving a long context costs VRAM, not speed. Summon defaults to 128K with the most precise weights that still fit, and never uses a 4-bit cache.
- **Never past training.** The context menu goes as far as the card allows, but never beyond what the model was trained for (read from its GGUF or config): 256K for the Qwen3.8, Ornith and feyospace models, 512K for Nanbeige, 1M for DeepSeek V4.1 Flash and Inkling 2 Mini. `CTX=` above that is refused, because quality degrades quietly past it.
- **Each model's own chat template** (llama.cpp `--jinja`), so native tool calls and thinking blocks work. That's why Nanbeige and Ornith run on llama.cpp: Ollama's builds of them ship without tool support.
- **Every layer on the GPU** (`--fit off`): if it doesn't fit, it fails loudly instead of silently crawling on the CPU.
- **Host-RAM prompt cache** so side requests (chat titles, subagents) don't evict a 200K-token conversation.
- **Speculative decoding** (v1.1.0): each model's draft plus n-gram drafts. Qwen3.8 uses its MTP head (+60-90% measured on base Qwen3.8), Ornith and ThinkingCap their built-in MTP layer (ThinkingCap: ~1.3x, measured by its makers), Nanbeige the optional Nanbeige4.1-3B draft model (~70% faster decoding, ~3x faster tool calls, community-measured). A draft that would cost weight precision on your card is offered, never forced. The model checks every drafted token, so quality holds on average, but outputs aren't bit-identical to a run without it: `SPEC=off` for reproducible runs.
- **MoE experts in system RAM** (v1.1.0): a mixture-of-experts model that doesn't fit VRAM (Ornith 35B on a 24 GB card) keeps its experts in RAM and caches the hot ones on the GPU (llama.cpp `--cpu-moe` + `--moe-cache-mib`), shown as `+RAM` in the menu. Summon only offers it when the GPU cache holds at least 60% of the experts, and never pre-selects it.

## Tested on real hardware

**ThinkingCap Qwen3.8 27B on an NVIDIA L40 (48 GB)**, installed with the one-line installer. It ran Q8_0 weights at 128K context with an f16 cache, using the built-in MTP layer plus n-gram drafts for speculative decoding. The tool-call test passed.

| Request | Prompt | Generated | Decode | Draft acceptance |
|---|---|---|---|---|
| Short | 316 tokens | 46 tokens | **56.6 tok/s** | 82% (mean 3.46 tokens per step) |
| Long | 34,748 tokens at 1,646 tok/s | 2,235 tokens | **41.5 tok/s** | 49% (mean 2.48 tokens per step) |

The acceptance rates show speculative decoding is working: the model accepts about half of the drafted tokens even on a 35K-token context. No baseline with speculation off was recorded, so the exact speed-up on this card is still unmeasured.

**Nanbeige4.3 33B on an NVIDIA A100 80GB PCIe**, with Q6_K weights, a 256K context, a q8_0 cache, and speculative decoding from a 3B draft model plus n-gram drafts. Measured with `python3 tools/bench.py`.

| Workload (thinking off, median of 3) | Decode | Draft acceptance |
|---|---|---|
| Prose | **35.0 tok/s** | 46% |
| New code | **66.2 tok/s** (62–94) | 77% |
| Rewriting an existing file | **471 tok/s** | 100% |

| Prompt | Prefill | Time to first token | Decode | Needle at 10 / 50 / 90% depth |
|---|---|---|---|---|
| 8K tokens | 958 tok/s | 8 s | 45–62 tok/s | found / found / found |
| 32K tokens | 783 tok/s | 40 s | 26–37 tok/s | found / found / found |
| 127K tokens | 448 tok/s | 4.7 min | 14–19 tok/s | found / found / not run |

Prefill is compute-bound here: the card sat at 100% use and its 300 W power limit. A larger prompt batch (`-ub 2048 -b 2048`, +1.7 GB VRAM) gave 8–10% faster prefill (1,056 / 851 / 483 tok/s at 8K / 32K / 127K) with no decode loss. A 239K-token prompt took 12.2 minutes to prefill at 325 tok/s, then decoded at 8.8 tok/s, and the needle was still found.

**Ternary Bonsai 2 27B on the same A100**, installed with `./summon.sh bonsai27b` on stock llama.cpp: Q2_0 weights (7.6 GB), a 256K context with an f16 cache (24.4 GB of VRAM in use), and n-gram drafts. The tool-call test passed.

| Workload (thinking off, median of 2) | Decode | Draft acceptance |
|---|---|---|
| Prose | **65.5 tok/s** | 0% |
| New code | **66.5 tok/s** | 28% |
| Rewriting an existing file | **441 tok/s** | 100% |

| Prompt | Prefill | Time to first token | Decode | Needle at 50% depth |
|---|---|---|---|---|
| 8K tokens | 1,329 tok/s | 7 s | 64 tok/s | found |
| 32K tokens | 1,288 tok/s | 27 s | 59 tok/s | found |
| 131K tokens | 998 tok/s | 2.2 min | 47 tok/s | found |
| 247K tokens | 782 tok/s | 5.4 min | 39 tok/s | found |

Coherence, with thinking on: 24 of 24 graded runs passed, the same tasks as Nanbeige. A 2,040-word essay had no repeated passages. Next to Nanbeige on this card it answers about twice as fast, and its lead grows with the prompt: only 16 of its 64 layers keep a cache.

Coherence, with thinking on: it passed all 24 graded runs (12 tasks, twice each). The tasks were math, logic, two coding tasks checked against tests, strict JSON, an exact output format, a tool call, answering from a tool result, multi-turn recall, and saying so when a fact can't be known. In an earlier quick pass it missed the number-theory sum once and returned an empty answer after a tool result once. A 3,478-word essay had 99.3% distinct 4-grams and no repeated passages.

| | |
|---|---|
| ![Summon installing and serving ThinkingCap on an L40](docs/img/12-l40-thinkingcap-install.png) | ![llama-server timings: ~42 tok/s decode with MTP drafts](docs/img/13-l40-thinkingcap-speed.png) |

## Screenshots

| | |
|---|---|
| ![24 GB card](docs/img/06-model-menu-3090.png) | ![Context menu](docs/img/03-context-menu.png) |
| ![Weights menu](docs/img/04-weights-menu.png) | ![Dry run plan](docs/img/05-dry-run-plan.png) |
| ![Ollama pick](docs/img/09-ollama-menu.png) | ![Ollama plan](docs/img/10-ollama-plan.png) |

Regenerate them with `python3 tools/termshot.py` (drives the real script in a pseudo-terminal).

## Status

Young project, [MIT licensed](LICENSE). The Nanbeige llama.cpp setup has served real agent work on rented RTX A6000 pods, and ThinkingCap 27B has been run end to end on an L40 ([results](#tested-on-real-hardware)). The full wizard, the Windows/WSL path and the Ornith models have only been tested with dry runs and simulated GPUs so far. Ornith's benchmarks are the vendor's own. Issues and pull requests welcome.

Models belong to their creators and carry their own licences: Nanbeige (Apache-2.0), Ornith (MIT), Bonsai / Granite / DaisyRose / Inkling 2 Mini (Apache-2.0), Ollama models as listed. All four new models' benchmark numbers are the vendors' own; none of them has been run end to end on real hardware by Summon yet.

## Questions?

> **Cody Volckmar** · [cody.volckmar@gmail.com](mailto:cody.volckmar@gmail.com) · X: [x.com/Chillisock7](https://x.com/Chillisock7)
>
> Summon is a free, open-source side project, not a service. Questions, bugs and ideas: [GitHub issues](https://github.com/codyvolckmar/summon/issues). Want to build on it together? Say hi by email or on X.
