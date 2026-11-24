# Changelog

## Unreleased

- **Granite 4.2 30B** (`granite42`, ibm-granite, Apache-2.0): dense 30B reasoning model, official IBM GGUFs (Q8_0 down to Q3_K_M). Vendor: SWE-bench Verified 57.0, AIME25 89.17, RULER 128K 81.38. Native 128K context (no rope scaling — the 512K extension is a separate artifact); heavy KV: 256 KB/token at f16, so 128K wants q8_0 cache or a 48 GB card.

- **Ternary Bonsai 2.5 27B** (`bonsai25`, prism-ml, Apache-2.0): the agentic release of Bonsai 27B — same ternary Qwen3.8-27B weights (7.7 GB, `Q2_0-mtp`, stock llama.cpp b11556+) plus an in-file MTP head (vendor: 1.5-1.8x faster decode, FP16-level tool calling, SWE-bench Verified 75.6). 256K on a 24 GB card, 128K on 16 GB.

- **Four CPU-friendly models added to the Ollama menu** (fits a RAM-only box, e.g. 6-core Xeon + 27 GB RAM): gpt-oss 20B (rank 7, ~24 KB/token KV thanks to sliding-window attention — the fastest quality option on CPU), Qwen3 30B-A3B (rank 8, 3B active params), Phi-4 14B (rank 9) and Qwen2.5 14B (rank 10). Ranked below the GPU-first models so 24 GB+ cards keep their current recommendations.

- **Ternary Bonsai 2 27B** (`bonsai27b`, prism-ml, Apache-2.0): Qwen3.8 27B in ternary weights, 7.6 GB. Uses the `Q2_0` file, which runs on stock llama.cpp (b11556+); the smaller `PQ2_0`/`PTQ1_0` files still need PrismML's fork. 256K on a 24 GB card, 128K on 16 GB. No MTP layer, so n-gram drafts only. Tested on an A100 80GB: install, tool-call test, 24/24 coherence, needle recall.
- `tools/bench.py`: speed and coherence benchmark for any Summon server.
- **Model menu shows each model's longest context** on this machine (`max 384K`), so contexts past 128K are visible before the context step.
- **Just go:** `./summon.sh --auto` (or `GO=1`) asks nothing: the recommended model, context and weights for the machine, port 20001.
- **`./summon.sh status`:** engine, model, context, port, health, last decode speed, HTTPS URL and GPU memory in one screen.
- **Ollama survives reboots:** it runs as the `summon-ollama` systemd service (starts on boot, re-applies the IP lock). No systemd (many pod containers): the old `screen` session, and Summon says so.
- **Plainer wording:** tagline "Your GPU. Your model. Your API."; the banner drops the stars and the "summoning" line; menus ask "Which model do you want to run?". Screenshots and the hero image regenerated.
- **Context limits:** the context menu now goes past 256K for models trained further (Nanbeige 4.3: 512K, DeepSeek V4.1 Flash: 1M), shown only when they fit. `CTX=` above a model's trained context (read from its GGUF) is refused instead of silently degrading it.
- **"What should I rent?"** table on the site and in the README: Lium's prices for 24, 48 and 96 GB cards and what Summon runs on each.

## v1.2.0 (2026-11-20)

- Docs site, terminal UI (banner + whiptail theme) and all screenshots restyled: enterprise slate/blue theme (replaces purple).
- **Logic-precision fixes (full-script audit).**
  - `--lock` works with no server running (Ollama's screen session dies on a reboot): `./summon.sh --lock PORT` defaults to host mode; a llama.cpp container on that port still locks via DOCKER-USER. `--dry-run --lock` now previews instead of locking.
  - `ENGINE=ollama` on a card where no Ollama model fits now errors out instead of silently running a llama.cpp model; Ollama also never starts with zero models.
  - DSpark no longer receives `--spec-ngram-mod-*` flags (its makers measured no gain from them).
  - The host-RAM prompt cache is half the RAM left after the MoE experts, never negative, with no fixed 8 GB floor (which could exceed free RAM after the expert deduction).
  - Leading-zero ports (`08001`) are accepted and normalized; `read` in the plain-text menu/confirm fallbacks no longer dies under `set -e` on piped stdin.
  - Small: menu descriptions live in the model definitions (`LC_BLURB`, can't drift from the id list), the Ollama plan prints the recommendation, the "locked" access line falls back to `SSH_CONNECTION`, the speed-reference quant falls back to the first quant's size, `/tmp/Modelfile.ctx` is removed after use, the EXIT trap is append-only, and the busy-port error shows the listening address (`ss -ltn`).
- **API keys instead of IP locking (llama.cpp).** A key is generated once, kept in the data folder across reruns and model switches, and passed to llama-server as a file (`--api-key-file`, so it never shows in `docker inspect`). Rented pods no longer break when your IP changes or the provider NATs traffic. `API_KEY=...` sets your own, `API_KEY=none` restores IP locking, `--new-key` rotates it, `ALLOW_IP` adds a firewall on top. Ollama models keep the IP lock (Ollama has no keys).
- **`TUNNEL=1`:** a free public HTTPS URL through a Cloudflare quick tunnel (`cloudflare/cloudflared:2026.11.1`). The raw port then only listens on 127.0.0.1. Cursor needs this.
- **Connect your tools.** The Done screen prints llama-server's browser chat page, the key, and copy-paste setup for Hermes Agent, pi, opencode, OpenClaw and Cursor with the real URL, model, key and context filled in. `./summon.sh --connect` prints it again (it also picks up a restarted tunnel's new URL).
- The tool-call test sends the key; the public address is detected via api.ipify.org (`PUBLIC_HOST=...` overrides).

- **OBLITERATUS models (uncensored abliterations, published Aug-Nov 2026).** Listed last, so they are never pre-selected:
  - llama.cpp: Qwen3.8 27B OBLITERATED (V3), Ornith 1.5 9B OBLITERATED, feyospace-s1 OBLITERATED (cybersecurity ~28B, mradermacher's GGUFs). All three use their built-in MTP layer.
  - Ollama (opt-in Hugging Face rows): Llama 3.1 8B Lexi V2 OBLITERATED, Qwen3 4B structured-output stage A OBLITERATED.
  - Qwen3.8 27B OBLITERATED's GGUFs ship a template without tool support, so Summon serves Qwen3.8's official template (pinned revision, sha256-checked) with thinking off and the card's repetition penalty 1.15.
  - DeepSeek V4.1 Flash OBLITERATED (748B MoE, HumanjavaEnterprises' 2.66-bit GGUF, 237 GB in 26 files): routed experts in RAM with a GPU expert cache, the 103 GB Engram table read from disk, the official template for DSML tool calls, and unsloth's DSpark drafter when it fits. Offered on a 96 GB card with ~130 GB free RAM, or ~160 GB of VRAM across cards (all on the GPUs).
  - Not added: Mistral 7B v0.3 (base model, no chat), GLM-4 9B chat (its GGUF lost the 128K RoPE scaling), Qwen3.8 27B E03 (already here via `robit/` builds; OBLITERATUS ships no GGUF).
- **The menu shows only models that fit** this machine (VRAM, and RAM for experts-in-RAM). The rest are listed in one line instead of as `TOO BIG` rows.
- llama.cpp catalog: per-model chat template override (`LC_TEMPLATE`), extra server arguments (`LC_ARGS`), file-name separator (`LC_FILE_SEP`), split GGUFs (`LC_PARTS`, every part verified, disk checked up front), weights kept on host/disk (`LC_HOST_PCT`) and DSpark drafters.
- Fix: with experts in RAM, the host-RAM prompt cache no longer counts the RAM the experts will use.
- `SUMMON_GPU="H100:81559:2"` previews multi-GPU machines in dry runs.

## v1.1.0 (2026-11-19)

- **Draft models (speculative decoding).** Each llama.cpp model gets its draft, combined with n-gram drafts:
  - Qwen3.8 27B pi: Qwen3.8's MTP head (ggml-org's separate file, 3 GB). Measured on base Qwen3.8 27B: +60% decode on an RTX 3090, 1.9x on code.
  - Ornith 1.5 35B / 9B: the MTP layer built into the model file, no extra download.
  - ThinkingCap Qwen3.8 27B (new on llama.cpp; the Ollama row stays): its built-in MTP layer, `--spec-draft-n-max 3` as its makers serve it (~1.3x measured on Q4_K_M). It thinks ~37% less than Qwen3.8, so the gains multiply. PolyForm Small Business licence; listed last, so it's never the default.
  - Nanbeige 4.3 33B: the Nanbeige4.1-3B draft model, padded once at install to the 33B's vocabulary (pad_draft.py, MIT, run in a throwaway container). Community-measured: ~70% faster decoding, ~3x faster tool calls.
  - The VRAM fit counts every draft. A draft that costs nothing is on by default; one that would lower the weights (Nanbeige's 3B on a 48 GB card: Q4_K_M instead of Q6_K at 128K) is asked about, with "No" pre-selected. `DRAFT=on|off` decides up front, `SPEC=off` turns all speculation off.
- **MoE experts in system RAM.** Mixture-of-experts models that don't fit VRAM run with their routed experts in RAM and the hot ones cached on the GPU (`--cpu-moe` + `--moe-cache-mib`, which llama.cpp leaves off by default). Ornith 1.5 35B now runs on a 24 GB card (Q6_K at 128K with ~30 GB free RAM). Offered only when the GPU cache holds >= 60% of the experts; shown as `+RAM`; never pre-selected.
- **Accuracy fix:** speculative decoding was described as lossless. Quality holds on average, but outputs aren't bit-identical (batched verification changes float rounding); docs and messages now say so.
- Dry runs can preview another machine's RAM: `SUMMON_RAM=65536`.
- llama.cpp models serve Prometheus metrics at `/metrics` (`--metrics`): tokens/s, draft acceptance, cache use. Same port and access rules as the API.

## v1.0.1 (2026-11-19)

- **One menu for every model.** llama.cpp and Ollama models share a single list, grouped under engine headers, with a description from each model's card and what fits on your card (weights ~VRAM@context, or TOO BIG). It fits on one screen; short terminals get a compact list.
- **The port is the last question**, after the model, context and weights.
- **New models:** Qwen3.8 27B pi (Qwen3.8 27B + harness-hub's pi-adapted LoRA, on llama.cpp) and `robit/qwen3.8-27b-e03-pi-adapted` (Ollama, Q8_0).
- **LoRA support** in the llama.cpp catalog: adapters are downloaded, verified and counted in the VRAM fit like the weights.
- Community and Hugging Face Ollama models are opt-in; `ONLY=1,3` pulls several Ollama models without the menu.
- `--version`; the version shows in the banner and the menu title.
- Fixes: menus no longer break on labels starting with `-`; prompts fall back to defaults when there is no terminal (e.g. `curl | bash` without a tty); dry runs flag Ollama models that don't fit.

## v1.0.0 (2026-11-18)

First public release: one-command setup for Nanbeige 4.3 33B, Ornith 1.5 35B/9B (llama.cpp) and a ranked Ollama list; dry runs, local/locked access, Linux and Windows (WSL 2) one-line installers, GitHub Pages site.
