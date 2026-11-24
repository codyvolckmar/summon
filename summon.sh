#!/usr/bin/env bash
# Summon - Your GPU. Your model. Your API. One-shot setup that turns an NVIDIA GPU (rented pod or your
# own PC) into a private, OpenAI-compatible AI endpoint: picks the model, weights and context that fit the card,
# verifies the download, keeps the port private and tests tool calling before handing you the URL.
#
# HOW IT WORKS
#   1. Detect   GPUs (VRAM summed over all cards), free RAM and whether you're on SSH (rented pod) or local.
#   2. Choose   One menu lists only the models that fit this machine; the rest are named on one line.
#               Then context (step 1) and weights (step 2): context first, then the most precise weights that
#               fit. Each model's draft for speculative decoding is switched on if it costs no precision.
#   3. Fetch    Docker + NVIDIA container toolkit if missing, then the model from Hugging Face: resumable, exact
#               byte count and GGUF header checked (full sha256 with --verify), split models file by file.
#   4. Serve    llama-server in Docker (container "summon"): every layer on the GPU (it fails loudly rather than
#               crawl on the CPU), or MoE experts in RAM with the hot ones cached on the GPU ("+RAM" in the
#               menu); each model's own chat template or a pinned fixed one; host-RAM prompt cache; /metrics.
#               Ollama models instead install Ollama (no Docker), pull the model and serve it with its context set.
#   5. Secure   llama.cpp: an API key (generated once, kept across reruns). Your own PC: 127.0.0.1 only. Rented
#               pod: port open, key required; TUNNEL=1 adds a free HTTPS URL instead. Ollama has no keys, so its
#               port is firewalled to your SSH IP (rented pods) or bound to 127.0.0.1 (your own PC).
#   6. Test     A tool call must come back as an OpenAI tool_calls entry.
#   7. Connect  Prints the browser chat page, the API key and copy-paste setup for Hermes Agent, pi, opencode,
#               OpenClaw and Cursor (again any time: --connect).
#   One engine runs at a time: starting one stops the other (they would fight over VRAM and the port).
#   Rerunning is cheap: downloaded models are kept, so switching models or settings only restarts the server.
#
# MODELS
#   llama.cpp, tuned per model (id in brackets):
#     Nanbeige 4.3 33B [nanbeige]      the agentic-coding daily driver (48 GB cards: 128-256K; trained for 512K)
#     Qwen3.8 27B pi [qwen38pi]        Qwen3.8 27B + a LoRA RL-tuned to drive the pi coding agent (thinking off)
#     Ornith 1.5 35B [ornith35b]       fast MoE (3B active), 256K with 8-bit weights on 48 GB
#     Ornith 1.5 9B [ornith9b]         fast and light: 256K with 8-bit weights on a 24 GB card
#     ThinkingCap 27B [thinkingcap]    Qwen3.8 27B that thinks ~37% less, built-in MTP head (PolyForm Small Business)
#     Ternary Bonsai 2 27B [bonsai27b] Qwen3.8 27B in ternary weights (7.6 GB): 256K on 24 GB cards, 128K on 16 GB
#     Uncensored (OBLITERATUS abliterations; never pre-selected), see OBLITERATUS.md:
#       [qwen38obl] Qwen3.8 27B  [ornith9bobl] Ornith 1.5 9B  [feyospace] feyospace-s1 (cybersecurity ~28B)
#       [deepseek41obl] DeepSeek V4.1 Flash 748B MoE (96 GB card + ~130 GB RAM, or ~160 GB VRAM across cards)
#   Ollama: a ranked list of other open models (Qwen3.8 27B, Qwen3.6 35B-A3B, Gemma 4 ...), sized to the card.
#
# COMMANDS
#   ./summon.sh [PORT]                    the wizard: one menu with every model that fits, then context/weights
#   ./summon.sh MODEL_ID [PORT]           skip the model menu, e.g. ./summon.sh thinkingcap 20000 (= ENGINE=...)
#   ./summon.sh ollama [PORT]             the menu, with the recommended Ollama model pre-selected
#   ./summon.sh --auto [PORT]             just go: no questions, the recommended model/context/weights, port 20001
#                                         (GO=1 does the same, e.g. curl ... | GO=1 bash)
#   ./summon.sh status                    what's running: model, context, port, health, last speed, GPU memory
#   ./summon.sh --dry-run [PORT]          the whole wizard, then print the plan: installs and downloads nothing
#   ./summon.sh --verify ...              also hash the whole model file with sha256 (slow)
#   ./summon.sh --connect                 show the chat page, API key and client setup of the running server
#   ./summon.sh --new-key MODEL_ID [PORT] start with a new API key (the old one stops working)
#   ./summon.sh --lock [PORT]             firewall the port to your current IP (also works with no server
#                                         running, e.g. after a reboot; ALLOW_IP=... to choose)
#   ./summon.sh --uninstall [llamacpp|ollama] [-y]   remove one engine (or both) and its model files
#   ./summon.sh --help | --version        this text / the version
#   The one-line installer saves this script as ~/.summon/summon.sh: rerun that for every command above.
#
# OPTIONS (environment variables, put them before the command; they skip the matching menu step)
#   Access
#     API_KEY=sk-...                use this API key (llama.cpp); API_KEY=none = no key, IP lock as for Ollama
#     TUNNEL=1                      public HTTPS URL via a Cloudflare quick tunnel (free, no account; needed for
#                                   Cursor). The URL changes when the tunnel restarts: --connect shows the current one
#     ALLOW_IP=1.2.3.4,5.6.7.0/24   firewall the port to these addresses (llama.cpp: on top of the key)
#     ALLOW_IP=any                  your own PC: listen on all interfaces (default there: 127.0.0.1 only)
#     PUBLIC_HOST=gpu.example.com   the address to print in client setup (default: public IP via api.ipify.org)
#   llama.cpp models
#     CTX=131072                    context window in tokens (skips step 1). At most what the model was trained
#                                   for: 256K for most, 512K for Nanbeige, 1M for DeepSeek (the menu offers up to that)
#     QUANT=Q8_0                    weights, e.g. Q8_0 Q6_K Q4_K_M IQ4_XS (skips step 2; must fit at CTX)
#     DRAFT=on|off                  the model's draft (MTP head, draft model or DSpark): on/off up front
#     SPEC=off                      no speculative decoding at all (no draft, no n-gram): reproducible runs and
#                                   speed baselines
#     THINK_BUDGET=4096             cap thinking tokens per answer
#     TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20   server sampling defaults (clients can still override per request)
#     VERIFY_SHA256=1               same as --verify
#   Ollama models
#     ONLY=1,3                      skip the menu, pull these rows (1 = Qwen3.8 27B ... in list order)
#     FAST=1                        try Q4 before Q8 (~1.7x decode speed, ~11 GB less VRAM)
#     OLLAMA_KV_CACHE_TYPE=q8_0     halve K/V memory (global: affects every model); q4_0 quarters it
#     OLLAMA_KEEP_ALIVE=10m         unload an idle model after 10 min (default: keep it loaded)
#   Dry-run previews
#     SUMMON_GPU="RTX 3090:24576"   pretend this card is installed; "H100:81559:2" = two of them
#     SUMMON_RAM=262144             pretend this much free RAM (MB), for models with experts in RAM
#
# EXAMPLES
#   CTX=131072 QUANT=Q8_0 ./summon.sh thinkingcap 20000      no menus: ThinkingCap, 128K, 8-bit weights
#   SPEC=off CTX=131072 QUANT=Q8_0 ./summon.sh thinkingcap 20000   same without speculation (speed baseline)
#   SUMMON_GPU="RTX PRO 6000:97887" SUMMON_RAM=262144 ./summon.sh --dry-run deepseek41obl   would DeepSeek fit?
#   TUNNEL=1 ./summon.sh thinkingcap 20000                   HTTPS URL for Cursor and phones, port stays private
#   ALLOW_IP=203.0.113.7 ./summon.sh nanbeige 20000          key + firewall to one address
#
# NOTES
#   Speculative decoding keeps quality on average but isn't bit-identical: verifying a batch of drafted tokens
#   changes float rounding, so near-tied tokens can flip. Use SPEC=off when you need reproducible outputs.
#   The API key travels in plain text over http://IP:PORT; use TUNNEL=1 (HTTPS) on untrusted networks.
#   Logs: docker logs -f summon. Speed and draft acceptance: http://<host>:PORT/metrics (send the key).
set -e
export DEBIAN_FRONTEND=noninteractive
SUMMON_VERSION=1.2.0

# ======================================================================================================
# llama.cpp models (one runs at a time; all share the image, container, data folder and server tuning)
# ======================================================================================================
LC_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b12429   # llama.cpp b12429 (2026-11-16), CUDA 12.8
LC_NAME=summon
LC_OLD_NAMES=(nanbeige)                         # container names used by earlier versions of this script
LC_CTX_TIERS=(1048576 524288 393216 327680 262144 196608 131072 98304 65536 49152 32768)   # each model uses those <= LC_CTX_TRAIN
LC_OVERHEAD_MB=3584                             # CUDA context + compute buffers at long context + ~1.5 GB margin
# The data folder is bind-mounted into the container. On Lium, /root is an encrypted FUSE mount that the pod's
# Sysbox runtime can't bind into nested containers, so the first folder that mounts wins. (Named after the first
# model for compatibility: earlier downloads are found again.)
LC_DIRS=(/root/.nanbeige /var/lib/nanbeige /var/lib/docker/nanbeige-data)
LC_MODEL_IDS=(nanbeige qwen38pi ornith35b ornith9b thinkingcap bonsai27b qwen38obl ornith9bobl feyospace deepseek41obl)   # order = menu order; first that reaches 128K is the default
lc_model() {   # $1 = model id: sets the LC_* settings of that model
  LC_LORA="" LC_EXTRA_BYTES=0   # optional LoRA adapter: "file|bytes|sha256|repo URL" (loaded with --lora)
  LC_FILE_SEP=-                 # weights file = $LC_FILE_PREFIX$LC_FILE_SEP$QUANT.gguf (mradermacher uses ".")
  LC_TEMPLATE=""                # optional chat template replacing the GGUF's own: "file|sha256|URL" (--chat-template-file)
  LC_BLURB=""                   # one-line menu description (the model's own summary, shortened; lives here so it can't drift from the id list)
  LC_ARGS=()                    # extra llama-server arguments (e.g. a repeat penalty the model card requires)
  LC_PARTS=()                   # split GGUF: "NNNNN-of-NNNNN|bytes|sha256" per file, named $prefix$sep$QUANT-$part.gguf
  LC_CTX_TRAIN=262144           # longest context the model was trained for (GGUF <arch>.context_length); never served beyond it
  LC_HOST_PCT=0                 # share of the weights llama.cpp keeps in host memory/disk (never VRAM), e.g. Engram tables
  # Speculative draft: LC_DRAFT_KIND = model (a small model, -md) | mtp (MTP head file, -md) | mtp-inline (MTP layer
  # inside the model file) | dspark (DeepSeek's DSpark drafter, -md) | "" (none). LC_DRAFT = "file|bytes|sha256|repo URL"; LC_DRAFT_KV_F16 = its cache per
  # token. LC_EXPERT_PCT = share of the weights that are MoE routed experts (they can live in system RAM).
  LC_DRAFT_KIND="" LC_DRAFT="" LC_DRAFT_KV_F16=0 LC_DRAFT_PAD=0 LC_DRAFT_TITLE="" LC_DRAFT_GAIN="" LC_EXPERT_PCT=0
  LC_DRAFT_NMAX=""   # tokens drafted per step (default: 4 for a draft model, 2 for MTP)
  case "$1" in
    nanbeige)
      # Verified against the model card and the GGUF's own Jinja template (2026-11-17): thinking on by default,
      # XML tool calls (Qwen3-Coder format). The card recommends preserve_thinking=True for code agents and
      # temp 1.0 / top_p 0.95 for agentic work. AA (independent): Intelligence Index 20 (top of the <=40B class),
      # lowest hallucination rate in it (18%), Terminal-Bench 4 0.30 (Qwen3.8 27B: 0.06).
      # Ollama's build of this model had no tool support in its template, hence llama.cpp.
      # Full attention in all 52 layers: 8 KV heads x (128 K + 128 V) -> ~110 MB of K/V per 1K tokens at q8_0.
      # RTX A6000 48 GB: 128K fits Q6_K, 256K only Q3_K_M ("low quality" per bartowski). 24 GB cards: ~48K.
      LC_TITLE="Nanbeige 4.3 33B" LC_ALIAS=nanbeige-33b
      LC_CTX_TRAIN=524288           # GGUF llama.context_length (read 2026-11-21)
      LC_BLURB="Lightweight agent for coding and research. AA index 20"
      LC_REPO=https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF/resolve/main
      LC_FILE_PREFIX=Nanbeige_Nanbeige4.3-33B
      LC_QUANTS=(   # bartowski's GGUFs, best first: "quant|bytes|sha256" (from the Hugging Face file list)
        "Q8_0|35195128960|8523452a975fa67d84a926c9a1bbdbb32b78b2a516dbf14d9360271fc53cd7f1"
        "Q6_K|29016857728|b11c9f21e69d896a1c20b12f88174f069e0e15623140733a49ff8e47c5541fbc"
        "Q5_K_M|25073368192|2b10206e3ba0a6a8f88b8c23fdbc49e9efccb3331564e711e4fc2901b18288cc"
        "Q4_K_M|20788825216|2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37"
        "IQ4_XS|18635041920|510cb2e07497ee3e9b7cb20aae3f50998adb29d95a71e363ba342544a47647a3"
        "IQ3_M|17802005632|7e4f92185c3b7917e68752364fbee2549b9a00b78fc3bbf31afa30952dd02d5e"
        "Q3_K_M|15919746176|671a48692387750eaa94f113f2635386d9d1c309b2d16c77b2890876d0e04e51"
        "IQ3_XS|15212665984|fdb7e52deb939c3f08e520c7dd968802c4cdb84c7d308637a989839d2916e1af"
        "IQ3_XXS|14489959552|5d959f2812f07632fc6e8de21373f90d12bda81d0178a2a8868fc3e6812a25da"
      )
      LC_KV_F16=$(( 52 * 8 * 256 * 2 ))            # 212992 bytes/token
      LC_TEMP=1.0 LC_TOP_P=0.95 LC_TOP_K="" LC_KWARGS='{"preserve_thinking": true}'
      # Draft: Nanbeige4.1-3B (same tokenizer; llama arch, 32 layers x 4 KV heads x 256 -> 64 KB/token at f16).
      # Its GGUF pads the vocabulary to 166,144 entries vs the 33B's 166,912, more than llama.cpp's 128 tolerance,
      # so it's padded once at install (pad_draft.py, MIT, from github.com/dedene/Nanbeige4.3-33B-DGX-Spark).
      # Measured there (DGX Spark, Q6_K, b11870): decode 7.3 -> 12.6 tok/s, tool calls 7.8 -> 24.4, prefill -15%.
      LC_DRAFT_KIND=model LC_DRAFT_PAD=1 LC_DRAFT_KV_F16=$(( 32 * 4 * 256 * 2 ))
      LC_DRAFT="Nanbeige4.1-3B.Q4_K_M.gguf|2443113024|0ba28f064c66071646e7959c752851abf02a096f1d37e7d0e443e3502e12ca02|https://huggingface.co/mradermacher/Nanbeige4.1-3B-GGUF/resolve/main"
      LC_DRAFT_TITLE="Nanbeige4.1-3B draft model" LC_DRAFT_GAIN="~70% faster decoding, ~3x faster tool calls (community-measured)"
      LC_REF_Q=Q3_K_M
      LC_SPEED_NOTE="Q3_K_M measured 38.7 tok/s on an empty context, ~15 at 58K (RTX A6000)."
      ;;
    ornith9b)
      # Ornith 1.5 9B (ornith-ai, MIT): dense 9B on the Qwen3.5 hybrid layout (GGUF arch qwen35). Read from the GGUF
      # (2026-11-18): 32 layers, full attention every 4th (8 layers), 4 KV heads x (256 K + 256 V) -> 32 KB/token
      # at f16; the other 24 layers keep a small fixed recurrent state instead of a cache. So 256K costs 8 GB:
      # Q8_0 + f16 cache at 256K is ~21 GB, comfortable on a 24 GB card. Vendor benchmarks only (not tested by AA):
      # SWE-bench Verified 70.6, Terminal-Bench 2.1 46.2, BrowseComp 56.4. Same XML tool calls as Qwen3.5.
      # Ollama's ornith-1.5:9b ships no chat template or tool parser, hence llama.cpp with the GGUF's own template.
      # Sampling: the card's "precise coding" row (temp 0.6, top_p 0.95, top_k 20, no presence penalty).
      # llama.cpp turns --cache-reuse off for hybrid models by itself, and speculation uses checkpoints there.
      LC_TITLE="Ornith 1.5 9B" LC_ALIAS=ornith-9b
      LC_BLURB="Self-improving 9B: light, fast, 256K on 24 GB cards"
      LC_REPO=https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/main
      LC_FILE_PREFIX=Ornith-1.5-9B
      LC_QUANTS=(   # official GGUFs (ornith-ai), best first: "quant|bytes|sha256"
        "Q8_0|9786060384|22086870b009dbe9815ee752c48a82de930118a7c5ce5599590892ae03b8b010"
        "Q6_K|7558901696|b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a"
        "Q5_K_M|6642544576|e4d9634a3b6546a5c00a8680568fe1125f6c98c704ee51ae52ba07650fb4247d"
        "Q4_K_M|5780090816|70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6"
      )
      LC_KV_F16=$(( 8 * 4 * 512 * 2 ))             # 32768 bytes/token
      LC_TEMP=0.6 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      # MTP layer inside the file (blk.32, one more full-attention layer -> 4 KB/token).
      LC_DRAFT_KIND=mtp-inline LC_DRAFT_KV_F16=$(( 4 * 512 * 2 ))
      LC_DRAFT_TITLE="Ornith's built-in MTP layer" LC_DRAFT_GAIN="faster decoding, no extra download (speed-up not yet measured)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Not measured yet on Summon; a 9B model runs several times faster than Nanbeige 33B."
      ;;
    qwen38pi)
      # Qwen3.8 27B + harness-hub/Qwen3.8-27B-pi-adapted (LoRA, Apache-2.0): RL-tuned (AsyncGRPO, LoRA r=32) to be
      # the model behind the pi coding agent on software-engineering and terminal tasks, with thinking OFF. Vendor
      # numbers (pi harness, thinking off): SWE-bench Pro hard-51 39.2% -> 64.7%, Terminal-Bench 2.1 41.0% -> 56.6%,
      # ~4.5x fewer generated tokens than the base with thinking. Not independently tested, and its authors haven't
      # re-evaluated the GGUF adapter. Base GGUF from ggml-org (text only: the vision projector isn't loaded).
      # Qwen3.8 hybrid layout: 16 of 64 layers keep a cache, 4 KV heads x (256 K + 256 V) -> 64 KB/token at f16.
      # Card settings: thinking off, temp 0.8, top_p 1.0. In pi use "api": "openai-completions" (the adapter wasn't
      # trained with the Responses API).
      LC_TITLE="Qwen3.8 27B pi" LC_ALIAS=qwen3.8-pi
      LC_BLURB="Qwen3.8 + RL-tuned LoRA for the pi agent. No thinking"
      LC_REPO=https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main
      LC_FILE_PREFIX=Qwen3.8-27B
      LC_QUANTS=(   # ggml-org's GGUFs, best first: "quant|bytes|sha256"
        "Q8_0|28595763840|6bc9e7220d8400574fd62df25f04a961cb5cbb0d766cf0118c21c36c0059c35a"
        "Q4_K_M|19403684480|d15d4b6de55f662ac40e867702d31afe738d032fa06b829b2b1a21a49d4a48ee"
      )
      LC_LORA="Qwen3.8-27B-pi-lora-F16.gguf|318802304|a64f91844f19e67da500d8628bdf9383aa334e61b7ab687805802e7d272d7b67|https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF/resolve/main"
      LC_EXTRA_BYTES=318802304                      # the adapter sits in VRAM next to the weights
      LC_KV_F16=$(( 16 * 4 * 512 * 2 ))             # 65536 bytes/token
      LC_TEMP=0.8 LC_TOP_P=1.0 LC_TOP_K="" LC_KWARGS='{"enable_thinking": false}'
      # Draft: Qwen3.8's own MTP head, shipped by ggml-org as a separate file (one full-attention layer -> 4 KB/token).
      # Measured on base Qwen3.8 27B: +60% decode on an RTX 3090 (preregistered study, n-max 2), 1.9x on code. With
      # the pi LoRA on the target, the head's acceptance rate is unmeasured.
      LC_DRAFT_KIND=mtp LC_DRAFT_KV_F16=$(( 4 * 512 * 2 ))
      LC_DRAFT="mtp-Qwen3.8-27B-Q8_0.gguf|3164006976|572ebcd0349321a71ccc6886c2611ff9d8c9aa160312ed4c90ae5bf9a055f4c6|https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main"
      LC_DRAFT_TITLE="Qwen3.8 MTP head" LC_DRAFT_GAIN="~60-90% faster decoding on base Qwen3.8 (measured)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Thinking is off (as trained): ~4.5x fewer tokens per task than Qwen3.8 with thinking (vendor figure)."
      ;;
    ornith35b)
      # Ornith 1.5 35B-A3B (ornith-ai, MIT): MoE, 256 experts / 8 active (~3B active), Qwen3.5 hybrid layout (GGUF
      # arch qwen35moe). Read from the GGUF (2026-11-18): 40 layers + 1 MTP layer, full attention every 4th, 2 KV
      # heads x (256 K + 256 V) -> 20 KB/token at f16 for 10 layers; sized for 11 to cover the MTP layer.
      # Q8_0 + f16 cache at 256K is ~45 GB: fits a 48 GB card. A 24 GB card can't hold even Q4_K_M with a useful
      # context. Vendor benchmarks only (not tested by AA): Terminal-Bench 2.1 68.5, SWE Atlas QnA 39.8.
      # Ollama's ornith-1.5:35b ships no chat template or tool parser, hence llama.cpp.
      LC_TITLE="Ornith 1.5 35B" LC_ALIAS=ornith-35b
      LC_BLURB="Self-improving MoE for coding agents. 3B active, fast"
      LC_REPO=https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main
      LC_FILE_PREFIX=Ornith-1.5-35B
      LC_QUANTS=(   # official GGUFs (ornith-ai), best first: "quant|bytes|sha256"
        "Q8_0|37802149280|de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3"
        "Q6_K|29208731392|15d4658bbfc9c6034621729c15bbb50662c82b32a7ddd9624a1e545a74bdbb4b"
        "Q5_K_M|25347532544|91df97de5845100e850b4b5ec5ff35695382020b880fad6f7f51787b3a953bd0"
        "Q4_K_M|21713463040|42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
      )
      LC_KV_F16=$(( 11 * 2 * 512 * 2 ))            # 22528 bytes/token (includes the MTP layer)
      LC_TEMP=0.6 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      # MTP layer inside the file (blk.40). Routed experts: 92% of the weights (19.1 of 20.7 GB at Q4_K_M, read
      # from the GGUF tensor table), so on a 24 GB card they live in RAM with the hot ones cached on the GPU.
      LC_DRAFT_KIND=mtp-inline LC_DRAFT_KV_F16=0 LC_EXPERT_PCT=92
      LC_DRAFT_TITLE="Ornith's built-in MTP layer" LC_DRAFT_GAIN="faster decoding, no extra download (speed-up not yet measured)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Not measured yet on Summon; ~3B active parameters per token, so it decodes far faster than a dense 33B."
      ;;
    thinkingcap)
      # ThinkingCap Qwen3.8 27B (bottlecapai): Qwen3.8 27B fine-tuned to think less: -37.2% mean thinking tokens for
      # -0.86 pts accuracy over 12 benchmarks (vendor). Licence: PolyForm Small Business 1.0.0 + a personal-use grant
      # (commercial use above the small-business limits needs a BottleCap licence). The GGUFs are open (not gated).
      # Dense model: no MoE experts. Its GGUFs carry Qwen3.8's MTP layer (blk.64, read from the Q4_K_M tensor table):
      # the makers serve it with --spec-type draft-mtp --spec-draft-n-max 3 and measured ~1.30x decode on Q4_K_M.
      # Qwen3.8 hybrid layout: 16 of 64 layers keep a cache, 4 KV heads x 512 -> 64 KB/token + 4 KB for the MTP layer.
      # Sampling: the base model's thinking-mode settings, as the card says (greedy decoding can loop). Vision needs
      # the separate mmproj file, which Summon doesn't load (text only).
      LC_TITLE="ThinkingCap Qwen3.8 27B" LC_ALIAS=thinkingcap-27b
      LC_BLURB="Qwen3.8 that thinks 37% less, MTP on. PolyForm SB"
      LC_REPO=https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF/resolve/main
      LC_FILE_PREFIX=ThinkingCap-Qwen3.8-27B
      LC_QUANTS=(   # bottlecapai's GGUFs, best first: "quant|bytes|sha256"
        "Q8_0|29047085152|41070725606f4be781db804e8458f3346c699d0dac24f2b96d2a734556c6c0f7"
        "Q6_K|23860566112|36ab5032dc4725f8acdb237070f9e5574171e0e29cd583fcc417a6c8ef87f557"
        "Q4_K_M|17442400352|fafa890ce2ce8531b4ade225c7dbd5f5d72a92303ca9ef72890c6cf78f19f299"
        "IQ4_XS|15475951712|c69ef3bbd56fe528bc9683186d14047243b00f41d3707016718775841d877cb2"
      )
      LC_KV_F16=$(( 16 * 4 * 512 * 2 ))             # 65536 bytes/token
      LC_TEMP=1.0 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      LC_DRAFT_KIND=mtp-inline LC_DRAFT_KV_F16=$(( 4 * 512 * 2 )) LC_DRAFT_NMAX=3
      LC_DRAFT_TITLE="ThinkingCap's built-in MTP layer" LC_DRAFT_GAIN="~1.3x faster decoding (measured by its makers on Q4_K_M)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Thinks ~37% less than Qwen3.8 (vendor, -0.86 pts), so answers finish sooner at the same tok/s."
      ;;
    bonsai27b)
      # Ternary Bonsai 2 27B (prism-ml, Apache-2.0): Qwen3.8 27B with ternary {-1,0,+1} weights (g128, FP16 scales) on
      # a Hadamard-rotated basis. Vendor: 98.2% of FP16's thinking-mode average over 14 benchmarks (84.78), BFCL v3
      # 74.92 vs 76.74. Not tested by AA. Its PQ2_0/PTQ1_0 files need PrismML's llama.cpp fork; the Q2_0 file (2.25
      # bpw) runs on stock llama.cpp from b11556 (the Hadamard runtime; older builds load it and output garbage), so
      # Summon uses Q2_0 only. F16 holds the same ternary values unpacked (7x the size, no quality gain): not offered.
      # Read from the GGUF (2026-11-21): qwen35, 64 layers, full attention every 4th (16 layers), 4 KV heads x
      # (256 K + 256 V) -> 64 KB/token at f16; no MTP layer (blocks 0-63), so n-gram drafts only. They do run on stock
      # llama.cpp (PrismML's notes say they don't in the fork): A100, rewriting a file 66 -> 421-461 tok/s, new text ~0.
      # Measured on an A100 80GB (b12429, 2026-11-21): 24.4 GB in use at 256K with an f16 cache. Its template has
      # tools, thinking on, and reasoning_effort xhigh (default) | medium | low ("high" is an error; low thinks about as
      # long as xhigh per the vendor; enable_thinking false turns thinking off). Vision needs
      # the separate mmproj file, which Summon doesn't load (text only).
      # Sampling: the card's thinking-mode row (temp 1.0, top_p 0.95, top_k 20; llama.cpp's min_p 0.05 default).
      LC_TITLE="Ternary Bonsai 2 27B" LC_ALIAS=bonsai-27b
      LC_BLURB="Qwen3.8 27B in ternary weights: 7.6 GB, 98% of FP16 (vendor)"
      LC_REPO=https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main
      LC_FILE_PREFIX=Ternary-Bonsai-2-27B
      LC_QUANTS=(   # prism-ml's stock-llama.cpp packing: "quant|bytes|sha256"
        "Q2_0|7626008928|4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e"
      )
      LC_KV_F16=$(( 16 * 4 * 512 * 2 ))             # 65536 bytes/token
      LC_TEMP=1.0 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      LC_REF_Q=Q2_0
      LC_SPEED_NOTE="Measured on an A100 80GB: 66 tok/s decode, ~1,300 tok/s prefill. Thinks at xhigh effort: allow 16K+ output."
      ;;
    qwen38obl)
      # Qwen3.8 27B OBLITERATED V3 (OBLITERATUS, Apache-2.0, 2026-08-19): abliterated Qwen3.8 27B, refusals and
      # "safety lecture" deflections removed. Vendor: MMLU 82.3% vs 84.5% stock (-2.1pp). Not tested by AA.
      # Its GGUFs ship a 506-byte template (no tools, empty-think prefill), so tool calls would fail: Summon serves
      # Qwen3.8's official template instead (same tokenizer) with enable_thinking false = the same no-think prefill.
      # Card settings for agents: temp 0.1-0.3 and repetition penalty 1.15 ("critical", greedy loops on tool calls),
      # no top_p/top_k, thinking off. Read from the GGUF (2026-11-20): qwen35, 64 layers + MTP layer (blk.64).
      LC_TITLE="Qwen3.8 27B OBLITERATED" LC_ALIAS=qwen3.8-obliterated
      LC_BLURB="Uncensored Qwen3.8 (abliterated V3), MMLU -2.1pp. No thinking"
      LC_REPO=https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/resolve/main
      LC_FILE_PREFIX=Qwen3.8-27B-OBLITERATED
      LC_QUANTS=(   # OBLITERATUS's GGUFs, best first: "quant|bytes|sha256"
        "Q8_0|29047075872|afa839b2fa5bc890e5735031dda2c6239d3b6bba3b6ffa29477cbc14a2e1f221"
        "Q6_K|22430991392|3535d4a15b75840fb391138ce04e6c73fde709320c6f5fe788cdbd586ee08e3a"
        "Q5_K_M|19535692832|87a69c85238854607fb58876347471f37db939d121ffc9c3dd18706186fd65f9"
        "Q4_K_M|16810705952|1f74330b211a8253c96f1bf586cba6eb56d37117c97ed9e6eec18c198a4e7fe5"
        "IQ4_XS|15420441632|f269deda48b0b703488bed23d7b3152f8003d7bfc14e6a327bd8757a5baac331"
      )
      LC_TEMPLATE="Qwen3.8-27B.chat_template.jinja|af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375|https://huggingface.co/Qwen/Qwen3.8-27B/resolve/37561816e66b5b9153b1b64c10968c18c1e6f4a0/chat_template.jinja"
      LC_KV_F16=$(( 16 * 4 * 512 * 2 ))             # 65536 bytes/token
      LC_TEMP=0.2 LC_TOP_P=1.0 LC_TOP_K="" LC_KWARGS='{"enable_thinking": false}'
      LC_ARGS=(--repeat-penalty 1.15)
      LC_DRAFT_KIND=mtp-inline LC_DRAFT_KV_F16=$(( 4 * 512 * 2 ))
      LC_DRAFT_TITLE="Qwen3.8's built-in MTP layer" LC_DRAFT_GAIN="faster decoding, no extra download (acceptance after abliteration unmeasured)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Thinking off (as its card recommends), so answers start right away. Not measured on Summon."
      ;;
    ornith9bobl)
      # Ornith 1.5 9B OBLITERATED (OBLITERATUS, MIT, 2026-08-27): 3-round SVD abliteration + per-head attention
      # surgery of Ornith 1.5 9B. Vendor: MMLU (n=100) 74.8% vs 78.8% stock (-4pp); Q8_0/Q6_K keep the abliteration
      # best. Same GGUF layout as Ornith 1.5 9B (read 2026-11-20: 32 layers, full attention every 4th, MTP blk.32);
      # its own template has tools. The card says to run with reasoning off (thinking loops), hence enable_thinking false.
      LC_TITLE="Ornith 1.5 9B OBLITERATED" LC_ALIAS=ornith-9b-obliterated
      LC_BLURB="Uncensored Ornith 9B (abliterated), MMLU -4pp. No thinking"
      LC_REPO=https://huggingface.co/OBLITERATUS/Ornith-1.5-9B-OBLITERATED/resolve/main
      LC_FILE_PREFIX=Ornith-1.5-9B-OBLITERATED
      LC_QUANTS=(   # OBLITERATUS's GGUFs, best first: "quant|bytes|sha256"
        "Q8_0|9786060064|e632573df9285acbceac28293ca19d95ffdbd108c05f39d54604d7f97ea4330c"
        "Q6_K|7558901024|1bbe46fdbf1686ffe49e204e9baa192b256dea4f80f6416ce045bce8126c4a81"
        "Q5_K_M|6642543904|81b3e622b99126cc5ce35f9c3635642bdf716bf0345726323840d3d38d6fad00"
        "Q4_K_M|5780090144|88ce99bb696fbc921edc4f995c4beb12a66bf94319d00bfd5f44265d02fa6dfb"
      )
      LC_KV_F16=$(( 8 * 4 * 512 * 2 ))             # 32768 bytes/token
      LC_TEMP=0.6 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS='{"enable_thinking": false}'
      LC_DRAFT_KIND=mtp-inline LC_DRAFT_KV_F16=$(( 4 * 512 * 2 ))
      LC_DRAFT_TITLE="Ornith's built-in MTP layer" LC_DRAFT_GAIN="faster decoding, no extra download (speed-up not yet measured)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Not measured yet on Summon; a 9B model runs several times faster than a 27B."
      ;;
    feyospace)
      # feyospace-s1 OBLITERATED (OBLITERATUS, Apache-2.0, 2026-10-14): cybersecurity-targeted abliteration of
      # feyospace-v1.1-small (Verapraxis; ~28B, Qwen3.5 layout) for pentesting, CTFs and vulnerability research.
      # Vendor: MMLU 82.4% (unchanged), WMDP-Cyber 63.8%, CyberMetric-10K 85.3%; base: CyberGym 84.4%. Its card warns
      # of sandbox escapes and goal drift in agent runs: isolate the tools. OBLITERATUS ships safetensors only, so the
      # GGUFs are mradermacher's (read 2026-11-20: qwen35, 64 layers + MTP blk.64, template with tools + thinking).
      # Base card: thinking on, keep reasoning_content across turns (preserve_thinking defaults to true). No sampling
      # given: Qwen3.8's thinking-mode settings, as for ThinkingCap.
      LC_TITLE="feyospace-s1 OBLITERATED" LC_ALIAS=feyospace-s1-obliterated
      LC_BLURB="Uncensored cybersecurity 27B for pentests and CTFs"
      LC_REPO=https://huggingface.co/mradermacher/feyospace-s1-OBLITERATED-GGUF/resolve/main
      LC_FILE_PREFIX=feyospace-s1-OBLITERATED LC_FILE_SEP=.
      LC_QUANTS=(   # mradermacher's GGUFs, best first: "quant|bytes|sha256"
        "Q8_0|29047085184|f0b84738bcb67f6a3981842a5f776eaefeab42d1ecc05aa44b6a263ee39e073e"
        "Q6_K|22431000704|4134c16e42f7cf5caa50362fe5ad91d13354581dcd217796f25b1e58cf5d2604"
        "Q5_K_M|19535702144|6875dd3b152f93fae6aaaa4b990aab15af3ba3dbc37998f1cc11dbbddc02009e"
        "Q4_K_M|16810715264|9facf38c3e7f4ac140ba48847a4b1657198721a917f6f727ccde8d1825701ae4"
        "IQ4_XS|15420450944|da9b0509d32c94f5a5cbe951d8a146475925ed7af423cc2433baf3d0761c7f5e"
      )
      LC_KV_F16=$(( 16 * 4 * 512 * 2 ))             # 65536 bytes/token
      LC_TEMP=1.0 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      LC_DRAFT_KIND=mtp-inline LC_DRAFT_KV_F16=$(( 4 * 512 * 2 ))
      LC_DRAFT_TITLE="feyospace's built-in MTP layer" LC_DRAFT_GAIN="faster decoding, no extra download (speed-up not yet measured)"
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Thinking on: long reasoning before answers. Not measured on Summon."
      ;;
    deepseek41obl)
      # DeepSeek V4.1 Flash 1017 OBLITERATED (OBLITERATUS, MIT, 2026-10-23): abliterated DeepSeek V4.1 Flash, a 748B MoE
      # (384 experts, 6 active). OBLITERATUS ships 1.5 TB of safetensors only; this is HumanjavaEnterprises' upstream
      # GGUF (IQ2_XXS mix, 2.66 bpw, own imatrix; repo rev f80c486): 237,451 MiB in 26 files. Their evals: MMLU 83.6%
      # (base quant 84.4%), GSM8K 97.2%; refusals 7/20 vs 19/20 for the base (reduced, not removed).
      # Tensor table, summed over all 26 files (2026-11-20): routed experts 125,955 MiB (53%), Engram tables + per-layer
      # embeddings 105,678 MiB (44.5%, llama.cpp reads them lazily from disk, never VRAM), the rest 5,818 MiB (GPU).
      # Their recipe (one 96 GB card + ~130 GB RAM: 63-74 tok/s decode): experts in RAM, --moe-cache-mib ~80000 (63%),
      # -lm none (126 GiB resident), -ub 2048, thinking off. Needs llama.cpp >= 2026-11-03: b12429 has deepseek41,
      # draft-dspark and the DSML tool parser. The GGUF's own template (702 bytes) has no tools, so Summon serves the
      # official chat_template.jinja (pinned), which llama.cpp detects as DeepSeek V4.1 (DSML tool calls).
      # K/V per token: no simple formula (compressed + shared attention); 40 layers x 1 KV head x (512 K + 512 V) at f16
      # plus the indexer keys is an upper bound, so the fit is conservative.
      # Draft: unsloth's DSpark drafter (MIT, 8 GB). Measured on this build: code +30-45% (80-102 tok/s), prose same.
      LC_TITLE="DeepSeek V4.1 Flash OBLITERATED" LC_ALIAS=deepseek-v4.1-flash-obliterated
      LC_CTX_TRAIN=1048576          # GGUF: 1M via the YaRN scaling baked into the file (x16 over 64K), read 2026-11-21
      LC_BLURB="Uncensored 748B MoE, MMLU 83.6%. Experts in RAM (~126 GB)"
      LC_REPO=https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF/resolve/f80c4863ac39a6a7002b3a3675a7c46580f0038b
      LC_FILE_PREFIX=DeepSeek-V4.1-Flash-1017-obliterated-hjimatrix
      LC_QUANTS=("IQ2_XXS|248991528992|split")   # total of the parts below
      LC_PARTS=(   # HF file list: "part|bytes|sha256"
        "00001-of-00026|4960285408|4f030340ddbd5eb689d2d8015065834383a14c95c86a80220b1566dcfd41d986"
        "00002-of-00026|4954522080|b621942d658eda514d360cf3a03df0bcd7c39e68c3f82060fa4a4ade799f375c"
        "00003-of-00026|4954522080|3b5af260b998f562a7bd13fde6b901e63b8dfdbbdfdd15ebb8deae57d614315d"
        "00004-of-00026|4954522080|061c1f2a67b5c5682680dd8151cf1c663b6c2472272923d9a34d3235775a2732"
        "00005-of-00026|5131469280|36105ea495c97351cbc8dec1bfc229581eb2bb181a598d70f6a8913c6006c17b"
        "00006-of-00026|5131469280|687e532f523869eb3dbedb37c899e2420a1536324b96ee7b2c9b78de27c2daf7"
        "00007-of-00026|5662310880|3dc062fee6602dbc99b7ed02c90d7a1a2bbd58f3fcbf431012122da38a2aef22"
        "00008-of-00026|5485363680|c60a2b04766dc5c85317b484645a1b2993df5647d7f283e9ad24170df39908f5"
        "00009-of-00026|4954522080|c91b4aa790ce357cd76d44bd803807ff396293300f851b3c7346d16b5c19ac2d"
        "00010-of-00026|5308416480|abca5fc169766ee966626817902aff2065ef4da6c4fef9455ce3e47125b7734a"
        "00011-of-00026|5768479200|12d4819746441720b6b82116d77ea5ce9439c14eaced2eb816b35d5af67ff752"
        "00012-of-00026|5945426400|1c845eba7ceed451dbe08dfe4f456a7c9fd01980a3d19cc71de446de5b043043"
        "00013-of-00026|4954522080|510aa8b3d8f9a594e87a6390f5dbfde8d39fac7d9324263bd45147dd4d341a49"
        "00014-of-00026|5662310880|26c0e6b7b7b83f121a95f4ed6f6ba6e8d3f8b73a4a6f0f83b55df55345534948"
        "00015-of-00026|5803868640|25854cc0ef73a233ea975becd081d2445d0b997ee7ccb9e6e6c5bdd2456b9531"
        "00016-of-00026|5980815840|3758c1879a87525a311bf14e4d3e06e853de1b9ca0af61b703dc6631c9a3b624"
        "00017-of-00026|5945426400|334c998cde2674c2b2d55adaf6362cfa5319c1ba598addba1a5ee7d6a9f0ba66"
        "00018-of-00026|5662310880|d3a4121b716a5ec66c3094aef8abb1e6189aa4183543e17038317142ddd212a9"
        "00019-of-00026|5131469280|8e011c4d874d19279fbce29525e05590f3ede385438ed727a64205c7e727df5e"
        "00020-of-00026|5662310880|2aa0c929a29ac80218c889e6fde0a0d80f11aa65222bf9f1a42540b1289ead48"
        "00021-of-00026|6086984160|aaa4e67ea7154aae6901a4040ddcdb279e0ac54e55ba8e3f3e17f09ae9a16447"
        "00022-of-00026|5768479200|f3d1c21bb0e9ef1d9fd84d70da42559664f6175a2cd66beba8d98b53b9557bb3"
        "00023-of-00026|5803868640|d28409d20a52d9cc4d89d2ad8a7805417ece014bfaa9902c1b6e10859604961d"
        "00024-of-00026|6405489120|17b1c028969b00d2ddb98fddd63f79c40d42619a68e8aaffc403095c0b93d1b6"
        "00025-of-00026|110595290592|58acb958598ee8d1d35623750e720b4822e206a9352448c3114c1129813495e7"
        "00026-of-00026|6317073472|7406e6a72c7e2e44a0c652d0106dcf6bfae0d3810c0151d0400d6d2666d320d2"
      )
      LC_EXPERT_PCT=53 LC_HOST_PCT=44
      LC_TEMPLATE="DeepSeek-V4.1-Flash-1017.chat_template.jinja|95b7033a693b321ceaefb6f89ad07d1e0cf2b095305711f934a4ac649456739d|https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/08c8657f02291e11b6f6b63e0e16014f8e748518/chat_template.jinja"
      LC_KV_F16=$(( 40 * 1024 * 2 + 8 * 128 * 2 ))   # 83968 bytes/token (upper bound)
      LC_TEMP=1.0 LC_TOP_P=1.0 LC_TOP_K="" LC_KWARGS='{"enable_thinking": false}'
      LC_ARGS=(-lm none -ub 2048)
      LC_DRAFT_KIND=dspark LC_DRAFT_KV_F16=$(( 3 * 1024 * 2 ))   # drafter cache: 3 blocks (estimate)
      LC_DRAFT="dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf|7976914240|6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a|https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main"
      LC_DRAFT_TITLE="DeepSeek's DSpark drafter" LC_DRAFT_GAIN="code +30-45%, prose about the same (measured on this build)"
      LC_REF_Q=IQ2_XXS
      LC_SPEED_NOTE="Measured by its makers on one 96 GB card: 63-74 tok/s decode, ~500-580 tok/s prefill."
      ;;
    *) echo "Unknown llama.cpp model '$1' (use: ${LC_MODEL_IDS[*]})"; exit 1 ;;
  esac
  LC_ID=$1
  # An active draft costs VRAM: its weights (if a separate file), its own cache per token, ~0.5 GB of buffers.
  LC_DRAFT_ON=0
  if [ -n "$LC_DRAFT_KIND" ] && [ "${DRAFT:-on}" != off ] && [ "${SPEC:-}" != off ]; then
    LC_DRAFT_ON=1
    LC_KV_F16=$(( LC_KV_F16 + LC_DRAFT_KV_F16 ))
    LC_EXTRA_BYTES=$(( LC_EXTRA_BYTES + 536870912 ))
    [ -n "$LC_DRAFT" ] && LC_EXTRA_BYTES=$(( LC_EXTRA_BYTES + $(echo "$LC_DRAFT" | cut -d'|' -f2) ))
  fi
  LC_KV_Q8=$(( LC_KV_F16 * 17 / 32 ))           # q8_0 = 1 byte + a 2-byte scale per 32 values (vs 2 bytes at f16)
  local e; LC_REF_BYTES=""
  for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$LC_REF_Q" ] && LC_REF_BYTES=$(echo "$e" | cut -d'|' -f2); done
  [ -n "$LC_REF_BYTES" ] || LC_REF_BYTES=$(echo "${LC_QUANTS[0]}" | cut -d'|' -f2)   # REF_Q stopped matching: fall back to the first quant
  return 0
}

# ======================================================================================================
# Ollama models
# ======================================================================================================
# Curated <=80B open models, ranked for long-context coherence. Data: Artificial Analysis (snapshot
# 2026-11-16): LCR = AA Long Context Reasoning, II = Intelligence Index v5, hall = hallucination rate,
# tau2 = tool-calling benchmark. All have tool calling (T) in Ollama; V = vision input, R = reasoning.
# KV KB/token (f16) and fixed MB (sliding-window cache) come from each GGUF's attention layout.
# The hf.co entries are faster-thinking Qwen3.8-27B derivatives (ThinkingCap: -37% thinking tokens, -0.86pp;
# Swift 1.5: -58.5% median thinking, +0.35%). Both carry an MTP head and restrictive licences (PolyForm Small
# Business; Swift Open License, free <= US$1M revenue). HF refs start OFF.
# robit/qwen3.8-27b-e03-pi-adapted (2026-11-19): Qwen3.8-27B Obliterated E03 with harness-hub's pi LoRA merged into
# the BF16 text weights, then Q8_0 (the only build: latest/27b/q8_0 are one file, 27.3 GB + 0.9 GB vision projector).
# GGUF: qwen35, 64 layers, full attention every 4th, 4 KV heads x (256 K + 256 V) -> 64 KB/token at f16. Ollama
# ships the qwen3.8 renderer + qwen3.5 parser (tools, thinking, vision); its thinking default is "medium", but the
# adapter was trained with thinking off, so clients should send think:false. Too big for 24 GB cards.
# AA data behind the ranking (2026-11-16): Qwen3.8 27B LCR .82 II20 hall 29% (lowest) | Muse Glimmer LCR .83 (top)
# II6 hall 84% | Qwen3.6 35B-A3B LCR .72 tau2 .95 (best tool calling) | Gemma 4 31B LCR .70 IFBench .76 tau2 .60 |
# K2 Horizon 36B-A4B LCR .72, 512K | K2 Horizon 7B LCR .70.
# OBLITERATUS (abliterated, uncensored) rows (2026-11-20), GGUFs by mradermacher: Llama 3.1 8B Lexi V2 (llama, 32 layers,
# 8 KV heads x 128 -> 128 KB/token at f16, 128K) and Qwen3 4B structured-output stage A (qwen3, 36 layers, 8 KV heads
# x 128 -> 144 KB/token). Not added: Mistral 7B v0.3 (a base model, no chat template), GLM-4 9B chat (its -hf
# conversion lost the 128K RoPE scaling: GGUF trained context 4096), DeepSeek V4.1 Flash (smallest GGUF ~250 GB).
# Descriptions: each model's own summary (ollama.com / Hugging Face page), shortened for the menu (2026-11-19).
# "rank|Q8 ref|Q4 ref|KV KB/token|fixed MB|caps|name|description"   (rank "-" = community / HF model, opt-in)
OL_MODELS=(
  "1|qwen3.8:27b-q8_0|qwen3.8:27b-q4_K_M|64|0|TVR|Qwen3.8 27B|Coding, research and long-horizon agentic tasks. Vision"
  "2|muse-glimmer:30b-q8_0|muse-glimmer:30b-q4_K_M|13|80|TVR|Muse Glimmer 30B|Meta's open model for always-on local agents and tool use"
  "3|qwen3.6:35b-a3b-q8_0|qwen3.6:35b-a3b-q4_K_M|20|0|TVR|Qwen3.6 35B-A3B|Agentic coding with thinking preservation. Fast MoE"
  "4|gemma4:31b-it-q8_0|gemma4:31b-it-qat|80|1600|TVR|Gemma 4 31B|Google: reasoning, agentic workflows, coding, multimodal"
  "5|ifm/k2-horizon:36b-q8_0|ifm/k2-horizon:36b-q4_K_M|192|0|T-R|K2 Horizon MoVA 36B-A4B|IFM open agentic MoE: reasons, calls tools, 512K context"
  "6|ifm/k2-horizon:7b-q8_0|ifm/k2-horizon:7b-q4_K_M|144|0|T-R|K2 Horizon 7B|IFM open agentic 7B with 512K context. Budget pick"
  "-|robit/qwen3.8-27b-obliterated-e03:27b||64|0|TVR|Qwen3.8 27B obliterated (Q4)|Unshackled (uncensored) Qwen3.8 27B via Obliteratus"
  "-|robit/qwen3.8-27b-e03-pi-adapted:q8_0||64|0|TVR|Qwen3.8 27B E03 pi (Q8)|Obliterated E03 + pi LoRA for the pi agent. Think off"
  "-|hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q8_0|hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M|64|0|TVR|ThinkingCap Qwen3.8 27B|Qwen3.8 that thinks 37% less, -0.86 pts. PolyForm SB"
  "-|hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q8_0|hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q4_K_M|64|0|TVR|Swift 1.5 Qwen3.8 27B|Qwen3.8 with 25-54% less thinking. Swift licence <=\$1M"
  "-|hf.co/mradermacher/Llama-3.1-8B-Lexi-Uncensored-V2-OBLITERATED-GGUF:Q8_0|hf.co/mradermacher/Llama-3.1-8B-Lexi-Uncensored-V2-OBLITERATED-GGUF:Q4_K_M|128|0|T--|Llama3.1 8B Lexi OBLITERATED|Uncensored Llama 3.1 8B (Lexi V2, abliterated). Older, small"
  "-|hf.co/mradermacher/qwen3-4b-structured-output-merged-stage-a-OBLITERATED-GGUF:Q8_0|hf.co/mradermacher/qwen3-4b-structured-output-merged-stage-a-OBLITERATED-GGUF:Q4_K_M|144|0|T-R|Qwen3 4B JSON OBLITERATED|Uncensored Qwen3 4B tuned for JSON output (stage-A checkpoint)"
  "7|gpt-oss:20b|gpt-oss:20b|24|4|T-R|gpt-oss 20B|OpenAI open MoE (3.1B active): fast reasoning, strong tool use. Best CPU pick"
  "8|qwen3:30b|qwen3:30b|96|0|T-R|Qwen3 30B-A3B|Qwen3 MoE (3B active), 256K window. Near-frontier quality, runs well on CPU"
  "9|phi4:14b|phi4:14b|200|0|T--|Phi-4 14B|Microsoft: reasoning quality far above its size. Full attention, heavy KV"
  "10|qwen2.5:14b-instruct|qwen2.5:14b-instruct|192|0|T--|Qwen2.5 14B|Older but proven all-rounder with tool calling"
)
# Hugging Face GGUF refs have no Ollama registry manifest: exact sizes in MiB from the HF file list (2026-11-17).
declare -A HF_MB=(
  ["hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q8_0"]=27701
  ["hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M"]=16634
  ["hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q8_0"]=27701
  ["hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q4_K_M"]=16634
  ["hf.co/mradermacher/Llama-3.1-8B-Lexi-Uncensored-V2-OBLITERATED-GGUF:Q8_0"]=8145
  ["hf.co/mradermacher/Llama-3.1-8B-Lexi-Uncensored-V2-OBLITERATED-GGUF:Q4_K_M"]=4692
  ["hf.co/mradermacher/qwen3-4b-structured-output-merged-stage-a-OBLITERATED-GGUF:Q8_0"]=4082
  ["hf.co/mradermacher/qwen3-4b-structured-output-merged-stage-a-OBLITERATED-GGUF:Q4_K_M"]=2381
)
OL_CTX_MAX=131072 OL_CTX_MIN=65536
OL_OVERHEAD_MB=1536                            # CUDA context + compute buffers

# ======================================================================================================
# Shared helpers
# ======================================================================================================
# --- EXIT trap: components append their cleanup to SUMMON_EXIT_CMDS (a later trap must not clobber an earlier one) ---
SUMMON_EXIT_CMDS=""
run_exit_cmds() { [ -n "$SUMMON_EXIT_CMDS" ] && eval "$SUMMON_EXIT_CMDS"; return 0; }
trap run_exit_cmds EXIT
APT_UPDATED=0
apt_get() {   # install packages; runs apt-get update once per run first (fresh pods have no package lists)
  if (( ! APT_UPDATED )); then
    apt-get update -qq; APT_UPDATED=1
    # Minimal images lack apt-utils, so every install prints "debconf: delaying package configuration, since
    # apt-utils is not installed". Harmless, but noisy: install it first (~1 MB), quietly, best effort.
    dpkg -s apt-utils >/dev/null 2>&1 || apt-get install -y -qq apt-utils >/dev/null 2>&1 || true
  fi
  apt-get install -y -qq "$@" >/dev/null
}

# --- Branding: banner + whiptail theme (steel blue; your own NEWT_COLORS wins) ---
TAGLINE="Your GPU. Your model. Your API."
BACKTITLE="SUMMON v$SUMMON_VERSION  -  $TAGLINE"
export NEWT_COLORS=${NEWT_COLORS:-'root=brightblue,black roottext=brightblue,black window=white,black border=blue,black title=brightblue,black textbox=white,black listbox=white,black actlistbox=black,blue sellistbox=brightblue,black actsellistbox=black,brightblue checkbox=brightblue,black actcheckbox=black,blue button=black,blue actbutton=white,brightblue compactbutton=white,black entry=white,black disentry=gray,black emptyscale=,black fullscale=,blue helpline=white,black'}
banner() {   # wordmark, tagline, one plain line on what it does
  local c1="" c2="" c3="" r="" utf=0
  if [ -t 1 ]; then
    if [[ ${TERM:-} == *256color* ]]; then c1=$'\e[38;5;75m' c2=$'\e[38;5;111m' c3=$'\e[38;5;153m'
    else c1=$'\e[34m' c2=$'\e[94m' c3=$'\e[96m'; fi
    r=$'\e[0m'
  fi
  [[ "$(locale charmap 2>/dev/null)" == UTF-8 ]] && utf=1
  echo
  if (( utf )); then
    printf '%s\n' \
      "${c1}   ███████╗██╗   ██╗███╗   ███╗███╗   ███╗ ██████╗ ███╗   ██╗${r}" \
      "${c1}   ██╔════╝██║   ██║████╗ ████║████╗ ████║██╔═══██╗████╗  ██║${r}" \
      "${c2}   ███████╗██║   ██║██╔████╔██║██╔████╔██║██║   ██║██╔██╗ ██║${r}" \
      "${c2}   ╚════██║██║   ██║██║╚██╔╝██║██║╚██╔╝██║██║   ██║██║╚██╗██║${r}" \
      "${c2}   ███████║╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║${r}" \
      "${c1}   ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝${r}"
  else
    printf '%s\n' "${c2}   SUMMON${r}"
  fi
  echo "${c3}   $TAGLINE  v$SUMMON_VERSION${r}"
  echo "   A private, OpenAI-compatible LLM endpoint on your own GPU. No cloud, no per-token bill."
  echo
}

choose() {   # $1 = title, $2 = text, $3 = default tag, then tag/label pairs -> prints the chosen tag
  # CHOOSE_NOTAGS=1 hides the tag column (rows read as plain labels); CHOOSE_W caps the width (default 110).
  local title=$1 text=$2 def=$3 i=0 n=1 c tags=(); shift 3
  if command -v whiptail >/dev/null; then
    local w=$(( $(tput cols 2>/dev/null || echo 110) - 4 )) h lh rows=$(( $# / 2 ))
    (( w > ${CHOOSE_W:-110} )) && w=${CHOOSE_W:-110}
    local chrome=$(( 10 + ${CHOOSE_TEXTLINES:-2} ))   # borders, buttons, blank lines + the prompt text
    h=$(( rows + chrome )); (( h > $(tput lines 2>/dev/null || echo 40) - 2 )) && h=$(( $(tput lines 2>/dev/null || echo 40) - 2 ))
    lh=$(( h - chrome )); (( lh < 3 )) && lh=3; (( lh > rows )) && lh=$rows
    whiptail --backtitle "$BACKTITLE" --title "$title" ${CHOOSE_NOTAGS:+--notags} --default-item "$def" \
      --menu "$text" "$h" "$w" "$lh" "$@" 3>&1 1>&2 2>&3
    return
  fi
  printf '\n%s\n%b\n' "$title" "$text" >&2
  while (( $# )); do
    i=$((i + 1)); tags+=("$1"); [ "$1" = "$def" ] && n=$i
    if [[ $1 =~ ^[0-9]+$ ]] || [ -n "${CHOOSE_NOTAGS:-}" ]; then printf '%2d) %s\n' "$i" "$2" >&2
    else printf '%2d) %-8s %s\n' "$i" "$1" "$2" >&2; fi
    shift 2
  done
  read -rp "Choice [$n]: " c || true; c=${c:-$n}   # EOF on piped stdin: the default, not a silent death under set -e
  [[ "$c" =~ ^[0-9]+$ ]] && (( c >= 1 && c <= i )) || { echo "Invalid choice: '$c'" >&2; return 1; }
  echo "${tags[c-1]}"
}

# --- Port lock: only your IP may reach the server (firewall rule, no VPN) ---
# Default: the IP of your current SSH session. Ollama listens on the host (INPUT chain); Docker-published ports
# skip INPUT, so the llama.cpp rule lives in DOCKER-USER and matches the original (pre-NAT) port.
# iptables rules don't survive a reboot: rerun the script, or `--lock`, after one.
unlock() {   # $1 = INPUT|DOCKER-USER: remove every LIUM-LOCK jump there, and its chain
  command -v iptables >/dev/null || return 0
  local r
  iptables -S "$1" 2>/dev/null | grep -E -- '-j LIUM-LOCK-[0-9]+$' | sed 's/^-A /-D /' | while read -r r; do
    iptables $r; iptables -F "${r##* }" 2>/dev/null; iptables -X "${r##* }" 2>/dev/null || true
  done
}
lock_port() {   # $1 = external port, $2 = docker|host
  local port=$1 chain=LIUM-LOCK-$1 allow=${ALLOW_IP:-} ip parent=INPUT
  [ "$2" = docker ] && parent=DOCKER-USER
  if [ -z "$allow" ]; then
    allow=${SSH_CLIENT%% *}; [ -n "$allow" ] || allow=${SSH_CONNECTION%% *}
    if [ -z "$allow" ]; then read -rp "Public IP allowed to use port $port (blank = open to everyone): " allow || true; fi
    # A private source address means the provider NATs inbound traffic: every client would look the same.
    if [[ $allow =~ ^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.) ]]; then
      echo "WARNING: your SSH session comes from a private address ($allow), so this host can't see real client IPs."
      echo "Port lock left unchanged (none on a first install). Set ALLOW_IP=<your public IP> only if you know"
      echo "inbound traffic keeps its source IP here."
      return 0
    fi
  fi
  [ -n "$allow" ] && [ "$allow" != any ] || { unlock "$parent"; echo "Port $port is open to everyone (no ALLOW_IP)."; return 0; }
  for ip in ${allow//,/ }; do
    [[ $ip =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}(/[0-9]{1,2})?$ ]] || { echo "Invalid IP in ALLOW_IP: '$ip'"; exit 1; }
  done
  command -v iptables >/dev/null || apt_get iptables >/dev/null 2>&1 || true
  command -v iptables >/dev/null || { echo "iptables is missing; can't lock port $port."; exit 1; }
  unlock "$parent"
  iptables -N "$chain" 2>/dev/null || iptables -F "$chain"
  for ip in ${allow//,/ }; do iptables -A "$chain" -s "$ip" -j RETURN; done
  iptables -A "$chain" -s 127.0.0.0/8 -j RETURN
  iptables -A "$chain" -j DROP
  if [ "$2" = docker ]; then
    iptables -N DOCKER-USER 2>/dev/null || true
    iptables -C FORWARD -j DOCKER-USER 2>/dev/null || iptables -I FORWARD 1 -j DOCKER-USER
    iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctorigdstport "$port" --ctdir ORIGINAL -j "$chain"
  else
    iptables -I INPUT 1 -p tcp --dport "$port" -j "$chain"
  fi
  LOCKED_TO=$allow
  echo "Port $port locked to: ${allow//,/, }   (check hits: iptables -L $chain -v -n)"
}

# --- Access: who may reach the endpoint ---
# SSH session (rented pod): bind 0.0.0.0, firewall-lock to your SSH IP. No SSH session (your own PC): bind
# 127.0.0.1, so only this computer reaches it (WSL forwards Windows' localhost to it too). ALLOW_IP overrides.
access_mode() {
  if [ -n "${ALLOW_IP:-}" ]; then if [ "$ALLOW_IP" = any ]; then ACCESS=open; else ACCESS=locked; fi
  elif [ -n "${SSH_CLIENT:-}${SSH_CONNECTION:-}" ]; then ACCESS=locked
  else ACCESS=local; fi
  BIND=0.0.0.0; [ "$ACCESS" = local ] && BIND=127.0.0.1
  HOST_HINT="<this-host>"; [ "$ACCESS" = local ] && HOST_HINT=localhost
  return 0
}
access_text() {   # $1 = llamacpp|ollama
  if [ "${1:-}" = llamacpp ] && (( KEY_ON )); then
    if (( TUNNEL_ON )); then echo "API key + HTTPS through a Cloudflare tunnel (the port itself only listens on 127.0.0.1)"
    elif [ "$ACCESS" = local ]; then echo "API key, and only this computer (127.0.0.1). Share it: ALLOW_IP=any or TUNNEL=1"
    elif [ -n "${ALLOW_IP:-}" ] && [ "$ALLOW_IP" != any ]; then echo "API key + firewall: only ${ALLOW_IP}"
    else echo "API key required (port open to the internet, no IP lock; ALLOW_IP=<ip> adds one)"; fi
    return 0
  fi
  case "$ACCESS" in
    local)  echo "local: only this computer (127.0.0.1). Share it with ALLOW_IP=<ip> or ALLOW_IP=any" ;;
    locked) local ssh_ip=${SSH_CLIENT:-${SSH_CONNECTION:-}}   # both start with the client IP, then ports
            echo "locked to ${ALLOW_IP:-your SSH IP ${ssh_ip%% *}}" ;;
    open)   echo "OPEN to anyone who can reach this machine (ALLOW_IP=any)" ;;
  esac
}
apply_access() {   # $1 = docker|host
  # llama.cpp with an API key: the key is the lock. The firewall is only added on request (ALLOW_IP=<ips>).
  if [ "$1" = docker ] && (( KEY_ON )); then
    if [ "$ACCESS" = local ] || (( TUNNEL_ON )); then unlock DOCKER-USER
    elif [ -n "${ALLOW_IP:-}" ] && [ "$ALLOW_IP" != any ]; then lock_port "$PORT" docker
    else unlock DOCKER-USER; fi
    return 0
  fi
  if [ "$ACCESS" = local ]; then
    if [ "$1" = docker ]; then unlock DOCKER-USER; else unlock INPUT; fi
    echo "Local mode: only this computer can reach port $PORT (ALLOW_IP=<ip> to share it)."
  else
    lock_port "$PORT" "$1"
  fi
}

# --- API key, public address, HTTPS tunnel and client setup (llama.cpp gets a key; Ollama has no auth) ---
# The key is generated once and kept in the data folder, so clients keep working across reruns and model
# switches. API_KEY=<value> sets your own, API_KEY=none turns it off (back to IP locking), --new-key rotates it.
SUMMON_HOME=${HOME:-/root}/.summon
CONNECT_STATE=$SUMMON_HOME/connect.env
TUNNEL_NAME=summon-tunnel
TUNNEL_IMAGE=cloudflare/cloudflared:2026.11.1      # Cloudflare quick tunnel: free, no account, HTTPS
KEY_ON=1; [ "${API_KEY:-}" = none ] && KEY_ON=0
TUNNEL_ON=0; [ "${TUNNEL:-}" = 1 ] && TUNNEL_ON=1
if (( TUNNEL_ON && ! KEY_ON )); then
  echo "TUNNEL=1 with API_KEY=none would publish an unauthenticated public URL. Keep the key (drop API_KEY=none)."; exit 1
fi
ensure_api_key() {   # $1 = key file: keep it, or create it (API_KEY=... or random); never printed to logs
  if [ -n "${API_KEY:-}" ] && [ "$API_KEY" != none ]; then printf '%s\n' "$API_KEY" > "$1"
  elif (( NEW_KEY )) || [ ! -s "$1" ]; then
    printf 'sk-summon-%s\n' "$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-32)" > "$1"
  fi
  chmod 600 "$1"
}
public_host() {   # how clients reach this machine: PUBLIC_HOST, localhost, or the public IP (api.ipify.org)
  local ip
  if [ -n "${PUBLIC_HOST:-}" ]; then echo "$PUBLIC_HOST"; return 0; fi
  if [ "$ACCESS" = local ]; then echo localhost; return 0; fi
  ip=$(curl -4 -fsS --max-time 5 https://api.ipify.org 2>/dev/null || true)
  if [[ $ip =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then echo "$ip"; else echo "<this-host>"; fi
}
tunnel_url() { docker logs "$TUNNEL_NAME" 2>&1 | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1; }
start_tunnel() {   # $1 = local port -> TUNNEL_URL. A quick tunnel gets a new URL each time it starts.
  local i
  docker rm -f "$TUNNEL_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$TUNNEL_NAME" --restart unless-stopped --network host "$TUNNEL_IMAGE" \
    tunnel --no-autoupdate --url "http://127.0.0.1:$1" >/dev/null
  TUNNEL_URL=""
  for i in $(seq 60); do TUNNEL_URL=$(tunnel_url); [ -n "$TUNNEL_URL" ] && break; sleep 1; done
  return 0
}
save_connect() {   # remember how to reach the server, for --connect
  mkdir -p "$SUMMON_HOME"
  { printf 'C_ENGINE=%q\nC_URL=%q\nC_MODEL=%q\nC_TITLE=%q\nC_CTX=%q\nC_REASON=%q\nC_KEYFILE=%q\nC_TUNNEL=%q\nC_CHAT=%q\n' \
      "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; } > "$CONNECT_STATE"
  chmod 600 "$CONNECT_STATE"
}
print_connect() {   # reads $CONNECT_STATE: chat page, endpoint and copy-paste setup for common agents
  # shellcheck disable=SC1090
  . "$CONNECT_STATE"
  local url=$C_URL key=ollama https=0 t
  if [ "$C_TUNNEL" = 1 ]; then t=$(tunnel_url); [ -n "$t" ] && url=$t; fi
  [[ $url == https://* ]] && https=1
  [ -n "$C_KEYFILE" ] && [ -s "$C_KEYFILE" ] && key=$(head -1 "$C_KEYFILE")
  local base="$url/v1" out=$((C_CTX >= 65536 ? 32768 : C_CTX / 4))
  cat <<EOF

==== Connect ====  (show this again: $0 --connect)
$( [ "$C_CHAT" = 1 ] && echo "  Chat in your browser:  $url   (it asks for the API key once)" )
  Base URL:  $base
  Model:     $C_MODEL        Context window: $C_CTX tokens
  API key:   $key$( [ -n "$C_KEYFILE" ] && echo "   (kept in $C_KEYFILE; rotate: $0 --new-key ...)" )
$( [[ $url == *"<this-host>"* ]] && echo "  Replace <this-host> with this machine's public address (or rerun with PUBLIC_HOST=... / TUNNEL=1)." )
-- Hermes Agent: ~/.hermes/config.yaml (or run: hermes model -> Custom endpoint)
model:
  default: $C_MODEL
  provider: custom
  base_url: $base
  api_key: $key
  context_length: $C_CTX

-- pi: add this provider to ~/.pi/agent/models.json, then /model -> summon/$C_MODEL
{"providers": {"summon": {"baseUrl": "$base", "api": "openai-completions", "apiKey": "$key",
  "models": [{"id": "$C_MODEL", "name": "$C_TITLE", "reasoning": $C_REASON, "contextWindow": $C_CTX, "maxTokens": $out}]}}}

-- opencode: add to ~/.config/opencode/opencode.json (or a project's opencode.json), then /models
{"\$schema": "https://opencode.ai/config.json", "model": "summon/$C_MODEL",
 "provider": {"summon": {"npm": "@ai-sdk/openai-compatible", "name": "Summon", "options": {"baseURL": "$base", "apiKey": "$key"},
  "models": {"$C_MODEL": {"name": "$C_TITLE", "limit": {"context": $C_CTX, "output": $out}}}}}}

-- OpenClaw: merge into ~/.openclaw/openclaw.json
{"models": {"mode": "merge", "providers": {"summon": {"baseUrl": "$base", "apiKey": "$key", "api": "openai-completions",
  "models": [{"id": "$C_MODEL", "name": "$C_TITLE", "reasoning": $C_REASON, "input": ["text"], "contextWindow": $C_CTX, "maxTokens": $out}]}}},
 "agents": {"defaults": {"model": {"primary": "summon/$C_MODEL"}}}}

-- Cursor: Settings > Models > API Keys: OpenAI API Key = the key above, Override OpenAI Base URL = $base,
   then add a custom model named $C_MODEL. Cursor calls models from its own servers, so it needs a public HTTPS URL:
$( if (( https )); then echo "   this one is (TUNNEL=1). It changes when the tunnel restarts: rerun $0 --connect for the current one."
   elif [ "$C_ENGINE" = ollama ]; then echo "   not available for Ollama models here; use a llama.cpp model with TUNNEL=1."
   else echo "   rerun with TUNNEL=1 (e.g. TUNNEL=1 $0 PORT) to get one."; fi )
EOF
}

# --- Our servers ---
lc_container() {   # name of our running llama.cpp container (current or legacy name), or nothing
  command -v docker >/dev/null || return 0
  local n
  for n in "$LC_NAME" "${LC_OLD_NAMES[@]}"; do
    docker ps -q -f "name=^${n}\$" 2>/dev/null | grep -q . && { echo "$n"; return 0; }
  done
  return 0
}
lc_running() { [ -n "$(lc_container)" ]; }
lc_port() { docker port "$(lc_container)" 8080/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}'; }
ol_running() { pgrep -x ollama >/dev/null 2>&1; }
ol_port() { ss -ltnp 2>/dev/null | awk '/"ollama"/ {n=split($4,a,":"); print a[n]; exit}'; }
stop_llamacpp() {
  local n
  if command -v docker >/dev/null; then
    for n in "$LC_NAME" "${LC_OLD_NAMES[@]}" "$TUNNEL_NAME"; do docker rm -f "$n" >/dev/null 2>&1 || true; done
  fi
  unlock DOCKER-USER
}
stop_ollama() {
  systemctl disable --now summon-ollama >/dev/null 2>&1 || true
  screen -S ollama -X quit 2>/dev/null || true; unlock INPUT
}
has_systemd() { [ -d /run/systemd/system ] && command -v systemctl >/dev/null; }
OL_UNIT=/etc/systemd/system/summon-ollama.service
start_ollama_server() {   # systemd service if available (survives reboots), else a screen session
  if has_systemd; then
    local self lock=""
    self=$(readlink -f "$0")
    # A reboot clears iptables: re-apply the IP lock once the server is up (Ollama has no API keys).
    [ "$ACCESS" != local ] && [ -n "${LOCKED_TO:-}" ] && [ "$LOCKED_TO" != any ] && \
      lock="ExecStartPost=-/usr/bin/env ALLOW_IP=$LOCKED_TO /bin/bash $self --lock $PORT"
    cat > "$OL_UNIT" <<UNIT
[Unit]
Description=Summon: Ollama on port $PORT
After=network-online.target
Wants=network-online.target

[Service]
User=root
Environment=HOME=${HOME:-/root}
Environment=OLLAMA_HOST=$BIND:$PORT
Environment=OLLAMA_CONTEXT_LENGTH=$OL_CTX_MAX
Environment=OLLAMA_KV_CACHE_TYPE=$KV_TYPE
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}
${OLLAMA_MODELS:+Environment=OLLAMA_MODELS=$OLLAMA_MODELS}
ExecStart=$(command -v ollama) serve
$lock
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable --now summon-ollama >/dev/null
    OL_MODE="systemd service summon-ollama (starts on boot)"
  else
    OLLAMA_HOST=$BIND:$PORT OLLAMA_CONTEXT_LENGTH=$OL_CTX_MAX OLLAMA_KV_CACHE_TYPE="$KV_TYPE" \
      screen -dmS ollama ollama serve
    OL_MODE="screen session (no systemd here: rerun $0 after a reboot)"
  fi
}
wait_port_free() {   # after stopping our servers: give them 5 s to let go, then refuse a port someone else holds
  local i
  for i in 1 2 3 4 5; do curl -s --max-time 1 -o /dev/null "http://127.0.0.1:$PORT" || return 0; sleep 1; done
  # curl proves an HTTP holder; ss names any other kind of listener on the port
  local holder; holder=$(ss -ltn 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $4; exit}')
  [ -n "$holder" ] && echo "Port $PORT is already in use (listening: $holder). Pick another port: $0 <port>" \
    || echo "Port $PORT is already in use by another server. Pick another port: $0 <port>"
  exit 1
}

# ======================================================================================================
# Arguments
# ======================================================================================================
ACTION=install PORT="" TARGET="" YES=0 DRY_RUN=0 NEW_KEY=0 AUTO=0; [ "${GO:-}" = 1 ] && AUTO=1
while (( $# )); do
  case "$1" in
    --uninstall) ACTION=uninstall ;;
    --lock) ACTION=lock ;;
    --connect) ACTION=connect ;;
    status|--status) ACTION=status ;;
    --auto|--go) AUTO=1 ;;
    --new-key) NEW_KEY=1 ;;
    --verify) VERIFY_SHA256=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --version) echo "summon $SUMMON_VERSION"; exit 0 ;;
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed -e '/^set -e/d' -e 's/^# \{0,1\}//'; exit 0 ;;
    -y) YES=1 ;;
    nanbeige|qwen38pi|ornith35b|ornith9b|thinkingcap|bonsai27b|qwen38obl|ornith9bobl|feyospace|deepseek41obl|ollama|llamacpp|llama.cpp) TARGET=$1 ;;
    -*) echo "Unknown option: $1"; exit 1 ;;
    *) [ -z "$PORT" ] || { echo "Only one port argument is accepted: '$1'"; exit 1; }; PORT=$1 ;;
  esac
  shift
done
if [ -n "$PORT" ]; then
  # 10#: a leading zero ("08001") would be parsed as (invalid) octal by the arithmetic below
  [[ "$PORT" =~ ^[0-9]+$ ]] && PORT=$((10#$PORT)) && (( PORT >= 1 && PORT <= 65535 )) || { echo "Invalid port: '$PORT'"; exit 1; }
fi

# --- Re-lock only: no restart. Locking needs no running server (e.g. the Ollama service re-locks itself on boot): ---
# --- --lock PORT works standalone (host mode); a matching llama.cpp container switches the rule to DOCKER-USER. ---
if [ "$ACTION" = lock ]; then
  MODE=""
  if [ -n "$PORT" ]; then
    MODE=host
    if lc_running && [ "$(lc_port)" = "$PORT" ]; then MODE=docker; fi
  elif lc_running; then MODE=docker; PORT=$(lc_port)
  elif ol_running; then MODE=host; PORT=$(ol_port); fi
  [ -n "$MODE" ] || { echo "No running server found, and no port passed: $0 --lock PORT"; exit 1; }
  if (( DRY_RUN )); then
    echo "DRY RUN: would lock port $PORT to your IP ($MODE mode). Nothing done."
    exit 0
  fi
  lock_port "$PORT" "$MODE"
  exit 0
fi

# --- Connect: print the chat page, endpoint and client setup of the server we started last ---
if [ "$ACTION" = connect ]; then
  [ -s "$CONNECT_STATE" ] || { echo "No Summon server set up yet (nothing in $CONNECT_STATE). Run $0 first."; exit 1; }
  lc_running || ol_running || echo "Note: no Summon server is running right now; this is the last one set up."
  print_connect; exit 0
fi

# --- Status: what's running, where, how fast ---
if [ "$ACTION" = status ]; then
  echo "Summon v$SUMMON_VERSION status"
  if lc_running; then
    c=$(lc_container); p=$(lc_port)
    args=$(docker inspect -f '{{join .Args " "}}' "$c" 2>/dev/null)
    m=$(sed -n 's/.*--alias \([^ ]*\).*/\1/p' <<< "$args"); ctx=$(sed -n 's/.* -c \([0-9]*\).*/\1/p' <<< "$args")
    f=$(sed -n 's#.*-m /models/\([^ ]*\).*#\1#p' <<< "$args")
    since=$(docker inspect -f '{{.State.StartedAt}}' "$c" 2>/dev/null | cut -d. -f1 | tr T ' ')
    h=$(curl -fs --max-time 3 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"ok"' && echo ready || echo "loading or down")
    speed=$(docker logs --tail 400 "$c" 2>&1 | grep -E '^.* eval time' | grep -v prompt | tail -1 | grep -oE '[0-9.]+ tokens per second' || true)
    echo "  Engine:    llama.cpp (container $c), $h, up since $since UTC"
    echo "  Model:     $m  ($f)"
    echo "  Context:   ${ctx:-?} tokens        Port: $p ($(docker port "$c" 8080/tcp | head -1))"
    echo "  Last decode speed: ${speed:-no request yet}"
    if docker ps -q -f "name=^$TUNNEL_NAME\$" | grep -q .; then echo "  HTTPS:     $(tunnel_url)"; fi
  elif ol_running; then
    p=$(ol_port)
    if systemctl is-active summon-ollama >/dev/null 2>&1; then how="systemd service summon-ollama (starts on boot)"
    elif screen -ls 2>/dev/null | grep -q '\.ollama'; then how="screen session (not restarted after a reboot)"
    else how="not started by Summon"; fi
    echo "  Engine:    Ollama on port ${p:-?}, $how"
    OLLAMA_HOST=127.0.0.1:${p:-11434} ollama ps 2>/dev/null | sed 's/^/  /'
  else
    echo "  No Summon server is running. Start one: $0"
  fi
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
      awk -F', ' '{printf "  GPU:       %s, %d / %d MB VRAM, %d%% busy\n", $1, $2, $3, $4}'
  fi
  echo "  Connect:   $0 --connect      Logs: docker logs -f summon / journalctl -u summon-ollama -f"
  exit 0
fi

# --- Uninstall: one engine by name, or both (a llama.cpp model name means the whole llama.cpp engine) ---
if [ "$ACTION" = uninstall ]; then
  case "$TARGET" in nanbeige|qwen38pi|ornith35b|ornith9b|thinkingcap|bonsai27b|qwen38obl|ornith9bobl|feyospace|deepseek41obl|llama.cpp) TARGET=llamacpp ;; esac
  DIRS=()
  if [ "$TARGET" != ollama ]; then for d in "${LC_DIRS[@]}"; do [ -d "$d" ] && DIRS+=("$d"); done; fi
  if [ "$TARGET" != llamacpp ]; then
    for d in /root/.ollama /usr/share/ollama ${OLLAMA_MODELS:+"$OLLAMA_MODELS"}; do [ -d "$d" ] && DIRS+=("$d"); done
  fi
  USED=$(du -shc "${DIRS[@]}" 2>/dev/null | tail -1 | cut -f1)
  if (( DRY_RUN )); then
    echo "DRY RUN: would remove ${TARGET:-llama.cpp and Ollama} and delete ${DIRS[*]:-(no model folders found)} (~${USED:-0}). Nothing done."
    exit 0
  fi
  if (( ! YES )); then
    read -rp "Remove ${TARGET:-llama.cpp and Ollama} and DELETE the model files in ${DIRS[*]:-(none)} (~${USED:-0})? [y/N] " a || true
    [[ $a == [yY]* ]] || { echo "Cancelled."; exit 0; }
  fi
  if [ "$TARGET" != ollama ]; then
    stop_llamacpp
    if command -v docker >/dev/null; then docker rmi "$LC_IMAGE" >/dev/null 2>&1 || true; fi
    for d in "${LC_DIRS[@]}"; do rm -rf "$d"; done
    docker rmi "$TUNNEL_IMAGE" >/dev/null 2>&1 || true
    grep -q '^C_ENGINE=llamacpp' "$CONNECT_STATE" 2>/dev/null && rm -f "$CONNECT_STATE"
    echo "llama.cpp server and its model files removed. Docker itself is left installed."
  fi
  if [ "$TARGET" != llamacpp ]; then
    stop_ollama
    if [ -f "$OL_UNIT" ]; then rm -f "$OL_UNIT"; systemctl daemon-reload 2>/dev/null || true; fi
    if [ -f /etc/systemd/system/ollama.service ]; then
      systemctl stop ollama 2>/dev/null || true
      systemctl disable ollama 2>/dev/null || true
      rm -f /etc/systemd/system/ollama.service
      systemctl daemon-reload 2>/dev/null || true
    fi
    pgrep -x ollama | xargs -r kill 2>/dev/null || true
    BIN=$(command -v ollama || true)
    if [ -n "$BIN" ]; then rm -f "$BIN"; rm -rf "$(dirname "$(dirname "$BIN")")/lib/ollama"; fi
    rm -rf /root/.ollama /usr/share/ollama ${OLLAMA_MODELS:+"$OLLAMA_MODELS"}
    grep -q '^C_ENGINE=ollama' "$CONNECT_STATE" 2>/dev/null && rm -f "$CONNECT_STATE"
    id ollama >/dev/null 2>&1 && userdel ollama 2>/dev/null || true
    getent group ollama >/dev/null && groupdel ollama 2>/dev/null || true
    echo "Ollama and all its models removed."
  fi
  echo "Freed ~${USED:-0}."
  exit 0
fi

# ======================================================================================================
# llama.cpp sizing helpers (used by the first menu and the llama.cpp engine)
# ======================================================================================================
lc_file() {   # $1 = quant -> the file llama-server opens (a split GGUF's first part)
  if (( ${#LC_PARTS[@]} )); then echo "$LC_FILE_PREFIX$LC_FILE_SEP$1-${LC_PARTS[0]%%|*}.gguf"
  else echo "$LC_FILE_PREFIX$LC_FILE_SEP$1.gguf"; fi
}
ctx_tiers() { local c; for c in "${LC_CTX_TIERS[@]}"; do (( c <= LC_CTX_TRAIN )) && echo "$c"; done; return 0; }   # tiers this model was trained for
# weight bytes, ctx, kv bytes/token -> GPU MB (+ LoRA/draft), without the host-only share (LC_HOST_PCT)
nb_need_mb() { echo $(( ($1 * (100 - LC_HOST_PCT) / 100 + LC_EXTRA_BYTES) / 1048576 + $2 * $3 / 1048576 )); }
kv_bytes() { if [ "${1%+ram}" = f16 ]; then echo "$LC_KV_F16"; else echo "$LC_KV_Q8"; fi; }   # "q8_0+ram" -> q8_0
MOE_MIN_CACHE_PCT=60                           # the GPU expert cache must hold >= 60% of the experts (few RAM reads)
ram_gpu_mb() {   # MoE with experts in RAM: GPU MB for the non-expert weights + extras + cache (no expert cache)
  echo $(( $1 * (100 - LC_EXPERT_PCT - LC_HOST_PCT) / 100 / 1048576 + LC_EXTRA_BYTES / 1048576 + $2 * $(kv_bytes "$3") / 1048576 ))
}
vram_text() {   # $1 weight bytes, $2 ctx, $3 cache type -> "~44G" or, experts in RAM, "~23G+18G RAM"
  if [[ $3 == *+ram ]]; then echo "~$(( (LC_BUDGET_MB + LC_OVERHEAD_MB) / 1024 ))G VRAM + $(( $1 * LC_EXPERT_PCT / 100 / 1073741824 ))G RAM"
  else echo "~$(( ($(nb_need_mb "$1" "$2" "$(kv_bytes "$3")") + LC_OVERHEAD_MB) / 1024 ))G"; fi
}
draft_auto() {   # $1 = model id -> sets DRAFT=on/off when unset: on if it doesn't lower the best weights that fit
  [ -n "${DRAFT:-}" ] && return 0
  local q_on q_off c=131072
  DRAFT=off; lc_model "$1"; lc_budget; q_off=$(best_quant $c)
  if [ -z "$q_off" ]; then for c in $(ctx_tiers); do q_off=$(best_quant "$c"); [ -n "$q_off" ] && break; done; fi
  DRAFT=on;  lc_model "$1"; lc_budget; q_on=$(best_quant $c)
  if [ -n "$q_on" ] && [ "$q_on" = "$q_off" ]; then DRAFT=on; else DRAFT=off; fi
  DRAFT_AUTO_ON=$q_on DRAFT_AUTO_OFF=$q_off DRAFT_AUTO_CTX=$c
  return 0
}
lc_budget() { LC_BUDGET_MB=$(( FREE_MB - LC_OVERHEAD_MB - (GPU_COUNT - 1) * 1024 )); }   # llama.cpp splits layers over all cards
fit_kv() {   # $1 = weight bytes, $2 = ctx -> most precise cache that fits (f16, then q8_0), or nothing.
  # MoE models that don't fit try the experts-in-RAM tier ("f16+ram"/"q8_0+ram"): the non-expert weights, the
  # cache and an expert cache holding >= 60% of the experts on the GPU, all routed experts in system RAM (+4 GB).
  # The 60% rule keeps most expert reads on the GPU, so it favours smaller weights over slow RAM traffic.
  local kv
  for kv in f16 q8_0; do
    if (( $(nb_need_mb "$1" "$2" "$(kv_bytes $kv)") <= LC_BUDGET_MB )); then echo "$kv"; return 0; fi
  done
  if (( LC_EXPERT_PCT > 0 )) && (( $1 * LC_EXPERT_PCT / 100 / 1048576 + 4096 <= AVAIL_RAM_MB )); then
    for kv in f16 q8_0; do
      if (( LC_BUDGET_MB - $(ram_gpu_mb "$1" "$2" "$kv") >= $1 * LC_EXPERT_PCT / 100 / 1048576 * MOE_MIN_CACHE_PCT / 100 )); then
        echo "$kv+ram"; return 0
      fi
    done
  fi
  return 0
}
best_quant() {   # $1 = ctx -> best quant that fits (respects QUANT), or nothing
  local e q b s
  for e in "${LC_QUANTS[@]}"; do
    IFS='|' read -r q b s <<< "$e"
    [ -n "${QUANT:-}" ] && [ "$q" != "$QUANT" ] && continue
    [ -n "$(fit_kv "$b" "$1")" ] && { echo "$q"; return 0; }
  done
  return 0   # nothing fits: print nothing (a non-zero status would end the script under set -e)
}
mount_ok() {   # $1 = folder: can a container see a file in it?
  local new=0; [ -d "$1" ] || new=1
  mkdir -p "$1" && echo ok > "$1/.mount-probe"
  docker run --rm -v "$1:/p" busybox cat /p/.mount-probe 2>/dev/null | grep -qx ok
  local r=$?; rm -f "$1/.mount-probe"; [ "$new" = 1 ] && [ "$r" != 0 ] && rmdir "$1" 2>/dev/null; return $r
}

ask_port() {   # the last question: which port to serve on (skipped when given as an argument)
  [ -n "$PORT" ] && return 0
  if (( AUTO )); then PORT=20001; return 0; fi
  if (( HAS_WHIPTAIL && INTERACTIVE )); then
    PORT=$(whiptail --backtitle "$BACKTITLE" --title "Port" --inputbox "Last step: port for the AI endpoint on this machine:" 9 60 20001 3>&1 1>&2 2>&3) || { echo "Cancelled."; exit 0; }
  else
    read -rp "Port for the AI endpoint on this machine [20001]: " PORT || true   # no terminal: the default
    PORT=${PORT:-20001}
  fi
  [[ "$PORT" =~ ^[0-9]+$ ]] && PORT=$((10#$PORT)) && (( PORT >= 1 && PORT <= 65535 )) || { echo "Invalid port: '$PORT'"; exit 1; }
}

# --- Ollama: sizes and fit for every model in OL_MODELS (registry lookups, no installs) ---
model_mb() {   # download size of an Ollama registry ref (or a HF GGUF), in MB; empty on failure
  if [ -n "${HF_MB[$1]:-}" ]; then echo "${HF_MB[$1]}"; return 0; fi
  local name=${1%%:*} tag=latest
  [[ $1 == *:* ]] && tag=${1##*:}
  [[ $name == */* ]] || name="library/$name"
  curl -fsSL -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    "https://registry.ollama.ai/v2/$name/manifests/$tag" 2>/dev/null \
    | grep -o '"size":[0-9]*' | awk -F: '{s+=$2} END{if (s) print int(s/1048576)}'
}
ol_need_mb() { echo $(( $2 + KV[$1] * $3 * KV_PCT / 100 / 1024 + FIXED[$1] + OL_OVERHEAD_MB )); }   # model i, weights MB, ctx
confirm_yesno() {   # $1 = question (may contain \n); returns 0 for yes
  local a
  [ -n "${ONLY:-}" ] && return 0   # rows were named explicitly; don't re-ask
  if (( HAS_WHIPTAIL )); then
    whiptail --backtitle "$BACKTITLE" --title "Won't fit in VRAM" --yesno "$1" 16 78
  else
    printf '%b\n' "$1"; read -rp "[y/N] " a || true; [[ $a == [yY]* ]]
  fi
}

ol_prepare() {   # sets KV_TYPE/KV_PCT/QUANT_ORDER, the per-model arrays (REF, Q, MCTX, GB, ...) and REC
  # OLLAMA_KV_CACHE_TYPE is global: Ollama applies it to every model. q8_0 ~53% of the f16 bytes, q4_0 ~27%.
  KV_TYPE=${OLLAMA_KV_CACHE_TYPE:-f16}
  case "$KV_TYPE" in
    f16)  KV_PCT=100 ;;
    q8_0) KV_PCT=53 ;;
    q4_0) KV_PCT=27 ;;
    *) echo "Unsupported OLLAMA_KV_CACHE_TYPE '$KV_TYPE' (use f16, q8_0 or q4_0)"; exit 1 ;;
  esac
  export OLLAMA_KV_CACHE_TYPE="$KV_TYPE"
  export OLLAMA_FLASH_ATTENTION=1          # required for a quantized K/V cache
  export OLLAMA_NUM_PARALLEL=1             # each parallel slot needs its own full KV cache; subagents queue
  export OLLAMA_MAX_LOADED_MODELS=1        # sizes assume one model in VRAM at a time
  # Ollama unloads an idle model after 5 min, then the next request waits ~20-60 s to reload it. Keep it loaded.
  export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}
  QUANT_ORDER="Q8-first"; [ "${FAST:-0}" = 1 ] && QUANT_ORDER="Q4-first (FAST=1)"

  # --- Sizes: look every ref up in parallel (~0.5 s per registry call), then pick per model what fits ---
  echo "Checking Ollama model sizes..."
  SIZE_DIR=$(mktemp -d)   # cleanup appends to the shared EXIT trap, so it can't clobber an earlier handler
  SUMMON_EXIT_CMDS="${SUMMON_EXIT_CMDS:+$SUMMON_EXIT_CMDS; }rm -rf \"$SIZE_DIR\""
  local e r r8 r4 rk kv fx cp n note i s8 s4 q c s opt
  for e in "${OL_MODELS[@]}"; do
    IFS='|' read -r _ r8 r4 _ <<< "$e"
    for r in $r8 $r4; do model_mb "$r" > "$SIZE_DIR/$(echo "$r" | tr '/:' '__')" & done
  done
  wait
  cached_mb() { cat "$SIZE_DIR/$(echo "$1" | tr '/:' '__')" 2>/dev/null; }

  # Per model: the best option that fits -> REF (tag to pull), Q, MCTX (0 = too big), GB
  RANKS=() CAPS=() MNAME=() NOTE=() KV=() FIXED=() REF=() Q=() MCTX=() GB=()
  for e in "${OL_MODELS[@]}"; do
    IFS='|' read -r rk r8 r4 kv fx cp n note <<< "$e"
    i=${#RANKS[@]}
    RANKS+=("$rk") CAPS+=("$cp") MNAME+=("$n") NOTE+=("$note") KV+=("$kv") FIXED+=("$fx")
    s8=$(cached_mb "$r8"); s4=""; [ -n "$r4" ] && s4=$(cached_mb "$r4")
    REF+=("") Q+=("") MCTX+=(0) GB+=("?")
    # Q8-first reads ~29 GB of weights per decoded token on a 48 GB card; FAST=1 tries Q4 (~17 GB) first.
    if [ "${FAST:-0}" = 1 ]; then OPTS=("4 $OL_CTX_MAX" "4 $OL_CTX_MIN" "8 $OL_CTX_MAX" "8 $OL_CTX_MIN")
    else OPTS=("8 $OL_CTX_MAX" "8 $OL_CTX_MIN" "4 $OL_CTX_MAX" "4 $OL_CTX_MIN"); fi
    for opt in "${OPTS[@]}"; do
      read -r q c <<< "$opt"
      if [ "$q" = 8 ]; then r=$r8 s=$s8; else r=$r4 s=$s4; fi
      [ -n "$s" ] || continue
      if (( $(ol_need_mb $i $s $c) <= FREE_MB )); then REF[i]=$r Q[i]=$q MCTX[i]=$c GB[i]=$(( $(ol_need_mb $i $s $c) / 1024 )); break; fi
    done
    if [ -z "${REF[i]}" ]; then   # too big for any option: smallest @64K, flagged
      if [ -n "$s4" ]; then REF[i]=$r4 Q[i]=4 s=$s4; else REF[i]=$r8 Q[i]=8 s=${s8:-0}; fi
      GB[i]=$(( $(ol_need_mb $i $s $OL_CTX_MIN) / 1024 ))
    fi
    [ -z "$r4" ] && Q[i]=""   # own models: single tag, quant not shown
  done

  REC=-1
  for i in "${!RANKS[@]}"; do [[ ${RANKS[i]} != - && ${MCTX[i]} != 0 ]] && { REC=$i; break; }; done
  if (( REC >= 0 )); then REC_TXT="Recommended: ${MNAME[REC]} at ${Q[REC]:+Q${Q[REC]} }$((MCTX[REC]/1024))K context, $KV_TYPE KV"
  else REC_TXT="None of the curated models fit in VRAM."; fi
  fit_label() {
    local j=$1 qq=${Q[$1]:+Q${Q[$1]} }
    if (( MCTX[j] == 0 )); then echo "TOO BIG ~${GB[j]}G@64K"; else echo "${qq}~${GB[j]}G@$((MCTX[j]/1024))K"; fi
  }
  return 0
}

# ======================================================================================================
# Install: GPU, engine, port
# ======================================================================================================
banner
access_mode
(( DRY_RUN )) && echo "DRY RUN: the full wizard, then the plan. Nothing is installed, downloaded or started."
if (( DRY_RUN )) && [ -n "${SUMMON_GPU:-}" ]; then
  # Preview another machine: "NAME:TOTAL_MB" or "NAME:TOTAL_MB:CARDS" (free VRAM = total - 500 MB, like an idle card)
  IFS=: read -r g_name g_mb g_n <<< "$SUMMON_GPU"; g_n=${g_n:-1}
  [[ $g_mb =~ ^[0-9]+$ && $g_n =~ ^[1-9][0-9]*$ ]] || { echo "SUMMON_GPU must look like \"RTX 3090:24576\" (name:MB[:cards])"; exit 1; }
  GPU_INFO=$(for ((k = 0; k < g_n; k++)); do echo "$g_name, $g_mb, $(( g_mb - 500 ))"; done)
else
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found - no NVIDIA GPU/driver? (Preview: SUMMON_GPU=\"RTX 3090:24576\" $0 --dry-run)"; exit 1; }
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
fi
GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
GPU_NAME=$(echo "$GPU_INFO" | head -1 | cut -d, -f1)
FREE_MB=$(echo "$GPU_INFO" | awk -F', ' '{s+=$3} END{print s}')     # free VRAM summed across GPUs
TOTAL_MB=$(echo "$GPU_INFO" | awk -F', ' '{s+=$2} END{print s}')
# A server of ours that is still running holds its VRAM, but it gets replaced: count that VRAM as free, or a
# rerun to switch weights would find "no room". ~1 GB per card stays reserved for the driver/CUDA context.
if (( ! DRY_RUN || ${#SUMMON_GPU} == 0 )) && { lc_running || ol_running; }; then
  FREE_MB=$(( TOTAL_MB - GPU_COUNT * 1024 ))
  echo "Detected ${GPU_COUNT}x $GPU_NAME, ~$((FREE_MB/1024)) GB usable VRAM (counting the running server's, which gets replaced)"
else
  echo "Detected ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB free VRAM"
fi

AVAIL_RAM_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
if (( DRY_RUN )) && [ -n "${SUMMON_RAM:-}" ]; then AVAIL_RAM_MB=$SUMMON_RAM; fi   # preview: SUMMON_RAM=65536
INTERACTIVE=0; [ -t 0 ] && [ -t 1 ] && INTERACTIVE=1
if (( AUTO )); then   # just go: the recommended model, context and weights for this machine, port 20001
  INTERACTIVE=0
  echo "Auto mode: no questions. Picking the recommended model, context and weights for this machine."
fi
if (( INTERACTIVE && ! DRY_RUN )) && ! command -v whiptail >/dev/null; then
  apt_get whiptail >/dev/null 2>&1 || true   # optional: plain-text menus work too
fi
HAS_WHIPTAIL=0; command -v whiptail >/dev/null && HAS_WHIPTAIL=1

# --- One menu with every model: llama.cpp (tuned per model) and Ollama, side by side ---
# ENGINE=... or a model id as the first argument skips it (./summon.sh ornith9b 40000). "ollama" picks the
# recommended Ollama model (or ONLY=1,3 for several). Default: the first llama.cpp model that reaches 128K on this
# card (Nanbeige on 48 GB, Ornith 9B on 24 GB), else the best-ranked Ollama model that fits.
[ -n "$TARGET" ] && [ -z "${ENGINE:-}" ] && ENGINE=$TARGET
case "${ENGINE:-}" in llamacpp|llama.cpp) ENGINE=nanbeige ;; esac
if [ -z "${ENGINE:-}" ] && { [ -n "${ONLY:-}" ] || [ -n "${FAST:-}" ]; }; then ENGINE=ollama; fi
case "${ENGINE:-}" in
  nanbeige|qwen38pi|ornith35b|ornith9b|thinkingcap|bonsai27b|qwen38obl|ornith9bobl|feyospace|deepseek41obl|ollama|"") ;;
  *) echo "ENGINE must be one of: ${LC_MODEL_IDS[*]} ollama (got '$ENGINE')"; exit 1 ;;
esac
OL_PICK=""
if [ -z "${ENGINE:-}" ] || [ "$ENGINE" = ollama ]; then ol_prepare; fi
if [ -z "${ENGINE:-}" ] || { [ "$ENGINE" = ollama ] && [ -z "${ONLY:-}" ]; }; then
  # Spaced layout: an engine header per group and a gap between the groups (header rows can't be picked). Short
  # terminals get the compact list instead, so everything stays on one screen; descriptions are cut to the width.
  DEF_ENGINE="" LC_ITEMS=() OL_ITEMS=() ITEMS=() LC_HIDDEN=() OL_HIDDEN=()
  MENU_COLS=$(( $(tput cols 2>/dev/null || echo 124) - 4 )); (( MENU_COLS > 150 )) && MENU_COLS=150
  MENU_LINES=$(tput lines 2>/dev/null || echo 34)
  DESC_W=$(( MENU_COLS - 12 - 28 - 2 - 17 - 2 - 9 - 2 )); (( DESC_W < 12 )) && DESC_W=12   # borders/indent + name + fit + max
  row() {   # $1 name, $2 fit, $3 description, $4 longest context that fits ("max 384K", may be empty) -> one menu line
    local d=$3; (( ${#d} > DESC_W )) && d="${d:0:DESC_W-3}..."
    printf '%-28.28s  %-17s  %-9s  %s' "$1" "$2" "${4:-}" "$d"
  }
  DRAFT_USER=${DRAFT:-}
  for id in "${LC_MODEL_IDS[@]}"; do
    DRAFT=$DRAFT_USER; draft_auto "$id"; lc_model "$id"; lc_budget
    # Fit label like Ollama's: weights ~total VRAM @ context (128K, or the longest context that fits)
    q=$(best_quant 131072); c=131072
    if [ -z "$q" ]; then for c in $(ctx_tiers); do q=$(best_quant "$c"); [ -n "$q" ] && break; done; fi
    kv=""
    if [ -n "$q" ]; then
      for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$q" ] && b=$(echo "$e" | cut -d'|' -f2); done
      kv=$(fit_kv "$b" "$c")
    fi
    # Pre-select only models that run fully on the GPU (experts in RAM work, but are slower and unmeasured)
    [ -z "$DEF_ENGINE" ] && [ "$c" = 131072 ] && [ -n "$q" ] && [[ $kv != *+ram ]] && DEF_ENGINE=$id
    [ -n "$q" ] || { LC_HIDDEN+=("$LC_TITLE"); continue; }   # doesn't fit this machine: not offered
    if [[ $kv == *+ram ]]; then fit="$q +RAM@$((c/1024))K"
    else fit="$q ~$(( ($(nb_need_mb "$b" "$c" "$(kv_bytes "$kv")") + LC_OVERHEAD_MB) / 1024 ))G@$((c/1024))K"; fi
    maxc=""; for m in $(ctx_tiers); do [ -n "$(best_quant "$m")" ] && { maxc="max $((m/1024))K"; break; }; done   # step 1 offers up to this
    LC_ITEMS+=("$id" "$(row "$LC_TITLE" "$fit" "$LC_BLURB" "$maxc")")   # blurb from lc_model: can't drift from the id list
  done
  DRAFT=$DRAFT_USER
  for i in "${!RANKS[@]}"; do
    if (( MCTX[i] == 0 )); then OL_HIDDEN+=("${MNAME[i]}"); continue; fi   # too big even at 64K: not offered
    OL_ITEMS+=("ol:$((i+1))" "$(row "${MNAME[i]}" "$(fit_label $i)" "${NOTE[i]}")")
  done
  if [ "$ENGINE" = ollama ] && (( ${#OL_ITEMS[@]} == 0 )); then
    echo "ENGINE=ollama: no Ollama model fits this machine ($((FREE_MB/1024)) GB VRAM, $((AVAIL_RAM_MB/1024)) GB free RAM); the menu would only offer llama.cpp."; exit 1
  fi
  if (( ${#LC_ITEMS[@]} + ${#OL_ITEMS[@]} == 0 )); then
    echo "No model fits this machine ($((FREE_MB/1024)) GB VRAM, $((AVAIL_RAM_MB/1024)) GB free RAM)."; exit 1
  fi
  if (( ${#LC_ITEMS[@]} && ${#OL_ITEMS[@]} )) && (( MENU_LINES >= ${#LC_ITEMS[@]}/2 + ${#OL_ITEMS[@]}/2 + 3 + 11 )); then
    # (labels must not start with "-": whiptail would read them as options)
    ITEMS=(_h1 "== llama.cpp  .  tuned per model, verified downloads ================================="
           "${LC_ITEMS[@]}" _gap " "
           _h2 "== Ollama  .  pulled from ollama.com, sized to this card ============================="
           "${OL_ITEMS[@]}")
  else
    ITEMS=("${LC_ITEMS[@]}" "${OL_ITEMS[@]}")
  fi
  (( REC >= 0 )) && [ -z "$DEF_ENGINE" ] && DEF_ENGINE="ol:$((REC+1))"
  [ "$ENGINE" = ollama ] && (( ${#OL_ITEMS[@]} )) && DEF_ENGINE=$( (( REC >= 0 )) && echo "ol:$((REC+1))" || echo "${OL_ITEMS[0]}")
  DEF_ENGINE=${DEF_ENGINE:-${ITEMS[0]}}
  [[ $DEF_ENGINE == _* ]] && DEF_ENGINE=${LC_ITEMS[0]:-${OL_ITEMS[0]}}
  HIDDEN=("${LC_HIDDEN[@]}" "${OL_HIDDEN[@]}")
  (( ${#HIDDEN[@]} )) && echo "Not offered (too big for this machine): $(IFS=,; echo "${HIDDEN[*]}" | sed 's/,/, /g')"
  if [ -n "${CTX:-}" ] || [ -n "${QUANT:-}" ] || (( ! INTERACTIVE )); then PICKED=$DEF_ENGINE
  else
    PICKED=_h1
    while [[ $PICKED == _* ]]; do   # a header or the gap: show the menu again
      PICKED=$(CHOOSE_NOTAGS=1 CHOOSE_W=150 CHOOSE_TEXTLINES=1 choose "Summon v$SUMMON_VERSION - ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB" \
        "Which model do you want to run? Pre-selected: best at 128K. Size = weights ~VRAM@context, max = longest context." \
        "$DEF_ENGINE" "${ITEMS[@]}") || { echo "Cancelled."; exit 0; }
    done
  fi
  if [[ $PICKED == ol:* ]]; then ENGINE=ollama OL_PICK=$(( ${PICKED#ol:} - 1 )); else ENGINE=$PICKED; fi
fi

# ======================================================================================================
# Engine: llama.cpp (the model chosen above)
# ======================================================================================================
fetch_gguf() {   # $1 = repo URL, $2 = file, $3 = bytes, $4 = sha256 -> $DDIR/models/$2 (resumable, verified)
  # O(1) checks catch the realistic failures (truncated download: exact byte count; an HTML error page: GGUF magic).
  # VERIFY_SHA256=1 / --verify adds a full-file hash, which re-reads the whole file.
  local URL=$1 FILE=$2 BYTES=$3 SHA=$4
  local PART="$DDIR/models/$FILE.part" MODEL="$DDIR/models/$FILE"
  part_size() { stat -c %s "$PART" 2>/dev/null || echo 0; }
  if [ -f "$MODEL" ]; then
    echo "Already present: $MODEL (no download, no hash)"
  else
    # One HEAD request: HF publishes the blob's exact size and sha256 (x-linked-size / x-linked-etag).
    local EXPECT_SHA=$SHA META REMOTE_SIZE REMOTE_SHA SIZE
    META=$(curl -fsSIL --connect-timeout 10 --max-time 30 "$URL/$FILE" 2>/dev/null | tr -d '\r' || true)
    if [ -n "$META" ]; then
      REMOTE_SIZE=$(echo "$META" | awk -F': ' 'tolower($1)=="x-linked-size" {print $2; exit}')
      REMOTE_SHA=$(echo "$META" | awk -F': ' 'tolower($1)=="x-linked-etag" {gsub(/"/,"",$2); print $2; exit}')
      if [ -n "$REMOTE_SIZE" ] && [ "$REMOTE_SIZE" != "$BYTES" ]; then
        echo "HF reports $REMOTE_SIZE bytes for $FILE but this script expects $BYTES (repacked file?)."
        echo "Update the file table for $LC_ID in $0 before retrying."; exit 1
      fi
      if [ -n "$REMOTE_SHA" ] && [ "$REMOTE_SHA" != "$SHA" ]; then
        echo "Note: HF's sha256 for $FILE differs from the script's table; using HF's value."
        EXPECT_SHA=$REMOTE_SHA
      fi
    fi
    # -C - resumes a partial file, but curl errors (33/416) if the .part is already complete: reuse it instead.
    if [ -f "$PART" ] && (( $(part_size) > BYTES )); then
      echo "Removing an oversized partial download ($(part_size) bytes > $BYTES)."; rm -f "$PART"
    fi
    if [ -f "$PART" ] && (( $(part_size) == BYTES )); then
      echo "Reusing the completed download $PART"
    else
      local NEED_MB FREE_DISK_MB RETRY
      NEED_MB=$(( (BYTES - $(part_size)) / 1048576 + 1024 ))
      FREE_DISK_MB=$(df -Pm "$DDIR/models" | awk 'NR==2 {print $4}')
      if [ -n "$FREE_DISK_MB" ] && (( FREE_DISK_MB < NEED_MB )); then
        echo "Not enough disk: $FREE_DISK_MB MB free in $DDIR/models, the download needs ~$NEED_MB MB."
        echo "Delete other quants there (ls $DDIR/models) or rent a box with more disk."; exit 1
      fi
      [ -f "$PART" ] && echo "Resuming $PART from $(part_size) bytes..."
      if (( BYTES >= 1073741824 )); then echo "Downloading $FILE ($((BYTES/1073741824)) GB)..."
      else echo "Downloading $FILE ($((BYTES/1048576)) MB)..."; fi
      RETRY=(--retry 5 --retry-delay 5)   # --retry-all-errors (curl >= 7.71) also retries mid-transfer drops
      if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then RETRY+=(--retry-all-errors); fi
      curl -fL "${RETRY[@]}" --connect-timeout 20 -C - --progress-bar -o "$PART" "$URL/$FILE"
    fi
    SIZE=$(part_size)
    if (( SIZE != BYTES )); then
      echo "Incomplete download: $SIZE of $BYTES bytes in $PART. Rerun to resume."
      echo "If it never grows, the size in the script is stale (check the Hugging Face file list)."; exit 1
    fi
    [ "$(head -c 4 "$PART")" = GGUF ] || { echo "$PART doesn't start with the GGUF magic (bad download?); delete it and retry."; exit 1; }
    if [ "$VERIFY_SHA256" = 1 ]; then
      echo "Checking sha256 (this reads the whole file)..."
      echo "$EXPECT_SHA  $PART" | sha256sum -c --status || { echo "sha256 mismatch: delete $PART and retry"; exit 1; }
    else
      echo "Verified $SIZE bytes + GGUF header; sha256 skipped (VERIFY_SHA256=1 / --verify to check)."
    fi
    mv "$PART" "$MODEL"
  fi
  return 0
}

pad_draft() {   # $1 target file, $2 draft file, $3 draft sha256, $4 output: pad the draft's vocabulary, once
  # Runs tools/pad_draft.py from github.com/dedene/Nanbeige4.3-33B-DGX-Spark (MIT, (c) its author) in a throwaway
  # python:3.12-slim container with pinned gguf/numpy: the host gets nothing installed.
  local out="$DDIR/models/$4" stamp="$DDIR/models/.$4.inputs" inputs="$3 gguf==0.19.0 numpy==2.5.4"
  if [ -f "$out" ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$inputs" ]; then echo "Already present: $out"; return 0; fi
  rm -f "$out" "$out.part" "$stamp"
  cat > "$DDIR/models/.pad_draft.py" <<'PADPY'
"""Pad a drafter GGUF's vocabulary to the target's, so llama.cpp accepts it for speculative decoding.
From github.com/dedene/Nanbeige4.3-33B-DGX-Spark (tools/pad_draft.py), MIT licence."""
import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

VOCAB_TENSORS = ("token_embd.weight", "output.weight")


def value(field):
    kind = field.types[0]
    if kind == GGUFValueType.ARRAY:
        if field.types[-1] == GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8", errors="surrogateescape") for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    if kind == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode()
    return field.parts[field.data[0]].tolist()[0]


def copy_field(writer, name, field):
    kind = field.types[0]
    if kind == GGUFValueType.ARRAY and field.types[-1] == GGUFValueType.STRING:
        writer.add_array(name, value(field))
    elif kind == GGUFValueType.ARRAY:
        writer.add_key_value(name, value(field), kind, sub_type=field.types[-1])
    else:
        writer.add_key_value(name, value(field), kind)


def main(target_path, draft_path, out_path):
    target, draft = GGUFReader(target_path), GGUFReader(draft_path)
    tokens_tgt = value(target.fields["tokenizer.ggml.tokens"])
    tokens_dft = value(draft.fields["tokenizer.ggml.tokens"])
    n_tgt, n_dft = len(tokens_tgt), len(tokens_dft)
    if n_tgt < n_dft or tokens_tgt[:n_dft] != tokens_dft:
        sys.exit("the drafter's tokens are not a prefix of the target's: not the same tokenizer")
    print(f"vocab {n_dft:,} -> {n_tgt:,}")
    writer = GGUFWriter(out_path, arch=value(draft.fields["general.architecture"]))
    for name, field in draft.fields.items():
        if not (name.startswith(("GGUF.", "tokenizer.")) or name == "general.architecture"):
            copy_field(writer, name, field)
    for name, field in target.fields.items():
        if name.startswith("tokenizer."):
            copy_field(writer, name, field)
    for tensor in draft.tensors:
        data = np.asarray(tensor.data)
        if tensor.name in VOCAB_TENSORS and int(tensor.shape[-1]) == n_dft:
            data = np.concatenate([data, np.zeros((n_tgt - n_dft,) + data.shape[1:], dtype=data.dtype)])
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
PADPY
  echo "Padding the draft model's vocabulary to the target's (one-time, ~1-2 min)..."
  docker run --rm -v "$DDIR/models:/m" python:3.12-slim sh -c \
    "pip install -q --root-user-action=ignore --disable-pip-version-check gguf==0.19.0 numpy==2.5.4 && python /m/.pad_draft.py '/m/$1' '/m/$2' '/m/$4.part'" \
    || { echo "Padding the draft model failed; rerun, or skip the draft model with DRAFT=off."; exit 1; }
  [ "$(head -c 4 "$out.part" 2>/dev/null)" = GGUF ] || { echo "Padded draft is not a GGUF; rerun or use DRAFT=off."; exit 1; }
  mv "$out.part" "$out"; echo "$inputs" > "$stamp"; rm -f "$DDIR/models/.pad_draft.py"
  echo "Built $out"
}

install_llamacpp() {
  lc_model "$1"; lc_budget
  # --- Speed boost: the model's draft (MTP head / small draft model), when it has one ---
  # Default (DRAFT unset): on when it doesn't lower the weights that fit at 128K; otherwise the user decides, with
  # the trade-off shown and "No" pre-selected (precision first). DRAFT=on/off or CTX/QUANT skip the question.
  if [ -n "$LC_DRAFT_KIND" ] && [ "${SPEC:-}" != off ] && [ -z "${DRAFT:-}" ]; then
    draft_auto "$1"
    if [ "$DRAFT" = off ] && (( INTERACTIVE )) && [ -z "${CTX:-}${QUANT:-}" ] && [ -n "$DRAFT_AUTO_ON" ]; then
      local cost txt a
      cost="no extra download"
      [ -n "$LC_DRAFT" ] && cost="a $(( $(echo "$LC_DRAFT" | cut -d'|' -f2) / 1048576 )) MB download"
      txt="Speed boost: $LC_DRAFT_TITLE.\nIt drafts tokens that $LC_TITLE then checks: $LC_DRAFT_GAIN.\nQuality stays the same on average; outputs aren't bit-identical to a run without it.\n\nCost: $cost + VRAM. On this card, $((DRAFT_AUTO_CTX/1024))K context would then fit $DRAFT_AUTO_ON weights instead of $DRAFT_AUTO_OFF (less precise)."
      if (( HAS_WHIPTAIL )); then
        if whiptail --backtitle "$BACKTITLE" --title "Draft model: speed vs precision" --defaultno --yesno "$txt\n\nTrade precision for speed?" 16 92; then DRAFT=on; fi
      else
        printf '%b\n' "$txt"; read -rp "Trade precision for speed? [y/N] " a || true; [[ $a == [yY]* ]] && DRAFT=on
      fi
    fi
  fi
  lc_model "$1"; lc_budget
  if [ -n "${CTX:-}" ]; then
    [[ "$CTX" =~ ^[0-9]+$ ]] && (( CTX > 0 )) || { echo "CTX must be a positive number of tokens"; exit 1; }
    if (( CTX > LC_CTX_TRAIN )); then   # past its training the model degrades quietly (llama.cpp only warns in the log)
      echo "$LC_TITLE was trained for up to $LC_CTX_TRAIN tokens ($((LC_CTX_TRAIN/1024))K); CTX=$CTX would degrade it. Use CTX=$LC_CTX_TRAIN or less."
      exit 1
    fi
  fi
  if [ -n "${QUANT:-}" ]; then
    local qok=0 e
    for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$QUANT" ] && qok=1; done
    (( qok )) || { echo "Unknown QUANT '$QUANT' for $LC_TITLE (have: $(printf '%s ' "${LC_QUANTS[@]%%|*}"))"; exit 1; }
  fi
  TEMPERATURE=${TEMPERATURE:-$LC_TEMP}
  TOP_P=${TOP_P:-$LC_TOP_P}
  TOP_K=${TOP_K:-$LC_TOP_K}
  if [ -n "$TOP_K" ]; then [[ "$TOP_K" =~ ^[0-9]+$ ]] || { echo "TOP_K must be a whole number"; exit 1; }; fi
  for v in "$TEMPERATURE" "$TOP_P"; do
    [[ "$v" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "TEMPERATURE/TOP_P must be numbers (got '$v')"; exit 1; }
  done
  if [ -n "${THINK_BUDGET:-}" ]; then
    [[ "$THINK_BUDGET" =~ ^[0-9]+$ ]] || { echo "THINK_BUDGET must be a non-negative integer of tokens"; exit 1; }
  fi
  case "${VERIFY_SHA256:-0}" in 1|true|yes|on) VERIFY_SHA256=1 ;; *) VERIFY_SHA256=0 ;; esac

  # --- Context, then weights ---
  # Speed: every generated token reads all the weights plus the part of the cache in use (llama.cpp only attends
  # over used cells, padded to 256), so reserving 256K costs no speed until a conversation is that long.
  # Bigger weights are more precise and slower at every length. Default: 128K with the best weights that fit.
  local DEF_CTX="" c q b s e kv tot ITEMS DEF_Q PICK
  [ -n "$(best_quant 131072)" ] && DEF_CTX=131072
  if [ -z "$DEF_CTX" ]; then
    for c in $(ctx_tiers); do [ -n "$(best_quant "$c")" ] && { DEF_CTX=$c; break; }; done
  fi
  if [ -z "$DEF_CTX" ] && [ -z "${CTX:-}" ]; then
    echo "$LC_TITLE doesn't fit in $((FREE_MB/1024)) GB of VRAM$( (( LC_EXPERT_PCT )) && echo " + $((AVAIL_RAM_MB/1024)) GB free RAM")${QUANT:+ with QUANT=$QUANT} at 32K or more."
    echo "Try a smaller model (./summon.sh ornith9b) or Ollama (./summon.sh ollama)."
    exit 1
  fi
  if (( INTERACTIVE )) && [ -z "${CTX:-}" ] && [ -z "${QUANT:-}" ]; then
    ITEMS=()
    for c in $(ctx_tiers); do
      q=$(best_quant "$c"); [ -n "$q" ] || continue
      for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$q" ] && b=$(echo "$e" | cut -d'|' -f2); done
      kv=$(fit_kv "$b" "$c")
      ITEMS+=("$c" "$(printf '%3dK context    best weights here: %-7s + %-4s cache%s' $((c / 1024)) "$q" "${kv%+ram}" \
        "$([[ $kv == *+ram ]] && echo ', experts in RAM')")")
    done
    CTX=$(choose "$LC_TITLE - ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB" \
      "Step 1/2: context window. Reserving more costs VRAM (so smaller weights), not speed.\nSpeed drops with the tokens actually in a conversation, at any setting." \
      "$DEF_CTX" "${ITEMS[@]}") || { echo "Cancelled."; exit 0; }
  fi
  CTX=${CTX:-$DEF_CTX}
  if (( INTERACTIVE )) && [ -z "${QUANT:-}" ]; then
    ITEMS=() DEF_Q=""
    for e in "${LC_QUANTS[@]}"; do
      IFS='|' read -r q b s <<< "$e"
      kv=$(fit_kv "$b" "$CTX"); [ -n "$kv" ] || continue
      [ -n "$DEF_Q" ] || DEF_Q=$q
      if [[ $kv == *+ram ]]; then
        ITEMS+=("$q" "$(printf '%2d.%d GB weights + %-4s cache = %s   experts in RAM: slower' \
          $((b / 1073741824)) $((b * 10 / 1073741824 % 10)) "${kv%+ram}" "$(vram_text "$b" "$CTX" "$kv")")")
      else
        ITEMS+=("$q" "$(printf '%2d.%d GB weights + %-4s cache = %-5s GB   speed ~%3d%% of %s' \
          $((b / 1073741824)) $((b * 10 / 1073741824 % 10)) "$kv" "$(vram_text "$b" "$CTX" "$kv" | tr -d G)" \
          $((LC_REF_BYTES * 100 / b)) "$LC_REF_Q")")
      fi
    done
    (( ${#ITEMS[@]} )) || { echo "Nothing fits at $((CTX/1024))K on this card; pick a shorter CTX."; exit 1; }
    QUANT=$(choose "$LC_TITLE at $((CTX/1024))K context" \
      "Step 2/2: weights. Higher = more precise but slower: every token reads all of them.\n$LC_SPEED_NOTE Speed % is for an empty context;\nthe gap narrows as the conversation (and its cache) grows." \
      "$DEF_Q" "${ITEMS[@]}") || { echo "Cancelled."; exit 0; }
  fi
  QUANT=${QUANT:-$(best_quant "$CTX")}
  PICK=""
  for e in "${LC_QUANTS[@]}"; do
    IFS='|' read -r q b s <<< "$e"
    [ "$q" = "$QUANT" ] || continue
    kv=$(fit_kv "$b" "$CTX"); [ -n "$kv" ] && PICK="$q|$b|$s|$CTX|$kv"
  done
  if [ -z "$PICK" ]; then
    echo "$LC_TITLE ${QUANT:+$QUANT }doesn't fit in $((FREE_MB/1024)) GB of VRAM at $((CTX/1024))K context."
    echo "Pick a shorter CTX or a smaller QUANT (run without them for the menu)."
    exit 1
  fi
  IFS='|' read -r QUANT_SEL BYTES SHA CTX_SEL KV_SEL <<< "$PICK"
  ask_port
  local KV_TYPE_SEL=${KV_SEL%+ram} EXPERTS_RAM=0 MOE_CACHE_MB=0
  if [[ $KV_SEL == *+ram ]]; then
    EXPERTS_RAM=1
    MOE_CACHE_MB=$(( LC_BUDGET_MB - $(ram_gpu_mb "$BYTES" "$CTX_SEL" "$KV_SEL") ))
    (( MOE_CACHE_MB > BYTES * LC_EXPERT_PCT / 100 / 1048576 )) && MOE_CACHE_MB=$(( BYTES * LC_EXPERT_PCT / 100 / 1048576 ))
  fi
  echo "Plan: $QUANT_SEL weights ($((BYTES/1073741824)) GB$( (( LC_HOST_PCT )) && echo ", $(( BYTES * LC_HOST_PCT / 100 / 1073741824 )) GB of it read from disk")) + $KV_TYPE_SEL K/V at $((CTX_SEL/1024))K context = $(vram_text "$BYTES" "$CTX_SEL" "$KV_SEL")$( (( EXPERTS_RAM )) || echo " of $((FREE_MB/1024)) GB")"
  (( EXPERTS_RAM )) && echo "      MoE experts in system RAM ($(( BYTES * LC_EXPERT_PCT / 100 / 1073741824 )) GB of $((AVAIL_RAM_MB/1024)) GB free); $(( MOE_CACHE_MB * 100 / (BYTES * LC_EXPERT_PCT / 100 / 1048576) ))% of them cached on the GPU ($((MOE_CACHE_MB/1024)) GB)"
  (( LC_DRAFT_ON )) && echo "      Draft: $LC_DRAFT_TITLE ($LC_DRAFT_GAIN)"
  if (( DRY_RUN )); then
    echo
    echo "Would do (dry run, nothing done):"
    echo "  1. Install what's missing: curl, Docker, NVIDIA container toolkit"
    echo "  2. Download $(lc_file "$QUANT_SEL")$( (( ${#LC_PARTS[@]} > 1 )) && echo " + $(( ${#LC_PARTS[@]} - 1 )) more parts") ($((BYTES/1073741824)) GB), check size + GGUF header (sha256 with --verify)"
    echo "     from ${LC_REPO%/resolve/*}"
    if [ -n "$LC_LORA" ]; then
      echo "     + LoRA adapter ${LC_LORA%%|*} ($(( $(echo "$LC_LORA" | cut -d'|' -f2) / 1048576 )) MB) from $(echo "$LC_LORA" | cut -d'|' -f4 | sed 's#/resolve/main##')"
    fi
    if (( LC_DRAFT_ON )) && [ -n "$LC_DRAFT" ]; then
      echo "     + draft ${LC_DRAFT%%|*} ($(( $(echo "$LC_DRAFT" | cut -d'|' -f2) / 1048576 )) MB) from $(echo "$LC_DRAFT" | cut -d'|' -f4 | sed 's#/resolve/main##')"
      (( LC_DRAFT_PAD )) && echo "     + pad its vocabulary to the 33B's once (python:3.12-slim container, gguf 0.19.0)"
    fi
    echo "  3. Pull $LC_IMAGE"
    echo "  4. Start llama-server: $LC_TITLE $QUANT_SEL, -c $CTX_SEL, $KV_TYPE_SEL cache, $( (( EXPERTS_RAM )) && echo "experts in RAM + ${MOE_CACHE_MB} MB GPU expert cache" || echo "all layers on the GPU"),"
    echo "     temp ${TEMPERATURE} / top_p ${TOP_P}${TOP_K:+ / top_k $TOP_K}, speculation: $( [ "${SPEC:-}" = off ] && echo off || { (( LC_DRAFT_ON )) && echo "$LC_DRAFT_KIND draft$( [ "$LC_DRAFT_KIND" = dspark ] || echo " + n-gram")" || echo n-gram; }), host-RAM prompt cache"
    [ -n "$LC_KWARGS" ] && echo "     chat template: $LC_KWARGS"
    [ -n "$LC_TEMPLATE" ] && echo "     chat template file: ${LC_TEMPLATE##*|} (sha256-checked; replaces the GGUF's own)"
    (( ${#LC_ARGS[@]} )) && echo "     extra: ${LC_ARGS[*]}"
    echo "  5. Access: $(access_text llamacpp)"
    (( KEY_ON )) && echo "     API key: generated once and kept in the data folder (API_KEY=... to set your own)"
    (( TUNNEL_ON )) && echo "     + Cloudflare quick tunnel ($TUNNEL_IMAGE): a public https://...trycloudflare.com URL"
    echo "  6. Wait for /health, then test a tool call. Metrics at /metrics (Prometheus format)"
    echo "  7. Print the chat page URL, the API key and setup for Hermes, pi, opencode, OpenClaw and Cursor"
    echo
    echo "Endpoint: http://$HOST_HINT:$PORT/v1   model: $LC_ALIAS   context window: $CTX_SEL   chat page: http://$HOST_HINT:$PORT"
    echo "Run it for real:       $0 $LC_ID $PORT"
    echo "Same, without menus:   CTX=$CTX_SEL QUANT=$QUANT_SEL DRAFT=$( (( LC_DRAFT_ON )) && echo on || echo off) $0 $LC_ID $PORT"
    return 0
  fi

  # --- Prerequisites: only touch apt when something is missing ---
  local APT_PKGS=()
  command -v curl   >/dev/null || APT_PKGS+=(curl ca-certificates)
  command -v gpg    >/dev/null || APT_PKGS+=(gnupg)
  command -v docker >/dev/null || APT_PKGS+=(docker.io)
  if (( ${#APT_PKGS[@]} )); then
    apt_get "${APT_PKGS[@]}"
    apt_get btop bmon >/dev/null 2>&1 || echo "Note: optional monitors (btop/bmon) unavailable, continuing."
  fi

  # --- Docker + NVIDIA container toolkit (rented pods often have no systemd: start dockerd directly) ---
  local RESTART_DOCKER=""
  if ! command -v nvidia-ctk >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    APT_UPDATED=0; apt_get nvidia-container-toolkit   # new repo: needs its own apt-get update
    nvidia-ctk runtime configure --runtime=docker >/dev/null
    RESTART_DOCKER=1
  fi
  if [ -d /run/systemd/system ]; then
    if [ -n "$RESTART_DOCKER" ]; then systemctl restart docker; else systemctl start docker; fi
  else
    if [ -n "$RESTART_DOCKER" ] && [ -f /var/run/docker.pid ]; then kill "$(cat /var/run/docker.pid)" 2>/dev/null || true; sleep 3; fi
    if ! docker info >/dev/null 2>&1; then
      nohup dockerd >/var/log/dockerd.log 2>&1 &
      for i in $(seq 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
    fi
  fi
  docker info >/dev/null 2>&1 || { echo "Docker cannot run on this machine. Rent a VM or a Docker-in-Docker pod, or use ENGINE=ollama."; tail -5 /var/log/dockerd.log 2>/dev/null; exit 1; }
  local DDIR="" d
  for d in "${LC_DIRS[@]}"; do if mount_ok "$d"; then DDIR=$d; break; fi; done
  [ -n "$DDIR" ] || { echo "Docker can't mount any folder into containers here. Use a VM or a privileged pod, or ENGINE=ollama."; exit 1; }
  echo "Docker OK: $(docker --version). Data folder: $DDIR"

  # --- Model download(s): the weights, plus the LoRA adapter if the model has one ---
  local FILE LORA_FILE="" LORA_BYTES LORA_SHA LORA_REPO DRAFT_FILE="" DRAFT_SRC="" PART_FILES=() p pf pb ps need=0 free
  FILE=$(lc_file "$QUANT_SEL")
  mkdir -p "$DDIR/models"
  docker pull -q "$LC_IMAGE" >/dev/null 2>&1 & local PULL_PID=$!   # pull the ~3 GB image during the download
  if (( ${#LC_PARTS[@]} )); then   # split GGUF: every file, each verified; llama.cpp opens the first and finds the rest
    for p in "${LC_PARTS[@]}"; do
      IFS='|' read -r pf pb ps <<< "$p"; pf="$LC_FILE_PREFIX$LC_FILE_SEP$QUANT_SEL-$pf.gguf"; PART_FILES+=("$pf")
      [ -f "$DDIR/models/$pf" ] || need=$(( need + pb - $(stat -c %s "$DDIR/models/$pf.part" 2>/dev/null || echo 0) ))
    done
    free=$(df -Pm "$DDIR/models" | awk 'NR==2 {print $4}')
    if [ -n "$free" ] && (( free < need / 1048576 + 1024 )); then
      echo "Not enough disk: $LC_TITLE needs ~$(( need / 1073741824 + 1 )) GB more in $DDIR/models, $((free/1024)) GB free."; exit 1
    fi
    for p in "${LC_PARTS[@]}"; do
      IFS='|' read -r pf pb ps <<< "$p"
      fetch_gguf "$LC_REPO" "$LC_FILE_PREFIX$LC_FILE_SEP$QUANT_SEL-$pf.gguf" "$pb" "$ps"
    done
  else
    fetch_gguf "$LC_REPO" "$FILE" "$BYTES" "$SHA"
  fi
  if [ -n "$LC_LORA" ]; then
    IFS='|' read -r LORA_FILE LORA_BYTES LORA_SHA LORA_REPO <<< "$LC_LORA"
    fetch_gguf "$LORA_REPO" "$LORA_FILE" "$LORA_BYTES" "$LORA_SHA"
  fi
  local TPL_FILE="" TPL_SHA TPL_URL
  if [ -n "$LC_TEMPLATE" ]; then   # a few KB: always hash-checked
    IFS='|' read -r TPL_FILE TPL_SHA TPL_URL <<< "$LC_TEMPLATE"
    if ! echo "$TPL_SHA  $DDIR/models/$TPL_FILE" | sha256sum -c --status 2>/dev/null; then
      curl -fsSL --retry 3 --connect-timeout 20 -o "$DDIR/models/$TPL_FILE" "$TPL_URL" \
        && echo "$TPL_SHA  $DDIR/models/$TPL_FILE" | sha256sum -c --status \
        || { rm -f "$DDIR/models/$TPL_FILE"; echo "Chat template download failed or its sha256 changed: $TPL_URL"; exit 1; }
    fi
  fi
  if (( LC_DRAFT_ON )) && [ -n "$LC_DRAFT" ]; then
    local D_FILE D_BYTES D_SHA D_REPO
    IFS='|' read -r D_FILE D_BYTES D_SHA D_REPO <<< "$LC_DRAFT"
    fetch_gguf "$D_REPO" "$D_FILE" "$D_BYTES" "$D_SHA"
    DRAFT_SRC=$D_FILE DRAFT_FILE=$D_FILE
    if (( LC_DRAFT_PAD )); then DRAFT_FILE="${D_FILE%.gguf}.pad.gguf"; pad_draft "$FILE" "$D_FILE" "$D_SHA" "$DRAFT_FILE"; fi
  fi
  local OTHERS
  OTHERS=$(ls "$DDIR/models" | grep -vxF -e "$FILE" "${PART_FILES[@]/#/-e}" -e "${LORA_FILE:-//}" -e "${TPL_FILE:-//}" -e "${DRAFT_SRC:-//}" -e "${DRAFT_FILE:-//}" || true)
  [ -n "$OTHERS" ] && echo "Other model files in $DDIR/models (kept, switching back is instant): $OTHERS"

  # --- Start (or replace) the server ---
  wait "$PULL_PID" || docker pull -q "$LC_IMAGE" >/dev/null   # retry in the foreground so a pull error shows
  stop_ollama; stop_llamacpp; wait_port_free
  (( TUNNEL_ON )) && BIND=127.0.0.1   # the tunnel reaches it locally; nothing else needs the raw port
  local KEY_FILE="" DOCKER_ARGS=()
  if (( KEY_ON )); then
    KEY_FILE="$DDIR/api-key"; ensure_api_key "$KEY_FILE"
    DOCKER_ARGS=(-v "$KEY_FILE:/run/summon/api-key:ro")
  fi
  apply_access docker   # before the port opens
  # Prompt cache in host RAM: with one slot, a side request (chat title, subagent) overwrites the main
  # conversation's K/V. llama-server parks the old state in RAM first, but its default 8 GiB cap holds only ~75K
  # tokens of this model. Give it half the available host RAM (minus the MoE experts that will live there).
  local AVAIL_MB CACHE_RAM_MB SERVER_ARGS
  AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  (( EXPERTS_RAM )) && AVAIL_MB=$(( AVAIL_MB - BYTES * LC_EXPERT_PCT / 100 / 1048576 ))
  # Half of the RAM left after the experts; never negative (0 turns the prompt cache off rather than overcommitting RAM)
  CACHE_RAM_MB=$(( AVAIL_MB / 2 )); (( CACHE_RAM_MB < 0 )) && CACHE_RAM_MB=0
  # --metrics: Prometheus-format stats at /metrics (tokens/s, prompt and decode counts, draft acceptance, KV use).
  # Same port and access rules as the API: local mode / your IP only.
  SERVER_ARGS=(--cache-ram "$CACHE_RAM_MB" --metrics)
  (( KEY_ON )) && SERVER_ARGS+=(--api-key-file /run/summon/api-key)   # chat page and /health stay public
  if [ -n "${THINK_BUDGET:-}" ]; then SERVER_ARGS+=(--reasoning-budget "$THINK_BUDGET"); fi
  if [ -n "$TOP_K" ]; then SERVER_ARGS+=(--top-k "$TOP_K"); fi
  if [ -n "$LC_KWARGS" ]; then SERVER_ARGS+=(--chat-template-kwargs "$LC_KWARGS"); fi
  if [ -n "$LORA_FILE" ]; then SERVER_ARGS+=(--lora "/models/$LORA_FILE"); fi
  if [ -n "$TPL_FILE" ]; then SERVER_ARGS+=(--chat-template-file "/models/$TPL_FILE"); fi
  SERVER_ARGS+=("${LC_ARGS[@]}")
  # Speculative decoding: the model's draft (MTP head or a small draft model) plus n-gram drafts from text already
  # in the context. The model verifies every drafted token: quality holds on average, but outputs aren't
  # bit-identical (batched verification changes float rounding). SPEC=off turns all of it off.
  if [ "${SPEC:-}" != off ]; then
    local SPEC_TYPES=ngram-mod
    if (( LC_DRAFT_ON )); then
      case "$LC_DRAFT_KIND" in
        model) SPEC_TYPES=draft-simple,ngram-mod
               SERVER_ARGS+=(-md "/models/$DRAFT_FILE" --spec-draft-ngl all --spec-draft-n-max "${LC_DRAFT_NMAX:-4}" --spec-draft-p-min 0.6) ;;
        mtp)   SPEC_TYPES=draft-mtp,ngram-mod; SERVER_ARGS+=(-md "/models/$DRAFT_FILE" --spec-draft-n-max "${LC_DRAFT_NMAX:-2}") ;;
        mtp-inline) SPEC_TYPES=draft-mtp,ngram-mod; SERVER_ARGS+=(--spec-draft-n-max "${LC_DRAFT_NMAX:-2}") ;;
        # DSpark: n-gram drafts added nothing on top of it in its makers' tests, and 3 drafted tokens were slower than 2
        dspark) SPEC_TYPES=draft-dspark
                SERVER_ARGS+=(-md "/models/$DRAFT_FILE" --spec-draft-ngl all --spec-draft-n-max "${LC_DRAFT_NMAX:-2}") ;;
      esac
      SERVER_ARGS+=(-ctkd "$KV_TYPE_SEL" -ctvd "$KV_TYPE_SEL")
    fi
    SERVER_ARGS+=(--spec-type "$SPEC_TYPES")
    if [[ $SPEC_TYPES == *ngram-mod* ]]; then   # dspark's SPEC_TYPES has no ngram-mod: n-gram drafts added nothing on top of it
      SERVER_ARGS+=(--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 48)
    fi
  fi
  # MoE experts in system RAM: --cpu-moe keeps all routed experts on the CPU side, --moe-cache-mib caches the hot
  # ones on the GPU (llama.cpp leaves that cache off by default, which roughly halves throughput).
  if (( EXPERTS_RAM )); then SERVER_ARGS+=(--cpu-moe --moe-cache-mib "$MOE_CACHE_MB"); fi
  # -ngl 999 + --fit off: every layer on the GPU or fail loudly (no silent CPU offload, no silently smaller -c).
  # -ctv q8_0 needs Flash Attention (-fa on). -np 1: one slot, the whole context for one conversation.
  # --cache-reuse: reuse cached K/V chunks when an agent edits earlier messages. --jinja: native XML tool calls.
  docker run -d --name "$LC_NAME" --restart unless-stopped --gpus all \
    -v "$DDIR/models:/models:ro" "${DOCKER_ARGS[@]}" -p "$BIND:$PORT:8080" "$LC_IMAGE" \
    -m "/models/$FILE" --alias "$LC_ALIAS" --host 0.0.0.0 --port 8080 \
    --jinja \
    -c "$CTX_SEL" -np 1 -ngl 999 --fit off -fa on -ctk "$KV_TYPE_SEL" -ctv "$KV_TYPE_SEL" \
    --cache-reuse 256 --temp "$TEMPERATURE" --top-p "$TOP_P" "${SERVER_ARGS[@]}" >/dev/null

  echo "Loading the model (1-3 min)..."
  for i in $(seq 300); do
    # a crash at load (e.g. out of VRAM) shows as a stopped container or, with the restart policy, a restart count
    if ! docker ps --format '{{.Names}}' | grep -qx "$LC_NAME" || [ "$(docker inspect -f '{{.RestartCount}}' "$LC_NAME" 2>/dev/null)" != 0 ]; then
      docker update --restart=no "$LC_NAME" >/dev/null 2>&1 || true
      echo "The server stopped. Last log lines:"; docker logs --tail 25 "$LC_NAME" 2>&1
      echo "If it ran out of memory, rerun with a smaller context, e.g.: CTX=$(( CTX_SEL / 2 )) $0 $PORT"; exit 1
    fi
    curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 2
  done
  curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "Server not healthy after 10 min:"; docker logs --tail 25 "$LC_NAME" 2>&1; exit 1; }

  # --- Smoke test: a tool call must come back as an OpenAI tool_calls entry ---
  echo "Testing tool calling..."
  local R TOOLS VRAM
  local AUTH=(); (( KEY_ON )) && AUTH=(-H "Authorization: Bearer $(head -1 "$KEY_FILE")")
  R=$(curl -fs --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" "${AUTH[@]}" -H 'Content-Type: application/json' -d '{
    "model": "'"$LC_ALIAS"'", "max_tokens": 4096,
    "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}' || true)
  if echo "$R" | grep -q '"tool_calls"' && echo "$R" | grep -q 'get_weather'; then TOOLS="OK"; else TOOLS="NOT DETECTED - check: docker logs $LC_NAME"; fi
  VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tr '\n' ' ')

  local URL
  if (( TUNNEL_ON )); then
    echo "Opening the HTTPS tunnel..."
    start_tunnel "$PORT"
    [ -n "$TUNNEL_URL" ] || echo "The tunnel didn't report a URL; check: docker logs $TUNNEL_NAME"
    URL=${TUNNEL_URL:-http://$(public_host):$PORT}
  else
    URL="http://$(public_host):$PORT"
  fi
  echo
  echo "Done. $LC_TITLE $QUANT_SEL is serving on $BIND:$PORT (OpenAI API, llama.cpp)"
  echo "  Context:     $CTX_SEL tokens, $KV_TYPE_SEL K/V cache (VRAM used: $VRAM)"
  echo "  Sampling:    temp $TEMPERATURE / top_p $TOP_P${TOP_K:+ / top_k $TOP_K}${THINK_BUDGET:+, thinking capped at $THINK_BUDGET tokens}"
  echo "  Speculation: $( [ "${SPEC:-}" = off ] && echo off || { (( LC_DRAFT_ON )) && echo "$LC_DRAFT_TITLE + n-gram drafts" || echo "n-gram drafts"; }) (SPEC=off: reproducible runs)"
  (( EXPERTS_RAM )) && echo "  MoE experts: in system RAM, ${MOE_CACHE_MB} MB of hot experts cached on the GPU"
  echo "  Tool calls:  $TOOLS"
  echo "  Access:      $(access_text llamacpp)"
  echo "  Clients:     set the context window to $CTX_SEL; one request at a time (others queue)"
  echo "  Prompt cache: up to $((CACHE_RAM_MB/1024)) GB of host RAM keeps conversations across side requests"
  echo "  Metrics:     $URL/metrics   (Prometheus format; send the API key)"
  echo "  Logs:        docker logs -f $LC_NAME          Monitor: btop / nvidia-smi"
  local REASON=true; [[ $LC_KWARGS == *'"enable_thinking": false'* ]] && REASON=false
  save_connect llamacpp "$URL" "$LC_ALIAS" "$LC_TITLE" "$CTX_SEL" "$REASON" "$KEY_FILE" "$TUNNEL_ON" 1
  print_connect
}

# ======================================================================================================
# Engine: Ollama
# ======================================================================================================
install_ollama() {
  # --- Prerequisites (not in a dry run) ---
  local APT_PKGS=()
  if (( ! DRY_RUN )); then
    command -v curl   >/dev/null || APT_PKGS+=(curl ca-certificates)
    command -v screen >/dev/null || APT_PKGS+=(screen)
    command -v zstd   >/dev/null || APT_PKGS+=(zstd)
    if (( ${#APT_PKGS[@]} )); then
      apt_get "${APT_PKGS[@]}"
      apt_get btop bmon >/dev/null 2>&1 || echo "Note: optional monitors (btop/bmon) unavailable, continuing."
    fi
    command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  fi

  # What to pull: the model picked in the menu, or the rows in ONLY=1,3 (several at once, no menu)
  local N=${#RANKS[@]} t ONLY_PICKS=""
  SEL=()
  if [ -n "${ONLY:-}" ]; then
    for t in ${ONLY//,/ }; do
      if [[ "$t" =~ ^[0-9]+$ ]] && (( t >= 1 && t <= N )); then SEL+=($((t-1))); ONLY_PICKS+=" $t"
      else echo "Ignoring invalid row in ONLY: '$t' (rows 1..$N: $0 --help)"; fi
    done
    (( ${#SEL[@]} )) || { echo "ONLY=${ONLY} has no valid Ollama rows (1..$N)."; exit 1; }
  elif [ -n "$OL_PICK" ]; then
    SEL=("$OL_PICK")
  elif (( REC >= 0 )); then
    SEL=("$REC")
  fi
  (( ${#SEL[@]} )) || { echo "No Ollama model fits this machine ($((FREE_MB/1024)) GB VRAM, $((AVAIL_RAM_MB/1024)) GB free RAM); starting an empty server would be pointless."; exit 1; }
  (( REC >= 0 )) && echo "$REC_TXT"
  echo "Ollama: $QUANT_ORDER, $KV_TYPE KV cache. Pulling: $(for i in "${SEL[@]}"; do printf '%s ' "${MNAME[i]}"; done)"

  # Warn if anything selected won't fit even at Q4 @64K
  local TOO_BIG="" KEEP
  for i in "${SEL[@]}"; do [ "${MCTX[i]}" = 0 ] && TOO_BIG+="${MNAME[i]} (~${GB[i]}GB @64K)\n"; done
  if [ -n "$TOO_BIG" ] && ! confirm_yesno \
      "Only $((FREE_MB/1024))GB VRAM. These will spill to CPU (slow):\n\n${TOO_BIG}\nPull them anyway? (No = skip them)"; then
    KEEP=(); for i in "${SEL[@]}"; do [ "${MCTX[i]}" = 0 ] || KEEP+=($i); done; SEL=("${KEEP[@]}")
  fi
  ask_port

  if (( DRY_RUN )); then
    echo
    echo "Would do (dry run, nothing done):"
    echo "  1. Install what's missing: curl, screen, zstd, Ollama"
    echo "  2. Start Ollama on $BIND:$PORT ($KV_TYPE KV cache, 128K default context, models stay loaded)"
    echo "     as $(has_systemd && echo "a systemd service (summon-ollama, starts on boot)" || echo "a screen session (no systemd: not restarted after a reboot)")"
    echo "  3. Access: $(access_text ollama)  (Ollama has no API keys)"
    echo "  4. Pull:"
    for i in "${SEL[@]}"; do
      if (( MCTX[i] )); then echo "       ${REF[i]}  ->  $((MCTX[i]/1024))K context, ~${GB[i]} GB VRAM"
      else echo "       ${REF[i]}  ->  TOO BIG for this card (~${GB[i]} GB at 64K): would spill to the CPU (slow)"; fi
    done
    (( ${#SEL[@]} )) || echo "       (nothing selected)"
    echo
    echo "Endpoint: http://$HOST_HINT:$PORT/v1"
    return 0
  fi

  # --- Start the server detached in screen, default context 128K ---
  stop_llamacpp; stop_ollama; wait_port_free
  apply_access host   # before the port opens
  start_ollama_server
  echo "Waiting for ollama on port $PORT..."
  for i in $(seq 60); do curl -fs --max-time 2 "http://127.0.0.1:$PORT" >/dev/null 2>&1 && break; sleep 1; done
  curl -fs --max-time 2 "http://127.0.0.1:$PORT" >/dev/null 2>&1 || {
    echo "Ollama didn't come up on $PORT after 60 s. Check: journalctl -u summon-ollama or screen -r ollama"; exit 1; }

  # Pull selected models. Ones that don't fit at 128K get num_ctx 64K baked in (same name).
  for i in "${SEL[@]}"; do
    OLLAMA_HOST=127.0.0.1:$PORT ollama pull "${REF[i]}"
    if [ "${MCTX[i]}" != "$OL_CTX_MAX" ]; then
      if [[ ${REF[i]} == hf.co/* ]]; then
        echo "${REF[i]}: fits only at 64K here - set num_ctx 65536 in the client instead"
      else
        printf 'FROM %s\nPARAMETER num_ctx %s\n' "${REF[i]}" $OL_CTX_MIN > /tmp/Modelfile.ctx
        OLLAMA_HOST=127.0.0.1:$PORT ollama create "${REF[i]}" -f /tmp/Modelfile.ctx
        rm -f /tmp/Modelfile.ctx
        echo "${REF[i]}: context set to 64K"
      fi
    fi
  done
  # Load each model once to confirm it really sits in VRAM at its context
  local hdr=1 m
  if (( ${#SEL[@]} )); then
    echo; echo "Checking real VRAM fit (PROCESSOR should be 100% GPU):"
    for i in "${SEL[@]}"; do
      m=${REF[i]}
      curl -s "http://127.0.0.1:$PORT/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":\"1m\"}" >/dev/null || true
      OLLAMA_HOST=127.0.0.1:$PORT ollama ps | awk -v m="$m" -v hdr=$hdr '(NR==1 && hdr) || $1==m'; hdr=0
      curl -s "http://127.0.0.1:$PORT/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true
    done
  fi
  echo
  echo "Done. Ollama is serving on $BIND:$PORT ($KV_TYPE KV cache, $((OL_CTX_MAX/1024))K default context, $QUANT_ORDER)"
  for i in "${SEL[@]}"; do echo "  ${REF[i]}  ->  $((MCTX[i] ? MCTX[i]/1024 : 64))K context"; done
  echo "  Access:            $(access_text ollama)  (Ollama has no API keys)"
  echo "  Runs as:           $OL_MODE"
  [[ $OL_MODE == screen* ]] && echo "  Attach to server:  screen -r ollama   (detach: Ctrl+A then D)"
  [[ $OL_MODE == systemd* ]] && echo "  Logs:              journalctl -u summon-ollama -f"
  echo "  Run a model:       OLLAMA_HOST=127.0.0.1:$PORT ollama run <model>"
  echo "  Monitor:           btop / bmon"
  if (( ${#SEL[@]} )); then
    local i0=${SEL[0]}
    save_connect ollama "http://$(public_host):$PORT" "${REF[i0]}" "${MNAME[i0]}" "$(( MCTX[i0] ? MCTX[i0] : OL_CTX_MIN ))" \
      "$( [[ ${CAPS[i0]} == *R* ]] && echo true || echo false)" "" 0 0
    print_connect
  fi
}

# ======================================================================================================
# Run the chosen engine (each asks for the port last, after its model choices)
# ======================================================================================================
if [ "$ENGINE" = ollama ]; then install_ollama; else install_llamacpp "$ENGINE"; fi
