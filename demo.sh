#!/usr/bin/env bash
# Try the demo: GPT-2 running in your browser, from a fresh clone, in one command.
#
#     ./demo.sh             check, set up, build, serve, open the browser
#     ./demo.sh --check     only check the prerequisites
#     ./demo.sh --no-open   serve, but do not open a browser
#     ./demo.sh --port 8080 serve on another port (default 8000, or the next free one)
#
# Every step is skipped when its output already exists, so running it again
# just serves the page. First run: about 1.5 GB of downloads (the model and
# CPU-only PyTorch) and 2.5 GB of disk; a few minutes on a fast connection.
#
# What it does, in order -- each step is an ordinary command in the repo, so
# nothing here is hidden:
#   1. a Python venv with torch and numpy -- used to convert the checkpoint and
#      pick which embedding columns stay fp32; nothing torch-based runs in the page
#   2. scripts/download_weights.py      GPT-2 124M from Hugging Face, checksum-verified
#   3. scripts/convert_weights.py       -> weights/gpt2-124m.bin, the engine's format
#   4. scripts/quantize_weights.py --wte-outliers 8
#                                       -> weights/gpt2-124m-int8-wte-o8.bin, 129 MB
#   5. web/build.sh wasm_simd128        the C++ engine compiled to WebAssembly
#   6. web/pack.py                      assemble the static site -- the same one
#                                       GitHub Pages publishes
#   7. web/serve.sh                     a local web server, localhost only
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

check_only=0
open_browser=1
port=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) check_only=1 ;;
    --no-open) open_browser=0 ;;
    --port) port="${2:?--port needs a number}"; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

bold=$'\e[1m'; dim=$'\e[2m'; red=$'\e[31m'; green=$'\e[32m'; reset=$'\e[0m'
[[ -t 1 ]] || { bold=""; dim=""; red=""; green=""; reset=""; }
step() { printf '\n%s==> %s%s\n' "$bold" "$1" "$reset"; }
ok()   { printf '  %s✓%s %s\n' "$green" "$reset" "$1"; }
skip() { printf '  %s✓ %s (already done)%s\n' "$dim" "$1" "$reset"; }

# --- prerequisites -----------------------------------------------------------
# All of them, before anything is downloaded, so a missing tool is found in
# seconds rather than after a gigabyte.
step "Checking prerequisites"

# Emscripten is often installed without its drivers on PATH: Arch keeps them
# in /usr/lib/emscripten, and emsdk needs its env script sourced.
if ! command -v emcmake >/dev/null 2>&1; then
  for d in /usr/lib/emscripten "${EMSDK:-}/upstream/emscripten" "$HOME/emsdk/upstream/emscripten"; do
    if [[ -x "$d/emcmake" ]]; then export PATH="$PATH:$d"; break; fi
  done
fi

missing=0
need() {  # need <command> <what it is for> <how to install>
  if command -v "$1" >/dev/null 2>&1; then
    ok "$1 ${dim}($2)${reset}"
  else
    printf '  %s✗ %s%s -- %s\n      install: %s\n' "$red" "$1" "$reset" "$2" "$3"
    missing=1
  fi
}
need python3 "setup scripts" "https://www.python.org/downloads/ (3.10 or newer)"
if command -v python3 >/dev/null 2>&1 &&
   ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  printf '  %s✗ python3 is %s; 3.10 or newer is needed%s\n' "$red" \
    "$(python3 -c 'import platform; print(platform.python_version())')" "$reset"
  missing=1
fi
need cmake "configures the engine build" "https://cmake.org/download/, or your package manager"
need ninja "runs the engine build" "your package manager (ninja or ninja-build), or pip install ninja"
need emcmake "Emscripten, compiles C++ to WebAssembly" \
  "https://emscripten.org/docs/getting_started/downloads.html -- git clone https://github.com/emscripten-core/emsdk && cd emsdk && ./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh"

if [[ $missing -ne 0 ]]; then
  printf '\n%sInstall the missing tools above, then run ./demo.sh again.%s\n' "$bold" "$reset"
  exit 1
fi
if [[ $check_only -eq 1 ]]; then
  printf '\n%sEverything needed is installed.%s Run ./demo.sh to set up and serve the demo.\n' "$green" "$reset"
  exit 0
fi

# --- 1. Python environment ---------------------------------------------------
step "1/7  Python environment (.venv)"
py=".venv/bin/python"
if [[ -x "$py" ]] && "$py" -c 'import torch, numpy' 2>/dev/null; then
  skip "torch and numpy in .venv"
else
  [[ -x "$py" ]] || python3 -m venv .venv
  # The CPU-only wheel: a few hundred MB instead of several GB of CUDA.
  "$py" -m pip install --quiet --upgrade pip
  "$py" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  "$py" -m pip install --quiet numpy
  ok "torch and numpy installed in .venv"
fi

# --- 2-4. Weights -------------------------------------------------------------
step "2/7  GPT-2 124M checkpoint (548 MB, checksum-verified)"
"$py" scripts/download_weights.py | sed 's/^/  /'

step "3/7  Convert to the engine's format"
if [[ -f weights/gpt2-124m.bin ]]; then
  skip "weights/gpt2-124m.bin"
else
  "$py" scripts/convert_weights.py | sed 's/^/  /'
fi

step "4/7  Quantize for the browser (int8, 129 MB)"
if [[ -f weights/gpt2-124m-int8-wte-o8.bin ]]; then
  skip "weights/gpt2-124m-int8-wte-o8.bin"
else
  "$py" scripts/quantize_weights.py --wte-outliers 8 | tail -4 | sed 's/^/  /'
fi

# --- 5. The engine, as WebAssembly -------------------------------------------
step "5/7  Build the engine for WebAssembly"
# Incremental: a no-op when nothing changed. The first run also compiles
# Emscripten's system libraries, which is the slow part.
mkdir -p web/build
web/build.sh wasm_simd128 >web/build/build.log 2>&1 ||
  { printf '  %s✗ build failed; the last lines of web/build/build.log:%s\n' "$red" "$reset"; tail -20 web/build/build.log; exit 1; }
ok "web/build/engine-wasm_simd128/tools/gpt2_web.wasm"

# --- 6. The site -------------------------------------------------------------
step "6/7  Assemble the site"
# Refuses to build unless the weights are byte-identical to the pinned,
# evaluated file -- the same check the GitHub Pages deploy makes.
python3 web/pack.py | sed 's/^/  /'

# --- 7. Serve ----------------------------------------------------------------
step "7/7  Serve the demo"
free_port() {
  python3 - "$1" <<'EOF'
import socket, sys
for p in range(int(sys.argv[1]), int(sys.argv[1]) + 50):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p)); print(p); break
        except OSError:
            pass
EOF
}
port="$(free_port "${port:-8000}")"
url="http://localhost:$port/"

if [[ $open_browser -eq 1 ]]; then
  # `open` only on macOS: on Linux it is often openvt, which is not a browser.
  opener=""
  if [[ "$(uname)" == Darwin ]]; then opener=open
  elif command -v xdg-open >/dev/null 2>&1; then opener=xdg-open; fi
  if [[ -n "$opener" ]]; then ( sleep 1; "$opener" "$url" >/dev/null 2>&1 || true ) & fi
fi

cat <<EOF

  ${bold}Open ${url}${reset}

  The page downloads the 129 MB model into your browser and runs it there --
  nothing is sent anywhere. Temperature 0 gives the engine's validated greedy
  output; raise it for varied text. Ctrl-C here stops the server; its request
  log is in web/build/serve.log.

EOF
exec web/serve.sh "$port" >web/build/serve.log 2>&1
