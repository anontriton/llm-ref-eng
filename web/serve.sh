#!/usr/bin/env bash
# Serve the demo locally.
#
#     web/serve.sh [port]        # then open http://localhost:8000/
#
# Serves web/build/site/, the static site web/pack.py assembles -- the same
# directory GitHub Pages publishes, so what you try here is what ships. Build it
# first (./demo.sh does all of this):
#
#     web/build.sh wasm_simd128
#     .venv/bin/python scripts/quantize_weights.py --wte-outliers 8
#     python3 web/pack.py
#
# Bound to localhost.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
site="$root/web/build/site"
port="${1:-8000}"

if [[ ! -f "$site/model/weights.json" ]]; then
  echo "no site at web/build/site -- run python3 web/pack.py (see the header of $0)" >&2
  exit 1
fi

echo "http://localhost:$port/"
# http.server, plus the two headers that make the page cross-origin isolated
# -- what the threaded engine needs, and what web/coi-sw.js supplies on GitHub
# Pages, which cannot send them.
exec python3 - "$port" "$site" <<'PY'
import functools, http.server, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

port, site = int(sys.argv[1]), sys.argv[2]
http.server.ThreadingHTTPServer(
    ("127.0.0.1", port), functools.partial(Handler, directory=site)).serve_forever()
PY
