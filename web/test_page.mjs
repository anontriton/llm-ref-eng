// The demo page itself, in headless Chrome -- the last check before a deploy.
//
//     python3 web/pack.py && node web/test_page.mjs
//
// web/test_web.mjs proves the wasm module; this proves the page built from it,
// served exactly as GitHub Pages will serve web/build/site/:
//
//   load      it downloads the parts, checks the sha256, and says Ready with
//             the verified checksum on screen
//   greedy    50 greedy tokens on "The capital of France is" put on screen
//             exactly the text gpt2_generate.js's ids decode to
//   stop      Stop ends a long sampling run
//   cache     a reload takes the weights from the browser's cache
//   tamper    one flipped byte in one part: the page refuses the file and
//             never becomes ready
//   mobile    a phone asks before downloading anything
//   console   no errors, other than the tamper run's deliberate one
//
// No dependencies: a small static server in Node, and the DevTools protocol
// over Node's built-in WebSocket. Needs Chrome or Chromium; CHROME=/path
// overrides the search. Exit 0 pass, 1 any failure.
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

import { Tokenizer } from "./tokenizer.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SITE = join(ROOT, "web/build/site");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let failures = 0;
const check = (ok, what) => {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) ++failures;
};

if (!existsSync(join(SITE, "model/weights.json"))) {
  console.error("no site at web/build/site -- run python3 web/pack.py first");
  process.exit(1);
}

// --- a static server, with a switch to corrupt one part --------------------
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
                ".wasm": "application/wasm", ".json": "application/json", ".txt": "text/plain",
                ".bin": "application/octet-stream" };
let tamper = false;
const server = createServer((req, res) => {
  let path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (path.endsWith("/")) path += "index.html";
  const file = join(SITE, path);
  if (!file.startsWith(SITE) || !existsSync(file) || statSync(file).isDirectory()) {
    res.writeHead(404).end();
    return;
  }
  let body = readFileSync(file);
  if (tamper && path.endsWith("weights-001.bin")) {
    body = Buffer.from(body);
    body[12345] ^= 0x01;
  }
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream",
                       "Content-Length": body.length });
  res.end(body);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const URL_ = `http://127.0.0.1:${server.address().port}/`;

// --- Chrome over the DevTools protocol --------------------------------------
function findChrome() {
  const names = [process.env.CHROME, "google-chrome-stable", "google-chrome", "chromium",
                 "chromium-browser",
                 "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
  for (const n of names) {
    if (n.startsWith("/") && existsSync(n)) return n;
    try {
      return execFileSync("sh", ["-c", `command -v "${n}"`], { encoding: "utf8" }).trim();
    } catch { /* next */ }
  }
  console.error("no Chrome or Chromium found; set CHROME=/path/to/chrome");
  process.exit(1);
}
const CHROME = findChrome();

async function browser() {
  const profile = mkdtempSync(join(tmpdir(), "gpt2-page-"));
  const port = 9300 + Math.floor(Math.random() * 500);
  const args = ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
                "--no-first-run", "--no-default-browser-check", "about:blank"];
  if (process.env.CI) args.unshift("--no-sandbox");  // CI runners lack the sandbox's namespaces
  const proc = spawn(CHROME, args, { stdio: "ignore" });
  let targets;
  for (let i = 0; i < 100 && !targets; ++i) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { await sleep(100); }
  }
  const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  const errors = [];
  ws.onmessage = ({ data }) => {
    const m = JSON.parse(data);
    if (pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === "Runtime.exceptionThrown") errors.push(m.params.exceptionDetails.text);
    if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
      errors.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
    }
    if (m.method === "Log.entryAdded" && m.params.entry.level === "error") errors.push(m.params.entry.text);
  };
  const send = (method, params = {}) => new Promise((r) => {
    pending.set(++id, r);
    ws.send(JSON.stringify({ id, method, params }));
  });
  await send("Runtime.enable");
  await send("Log.enable");
  await send("Page.enable");
  const js = async (expr) => (await send("Runtime.evaluate",
    { expression: expr, returnByValue: true, awaitPromise: true })).result.value;
  const statusText = () => js(`document.getElementById("status-msg").textContent`);
  const waitStatus = async (re, seconds) => {
    for (let i = 0; i < seconds * 4; ++i) {
      const s = await statusText();
      if (re.test(s)) return s;
      await sleep(250);
    }
    return await statusText();
  };
  // Close Chrome through the protocol, which shuts down its helper processes
  // too -- killing the main one leaves them writing to the profile. Removing
  // the profile is best-effort: a leftover temp directory is not a failure.
  const exited = new Promise((r) => proc.once("exit", r));
  const close = async () => {
    try {
      const { webSocketDebuggerUrl } = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
      const bws = new WebSocket(webSocketDebuggerUrl);
      await new Promise((r) => (bws.onopen = r));
      bws.send(JSON.stringify({ id: 1, method: "Browser.close" }));
    } catch {
      proc.kill();
    }
    ws.close();
    await Promise.race([exited, sleep(5000).then(() => proc.kill("SIGKILL"))]);
    try {
      rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
    } catch {
      console.log(`  (could not remove ${profile})`);
    }
  };
  return { send, js, statusText, waitStatus, errors, close };
}

const manifest = JSON.parse(readFileSync(join(SITE, "model/weights.json"), "utf8"));
console.log(`site ${URL_}, weights ${manifest.policy} sha256 ${manifest.sha256.slice(0, 16)}...`);

// --- load, greedy, stop, cache ----------------------------------------------
{
  const b = await browser();
  await b.send("Page.navigate", { url: URL_ });
  const ready = await b.waitStatus(/^Ready|error|match/i, 120);
  check(/^Ready/.test(ready), `load: ${ready}`);
  const verified = await b.js(`document.getElementById("verified").title`);
  check(verified === manifest.sha256, "load: the verified sha256 on screen is the manifest's");

  await b.js(`(() => {
    document.getElementById("prompt").value = "The capital of France is";
    document.getElementById("temp").value = "0";
    document.getElementById("max").value = "50";
    document.getElementById("go").click();
  })()`);
  await b.waitStatus(/^Done|error/i, 120);
  const shown = await b.js(`document.getElementById("output").textContent`);
  const gen = JSON.parse(execFileSync("node", [
    join(ROOT, "web/build/engine-wasm_simd128/tools/gpt2_generate.js"),
    "--weights", join(ROOT, "weights/gpt2-124m-int8-wte-o8.bin"),
    "--run", "capital", "--steps", "50", "--kv-cache"],
    { cwd: ROOT, stdio: ["ignore", "pipe", "ignore"] }));
  const tok = new Tokenizer(JSON.parse(readFileSync(join(SITE, "model/vocab.json"), "utf8")),
                            readFileSync(join(SITE, "model/merges.txt"), "utf8"));
  const want = tok.decode(gen.prompt_ids.concat(gen.generated_ids));
  check(shown === want, "greedy: the page shows exactly gpt2_generate's 50 tokens");

  await b.js(`(() => {
    document.getElementById("temp").value = "0.8";
    document.getElementById("max").value = "500";
    document.getElementById("go").click();
  })()`);
  await sleep(1500);
  await b.js(`document.getElementById("stop").click()`);
  await b.waitStatus(/^Done|error/i, 20);
  const end = await b.js(`document.querySelector(".output .end")?.textContent ?? ""`);
  check(end === "[stopped]", `stop: the run ends ${end || "(no marker)"}`);

  await b.send("Page.reload");
  const again = await b.waitStatus(/^Ready|error/i, 60);
  check(/from cache/.test(again), `cache: ${again}`);
  check(b.errors.length === 0, `console: ${b.errors.length ? b.errors.join(" | ") : "no errors"}`);
  await b.close();
}

// --- tamper: a fresh profile, so nothing is cached --------------------------
{
  tamper = true;
  const b = await browser();
  await b.send("Page.navigate", { url: URL_ });
  const s = await b.waitStatus(/^Ready|checksum/i, 120);
  const goDisabled = await b.js(`document.getElementById("go").disabled`);
  check(/checksum/.test(s) && goDisabled, `tamper: ${s}`);
  await b.close();
  tamper = false;
}

// --- mobile: asks before downloading ----------------------------------------
{
  const b = await browser();
  await b.send("Emulation.setUserAgentOverride", {
    userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148" });
  await b.send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await b.send("Page.navigate", { url: URL_ });
  await sleep(2000);
  const button = await b.js(`!document.getElementById("load").hidden`);
  const s = await b.statusText();
  const overflow = await b.js(`document.documentElement.scrollWidth > window.innerWidth`);
  check(button && !/Downloading|Ready/.test(s), "mobile: asks before downloading");
  check(!overflow, "mobile: no horizontal overflow at 390 px");
  await b.close();
}

server.close();
console.log(failures === 0 ? "\nPAGE PASSED" : `\nPAGE FAILED  (${failures})`);
process.exit(failures === 0 ? 0 : 1);
