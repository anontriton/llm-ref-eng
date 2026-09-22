// Hold web/tokenizer.js to the Python side's tokenization.
//
//     node web/test_tokenizer.mjs
//
// Four checks, each against something committed rather than something this
// file computes:
//
//   fixture   web/tokenizer_cases.json -- HF's ids for hand-picked edge cases
//             and a seeded fuzz set, and HF's text for decodes that cut a
//             character in half. scripts/export_tokenizer_cases.py writes it.
//   prompts   the oracle's five prompts (oracle/manifest.json) and the
//             benchmark prompt (bench/prompts.tsv), text and ids both pinned.
//   corpus    eval/corpus.tsv and eval/calib.tsv: 24,576 real WikiText tokens.
//             Their text is not committed, but byte-level BPE is lossless, so
//             decode(ids) is exactly the text they were cut from, and encoding
//             it again must give the same ids.
//   stream    streamDecoder() must produce what decode() does, one id at a
//             time, for every fixture sequence.
//
// Needs the tokenizer files scripts/download_weights.py fetches. Exit 0 pass,
// 1 any mismatch.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { Tokenizer } from "./tokenizer.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => readFileSync(join(ROOT, p), "utf8");

const tok = new Tokenizer(
  JSON.parse(read("weights/gpt2-124m/vocab.json")),
  read("weights/gpt2-124m/merges.txt"),
);

let failures = 0;
const fail = (what) => {
  if (failures < 10) console.log(`  FAIL ${what}`);
  ++failures;
};
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
// A tokenizer that throws on some input has failed on that input, and should
// be reported like any other mismatch -- with the case -- rather than ending
// the run on the first one.
const attempt = (f) => { try { return f(); } catch (e) { return e; } };
const err = (v) => v instanceof Error;
const show = (s) => JSON.stringify(s.length > 60 ? s.slice(0, 60) + "..." : s);

// The TSVs' id column, and the escaping their writers use.
function tsvRows(path) {
  return read(path).split("\n")
    .filter((l) => l && !l.startsWith("#"))
    .map((l) => {
      const [name, field, ids] = l.split("\t");
      const text = field.replace(/\\(.)/g, (_, c) =>
        ({ n: "\n", r: "\r", t: "\t", "\\": "\\" })[c] ?? `\\${c}`);
      return { name, text, ids: ids.split(",").map(Number) };
    });
}

// --- fixture ---------------------------------------------------------------
const cases = JSON.parse(read("web/tokenizer_cases.json"));
let n = 0;
for (const c of cases.encode) {
  const got = attempt(() => tok.encode(c.text));
  if (err(got) || !same(got, c.ids)) fail(`encode ${show(c.text)}: got ${err(got) ? got.message : `[${got}]`} want [${c.ids}]`);
  ++n;
}
for (const c of cases.decode) {
  const got = attempt(() => tok.decode(c.ids));
  if (err(got) || got !== c.text) fail(`decode [${c.ids}]: got ${err(got) ? got.message : show(got)} want ${show(c.text)}`);
  ++n;
}
console.log(`fixture   ${n} cases vs ${cases.reference}`);

// --- prompts ---------------------------------------------------------------
const manifest = JSON.parse(read("oracle/manifest.json"));
const prompts = manifest.runs.map((r) => ({ name: r.name, text: r.prompt, ids: r.input_ids }))
  .concat(tsvRows("bench/prompts.tsv"));
for (const p of prompts) {
  const got = attempt(() => tok.encode(p.text));
  if (err(got) || !same(got, p.ids)) fail(`prompt ${p.name}: got ${err(got) ? got.message : `[${got.slice(0, 8)}...]`} want [${p.ids.slice(0, 8)}...]`);
  if (attempt(() => tok.decode(p.ids)) !== p.text) fail(`prompt ${p.name}: decode does not give the prompt back`);
}
console.log(`prompts   ${prompts.length} pinned prompts: ${prompts.map((p) => p.name).join(", ")}`);

// --- corpus ----------------------------------------------------------------
for (const [tsv, json] of [["eval/corpus.tsv", "eval/corpus.json"], ["eval/calib.tsv", "eval/calib.json"]]) {
  const { ids } = tsvRows(tsv)[0];
  // First make sure these are the pinned ids: the checksum the Python side
  // recorded, over the same little-endian int32 bytes.
  const le = Buffer.from(new Int32Array(ids).buffer);
  const sha = createHash("sha256").update(le).digest("hex");
  if (sha !== JSON.parse(read(json)).ids_sha256) fail(`${tsv}: ids do not match ${json}'s checksum`);

  const t0 = performance.now();
  const text = attempt(() => tok.decode(ids));
  const again = err(text) ? text : attempt(() => tok.encode(text));
  if (err(again)) {
    fail(`${tsv}: ${again.message}`);
    continue;
  }
  const ms = performance.now() - t0;
  // Count agreement token by token and report the first disagreement, so a
  // failure says where rather than just that.
  let first = -1;
  for (let i = 0; i < Math.max(ids.length, again.length); ++i) {
    if (ids[i] !== again[i]) { first = i; break; }
  }
  if (first >= 0) {
    fail(`${tsv}: re-encoding diverges at token ${first} of ${ids.length}: ` +
         `got ${again[first]} want ${ids[first]} near ${show(tok.decode(ids.slice(Math.max(0, first - 3), first + 3)))}`);
  }
  console.log(`corpus    ${tsv}: ${ids.length} ids, ${text.length} chars, ` +
              `round trip ${first < 0 ? "exact" : "DIVERGES"}  (${ms.toFixed(0)} ms)`);
}

// --- stream ----------------------------------------------------------------
let streamed = 0;
for (const c of cases.encode.concat(cases.decode)) {
  const text = attempt(() => {
    const sd = tok.streamDecoder();
    let t = "";
    for (const id of c.ids) t += sd.push(id);
    return t + sd.flush();
  });
  const want = attempt(() => tok.decode(c.ids));
  if (err(text) || err(want) || text !== want) {
    fail(`stream [${c.ids}]: got ${err(text) ? text.message : show(text)} ` +
         `want ${err(want) ? want.message : show(want)}`);
  }
  ++streamed;
}
console.log(`stream    ${streamed} sequences, one id at a time`);

console.log(failures === 0 ? "\nTOKENIZER PASSED" : `\nTOKENIZER FAILED  (${failures} mismatches)`);
process.exit(failures === 0 ? 0 : 1);
