// Hold the browser build to the command-line engine.
//
//     web/build.sh wasm_simd128 && node web/test_web.mjs
//
// gpt2_web.mjs is what the demo page runs; gpt2_generate.js is the tool
// oracle/check_greedy.py already holds to the PyTorch reference. Same engine
// underneath, so greedy decoding through web/engine.js -- weights streamed in
// chunks the way a fetch body delivers them, the prompt prefilled, then one
// token at a time through the KV cache -- must produce exactly gpt2_generate's
// ids. Checked on the fp32 file, where that chains back to the oracle, and on
// the int8-wte-o8 file the page actually downloads.
//
// Also: seeded sampling repeats itself, and running off the end of the
// context comes back as an error rather than a crash.
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { Engine } from "./engine.js";
import { argmax, rng, sample } from "./sampling.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BUILD = join(ROOT, "web/build/engine-wasm_simd128/tools");
const STEPS = 50;
const RUNS = ["capital", "code", "newline"];

let failures = 0;
const check = (ok, what) => {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) ++failures;
};

const { default: createGpt2 } = await import(join(BUILD, "gpt2_web.mjs"));
const runs = Object.fromEntries(
  JSON.parse(readFileSync(join(ROOT, "oracle/manifest.json"), "utf8"))
    .runs.map((r) => [r.name, r.input_ids]));

for (const file of ["weights/gpt2-124m.bin", "weights/gpt2-124m-int8-wte-o8.bin"]) {
  const path = join(ROOT, file);
  if (!existsSync(path)) {
    check(false, `${file} is missing`);
    continue;
  }
  const engine = await Engine.create(createGpt2);
  const t0 = performance.now();
  // 1 MiB reads, so the copy-in path is exercised many times over, as it is
  // behind a network fetch.
  await engine.loadWeights(statSync(path).size,
                           createReadStream(path, { highWaterMark: 1 << 20 }));
  console.log(`${file}: ${engine.policy}, ${engine.backend}, ` +
              `loaded in ${(performance.now() - t0).toFixed(0)} ms`);

  for (const name of RUNS) {
    engine.reset();
    const got = [];
    let logits = engine.forward(runs[name]);
    for (let s = 0; s < STEPS; ++s) {
      const next = argmax(logits);
      got.push(next);
      if (s + 1 < STEPS) logits = engine.forward([next]);
    }
    const want = JSON.parse(execFileSync("node", [
      join(BUILD, "gpt2_generate.js"), "--weights", path, "--run", name,
      "--steps", String(STEPS), "--kv-cache"], { cwd: ROOT, stdio: ["ignore", "pipe", "ignore"] }))
      .generated_ids;
    const first = got.findIndex((v, i) => v !== want[i]);
    check(first < 0, `${name}: ${STEPS} greedy ids ` +
          (first < 0 ? "identical to gpt2_generate" : `diverge at step ${first}`));
  }

  // Seeded sampling is repeatable, and a different seed goes elsewhere.
  const draw = (seed) => {
    engine.reset();
    const random = rng(seed);
    const out = [];
    let logits = engine.forward(runs.capital);
    for (let s = 0; s < 20; ++s) {
      const next = sample(logits, { temperature: 0.9, topK: 40 }, random);
      out.push(next);
      logits = engine.forward([next]);
    }
    return out.join(",");
  };
  const a = draw(7), b = draw(7), c = draw(8);
  check(a === b && a !== c, "sampling: same seed repeats, another seed differs");

  // One past the context window is an error with a reason, and the engine
  // still works afterwards.
  engine.reset();
  engine.forward(new Array(engine.nCtx).fill(464));
  let message = "";
  try {
    engine.forward([464]);
  } catch (e) {
    message = e.message;
  }
  check(message.length > 0, `past n_ctx: throws "${message}"`);
  engine.reset();
  check(argmax(engine.forward(runs.capital)) > 0, "usable after the error and a reset");
}

console.log(failures === 0 ? "\nWEB ENGINE PASSED" : `\nWEB ENGINE FAILED  (${failures})`);
process.exit(failures === 0 ? 0 : 1);
