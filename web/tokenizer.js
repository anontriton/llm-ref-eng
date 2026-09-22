// GPT-2's byte-level BPE, hand-written, for the browser.
//
// The engine takes token ids and never text -- a decision from Phase 1 -- so
// the demo needs this, and it is the one part of Phase 5 with no oracle behind
// it. What it is held to instead is web/test_tokenizer.mjs: exact agreement
// with HF's tokenizer on a committed fixture of edge cases, with the Python
// side's ids for every committed prompt, and a decode/re-encode round trip over
// the 24,576 committed corpus tokens.
//
// No dependencies and no Node or DOM APIs, so the same file runs in Node for
// the test and in the page. Data comes from the checkpoint's own vocab.json
// and merges.txt, passed in as a parsed object and a string.
//
// The algorithm, in the order encode() runs it:
//   0. Split out the special token <|endoftext|>, which HF recognises when it
//      appears in text: it becomes id 50256, and the text on either side is
//      encoded as if the other side were not there. Whitespace next to it is
//      kept, not stripped -- the fixture pins that.
//   1. Split the text into pre-tokens with GPT-2's regex. BPE never merges
//      across these boundaries.
//   2. Encode each pre-token as UTF-8 and map every byte to a printable
//      character (bytesToUnicode), so the vocabulary never has to contain a
//      raw control byte or a space.
//   3. Merge adjacent symbols, lowest merge rank first, until no ranked pair
//      remains.
//   4. Look each resulting symbol up in the vocabulary.
// decode() inverts 4 and 2, and UTF-8-decodes the bytes.

// GPT-2's pre-tokenizer pattern, from the original encoder.py. `\s` there is
// Unicode White_Space, which JavaScript's `\s` is not quite: JS adds U+FEFF and
// leaves out U+0085. \p{White_Space} is the property itself; the fixture holds
// both characters to keep it that way.
const PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\p{White_Space}\p{L}\p{N}]+|\p{White_Space}+(?!\P{White_Space})|\p{White_Space}+/gu;

// GPT-2's only special token, and the end-of-text id generation stops at.
export const ENDOFTEXT = "<|endoftext|>";

// The byte <-> character table from GPT-2's encoder.py. Printable Latin-1
// bytes stand for themselves; the other 68 are shifted up past U+0100 in
// order. A space is byte 32, which becomes U+0120 -- the "Ġ" all over vocab.json.
function bytesToUnicode() {
  const bs = [];
  for (let b = 0x21; b <= 0x7e; ++b) bs.push(b);
  for (let b = 0xa1; b <= 0xac; ++b) bs.push(b);
  for (let b = 0xae; b <= 0xff; ++b) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; ++b) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      ++n;
    }
  }
  const byteToChar = new Array(256);
  const charToByte = new Map();
  for (let i = 0; i < bs.length; ++i) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

export class Tokenizer {
  // vocab: the parsed vocab.json, token string -> id.
  // merges: merges.txt as text; line i (after the version header) is the
  // merge of rank i.
  constructor(vocab, merges) {
    this.encoder = new Map(Object.entries(vocab));
    this.decoder = new Map();
    for (const [token, id] of this.encoder) this.decoder.set(id, token);

    this.ranks = new Map();
    let rank = 0;
    for (const line of merges.split("\n")) {
      if (line.startsWith("#version") || line.trim() === "") continue;
      const sp = line.indexOf(" ");
      // A pair is keyed by its two symbols joined with a space. No symbol
      // contains a space -- bytesToUnicode maps it away -- so this is unique.
      this.ranks.set(line.slice(0, sp) + " " + line.slice(sp + 1), rank++);
    }

    const { byteToChar, charToByte } = bytesToUnicode();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;
    this.utf8 = new TextEncoder();
    // Pre-tokens repeat constantly in real text; BPE on each is the expensive
    // part. Bounded so a long session cannot grow it without limit.
    this.cache = new Map();
    this.cacheLimit = 1 << 16;
  }

  get vocabSize() {
    return this.encoder.size;
  }

  // Merge one pre-token's symbols, lowest rank first. Returns the symbols.
  bpe(word) {
    const hit = this.cache.get(word);
    if (hit !== undefined) return hit;

    let symbols = Array.from(word);
    while (symbols.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; ++i) {
        const r = this.ranks.get(symbols[i] + " " + symbols[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      // Merge every occurrence of that pair, left to right and without
      // overlap, as the reference does -- not just the first.
      const a = symbols[best];
      const b = symbols[best + 1];
      const merged = [];
      for (let i = 0; i < symbols.length; ) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          merged.push(a + b);
          i += 2;
        } else {
          merged.push(symbols[i]);
          i += 1;
        }
      }
      symbols = merged;
    }

    if (this.cache.size >= this.cacheLimit) this.cache.clear();
    this.cache.set(word, symbols);
    return symbols;
  }

  encode(text) {
    const ids = [];
    const parts = text.split(ENDOFTEXT);
    for (let i = 0; i < parts.length; ++i) {
      if (i > 0) ids.push(this.encoder.get(ENDOFTEXT));
      this.encodeOrdinary(parts[i], ids);
    }
    return ids;
  }

  // Text with no special tokens in it, appended to `ids`.
  encodeOrdinary(text, ids) {
    for (const match of text.matchAll(PATTERN)) {
      let word = "";
      for (const byte of this.utf8.encode(match[0])) word += this.byteToChar[byte];
      for (const symbol of this.bpe(word)) {
        const id = this.encoder.get(symbol);
        // Unreachable with GPT-2's own files: every byte is a symbol in the
        // vocabulary and every merge result is too. Mismatched files are the
        // way to get here, and they should fail rather than drop text.
        if (id === undefined) throw new Error(`tokenizer: no id for symbol ${JSON.stringify(symbol)}`);
        ids.push(id);
      }
    }
  }

  // The raw bytes a list of ids stands for.
  bytes(ids) {
    const out = [];
    for (const id of ids) {
      const token = this.decoder.get(id);
      if (token === undefined) throw new Error(`tokenizer: id ${id} is not in the vocabulary`);
      for (const ch of token) out.push(this.charToByte.get(ch));
    }
    return new Uint8Array(out);
  }

  // Invalid UTF-8 -- which a list of ids can easily be, since a character can
  // span tokens -- becomes U+FFFD, as it does in HF's decode.
  decode(ids) {
    return new TextDecoder("utf-8").decode(this.bytes(ids));
  }

  // For generation, one token at a time. A character split across tokens is
  // held back until its last byte arrives, rather than shown as U+FFFD and
  // then never corrected. flush() emits whatever is left, replaced if invalid.
  streamDecoder() {
    const td = new TextDecoder("utf-8");
    return {
      push: (id) => td.decode(this.bytes([id]), { stream: true }),
      flush: () => td.decode(),
    };
  }
}
