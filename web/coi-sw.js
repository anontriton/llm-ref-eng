// Cross-origin isolation, for a host that cannot send headers.
//
// The threaded engine needs SharedArrayBuffer, and a browser hands that only
// to a page that is cross-origin isolated: served with
//
//     Cross-Origin-Opener-Policy: same-origin
//     Cross-Origin-Embedder-Policy: require-corp
//
// GitHub Pages sends neither and offers no way to add them. A service worker
// sits between the page and the network, so it can: it re-serves every
// same-origin response with the two headers added. The page registers this on
// its first visit and reloads once, and from then on is isolated -- see
// isolate() in index.html, which also falls back to the single-threaded engine
// if any of this is refused. web/serve.sh sends the real headers, so locally
// this never has anything to do.
//
// Everything the page loads is same-origin -- the page, its modules, the wasm,
// the weights -- so require-corp costs nothing here. Cross-origin requests are
// left alone entirely.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return;
  // A DevTools quirk: this combination throws if fetched from here.
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;

  event.respondWith((async () => {
    const response = await fetch(request);
    if (response.status === 0) return response;  // opaque: nothing to add to
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    // The body is passed through as a stream, not buffered: the weights are
    // 32 MB a part, and the page's progress bar and retry logic both depend
    // on seeing them arrive.
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  })());
});
