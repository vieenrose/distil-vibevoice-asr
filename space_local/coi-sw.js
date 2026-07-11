/* Minimal COOP/COEP service worker: static Spaces cannot set response
 * headers, but SharedArrayBuffer (multi-threaded wasm) requires
 * cross-origin isolation. This SW re-serves every same-scope response with
 * the isolation headers; COEP:credentialless keeps cross-origin model
 * fetches (HF CDN) working without CORP headers on them. */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0 || res.type === "opaque") return res;
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Embedder-Policy", "credentialless");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, {
        status: res.status, statusText: res.statusText, headers: h,
      });
    })
  );
});
