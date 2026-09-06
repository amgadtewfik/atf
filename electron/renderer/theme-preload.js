// theme-preload.js
// Runs before paint to set data-* attributes on <html> so the renderer
// never flashes the wrong theme. Reads from localStorage, falls back to
// dark mode if anything goes wrong (e.g. localStorage unavailable).

(function () {
  "use strict";
  var fallback = { mode: "dark", accent: "indigo", fontSize: "comfortable", density: "comfortable" };
  var theme;
  try {
    var raw = localStorage.getItem("atf-theme");
    theme = raw ? JSON.parse(raw) : null;
  } catch (e) { theme = null; }
  if (!theme || typeof theme !== "object") theme = fallback;
  if (theme.mode === "auto") {
    theme.mode = (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark";
  }
  if (theme.mode !== "light" && theme.mode !== "dark") theme.mode = "dark";
  var root = document.documentElement;
  root.setAttribute("data-theme", theme.mode);
  root.setAttribute("data-accent", theme.accent || "indigo");
  root.setAttribute("data-density", theme.density || "comfortable");
  root.setAttribute("data-fontsize", theme.fontSize || "comfortable");
})();
