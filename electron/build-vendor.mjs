// Bundles browser libraries into IIFE files for the renderer (CSP 'self').
// Run: npm run vendor
import { build } from "esbuild";
import { mkdirSync } from "fs";

mkdirSync("renderer/vendor", { recursive: true });

await build({
  entryPoints: ["vendor-src/markdown.js"],
  bundle: true,
  minify: true,
  format: "iife",
  outfile: "renderer/vendor/markdown-it.min.js",
});

await build({
  entryPoints: ["vendor-src/highlight.js"],
  bundle: true,
  minify: true,
  format: "iife",
  outfile: "renderer/vendor/highlight.min.js",
});

await build({
  entryPoints: ["vendor-src/purify.js"],
  bundle: true,
  minify: true,
  format: "iife",
  outfile: "renderer/vendor/purify.min.js",
});

console.log("vendor bundles written to renderer/vendor/");
