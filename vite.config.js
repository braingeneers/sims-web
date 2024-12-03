import { resolve } from "path";

export default {
  root: resolve(__dirname, "src"),
  resolve: {
    alias: {
      "~bootstrap": resolve(__dirname, "node_modules/bootstrap"),
    },
  },
  build: {
    target: "esnext",
    outDir: "../dist",
    emptyOutDir: false,
  },
  server: {
    port: 8080,
    hot: true,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
};
