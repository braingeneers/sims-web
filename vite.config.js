import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  base: "./", // Relative base path so we can live anywhere
  // base: "/sims-web/", // Relative base path so we can live anywhere
  server: {
    cors: true,
    watch: {
      usePolling: false,
      ignored: ["**/venv/**", "**/node_modules/**", "**/data/**"],
    },
    headers: {
      // Required for the onnxruntime-web to use multiple threads
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
      // "Access-Control-Expose-Headers": "*",kkkk
      // "Access-Control-Allow-Headers": "*",
    },
  },
  preview: {
    cors: true,
    headers: {
      // Required for the onnxruntime-web to use multiple threads
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
      // "Access-Control-Expose-Headers": "*",
      // "Access-Control-Allow-Headers": "*",
    },
  },
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "assets",
        },
      ],
    }),
  ],
});
