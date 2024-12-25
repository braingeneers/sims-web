import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  base: "./", // Relative base path so we can live anywhere
  // base: "/sims-web/", // Relative base path so we can live anywhere
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
