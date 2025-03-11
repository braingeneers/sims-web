import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  base: './', // Relative base path so we can live anywhere
  server: {
    cors: true,
    watch: {
      usePolling: false,
      ignored: ['**/venv/**', '**/node_modules/**', '**/data/**'],
    },
    headers: {
      // Required for the onnxruntime-web to use multiple threads
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
      // "Access-Control-Expose-Headers": "*",kkkk
      // "Access-Control-Allow-Headers": "*",
    },
  },
  plugins: [
    vue(),
    vueJsx(),
    vueDevTools(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'assets',
        },
      ],
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
