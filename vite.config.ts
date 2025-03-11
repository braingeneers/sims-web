import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'
import vueDevTools from 'vite-plugin-vue-devtools'
import { emptyDir } from 'rollup-plugin-empty-dir'

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
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  preview: {
    cors: true,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
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
        {
          src: 'public/*',
          dest: './',
        },
        {
          src: 'public/models/*',
          dest: './models',
        },
      ],
    }),
    emptyDir(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue'],
        },
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]',
      },
      onwarn(warning, warn) {
        if (warning.message.includes('"use client"')) {
          return
        }
        warn(warning)
      },
    },
  },
})
