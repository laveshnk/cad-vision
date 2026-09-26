import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: {
    port: 5173,
    open: false,
    // The voice backend (npm run dev:server) holds the API keys; proxying it
    // keeps the browser on one origin, so no CORS and no mixed-content rules.
    proxy: {
      '/api': { target: 'http://localhost:8787', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8787', ws: true },
    },
  },
  build: {
    target: 'es2020',
    sourcemap: true,
  },
});
