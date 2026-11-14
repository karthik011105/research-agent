import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run dev` talks to a running API (local or the deployed one): API_URL=http://<alb-dns> npm run dev
const api = process.env.API_URL || 'http://localhost:8000'
const apiPaths = ['/research', '/result', '/stream', '/session', '/reports', '/monitors', '/stats', '/feedback', '/health', '/diff']

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: Object.fromEntries(apiPaths.map(p => [p, { target: api, changeOrigin: true }])),
    fs: { allow: ['..'] },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 900 },
})
