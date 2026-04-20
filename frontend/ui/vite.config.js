// frontend/ui/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// VITE_API_URL is set to http://backend:8000 in docker-compose
// Falls back to http://localhost:8000 for local dev without Docker
const API_URL = process.env.VITE_API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',   // required for Docker — expose outside container
    port: 5173,
    proxy: {
      '/api': {
        target:       API_URL,
        changeOrigin: true,
        rewrite:      path => path.replace(/^\/api/, ''),
      },
      '/stream': {
        target:       API_URL,
        changeOrigin: true,
      },
    },
  },
})