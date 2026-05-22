import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Test config lives in vitest.config.ts (kept separate to avoid the
// vite 8 / vitest nested-vite plugin type clash).
export default defineConfig({
  plugins: [react()],
})
