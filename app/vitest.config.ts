import { defineConfig } from 'vitest/config'

// Standalone test config. No vite plugins here (pure-logic tests today);
// add JSX/component handling when we start testing React components.
export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
