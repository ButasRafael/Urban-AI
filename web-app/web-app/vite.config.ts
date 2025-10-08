import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'firebase-messaging-sw.ts',
      injectManifest: {
        globPatterns: [], // Don't precache anything, just use for Firebase messaging
        injectionPoint: undefined // Use default self.__WB_MANIFEST
      },
      devOptions: {
        enabled: true, // Enable service worker in development
        type: 'module'
      }
    })
  ],
  server: { port: 5173 }
})
