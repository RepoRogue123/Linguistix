import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Builds into the Flask static tree so the whole app ships as one container and
// one origin. That also means microphone access rides on the same HTTPS origin
// as the API, with no CORS and no cross-origin latency per prediction.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../ml_website/static/dist',
    emptyOutDir: true,
    // onnxruntime-web ships large wasm binaries; keep them out of the JS bundle
    // and let the browser fetch them on demand.
    assetsInlineLimit: 4096,
    rollupOptions: {
      output: {
        manualChunks: {
          onnx: ['onnxruntime-web'],
          react: ['react', 'react-dom', 'react-router-dom'],
          // three is the single largest dependency. Splitting it out means the
          // routes that use 3D share one cached copy rather than each carrying
          // its own, and routes that do not never download it.
          three: ['three', '@react-three/fiber', '@react-three/drei'],
          motion: ['framer-motion'],
        },
      },
    },
  },
  // onnxruntime-web needs these headers for threaded wasm. Harmless otherwise.
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:7860', changeOrigin: true },
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
