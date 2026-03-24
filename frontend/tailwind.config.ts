import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        retro: {
          bg: '#0a0e1a',
          surface: '#111827',
          card: '#1a2236',
          border: '#2a3550',
          accent: '#3b82f6',
          'accent-bright': '#60a5fa',
          success: '#10b981',
          warning: '#f59e0b',
          danger: '#ef4444',
          text: '#e5e7eb',
          'text-dim': '#9ca3af',
          'speed-glow': '#00ff88',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 5px rgba(59, 130, 246, 0.3)' },
          '50%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.6)' },
        },
      },
    },
  },
  plugins: [],
}
export default config
