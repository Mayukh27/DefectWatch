/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        // Dark base palette — GitHub-style neutral darks
        bg:      { DEFAULT: '#0a0c0e', raised: '#0d1117', card: '#161b22', border: '#21262d', muted: '#30363d' },
        text:    { primary: '#e6edf3', secondary: '#8b949e', muted: '#484f58' },
        // Signal colours — desaturated, professional
        ok:      '#238636',    // green — active / healthy
        defect:  '#da3633',    // red   — defect / alert
        warn:    '#9e6a03',    // amber — warning
        info:    '#388bfd',    // blue  — info / roi border
        // Overlay colours for detection masks (BGR in Python = RGB here reversed)
        mask:    { crack: 'rgba(218,54,51,0.45)', movement: 'rgba(56,139,253,0.35)', general: 'rgba(35,134,54,0.35)' },
      },
      animation: {
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
        'fade-in':   'fadeIn 0.2s ease-out',
        'slide-up':  'slideUp 0.25s ease-out',
      },
      keyframes: {
        pulseDot: {
          '0%,100%': { opacity: '1' },
          '50%':     { opacity: '0.4' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};