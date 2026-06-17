/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0f172a',
          800: '#1e293b',
          700: '#334155',
        },
        neon: {
          blue: '#38bdf8',
          green: '#4ade80',
          red: '#f87171'
        }
      }
    },
  },
  plugins: [],
}
