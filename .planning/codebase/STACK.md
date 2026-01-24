# Technology Stack

**Analysis Date:** 2026-01-23

## Languages

**Primary:**
- Python 3.10.12 - Core application logic, backtesting engines, data processing, and Streamlit web UI
- TypeScript 5.9.2 - Mobile application (Expo React Native) for iOS, Android, and Web platforms
- JavaScript - Supporting build tooling and Expo configuration

**Secondary:**
- Markdown - Documentation files

## Runtime

**Environment:**
- Python 3.10.12 for backend and Streamlit application
- Node.js v20.19.5 for Expo React Native mobile app
- Expo CLI for cross-platform mobile development

**Package Manager:**
- pip - Python dependency management
- npm 10.8.2 - JavaScript/Node.js dependency management
- Lockfile: `margin-stresser-mobile/package-lock.json` (present, but uses npm format)

## Frameworks

**Core:**
- Streamlit >=1.45.0 - Web application framework for interactive financial analysis UI (`testfol_charting.py` as main entry point)
- FastAPI - REST API backend for mobile app and external consumers (`app/services/backend.py`)

**Mobile:**
- Expo ~54.0.29 - Cross-platform mobile development framework
- React Native 0.81.5 - Native mobile UI components
- React 19.1.0 - Component framework for web and mobile
- Expo Router ~6.0.19 - Navigation and routing for Expo apps
- React Native Paper 5.14.5 - Material Design UI components
- NativeWind 2.0.11 - Tailwind CSS for React Native

**Testing:**
- pytest - Python unit testing (not explicitly listed in requirements but standard practice)

**Build/Dev:**
- PostCSS 8.5.6 - CSS processing for Tailwind
- TailwindCSS 3.3.2 - Utility-first CSS framework
- Autoprefixer 10.4.23 - CSS vendor prefixing
- ESLint 9.25.0 - JavaScript linting
- ESLint Config Expo ~10.0.0 - Expo-specific linting rules

**Data & Visualization:**
- pandas - Data manipulation and time series analysis
- numpy - Numerical computing
- Plotly - Interactive charting (Plotly Graph Objects and Express)
- matplotlib - Static plotting library
- openpyxl - Excel file reading/writing

## Key Dependencies

**Critical:**
- requests - HTTP client library for API calls (Testfol API, FRED API, yfinance fallback)
- yfinance - Yahoo Finance data fetching, used as fallback for equity prices and as proxy (QBIG) for simulated tickers
- pandas - Core data manipulation for backtesting and calculations
- numpy - Numerical operations for simulation engines

**Infrastructure:**
- @expo/vector-icons ~15.0.3 - Icon library for mobile UI
- @react-native-async-storage/async-storage ~2.2.0 - Local storage for mobile app
- @react-navigation packages (v7.x) - Navigation infrastructure for tabs and stacks
- expo-blur, expo-image, expo-linear-gradient, expo-web-browser - Expo modules for UI enhancements
- react-native-gestures, react-native-reanimated - Touch and animation handling
- react-native-safe-area-context, react-native-screens - Safe area and screen management
- react-native-svg - SVG rendering in React Native
- react-native-gifted-charts - Chart rendering for mobile (alternative to Plotly)

## Configuration

**Environment:**
- TESTFOL_API_KEY - Bearer token for authenticating with testfol.io API (optional for standard users; required for authenticated endpoints)
- Local file-based caching in `data/api_cache/` using pickle format (MD5-keyed cache files)
- No explicit .env file required; environment variables read via `os.environ.get()`

**Build:**
- `tsconfig.json` - TypeScript compiler configuration in `margin-stresser-mobile/`
  - Paths alias: `@/*` maps to root directory
  - Strict mode enabled
- Expo app config in `margin-stresser-mobile/package.json` with entry point: `expo-router/entry`

## Platform Requirements

**Development:**
- macOS, Windows, or Linux for Python environment
- Python 3.10+ installed and in PATH
- Node.js v18+ for Expo development
- Expo Go app (iOS/Android) or iOS/Android simulators for mobile testing
- Optional: Testfol API key for authenticated access

**Production:**
- Streamlit Cloud (testfol-marginstresser.streamlit.app) for web app deployment
- Expo over-the-air updates for mobile app distribution
- Optional local FastAPI server (app/services/backend.py) for REST API access by mobile clients
- Python 3.10+ runtime for local/server deployments

---

*Stack analysis: 2026-01-23*
