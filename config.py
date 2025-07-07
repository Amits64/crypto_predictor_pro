import os

# ─────────────────────────────────────────────────────
# Streamlit page configuration
# ─────────────────────────────────────────────────────
PAGE_TITLE = "🚀 Ultra-Enhanced Crypto Predictor Pro"
PAGE_LAYOUT = "wide"

# ─────────────────────────────────────────────────────
# Environment config
# ─────────────────────────────────────────────────────
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────────────
# App-wide CSS for enterprise-style UI
# ─────────────────────────────────────────────────────
APP_CSS = """
<style>
  body {
    background: #f5f7fa;
    font-family: 'Segoe UI', sans-serif;
    zoom: 0.98;
  }

  .app-header {
    background: linear-gradient(135deg, #007cf0, #00dfd8);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
  }

  .stButton > button {
    background-color: #007cf0;
    color: white;
    font-weight: 600;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    transition: background 0.3s ease;
  }

  .stButton > button:hover {
    background-color: #005bb5;
  }

  .stSlider > div {
    color: #007cf0;
  }

  .stMetric {
    border: 1px solid #e6e6e6;
    background: #fff;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  }

  .footer-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
    font-size: 0.8em;
  }

  .prediction-box {
    background: linear-gradient(135deg, #43e97b, #38f9d7);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 18px;
    text-align: center;
  }

  .training-message {
    font-size: 16px;
    font-weight: 500;
    color: #ff4081;
    margin-top: 10px;
  }
</style>
"""
