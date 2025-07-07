import os

# Streamlit page config
PAGE_TITLE = "🚀 Ultra-Enhanced Crypto Predictor Pro"
PAGE_LAYOUT = "wide"

# Environment variables
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CSS styling
APP_CSS = """
<style>
  body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: "Segoe UI", sans-serif;
    zoom: 0.98;
  }
  .app-header {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    color: white; padding: 15px; border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 10px;
  }
  .metric-card {
    background: white; padding: 10px; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 5px;
    font-size: 0.9em;
  }
  .prediction-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white; padding: 15px; border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 10px;
  }
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 10px; margin-bottom: 10px;
  }
  .footer-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px; font-size: 0.8em;
  }
</style>
"""
