import requests
import streamlit as st

def safe_get(url: str, timeout: int = 15):
    """Wrapper around requests.get with error handling."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ HTTP Error {resp.status_code}: {e} \nURL: {url}")
        return []
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return []
