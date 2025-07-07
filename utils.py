import requests

def safe_get(url: str, timeout: int = 15):
    """Wrapper around requests.get with error handling."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
