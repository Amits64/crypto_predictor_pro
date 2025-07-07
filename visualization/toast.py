import streamlit.components.v1 as components

def inject_toast_js():
    toast_script = """
    <script>
    function showToast(msg) {
      let toast = document.createElement('div');
      toast.innerHTML = "🔔 " + msg;
      toast.style.position = 'fixed';
      toast.style.bottom = '20px';
      toast.style.right = '20px';
      toast.style.background = '#333';
      toast.style.color = '#fff';
      toast.style.padding = '12px 18px';
      toast.style.borderRadius = '8px';
      toast.style.zIndex = 10000;
      toast.style.fontWeight = 'bold';
      toast.style.boxShadow = '0px 0px 10px #000';
      document.body.appendChild(toast);
      setTimeout(() => { toast.remove(); }, 6000);
    }
    </script>
    """
    components.html(toast_script, height=0)

def show_toast(message: str):
    script = f"<script>showToast('{message}')</script>"
    components.html(script, height=0)
