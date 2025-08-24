import os
import logging

def _login_if_token_exists():
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        logging.getLogger(__name__).info("[HF] token not set; proceeding without authentication.")
        return
    try:
        from huggingface_hub import login, set_access_token
        # login() はトークンを keyring/キャッシュに保存。失敗しても起動は続行。
        login(token=token, add_to_git_credential=True)
        set_access_token(token)  # 明示的にプロセスにも設定
        logging.getLogger(__name__).info("[HF] authentication succeeded.")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[HF] authentication failed: {e}. Continuing without token.")

_login_if_token_exists()
