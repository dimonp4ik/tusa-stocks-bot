"""
Encrypted storage helpers for user OKX API credentials.

Secrets are encrypted with Fernet (AES-128-CBC + HMAC) before hitting the DB.
The encryption key lives ONLY in the AUTOTRADE_ENC_KEY env var — a leaked
signals.db alone is useless without it.

Generate a key once:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
and set it as AUTOTRADE_ENC_KEY on the host (Railway → Variables).
"""
import os
import logging

_log = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken
    _CRYPTO_OK = True
except ImportError:          # cryptography not installed
    Fernet, InvalidToken = None, Exception
    _CRYPTO_OK = False

_fernet = None


def _get_fernet():
    global _fernet
    if _fernet is not None:
        return _fernet
    if not _CRYPTO_OK:
        raise RuntimeError("cryptography package not installed — add it to requirements.txt")
    key = os.getenv("AUTOTRADE_ENC_KEY", "").strip()
    if not key:
        raise RuntimeError("AUTOTRADE_ENC_KEY env var not set — autotrade key storage disabled")
    _fernet = Fernet(key.encode())
    return _fernet


def keystore_ready() -> bool:
    """True when encryption is configured (lib installed + key set)."""
    try:
        _get_fernet()
        return True
    except Exception as e:
        _log.warning(f"keystore not ready: {e}")
        return False


def encrypt_secret(plain: str) -> str:
    """Encrypt a secret string → base64 token for DB storage."""
    return _get_fernet().encrypt(plain.encode()).decode()


def decrypt_secret(token: str) -> str:
    """Decrypt a DB token back to the secret string."""
    return _get_fernet().decrypt(token.encode()).decode()
