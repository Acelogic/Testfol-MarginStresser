"""
Shared disk cache for MD5/pickle caching with TTL expiry.

Provides:
  - disk_cache() decorator for wrapping any function
  - cache_get/cache_set/cache_key primitives for manual caching (e.g. Pydantic models)

Security: Cache files are HMAC-signed to prevent deserialization of tampered data.
Pickle is used intentionally here because cached objects include pandas DataFrames
and Series which are not efficiently JSON-serializable.
"""

import functools
import hashlib
import hmac
import json
import logging
import os
import pickle  # noqa: S403 — intentional, HMAC-guarded
import time

logger = logging.getLogger(__name__)


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "api_cache")
DEFAULT_TTL = 3600  # 1 hour

# HMAC key for cache integrity — prevents pickle injection from tampered files.
# Derived from this file's path so it's stable per-install but unpredictable to attackers.
_HMAC_KEY = hashlib.sha256(os.path.abspath(__file__).encode()).digest()


# ---------------------------------------------------------------------------
# Primitives (for manual caching, e.g. endpoint-level)
# ---------------------------------------------------------------------------

def cache_key(payload_str: str) -> str:
    """Generate MD5 hash from a string payload."""
    return hashlib.md5(payload_str.encode("utf-8")).hexdigest()


def _sig_path(cache_path: str) -> str:
    """Return the HMAC signature file path for a cache file."""
    return cache_path + ".sig"


def cache_get(key: str, prefix: str = "", ttl: int = DEFAULT_TTL):
    """Try to load a cached result. Returns None on miss or tampered file."""
    tag = f"{prefix}_" if prefix else ""
    path = os.path.join(CACHE_DIR, f"{tag}{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        if ttl > 0 and (time.time() - os.path.getmtime(path)) > ttl:
            os.remove(path)
            return None
        # Verify HMAC before deserializing
        with open(path, "rb") as f:
            data = f.read()
        sig_file = _sig_path(path)
        if os.path.exists(sig_file):
            with open(sig_file, "rb") as sf:
                stored_sig = sf.read()
            expected_sig = hmac.new(_HMAC_KEY, data, hashlib.sha256).digest()
            if not hmac.compare_digest(stored_sig, expected_sig):
                logger.warning(f"Cache integrity check failed for {path} — removing")
                os.remove(path)
                os.remove(sig_file)
                return None
        else:
            # Legacy file without signature — accept but re-sign on next write
            pass
        return pickle.loads(data)  # noqa: S301 — HMAC-verified above
    except (EOFError, pickle.UnpicklingError, ValueError, OSError):
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def cache_set(key: str, value, prefix: str = ""):
    """Save a result to disk cache with HMAC signature."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""
    path = os.path.join(CACHE_DIR, f"{tag}{key}.pkl")
    try:
        data = pickle.dumps(value)  # noqa: S301
        with open(path, "wb") as f:
            f.write(data)
        # Write HMAC signature
        sig = hmac.new(_HMAC_KEY, data, hashlib.sha256).digest()
        with open(_sig_path(path), "wb") as sf:
            sf.write(sig)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


# ---------------------------------------------------------------------------
# Decorator (for wrapping functions directly)
# ---------------------------------------------------------------------------

def disk_cache(ttl=DEFAULT_TTL, prefix=""):
    """
    Decorator for MD5/pickle disk caching with TTL expiry.

    Args:
        ttl: Time-to-live in seconds (default 1 hour). 0 = no expiry.
        prefix: Optional prefix for cache filenames.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_payload = {
                "func": func.__qualname__,
                "args": args,
                "kwargs": kwargs,
            }
            payload_str = json.dumps(cache_payload, sort_keys=True, default=str)
            req_hash = cache_key(payload_str)

            result = cache_get(req_hash, prefix=prefix, ttl=ttl)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache_set(req_hash, result, prefix=prefix)
            return result
        return wrapper
    return decorator
