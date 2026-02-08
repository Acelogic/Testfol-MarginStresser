"""
Shared disk cache for MD5/pickle caching with TTL expiry.

Provides:
  - disk_cache() decorator for wrapping any function
  - cache_get/cache_set/cache_key primitives for manual caching (e.g. Pydantic models)
"""

import functools
import hashlib
import json
import logging
import os
import pickle
import time

logger = logging.getLogger(__name__)


CACHE_DIR = "data/api_cache"
DEFAULT_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Primitives (for manual caching, e.g. endpoint-level)
# ---------------------------------------------------------------------------

def cache_key(payload_str: str) -> str:
    """Generate MD5 hash from a string payload."""
    return hashlib.md5(payload_str.encode("utf-8")).hexdigest()


def cache_get(key: str, prefix: str = "", ttl: int = DEFAULT_TTL):
    """Try to load a cached result. Returns None on miss."""
    tag = f"{prefix}_" if prefix else ""
    path = os.path.join(CACHE_DIR, f"{tag}{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        if ttl > 0 and (time.time() - os.path.getmtime(path)) > ttl:
            os.remove(path)
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, ValueError, OSError):
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def cache_set(key: str, value, prefix: str = ""):
    """Save a result to disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""
    path = os.path.join(CACHE_DIR, f"{tag}{key}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
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
