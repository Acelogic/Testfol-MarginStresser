"""Testfol.io Supabase authentication — auto-login and token refresh."""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = "https://rfqvszcghajvopjoabqt.supabase.co"
# Public anon key (safe to embed — it's in the testfol.io frontend JS bundle)
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJmcXZzemNnaGFqdm9wam9hYnF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5Mjk1MzksImV4cCI6MjA4MTUwNTUzOX0.2cMl7sVDSPjkyMouK_jkTdqNR6wmpRUUH0R5xZdG8d0"

_token_cache: dict = {}


def get_user_email() -> str | None:
    """Extract email from the cached access token JWT (no network call)."""
    token = _token_cache.get("access_token")
    if not token:
        return None
    try:
        import base64, json as _json
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        data = _json.loads(base64.b64decode(payload))
        return data.get("email")
    except Exception:
        return None


def _login(email: str, password: str) -> dict:
    """Sign in with email/password via Supabase Auth REST API."""
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        json={"email": email, "password": password},
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _refresh(refresh_token: str) -> dict:
    """Refresh an expired access token."""
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
        json={"refresh_token": refresh_token},
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def login_with_credentials(email: str, password: str) -> str:
    """
    Login with explicit credentials (from UI). Returns access token.
    Raises on failure.
    """
    global _token_cache
    data = _login(email, password)
    _token_cache["access_token"] = data["access_token"]
    _token_cache["refresh_token"] = data["refresh_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    logger.info("Testfol login successful (UI credentials)")
    return _token_cache["access_token"]


def get_token() -> str | None:
    """
    Return a valid Testfol bearer token.

    Priority:
    1. Cached token (if not expired)
    2. Refresh via stored refresh_token
    3. Fresh login via TESTFOL_EMAIL + TESTFOL_PASSWORD env vars
    4. Static TESTFOL_API_KEY env var (manual fallback)
    """
    global _token_cache

    # Check cached token (refresh 5 min before expiry)
    if _token_cache.get("access_token") and _token_cache.get("expires_at", 0) > time.time() + 300:
        return _token_cache["access_token"]

    # Try refresh
    if _token_cache.get("refresh_token"):
        try:
            data = _refresh(_token_cache["refresh_token"])
            _token_cache["access_token"] = data["access_token"]
            _token_cache["refresh_token"] = data["refresh_token"]
            _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
            logger.info("Testfol token refreshed")
            return _token_cache["access_token"]
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            _token_cache.clear()

    # Try login via env vars
    email = os.environ.get("TESTFOL_EMAIL")
    password = os.environ.get("TESTFOL_PASSWORD")
    if email and password:
        try:
            return login_with_credentials(email, password)
        except Exception as e:
            logger.error(f"Testfol login failed: {e}")

    # Fall back to static key
    return os.environ.get("TESTFOL_API_KEY")
