"""Tests for app.common.cache module."""

import os
import time

import pytest

from app.common.cache import cache_key, cache_get, cache_set, disk_cache, CACHE_DIR


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

def test_cache_key_deterministic():
    """Same input produces the same hash."""
    assert cache_key("hello") == cache_key("hello")


def test_cache_key_unique():
    """Different inputs produce different hashes."""
    assert cache_key("hello") != cache_key("world")


# ---------------------------------------------------------------------------
# cache_get / cache_set round-trip
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path, monkeypatch):
    """cache_set followed by cache_get returns the same value."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))
    cache_set("test_key", {"a": 1, "b": [2, 3]})
    result = cache_get("test_key", ttl=0)
    assert result == {"a": 1, "b": [2, 3]}


def test_cache_miss(tmp_path, monkeypatch):
    """cache_get returns None when key doesn't exist."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))
    assert cache_get("nonexistent_key") is None


def test_cache_ttl_expiry(tmp_path, monkeypatch):
    """Expired entries return None and are removed."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))
    cache_set("ttl_key", "value")

    # Backdate the file modification time
    path = os.path.join(str(tmp_path), "ttl_key.pkl")
    old_time = time.time() - 7200  # 2 hours ago
    os.utime(path, (old_time, old_time))

    result = cache_get("ttl_key", ttl=3600)
    assert result is None
    assert not os.path.exists(path)


def test_cache_ttl_zero_no_expiry(tmp_path, monkeypatch):
    """TTL=0 means no expiry."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))
    cache_set("forever_key", "value")

    # Backdate heavily
    path = os.path.join(str(tmp_path), "forever_key.pkl")
    old_time = time.time() - 999999
    os.utime(path, (old_time, old_time))

    assert cache_get("forever_key", ttl=0) == "value"


def test_cache_corrupt_file(tmp_path, monkeypatch):
    """Corrupt pickle files return None and are cleaned up."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))
    path = os.path.join(str(tmp_path), "corrupt_key.pkl")
    with open(path, "wb") as f:
        f.write(b"not-a-pickle")

    result = cache_get("corrupt_key", ttl=0)
    assert result is None
    assert not os.path.exists(path)


def test_cache_prefix(tmp_path, monkeypatch):
    """Prefix creates separate cache namespace."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))
    cache_set("key1", "val_a", prefix="ns1")
    cache_set("key1", "val_b", prefix="ns2")

    assert cache_get("key1", prefix="ns1", ttl=0) == "val_a"
    assert cache_get("key1", prefix="ns2", ttl=0) == "val_b"


# ---------------------------------------------------------------------------
# disk_cache decorator
# ---------------------------------------------------------------------------

def test_disk_cache_decorator(tmp_path, monkeypatch):
    """Decorated function caches its return value."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))

    call_count = 0

    @disk_cache(ttl=0, prefix="test")
    def expensive(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    assert expensive(5) == 10
    assert expensive(5) == 10  # cached
    assert call_count == 1


def test_disk_cache_different_args(tmp_path, monkeypatch):
    """Different arguments produce different cache entries."""
    monkeypatch.setattr("app.common.cache.CACHE_DIR", str(tmp_path))

    @disk_cache(ttl=0)
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add(3, 4) == 7
