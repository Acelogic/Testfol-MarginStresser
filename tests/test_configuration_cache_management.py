"""Tests for runtime cache clearing in app.ui.configuration."""

from app.core import tax_library
from app.services import testfol_auth
from app.ui.configuration import clear_runtime_caches


def test_clear_runtime_caches_clears_in_memory_and_session(monkeypatch):
    calls = {"data": 0, "resource": 0, "provider_reset": 0}

    monkeypatch.setattr(
        "app.ui.configuration.st.cache_data.clear",
        lambda: calls.__setitem__("data", calls["data"] + 1),
    )
    monkeypatch.setattr(
        "app.ui.configuration.st.cache_resource.clear",
        lambda: calls.__setitem__("resource", calls["resource"] + 1),
    )
    monkeypatch.setattr(
        "app.services.price_providers.reset_provider",
        lambda: calls.__setitem__("provider_reset", calls["provider_reset"] + 1),
    )

    session_state = {
        "ae_cache": {"SPY": [1, 2, 3]},
        "ae_future": object(),
        "keep_me": 123,
    }
    testfol_auth._token_cache.clear()
    testfol_auth._token_cache.update({"access_token": "abc", "expires_at": 123})
    tax_library._STATE_TAX_TABLES.clear()
    tax_library._STATE_TAX_TABLES.update({"CA": {"brackets": {}}})

    summary = clear_runtime_caches(session_state=session_state)

    assert calls == {"data": 1, "resource": 1, "provider_reset": 1}
    assert summary == {"streamlit_data": 1, "streamlit_resource": 1, "session_keys_removed": 2}
    assert "ae_cache" not in session_state
    assert "ae_future" not in session_state
    assert session_state["keep_me"] == 123
    assert testfol_auth._token_cache == {}
    assert tax_library._STATE_TAX_TABLES == {}
