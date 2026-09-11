import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def date_config():
    app = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from app.ui import configuration, render_sidebar

st.session_state.setdefault("margin_rate_model", "Fixed")
# Isolate unrelated data panels/auth. Streamlit 1.41 AppTest cannot serialize
# the portfolio segmented control on rerun, so omit that control here.
with (
    patch.object(configuration.asset_explorer, "render_asset_explorer"),
    patch.object(configuration.ndx_scanner, "render_ndx_scanner"),
    patch("streamlit.segmented_control"),
    patch("app.services.testfol_auth.get_token", return_value=None),
):
    start_date, _, _, _ = render_sidebar()
    st.session_state["rendered_config"] = configuration.render(start_date=start_date)
""",
        default_timeout=20,
    )
    app.session_state["draw_monthly"] = 1000.0
    app.session_state["draw_monthly_retirement"] = 3333.0
    app.run()
    assert not app.exception
    _set_start_date(app, dt.date(1990, 1, 1))
    return app


def _set_start_date(app, value):
    next(widget for widget in app.date_input if widget.label == "Start Date").set_value(value)
    app.run()
    assert not app.exception


@pytest.mark.parametrize("key", ["retirement_date", "draw_start_date"])
def test_withdrawal_date_tracks_backtest_start_without_losing_selection(date_config, key):
    app = date_config
    assert app.date_input(key=key).min == dt.date(1990, 1, 1)
    assert app.date_input(key=key).value is None

    # Historical dates are accepted and reach the backtest configuration.
    app.date_input(key=key).set_value(dt.date(1995, 1, 1)).run()
    assert not app.exception
    assert app.session_state["rendered_config"][key] == dt.date(1995, 1, 1)

    # Moving the start past a selected date clamps it to the new minimum.
    _set_start_date(app, dt.date(1998, 1, 1))
    assert app.date_input(key=key).min == dt.date(1998, 1, 1)
    assert app.date_input(key=key).value == dt.date(1998, 1, 1)
    assert app.session_state["rendered_config"][key] == dt.date(1998, 1, 1)

    # Expanding the range preserves a valid selection and permits its boundary.
    _set_start_date(app, dt.date(1980, 1, 1))
    assert app.date_input(key=key).min == dt.date(1980, 1, 1)
    assert app.date_input(key=key).value == dt.date(1998, 1, 1)
    app.date_input(key=key).set_value(dt.date(1980, 1, 1)).run()
    assert not app.exception
    assert app.session_state["rendered_config"][key] == dt.date(1980, 1, 1)

    # An intentionally unset date stays unset when the range changes.
    app.date_input(key=key).set_value(None).run()
    _set_start_date(app, dt.date(1990, 1, 1))
    assert app.date_input(key=key).value is None
    assert app.session_state["rendered_config"][key] is None
