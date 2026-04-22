"""Testfol preset ticker metadata and provider fallback aliases."""

from __future__ import annotations

import pandas as pd


# Provider fallbacks are only approximations for when Testfol's preset data is
# unavailable. The Testfol API remains the preferred source for these presets.
TESTFOL_PRESET_PROVIDER_FALLBACKS: dict[str, str] = {
    # Cash/Bills
    "TBILL": "BIL",
    "CASHX": "BIL",
    "EFFRX": "BIL",

    # US broad market and style boxes
    "SPYSIM": "SPY",
    "SPYTR": "SPY",
    "OEFSIM": "OEF",
    "MDYSIM": "MDY",
    "IJRSIM": "IJR",
    "IWMSIM": "IWM",
    "USMVSIM": "USMV",
    "VTISIM": "VTI",
    "VTITR": "VTI",
    "VTSIM": "VT",
    "VVSIM": "VOO",
    "VOOSIM": "VOO",
    "VTVSIM": "VTV",
    "VUGSIM": "VUG",
    "VOSIM": "VO",
    "VOESIM": "VOE",
    "VOTSIM": "VOT",
    "VBSIM": "VB",
    "VBRSIM": "VBR",
    "VBKSIM": "VBK",
    "IWCSIM": "IWC",
    "MTUMSIM": "MTUM",
    "REITSIM": "VNQ",

    # International
    "VXUSSIM": "VXUS",
    "VXUSX": "VXUS",
    "EFASIM": "EFA",
    "VEASIM": "VEA",
    "VWOSIM": "VWO",
    "VSSSIM": "VSS",
    "EFVSIM": "EFV",

    # Bonds
    "TLTSIM": "TLT",
    "TLTTR": "TLT",
    "ZROZSIM": "ZROZ",
    "ZROZX": "ZROZ",
    "IEFSIM": "IEF",
    "IEFTR": "IEF",
    "IEISIM": "IEI",
    "IEITR": "IEI",
    "SHYSIM": "SHY",
    "SHYTR": "SHY",
    "TIPSIM": "TIP",
    "BNDSIM": "BND",

    # Metals and commodities
    "GLDSIM": "GLD",
    "GOLDX": "GLD",
    "SLVSIM": "SLV",
    "GSGSIM": "GSG",
    "GSGTR": "GSG",
    "UUPSIM": "UUP",

    # Managed futures / return stacking
    "KMLMSIM": "KMLM",
    "KMLMX": "KMLM",
    "DBMFSIM": "DBMF",
    "DBMFX": "DBMF",
    "CAOSSIM": "CAOS",
    "GDESIM": "GDE",
    "RSSBSIM": "RSSB",
    "NTSDSIM": "NTSD",

    # Volatility
    "VIXSIM": "^VIX",
    "VOLIX": "^VIX",
    "SVIXSIM": "SVIX",
    "SVIXX": "SVIX",
    "UVIXSIM": "UVIX",
    "ZVOLSIM": "ZVOL",
    "ZIVBX": "ZVOL",

    # Crypto
    "BTCSIM": "BTC-USD",
    "BTCTR": "BTC-USD",
    "ETHSIM": "ETH-USD",
    "ETHTR": "ETH-USD",

    # Sector ETFs
    "XLBSIM": "XLB",
    "XLBTR": "XLB",
    "XLCSIM": "XLC",
    "XLCTR": "XLC",
    "XLESIM": "XLE",
    "XLETR": "XLE",
    "XLFSIM": "XLF",
    "XLFTR": "XLF",
    "XLISIM": "XLI",
    "XLITR": "XLI",
    "XLKSIM": "XLK",
    "XLKTR": "XLK",
    "XLPSIM": "XLP",
    "XLPTR": "XLP",
    "XLUSIM": "XLU",
    "XLUTR": "XLU",
    "XLVSIM": "XLV",
    "XLVTR": "XLV",
    "XLYSIM": "XLY",
    "XLYTR": "XLY",
    "QQQSIM": "QQQ",
    "QQQTR": "QQQ",

    # Leveraged/special products
    "FNGUSIM": "FNGU",
    "MCISIM": "MCI",

    # Legacy local alias
    "DIA_SIM": "DIA",
}


TESTFOL_PRESET_TICKERS: frozenset[str] = frozenset(
    set(TESTFOL_PRESET_PROVIDER_FALLBACKS)
    | {
        "ZEROX",
        "INFLATION",
    }
)


def clean_ticker_symbol(ticker: str) -> str:
    """Return the uppercase base ticker without query params or share overrides."""
    return str(ticker).split("?")[0].split("@")[0].strip().upper()


def is_fama_french_factor_ticker(ticker: str) -> bool:
    """Return True for Testfol's FF3*/FF5* factor pseudo-tickers."""
    base = clean_ticker_symbol(ticker)
    return base.startswith("FF3") or base.startswith("FF5")


def is_testfol_preset_ticker(ticker: str) -> bool:
    """Return True when ticker is one of Testfol's special preset tickers."""
    base = clean_ticker_symbol(ticker)
    return base in TESTFOL_PRESET_TICKERS or is_fama_french_factor_ticker(base)


def provider_fallback_ticker(ticker: str) -> str:
    """Return a provider-compatible fallback ticker when one is known."""
    base = clean_ticker_symbol(ticker)
    return TESTFOL_PRESET_PROVIDER_FALLBACKS.get(base, base)


def zero_return_series(start_date, end_date, name: str = "ZEROX") -> pd.Series:
    """Build a constant-price series for Testfol's 0% nominal return ticker."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        return pd.Series(dtype=float, name=name)
    index = pd.bdate_range(start_ts, end_ts)
    return pd.Series(100.0, index=index, name=name)
