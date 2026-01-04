import json
import csv
import os
import requests
import difflib
import yfinance as yf
import time
import re
import config

# Configuration
INPUT_CSV = config.COMPONENTS_FILE
SEC_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_JSON_FILE = os.path.join(config.ASSETS_DIR, "company_tickers.json")
MAPPING_FILE = os.path.join(config.ASSETS_DIR, "name_mapping.json")
USER_AGENT = "Antigravity/1.0 (antigravity_agent@google.com)"

def download_sec_json():
    print("Downloading SEC Tickers JSON...")
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(SEC_JSON_URL, headers=headers)
        resp.raise_for_status()
        with open(SEC_JSON_FILE, "wb") as f:
            f.write(resp.content)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def load_sec_data():
    if not os.path.exists(SEC_JSON_FILE) or os.path.getsize(SEC_JSON_FILE) < 5000:
        if not download_sec_json():
            return {}
            
    with open(SEC_JSON_FILE, "r") as f:
        data = json.load(f)
    
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    # Convert to list of dicts
    tickers = []
    for k, v in data.items():
        tickers.append(v)
    return tickers

def get_unique_names():
    names = set()
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["Company"])
    return sorted(list(names))

def clean_name(name):
    # Simplify name for matching
    # Remove common suffixes
    n = name.lower()
    n = re.sub(r'[\.,]', '', n)
    n = re.sub(r'\b(inc|corp|corporation|ltd|limited|company|co|plc)\b', '', n)
    n = n.replace('*', '').strip()
    return n

def price_validate(ticker, date_str, shares, value):
    # Check if Price(date) * shares ~= value within 10%
    # shares and value are strings
    try:
        shares = float(shares)
        val = float(value)
        if shares == 0: return False
        
        implied_price = val / shares
        
        # Determine 5-day window around date
        # YF expects YYYY-MM-DD
        dt = date_str
        
        # We need Split-Adjusted Price from YF?
        # NO. The Shares in CSV are Raw. The Value is Raw.
        # But YF gives Split-Adjusted Price.
        # For validation, we need to know if the implied price matches the YF price (adjusted or raw).
        # Actually: Implied Price is RAW Price.
        # YF Price is ADJ Price.
        
        # If no split happened, match.
        # If split happened, mismatched.
        
        # Strategy: Fetch history + splits.
        # Calc Raw Price from YF data.
        # Raw Price = Adj Close / (Product of split factors since date?) 
        # Actually simpler: YF data has "Stock Splits".
        # We can detect splits.
        
        # But for quick validation, we can just fetch the data.
        # If implied price is 90 and YF is 22... maybe split 4:1.
        # If implied price is 90 and YF is 90... no split.
        
        # Let's fetch the data first.
        data = yf.download(ticker, start=dt, end=pd.to_datetime(dt) + pd.Timedelta(days=5), progress=False)
        if data.empty:
            return False
            
        # Get first available close
        yf_price_adj = data['Close'].iloc[0] # auto_adjust=True default in new YF
        if isinstance(yf_price_adj, pd.Series): yf_price_adj = yf_price_adj.iloc[0]
        
        ratio = implied_price / yf_price_adj
        
        # If ratio is close to 1.0, 2.0, 4.0, 0.5 etc. it's likely a match with split.
        # Allow crude validation.
        
        valid_ratios = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 0.5, 0.25, 0.1]
        
        for r in valid_ratios:
            if 0.85 < (ratio / r) < 1.15:
                return True
                
        return False
    except:
        return False

# Manual overrides for difficult to map companies (ADRs, complex class structures)
MANUAL_OVERRIDES = {
    "ARM Holdings PLC, ADR (b)": "ARM",
    "ASML Holding N.V., New York Shares (Netherlands)": "ASML",
    "Airbnb, Inc., Class A (b)": "ABNB",
    "AstraZeneca PLC, ADR (United Kingdom)": "AZN",
    "Atlassian Corp., Class A (b)": "TEAM",
    "Autodesk, Inc. (b)": "ADSK",
    "Baker Hughes Co., Class A": "BKR",
    "Biogen, Inc. (b)": "BIIB",
    "Charter Communications, Inc., Class A (b)": "CHTR",
    "Coca-Cola Europacific Partners PLC (United Kingdom)": "CCEP",
    "Copart, Inc. (b)": "CPRT",
    "CrowdStrike Holdings, Inc., Class A (b)": "CRWD",
    "Datadog, Inc., Class A (b)": "DDOG",
    "DexCom, Inc. (b)": "DXCM",
    "DoorDash, Inc., Class A (b)": "DASH",
    "Fortinet, Inc. (b)": "FTNT",
    "GLOBALFOUNDRIES, Inc. (b)": "GFS",
    "IDEXX Laboratories, Inc. (b)": "IDXX",
    "Illumina, Inc. (b)": "ILMN",
    "Intuitive Surgical, Inc. (b)": "ISRG",
    "Kraft Heinz Co. (The)": "KHC",
    "Monolithic Power Systems, Inc.": "MPWR",
    "Old Dominion Freight Line, Inc.": "ODFL",
    "Palo Alto Networks, Inc. (b)": "PANW",
    "Ross Stores, Inc.": "ROST",
    "Synopsys, Inc. (b)": "SNPS",
    "Take-Two Interactive Software, Inc. (b)": "TTWO",
    "The Trade Desk, Inc., Class A": "TTD",
    "Trade Desk, Inc. (The), Class A (b)": "TTD",
    "Warner Bros. Discovery, Inc. (b)": "WBD",
    "Workday, Inc., Class A (b)": "WDAY",
    "Zscaler, Inc. (b)": "ZS",
    "lululemon athletica, inc. (b)": "LULU",
    "O’Reilly Automotive, Inc. (b)": "ORLY",
    "Cadence Design Systems, Inc. (b)": "CDNS",
    "CoStar Group, Inc. (b)": "CSGP",
    "GE HealthCare Technologies, Inc.": "GEHC",
    "Marriott International, Inc., Class A": "MAR",
    "MercadoLibre, Inc.": "MELI",
    "MercadoLibre, Inc. (Brazil) (b)": "MELI",
    "Microchip Technology, Inc.": "MCHP",
    "Micron Technology, Inc.": "MU",
    "Moderna, Inc.": "MRNA",
    "Moderna, Inc. (b)": "MRNA",
    "MongoDB, Inc. (b)": "MDB",
    "Monster Beverage Corp.": "MNST",
    "ON Semiconductor Corp. (b)": "ON",
    "PACCAR, Inc.": "PCAR",
    "Regeneron Pharmaceuticals, Inc.": "REGN",
    "Super Micro Computer, Inc. (b)": "SMCI",
    "T-Mobile US, Inc.": "TMUS",
    "Verisk Analytics, Inc.": "VRSK",
    "Vertex Pharmaceuticals, Inc.": "VRTX",
    "Tesla, Inc. (b)": "TSLA",
    "Amazon.com, Inc. (b)": "AMZN",
    "PDD Holdings, Inc., ADR (China) (b)": "PDD",
    "Netflix, Inc. (b)": "NFLX",
    "Alphabet, Inc., Class A": "GOOGL",
    "Alphabet, Inc., Class C": "GOOG",
    "Meta Platforms, Inc., Class A": "META",
    "Comcast Corp., Class A": "CMCSA",
    "Adobe, Inc. (b)": "ADBE",
    "ANSYS, Inc. (b)": "ANSS"
}

def main():
    sec_data = load_sec_data()
    csv_names = get_unique_names()
    
    # Build lookup for SEC titles
    sec_map = {}
    for item in sec_data:
        c_title = clean_name(item['title'])
        sec_map[c_title] = item['ticker']
    
    mapping = {}
    
    # Pre-populate with manual overrides
    mapping.update(MANUAL_OVERRIDES)
    
    print(f"Mapping {len(csv_names)} names...")
    
    matches_found = 0
    
    for name in csv_names:
        # Check manual overrides first
        if name in mapping:
            matches_found += 1
            continue
            
        c_name = clean_name(name)
        ticker = None
        
        # 1. Exact cleaned match
        if c_name in sec_map:
            ticker = sec_map[c_name]
        else:
            # 2. Difflib match
            matches = difflib.get_close_matches(c_name, sec_map.keys(), n=1, cutoff=0.8)
            if matches:
                 ticker = sec_map[matches[0]]
        
        if ticker:
            mapping[name] = ticker
            matches_found += 1
            # print(f"Mapped '{name}' -> {ticker}")
        else:
            # print(f"Failed to map '{name}'")
            pass

    print(f"Mapped {matches_found}/{len(csv_names)} companies.")
    
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

if __name__ == "__main__":
    import pandas as pd
    main()
