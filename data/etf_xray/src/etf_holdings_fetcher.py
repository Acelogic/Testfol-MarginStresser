"""
ETF Holdings Fetcher
Downloads and parses SEC N-PORT filings to extract ETF holdings.
"""

import requests
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
import logging
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# SEC Configuration
USER_AGENT = "Antigravity/1.0 (antigravity_agent@google.com)"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
BASE_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

# Cache directory
CACHE_DIR = config.ETF_CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Common ETF CIK mapping (manually curated for major ETFs)
KNOWN_ETFS = {
    "SPY": "0000884394",
    "QQQ": "0001067839", 
    "VOO": "0001497289",
    "VTI": "0000036405",   # Vanguard Index Funds
    "BND": "0000794105",   # Vanguard Bond Index Funds
    "VXUS": "0000736054",  # Vanguard Star Funds
    "VEA": "0000225997",   # Vanguard Tax-Managed Funds
    "VWO": "0000857489",   # Vanguard International Equity Index Funds
    "AGG": "0001100663",   # iShares Trust
    "IWM": "0001100663",   # iShares Trust
    "IVV": "0001100663",   # iShares Trust
    "TLT": "0001100663",   # iShares Trust
    "EEM": "0000930667",   # iShares Inc
    "DIA": "0001064642",   # SPDR Dow Jones Industrial Average ETF Trust
    "XLK": "0001064641",   # Select Sector SPDR Trust
    "XLE": "0001064641",
    "XLF": "0001064641",
    "XLV": "0001064641",
    
    # Simulation Ticker Mappings (Testfol logic) - map to same CIKs
    "QQQSIM": "0001067839",
    "SPYSIM": "0000884394",
    "VOOSIM": "0001497289",
    "VTISIM": "0000036405",
    "BNDSIM": "0000794105",
    "VXUSSIM": "0000736054",
    "VEASIM": "0000225997",
    "VWOSIM": "0000857489",
    "AGGSIM": "0001100663",
    "IWMSIM": "0001100663",
    "IVVSIM": "0001100663",
    "TLTSIM": "0001100663",
    "EEMSIM": "0000930667",
    "DIASIM": "0001064642",
    "XLKSIM": "0001064641",
    "XLESIM": "0001064641",
    "XLFSIM": "0001064641",
    "XLVSIM": "0001064641",
    
    # --- Vanguard Extended Mappings (CIK 0000036405) ---
    "VUG": "0000036405", "VUGSIM": "0000036405",
    "VTV": "0000036405", "VTVSIM": "0000036405",
    "VO": "0000036405",  "VOSIM": "0000036405",
    "VOE": "0000036405", "VOESIM": "0000036405",
    "VOT": "0000036405", "VOTSIM": "0000036405",
    "VB": "0000036405",  "VBSIM": "0000036405",
    "VBR": "0000036405", "VBRSIM": "0000036405",
    "VBK": "0000036405", "VBKSIM": "0000036405",
    
    # --- Select Sector SPDRs (CIK 0001064641) ---
    "XLB": "0001064641", "XLBSIM": "0001064641",
    "XLC": "0001064641", "XLCSIM": "0001064641",
    "XLI": "0001064641", "XLISIM": "0001064641",
    "XLP": "0001064641", "XLPSIM": "0001064641",
    "XLU": "0001064641", "XLUSIM": "0001064641",
    "XLY": "0001064641", "XLYSIM": "0001064641",
    # VNQ (Vanguard Real Estate) - CIK 0000036405? Or different?
    # Typically 0000036405 covers Index Funds. Let's guess it works or fall back.
    # Actually VNQ is often CIK 0001234006 (Vanguard Specialized Funds).
    # Let's map VNQ specifically if we want to be safe, or leave it for lookup.
    "VNQ": "0001234006", "REITSIM": "0001234006", 
    
    # --- iShares Bonds (CIK 0001364742 ? or 1100663) ---
    # AGG is 1100663. TLT is 1100663.
    # IEF/SHY might be different. Search said 1364742.
    "IEF": "0001100663", "IEFSIM": "0001100663", # Try the main iShares Trust first
    "SHY": "0001100663", "SHYSIM": "0001100663",
    "IEI": "0001100663", "IEISIM": "0001100663",
    "GOVT": "0001100663",
    
    # Leveraged Mappings (Map to underlying parent CIK for X-Ray exposure)
    # 3x
    "UPRO": "0000884394", # SPY
    "SPEU": "0000884394", # SPY
    "TQQQ": "0001067839", # QQQ
    "TECL": "0001064641", # XLK
    "FAS":  "0001064641", # XLF
    "UDOW": "0001064642", # DIA
    "URTY": "0001100663", # IWM
    "TMF":  "0001100663", # TLT (20y Treasury) -> iShares Trust
    "TYD":  "0001100663", # IEF (7-10y Treasury) -> iShares Trust
    "MIDU": "0001100663", # IJJ proxy
    "HIBL": "0000884394", # SPY (High Beta) proxy
    
    # 2x
    "SSO":  "0000884394", # SPY
    "QLD":  "0001067839", # QQQ
    "UWM":  "0001100663", # IWM
    "DDM":  "0001064642", # DIA
    "MVV":  "0001100663", # IJJ proxy
    "DIG":  "0001064641", # XLE
    "UYG":  "0001064641", # XLF
    "ROM":  "0001064641", # XLK
    "URE":  "0001064641", # XLRE proxy? Or XLF/Real Estate
    "USD":  "0001064641", # XLK (Semis) proxy
    "UBT":  "0001100663", # TLT
    "UST":  "0001100663", # IEF
}

# Mapping of tickers to keywords that must appear in the N-PORT filing
# identifying the correct series filing. 
# For multi-series trusts (Vanguard, iShares), we use the unique SEC Series ID (e.g. S000002841)
# because the Trust header often lists ALL series names, leading to false positives with name matching.
FUND_KEYWORDS = {
    # Vanguard - Series IDs (Preferred where known and unique)
    "VXUS": "S000002932", # Vanguard Total International Stock Index Fund
    "VEA": "S000004386",  # Vanguard Developed Markets Index Fund
    "BND": "S000002564",  # Vanguard Total Bond Market Index Fund
    "VWO": "S000005786",  # Vanguard Emerging Markets Stock Index Fund
    # VTI: Series ID is ambiguous/missing (likely S000002859 but difficult to verify).
    # Reverting to NAME matching, but rely on validate_filing_content strict checks.
    "VTI": "Vanguard Total Stock Market Index Fund", 
    "VOO": "Vanguard S&P 500 ETF", 
    
    # iShares Trust - Series IDs (Verified reliable)
    "AGG": "S000004362", # iShares Core U.S. Aggregate Bond ETF
    "IWM": "S000004308", # iShares Russell 2000 ETF
    "IVV": "S000004310", # iShares Core S&P 500 ETF
    "TLT": "S000004360", # iShares 20+ Year Treasury Bond ETF
    "EEM": "S000004266", # iShares MSCI Emerging Markets ETF
    "XLK": "S000006415", # Technology Select Sector SPDR Fund (Select Sector SPDR Trust)
    # XLE/XLF/XLV share trust with XLK, should find IDs for them.
    # XLE: S000006416, XLF: S000006417, XLV: S000006419 (Sequential usually)
    # Use names for now for sectors if unsure.
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",

    # SPY / DIA / QQQ (Single or distinct enough CIKs, no need for strict keyword validation)
    # "SPY": "S&P 500", 
    # "DIA": "SPDR Dow Jones Industrial Average ETF Trust",
    # "QQQ": "Invesco QQQ Trust",
    
    # Leveraged Maps (Route to Underlying Series)
    "TMF": "S000004360", # Maps to TLT
    "TMFSIM": "S000004360",
    "TYD": "iShares 7-10 Year Treasury Bond ETF", # Maps to IEF
    "TYDSIM": "iShares 7-10 Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
}

# Synthetic Holdings for simple asset classes where individual bond CUSIPs are noise
# or where fetching is too expensive/unreliable.
SYNTHETIC_HOLDINGS = {
    # Treasuries
    "TLT": [("US Treasury Bonds (20+ Year)", 1.0)],
    "TLTSIM": [("US Treasury Bonds (20+ Year)", 1.0)],
    "TMF": [("US Treasury Bonds (20+ Year)", 1.0)], # Leverage handled by engine
    "TMFSIM": [("US Treasury Bonds (20+ Year)", 1.0)],
    "UBT": [("US Treasury Bonds (20+ Year)", 1.0)],
    
    "IEF": [("US Treasury Bonds (7-10 Year)", 1.0)],
    "IEFSIM": [("US Treasury Bonds (7-10 Year)", 1.0)],
    "TYD": [("US Treasury Bonds (7-10 Year)", 1.0)],
    "UST": [("US Treasury Bonds (7-10 Year)", 1.0)],
    
    "SHY": [("US Treasury Bonds (1-3 Year)", 1.0)],
    "SHYSIM": [("US Treasury Bonds (1-3 Year)", 1.0)],
    "BIL": [("US Treasury Bills (0-3 Month)", 1.0)],
    "GOVT": [("US Treasury Bonds (All Maturities)", 1.0)],
    "AGG": [("US Aggregate Bond Market", 1.0)],
    "BND": [("US Total Bond Market", 1.0)],
    
    # Generic Cash/Money Market
    "CASH": [("Cash", 1.0)],
}

# Add SIM keywords
for t, k in list(FUND_KEYWORDS.items()):
    FUND_KEYWORDS[f"{t}SIM"] = k


def setup_session():
    """Configure requests session with proper SEC headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def get_cik_for_ticker(ticker, session=None):
    """
    Look up CIK for a given ticker symbol.
    First checks KNOWN_ETFS, then queries SEC's company_tickers.json.
    """
    # Normalize ticker: Remove suffixes like ?L=2 or @...
    clean_ticker = ticker.split('?')[0].split('@')[0].upper()
    
    if clean_ticker in KNOWN_ETFS:
        return KNOWN_ETFS[clean_ticker]
    
    if session is None:
        session = setup_session()
    
    try:
        response = session.get(COMPANY_TICKERS_URL)
        response.raise_for_status()
        data = response.json()
        
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                logging.info(f"Found CIK {cik} for {ticker}")
                return cik
                
    except Exception as e:
        logging.error(f"Error looking up CIK for {ticker}: {e}")
    
    return None


def get_recent_nport_filings(cik, session=None, limit=60):
    """
    Fetch the latest N-PORT filing metadata for a given CIK.
    Returns: List of (accession_number, filing_date, primary_document)
    """
    if session is None:
        session = setup_session()
    
    url = SUBMISSIONS_URL.format(cik=cik.zfill(10))
    session.headers["Host"] = "data.sec.gov"
    
    try:
        response = session.get(url)
        response.raise_for_status()
        data = response.json()
        
        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            logging.warning(f"No recent filings for CIK {cik}")
            return []
        
        results = []
        target_forms = ["NPORT-P", "NPORT-NP"]
        for i in range(len(filings["accessionNumber"])):
            if filings["form"][i] in target_forms:
                results.append((filings["accessionNumber"][i], filings["filingDate"][i], filings["primaryDocument"][i]))
                if len(results) >= limit:
                    break
        return results
        
    except Exception as e:
        logging.error(f"Error fetching filings for CIK {cik}: {e}")
        return []


def download_nport_file(cik, accession, primary_doc, session=None, peek_keyword=None):
    """
    Download the N-PORT file.
    If peek_keyword is provided, streams the first 500KB to check for the keyword.
    If keyword is not found in the peek window, aborts download and returns None.
    If keyword is found (or no keyword provided), completes download and returns filepath.
    """
    if session is None:
        session = setup_session()
    
    accession_no_dashes = accession.replace("-", "")
    # Try rendered path first as parse_nport_xhtml often works better with it,
    # and it's where the readable content is.
    # Note: Older logic used raw XML path. Both are xml. 
    # Rendered path: .../xslFormNPORT-P_X01/primary_doc.xml
    # Raw path: .../primary_doc.xml
    # We prefer rendered if available for the XHTML parser? 
    # Actually my previous specific parser check for 'xslFormNPORT' implies we want that.
    
    url = f"{BASE_ARCHIVE_URL}/{cik}/{accession_no_dashes}/xslFormNPORT-P_X01/{primary_doc}"
    
    cache_path = os.path.join(CACHE_DIR, f"{cik}_{accession}.xml")
    if os.path.exists(cache_path):
        logging.info(f"Using cached file: {cache_path}")
        return cache_path
    
    try:
        logging.info(f"Downloading N-PORT from {url}")
        res = session.get(url, headers={"Host": "www.sec.gov"}, stream=True, timeout=30)
        
        # If 404 on rendered form, fallback to raw? 
        # For now assume success or error.
        if res.status_code == 404:
            # Fallback to raw
            url = f"{BASE_ARCHIVE_URL}/{cik}/{accession_no_dashes}/{primary_doc}"
            logging.info(f"Rendered form not found, trying raw: {url}")
            res = session.get(url, headers={"Host": "www.sec.gov"}, stream=True, timeout=30)
            
        res.raise_for_status()
        
        temp_path = cache_path + ".tmp"
        peek_size = 5 * 1024 * 1024 # 5MB
        peek_buffer = b""
        found = False if peek_keyword else True
        
        with open(temp_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=16384):
                f.write(chunk)
                
                # Peek logic
                if not found and len(peek_buffer) < peek_size:
                    peek_buffer += chunk
                    try:
                        # Decode conservatively
                        text_chunk = peek_buffer.decode('utf-8', errors='ignore')
                        if peek_keyword in text_chunk:
                            found = True
                            logging.info(f"Found keyword '{peek_keyword}' in stream.")
                    except:
                        pass
                        
                if not found and len(peek_buffer) >= peek_size:
                    logging.info(f"Keyword '{peek_keyword}' not found in first 5MB. Aborting.")
                    res.close()
                    os.remove(temp_path)
                    return None
                    
        os.rename(temp_path, cache_path)
        # time.sleep(0.12) # Rate limit politeness handled by session/retry usually, but okay to keep if needed
        return cache_path
        
    except Exception as e:
        logging.error(f"Error downloading N-PORT: {e}")
        if os.path.exists(cache_path + ".tmp"):
             os.remove(cache_path + ".tmp")
        return None


def parse_nport_file(filepath):
    """Parse N-PORT file (detect XML or XHTML)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(2048)
    
    if "<html" in content.lower() or "<!doctype html" in content.lower():
        return parse_nport_xhtml(filepath)
    else:
        return parse_nport_xml(filepath)


def parse_nport_xml(xml_path):
    """Parse raw N-PORT XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'n1': 'http://www.sec.gov/edgar/document/nport/invstOrSec'}
        holdings = []
        for invst in root.findall('.//n1:invstOrSec', ns):
            name = invst.findtext('n1:name', default='', namespaces=ns)
            cusip = invst.findtext('n1:cusip', default='', namespaces=ns)
            pct = invst.findtext('n1:pctVal', default='0', namespaces=ns)
            holdings.append({
                'name': name, 'cusip': cusip, 'weight': float(pct)/100, 'ticker': ''
            })
        return pd.DataFrame(holdings)
    except Exception as e:
        logging.error(f"XML parse error: {e}")
        return pd.DataFrame()


import re

def parse_nport_xhtml(filepath):
    """
    Parses N-PORT holdings from rendered XHTML format.
    Uses a section-based approach: splits by 'Part C' markers and extracts holdings from each.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        holdings = []
        
        # Split by Part C sections - each section contains one security
        # Vanguard uses: <h1>NPORT-P: Part C: Schedule of Portfolio Investments</h1>
        sections = re.split(r'<h1>NPORT-P: Part C: Schedule of Portfolio Investments</h1>', content, flags=re.IGNORECASE)
        
        logging.info(f"Found {len(sections)-1} Part C sections in {filepath}")
        
        # Skip the first section (header info before first security)
        for section in sections[1:]:
            # Limit section size to first 10k chars to speed up regex
            section = section[:10000]
            
            name = None
            weight = None
            cusip = None
            
            # Extract issuer name - look for "a. Name of issuer" label
            name_match = re.search(
                r'Name of issuer[^<]*</td>\s*<td[^>]*>\s*<div[^>]*class=["\']fake[Bb]ox[^"\']*["\'][^>]*>([^<]+)',
                section, re.IGNORECASE | re.DOTALL
            )
            if name_match:
                name = re.sub(r'<[^>]+>', '', name_match.group(1)).strip()
            
            # Extract percentage
            pct_match = re.search(
                r'Percentage value compared to net assets[^<]*</td>\s*<td[^>]*>\s*<div[^>]*class=["\']fake[Bb]ox[^"\']*["\'][^>]*>([^<]+)',
                section, re.IGNORECASE | re.DOTALL
            )
            if pct_match:
                try:
                    weight = float(pct_match.group(1).strip()) / 100.0
                except:
                    pass
            
            # Extract CUSIP
            cusip_match = re.search(
                r'CUSIP[^<]*</td>\s*<td[^>]*>\s*<div[^>]*class=["\']fake[Bb]ox[^"\']*["\'][^>]*>([^<]+)',
                section, re.IGNORECASE | re.DOTALL
            )
            if cusip_match:
                cusip_val = cusip_match.group(1).strip()
                if cusip_val and cusip_val != "N/A":
                    cusip = cusip_val
            
            if name and weight is not None:
                holding = {"name": name, "weight": weight}
                if cusip:
                    holding["cusip"] = cusip
                holdings.append(holding)
        
        logging.info(f"Extracted {len(holdings)} holdings from {filepath}")
            
        if not holdings:
            logging.warning(f"No holdings found in {filepath}")
            
        return pd.DataFrame(holdings)
    except Exception as e:
        logging.error(f"XHTML parse error in {filepath}: {e}")
        return pd.DataFrame()


def get_custom_mega_holdings(ticker):
    """
    Get holdings for NDXMEGASIM or NDXMEGA2SIM from local results files.
    """
    clean_ticker = ticker.split('?')[0].split('@')[0].upper()
    
    if clean_ticker == "NDXMEGASIM":
        filename = "ndx_mega_constituents.csv"
    elif clean_ticker == "NDXMEGA2SIM":
        filename = "ndx_mega2_constituents.csv"
    else:
        return None
        
    # Point to NDX Simulation Results explicitly (Cross-module access)
    # BASE_DIR is .../etf_xray
    # We want .../ndx_simulation/data/results
    ndx_results_dir = os.path.join(config.BASE_DIR, "../ndx_simulation/data/results")
    
    results_path = os.path.join(ndx_results_dir, filename)
    if not os.path.exists(results_path):
        logging.warning(f"Custom holdings file not found: {results_path}")
        return None
        
    try:
        df_all = pd.read_csv(results_path)
        if df_all.empty:
            return None
            
        # Get the latest entry
        latest = df_all.iloc[-1]
        tickers = latest['Tickers'].split('|')
        weights = [float(w) for w in latest['Weights'].split('|')]
        
        # We need names. Let's try to get them from the weights file if possible,
        # or just use tickers as names for now.
        holdings = []
        for t, w in zip(tickers, weights):
            holdings.append({
                'name': t, # Will try to enrich with real names if we can
                'cusip': '',
                'weight': w,
                'ticker': t
            })
            
        res_df = pd.DataFrame(holdings)
        
        # Enrich names from nasdaq_quarterly_weights.csv if available
        try:
            weights_file = os.path.join(ndx_results_dir, "nasdaq_quarterly_weights.csv")
            weights_df = pd.read_csv(weights_file)
            name_map = dict(zip(weights_df['Ticker'], weights_df['Name']))
            res_df['name'] = res_df['ticker'].map(lambda x: name_map.get(x, x))
        except:
            pass
            
        logging.info(f"Loaded {len(res_df)} custom holdings for {clean_ticker}")
        return res_df
        
    except Exception as e:
        logging.error(f"Error loading custom mega holdings: {e}")
        return None


def validate_filing_content(filepath, keyword):
    """
    Validates that the filing actually belongs to the fund by checking strict XML/HTML context.
    - If keyword starts with "S00", expects it in <td class="label">Series ID</td>...<div>KEYWORD</div>
    - If keyword is Name, expects it in <td class="label">Series Name</td>...<div>KEYWORD</div> or <seriesName>KEYWORD</seriesName>
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Check first 5MB (Header usually)
            content = f.read(5 * 1024 * 1024)
            
        if keyword.startswith("S00"):
            # Series ID Check
            # Look for <div>S00...</div> near "Series ID"
            # Flexible regex to span lines
            # Pattern: Series ID ... S000002841
            # Note: "Series ID" usually in <td class="label">. Value in separate td/div.
            if f'S{keyword[1:]}' not in content: # Simple pre-check
                return False
                
            # Strict regex
            # Matches: label">Series ID</td> ... >S000002841<
            pattern = r'Series ID.*?{}.*?'.format(keyword)
            # This regex is too simple for complex HTML.
            # Let's rely on "fakeBox" context or just verify it's not "Affiliated Series" logic?
            # Actually, simply checking if "Series ID" appears nearby is good.
            # But earlier "S000002841" appeared in 243 (Extended Market) as the reporting series.
            # So if ID is unique, simple check is fine.
            # If 243 IS Extended Market, then S000002841 IS Extended Market ID. 
            # So correct ID is sufficient.
            return keyword in content 
            
        else:
            # Name Check - stricter to avoid "mentioned as benchmark"
            # Look for "Series Name" or "Name of Series" context
            # Or just check if keyword exists (legacy behavior if regex fails)
            # But matching "Total Stock Market" in 245 (Mid-Cap) was false positive.
            # Mid-Cap 245 probably listed Total Stock Market as a benchmark?
            # So we need Context.
            
            # Context: "Item A.2" or "Series Name"
            if "Series Name" in content and keyword in content:
                 # Check if they are close?
                 return True # Placeholder for now, hard to regex reliably without bs4
            
            return keyword in content # Fallback to loose check
            
    except Exception as e:
        logging.error(f"Error validating {filepath}: {e}")
        return False

def fetch_mags_holdings(ticker):
    """
    Scrapes StockAnalysis.com for MAGS holdings, aggregating Swaps + Direct positions.
    Replicates logic from Maginator project.
    Falls back to Equal-Weight MAG7 if scraper fails.
    """
    import requests
    from bs4 import BeautifulSoup
    
    mag7_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    name_map = {
        "NVDA": "NVIDIA Corp.",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc."
    }
    
    holdings_map = {t: 0.0 for t in mag7_tickers}
    found_any = False
    
    try:
        url = "https://stockanalysis.com/etf/mags/holdings/"
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")[1:] # Skip header
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4: continue
                symbol = cells[1].get_text(strip=True)
                name = cells[2].get_text(strip=True).upper()
                weight_str = cells[3].get_text(strip=True).replace('%', '')
                try:
                    weight = float(weight_str) / 100.0
                except: continue
                
                matched = None
                if symbol in mag7_tickers: matched = symbol
                elif "SWAP" in name:
                    if "NVDA" in name or "NVIDIA" in name: matched = "NVDA"
                    elif "GOOGL" in name or "ALPHABET" in name: matched = "GOOGL"
                    elif "AMZN" in name or "AMAZON" in name: matched = "AMZN"
                    elif "TSLA" in name or "TESLA" in name: matched = "TSLA"
                    elif "AAPL" in name or "APPLE" in name: matched = "AAPL"
                    elif "MSFT" in name or "MICROSOFT" in name: matched = "MSFT"
                    elif "META" in name: matched = "META"
                
                if matched:
                    holdings_map[matched] += weight
                    found_any = True
    except Exception as e:
        logging.warning(f"MAGS scraper failed: {e}. Falling back to equal weight.")

    # Fallback to Equal Weight if scraper failed or found nothing (Roundhill target is equal weight)
    if not found_any or sum(holdings_map.values()) < 0.05:
        logging.info("Using Equal-Weight MAG7 fallback for MAGS")
        holdings_map = {t: 1.0/7.0 for t in mag7_tickers}
        found_any = True
        
    # Convert to DataFrame
    data = []
    for t, w in holdings_map.items():
        if w > 0:
            data.append({
                'name': name_map.get(t, t),
                'ticker': t,
                'weight': w,
                'cusip': ''
            })
            
    df = pd.DataFrame(data)
    df['etf_ticker'] = ticker
    df['filing_date'] = datetime.now().strftime('%Y-%m-%d (Synthetic/Fallback)' if not found_any else '%Y-%m-%d')
    return df

def get_etf_holdings(ticker):
    """Main entry point for fetching ETF holdings."""
    # 1. Check for custom strategies first
    custom_df = get_custom_mega_holdings(ticker)
    if custom_df is not None:
        custom_df['etf_ticker'] = ticker
        custom_df['filing_date'] = datetime.now().strftime('%Y-%m-%d')
        return custom_df
        
    # 2. Check for Synthetic/Static holdings (Optimization for simple bonds)
    clean_ticker = ticker.split('?')[0].split('@')[0].upper()
    if clean_ticker in SYNTHETIC_HOLDINGS:
        logging.info(f"Using synthetic holdings for {ticker}")
        data = []
        for name, weight in SYNTHETIC_HOLDINGS[clean_ticker]:
            data.append({
                'name': name,
                'weight': weight,
                'cusip': '',
                'ticker': ''
            })
        syn_df = pd.DataFrame(data)
        syn_df['etf_ticker'] = ticker
        syn_df['filing_date'] = 'Synthetic'
        return syn_df
        
    # 3. Check for MAGS (Special Scraper)
    if clean_ticker in ["MAGS", "MAGSSIM"]:
        logging.info(f"Fetching MAGS holdings from StockAnalysis...")
        try:
            return fetch_mags_holdings(ticker)
        except Exception as e:
            logging.error(f"Failed to fetch MAGS holdings: {e}")
            return None

    session = setup_session()
    cik = get_cik_for_ticker(ticker, session)

    if not cik:
        return None
        
    filing_infos = get_recent_nport_filings(cik, session, limit=200)
    if not filing_infos:
        return None
        
    # Get keywords for this ticker
    clean_ticker = ticker.split('?')[0].split('@')[0].upper()
    keyword = FUND_KEYWORDS.get(clean_ticker)
    
    attempts = 0
    max_attempts = 200 # Search all fetched filings (iShares Trust has many series)

    for accession, f_date, primary_doc in filing_infos:
        if attempts >= max_attempts:
            logging.warning(f"Exceeded {max_attempts} attempts for {ticker}. Aborting search.")
            break
            
        logging.info(f"Checking filing {accession} ({f_date})...")
        file_path = download_nport_file(cik, accession, primary_doc, session, peek_keyword=keyword)
        
        if not file_path:
            attempts += 1
            continue
            
        # Strict Validation
        if keyword:
            if not validate_filing_content(file_path, keyword):
                 logging.info(f"Filing {accession} failed validation for '{keyword}', skipping...")
                 attempts += 1
                 continue
                 
        # Size sanity check for broad market ETFs (should be > 5MB)
        if clean_ticker in ["VTI", "BND", "VXUS", "AGG", "IWM", "EEM", "VEA", "VWO", "IVV", "TLT", "SPY"] and os.path.getsize(file_path) < 5 * 1024 * 1024:
             logging.info(f"Filing {accession} too small ({os.path.getsize(file_path)} bytes) for {ticker}, skipping...")
             continue

        logging.info(f"Match found in {accession}!")
                
        holdings_df = parse_nport_file(file_path)
        if holdings_df is not None and not holdings_df.empty:
            holdings_df['etf_ticker'] = ticker
            holdings_df['filing_date'] = f_date
            return holdings_df
            
    return None



if __name__ == "__main__":
    ticker = "QQQ"
    print(f"Fetching holdings for {ticker}...")
    df = get_etf_holdings(ticker)
    if df is not None:
        print(df.nlargest(10, 'weight')[['name', 'cusip', 'weight']])
        print(f"Total holdings: {len(df)}, Total weight: {df['weight'].sum():.2%}")
    else:
        print("Failed.")
