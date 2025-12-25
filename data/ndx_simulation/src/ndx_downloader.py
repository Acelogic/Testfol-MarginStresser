import requests
import json
import os
import time
import datetime
import config

# Configuration
CIK = "1067839"
FORM_TYPE = "485BPOS"
START_DATE = datetime.date(1999, 1, 1)
USER_AGENT = "Antigravity/1.0 (antigravity_agent@google.com)" # SEC requires a User-Agent with contact info
DOWNLOAD_DIR = config.NDX_CACHE_DIR

# SEC EDGAR URLs
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK.zfill(10)}.json"
BASE_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

def setup_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    })
    return session

def download_filing(session, accession_number, primary_document, filing_date):
    """
    Downloads a single filing.
    URL format: https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}
    Note: Accession number in URL usually DOES NOT have dashes.
    """
    accession_no_dashes = accession_number.replace("-", "")
    
    # Sometimes primary_document is empty or not what we want, but let's try the listing first
    # Actually, the bulk feed often gives the primary document.
    # The URL provided by the user: https://www.sec.gov/Archives/edgar/data/1067839/000091205700030669/a485bpos.txt
    # This implies we can just construct it.
    
    url = f"{BASE_ARCHIVE_URL}/{CIK}/{accession_no_dashes}/{primary_document}"
    
    # We might need to switch Host header for document download if it's different from data.sec.gov
    # The archives are usually on www.sec.gov
    download_headers = session.headers.copy()
    download_headers["Host"] = "www.sec.gov"
    
    print(f"Downloading {filing_date} - {url}...")
    
    try:
        response = session.get(url, headers=download_headers)
        response.raise_for_status()
        
        filename = f"{filing_date}_{accession_number}_{primary_document}"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False
    finally:
        time.sleep(0.12) # Rate limit: SEC allows 10 req/s, so > 0.1s sleep is safe

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    session = setup_session()
    
    print(f"Fetching submissions for CIK {CIK}...")
    try:
        response = session.get(SUBMISSIONS_URL)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching submissions: {e}")
        return

    filings = data.get("filings", {}).get("recent", {})
    
    if not filings:
        print("No recent filings found.")
        return

    # Filings data is a dictionary of lists. We need to iterate through them.
    # lists: accessionNumber, filingDate, reportDate, acceptanceDateTime, act, form, fileNumber, filmNumber, items, size, isXBRL, isInlineXBRL, primaryDocument, primaryDocDescription
    
    count = 0
    total_filings = len(filings["accessionNumber"])
    
    print(f"Processing {total_filings} filings...")
    
    for i in range(total_filings):
        form = filings["form"][i]
        filing_date_str = filings["filingDate"][i]
        filing_date = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
        
        if form == FORM_TYPE and filing_date >= START_DATE:
            accession_number = filings["accessionNumber"][i]
            primary_document = filings["primaryDocument"][i]
            
            # primaryDocument is sometimes just a filename like "doc.xml". 
            # If we want the text version, it is often just the accession number + .txt?
            # User example: .../000091205700030669/a485bpos.txt
            # Let's trust primaryDocument for now. If it fails, we might need logic to find the .txt
            
            success = download_filing(session, accession_number, primary_document, filing_date_str)
            if success:
                count += 1
                
    print(f"Done. Downloaded {count} filings.")

if __name__ == "__main__":
    main()
