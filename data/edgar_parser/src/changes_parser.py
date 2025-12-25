import pandas as pd
import config
import os

_cached_changes = None

def load_changes():
    """Loads and standardizes the changes CSV."""
    if not os.path.exists(config.CHANGES_FILE):
        print(f"Warning: Changes file {config.CHANGES_FILE} not found.")
        return pd.DataFrame(columns=['Date', 'Added Ticker', 'Removed Ticker'])
    
    try:
        df = pd.read_csv(config.CHANGES_FILE)
        # Parse Dates (e.g. "December 18, 2023")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.sort_values('Date')
    except Exception as e:
        print(f"Error parse changes file: {e}")
        return pd.DataFrame(columns=['Date', 'Added Ticker', 'Removed Ticker'])

def get_changes_between(start_date, end_date):
    """
    Returns DataFrame of changes strictly AFTER start_date and up to (inclusive) end_date.
    start_date/end_date can be strings or Datetime.
    """
    global _cached_changes
    if _cached_changes is None:
        _cached_changes = load_changes()
    
    if _cached_changes.empty:
        return _cached_changes
        
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Filter: Changes occurring WITHIN the holding period.
    # If change is on Start Date, we assume the Filing took care of it (or it's the rebal date).
    mask = (_cached_changes['Date'] > start_dt) & (_cached_changes['Date'] <= end_dt)
    return _cached_changes[mask]

def get_replacement_map(start_date, end_date):
    """
    Returns a dictionary mapping {RemovedTicker: AddedTicker} for the period.
    Useful for simple force-swap logic.
    """
    changes = get_changes_between(start_date, end_date)
    if changes.empty:
        return {}
        
    # We prioritize the LATEST change if multiple happen? 
    # Or just mapping.
    # Note: Added Ticker might be empty formatting! 
    # In CSV: "Added Ticker","Removed Ticker"
    # If "Added Ticker" is NaN, it's a deletion without direct replacement (or filler).
    
    replacements = {}
    for _, row in changes.iterrows():
        removed = row.get('Removed Ticker')
        added = row.get('Added Ticker')
        
        if pd.notna(removed) and pd.notna(added):
            replacements[removed] = added
            
    return replacements
