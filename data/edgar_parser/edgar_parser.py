import os
import re
import csv
import logging
import sys
from html.parser import HTMLParser

# Configuration
DOWNLOAD_DIR = "downloads"
OUTPUT_FILE = "nasdaq_components.csv"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class InvestmentHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row_data = []
        self.current_cell_text = ""
        self.rows = [] 
        self.table_stack = [] # Stack of tables, each is a list of rows

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'table':
            self.in_table = True
            self.table_stack.append([]) # Start a new table context
        elif tag.lower() == 'tr':
            if self.in_table:
                self.in_row = True
                self.current_row_data = []
        elif tag.lower() == 'td':
            if self.in_row:
                self.in_cell = True
                self.current_cell_text = ""

    def handle_endtag(self, tag):
        if tag.lower() == 'table':
            if self.table_stack:
                finished_table_rows = self.table_stack.pop()
                # Analyze the table we just finished
                if self._is_investment_table(finished_table_rows):
                    self.rows.extend(finished_table_rows)
            
            if not self.table_stack:
                self.in_table = False
                
        elif tag.lower() == 'tr':
            if self.in_row:
                self.in_row = False
                if self.current_row_data:
                    # Add to current table (top of stack)
                    if self.table_stack:
                        self.table_stack[-1].append(self.current_row_data)
        elif tag.lower() == 'td':
            if self.in_cell:
                self.in_cell = False
                clean_text = self.current_cell_text.strip()
                # Remove common garbage
                clean_text = clean_text.replace('\xa0', ' ').replace('&nbsp;', ' ')
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                self.current_row_data.append(clean_text)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell_text += data + " "
            
    def _is_investment_table(self, rows):
        """Checks if a parsed table looks like a Schedule of Investments."""
        if not rows:
            return False
            
        # Check first few rows for Headers
        header_found = False
        global DEBUG_FILE
        if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
             print(f"--- Checking Table with {len(rows)} rows ---")
             
        for i in range(min(5, len(rows))):
            row_text = " ".join(rows[i]).lower()
            if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
                 print(f"DEBUG ROW {i}: {row_text}")
            if "value" in row_text and ("shares" in row_text or "principal" in row_text or "security" in row_text):
                 header_found = True
                 if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
                     print("!!! MATCH FOUND !!!")
                 break
        
        if not header_found:
             # DEBUG: Print failures for 2017 file specifically if needed
             # if "2017" in current_file: ...
             return False
            
        # Optional: Check if it has data rows?
        return True

def parse_html_file(filepath):
    """Parses an HTML filing."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    parser = InvestmentHTMLParser()
    parser.feed(content)
    
    # Process any remaining tables in the stack (handle unclosed tables)
    while parser.table_stack:
        finished_table_rows = parser.table_stack.pop()
        global DEBUG_FILE
        if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
             print(f"--- Flushing/Closing Orphaned Table with {len(finished_table_rows)} rows ---")
        if parser._is_investment_table(finished_table_rows):
            parser.rows.extend(finished_table_rows)
            
    # Post-process extracted rows
    holdings = []
    
    # We expect rows with Name, Shares, Value
    for row in parser.rows:
        # cleanup row
        cleaned_row = [c for c in row if c and c not in ['$']]
        
        if len(cleaned_row) < 3:
            continue
            
        # Identify "Value" column and "Shares" column
        # Usually Value is last, Shares is second last.
        # But sometimes there are empty columns.
        
        try:
            # Filter out empty strings
            valid_cols = [c for c in cleaned_row if c.strip()]
            
            if len(valid_cols) < 3:
                continue
                
            val_str = valid_cols[-1].replace(',', '').replace('$', '').replace(')', '').replace('(', '')
            shares_str = valid_cols[-2].replace(',', '').replace('$', '')
            
            # Check if they are numbers
            if not (val_str.isdigit() and shares_str.isdigit()):
                continue
                
            name = " ".join(valid_cols[:-2]).strip()
            # Remove * or similar from name
            name = name.rstrip('*').strip()
            
            # Basic garbage filter: Name shouldn't be "Total" or start with numbers
            if "TOTAL" in name.upper() or "NET ASSETS" in name.upper():
                continue
            if not name[0].isalpha():
                continue

            holdings.append((name, shares_str, val_str))
        except:
            continue
            
    return holdings

def parse_html_as_text(filepath):
    """Fallback: Strip HTML and parse as text."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Strip tags
    text_content = re.sub(r'<[^>]+>', '   ', content)
    
    # Clean HTML entities
    text_content = text_content.replace('&nbsp;', ' ').replace('&#151;', ' ').replace('&amp;', '&')
    
    # Use array of lines
    lines = text_content.split('\n')
    
    # Use array of lines
    lines = text_content.split('\n')
    
    holdings = []
    
    # Reuse row pattern from text parser: Name ... Number ... Number
    # But HTML stripped might have extra spaces.
    # Regex patterns for fallback (single line)
    # 1. Name ... Shares ... Value (Standard Text)
    # 2. Shares ... Name ... Value (HTML 2017 style, if on same line)
    
    patterns = [
        (re.compile(r'^\s*([A-Za-z].*?)\s+([0-9,]+)\s+\$?([0-9,]+)\s*$'), "NSV"),
        (re.compile(r'^\s*([0-9,]+)\s+([A-Za-z].*?)\s+\$?([0-9,]+)\s*$'), "SNV")
    ]
    
    # Try Single Line Matching First
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Heuristics to skip junk
        if len(line) < 20: continue
        if "TOTAL" in line.upper(): continue
        
        match_found_single = False
        for regex, order in patterns:
            match = regex.match(line)
            if match:
                g1, g2, g3 = match.groups()
                if order == "NSV":
                    n, s, v = g1, g2, g3
                else: 
                    s, n, v = g1, g2, g3
                
                # Validation
                v_clean = v.replace(',', '')
                if len(v_clean) >= 6:
                     # Clean Name
                     n = n.strip()
                     n = re.sub(r'\*|\([a-z]\)', '', n).strip()
                     holdings.append((n, s, v))
                     match_found_single = True
                break
    
    # If single line matching worked well, return
    if len(holdings) > 10:
        return holdings
        
    logging.info("    Single-line text parse failed. Trying Multi-line Sequence Parser...")
    
    # Multi-Line Sequence Parser (Shares -> Name -> Value)
    # Trigger: Find valid Share count. Then look ahead for Name, then Value.
    
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if not l:
            i += 1
            continue
            
        # Check if line is a potential Share count (Number, e.g. 3,713,874)
        # Must be digits and commas, maybe dot.
        # Must be at least a reasonable size number?
        s_clean = l.replace(',', '').replace('.', '')
        if s_clean.isdigit() and len(s_clean) > 3: # > 1000 shares
             # Potential Shares found.
             shares = l
             
             # Look ahead for Name (within next 5 lines)
             name = None
             value = None
             j = i + 1
             name_idx = -1
             
             while j < min(len(lines), i + 6):
                 nl = lines[j].strip()
                 if nl and not nl.replace(',','').replace('.','').replace('$','').isdigit():
                      # Found text line. Is it a name?
                      if len(nl) > 3 and "TOTAL" not in nl.upper():
                          name = nl
                          name_idx = j
                          break
                 j += 1
                 
             if name:
                 # Look ahead for Value (within next 5 lines after Name)
                 k = name_idx + 1
                 while k < min(len(lines), name_idx + 6):
                     vl = lines[k].strip()
                     # Value might be "$" line, then Number line.
                     if vl == "$":
                         k += 1
                         continue
                         
                     vl_clean = vl.replace(',','').replace('$','')
                     if vl_clean.isdigit() and len(vl_clean) > 5: # Value > $100k
                         value = vl_clean
                         # Found complete set!
                         
                         name = name.strip()
                         name = re.sub(r'\*|\([a-z]\)', '', name).strip()
                         if len(name) > 2:
                             holdings.append((name, shares, value))
                         
                         # Advance main loop to k
                         i = k
                         break
                     k += 1
        i += 1
            
    return holdings

def parse_text_file(filepath):
    """Parses a text (ASCII) filing."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    holdings = []
    in_schedule = False
    
    # Regex for a line like: "Microsoft Corporation*.......  3,347,177   $303,128,717"
    # Name can contain spaces. Separator is usually multiple dots or spaces.
    # We look for a line ending with two numbers.
    
    row_pattern = re.compile(r'^(.*?)\s+([0-9,]+)\s+\$?([0-9,]+)$')

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "Schedule of Investments" in line:
            in_schedule = True
            continue
            
        if not in_schedule:
            continue
            
        # Stop condition: total or end of table
        if "TOTAL" in line.upper() or "</TABLE>" in line.upper():
            in_schedule = False # Or just break if we assume one table
            
        # Attempt to match row
        match = row_pattern.match(line)
        if match:
            name, shares, value = match.groups()
            
            # Cleanup
            name = name.strip('.').strip()
            name = name.rstrip('*').strip()
            shares = shares.replace(',', '')
            value = value.replace(',', '')
            
            holdings.append((name, shares, value))
            
    return holdings

def process_files():
    files = sorted(os.listdir(DOWNLOAD_DIR))
    all_data = []
    
    for filename in files:
        if not (filename.endswith('.txt') or filename.endswith('.htm') or filename.endswith('.html')):
            continue
            
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        # Extract date from filename (YYYY-MM-DD_...)
        file_date = filename.split('_')[0]
        
        logging.info(f"Processing {filename}...")
        
        holdings = []
        if filename.endswith('.txt'):
            holdings = parse_text_file(filepath)
        else:
            holdings = parse_html_file(filepath)
            
            # FALLBACK
            if len(holdings) < 10:
                 logging.info("  HTML parsing yielded few results. Using fallback text parsing.")
                 holdings = parse_html_as_text(filepath)
            
        logging.info(f"  Found {len(holdings)} holdings.")
        
        for h in holdings:
            all_data.append([file_date, filename, h[0], h[1], h[2]])

    # Write to CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "FilingID", "Company", "Shares", "Value"])
        writer.writerows(all_data)
        
    logging.info(f"Successfully wrote {len(all_data)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_files()
