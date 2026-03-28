Public Nasdaq component membership snapshots downloaded from:

`https://indexes.nasdaqomx.com/Index/WeightingData`

Files in this folder:

- `*_official_membership_daily.csv`: one row per available public trade date with pipe-delimited `Tickers` and `Names`
- `*_official_membership_periods.csv`: compressed change periods derived from the daily snapshots
- `official_membership_manifest.json`: generation metadata and detected public coverage windows

Important limitations:

- This public endpoint exposes component membership (`Name`, `Symbol`) but not full official weights.
- Coverage differs by index:
  - `NDX` public membership starts in 2003.
  - `NDXMEGA` public membership starts in 2024.
  - `NDX30` public membership starts in 2024.

To refresh these files, run:

```bash
python data/ndx_simulation/scripts/download_official_membership.py
```
