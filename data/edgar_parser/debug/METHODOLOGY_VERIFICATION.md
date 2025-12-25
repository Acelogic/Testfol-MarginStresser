# NDX & NDX Mega Methodology Verification Report

This report details the discrepancies found between the provided methodology documents and the current implementation in `data/edgar_parser`.

## Documents Reviewed
1.  **NDX (Nasdaq-100)**: `Methodology_NDX.pdf`
2.  **NDX Mega 1.0**: `NDXMEGA Methodology.pdf`
3.  **NDX Mega 2.0**: `NDXMEGA2_Methodology.pdf`

## Code Reviewed
1.  `reconstruct_weights.py` (Base NDX Reconstruction)
2.  `backtest_ndx_mega.py` (NDX Mega 1.0)
3.  `backtest_ndx_mega2.py` (NDX Mega 2.0)
4.  `config.py`

---

## 1. Breakdown: NDX Base Reconstruction
**File**: `reconstruct_weights.py`

### 🔴 Critical Issue: Missing Annual Reconstitution Logic
The methodology distinguishes between **Quarterly Rebalances** and the **Annual Reconstitution** (December). The code currently applies the **Quarterly** weighting logic to *every* quarter, including December.

*   **Methodology (Annual)**:
    *   **Stage 1 Adjustment**: Cap single security weights at **14%** (if > 15%).
    *   **Stage 2 Adjustment**: The aggregate weight of the **Top 5** securities is capped at **38.5%**.
*   **Current Code**:
    *   Applies **Quarterly** limits (24% Cap / 48% Aggregate Cap) to all dates.
    *   **Impact**: December weights will be incorrect, potentially leading to wrong base weights for the Mega strategies during their annual reconstitution.

---

## 2. Breakdown: NDX Mega 1.0 & 2.0 Common Issues
**Files**: `backtest_ndx_mega.py`, `backtest_ndx_mega2.py`

### 🔴 Critical Issue: Rebalancing Logic (Swap vs. Add/Drop)
The implementation uses a simplified "Buffer Zone" approach that fundamentally differs from the "Swap" logic described in the methodology.

*   **Methodology (Step 2 & 3)**:
    1.  **Retention**: Keep current constituents if they are in the **Top 50%**.
    2.  **Stability**: If *no* current constituents are outside the Top 50%, **NO changes are made** (even if a non-member enters the Top 47%).
    3.  **Swap Mechanism**: If there *are* constituents outside the Top 50% (let's say `N` "bad" stocks):
        *   Determine a "Dynamic Threshold" (max cumulative weight of the bad stocks).
        *   Select the **Top `N`** best non-members that fall below this threshold to replace them.
    *   *Essence*: This is a **Replacement** logic designed to minimize turnover and maintain count stability.
*   **Current Code**:
    *   **Logic**: `Next = (Current & Top 50%) | (Full Top Threshold)`.
    *   **Behavior**: It *always* adds any stock that enters the "Target" (Top 47%/40%), regardless of whether a removal occurred.
    *   **Impact**: The simulator will likely show higher turnover and a growing constituent count compared to the official index.

---

## 3. Breakdown: NDX Mega 2.0 Specifics
**File**: `backtest_ndx_mega2.py`

### 🟠 Discrepancy: Threshold Configuration
The `config.py` settings for Mega 2.0 contradict the provided PDF, though they match the user's previous description.

*   **Methodology PDF**:
    *   Selection Target: **Top 47%** (Same as Mega 1.0).
    *   Buffer Threshold: **Top 50%** (Same as Mega 1.0).
    *   *Note*: The PDF's *Description* mentions "target performance of... Top 45%", but the *Selection Criteria* explicitly states 47%.
*   **Current Code (`config.py`)**:
    *   Selection Target: **40%** (`MEGA2_TARGET_THRESHOLD = 0.40`).
    *   Buffer Threshold: **45%** (`MEGA2_BUFFER_THRESHOLD = 0.45`).
*   **Action Required**: Confirm if the implementation should follow the PDF (47%/50%) or the custom 40%/45% target.

### 🟡 Minor Note: Minimum Security Weighting
*   **Methodology**: "Standard securities ... allocated 99% ... Additional (Filler) securities ... 1%".
*   **Current Code**: Implements a pragmatic fallback. If the Standard securities hit their **30% individual cap** and cannot collectively absorb 99% of the weight, the excess flows to the fillers.
*   **Assessment**: This is a reasonable engineering decision to handle conflicting constraints (30% cap vs 99% group target), but strictly purely deviates from the text which doesn't define precedence.

---

## Summary of Recommended Actions

1.  **Update `reconstruct_weights.py`**: Add a check for `month == 12` to apply the specific Annual Weighting rules (14% Cap, 38.5% Top 5 Cap).
2.  **Rewrite Selection Logic**: Update both backtesters to implement the "Stability & Swap" logic:
    *   Do not add new stocks unless a current stock falls out of the Top 50%.
    *   If replacements are needed, swap the `N` dropouts for the `N` best candidates.
3.  **Clarify Mega 2.0 Thresholds**: Decide whether to stick to the PDF (47%/50%) or the current config (40%/45%).
