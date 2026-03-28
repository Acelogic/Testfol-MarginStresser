import pandas as pd

import config


def canonical_company(ticker):
    return getattr(config, "DUAL_CLASS_GROUPS", {}).get(ticker, ticker)


def attach_company_column(df, ticker_col="Ticker"):
    out = df.copy()
    out["Company"] = out[ticker_col].map(canonical_company)
    return out


def build_company_views(q_weights):
    mapped = attach_company_column(q_weights[q_weights["IsMapped"] == True].copy())
    mapped = mapped.sort_values(["Weight", "Ticker"], ascending=[False, True])

    company_weights = mapped.groupby("Company")["Weight"].sum().sort_values(ascending=False)
    company_cum_weights = company_weights.cumsum()
    company_tickers = mapped.groupby("Company")["Ticker"].apply(list).to_dict()
    return mapped, company_weights, company_cum_weights, company_tickers


def select_companies_up_to_threshold(company_weights, threshold):
    curr_sum = 0.0
    selected = []

    for company, company_weight in company_weights.items():
        if not selected or (curr_sum + company_weight) <= threshold + 1e-12:
            selected.append(company)
            curr_sum += company_weight
        else:
            break

    return selected


def quarterly_company_selection(company_weights, company_cum_weights, current_companies, buffer_threshold):
    current_companies = [company for company in current_companies if company in company_weights.index]
    if not current_companies:
        return []

    buffer_set = {
        company for company, cum_weight in company_cum_weights.items()
        if cum_weight <= buffer_threshold + 1e-12
    }

    retained = [company for company in current_companies if company in buffer_set]
    dropouts = [company for company in current_companies if company not in retained]

    if not dropouts:
        return retained

    dynamic_threshold = max(company_cum_weights.get(company, 1.0) for company in dropouts)

    candidates = [
        company for company, _ in company_weights.items()
        if company not in retained and company_cum_weights[company] <= dynamic_threshold + 1e-12
    ]

    if len(candidates) < len(dropouts):
        extras = [company for company in company_weights.index if company not in retained and company not in candidates]
        candidates.extend(extras)

    return retained + candidates[:len(dropouts)]


def expand_companies_to_tickers(selected_companies, company_tickers):
    selected_tickers = []
    for company in selected_companies:
        selected_tickers.extend(company_tickers.get(company, []))
    return selected_tickers


def pick_unique_fillers(q_weights, selected_tickers, selected_companies, needed):
    if needed <= 0:
        return []

    remaining = attach_company_column(q_weights[q_weights["IsMapped"] == True].copy())
    remaining = remaining[~remaining["Ticker"].isin(selected_tickers)]
    remaining = remaining[~remaining["Company"].isin(set(selected_companies))]
    remaining = remaining.sort_values(["Weight", "Ticker"], ascending=[False, True])
    remaining = remaining.drop_duplicates("Company", keep="first")
    return remaining.head(needed)["Ticker"].tolist()


def iterative_cap_series(weights, cap, total_target=1.0, max_iterations=None):
    max_iterations = max_iterations or config.MAX_CAP_ITERATIONS
    capped = weights.copy()
    if capped.sum() == 0:
        return capped

    capped = (capped / capped.sum()) * total_target

    for _ in range(max_iterations):
        excess = capped[capped > cap]
        if excess.empty:
            break

        surplus = (excess - cap).sum()
        capped[capped > cap] = cap

        others = capped[capped < cap]
        if others.empty or others.sum() <= 0:
            break

        capped[others.index] = others + (surplus * others / others.sum())

    return capped


def project_company_weights_to_tickers(subset_df, company_weights):
    weighted = attach_company_column(subset_df)
    company_totals = weighted.groupby("Company")["Weight"].transform("sum")
    weighted["WithinCompany"] = weighted["Weight"] / company_totals
    final_weights = weighted["Company"].map(company_weights) * weighted["WithinCompany"]
    final_weights = pd.Series(final_weights.values, index=weighted["Ticker"])
    return final_weights.sort_values(ascending=False)


def apply_company_cap(subset_df, cap, total_target=1.0):
    if subset_df.empty:
        return pd.Series(dtype=float)

    weighted = attach_company_column(subset_df)
    company_weights = weighted.groupby("Company")["Weight"].sum().sort_values(ascending=False)
    capped_company_weights = iterative_cap_series(company_weights, cap, total_target=total_target)
    return project_company_weights_to_tickers(weighted, capped_company_weights)


def apply_ndx30_company_cap(subset_df):
    if subset_df.empty:
        return pd.Series(dtype=float)

    weighted = attach_company_column(subset_df)
    company_weights = weighted.groupby("Company")["Weight"].sum().sort_values(ascending=False)
    company_weights = company_weights / company_weights.sum()

    for _ in range(config.MAX_CAP_ITERATIONS):
        over = company_weights[company_weights > config.NDX30_HARD_CAP]
        if over.empty:
            break
        surplus = (over - config.NDX30_HARD_CAP).sum()
        company_weights[company_weights > config.NDX30_HARD_CAP] = config.NDX30_HARD_CAP
        under = company_weights[company_weights < config.NDX30_HARD_CAP]
        if under.empty or under.sum() <= 0:
            break
        company_weights[under.index] = under + surplus * under / under.sum()

    for _ in range(50):
        above = company_weights[company_weights > config.NDX30_SOFT_CAP]
        if above.empty or above.sum() <= config.NDX30_AGG_LIMIT + 0.001:
            break

        min_company = above.idxmin()
        excess = company_weights[min_company] - config.NDX30_SOFT_CAP
        company_weights[min_company] = config.NDX30_SOFT_CAP

        below = company_weights[company_weights < config.NDX30_SOFT_CAP]
        if below.empty:
            break

        room = config.NDX30_SOFT_CAP - below
        total_room = room.sum()
        if total_room <= excess:
            company_weights[below.index] = config.NDX30_SOFT_CAP
        else:
            share = excess * (room / total_room)
            company_weights[below.index] = below + share

    return project_company_weights_to_tickers(weighted, company_weights)


def apply_ndx_quarterly_company_cap(subset_df):
    if subset_df.empty:
        return pd.Series(dtype=float)

    weighted = attach_company_column(subset_df)
    company_weights = weighted.groupby("Company")["Weight"].sum().sort_values(ascending=False)
    company_weights = company_weights / company_weights.sum()

    if (
        company_weights.max() <= 0.24 + 1e-12
        and company_weights[company_weights > 0.045].sum() <= 0.48 + 1e-12
    ):
        return project_company_weights_to_tickers(weighted, company_weights)

    current = company_weights.copy()
    for _ in range(config.MAX_CAP_ITERATIONS):
        current = current / current.sum()

        stage1 = current.copy()
        if stage1.max() > 0.24 + 1e-12:
            stage1 = iterative_cap_series(stage1, 0.20, total_target=1.0)

        final = stage1.copy()
        above = stage1[stage1 > 0.045]
        if above.sum() > 0.48 + 1e-12:
            target_agg = 0.40
            scale = target_agg / above.sum()
            final[above.index] = above * scale

            below = stage1[stage1 <= 0.045]
            released = above.sum() - target_agg
            if released > 0 and not below.empty and below.sum() > 0:
                final[below.index] = below + (released * below / below.sum())

        current = final.sort_values(ascending=False)
        if (
            current.max() <= 0.24 + 0.0001
            and current[current > 0.045].sum() <= 0.48 + 0.0001
        ):
            break

    return project_company_weights_to_tickers(weighted, current)


def apply_ndx_annual_security_cap(weight_series):
    if weight_series.empty:
        return pd.Series(dtype=float)

    current = weight_series.astype(float).copy()
    current = current / current.sum()

    w_sorted = current.sort_values(ascending=False)
    needs_stage1 = current.max() > 0.15 + 1e-12
    needs_stage2 = len(w_sorted) >= 5 and w_sorted.iloc[:5].sum() > 0.40 + 1e-12
    if not needs_stage1 and not needs_stage2:
        return current.sort_values(ascending=False)

    for _ in range(config.MAX_CAP_ITERATIONS):
        current = current / current.sum()

        stage1 = current.copy()
        if stage1.max() > 0.15 + 1e-12:
            stage1 = iterative_cap_series(stage1, 0.14, total_target=1.0)

        final = stage1.copy()
        w_sorted = stage1.sort_values(ascending=False)
        if len(w_sorted) >= 5 and w_sorted.iloc[:5].sum() > 0.40 + 1e-12:
            top5 = w_sorted.iloc[:5].index
            top5_sum = stage1[top5].sum()
            final[top5] = stage1[top5] * (0.385 / top5_sum)

            outside = stage1.index.difference(top5)
            released = top5_sum - 0.385
            if released > 0 and len(outside) > 0 and stage1[outside].sum() > 0:
                final[outside] = stage1[outside] + (released * stage1[outside] / stage1[outside].sum())

            fifth_weight = final[top5].min()
            outside_cap = min(0.044, fifth_weight)

            for _ in range(config.MAX_CAP_ITERATIONS):
                outside_weights = final[outside]
                over = outside_weights[outside_weights > outside_cap + 1e-12]
                if over.empty:
                    break

                surplus = (over - outside_cap).sum()
                final[over.index] = outside_cap

                under = final[outside][final[outside] < outside_cap - 1e-12]
                if under.empty or under.sum() <= 0:
                    break
                final[under.index] = under + (surplus * under / under.sum())

        current = final.sort_values(ascending=False)

        w_check = current.sort_values(ascending=False)
        stage1_ok = w_check.iloc[0] <= 0.15 + 0.0001
        stage2_ok = len(w_check) < 5 or w_check.iloc[:5].sum() <= 0.40 + 0.0001
        if len(w_check) >= 5:
            fifth = w_check.iloc[4]
            outside_limit = min(0.044, fifth) + 0.001
            outside_ok = len(w_check) < 6 or (w_check.iloc[5:] <= outside_limit).all()
        else:
            outside_ok = True

        if stage1_ok and stage2_ok and outside_ok:
            break

    return current.sort_values(ascending=False)
