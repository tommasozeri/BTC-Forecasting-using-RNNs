# BTC data library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# explanatory data analysis
def eda(df,target='BTC_CLOSE'):
    
    print()
    print('EDA on features')
    
    # Step 1: Data Description
    print("Data Description:")
    print(df.describe())
    
    print("\nData Info:")
    print(df.info())
    
    # Step 2: Time Plots for All Variables
    plt.figure(figsize=(15, 10))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title("Time Series of All Variables")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # Individual Time Plots for Each Variable
    for column in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column], label=column)
        plt.title(f"{column} onchain")
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        plt.show()
    
    # Step 3: Correlation Matrix (Highlighting MKPRU)
    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.show()
    
    # Highlight MKPRU
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
        cbar=True, annot_kws={"size": 10}, linewidths=0.5
    )
    plt.title("Correlation Matrix (BTC price Highlighted)")
    for i, variable in enumerate(corr_matrix.columns):
        if variable == target:
            plt.gca().add_patch(
                plt.Rectangle((i, i), 1, len(corr_matrix.columns), fill=False, edgecolor='red', lw=2)
            )
            plt.gca().add_patch(
                plt.Rectangle((0, i), len(corr_matrix.columns), 1, fill=False, edgecolor='red', lw=2)
            )
    plt.show()
    
    
    # Scatterplots of All Variables with MKPRU
    #target = 'MKPRU'
    num_cols = len(df.columns)
    
    # Determine optimal grid layout: Adjust columns and rows dynamically
    n_cols = 3  # Number of columns in the grid
    n_rows = -(-num_cols // n_cols)  # Compute rows needed, rounded up
    
    plt.figure(figsize=(15, n_rows * 4))  # Adjust the figure size dynamically
    for i, column in enumerate(df.columns):
        if column != target:
            plt.subplot(n_rows, n_cols, i + 1)  # Organizing subplots in rows and columns
            plt.scatter(df[column], df[target], alpha=0.6)
            plt.title(f"Scatterplot: {column} vs {target}")
            plt.xlabel(column)  # Feature on x-axis
            plt.ylabel(target)  # Target on y-axis
            plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    # scatterplot single
    for column in df.columns:
        if column != target:
            plt.figure(figsize=(12, 6))
            plt.scatter(df[column], df[target], alpha=0.6)
            plt.title(f"Scatterplot: {column} vs {target}")
            plt.xlabel(column)  # Feature on x-axis
            plt.ylabel(target)  # Target on y-axis
            plt.grid()
            plt.show()

def aggregate_onchain_weekends_to_monday(ONCHAIN, bbg_index, agg_map=None):
    """
    Aggregate calendar-day on-chain data to BBG weekday index.
    Weekend data (Sat/Sun) is rolled into Monday.
    
    Parameters
    ----------
    ONCHAIN : pd.DataFrame
        Calendar-day on-chain data (DatetimeIndex)
    bbg_index : pd.DatetimeIndex
        Bloomberg weekday index
    agg_map : dict or None
        Optional column-wise aggregation rules
        e.g. {'NTRAN': 'sum', 'DIFF': 'mean'}
    """

    onchain = ONCHAIN.copy()
    onchain.index = pd.to_datetime(onchain.index)

    # create target date
    onchain['bbg_date'] = onchain.index

    # shift weekends to next business day
    weekend_mask = onchain.index.weekday >= 5
    onchain.loc[weekend_mask, 'bbg_date'] = (
        onchain.loc[weekend_mask, 'bbg_date'] + pd.offsets.BDay(1)
    )

    # drop original index
    onchain = onchain.reset_index(drop=True)

    # aggregate
    if agg_map is None:
        agg = onchain.groupby('bbg_date').mean()
    else:
        agg = onchain.groupby('bbg_date').agg(agg_map)

    # keep only Bloomberg dates
    agg = agg.loc[agg.index.intersection(bbg_index)]

    return agg

def build_aligned_dataframe(ONCHAIN, BBG, agg_map=None):
    """
    Build final dataframe aligned on BBG weekdays.
    """

    ONCHAIN = ONCHAIN.copy()
    BBG = BBG.copy()

    ONCHAIN.index = pd.to_datetime(ONCHAIN.index)
    BBG.index = pd.to_datetime(BBG.index)

    # ensure BBG is weekday-only
    BBG = BBG[BBG.index.weekday < 5]

    # aggregate on-chain
    ONCHAIN_AGG = aggregate_onchain_weekends_to_monday(
        ONCHAIN,
        bbg_index=BBG.index,
        agg_map=agg_map
    )

    # merge
    df = BBG.join(ONCHAIN_AGG, how='left')

    return df

def normalize_onchain_features(
    df,
    onchain_cols,
    window=180,
    log_transform_cols=None
    ):
    
    """
    Rolling z-score normalization for on-chain variables.
    """

    out = df.copy()

    if log_transform_cols is None:
        log_transform_cols = []

    for col in onchain_cols:
        series = out[col]

        if col in log_transform_cols:
            series = np.log1p(series)

        rolling_mean = series.rolling(window, min_periods=window).mean()
        rolling_std  = series.rolling(window, min_periods=window).std()

        out[col + '_z'] = (series - rolling_mean) / rolling_std

    return out

def correlation_screen(df, target, features, min_year=None):
    dfx = df.copy()

    # ensure datetime index
    dfx.index = pd.to_datetime(dfx.index)

    if min_year is not None:
        dfx = dfx[dfx.index.year >= min_year]

    corr = (
        dfx[features + [target]]
        .dropna()
        .corr()[target]
        .drop(target)
    )
    return corr


ONCHAIN_AGG_MAP = {

    # ─────────────
    # Network activity / flows → SUM
    # ─────────────
    'NTRAN': 'sum',   # number of transactions
    'NTRAT': 'sum',   # transaction count adjusted
    'NTRBL': 'sum',   # transaction blocks
    'NTREP': 'sum',   # transaction reports / events
    'TRFEE': 'sum',   # total transaction fees
    'TRFUS': 'sum',   # fees in USD
    'TRVOU': 'sum',   # transaction volume (native)
    'TOUTV': 'sum',   # output volume
    'ETRAV': 'sum',   # estimated transaction value
    'ETRVU': 'sum',   # estimated transaction value (USD)
    'CPTRA': 'sum',   # coins per transaction (flow-like usage)
    'CPTRV': 'sum',   # coins per transaction value
    # ─────────────
    # Miner / security metrics → MEAN
    # ─────────────
    'DIFF':  'mean',  # mining difficulty
    'HRATE': 'mean',  # hash rate
    'MIREV': 'mean',  # miner revenue
    'MWNUS': 'mean',  # miner revenue USD
    # ─────────────
    # Supply / stock variables → MEAN
    # ─────────────
    'TOTBC': 'mean',  # total bitcoins in circulation
    'AVBLS': 'mean',  # average balance
    'BLCHS': 'mean',  # blockchain size
    'NADDU': 'mean',  # number of active addresses
    # ─────────────
    # Market valuation variables → MEAN
    # ─────────────
    'MKTCP': 'mean',  # market capitalization
    'MKPRU': 'mean',  # market price (USD) – NOT a target

}
