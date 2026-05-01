import pandas as pd
import numpy as np

def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-score normalization to each column: (x - mean) / std.
    If the std of a column is 0, set the entire normalized column to 0
    to avoid division by zero.
    """
    means = df.mean(axis=0, numeric_only=True)
    stds = df.std(axis=0, ddof=0, numeric_only=True)  # ddof=0 is commonly used for population normalization

    # Avoid division by zero: replace std=0 with NaN, then fill with 0 later
    stds_safe = stds.replace(0, np.nan)
    z = (df - means) / stds_safe
    return z.fillna(0.0)

def main():
    input_path = "data.csv"
    output_path = "dynamic_data.csv"

    # 1) Read data and use timestamp as the index
    df = pd.read_csv(input_path)

    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' not found. Please make sure data.csv contains a column named 'timestamp'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if df["timestamp"].isna().any():
        bad_rows = df[df["timestamp"].isna()]
        raise ValueError(f"Some timestamps could not be parsed. Example rows:\n{bad_rows.head()}")

    df = df.set_index("timestamp").sort_index()

    # Ensure all data columns are numeric.
    # If a column contains non-numeric values, try converting them to numeric values.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values as needed.
    # Here, forward-fill along the time axis first, then backward-fill boundary values.
    df = df.ffill().bfill()

    # 2) Apply z-score normalization to each column
    df_z = zscore_normalize(df)

    # 3) Resample by timestamp every 10 seconds
    # Common options:
    # - Mean: mean()
    # - Last value: last()
    # - Median: median()
    # - Resample first, then forward-fill: asfreq("10S").ffill()
    df_resampled = df_z.resample("10S").mean()

    # If you do not want NaN values caused by empty time windows,
    # fill them again here.
    df_resampled = df_resampled.ffill().bfill()

    # 4) Save the processed data
    # Keep the timestamp as the first column in the output file
    df_resampled.index.name = "date"
    df_resampled.to_csv(output_path, index=True, date_format="%Y-%m-%d %H:%M:%S")

    print(f"Processing complete. Output file saved as {output_path}")
    print(f"Output rows: {len(df_resampled)}, output columns: {df_resampled.shape[1]}")

if __name__ == "__main__":
    # main()

    # data = pd.read_csv("dynamic_data.csv", parse_dates=["date"], index_col="date")
    # print(data.head())
    with open("dynamic_data.csv", "r", encoding="utf-8") as f:
        for _ in range(3):
            print(f.readline().rstrip("\n"))