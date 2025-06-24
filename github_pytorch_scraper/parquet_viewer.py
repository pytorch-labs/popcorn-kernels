import argparse
import os

import pandas as pd
import pyarrow.parquet as pq


def view_parquet(file_path, rows=5):
    """
    View the contents of a parquet file.

    Parameters:
    file_path (str): Path to the parquet file
    rows (int): Number of rows to display (default: 5)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File: {file_path} ({file_size:.2f} MB)")

    # Get parquet metadata
    parquet_file = pq.ParquetFile(file_path)
    num_row_groups = parquet_file.num_row_groups
    schema = parquet_file.schema
    total_rows = parquet_file.metadata.num_rows

    print(f"Number of row groups: {num_row_groups}")
    print(f"Total rows: {total_rows}")
    print("\nSchema:")
    print(schema)

    # Read the parquet file into a pandas DataFrame
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return
    pd.set_option("display.max_columns", None)
    # Print head of DataFrame
    print(f"\nFirst {rows} rows:")
    print(df.head(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View contents of a parquet file")
    parser.add_argument("file_path", help="Path to the parquet file")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to display")

    args = parser.parse_args()
    view_parquet(args.file_path, args.rows)
