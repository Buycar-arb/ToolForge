#!/usr/bin/env python3
"""
Convert all parquet files in the specified directory to jsonl format
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List


def parquet_to_jsonl(input_dir: str, output_dir: str = None):
    """
    Convert all parquet files in the directory to jsonl format
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path, if None then use input directory
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in directory {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    for parquet_file in parquet_files:
        try:
            print(f"\nProcessing file: {parquet_file.name}")
            
            # Read parquet file
            df = pd.read_parquet(parquet_file, engine='pyarrow')
            
            if df.empty:
                print(f"  Warning: File {parquet_file.name} contains no data")
                continue
            
            # Convert JSON string columns back to original data structures
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to parse string as JSON
                    def try_parse_json(x):
                        if pd.isna(x) or x == '':
                            return x
                        if isinstance(x, str):
                            # Try to parse JSON string
                            try:
                                # Check if it's a JSON string (starts with [ or {)
                                x_stripped = x.strip()
                                if (x_stripped.startswith('[') and x_stripped.endswith(']')) or \
                                   (x_stripped.startswith('{') and x_stripped.endswith('}')):
                                    return json.loads(x)
                            except (json.JSONDecodeError, ValueError):
                                # If not valid JSON, keep as is
                                pass
                        return x
                    
                    # Only convert columns that might be JSON strings
                    sample = df[col].dropna()
                    if len(sample) > 0:
                        first_val = sample.iloc[0]
                        if isinstance(first_val, str):
                            first_stripped = first_val.strip()
                            # If the first value looks like a JSON string, try to convert the entire column
                            if (first_stripped.startswith('[') and first_stripped.endswith(']')) or \
                               (first_stripped.startswith('{') and first_stripped.endswith('}')):
                                df[col] = df[col].apply(try_parse_json)
            
            # Generate output filename (replace .parquet with .jsonl)
            output_file = output_path / parquet_file.name.replace('.parquet', '.jsonl')
            
            # Convert to records list and write to jsonl file
            records = df.to_dict('records')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    # Convert record to JSON string and write
                    json_line = json.dumps(record, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            print(f"  ✓ Successfully converted: {parquet_file.name} -> {output_file.name}")
            print(f"    Records: {len(df)}, Columns: {len(df.columns)}")
            
        except Exception as e:
            print(f"  ✗ Error processing file {parquet_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nConversion completed!")


if __name__ == "__main__":
    # Default to process the directory where the script is located
    script_dir = Path(__file__).parent
    input_directory = str(script_dir)
    
    print(f"Input directory: {input_directory}")
    print("=" * 60)
    
    parquet_to_jsonl(input_directory)
    
    print("=" * 60)
    print("All files processed!")

