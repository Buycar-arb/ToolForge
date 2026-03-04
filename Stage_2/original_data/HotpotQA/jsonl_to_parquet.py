#!/usr/bin/env python3
"""
Convert all jsonl files in the specified directory to parquet format
For uploading to GitHub
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List


def jsonl_to_parquet(input_dir: str, output_dir: str = None):
    """
    Convert all jsonl files in the directory to parquet format
    
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
    
    # Find all jsonl files
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No jsonl files found in directory {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} jsonl files")
    
    for jsonl_file in jsonl_files:
        try:
            print(f"\nProcessing file: {jsonl_file.name}")
            
            # Read jsonl file
            records = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Failed to parse JSON at line {line_num}: {e}")
                        continue
            
            if not records:
                print(f"  Warning: File {jsonl_file.name} contains no valid data")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Convert complex nested structures (lists, dicts) to JSON strings for parquet format support
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains lists or dicts
                    sample = df[col].dropna()
                    if len(sample) > 0:
                        first_val = sample.iloc[0]
                        if isinstance(first_val, (list, dict)):
                            # Convert complex objects to JSON strings
                            df[col] = df[col].apply(
                                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                            )
            
            # Generate output filename (replace .jsonl with .parquet)
            output_file = output_path / jsonl_file.name.replace('.jsonl', '.parquet')
            
            # Save as parquet format
            df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
            
            print(f"  ✓ Successfully converted: {jsonl_file.name} -> {output_file.name}")
            print(f"    Records: {len(df)}, Columns: {len(df.columns)}")
            
        except Exception as e:
            print(f"  ✗ Error processing file {jsonl_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nConversion completed!")


if __name__ == "__main__":
    # Default to process the directory where the script is located
    script_dir = Path(__file__).parent
    input_directory = str(script_dir)
    
    print(f"Input directory: {input_directory}")
    print("=" * 60)
    
    jsonl_to_parquet(input_directory)
    
    print("=" * 60)
    print("All files processed!")

