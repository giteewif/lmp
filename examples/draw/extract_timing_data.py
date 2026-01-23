#!/usr/bin/env python3
"""
Extract prefill time and decode step times from log files.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Base directory
BASE_DIR = Path("/mnt/zhengcf3/lmp/examples")

# Folders to process
FOLDERS = [
    "deepseek-16b",
    "deepseek-v2-lite", 
    "qwen1.5-moe",
    "qwen3-30B"
]

# GPU configurations and file patterns
GPU_CONFIGS = ["1g", "2g", "3g", "4g"]
FILE_SUFFIXES = ["1", "2", "3"]


def extract_timing_data(log_file: Path, num_steps: int = 31) -> List[float]:
    """Extract prefill time and decode step times from log file.
    
    Returns a list: [prefill_time, decode_step_0, decode_step_1, ..., decode_step_30, avg_decode_time]
    Returns None if extraction fails.
    """
    prefill_pattern = r'prefill time:\s*([\d.]+)\s*seconds'
    decode_pattern = r'decode step (\d+)\s+time:\s*([\d.]+)\s*seconds'
    
    prefill_time = None
    decode_times = {}
    found_prefill = False
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for prefill time (take the first one after we start collecting decode steps)
            prefill_match = re.search(prefill_pattern, line)
            if prefill_match and not found_prefill:
                prefill_time = float(prefill_match.group(1))
                found_prefill = True
                # Reset decode_times when we find a new prefill
                decode_times = {}
            
            # Look for decode steps
            decode_match = re.search(decode_pattern, line)
            if decode_match and found_prefill:
                step = int(decode_match.group(1))
                time = float(decode_match.group(2))
                if step < num_steps:
                    decode_times[step] = time
                    
                    # If we have all steps, we're done
                    if len(decode_times) == num_steps:
                        break
    
    if prefill_time is None or len(decode_times) != num_steps:
        return None
    
    # Return in order (step 0 to step 30)
    decode_list = [decode_times[i] for i in range(num_steps)]
    
    # Calculate average decode time
    avg_decode_time = sum(decode_list) / len(decode_list)
    
    # Return as list: [prefill_time, decode_step_0, ..., decode_step_30, avg_decode_time]
    result = [prefill_time] + decode_list + [avg_decode_time]
    return result


def process_log_file(log_file: Path) -> List[float]:
    """Process a single log file and extract timing data.
    
    Returns a list: [prefill_time, decode_step_0, decode_step_1, ..., decode_step_30, avg_decode_time]
    Returns None if extraction fails.
    """
    return extract_timing_data(log_file, num_steps=31)


def process_folder(folder_name: str) -> Dict:
    """Process all log files in a folder."""
    folder_path = BASE_DIR / folder_name
    results = {}
    
    for gpu_config in GPU_CONFIGS:
        for suffix in FILE_SUFFIXES:
            log_file = folder_path / f"generate_multi_{gpu_config}_{suffix}.log"
            
            if log_file.exists():
                data = process_log_file(log_file)
                if data:
                    key = f"{gpu_config}_{suffix}"
                    results[key] = data
                    print(f"Processed: {log_file.name}")
                else:
                    print(f"Warning: Could not extract data from {log_file.name}")
            else:
                print(f"Warning: File not found: {log_file}")
    
    return results


def main():
    """Main function to extract data from all folders."""
    all_results = {}
    
    for folder in FOLDERS:
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder}")
        print(f"{'='*60}")
        results = process_folder(folder)
        all_results[folder] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    for folder, results in all_results.items():
        print(f"\n{folder}:")
        for key, data in results.items():
            if data:
                print(f"  {key}:")
                print(f"    Prefill time: {data[0]:.4f} s")
                print(f"    Avg decode time: {data[-1]:.4f} s")
                print(f"    Decode steps: {len(data) - 2} steps")
                print(f"    List length: {len(data)} (prefill + 31 decode steps + avg)")
    
    # Save to JSON file
    output_file = BASE_DIR / "draw" / "timing_data.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nData saved to: {output_file}")
    
    # Also create Python list format
    output_py = BASE_DIR / "draw" / "timing_data.py"
    with open(output_py, 'w') as f:
        f.write("# Extracted timing data\n")
        f.write("timing_data = ")
        f.write(repr(all_results))
        f.write("\n")
    print(f"Python data saved to: {output_py}")
    
    return all_results


if __name__ == "__main__":
    main()
