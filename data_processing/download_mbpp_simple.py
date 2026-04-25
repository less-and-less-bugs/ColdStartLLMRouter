"""
Simple script to download MBPP dataset to data directory
"""

import os
from datasets import load_dataset
import json

def download_mbpp_to_data():
    """
    Download MBPP dataset and save to data/mbpp directory
    """
    
    # Create mbpp directory in data
    output_dir = "data/mbpp"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading MBPP dataset...")
    
    try:
        # Load the full MBPP dataset
        dataset = load_dataset("google-research-datasets/mbpp", "full")
        print("Successfully loaded MBPP full dataset")
        
        # Save the entire dataset as a single file with split information
        # : "You are an expert Python programmer, and here is your task: {{text}} Your code should pass these tests:\n\n{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n[BEGIN]\n"
        all_data = []
        for split_name, split_data in dataset.items():
            if split_name == 'prompt':
                continue
            print(f"Processing split: {split_name} ({len(split_data)} samples)")
            split_list = list(split_data)
            # Add split information to each sample and remove prompt field
            for sample in split_list:
                sample_with_split = sample.copy()
                sample_with_split['split'] = split_name
                # Remove prompt field if it exists
                all_data.append(sample_with_split)
        
        all_data_file = os.path.join(output_dir, "mbpp_all.json")
        with open(all_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved all {len(all_data)} samples to {all_data_file}")
        
        # Print dataset info
        print("\nDataset information:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} samples")
        
        # Show sample structure
        if len(all_data) > 0:
            print(f"\nSample data structure:")
            sample = all_data[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_mbpp_to_data()
    if success:
        print("\nMBPP dataset download completed successfully!")
    else:
        print("\nMBPP dataset download failed!") 