"""
Simple script to download GSM8K dataset train and test splits to data directory
"""

import os
from datasets import load_dataset
import json

def download_gsm8k_to_data():
    """
    Download GSM8K dataset train and test splits and save to data/GSM8K directory
    """
    
    # Create GSM8K directory in data
    output_dir = "data/GSM8K"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading GSM8K dataset...")
    
    try:
        # Load the GSM8K dataset
        dataset = load_dataset("openai/gsm8k", "main")
        print("Successfully loaded GSM8K dataset")

        
        
        # Process both train and test splits
        for split in ['train', 'test']:
            split_data = list(dataset[split])
            print(f"\nProcessing {split} split ({len(split_data)} samples)")
            
            # Format the data
            formatted_data = []
            for item in split_data:
                formatted_item = {
                    "answer": item['answer'],
                    "instruction": "Given a simple mathematical question, please directly provide the final answer.Question: {question};\nYour response should follow the structure outlined below:\nR: <Replace Here With Your Reasonings>;\nA: Place your Final Answer here as a clear numeric value. Ensure there are no additional words, signs, or explanations! Enclose the numeric value in angle brackets.\nAn example of the desired output is:\nR: First find the total number of starfish arms: 7 starfish * 5 arms/starfish = <<7*5=35>>35 arms\nThen add the number of seastar arms to find the total number of arms: 35 arms + 14 arms = <<35+14=49>>49 arms\nA: <49> \n",
                    "input": item['question']
                }
                formatted_data.append(formatted_item)
            
            # Save to file
            output_file = os.path.join(output_dir, f"GSM8K_{split}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(formatted_data)} {split} samples to {output_file}")
            
            # Show sample structure for first split only
            if split == 'train' and len(formatted_data) > 0:
                print(f"\nSample data structure:")
                sample = formatted_data[0]
                print("\nExample item:")
                for key, value in sample.items():
                    print(f"\n{key}:")
                    print(value)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Note: You need to login using `huggingface-cli login` to access this dataset")
    success = download_gsm8k_to_data()
    if success:
        print("\nGSM8K dataset download completed successfully!")
    else:
        print("\nGSM8K dataset download failed!") 