"""
Simple script to download Multi-News dataset test split to data directory
"""

import os
import json
import hashlib
import requests
from tqdm import tqdm

def download_file(url, local_path):
    """Download a file from URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def generate_id(text):
    """Generate a unique ID for each sample based on its content"""
    return hashlib.sha256(text.encode()).hexdigest()[:40]

def download_multi_news_to_data():
    """
    Download Multi-News dataset test split and save to data/multi_news directory
    Multi-News consists of news articles and human-written summaries
    from the site newser.com.
    """
    
    # Create multi_news directory in data
    output_dir = "data/multi_news"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define source URLs
    base_url = "https://huggingface.co/datasets/multi_news/resolve/main/data"
    source_url = f"{base_url}/test.src.cleaned"
    target_url = f"{base_url}/test.tgt"
    
    # Local paths for downloaded files
    source_file = os.path.join(output_dir, "test.src.cleaned")
    target_file = os.path.join(output_dir, "test.tgt")
    
    print("Downloading Multi-News test dataset files...")
    
    try:
        # Download source and target files
        download_file(source_url, source_file)
        download_file(target_url, target_file)
        
        print("\nProcessing downloaded files...")
        
        # Read and process the files
        formatted_data = []
        with open(source_file, 'r', encoding='utf-8') as src_f, \
             open(target_file, 'r', encoding='utf-8') as tgt_f:
            
            for src_line, tgt_line in zip(src_f, tgt_f):
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()
                
                # Format the data
                formatted_item = {
                    "input": src_text,
                    "length": len(src_text),
                    "dataset": "multi_news",
                    "language": "en",
                    "all_classes": None,
                    "_id": generate_id(src_text),
                    "instruction": "Please summarize the following content into a fluent passage.",
                    "output": tgt_text
                }
                formatted_data.append(formatted_item)
        
        # Save processed data
        output_file = os.path.join(output_dir, "multi_news.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(formatted_data)} test samples to {output_file}")
        
        # Show sample structure
        if len(formatted_data) > 0:
            print(f"\nSample data structure:")
            sample = formatted_data[0]
            print("\nExample item:")
            for key, value in sample.items():
                if key in ['input', 'output']:
                    print(f"\n{key} (first 200 chars):")
                    print(value[:200] + "...")
                else:
                    print(f"\n{key}:")
                    print(value)
        
        # Clean up downloaded files
        os.remove(source_file)
        os.remove(target_file)
        print("\nCleaned up temporary files")
        
    except Exception as e:
        print(f"Error downloading/processing dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_multi_news_to_data()
    if success:
        print("\nMulti-News dataset download completed successfully!")
    else:
        print("\nMulti-News dataset download failed!") 