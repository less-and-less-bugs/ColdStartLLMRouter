"""
Simple script to download WMT machine translation dataset to data directory
Downloads samples from multiple language pairs with English as source language
"""

import os
import json
import hashlib
import random
from datasets import load_dataset
from collections import defaultdict

def generate_id(text):
    """Generate a unique ID for each sample based on its content"""
    return hashlib.sha256(text.encode()).hexdigest()[:40]

def estimate_difficulty(src_text, tgt_text):
    """
    Estimate translation difficulty based on sentence length and complexity
    Returns: 'easy', 'medium', or 'hard'
    """
    src_len = len(src_text.split())
    tgt_len = len(tgt_text.split())
    
    # Simple heuristic: longer sentences are generally harder
    avg_len = (src_len + tgt_len) / 2
    
    if avg_len < 15:
        return 'easy'
    elif avg_len < 30:
        return 'medium'
    else:
        return 'hard'

def download_wmt_to_data(total_samples=2000):
    """
    Download WMT machine translation dataset samples and save to data/wmt directory
    Selects samples from multiple language pairs with English as source language
    
    Language pairs to include:
    - en-de (English-German)
    - en-fr (English-French)
    - en-es (English-Spanish)
    - en-ru (English-Russian)
    - en-zh (English-Chinese)
    """
    
    # Create wmt directory in data
    output_dir = "/hdd2/lh/agenticrouter_data/data/wmt"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Downloading WMT machine translation datasets...")
    
    all_samples = []
    
    # Define datasets to try in order of preference
    # Using opus_books which is reliable and has multiple language pairs
    datasets_to_try = [
        {
            'name': 'opus_books',
            'lang_pairs': ['en-de', 'en-fr', 'en-es', 'en-ru', 'en-zh'],
            'max_samples_per_pair': 300
        },
        {
            'name': 'europarl_bilingual',
            'lang_pairs': ['en-de', 'en-fr', 'en-es'],
            'max_samples_per_pair': 400
        }
    ]
    
    try:
        # Try datasets in order
        for dataset_config in datasets_to_try:
            dataset_name = dataset_config['name']
            lang_pairs = dataset_config['lang_pairs']
            max_samples = dataset_config['max_samples_per_pair']
            
            print(f"\nTrying {dataset_name} dataset...")
            
            for lang_pair in lang_pairs:
                try:
                    print(f"  Loading {lang_pair}...")
                    dataset = load_dataset(dataset_name, lang_pair, split='train')
                    print(f"    Successfully loaded {len(dataset)} samples")
                    
                    # Limit samples per language pair to avoid too many downloads
                    samples_to_process = min(len(dataset), max_samples)
                    dataset_subset = dataset.select(range(samples_to_process))
                    
                    # Process samples
                    for sample in dataset_subset:
                        if 'translation' in sample:
                            src_lang, tgt_lang = lang_pair.split('-')
                            translation = sample['translation']
                            
                            # Get source and target text
                            src_text = translation.get(src_lang, '').strip()
                            tgt_text = translation.get(tgt_lang, '').strip()
                            
                            # Validate that we have both texts and English is source
                            if src_text and tgt_text and src_lang == 'en':
                                difficulty = estimate_difficulty(src_text, tgt_text)
                                all_samples.append({
                                    'source': src_text,
                                    'target': tgt_text,
                                    'source_lang': src_lang,
                                    'target_lang': tgt_lang,
                                    'difficulty': difficulty,
                                    'dataset': dataset_name
                                })
                    
                    print(f"    Processed {samples_to_process} samples")
                    
                except Exception as e:
                    print(f"    Failed to load {lang_pair}: {e}")
                    continue
            
            # If we have enough samples, break early
            if len(all_samples) >= total_samples * 2:
                print(f"\nCollected enough samples ({len(all_samples)}), stopping...")
                break
        
        if not all_samples:
            raise Exception("No samples could be collected from any dataset source")
        
        print(f"\nCollected {len(all_samples)} total samples")
        
        # Group by difficulty and target language for balanced sampling
        samples_by_difficulty = defaultdict(list)
        for sample in all_samples:
            key = (sample['difficulty'], sample['target_lang'])
            samples_by_difficulty[key].append(sample)
        
        print(f"\nSample distribution by difficulty and target language:")
        for key, samples in sorted(samples_by_difficulty.items()):
            print(f"  {key}: {len(samples)} samples")
        
        # Select balanced samples across difficulty levels and target languages
        selected_samples = []
        
        if len(samples_by_difficulty) > 0:
            # Calculate samples per group, ensuring we get enough samples

            samples_per_group = max(1, total_samples // len(samples_by_difficulty))
            remaining = total_samples % len(samples_by_difficulty)
            
            # First pass: select balanced samples from each group
            for i, (key, samples) in enumerate(samples_by_difficulty.items()):
                count = samples_per_group + (1 if i < remaining else 0)
                if len(samples) >= count:
                    selected = random.sample(samples, count)
                else:
                    # Take all available samples if not enough
                    selected = samples
                selected_samples.extend(selected)
            
            # Second pass: if we don't have enough samples, randomly select more from remaining
            if len(selected_samples) < total_samples:
                remaining_needed = total_samples - len(selected_samples)
                remaining_samples = [s for s in all_samples if s not in selected_samples]
                if remaining_samples:
                    additional = random.sample(remaining_samples, min(remaining_needed, len(remaining_samples)))
                    selected_samples.extend(additional)
        else:
            # Fallback: just randomly select if grouping failed
            selected_samples = random.sample(all_samples, min(total_samples, len(all_samples)))
        
        # Shuffle final selection and ensure exact count
        random.shuffle(selected_samples)
        selected_samples = selected_samples[:total_samples]
        
        print(f"\nSelected {len(selected_samples)} samples for final dataset")
        
        # Format the data similar to multi_news format
        formatted_data = []
        target_lang_counts = defaultdict(int)
        
        for sample in selected_samples:
            src_text = sample['source']
            tgt_text = sample['target']
            target_lang = sample['target_lang']
            target_lang_counts[target_lang] += 1
            
            # Create instruction based on target language
            lang_names = {
                'de': 'German',
                'fr': 'French',
                'es': 'Spanish',
                'ru': 'Russian',
                'zh': 'Chinese',
                'ja': 'Japanese',
                'ro': 'Romanian'
            }
            lang_name = lang_names.get(target_lang, target_lang.upper())
            
            formatted_item = {
                "input": src_text,
                "length": len(src_text),
                "dataset": "wmt",
                "language": "en",
                "target_language": target_lang,
                "all_classes": None,
                "_id": generate_id(src_text + tgt_text),
                "instruction": f"Please translate the following English text to {lang_name}.",
                "output": tgt_text,
                "difficulty": sample['difficulty']
            }
            formatted_data.append(formatted_item)
        
        # Save processed data
        output_file = os.path.join(output_dir, "wmt.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(formatted_data)} samples to {output_file}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Total samples: {len(formatted_data)}")
        print(f"  Target language distribution:")
        for lang, count in sorted(target_lang_counts.items()):
            print(f"    {lang}: {count} samples")
        
        difficulty_counts = defaultdict(int)
        for item in formatted_data:
            difficulty_counts[item['difficulty']] += 1
        print(f"  Difficulty distribution:")
        for diff, count in sorted(difficulty_counts.items()):
            print(f"    {diff}: {count} samples")
        
        # Show sample structure
        if len(formatted_data) > 0:
            print(f"\nSample data structure:")
            sample = formatted_data[0]
            print("\nExample item:")
            for key, value in sample.items():
                if key in ['input', 'output']:
                    print(f"\n{key} (first 200 chars):")
                    print(value[:200] + "..." if len(str(value)) > 200 else value)
                else:
                    print(f"\n{key}:")
                    print(value)
        
    except Exception as e:
        print(f"Error downloading/processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Note: This script downloads machine translation datasets from HuggingFace")
    print("It may take some time depending on your internet connection.")
    success = download_wmt_to_data(total_samples=1000)
    if success:
        print("\nWMT dataset download completed successfully!")
    else:
        print("\nWMT dataset download failed!")

