"""
Simple script to download MMLU-Redux dataset to data directory
"""

import os
from datasets import load_dataset
import json

SUBJECTS = ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering', 'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy', 'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history', 'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law']

def download_mmlu_redux_to_data():
    """
    Download MMLU-Redux dataset and save to data/mmlu_redux directory
    """
    
    # Create mmlu_redux directory in data
    output_dir = "data/mmlu_redux"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading MMLU-Redux dataset...")
    
    try:
        all_data = []
        
        # Download each subject separately
        for subject in SUBJECTS:
            print(f"\nProcessing subject: {subject}")
            
            # Load the dataset for this subject
            dataset = load_dataset("edinburgh-dawg/mmlu-redux", subject)
            print(f"Successfully loaded {subject} dataset")
            
            # Process each split (train, validation, test)
            for split_name, split_data in dataset.items():
                print(f"  Processing split: {split_name} ({len(split_data)} samples)")
                split_list = list(split_data)
                
                # Add split and subject information to each sample
                for sample in split_list:
                    sample_with_info = sample.copy()
                    sample_with_info['split'] = split_name
                    sample_with_info['subject'] = subject
                    all_data.append(sample_with_info)
                
            # Save subject-specific data
            subject_data = [item for item in all_data if item['subject'] == subject]
            subject_file = os.path.join(output_dir, f"{subject}.json")
            with open(subject_file, 'w', encoding='utf-8') as f:
                json.dump(subject_data, f, ensure_ascii=False, indent=2)
            print(f"  Saved {len(subject_data)} samples to {subject_file}")
        
        # Save complete dataset
        all_data_file = os.path.join(output_dir, "mmlu_redux_all.json")
        with open(all_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved total {len(all_data)} samples to {all_data_file}")
        
        # Print summary
        print("\nDataset Summary:")
        subjects_count = {}
        splits_count = {}
        for item in all_data:
            subjects_count[item['subject']] = subjects_count.get(item['subject'], 0) + 1
            splits_count[item['split']] = splits_count.get(item['split'], 0) + 1
            
        print("\nSamples per subject:")
        for subject, count in subjects_count.items():
            print(f"  {subject}: {count}")
            
        print("\nSamples per split:")
        for split, count in splits_count.items():
            print(f"  {split}: {count}")
        
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
    print("Note: You need to login using `huggingface-cli login` to access this dataset")
    success = download_mmlu_redux_to_data()
    if success:
        print("\nMMLU-Redux dataset download completed successfully!")
    else:
        print("\nMMLU-Redux dataset download failed!") 