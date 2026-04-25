"""
Simple script to download MedMCQA dataset test split to data directory
MedMCQA is a large-scale Multiple-Choice Question Answering dataset 
for medical domain questions from AIIMS & NEET PG entrance exams
"""

import os
import json
import random
from datasets import load_dataset

def format_answer_option(cop):
    """
    Convert cop (correct option) to option letter
    cop: '0a', '1b', '2c', '3d' format
    Returns: 'A', 'B', 'C', or 'D'
    """
    if isinstance(cop, str):
        # Handle format like '0a', '1b', '2c', '3d'
        if len(cop) >= 2:
            option_char = cop[-1].lower()
            if option_char == 'a':
                return 'A'
            elif option_char == 'b':
                return 'B'
            elif option_char == 'c':
                return 'C'
            elif option_char == 'd':
                return 'D'
    elif isinstance(cop, int):
        # Handle numeric format: 0=A, 1=B, 2=C, 3=D
        if cop == 0:
            return 'A'
        elif cop == 1:
            return 'B'
        elif cop == 2:
            return 'C'
        elif cop == 3:
            return 'D'
    
    return 'A'  # Default fallback

def download_medmcqa_to_data():
    """
    Download MedMCQA dataset test split and save to data/medmcqa directory
    """
    
    # Create medmcqa directory in data
    output_dir = "/hdd2/lh/agenticrouter_data/data/medmcqa"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Downloading MedMCQA dataset...")
    
    try:
        # Load the test split
        dataset = load_dataset("openlifescienceai/medmcqa", split='test')
        print("Successfully loaded MedMCQA test dataset")
        
        split_data = list(dataset)
        print(f"\nProcessing test split ({len(split_data)} samples)")
        
        # Format the data
        formatted_data = []
        for item in split_data:
            question = item.get('question', '').strip()
            opa = item.get('opa', '').strip()
            opb = item.get('opb', '').strip()
            opc = item.get('opc', '').strip()
            opd = item.get('opd', '').strip()
            cop = item.get('cop', '')
            
            # Validate that we have question and options
            if question and opa and opb and opc and opd:
                correct_answer = format_answer_option(cop)
                
                # Create formatted question text
                question_text = f"{question}\n\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"
                
                formatted_item = {
                    "input": question_text,
                    "instruction": "Please answer the following multiple-choice medical question. Select only the correct option letter (A, B, C, or D).",
                    "output": correct_answer
                }
                formatted_data.append(formatted_item)
        
        # Randomly sample 1000 samples
        if len(formatted_data) > 1000:
            formatted_data = random.sample(formatted_data, 1000)
            print(f"\nRandomly selected 1000 samples from {len(split_data)} total samples")
        else:
            print(f"\nUsing all {len(formatted_data)} samples (less than 1000)")
        
        # Save to file
        output_file = os.path.join(output_dir, "medmcqa.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(formatted_data)} test samples to {output_file}")
        
        # Show sample structure
        if len(formatted_data) > 0:
            print(f"\nSample data structure:")
            sample = formatted_data[0]
            print("\nExample item:")
            for key, value in sample.items():
                print(f"\n{key}:")
                if key == 'input' and len(str(value)) > 300:
                    print(value[:300] + "...")
                else:
                    print(value)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Note: This script downloads MedMCQA test split from HuggingFace")
    print("It may take some time depending on your internet connection.")
    success = download_medmcqa_to_data()
    if success:
        print("\nMedMCQA dataset download completed successfully!")
    else:
        print("\nMedMCQA dataset download failed!")
