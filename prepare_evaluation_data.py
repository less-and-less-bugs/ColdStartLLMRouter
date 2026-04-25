"""
Prepare evaluation data - sample 150 rows and save for human evaluation
"""
import pandas as pd
import json
import random
from pathlib import Path

# Set random seed
random.seed(42)

def load_all_data():
    """Load data from all model CSV files in alpaca_data and GSM8K directories"""
    data_dir = Path("/hdd2/lh/agenticrouter_data/data")
    all_data = []
    
    datasets = ["alpaca_data", "GSM8K"]
    
    for dataset in datasets:
        dataset_dir = data_dir / dataset
        csv_files = list(dataset_dir.glob("*.csv"))
        
        # Exclude summary and split files
        csv_files = [f for f in csv_files if f.name not in ["performance_summary.csv", "test.csv", "train.csv", "val.csv"]]
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Only keep rows with both effect and llm_judge_effect
                df = df.dropna(subset=['effect', 'llm_judge_effect'])
                df['source_file'] = csv_file.name
                df['dataset'] = dataset
                all_data.append(df)
                print(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal rows loaded: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()

def sample_and_save(n=150):
    """Sample n rows and save for evaluation"""
    print("Loading data...")
    df = load_all_data()
    
    if len(df) < n:
        print(f"Warning: Only {len(df)} rows available, using all of them")
        sampled_df = df
    else:
        sampled_df = df.sample(n=n, random_state=42)
    
    print(f"\nSampled {len(sampled_df)} rows")
    
    # Prepare data for evaluation
    evaluation_data = []
    for idx, row in sampled_df.iterrows():
        item = {
            'id': idx,
            'dataset': row['dataset'],
            'model_name': row.get('model_name', 'unknown'),
            'question': str(row['query']),
            'ground_truth': str(row['ground_truth']),
            'answer': str(row['response']),
            'traditional_metric': float(row['effect']),
            'llm_judge_score': float(row['llm_judge_effect']),
            'metric_type': str(row.get('metric', 'unknown'))
        }
        evaluation_data.append(item)
    
    # Save to JSON file
    output_file = "evaluation_samples_150.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(evaluation_data)} samples to {output_file}")
    print(f"Dataset distribution:")
    print(sampled_df['dataset'].value_counts())
    print(f"\nModel distribution:")
    print(sampled_df['model_name'].value_counts())
    
    return output_file

if __name__ == "__main__":
    sample_and_save(150)

