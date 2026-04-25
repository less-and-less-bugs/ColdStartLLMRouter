"""
Generate train/val/test splits for all datasets according to specific requirements. We utilzied in our work.
"""

import pandas as pd
import numpy as np
from utils import loadjson
import os
import glob
from sklearn.model_selection import train_test_split

def process_mmlu_redux_data(path, total_samples=2000):
    """
    Process MMLU-Redux data to get balanced samples across subjects
    """
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(path, "*.json"))
    json_files = [f for f in json_files if "mmlu_redux_all" not in f]
    
    # Calculate samples per subject
    n_subjects = len(json_files)
    samples_per_subject = total_samples // n_subjects
    remaining_samples = total_samples % n_subjects
    
    all_data = []
    for json_file in json_files:
        data = loadjson(json_file)
        subject = os.path.basename(json_file).replace('.json', '')
        
        # Add extra sample if we have remaining samples to distribute
        current_samples = samples_per_subject + (1 if remaining_samples > 0 else 0)
        remaining_samples = max(0, remaining_samples - 1)
        
        # Process samples for this subject
        for item in data[:current_samples]:
            query = f"Q: {item['question'].strip()}\n(A) {item['choices'][0]} (B) {item['choices'][1]} (C) {item['choices'][2]} (D) {item['choices'][3]}\n"
            answer_map = ['(A)', '(B)', '(C)', '(D)']
            ground_truth = answer_map[item['answer']]
            
            all_data.append({
                'task_id': "mmlu_redux",
                'sub_task': subject,
                'query': query + "\nPlease wrap your final answer with tag <answer> your answer </answer>",
                'ground_truth': ground_truth,
                'metric': 'mmluredux',
                'task_description': 'The MMLU-Redux dataset is a comprehensive evaluation benchmark designed to test knowledge across various academic and professional domains.'
            })
    
    return pd.DataFrame(all_data)

def process_regular_dataset(data, config, sample_size=2000):
    """Helper function to process regular sized datasets with unified format"""
    processed_data = []
    
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
    
    for item in data.to_dict('records'):
        if isinstance(config['query_fields'], list):
            if config['task_name'] == 'mbpp':
                # Example for few-shot learning
                example_text = "Write a function to find the similar elements from the given two tuple lists."
                example_tests = [
                    "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                    "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                    "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
                ]
                example_code = "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)"
                
                # Format test cases for current task
                test_cases = '\n'.join(item['test_list'][:3]) if len(item['test_list']) >= 3 else '\n'.join(item['test_list'])
                
                query = """You are an expert Python programmer. Here is an example:

Task: {example_text}
Tests:
{example_tests}

<code>
{example_code}
</code>

Now, here is your task:
Task: {text}
Tests:
{tests}

Please output your code wrapped in <code> and </code> tags, following the example format above. Only output the code without any other text.""".format(
                    example_text=example_text,
                    example_tests='\n'.join(example_tests),
                    example_code=example_code,
                    text=item['text'],
                    tests=test_cases
                )
                ground_truth = item[config['ground_truth_field']]
            elif config['task_name'] == 'medmcqa' or config['task_name'] == 'legalbench':
                query = ''.join([item[field] for field in config['query_fields']]) + "\nPlease wrap your final answer with tag <answer> your answer </answer>."
                ground_truth = item[config['ground_truth_field']]
            else:
                query = ''.join([item[field] for field in config['query_fields']])
                ground_truth = item[config['ground_truth_field']]
        else:
            query = item[config['query_fields']]
            ground_truth = item[config['ground_truth_field']]
            
        processed_data.append({
            'task_id': config['task_name'],
            'sub_task': None,
            'query': query,
            'ground_truth': ground_truth,
            'metric': config['metric'],
            'task_description': config['task_description']
        })
    
    return pd.DataFrame(processed_data)

def process_parquet_dataset(data, config, sample_size=2000):
    """Helper function to process parquet datasets with unified format"""
    processed_data = []
    
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
    
    for item in data.itertuples():
        query = getattr(item, config['query_field'])
        
        if 'ground_truth_subfield' in config:
            ground_truth_container = getattr(item, config['ground_truth_field'])
            ground_truth = ground_truth_container[config['ground_truth_subfield']][config['ground_truth_index']]
        else:
            ground_truth = getattr(item, config['ground_truth_field'])
            
        processed_data.append({
            'task_id': config['task_name'],
            'sub_task': None,
            'query': query + "\nPlease wrap your final answer with tag <answer> your answer </answer>.",
            'ground_truth': ground_truth,
            'metric': config['metric'],
            'task_description': config['task_description']
        })
    
    return pd.DataFrame(processed_data)

def split_dataset(df, train_ratio=0.7, val_ratio=0.1):
    """Split dataset into train/val/test sets"""
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio/(1-train_ratio), random_state=42)
    return train_df, val_df, test_df

def save_splits(df_dict, output_dir):
    """Save train/val/test splits to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    for split_name, df in df_dict.items():
        output_file = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} samples to {output_file}")

def generate_dataset_splits(data_dir='data'):
    """Generate train/val/test splits for all datasets"""
    
    # Define dataset configurations
    dataset_configs = [
        # {
        #     'task_name': 'alpaca_data',
        #     'path': 'data/alpaca_data/alpaca_data.json',
        #     'format': 'json',
        #     'query_fields': ['instruction', 'input'],
        #     'ground_truth_field': 'output',
        #     'metric': 'f1_score',
        #     'task_description': 'The Alpaca dataset is designed for instruction-following tasks, where the model is required to generate coherent and contextually appropriate responses to given instructions or prompts.'
        # },
        # {
        #     'task_name': 'GSM8K',
        #     'path': 'data/GSM8K/GSM8K_train.json',
        #     'test_path': 'data/GSM8K/GSM8K_test.json',
        #     'format': 'json',
        #     'query_fields': ['instruction', 'input'],
        #     'ground_truth_field': 'answer',
        #     'metric': 'GSM8K',
        #     'task_description': 'The GSM8K dataset is tailored for mathematical problem-solving tasks.'
        # },
        # {
        #     'task_name': 'multi_news',
        #     'path': 'data/multi_news/multi_news.json',
        #     'format': 'json',
        #     'query_fields': ['instruction', 'input'],
        #     'ground_truth_field': 'output',
        #     'metric': 'f1_score',
        #     'task_description': 'The Multi-News dataset is aimed at text summarization tasks.'
        # },
        # {
        #     'task_name': 'SQUAD',
        #     'path': 'data/SQUAD/SQUAD.parquet',
        #     'format': 'parquet',
        #     'query_field': 'question',
        #     'ground_truth_field': 'answers',
        #     'ground_truth_subfield': 'text',
        #     'ground_truth_index': 0,
        #     'metric': 'f1_score',
        #     'task_description': 'The SQuAD dataset is focused on question-answering tasks.'
        # },
        # {
        #     'task_name': 'mbpp',
        #     'path': f'{data_dir}/data/mbpp/mbpp_all.json',
        #     'format': 'json',
        #     'query_fields': ["text", "test_list"],
        #     'ground_truth_field': 'code',
        #     'metric': 'pass@1',
        #     'task_description': 'The MBPP dataset is designed to measure the ability to synthesize short Python programs from natural language descriptions.'
        # },
        # {
        #     'task_name': 'wmt',
        #     'path': f'{data_dir}/data/wmt/wmt.json',
        #     'format': 'json',
        #     'query_fields': ["instruction", "input"],
        #     'ground_truth_field': 'output',
        #     'metric': 'bleu',
        #     'task_description': 'The WMT dataset is a machine translation dataset that contains parallel text data for various language pairs.'
        # },
        {
            'task_name': 'legalbench',
            'path': f'{data_dir}/data/legalbench/legalbench.json',
            'format': 'json',
            'query_fields': ["instruction", "input"],
            'ground_truth_field': 'answer',
            'metric': 'f1_score',
            'task_description': 'The LegalBench dataset is a collection of legal tasks that are designed to measure the ability of a model to reason about legal concepts and apply them to real-world legal scenarios.'
        },
        {
            'task_name': 'medmcqa',
            'path': f'{data_dir}/data/medmcqa/medmcqa.json',
            'format': 'json',
            'query_fields': ["instruction", "input"],
            'ground_truth_field': 'output',
            'metric': 'f1_score',
            'task_description': 'The MedMCQA dataset is a collection of medical questions and answers that are designed to measure the ability of a model to reason about medical concepts and apply them to real-world medical scenarios.'
        },
    ]
    # dataset_configs = [ {
    #         'task_name': 'SQUAD',
    #         'path': 'data/SQUAD/SQUAD.parquet',
    #         'format': 'parquet',
    #         'query_field': 'question',
    #         'ground_truth_field': 'answers',
    #         'ground_truth_subfield': 'text',
    #         'ground_truth_index': 0,
    #         'metric': 'f1_score',
    #         'task_description': 'The SQuAD dataset is focused on question-answering tasks.'
    #     }
    # ]
    
    # Process each dataset
    for config in dataset_configs:
        try:
            print(f"\nProcessing {config['task_name']}...")
            
            if config['task_name'] == 'GSM8K':
                # Special processing for GSM8K
                gsm8k_test = loadjson(config['test_path'])
                gsm8k_train = loadjson(config['path'])
                
                # Process test data for val and test (600 samples, 1:2 split)
                test_df = process_regular_dataset(pd.DataFrame(gsm8k_test[:600]), config, sample_size=600)
                val_df, test_df = train_test_split(test_df, train_size=1/3, random_state=42)
                
                # Process train data (1400 samples)
                train_df = process_regular_dataset(pd.DataFrame(gsm8k_train[:1400]), config, sample_size=1400)
                
            elif config['task_name'] == 'mbpp':
                # Special processing for MBPP: first 400 -> test, next 100 -> val, last 500 -> train
                data = pd.DataFrame(loadjson(config['path']))
         
                # Process test set (first 400 samples)
                test_df = process_regular_dataset(data.iloc[:400], config, sample_size=400)
                
                # Process val set (next 100 samples: 400-500)
                val_df = process_regular_dataset(data.iloc[400:500], config, sample_size=100)
                
                # Process train set (last 500 samples: 500-1000)
                train_df = process_regular_dataset(data.iloc[500:1000], config, sample_size=500)
                
            elif config['task_name'] == 'wmt':
                # Special processing for WMT: first 400 -> test, next 100 -> val, last 500 -> train
                data = pd.DataFrame(loadjson(config['path']))
                
                # Use first 1000 samples: first 400 -> test, next 100 -> val, last 500 -> train
                # Process test set (first 400 samples)
                test_df = process_regular_dataset(data.iloc[:400], config, sample_size=400)
                
                # Process val set (next 100 samples: 400-500)
                val_df = process_regular_dataset(data.iloc[400:500], config, sample_size=100)
                
                # Process train set (next 500 samples: 500-1000)
                train_df = process_regular_dataset(data.iloc[500:1000], config, sample_size=500)
            
            elif config['task_name'] == 'medmcqa' or config['task_name'] == 'legalbench':
                # Special processing for MedMCQA: first 400 -> test, next 100 -> val, last 500 -> train
                data = pd.DataFrame(loadjson(config['path']))
                
                # Use first 1000 samples: first 400 -> test, next 100 -> val, last 500 -> train
                # Process test set (first 400 samples)
                test_df = process_regular_dataset(data.iloc[:400], config, sample_size=400)
                
                # Process val set (next 100 samples: 400-500)
                val_df = process_regular_dataset(data.iloc[400:500], config, sample_size=100)
                
                # Process train set (next 500 samples: 500-1000)
                train_df = process_regular_dataset(data.iloc[500:1000], config, sample_size=500)
            

                
            else:
                if config['format'] == 'json':
                    data = pd.DataFrame(loadjson(config['path']))
                    # Sample 2000 for other datasets
                    processed_df = process_regular_dataset(data, config, sample_size=2000)
                    
                elif config['format'] == 'parquet': # SQUAD is in parquet format. QA Taks.
                    data = pd.read_parquet(config['path'])
                    processed_df = process_parquet_dataset(data, config, sample_size=2000)
                
                # Split into train/val/test (7:1:2)
                train_df, val_df, test_df = split_dataset(processed_df)
            
            # Save splits
            save_splits(
                {'train': train_df, 'val': val_df, 'test': test_df},
                f"{data_dir}/data/{config['task_name']}"
            )
            
        except Exception as e:
            print(f"Error processing {config['task_name']}: {str(e)}")
            continue
    
    # Process MMLU-Redux separately (2000 samples total, balanced across subjects)
    print("\nProcessing MMLU-Redux...")
    try:
        mmlu_df = process_mmlu_redux_data("data/mmlu_redux", total_samples=2000)
        mmlu_train, mmlu_val, mmlu_test = split_dataset(mmlu_df)
        save_splits(
            {'train': mmlu_train, 'val': mmlu_val, 'test': mmlu_test},
            'data/mmlu_redux'
        )
    except Exception as e:
        print(f"Error processing MMLU-Redux: {str(e)}")

def print_split_statistics(data_dir='data'):
    """Print statistics for all dataset splits"""
    datasets = [
        ('MMLU-Redux', 'data/mmlu_redux'),
        ('Alpaca', f'{data_dir}/data/alpaca_data'),
        ('Multi-News', f'{data_dir}/data/multi_news'),
        ('SQUAD', f'{data_dir}/data/SQUAD'),
        ('mbpp', f'{data_dir}/data/mbpp'),
        ('GSM8K', f'{data_dir}/data/GSM8K'),
        ('WMT', f'{data_dir}/data/wmt'),
        ('LegalBench', f'{data_dir}/data/legalbench'),
        ('MedMCQA', f'{data_dir}/data/medmcqa')
    ]
    
    print("\nDataset Split Statistics:")
    print("=" * 50)
    
    for dataset_name, path in datasets:
        print(f"\n{dataset_name}:")
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(path, f"{split}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"  {split}: {len(df)} samples")
                
                if dataset_name == 'MMLU-Redux':
                    subjects = df['sub_task'].value_counts()
                    print(f"  {split} subject distribution:")
                    for subject, count in subjects.items():
                        print(f"    - {subject}: {count}")

if __name__ == "__main__":
    generate_dataset_splits(data_dir='/hdd2/lh/agenticrouter_data')
    print_split_statistics(data_dir='/hdd2/lh/agenticrouter_data') 