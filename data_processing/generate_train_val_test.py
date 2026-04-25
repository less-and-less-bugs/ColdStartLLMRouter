"""
Generate train/validation/test splits for all datasets. All datasets share the same csv file.
"""

import pandas as pd
import numpy as np
from utils import loadjson, get_embedding, savepkl, loadpkl
import yaml
import os
import glob
from sklearn.model_selection import train_test_split

def process_mmlu_redux_data(path, test_size=20, train_val_split=0.8):
    """
    Process MMLU-Redux data from multiple JSON files with train/val/test split
    """
    train_data = []
    val_data = []
    test_data = []
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(path, "*.json"))
    
    for json_file in json_files:
        if "mmlu_redux_all" in json_file:  # Skip the combined file
            continue
            
        data = loadjson(json_file)
        subject = os.path.basename(json_file).replace('.json', '')
        
        # First get test samples
        test_samples = data[:test_size]
        remaining_samples = data[test_size:]
        
        # Split remaining samples into train and val
        if remaining_samples:
            train_samples, val_samples = train_test_split(
                remaining_samples, 
                train_size=train_val_split,
                random_state=42
            )
        else:
            train_samples, val_samples = [], []
        
        # Process test samples
        for item in test_samples:
            query = f"Q: {item['question'].strip()}\n(A) {item['choices'][0]} (B) {item['choices'][1]} (C) {item['choices'][2]} (D) {item['choices'][3]}\n"
            answer_map = ['(A)', '(B)', '(C)', '(D)']
            ground_truth = answer_map[item['answer']]
            
            test_data.append({
                'task_id': "mmlu_redux",
                'sub_task': subject,
                'query': query + "\nPlease wrap your final answer with tag <answer> your answer </answer>",
                'ground_truth': ground_truth,
                'metric': 'mmluredux',
                'task_description': 'The MMLU-Redux dataset is a comprehensive evaluation benchmark designed to test knowledge across various academic and professional domains. It presents multiple-choice questions that assess understanding and expertise in specific subject areas.',
                'split': 'test'
            })
        
        # Process train samples
        for item in train_samples:
            query = f"Q: {item['question'].strip()}\n(A) {item['choices'][0]} (B) {item['choices'][1]} (C) {item['choices'][2]} (D) {item['choices'][3]}\n"
            answer_map = ['(A)', '(B)', '(C)', '(D)']
            ground_truth = answer_map[item['answer']]
            
            train_data.append({
                'task_id': "mmlu_redux",
                'sub_task': subject,
                'query': query + "\nPlease wrap your final answer with tag <answer> your answer </answer>",
                'ground_truth': ground_truth,
                'metric': 'mmluredux',
                'task_description': 'The MMLU-Redux dataset is a comprehensive evaluation benchmark designed to test knowledge across various academic and professional domains. It presents multiple-choice questions that assess understanding and expertise in specific subject areas.',
                'split': 'train'
            })
            
        # Process validation samples
        for item in val_samples:
            query = f"Q: {item['question'].strip()}\n(A) {item['choices'][0]} (B) {item['choices'][1]} (C) {item['choices'][2]} (D) {item['choices'][3]}\n"
            answer_map = ['(A)', '(B)', '(C)', '(D)']
            ground_truth = answer_map[item['answer']]
            
            val_data.append({
                'task_id': "mmlu_redux",
                'sub_task': subject,
                'query': query + "\nPlease wrap your final answer with tag <answer> your answer </answer>",
                'ground_truth': ground_truth,
                'metric': 'mmluredux',
                'task_description': 'The MMLU-Redux dataset is a comprehensive evaluation benchmark designed to test knowledge across various academic and professional domains. It presents multiple-choice questions that assess understanding and expertise in specific subject areas.',
                'split': 'val'
            })
    
    return train_data, val_data, test_data

def generate_train_val_test_splits(output_dir='data', test_size=400, train_val_split=0.8, mmlu_test_size=20):
    """
    Generate train/validation/test splits for all datasets.
    
    Parameters:
    output_dir (str): Directory to save the output files
    test_size (int): Number of samples for test set (per task)
    train_val_split (float): Ratio for splitting remaining data into train/val
    mmlu_test_size (int): Number of test samples per MMLU subject
    """
    
    # Initialize result DataFrames for each split
    columns = [
        'task_id', 'sub_task', 'query', 'ground_truth', 'metric',
        'task_description', 'split'
    ]
    
    all_data = []

    # Define dataset paths and corresponding task names with descriptions
    dataset_configs = [
        {
            'task_name': 'alpaca_data',
            'path': 'data/alpaca_data/alpaca_data.json',
            'format': 'json',
            'query_fields': ['instruction', 'input'],
            'ground_truth_field': 'output',
            'metric': 'f1_score',
            'task_description': 'The Alpaca dataset is designed for instruction-following tasks, where the model is required to generate coherent and contextually appropriate responses to given instructions or prompts. It focuses on understanding diverse user requests and providing informative and accurate outputs based on those instructions.'
        },
        {
            'task_name': 'GSM8K',
            'path': 'data/GSM8K/GSM8K.json',
            'format': 'json',
            'query_fields': ['instruction', 'input'],
            'ground_truth_field': 'answer',
            'metric': 'GSM8K',
            'task_description': 'The GSM8K dataset is tailored for mathematical problem-solving tasks. It consists of natural language math problems that require the model to comprehend the problem statement, apply the correct mathematical operations, and provide the solution. The primary challenge lies in both parsing complex language and performing accurate calculations.'
        },
        {
            'task_name': 'multi_news',
            'path': 'data/multi_news/multi_news.json',
            'format': 'json',
            'query_fields': ['instruction', 'input'],
            'ground_truth_field': 'output',
            'metric': 'f1_score',
            'task_description': 'The Multi-News dataset is aimed at text summarization tasks. It contains multiple news articles on the same topic, and the model\'s objective is to generate a concise and comprehensive summary that integrates information from all the articles. The challenge is to distill key points while maintaining coherence and avoiding redundancy.'
        },
        {
            'task_name': 'SQUAD',
            'path': 'data/SQUAD/SQUAD.parquet',
            'format': 'parquet',
            'query_field': 'question',
            'ground_truth_field': 'answers',
            'ground_truth_subfield': 'text',
            'ground_truth_index': 0,
            'metric': 'f1_score',
            'task_description': 'The SQuAD dataset is focused on question-answering tasks, where the model is given a passage of text and needs to extract or generate a precise answer to a question based on the content of the passage. The dataset emphasizes comprehension, retrieval of relevant information, and concise answer generation.'
        },
        {
            'task_name': 'MBPP',
            'path': 'data/mbpp/mbpp_all.json',
            'format': 'json',
            'query_fields': ["text", "test_list"],
            'ground_truth_field': 'code',
            'metric': 'pass@1',
            'task_description': 'The MBPP dataset is designed to measure the ability to synthesize short Python programs from natural language descriptions.'
        },
        {
            'task_name': 'mmlu_redux',
            'path': 'data/mmlu_redux',
            'format': 'mmlu',  # Special format for MMLU-Redux
            'metric': 'mmluredux',
            'task_description': 'The MMLU-Redux dataset is a comprehensive evaluation benchmark designed to test knowledge across various academic and professional domains. It presents multiple-choice questions that assess understanding and expertise in specific subject areas.'
        }
    ]

    # Process each dataset
    for config in dataset_configs:
        try:
            if config['format'] == 'mmlu':
                # Special processing for MMLU-Redux
                train_data, val_data, test_data = process_mmlu_redux_data(
                    config['path'], 
                    test_size=mmlu_test_size,
                    train_val_split=train_val_split
                )
                all_data.extend(train_data + val_data + test_data)
                
            elif config['format'] == 'json':
                data = loadjson(config['path'])
                
                # Get test samples
                test_samples = data[:test_size]
                remaining_samples = data[test_size:test_size+100]  # Take at most 100 samples for train/val
                
                # Split remaining samples into train and val
                if remaining_samples:
                    train_samples, val_samples = train_test_split(
                        remaining_samples,
                        train_size=train_val_split,
                        random_state=42
                    )
                else:
                    train_samples, val_samples = [], []

                # Process all splits
                for split, samples in [('test', test_samples), ('train', train_samples), ('val', val_samples)]:
                    for item in samples:
                        if isinstance(config['query_fields'], list):
                            if config['task_name'] == 'MBPP':
                                query = "You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{test_list[0]}\n{test_list[1]}\n{test_list[2]}\nPlease wrap your final answer with tag <code> your codes </code>".format(text=item['text'], test_list=item['test_list'])
                            else:
                                query = ''.join([item[field] for field in config['query_fields']])
                                ground_truth = item[config['ground_truth_field']]
                        else:
                            query = item[config['query_fields']]
                            ground_truth = item[config['ground_truth_field']]

                        all_data.append({
                            'task_id': config['task_name'],
                            'sub_task': None,
                            'query': query,
                            'ground_truth': ground_truth,
                            'metric': config['metric'],
                            'task_description': config['task_description'],
                            'split': split
                        })

            elif config['format'] == 'parquet':
                data = pd.read_parquet(config['path'])
                
                # Get test samples
                test_samples = data[:test_size]
                remaining_samples = data[test_size:test_size+100]  # Take at most 100 samples for train/val
                
                # Split remaining samples into train and val
                if len(remaining_samples) > 0:
                    train_samples, val_samples = train_test_split(
                        remaining_samples,
                        train_size=train_val_split,
                        random_state=42
                    )
                else:
                    train_samples, val_samples = pd.DataFrame(), pd.DataFrame()

                # Process all splits
                for split, samples in [('test', test_samples), ('train', train_samples), ('val', val_samples)]:
                    for item in samples.itertuples():
                        query = getattr(item, config['query_field'])
                        
                        if 'ground_truth_subfield' in config:
                            ground_truth_container = getattr(item, config['ground_truth_field'])
                            ground_truth = ground_truth_container[config['ground_truth_subfield']][config['ground_truth_index']]
                        else:
                            ground_truth = getattr(item, config['ground_truth_field'])

                        all_data.append({
                            'task_id': config['task_name'],
                            'sub_task': None,
                            'query': query,
                            'ground_truth': ground_truth,
                            'metric': config['metric'],
                            'task_description': config['task_description'],
                            'split': split
                        })

        except Exception as e:
            print(f"Error processing {config['task_name']}: {str(e)}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=columns)
    
    # Save each split to a separate file
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        output_file = os.path.join(output_dir, f'unified_qa_data_{split}.csv')
        split_df.to_csv(output_file, index=False)
        print(f"Saved {len(split_df)} {split} samples to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate splits
    df = generate_train_val_test_splits(
        output_dir='data',
        test_size=400,
        train_val_split=0.8,
        mmlu_test_size=20
    )
    
    # Print statistics
    print("\nDataset Statistics:")
    print("==================")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"\n{split.upper()} Set:")
        print(f"Total samples: {len(split_df)}")
        print("\nSamples per task:")
        task_counts = split_df['task_id'].value_counts()
        for task, count in task_counts.items():
            print(f"{task}: {count}")
            if task == 'mmlu_redux':
                subject_counts = split_df[split_df['task_id'] == 'mmlu_redux']['sub_task'].value_counts()
                print("Samples per subject:")
                for subject, subj_count in subject_counts.items():
                    print(f"  - {subject}: {subj_count}") 