"""
Get the size of each dataset
"""

import pandas as pd
from utils import loadjson
import os
import glob

"""
alpaca_data: 51942

GSM8K: 1319 (test split) 7473 (train split)

multi_news: 5622

SQUAD: 10570

MBPP: 964

mmlu_redux: 3000
"""
def get_dataset_sizes():
    """
    Get the size of each dataset
    """
    dataset_configs = [
        {
            'task_name': 'alpaca_data',
            'path': 'data/alpaca_data/alpaca_data.json',
            'format': 'json'
        },
        {
            'task_name': 'GSM8K',
            'path': 'data/GSM8K/GSM8K.json',
            'format': 'json'
        },
        {
            'task_name': 'multi_news',
            'path': 'data/multi_news/multi_news.json',
            'format': 'json'
        },
        {
            'task_name': 'SQUAD',
            'path': 'data/SQUAD/SQUAD.parquet',
            'format': 'parquet'
        },
        {
            'task_name': 'MBPP',
            'path': 'data/mbpp/mbpp_all.json',
            'format': 'json'
        },
        {
            'task_name': 'mmlu_redux',
            'path': 'data/mmlu_redux',
            'format': 'mmlu'
        }
    ]

    sizes = {}
    
    for config in dataset_configs:
        try:
            if config['format'] == 'mmlu':
                # Get all JSON files in the directory
                json_files = glob.glob(os.path.join(config['path'], "*.json"))
                total_size = 0
                subject_sizes = {}
                
                for json_file in json_files:
                    if "mmlu_redux_all" in json_file:  # Skip the combined file
                        continue
                    
                    data = loadjson(json_file)
                    subject = os.path.basename(json_file).replace('.json', '')
                    subject_sizes[subject] = len(data)
                    total_size += len(data)
                
                sizes[config['task_name']] = {
                    'total': total_size,
                    'subjects': subject_sizes
                }
                
            elif config['format'] == 'json':
                data = loadjson(config['path'])
                sizes[config['task_name']] = len(data)
                
            elif config['format'] == 'parquet':
                data = pd.read_parquet(config['path'])
                sizes[config['task_name']] = len(data)
                
        except Exception as e:
            print(f"Error processing {config['task_name']}: {str(e)}")
            sizes[config['task_name']] = "Error: " + str(e)
            continue
    
    return sizes

if __name__ == "__main__":
    sizes = get_dataset_sizes()
    
    print("\n数据集大小统计：")
    print("=" * 50)
    
    for task_name, size in sizes.items():
        if task_name == 'mmlu_redux':
            print(f"\n{task_name}:")
            print(f"总样本数: {size['total']}")
            print("\n各科目样本数:")
            for subject, subject_size in size['subjects'].items():
                print(f"  - {subject}: {subject_size}")
        else:
            print(f"\n{task_name}: {size}")
    
    print("\n" + "=" * 50)
    
    # 计算总样本数
    total_samples = sum(
        size['total'] if isinstance(size, dict) else size 
        for size in sizes.values() 
        if not isinstance(size, str)  # Skip error messages
    )
    print(f"\n所有数据集总样本数: {total_samples}") 