"""
candidate datasets: 
coding tasks MBPP
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mbpp/README.md

"""

import pandas as pd
from utils import loadjson, get_embedding, savepkl, loadpkl
import yaml
import os
import glob

def process_mmlu_redux_data(path, sample_size):
    """
    Process MMLU-Redux data from multiple JSON files
    """
    all_data = []
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(path, "*.json"))
    
    for json_file in json_files:
        if "mmlu_redux_all" in json_file:  # Skip the combined file
            continue
            
        data = loadjson(json_file)
        subject = os.path.basename(json_file).replace('.json', '')
        
        # Process each item
        for item in data[:sample_size]:
            # Format query according to the specified template
            query = f"Q: {item['question'].strip()}\n(A) {item['choices'][0]} (B) {item['choices'][1]} (C) {item['choices'][2]} (D) {item['choices'][3]}\n"
            
            # Convert numeric answer to letter format
            answer_map = ['(A)', '(B)', '(C)', '(D)']
            ground_truth = answer_map[item['answer']]
            
            all_data.append({
                'task_id': "mmlu_redux",
                'sub_task': subject,
                'query': query,
                'ground_truth': ground_truth,
                'metric': 'mmluredux',
                'task_description': 'The MMLU-Redux dataset is a comprehensive evaluation benchmark designed to test knowledge across various academic and professional domains. It presents multiple-choice questions that assess understanding and expertise in specific subject areas.'
            })
    
    return all_data

def generate_unified_qa_dataset(output_path='data/unified_qa_data.csv',sample_size=60, mmlu_sample_size=30):
    """
    Generate a unified question-answering dataset from multiple data sources.

    Parameters:
    sample_size (int): Number of samples to extract from each dataset
    output_path (str): Path to save the output CSV file

    Returns:
    pandas.DataFrame: The generated unified dataset
    """
    # Initialize result DataFrame
    df = pd.DataFrame(columns=[
        'task_id', 'sub_task', 'query', 'ground_truth', 'metric',
        'task_description'  # Added task description column
    ])

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
                mmlu_data = process_mmlu_redux_data(config['path'], mmlu_sample_size)
                for item in mmlu_data:
                    # Add instruction for code wrapping
                    item['query'] = item['query'] + "\nPlease wrap your final answer with tag <answer> your answer </answer>"
                    df = df._append(item, ignore_index=True)
            elif config['format'] == 'json':
                data = loadjson(config['path'])[:sample_size]

                # Process JSON formatted data
                for item in data:
                    # Construct query text based on configuration
                    if isinstance(config['query_fields'], list):
                        if config['task_name'] == 'MBPP':
                            query = "You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{test_list[0]}\n{test_list[1]}\n{test_list[2]}\nPlease wrap your final answer with tag <code> your codes </code>".format(text=item['text'], test_list=item['test_list'])
                        else:
                            query = ''.join([item[field] for field in config['query_fields']])
                            ground_truth = item[config['ground_truth_field']]
                    else:
                        query = item[config['query_fields']]
                        ground_truth = item[config['ground_truth_field']]

                    # Add to dataset
                    new_row = {
                        'task_id': config['task_name'],
                        'sub_task': None,  # Non-MMLU tasks don't have sub_tasks
                        'query': query,
                        'ground_truth': ground_truth,
                        'metric': config['metric'],
                        'task_description': config['task_description']
                    }
                    df = df._append(new_row, ignore_index=True)

            elif config['format'] == 'parquet':
                data = pd.read_parquet(config['path'])[:sample_size]

                # Process Parquet formatted data
                for item in data.itertuples():
                    query = getattr(item, config['query_field'])

                    # Handle complex ground truth structures
                    if 'ground_truth_subfield' in config:
                        ground_truth_container = getattr(item, config['ground_truth_field'])
                        ground_truth = ground_truth_container[config['ground_truth_subfield']][config['ground_truth_index']]
                    else:
                        ground_truth = getattr(item, config['ground_truth_field'])

                    # add to dataset
                    new_row = {
                        'task_id': config['task_name'],
                        'sub_task': None,  # Non-MMLU tasks don't have sub_tasks
                        'query': query,
                        'ground_truth': ground_truth,
                        'metric': config['metric'],
                        'task_description': config['task_description']
                    }
                    df = df._append(new_row, ignore_index=True)

        except Exception as e:
            print(f"Error processing {config['task_name']}: {str(e)}")
            continue

    # Save results to CSV
    df.to_csv(output_path, index=False)
    return df


# Usage example
if __name__ == "__main__":
    # Open config file
    unified_dataset_path = 'data/unified_qa_data_ex_search.csv'
    # Generate dataset with default sample size
    unified_dataset = generate_unified_qa_dataset(unified_dataset_path, sample_size=600, mmlu_sample_size=20)

    # Or specify custom sample size
    # unified_dataset = generate_unified_qa_dataset(config['unified_qa_data_path'],sample_size=100)