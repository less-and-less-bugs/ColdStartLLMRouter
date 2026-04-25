"""
Script to check LegalBench dataset structure and evaluation protocols
"""

from datasets import load_dataset
from collections import defaultdict

def check_legalbench_evaluation_protocols():
    """
    Check LegalBench dataset to identify evaluation protocols for different tasks
    """
    
    try:
        print("Loading LegalBench dataset...")
        dataset = load_dataset("HazyResearch/legalbench")
        print("Successfully loaded LegalBench dataset\n")
        
        # Get all task names
        task_names = list(dataset.keys())
        print(f"Found {len(task_names)} tasks\n")
        
        # Exclude 'rule_qa' task
        excluded_tasks = ['rule_qa']
        task_names = [task for task in task_names if task not in excluded_tasks]
        
        # Analyze task structures and evaluation protocols
        evaluation_protocols = defaultdict(list)
        task_structures = {}
        
        for task_name in task_names[:20]:  # Check first 20 tasks as sample
            try:
                task_dataset = dataset[task_name]
                
                # Check dataset info/features
                if hasattr(task_dataset, 'info') and task_dataset.info:
                    print(f"Task: {task_name}")
                    print(f"  Info: {task_dataset.info}")
                
                # Check features
                if hasattr(task_dataset, 'features') and task_dataset.features:
                    print(f"  Features: {task_dataset.features}")
                
                # Check splits
                splits = list(task_dataset.keys())
                print(f"  Splits: {splits}")
                
                # Get a sample from test split if available
                if 'test' in task_dataset:
                    test_split = task_dataset['test']
                    if len(test_split) > 0:
                        sample = test_split[0]
                        print(f"  Sample keys: {list(sample.keys())}")
                        
                        # Analyze output type to infer evaluation protocol
                        if 'output' in sample:
                            output = sample['output']
                            if isinstance(output, str):
                                # Check if it's a classification (short string) or generation (long text)
                                if len(output.split()) < 10:
                                    evaluation_protocols['classification'].append(task_name)
                                else:
                                    evaluation_protocols['generation'].append(task_name)
                            elif isinstance(output, (int, float)):
                                evaluation_protocols['regression'].append(task_name)
                            elif isinstance(output, list):
                                evaluation_protocols['multi_label'].append(task_name)
                        elif 'label' in sample:
                            evaluation_protocols['classification'].append(task_name)
                        elif 'answer' in sample:
                            answer = sample['answer']
                            if isinstance(answer, str) and len(answer.split()) < 10:
                                evaluation_protocols['classification'].append(task_name)
                            else:
                                evaluation_protocols['generation'].append(task_name)
                        
                        task_structures[task_name] = {
                            'keys': list(sample.keys()),
                            'output_type': type(sample.get('output', sample.get('answer', sample.get('label', None))))
                        }
                
                print()
                
            except Exception as e:
                print(f"Error processing {task_name}: {e}\n")
                continue
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION PROTOCOL SUMMARY")
        print("="*60)
        
        for protocol, tasks in evaluation_protocols.items():
            print(f"\n{protocol.upper()} ({len(tasks)} tasks):")
            for task in tasks[:10]:  # Show first 10
                print(f"  - {task}")
            if len(tasks) > 10:
                print(f"  ... and {len(tasks) - 10} more")
        
        # Check if there's metadata about evaluation functions
        print("\n" + "="*60)
        print("TASK STRUCTURES")
        print("="*60)
        for task_name, structure in list(task_structures.items())[:10]:
            print(f"\n{task_name}:")
            print(f"  Keys: {structure['keys']}")
            print(f"  Output type: {structure['output_type']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_legalbench_evaluation_protocols()

