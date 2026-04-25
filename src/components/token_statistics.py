"""
Script to calculate token usage statistics for QA pair generation.
"""
import json
import random
import tiktoken
from typing import Dict, List, Any
from src.utils.query_type_graph import read_kg_data

# Initialize tiktoken encoder for GPT-4 (cl100k_base)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))

def format_qa_pair(qa_pair: List[str]) -> str:
    """Format QA pair as it would appear in LLM output."""
    question, answer = qa_pair
    return f"Q: {question}\nA: {answer}\n"

def calculate_output_tokens(qa_file: str, sample_size: int = 40) -> Dict[str, Any]:
    """
    Calculate output tokens by sampling QA pairs from each task type.
    
    Args:
        qa_file: Path to generated QA pairs JSON file
        sample_size: Number of QA pairs to sample per task type
        
    Returns:
        Dictionary with statistics
    """
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    total_output_tokens = 0
    task_stats = {}
    
    # Format markers that would be in the output
    start_marker = "<qa_pairs begin>\n"
    end_marker = "\n</qa_pairs end>"
    
    for task_key, qa_pairs in qa_data.items():
        if not qa_pairs:
            continue
            
        # Sample QA pairs
        sample_qa_pairs = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
        
        # Format as LLM output
        formatted_output = start_marker
        for qa_pair in sample_qa_pairs:
            formatted_output += format_qa_pair(qa_pair)
        formatted_output += end_marker
        
        # Count tokens
        task_tokens = count_tokens(formatted_output)
        total_output_tokens += task_tokens
        
        task_stats[task_key] = {
            "sampled_pairs": len(sample_qa_pairs),
            "total_pairs": len(qa_pairs),
            "output_tokens": task_tokens
        }
    
    # Calculate average tokens per task
    num_tasks = len(task_stats)
    avg_tokens_per_task = total_output_tokens / num_tasks if num_tasks > 0 else 0
    
    # Scale to total tasks (assuming each task generates 40 QA pairs)
    # Since we sampled 40, we need to scale based on actual generation
    # But for output token calculation, we use the sampled data directly
    # The actual output would be: num_tasks * avg_tokens_per_task * (40 / sample_size)
    
    return {
        "total_tasks": num_tasks,
        "sample_size_per_task": sample_size,
        "total_output_tokens_sampled": total_output_tokens,
        "avg_output_tokens_per_task": avg_tokens_per_task,
        "task_stats": task_stats
    }

def calculate_input_tokens(kg_path: str, end_tag: str, question_per_prompt: int = 8) -> Dict[str, Any]:
    """
    Calculate input tokens based on prompts used in generation.
    
    Args:
        kg_path: Path to knowledge graph JSON file
        end_tag: Type of nodes (e.g., "difficulty_nodes")
        question_per_prompt: Number of questions per prompt (batch size)
        
    Returns:
        Dictionary with input token statistics
    """
    # System prompt from task_annotator.py
    system_prompt = """You are a helpful assistant that generates high-quality question-answer pairs. 
    You must respond with a specific format using markers to structure your output.
    
    IMPORTANT: Your response must follow this exact format:
    
    <qa_pairs begin>
    Q: What is 2+2?
    A: 2+2 equals 4
    
    Q: What is the capital of France?
    A: The capital of France is Paris
    
    Q: How does photosynthesis work?
    A: Photosynthesis is the process by which plants convert sunlight into energy.
    </qa_pairs end>
    
    Rules:
    - Start with <qa_pairs begin> and end with </qa_pairs end>
    - Each QA pair should be separated by a blank line
    - Use "Q:" for questions and "A:" for answers
    - Ensure questions are clear, relevant, and the answers are accurate and comprehensive"""
    
    # User prompt template based on end_tag
    nodes_data = read_kg_data(kg_path)
    prompt_list = []
    
    if end_tag == "domain_nodes":
        prompt_template = "Task Domain: {domain_name}\n Domain Definition: {domain_definition}\n"
        for node in nodes_data:
            user_prompt = prompt_template.format(
                domain_name=node["name"], 
                domain_definition=node["definition"]
            )
            prompt_list.append(user_prompt)
    elif end_tag == "subcategory_nodes":
        prompt_template = "Task Domain: {domain_name}\n Domain Definition: {domain_definition}\n Task Subcategory: {subcategory_name}\n Subcategory Definition: {subcategory_definition}\n"
        for node in nodes_data:
            for subcategory in node["subcategory_nodes"]:
                user_prompt = prompt_template.format(
                    domain_name=node["name"], 
                    domain_definition=node["definition"],
                    subcategory_name=subcategory["name"], 
                    subcategory_definition=subcategory["definition"]
                )
                prompt_list.append(user_prompt)
    elif end_tag == "difficulty_nodes":
        prompt_template = "Task Domain: {domain_name}\n Domain Definition: {domain_definition}\n Task Subcategory: {subcategory_name}\n Subcategory Definition: {subcategory_definition} \n Task Difficulty: {difficulty_name}\n Difficulty Definition: {difficulty_definition}\n"
        for node in nodes_data:
            for subcategory in node["subcategory_nodes"]:
                for difficulty in subcategory["difficulty_nodes"]:
                    user_prompt = prompt_template.format(
                        domain_name=node["name"], 
                        domain_definition=node["definition"],
                        subcategory_name=subcategory["name"], 
                        subcategory_definition=subcategory["definition"],
                        difficulty_name=difficulty["name"], 
                        difficulty_definition=difficulty["definition"]
                    )
                    prompt_list.append(user_prompt)
    
    # Add prefix to each prompt
    prefix = f"""Please generate {question_per_prompt} different question-answer pairs according to all the above specification.
        The questions should be clear, relevant, and the answers should be comprehensive and accurate.
        Focus on creating diverse questions that cover different aspects of the topic."""
    
    # Count tokens for system prompt
    system_tokens = count_tokens(system_prompt)
    
    # Count tokens for each user prompt
    user_prompt_tokens = []
    for user_prompt in prompt_list:
        full_user_prompt = user_prompt + prefix
        tokens = count_tokens(full_user_prompt)
        user_prompt_tokens.append(tokens)
    
    # Calculate average user prompt tokens
    avg_user_prompt_tokens = sum(user_prompt_tokens) / len(user_prompt_tokens) if user_prompt_tokens else 0
    
    # For OpenAI Chat API, tokens are counted as:
    # - System message: system_tokens + 4 (formatting overhead)
    # - User message: user_tokens + 2 (formatting overhead)
    # Total per request = system_tokens + 4 + user_tokens + 2 = system_tokens + user_tokens + 6
    tokens_per_request = system_tokens + avg_user_prompt_tokens + 6
    
    # Number of tasks
    num_tasks = len(prompt_list)
    
    # Number of requests per task = ceil(40 / question_per_prompt) * 6 (safety factor)
    # But user said: 任务数量 * 6 (容错)
    requests_per_task = 6
    
    # Total input tokens
    total_input_tokens = num_tasks * requests_per_task * tokens_per_request
    
    return {
        "num_tasks": num_tasks,
        "system_prompt_tokens": system_tokens,
        "avg_user_prompt_tokens": avg_user_prompt_tokens,
        "tokens_per_request": tokens_per_request,
        "requests_per_task": requests_per_task,
        "total_input_tokens": total_input_tokens,
        "prompt_samples": {
            "min_tokens": min(user_prompt_tokens) if user_prompt_tokens else 0,
            "max_tokens": max(user_prompt_tokens) if user_prompt_tokens else 0,
            "avg_tokens": avg_user_prompt_tokens
        }
    }

def calculate_statistics_for_provider(provider_name: str, kg_path: str, qa_file: str, 
                                      end_tag: str, question_per_prompt: int = 8) -> Dict[str, Any]:
    """
    Calculate token statistics for a specific provider (GPT or Gemini).
    
    Args:
        provider_name: Name of the provider ("GPT" or "Gemini")
        kg_path: Path to knowledge graph JSON file
        qa_file: Path to generated QA pairs JSON file
        end_tag: Type of nodes (e.g., "difficulty_nodes")
        question_per_prompt: Number of questions per prompt (batch size)
        
    Returns:
        Dictionary with complete statistics
    """
    print(f"\n{'=' * 80}")
    print(f"Token Usage Statistics for {provider_name}")
    print("=" * 80)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Calculate output tokens
    print(f"\n[{provider_name}] [1] Calculating Output Tokens...")
    print("-" * 80)
    output_stats = calculate_output_tokens(qa_file, sample_size=40)
    
    print(f"Total tasks: {output_stats['total_tasks']}")
    print(f"Sample size per task: {output_stats['sample_size_per_task']}")
    print(f"Total output tokens (sampled): {output_stats['total_output_tokens_sampled']:,}")
    print(f"Average output tokens per task: {output_stats['avg_output_tokens_per_task']:.2f}")
    
    # Scale to actual generation (40 QA pairs per task)
    # Since we sampled 40, the sampled tokens represent the actual output
    total_output_tokens = output_stats['total_output_tokens_sampled']
    
    # Calculate input tokens
    print(f"\n[{provider_name}] [2] Calculating Input Tokens...")
    print("-" * 80)
    input_stats = calculate_input_tokens(kg_path, end_tag, question_per_prompt)
    
    print(f"Number of tasks: {input_stats['num_tasks']}")
    print(f"System prompt tokens: {input_stats['system_prompt_tokens']}")
    print(f"Average user prompt tokens: {input_stats['avg_user_prompt_tokens']:.2f}")
    print(f"Tokens per request: {input_stats['tokens_per_request']:.2f}")
    print(f"Requests per task (safety factor): {input_stats['requests_per_task']}")
    print(f"Total input tokens: {input_stats['total_input_tokens']:,}")
    
    # Convert to millions of tokens
    input_tokens_millions = input_stats['total_input_tokens'] / 1_000_000
    output_tokens_millions = total_output_tokens / 1_000_000
    
    print(f"\n[{provider_name}] SUMMARY")
    print("-" * 80)
    print(f"Input Tokens:  {input_tokens_millions:.4f} million tokens")
    print(f"Output Tokens: {output_tokens_millions:.4f} million tokens")
    print(f"Total Tokens:  {(input_tokens_millions + output_tokens_millions):.4f} million tokens")
    
    return {
        "provider": provider_name,
        "input_tokens": {
            "total": input_stats['total_input_tokens'],
            "millions": input_tokens_millions,
            "details": input_stats
        },
        "output_tokens": {
            "total": total_output_tokens,
            "millions": output_tokens_millions,
            "details": output_stats
        }
    }

def main():
    """Main function to calculate and display token statistics for both GPT and Gemini."""
    end_tag = "difficulty_nodes"
    question_per_prompt = 8
    
    print("=" * 80)
    print("Token Usage Statistics for QA Pair Generation")
    print("Comparing GPT and Gemini Providers")
    print("=" * 80)
    
    # Configuration for both providers
    configs = [
        {
            "name": "GPT",
            "kg_path": "/hdd2/lh/agenticrouter_data/kg_data/kg_data.json",
            "qa_file": "/hdd2/lh/agenticrouter_data/kg_data/generated_qa_difficulty_nodes.json"
        },
        {
            "name": "Gemini",
            "kg_path": "/hdd2/lh/agenticrouter_data/kg_data_gemini/kg_data.json",
            "qa_file": "/hdd2/lh/agenticrouter_data/kg_data_gemini/generated_qa_difficulty_nodes.json"
        }
    ]
    
    all_stats = {}
    
    # Calculate statistics for each provider
    for config in configs:
        try:
            stats = calculate_statistics_for_provider(
                provider_name=config["name"],
                kg_path=config["kg_path"],
                qa_file=config["qa_file"],
                end_tag=end_tag,
                question_per_prompt=question_per_prompt
            )
            all_stats[config["name"]] = stats
        except Exception as e:
            print(f"\nError processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall comparison
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    
    for provider_name, stats in all_stats.items():
        print(f"\n{provider_name}:")
        print(f"  Input Tokens:  {stats['input_tokens']['millions']:.4f} million tokens")
        print(f"  Output Tokens: {stats['output_tokens']['millions']:.4f} million tokens")
        print(f"  Total Tokens:  {(stats['input_tokens']['millions'] + stats['output_tokens']['millions']):.4f} million tokens")
    
    # Calculate totals
    total_input = sum(s['input_tokens']['millions'] for s in all_stats.values())
    total_output = sum(s['output_tokens']['millions'] for s in all_stats.values())
    
    print(f"\nTOTAL (Both Providers):")
    print(f"  Input Tokens:  {total_input:.4f} million tokens")
    print(f"  Output Tokens: {total_output:.4f} million tokens")
    print(f"  Total Tokens:  {(total_input + total_output):.4f} million tokens")
    
    # Save detailed statistics
    output_file = "/hdd2/lh/agenticrouter_data/kg_data/token_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed statistics saved to: {output_file}")

if __name__ == "__main__":
    main()

