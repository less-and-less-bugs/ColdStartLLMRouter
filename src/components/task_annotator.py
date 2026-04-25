"""
This file is used to generate qa pairs for each task type node.
"""
import json 
import asyncio
from typing import List, Dict, Any
from src.components.llm_provider import LLMProvider, ProviderType
import re
import os
from src.utils.query_type_graph import read_kg_data
from src.utils.similarity import SimilarityTool_hugg

# Initialize similarity tool
similarity_tool = SimilarityTool_hugg(model_name='sentence-transformers/all-MiniLM-L6-v2')

def parse_qa_response(text: str):
    """解析QA响应文本，直接用正则分割Q/A对"""
    try:
        # 提取标记内容
        start_marker = "<qa_pairs begin>"
        end_marker = "</qa_pairs end>"
        try:
            start_idx = text.index(start_marker)
            end_idx = text.index(end_marker)
            content = text[start_idx + len(start_marker):end_idx].strip()
        except ValueError:
            print(f"Error: Could not find qa_pairs markers in LLM output:")
            print(f"Text: {text[:200]}...")
            return None

        # 用正则表达式提取所有Q/A对
        pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=(?:Q:|$))"
        matches = re.findall(pattern, content, re.DOTALL)

        qa_pairs = []
        for q, a in matches:
            question = q.strip()
            answer = a.strip()
            if question and answer:
                qa_pairs.append([question, answer])
        return qa_pairs

    except Exception as e:
        print(f"Error parsing QA response: {str(e)}")
        return None


def annotate_data(kg_path: str, end_tag: str, question_num: int = 8) -> list:
    nodes_data = read_kg_data(kg_path)
    prompt_list = []
    if end_tag == "domain_nodes":
        prompt = "Task Domain: {domain_name}\n Domain Definition: {domain_definition}\n"
        for node in nodes_data:
            prompt_list.append([node["name"], prompt.format(domain_name=node["name"], domain_definition=node["definition"])])
    elif end_tag == "subcategory_nodes":
        prompt = "Task Domain: {domain_name}\n Domain Definition: {domain_definition}\n Task Subcategory: {subcategory_name}\n Subcategory Definition: {subcategory_definition}\n"
        for node in nodes_data:
            for subcategory in node["subcategory_nodes"]:
                prompt_list.append([node["name"], subcategory["name"], prompt.format(domain_name=node["name"], domain_definition=node["definition"], subcategory_name=subcategory["name"], subcategory_definition=subcategory["definition"])])
    elif end_tag == "difficulty_nodes":
        prompt = "Task Domain: {domain_name}\n Domain Definition: {domain_definition}\n Task Subcategory: {subcategory_name}\n Subcategory Definition: {subcategory_definition} \n Task Difficulty: {difficulty_name}\n Difficulty Definition: {difficulty_definition}\n"
        for node in nodes_data:
            for subcategory in node["subcategory_nodes"]:
                for difficulty in subcategory["difficulty_nodes"]:
                    prompt_list.append([node["name"], subcategory["name"], difficulty["name"], prompt.format(domain_name=node["name"], domain_definition=node["definition"], subcategory_name=subcategory["name"], subcategory_definition=subcategory["definition"], difficulty_name=difficulty["name"], difficulty_definition=difficulty["definition"])])
    elif end_tag == "preference_nodes":
        prompt = "Original Question-answer pair in json format: {original_qa}\n Task Preference: {preference_name}\n Preference Definition: {preference_definition}\n"
        # For preference_nodes, we need to handle differently as it requires original QA pairs
        # This would need to be implemented based on your specific data structure
    if end_tag in ["domain_nodes", "subcategory_nodes", "difficulty_nodes", "preference_nodes"]:
        prefix = f"""Please generate {question_num} different question-answer pairs according to all the above specification.
        The questions should be clear, relevant, and the answers should be comprehensive and accurate.
        Focus on creating diverse questions that cover different aspects of the topic."""

    else:
        prefix  = f"""Given the original {question_num} question-answer pairs, please rewrite them into corresponding new question-answer pairs that simulate the user's preference.         
        Ensure the new questions and answers are relevant to the user's preference while maintaining quality."""
    for i, prompt in enumerate(prompt_list):
        prompt_list[i][-1] = prompt[-1] + prefix
        
    return prompt_list

async def generate_qa_for_task(prompt_list: list, model: str, provider: str, config: Dict[str, Any], gen_config: Dict[str, Any], 
 question_num: int = 40, question_per_prompt: int = 8, output_file: str = "") -> Dict[str, Any]:
    """
    Generate question-answer pairs for different task types using LLM with structured output
    
    Args:
        prompt_list: List of prompts with task information
        model: LLM model name
        provider: LLM provider name
        config: Provider configuration
        gen_config: Generation configuration
        question_num: Number of questions to generate per prompt
        question_per_prompt: Number of questions to generate per LLM call
        output_file: Output file path for saving results
        
    Returns:
        Dictionary of generated question-answer pairs with metadata
    """
    llm_provider = LLMProvider(
        provider=ProviderType.OPENAI,
        model=model,
        config=config
    )
    
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
    
    print(f"Generating {question_num} questions per prompt using structured output...")
    
    for i, prompt_data in enumerate(prompt_list):
            # 如果输出文件存在，加载已有结果
        if output_file and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    all_qa_pairs = json.load(f)
                print(f"Loaded existing results from {output_file}")
            except Exception as e:
                print(f"Error loading existing file: {e}")
                all_qa_pairs = {}
        else:
            all_qa_pairs = {}

        prompt_text = prompt_data[-1]  # Get the actual prompt text
        task_info = prompt_data[:-1]   # Get task metadata
        task_key = i # 使用task_info作为key，转换为字符串
        
        print(f"Processing prompt {i+1}/{len(prompt_list)}: {task_info}")
        

        if str(task_key) in all_qa_pairs and len(all_qa_pairs[str(task_key)]) == question_num:
            print(f"Prompt {i+1} already completed with {len(all_qa_pairs[str(task_key)])} QA pairs, skipping...")
            continue
        
        # 如果部分完成，获取已有的QA对数量
        existing_count = len(all_qa_pairs.get(str(task_key), []))
        remaining_count = question_num - existing_count
        print(f"Existing count: {existing_count}, Remaining count: {remaining_count}")
        # if existing_count > 0:
        #     print(f"Resuming prompt {i+1}: {existing_count} QA pairs already generated, need {remaining_count} more")
        # exit(0)
        qa_pairs_for_prompt = all_qa_pairs.get(str(task_key), [])
        total_attempts = 0
        max_total_attempts = remaining_count // question_per_prompt + 10

        while len(qa_pairs_for_prompt) < question_num and total_attempts < max_total_attempts:
            total_attempts += 1
            try:
                # 1. 生成LLM输出
                result = await llm_provider.generate(
                    prompt=prompt_text,
                    system_prompt=system_prompt,
                    config=gen_config,
                )
                # 2. 用解析方法解析
                qa_pairs = parse_qa_response(result.text)
                if qa_pairs is None:
                    print(f"Warning: No structured output from LLM")
                    continue
                else:
                    print(f"Successfully parsed structured output with {len(qa_pairs)} QA pairs")
                    new_added_count = 0
                    for qa_pair in qa_pairs:
                        if len(qa_pairs_for_prompt) < question_num:
                            # Check similarity with existing QA pairs
                            is_similar = False
                            similarity_threshold = 0.9 # 可以根据需要调整阈值
                            
                            # 检查新问题与已有问题的相似度
                            for existing_pair in qa_pairs_for_prompt:
                                similarity = similarity_tool.similarity(qa_pair[0], existing_pair[0])
                                if similarity > similarity_threshold:
                                    is_similar = True
                                    print(f"Skipping similar question (similarity: {similarity:.2f})")
                                    break
                            
                            if not is_similar:
                                qa_pairs_for_prompt.append(qa_pair)
                                new_added_count += 1
                        else:
                            break
                    print(f"Added new QA pair (total: {new_added_count})")
                    
                    # 更新结果字典
                    all_qa_pairs[task_key] = qa_pairs_for_prompt
            except Exception as e:
                print(f"Warning: Only generated {len(qa_pairs_for_prompt)} QA pairs from prompt {i+1} (target: {question_num})")
                    
        # 保存到文件
        if output_file:
            try:
                # 确保目录存在
                output_dir = os.path.dirname(output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
                print(f"Saved progress to {output_file}")
            except Exception as e:
                print(f"Error saving to file: {e}")
    
    print(f"Successfully generated QA pairs from {len(prompt_list)} prompts")
    
    return all_qa_pairs

if __name__ == "__main__":
    config = {
        "api_key": None,
        "base_url": "https://api.zhizengzeng.com/"
    }
    model = "gemini-2.5-flash"  # 使用支持结构化输出的模型
    # model = "gpt-4.1-2025-04-14"  # 使用支持结构化输出的模型
    provider = "OPENAI"
    gen_config = {
        "temperature": 0.7,
        # "max_tokens": 2000  # Increased for longer responses

    }
    dir_path = "/hdd2/lh/agenticrouter_data/kg_data_supp"
    kg_path = os.path.join(dir_path, "kg_data.json")

    end_tag = "difficulty_nodes"

    async def main():
        question_num = 40
        question_per_prompt = 8
        prompt_list = annotate_data(kg_path, end_tag, question_num=question_per_prompt)
        print(f"Generated {len(prompt_list)} prompts")
        print("Sample prompt:", prompt_list[0] if prompt_list else "No prompts")
        
        output_file = os.path.join(dir_path, f"generated_qa_{end_tag}.json")
        
        # Generate QA pairs
        qa_pairs = await generate_qa_for_task(
            prompt_list=prompt_list, 
            model=model, 
            provider=provider, 
            config=config, 
            gen_config=gen_config,
            question_num=question_num,
            question_per_prompt=question_per_prompt,
            output_file=output_file
        )
        


        if qa_pairs:
            # 获取第一个QA对作为示例
            first_key = list(qa_pairs.keys())[0]
            first_qa_list = qa_pairs[first_key]
            if first_qa_list:
                print("Sample QA pair:", first_qa_list[0])

    # Run the async main function
    asyncio.run(main())