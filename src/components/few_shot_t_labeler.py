"""
Few-shot Task Type Labeler
Using Qwen-8B model for hierarchical task type classification
"""
import json
import yaml
import os
import re
from typing import List, Dict, Any
import torch
import pandas as pd
from src.utils.query_type_graph import extract_task_hierarchy, read_kg_data
from src.utils.data_loader import DatasetGen
from src.components.llm_provider import LLMProvider, ProviderType

# Make sure you have installed required packages:
# pip install torch pandas pyyaml

class FewShotTaskLabeler:

        
    def generate_domain_prompt(self, query: str) -> str:
        """Generate multiple choice prompt for domain classification"""
        domains = [d['name'] for d in self.hierarchy['domains']]
        domain_definitions = {d['name']: d['definition'] for d in self.hierarchy['domains']}
        
        # Create options with letters
        options = []
        for i, domain in enumerate(domains):
            letter = chr(65 + i)  # A, B, C, etc.
            definition = domain_definitions[domain]
            options.append(f"{letter}. {domain}: {definition}")
        
        prompt = f"""Based on the following query, select the most appropriate domain. 

Query: {query}

Options:
{chr(10).join(options)}

Instructions:
1. Choose ONLY ONE option
2. Respond with ONLY the letter (A, B, C, etc.)
3. Format your answer as: Answer: **X** (where X is your chosen letter)
4. Do not include any explanations or additional text.
5. Do not answer the question.

Answer: """
        return prompt

    def generate_subcat_prompt(self, query: str, domain: str) -> str:
        """Generate multiple choice prompt for subcategory classification"""
        domain_id = next(i for i, d in enumerate(self.hierarchy['domains']) if d['name'] == domain)
        subcats = self.hierarchy['domain_to_subcats'][domain_id]
        subcat_definitions = {s['name']: s['definition'] for s in self.hierarchy['subcategories'] if s['domain'] == domain}
        
        # Create options with letters
        options = []
        for i, subcat in enumerate(subcats):
            letter = chr(65 + i)  # A, B, C, etc.
            definition = subcat_definitions[subcat]
            options.append(f"{letter}. {subcat}: {definition}")
        
        prompt = f"""Within the domain "{domain}", select the most appropriate subcategory for the following query.

Query: {query}

Options:
{chr(10).join(options)}

Instructions:
1. Choose ONLY ONE option
2. Respond with ONLY the letter (A, B, C, etc.)
3. Format your answer as: Answer: **X** (where X is your chosen letter)
4. Do not include any explanations or additional text
5. Do not answer the question.
Answer: """
        return prompt

    def generate_difficulty_prompt(self, query: str, subcat: str) -> str:
        """Generate multiple choice prompt for difficulty level classification"""
        subcat_id = next(i for i, s in enumerate(self.hierarchy['subcategories']) if s['name'] == subcat)
        difficulties = self.hierarchy['subcat_to_difficulties'][subcat_id]
        difficulty_definitions = {d['name']: d['definition'] for d in self.hierarchy['difficulty_levels'] if d['subcategory'] == subcat}
        
        # Create options with letters
        options = []
        for i, diff in enumerate(difficulties):
            letter = chr(65 + i)  # A, B, C, etc.
            definition = difficulty_definitions[diff]
            options.append(f"{letter}. {diff}: {definition}")
        
        prompt = f"""Within the subcategory "{subcat}", select the most appropriate difficulty level for the following query.

Query: {query}

Options:
{chr(10).join(options)}

Instructions:
1. Choose ONLY ONE option
2. Respond with ONLY the letter (A, B, C, etc.)
3. Format your answer as: Answer: **X** (where X is your chosen letter)
4. Do not include any explanations or additional text
5. Do not answer the question.
Answer: """
        return prompt

    def __init__(self, 
                 kg_path: str = "/hdd2/lh/agenticrouter_data/kg_data_gemini/kg_data.json",
                 data_dir: str = "/hdd2/lh/agenticrouter_data/data",
                 task_names: List[str] = None,
                 model_names: List[str] = ["qwen3-8b"],
                 config_path: str = "configs/models.yaml"):
        """
        Initialize the hierarchical task type classifier
        
        Args:
            kg_path: Path to knowledge graph
            data_dir: Data directory
            task_names: List of task names
            model_names: List of model names
            config_path: Path to model configuration file
        """
        self.kg_path = kg_path
        self.hierarchy = extract_task_hierarchy(kg_path)
        self.data_loader = DatasetGen(data_dir=data_dir, 
                                    task_names=task_names,
                                    model_names=model_names)
        
        # Load training, validation and test data
        self.train_data, self.val_data, self.test_data = self.data_loader.load_data()
        
        # Initialize LLM provider
        with open(config_path, 'r', encoding='utf-8') as f:
            self.models_config = yaml.safe_load(f)
            
        self.llm_provider = LLMProvider(
            provider=ProviderType.ALIPAY,
            model="qwen3-235b-a22b",
            config={
                "base_url": self.models_config['providers']['alipay']['api']['base_url'],
                "api_key": self.models_config['providers']['alipay']['api']['key']
            }
        )
        
    def _extract_answer_letter(self, response: str) -> str:
        """
        Extract the answer letter from model response
        
        Args:
            response: Raw model response
            
        Returns:
            Single letter answer (A, B, C, etc.) or empty string if not found
        """
        # 移除所有空白字符
        clean_response = response.strip()
        
        # 尝试不同的模式匹配
        patterns = [
            r'\*\*([A-Z])\*\*',                          # **B**
            r'\*\*([A-Z])\.',                            # **H.
            r'Answer:\s*\*\*([A-Z])[\.|\*]',            # Answer: **B** 或 Answer: **B.
            r'Answer:\s*([A-Z])[\.|\*]',                # Answer: B. 或 Answer: B*
            r'^([A-Z])[\.\*]',                          # B. 或 B*
            r'\*\*([A-Z])\..*?\*\*',                    # **H. Programming**
            r'([A-Z])\..*?[^a-zA-Z]'                    # E. Mathematics
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_response)
            if match:
                return match.group(1)
                
        # 如果上述模式都没匹配到，尝试找第一个大写字母
        for char in clean_response:
            if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return char
                
        return ""
        
    def _map_letter_to_option(self, letter: str, options: List[str]) -> str:
        """
        Map answer letter to the corresponding option text
        
        Args:
            letter: Answer letter (A, B, C, etc.)
            options: List of options in format ["A. option1: def1", "B. option2: def2", ...]
            
        Returns:
            The option name without letter prefix and definition
        """
        if not letter or not options:
            return ""
            
        try:
            # Find the option that starts with the letter
            for option in options:
                if option.startswith(letter + "."):
                    # Extract the option name (between ". " and ":")
                    return option.split(". ")[1].split(":")[0].strip()
        except Exception:
            pass
        return ""

    async def call_qwen(self, prompt: str, options: List[str]) -> str:
        """
        Call Qwen-8B model for text generation and extract the answer
        
        Args:
            prompt: Input prompt
            options: List of options in format ["A. option1: def1", "B. option2: def2", ...]
            
        Returns:
            The selected option name
        """
        try:
            gen_config = {
                "temperature": 0,
                "max_tokens": 50,
                "stream": False,
                "extra_body": {"enable_thinking": False},
            }
            
            result = await self.llm_provider.generate(
                prompt=prompt,
                config=gen_config
            )
            print(result.text)
            # Extract the letter answer
            answer_letter = self._extract_answer_letter(result.text)
            
            # Map the letter to the actual option
            return self._map_letter_to_option(answer_letter, options)
            
        except Exception as e:
            print(f"Error calling Qwen model: {str(e)}")
            return ""

    async def classify_query(self, query: str) -> Dict[str, str]:
        """
        Perform hierarchical classification for a single query
        
        Args:
            query: Input query text
            
        Returns:
            Dictionary containing domain, subcategory and difficulty classifications
        """
        # 1. Classify domain
        domain_prompt = self.generate_domain_prompt(query)
        domains = [d['name'] for d in self.hierarchy['domains']]
        domain_options = [f"{chr(65 + i)}. {domain}: {d['definition']}" 
                         for i, (domain, d) in enumerate(zip(domains, self.hierarchy['domains']))]
        domain = await self.call_qwen(domain_prompt, domain_options)
        
        if not domain:
            return {"domain": "", "subcategory": "", "difficulty": ""}
        
        # 2. Classify subcategory
        subcat_prompt = self.generate_subcat_prompt(query, domain)
        domain_id = next(i for i, d in enumerate(self.hierarchy['domains']) if d['name'] == domain)
        subcats = self.hierarchy['domain_to_subcats'][domain_id]
        subcat_options = [f"{chr(65 + i)}. {subcat}: {s['definition']}" 
                         for i, (subcat, s) in enumerate(zip(subcats, 
                         [s for s in self.hierarchy['subcategories'] if s['domain'] == domain]))]
        subcategory = await self.call_qwen(subcat_prompt, subcat_options)
        
        if not subcategory:
            return {"domain": domain, "subcategory": "", "difficulty": ""}
        
        # 3. Classify difficulty
        diff_prompt = self.generate_difficulty_prompt(query, subcategory)
        subcat_id = next(i for i, s in enumerate(self.hierarchy['subcategories']) if s['name'] == subcategory)
        difficulties = self.hierarchy['subcat_to_difficulties'][subcat_id]
        difficulty_nodes = [d for d in self.hierarchy['difficulty_levels'] if d['subcategory'] == subcategory]
        diff_options = [f"{chr(65 + i)}. {diff}: {d['definition']}" 
                       for i, (diff, d) in enumerate(zip(difficulties, difficulty_nodes))]
        difficulty_name = await self.call_qwen(diff_prompt, diff_options)
        
        # Convert difficulty name to id
        difficulty_id = ""
        if difficulty_name:
            matching_node = next((d for d in difficulty_nodes if d['name'] == difficulty_name), None)
            if matching_node:
                difficulty_id = str(matching_node['id'])
        
        return {
            "domain": domain,
            "subcategory": subcategory,
            "difficulty": difficulty_id
        }

    def _load_saved_results(self, result_file: str) -> Dict[int, Dict[str, str]]:
        """
        Load previously saved classification results
        
        Args:
            result_file: Path to the saved results file
            
        Returns:
            Dictionary of saved results
        """
        try:
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading saved results: {e}")
            return {}
            
    def _save_results(self, results: Dict[int, Dict[str, str]], result_file: str):
        """
        Save current classification results
        
        Args:
            results: Results to save
            result_file: Path to save the results
        """
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving results: {e}")

    async def classify_dataset(self, data: pd.DataFrame, result_file: str = "classification_results_temp.json") -> Dict[int, Dict[str, str]]:
        """
        Classify an entire dataset with checkpoint saving
        
        Args:
            data: DataFrame containing queries
            result_file: Path to save intermediate results
            
        Returns:
            Dictionary of classification results
        """
        # Load existing results if any
        results = self._load_saved_results(result_file)
        print(f"Loaded {len(results)} existing results")
        
        try:
            for i, row in data.iterrows():
                # Skip if already classified
                if str(i) in results and results[str(i)]["difficulty"]:
                    continue
                    
                query = row['query']
                print(f"Processing query {i}/{len(data)}")
                
                # If we have partial results for this query
                if str(i) in results:
                    existing = results[str(i)]
                    if existing["domain"] and existing["subcategory"] and not existing["difficulty"]:
                        # Continue from difficulty classification
                        diff_prompt = self.generate_difficulty_prompt(query, existing["subcategory"])
                        subcat_id = next(i for i, s in enumerate(self.hierarchy['subcategories']) 
                                       if s['name'] == existing["subcategory"])
                        difficulties = self.hierarchy['subcat_to_difficulties'][subcat_id]
                        difficulty_nodes = [d for d in self.hierarchy['difficulty_levels'] 
                                          if d['subcategory'] == existing["subcategory"]]
                        diff_options = [f"{chr(65 + i)}. {diff}: {d['definition']}" 
                                      for i, (diff, d) in enumerate(zip(difficulties, difficulty_nodes))]
                        difficulty_name = await self.call_qwen(diff_prompt, diff_options)
                        
                        # Convert difficulty name to id
                        difficulty_id = ""
                        if difficulty_name:
                            matching_node = next((d for d in difficulty_nodes if d['name'] == difficulty_name), None)
                            if matching_node:
                                difficulty_id = str(matching_node['id'])
                        
                        existing["difficulty"] = difficulty_id
                        results[str(i)] = existing
                    elif existing["domain"] and not existing["subcategory"]:
                        # Continue from subcategory classification
                        result = await self.classify_query(query)
                        results[str(i)] = result
                else:
                    # Start new classification
                    result = await self.classify_query(query)
                    results[str(i)] = result
                
                # Save after each query
                self._save_results(results, result_file)
                
        except Exception as e:
            print(f"Error during classification: {e}")
            # Save current progress before raising the error
            self._save_results(results, result_file)
            raise
            
        return results

    async def evaluate_all_datasets(self, output_dir: str = "query_task_type_results"):
        """
        Evaluate classification results for all datasets with checkpoint saving
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting training set classification...")
        train_results = await self.classify_dataset(
            self.train_data,
            result_file=os.path.join(output_dir, "train_results_temp.json")
        )
        
        print("Starting validation set classification...")
        val_results = await self.classify_dataset(
            self.val_data,
            result_file=os.path.join(output_dir, "val_results_temp.json")
        )
        
        print("Starting test set classification...")
        test_results = await self.classify_dataset(
            self.test_data,
            result_file=os.path.join(output_dir, "test_results_temp.json")
        )
        
        # Save final classification results
        results = {
            "train": train_results,
            "val": val_results,
            "test": test_results
        }
        
        final_result_file = os.path.join(output_dir, "classification_results.json")
        with open(final_result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"Classification results have been saved to {final_result_file}")
        
        # Clean up temporary files
        for temp_file in ["train_results_temp.json", "val_results_temp.json", "test_results_temp.json"]:
            try:
                os.remove(os.path.join(output_dir, temp_file))
            except:
                pass
                
        return results

async def main():
    """Main function for task type classification"""
    # task_names = ["alpaca_data", "GSM8K", "multi_news", "SQUAD"]
    task_names = ["wmt", "legalbench","medmcqa","mbpp"]
    dir_path = "/hdd2/lh/agenticrouter_data/kg_data_supp"
    task_labeler_dir_path = "/hdd2/lh/agenticrouter_data/query_task_type_results_supp"
    labeler = FewShotTaskLabeler(task_names=task_names, kg_path= os.path.join(dir_path, "kg_data.json"))
    results = await labeler.evaluate_all_datasets( output_dir= task_labeler_dir_path)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
