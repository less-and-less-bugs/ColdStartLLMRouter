from typing import List, Dict, Optional, Any, Tuple
import asyncio
import json
from dataclasses import dataclass
from src.components.llm_provider import LLMProvider, GenerationConfig, ProviderType
from src.prompts.node_gen import (
    NodeGenPrompt, NodeGenRule, InitialDomains,
    DomainNodePrompt, SubcategoryNodePrompt,
    NodeRevisePrompt, NodeSetChoicePrompt,
    DifficultyLevelNodePrompt, UserPreferenceNodePrompt
)
from src.utils.logger import LLMLogger
import os
from datetime import datetime

config = {
    "api_key": None,
    "base_url": "https://api.zhizengzeng.com/"
}

@dataclass
class Node:
    """基础节点类"""
    name: str
    definition: str
    example: str

class KGConstructor:
    """知识图谱构建器"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.gen_config = {
            "temperature": 0.7,
            "max_tokens": 30000
        }
   
        self.logger = LLMLogger()
        
    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用LLM生成文本"""
        try:
            result = await self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.gen_config
            )
            # 记录交互
            self.logger.log_interaction(
                prompt=prompt,
                response=result.text,
                system_prompt=system_prompt,
                metadata={
                    "temperature": self.gen_config["temperature"],
                    "max_tokens": self.gen_config["max_tokens"]
                }
            )
            print(result)
            return result.text
        except Exception as e:
            # 记录错误
            self.logger.log_interaction(
                prompt=prompt,
                response=str(e),
                system_prompt=system_prompt,
                metadata={
                    "error": True,
                    "error_type": type(e).__name__,
                    "temperature": self.gen_config["temperature"],
                    "max_tokens": self.gen_config["max_tokens"]
                }
            )
            raise
        
    def _extract_node_content(self, text: str) -> Optional[str]:
        """从LLM输出中提取<node begin>和<node end>之间的内容"""
        try:
            start_idx = text.index("<node begin>")
            end_idx = text.index("<node end>")
            return text[start_idx + len("<node begin>"):end_idx].strip()
        except ValueError:
            print(f"Error: Could not find node markers in LLM output:")
            print(f"Text: {text[:200]}...")  # 只打印前200个字符
            return None
            
    def _parse_node_set(self, text: str, start_tag: str, end_tag: str, node_type: str) -> Tuple[List[Node], bool]:
        """解析LLM生成的节点集合文本
        
        Returns:
            Tuple[List[Node], bool]: (节点列表, 是否包含正确的标记)
        """
        try:
            # 首先检查是否包含正确的标记
            start_idx = text.index(start_tag)
            end_idx = text.index(end_tag)
            fail_flag = False
            try:
                text = text[start_idx + len(start_tag):end_idx].strip()
            except ValueError:
                print(f"Error: Invalid node markers in text:")
                print(f"Text: {text[:200]}...")
                fail_flag = True
            
            nodes = []
            if fail_flag:
                return nodes, False
            
            # 1. 按换行符分割
            lines = text.split('\n')
            
            # 2. 预处理每一行
            processed_lines = []
            keywords = [f'{node_type}:', 'Definition:', 'Example:']
            
            for line in lines:
                # 移除前后空白
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                
                # 检查是否包含关键词，如果不包含则跳过
                if not any(keyword in line for keyword in keywords):
                    continue
                
                # 移除**标记如果存在
                line = line.replace('**', '')
                processed_lines.append(line)
            
            # 验证处理后的行
            if len(processed_lines) % 3 != 0:
                print(f"Error: Number of lines ({len(processed_lines)}) is not a multiple of 3")
                return [], False
            
            # 验证关键词顺序并解析节点
            nodes = []
            for i in range(0, len(processed_lines), 3):
                if not (processed_lines[i].startswith(f'{node_type}:') and 
                       processed_lines[i+1].startswith('Definition:') and 
                       processed_lines[i+2].startswith('Example:')):
                    print(f"Error: Invalid keyword order at lines {i+1}-{i+3}:")
                    print(f"Expected: {node_type}, Definition, Example")
                    print(f"Got: {processed_lines[i:i+3]}")
                    return [], False
                
                name = processed_lines[i].split(':', 1)[1].strip()
                definition = processed_lines[i+1].split(':', 1)[1].strip()
                example = processed_lines[i+2].split(':', 1)[1].strip()
                nodes.append(Node(name=name, definition=definition, example=example))
            
            return nodes, True
        except Exception as e:
            print(f"Error parsing node set: {str(e)}")
            return [], False
        
    def _extract_preferred_set(self, text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """从LLM输出中提取<preferred set>标签中的内容并判断是A还是B"""
        try:
            start_idx = text.index(start_tag)
            end_idx = text.index(end_tag)
            preferred_text = text[start_idx + len(start_tag):end_idx].strip()
            print(preferred_text)
            if "A" in preferred_text:
                return "A"
            elif "B" in preferred_text:
                return "B"
            return None
        except ValueError:
            print(f"Error: Could not find preferred set markers in LLM output:")
            print(f"Text: {text[:200]}...")
            return None

    async def generate_domain_nodes(self, max_iterations: int = 5, stability_threshold: int = 2,) -> List[Node]:
        """生成Domain节点集合"""
        current_nodes = []
        stability_count = 0
        max_retries = 3
        self.logger.info(f"开始生成Domain节点，最大迭代次数: {max_iterations}, 稳定阈值: {stability_threshold}")
        
        for i in range(max_iterations):
            self.logger.info(f"开始第 {i+1} 次迭代")
            if not current_nodes:
                # 生成初始节点集合，如果格式不正确则重试
                for retry in range(max_retries):
                    self.logger.debug(f"尝试生成初始Domain节点 (尝试 {retry + 1}/{max_retries})")
                    response = await self._call_llm(DomainNodePrompt)
                    nodes, is_valid = self._parse_node_set(response, "<node begin>", "</node end>", "Domain")
                    if is_valid:
                        current_nodes = nodes
                        self.logger.info(f"成功生成 {len(nodes)} 个初始Domain节点")
                        break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for domain node generation")
                        continue
                    else:
                        self.logger.error(f"Failed to generate domain nodes after {max_retries} retries")
                        exit(0)
            else:
                self.logger.debug("开始修改现有Domain节点")
                # 使用NodeRevisePrompt进行修改
                revise_prompt = NodeRevisePrompt.format(
                    node_name="Domain Node",
                    candidate_node_set=self._format_nodes_for_prompt(current_nodes, "Domain"),
                    node_gen_rules=NodeGenRule[0],
                    special_note="""We have already defined the following six initial Domains:

- Mathematics  
- Creative Writing  
- Commonsense Knowledge 
- Programming 
- Long-context Understanding 
- Reading Comprehension.
""",
                    node_name_short="Domain"
                )
                
                # 添加重试逻辑
                revised_nodes = None
                is_valid = False
                
                for retry in range(max_retries):
                    self.logger.info(f"尝试修改Domain节点 (尝试 {retry + 1}/{max_retries})")
                    revised_response = await self._call_llm(revise_prompt)
                    revised_nodes, is_valid = self._parse_node_set(revised_response, "<revision node begin>", "</revision node end>", "Domain")
                    if is_valid:
                        self.logger.info(f"成功生成候选Domain节点，生成 {len(revised_nodes)} 个节点")
                        break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for node revision")
                        continue
                    else:
                        self.logger.error(f"Failed to revise nodes after {max_retries} retries")
                        exit(0)
                
                # 比较两个节点集合
                for retry in range(max_retries):
                    # 随机分配A/B顺序
                    import random
                    if random.random() < 0.5:
                        set_a, set_b = current_nodes, revised_nodes
                        is_current_a = True
                    else:
                        set_a, set_b = revised_nodes, current_nodes
                        is_current_a = False
                        
                    choice_prompt = NodeSetChoicePrompt.format(
                        node_name="Domain Node",
                        node_gen_rules=NodeGenRule[0],
                        special_note="""We have already defined the following six initial Domains:

- Mathematics  
- Creative Writing  
- Commonsense Knowledge 
- Programming 
- Long-context Understanding 
- Reading Comprehension.

Please generate up to 4 new high-level Domains that are distinct from the above.""",
                        candidate_node_set_a=self._format_nodes_for_prompt(set_a, "Domain"),
                        candidate_node_set_b=self._format_nodes_for_prompt(set_b, "Domain")
                    )
                    
                    choice_response = await self._call_llm(choice_prompt)
                    preferred_set = self._extract_preferred_set(choice_response, "<preferred set>", "</preferred set>")
                    
                    if preferred_set:
                        # 根据A/B选择和随机分配确定是否使用revised_nodes
                        use_revised = (preferred_set == "A" and not is_current_a) or (preferred_set == "B" and is_current_a)
                        if use_revised:
                            self.logger.info(f"使用候选Domain节点")
                            if current_nodes == revised_nodes:
                                stability_count += 1
                            else:
                                stability_count = 0
                            current_nodes = revised_nodes
                        else:
                            self.logger.info(f"使用上轮Domain节点")
                            stability_count += 1
                        break
                    else:  
                        if retry < max_retries - 1:
                            print(f"Retry {retry + 1} of {max_retries} for choice comparison")
                            continue
                        else:
                            print(f"Failed to get valid choice after {max_retries} retries")
                            exit(0)
                
            if stability_count >= stability_threshold:
                break
                
        return current_nodes
        
    async def generate_subcategory_nodes(self, domain_node: Node, max_iterations: int = 5, stability_threshold: int = 2, max_subcategory_nodes: int = 10) -> List[Node]:
        """为单个Domain生成Subcategory节点集合
        
        Args:
            domain_node: 单个Domain节点
            max_iterations: 最大迭代次数
            stability_threshold: 节点集合稳定所需的迭代次数
            max_subcategory_nodes: 最大子类别节点数量
            
        Returns:
            该Domain对应的Subcategory节点列表
        """
        max_retries = 3
        self.logger.info(f"开始为Domain '{domain_node.name}'生成Subcategory节点")
        current_nodes = []
        stability_count = 0
        
        for i in range(max_iterations):
            self.logger.info(f"开始第 {i+1} 次迭代")
            if not current_nodes:
                # 生成初始subcategory节点
                for retry in range(max_retries):
                    self.logger.debug(f"尝试生成初始Subcategory节点 (尝试 {retry + 1}/{max_retries})")
                    prompt = SubcategoryNodePrompt.format(
                        domain_name=domain_node.name,
                        domain_definition=domain_node.definition,
                        domain_example=domain_node.example,
                        max_subcategory_nodes=max_subcategory_nodes,
                        SubcategoryNodeRule=NodeGenRule[1].format(max_subcategory_nodes=max_subcategory_nodes)
                    )
                    response = await self._call_llm(prompt)
                    nodes, is_valid = self._parse_node_set(response, "<node begin>", "</node end>", "Subcategory")
                    if is_valid:
                        current_nodes = nodes
                        self.logger.info(f"成功生成 {len(nodes)} 个初始Subcategory节点")
                        if len(nodes) > max_subcategory_nodes:
                            self.logger.warning(f"生成超过{max_subcategory_nodes}个Subcategory节点，请重新生成")
                            continue
                        else:
                            break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for subcategory node generation")
                        continue
                    else:
                        self.logger.error(f"Failed to generate subcategory nodes after {max_retries} retries")
                        exit(0)
            else:
                self.logger.debug("开始修改现有Subcategory节点")
                # 使用NodeRevisePrompt进行修改
                revise_prompt = NodeRevisePrompt.format(
                    node_name="Subcategory Node",
                    candidate_node_set=self._format_nodes_for_prompt(current_nodes, "Subcategory"),
                    node_gen_rules=NodeGenRule[1].format(max_subcategory_nodes=max_subcategory_nodes),
                    special_note=f"Ensure subcategories are specific to the {domain_node.name} domain.",
                    node_name_short="Subcategory"
                )
                
                # 添加重试逻辑
                revised_nodes = None
                is_valid = False
                
                for retry in range(max_retries):
                    self.logger.info(f"尝试修改Subcategory节点 (尝试 {retry + 1}/{max_retries})")
                    revised_response = await self._call_llm(revise_prompt)
                    revised_nodes, is_valid = self._parse_node_set(revised_response, "<revision node begin>", "</revision node end>", "Subcategory")
                    if is_valid:
                        self.logger.info(f"成功生成候选Subcategory节点，生成 {len(revised_nodes)} 个节点")
                        if len(revised_nodes) > max_subcategory_nodes:
                            self.logger.warning(f"生成超过{max_subcategory_nodes}个Subcategory节点，请重新生成")
                            continue
                        else:
                            break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for node revision")
                        continue
                    else:
                        self.logger.error(f"Failed to revise nodes after {max_retries} retries")
                        exit(0)
                
                # 比较两个节点集合
                for retry in range(max_retries):
                    # 随机分配A/B顺序
                    import random
                    if random.random() < 0.5:
                        set_a, set_b = current_nodes, revised_nodes
                        is_current_a = True
                    else:
                        set_a, set_b = revised_nodes, current_nodes
                        is_current_a = False
                        
                    choice_prompt = NodeSetChoicePrompt.format(
                        node_name="Subcategory Node",
                        node_gen_rules=NodeGenRule[1].format(max_subcategory_nodes=max_subcategory_nodes),
                        special_note=f"Ensure subcategories are specific to the {domain_node.name} domain.",
                        candidate_node_set_a=self._format_nodes_for_prompt(set_a, "Subcategory"),
                        candidate_node_set_b=self._format_nodes_for_prompt(set_b, "Subcategory")
                    )
                    
                    choice_response = await self._call_llm(choice_prompt)
                    preferred_set = self._extract_preferred_set(choice_response, "<preferred set>", "</preferred set>")
                    
                    if preferred_set:
                        # 根据A/B选择和随机分配确定是否使用revised_nodes
                        use_revised = (preferred_set == "A" and not is_current_a) or (preferred_set == "B" and is_current_a)
                        if use_revised:
                            self.logger.info(f"使用候选Subcategory节点")
                            if current_nodes == revised_nodes:
                                stability_count += 1
                            else:
                                stability_count = 0
                            current_nodes = revised_nodes
                        else:
                            self.logger.info(f"使用上轮Subcategory节点")
                            stability_count += 1
                        break
                    else:
                        if retry < max_retries - 1:
                            self.logger.warning(f"Retry {retry + 1} of {max_retries} for choice comparison")
                            continue
                        else:
                            self.logger.error(f"Failed to get valid choice after {max_retries} retries")
                            exit(0)
            
            if stability_count >= stability_threshold:
                break
                
        self.logger.info(f"完成Domain '{domain_node.name}'的Subcategory节点生成，共 {len(current_nodes)} 个节点")
        return current_nodes
        
    async def generate_difficulty_nodes(self, domain_node: Node, subcategory_node: Node, max_iterations: int = 5, stability_threshold: int = 2, max_difficulty_nodes: int = 5) -> List[Node]:
        """为单个Subcategory生成Difficulty Level节点集合
        
        Args:
            domain_node: Domain节点
            subcategory_node: Subcategory节点
            max_iterations: 最大迭代次数
            stability_threshold: 节点集合稳定所需的迭代次数
            max_difficulty_nodes: 最大难度等级节点数量
            
        Returns:
            该Subcategory对应的Difficulty Level节点列表
        """
        max_retries = 3
        self.logger.info(f"开始为Subcategory '{subcategory_node.name}'生成Difficulty Level节点")
        current_nodes = []
        stability_count = 0
        
        for i in range(max_iterations):
            self.logger.info(f"开始第 {i+1} 次迭代")
            if not current_nodes:
                # 生成初始difficulty nodes
                for retry in range(max_retries):
                    self.logger.debug(f"尝试生成初始Difficulty Level节点 (尝试 {retry + 1}/{max_retries})")
                    prompt = DifficultyLevelNodePrompt.format(
                        domain_name=domain_node.name,
                        subcategory_name=subcategory_node.name,
                        subcategory_definition=subcategory_node.definition,
                        subcategory_example=subcategory_node.example,
                        max_difficulty_level_nodes=max_difficulty_nodes,
                        DifficultyLevelNodeRule=NodeGenRule[2].format(max_difficulty_level_nodes=max_difficulty_nodes)
                    )
                    response = await self._call_llm(prompt)
                    nodes, is_valid = self._parse_node_set(response, "<node begin>", "</node end>", "Level")
                    if is_valid:
                        current_nodes = nodes
                        self.logger.info(f"成功生成 {len(nodes)} 个初始Difficulty Level节点")
                        if len(nodes) > max_difficulty_nodes:
                            self.logger.warning(f"生成超过{max_difficulty_nodes}个Difficulty Level节点，请重新生成")
                            continue
                        else:
                            break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for difficulty level node generation")
                        continue
                    else:
                        self.logger.error(f"Failed to generate difficulty level nodes after {max_retries} retries")
                        exit(0)
            else:
                self.logger.debug("开始修改现有Difficulty Level节点")
                # 使用NodeRevisePrompt进行修改
                revise_prompt = NodeRevisePrompt.format(
                    node_name="Difficulty Level Node",
                    candidate_node_set=self._format_nodes_for_prompt(current_nodes, "Level"),
                    node_gen_rules=NodeGenRule[2].format(max_difficulty_level_nodes=max_difficulty_nodes),
                    special_note=f"Ensure difficulty levels are specific to the the output token length and LLM capability required. Please utilize a fixed global scale (e.g., Easy / Medium / Hard for three-level difficulty) as the short name of each level. However, the Definition of each level should be customized based on the nature of the specific Subcategory.",
                    node_name_short="Level"
                )
                
                # 添加重试逻辑
                revised_nodes = None
                is_valid = False
                
                for retry in range(max_retries):
                    self.logger.info(f"尝试修改Difficulty Level节点 (尝试 {retry + 1}/{max_retries})")
                    revised_response = await self._call_llm(revise_prompt)
                    revised_nodes, is_valid = self._parse_node_set(revised_response, "<revision node begin>", "</revision node end>", "Level")
                    if is_valid:
                        self.logger.info(f"成功生成候选Difficulty Level节点，生成 {len(revised_nodes)} 个节点")
                        if len(revised_nodes) > max_difficulty_nodes:
                            self.logger.warning(f"生成超过{max_difficulty_nodes}个Difficulty Level节点，请重新生成")
                            continue
                        else:
                            break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for node revision")
                        continue
                    else:
                        self.logger.error(f"Failed to revise nodes after {max_retries} retries")
                        exit(0)
                
                # 比较两个节点集合
                for retry in range(max_retries):
                    # 随机分配A/B顺序
                    import random
                    if random.random() < 0.5:
                        set_a, set_b = current_nodes, revised_nodes
                        is_current_a = True
                    else:
                        set_a, set_b = revised_nodes, current_nodes
                        is_current_a = False
                        
                    choice_prompt = NodeSetChoicePrompt.format(
                        node_name="Difficulty Level Node",
                        node_gen_rules=NodeGenRule[2].format(max_difficulty_level_nodes=max_difficulty_nodes),
                        special_note=f"Ensure difficulty levels are specific to the the output token length and LLM capability required.",
                        candidate_node_set_a=self._format_nodes_for_prompt(set_a, "Level"),
                        candidate_node_set_b=self._format_nodes_for_prompt(set_b, "Level")
                    )
                    
                    choice_response = await self._call_llm(choice_prompt)
                    preferred_set = self._extract_preferred_set(choice_response, "<preferred set>", "</preferred set>")
                    
                    if preferred_set:
                        # 根据A/B选择和随机分配确定是否使用revised_nodes
                        use_revised = (preferred_set == "A" and not is_current_a) or (preferred_set == "B" and is_current_a)
                        if use_revised:
                            self.logger.info(f"使用候选Difficulty Level节点")
                            if current_nodes == revised_nodes:
                                stability_count += 1
                            else:
                                stability_count = 0
                            current_nodes = revised_nodes
                        else:
                            self.logger.info(f"使用上轮Difficulty Level节点")
                            stability_count += 1
                        break
                    else:
                        if retry < max_retries - 1:
                            self.logger.warning(f"Retry {retry + 1} of {max_retries} for choice comparison")
                            continue
                        else:
                            self.logger.error(f"Failed to get valid choice after {max_retries} retries")
                            exit(0)
            
            if stability_count >= stability_threshold:
                break
                
        self.logger.info(f"完成Subcategory '{subcategory_node.name}'的Difficulty Level节点生成，共 {len(current_nodes)} 个节点")
        return current_nodes

    async def generate_preference_nodes(self, domain_node: Node, subcategory_node: Node, max_iterations: int = 5, stability_threshold: int = 2, max_preference_nodes: int = 6) -> List[Node]:
        """为单个Difficulty Level生成User Preference节点集合
        
        Args:
            domain_node: Domain节点
            subcategory_node: Subcategory节点
            max_iterations: 最大迭代次数
            stability_threshold: 节点集合稳定所需的迭代次数
            max_preference_nodes: 最大User Preference节点数量
            
        Returns:
            该Subcategory对应的User Preference节点列表
        """
        max_retries = 3
        self.logger.info(f"开始为Subcategory '{subcategory_node.name}'生成User Preference节点")
        current_nodes = []
        stability_count = 0
        
        for i in range(max_iterations):
            self.logger.info(f"开始第 {i+1} 次迭代")
            if not current_nodes:
                # 生成初始preference nodes
                for retry in range(max_retries):
                    self.logger.debug(f"尝试生成初始User Preference节点 (尝试 {retry + 1}/{max_retries})")
                    prompt = UserPreferenceNodePrompt.format(
                        domain_name=domain_node.name,
                        subcategory_name=subcategory_node.name,
                        subcategory_definition=subcategory_node.definition,
                        subcategory_example=subcategory_node.example,
                        max_user_preference_nodes=max_preference_nodes,
                        UserPreferenceNodeRule=NodeGenRule[3].format(max_user_preference_nodes=max_preference_nodes)
                    )
                    response = await self._call_llm(prompt)
                    nodes, is_valid = self._parse_node_set(response, "<node begin>", "</node end>", "Preference")
                    if is_valid:
                        current_nodes = nodes
                        self.logger.info(f"成功生成 {len(nodes)} 个初始User Preference节点")
                        if len(nodes) > max_preference_nodes:
                            self.logger.warning(f"生成超过{max_preference_nodes}个User Preference节点，请重新生成")
                            continue
                        else:
                            break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for user preference node generation")
                        continue
                    else:
                        self.logger.error(f"Failed to generate user preference nodes after {max_retries} retries")
                        exit(0)
            else:
                self.logger.debug("开始修改现有User Preference节点")
                # 使用NodeRevisePrompt进行修改
                revise_prompt = NodeRevisePrompt.format(
                    node_name="User Preference Node",
                    candidate_node_set=self._format_nodes_for_prompt(current_nodes, "Preference"),
                    node_gen_rules=NodeGenRule[3].format(max_user_preference_nodes=max_preference_nodes),
                    special_note=f"Ensure user preferences represent common output-related constraints or stylistic preferences that would influence output length and generation behavior. Must include a 'No Explicit Preference' node.",
                    node_name_short="Preference"
                )
                
                # 添加重试逻辑
                revised_nodes = None
                is_valid = False
                
                for retry in range(max_retries):
                    self.logger.info(f"尝试修改User Preference节点 (尝试 {retry + 1}/{max_retries})")
                    revised_response = await self._call_llm(revise_prompt)
                    revised_nodes, is_valid = self._parse_node_set(revised_response, "<revision node begin>", "</revision node end>", "Preference")
                    if is_valid:
                        self.logger.info(f"成功生成候选User Preference节点，生成 {len(revised_nodes)} 个节点")
                        if len(revised_nodes) > max_preference_nodes:
                            self.logger.warning(f"生成超过{max_preference_nodes}个User Preference节点，请重新生成")
                            continue
                        else:
                            break
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry + 1} of {max_retries} for node revision")
                        continue
                    else:
                        self.logger.error(f"Failed to revise nodes after {max_retries} retries")
                        exit(0)
                
                # 比较两个节点集合
                for retry in range(max_retries):
                    # 随机分配A/B顺序
                    import random
                    if random.random() < 0.5:
                        set_a, set_b = current_nodes, revised_nodes
                        is_current_a = True
                    else:
                        set_a, set_b = revised_nodes, current_nodes
                        is_current_a = False
                        
                    choice_prompt = NodeSetChoicePrompt.format(
                        node_name="User Preference Node",
                        node_gen_rules=NodeGenRule[3].format(max_user_preference_nodes=max_preference_nodes),
                        special_note=f"Ensure user preferences represent common output-related constraints or stylistic preferences that would influence output length and generation behavior.",
                        candidate_node_set_a=self._format_nodes_for_prompt(set_a, "Preference"),
                        candidate_node_set_b=self._format_nodes_for_prompt(set_b, "Preference")
                    )
                    
                    choice_response = await self._call_llm(choice_prompt)
                    preferred_set = self._extract_preferred_set(choice_response, "<preferred set>", "</preferred set>")
                    
                    if preferred_set:
                        # 根据A/B选择和随机分配确定是否使用revised_nodes
                        use_revised = (preferred_set == "A" and not is_current_a) or (preferred_set == "B" and is_current_a)
                        if use_revised:
                            self.logger.info(f"使用候选User Preference节点")
                            if current_nodes == revised_nodes:
                                stability_count += 1
                            else:
                                stability_count = 0
                            current_nodes = revised_nodes
                        else:
                            self.logger.info(f"使用上轮User Preference节点")
                            stability_count += 1
                        break
                    else:
                        if retry < max_retries - 1:
                            self.logger.warning(f"Retry {retry + 1} of {max_retries} for choice comparison")
                            continue
                        else:
                            self.logger.error(f"Failed to get valid choice after {max_retries} retries")
                            exit(0)
            
            if stability_count >= stability_threshold:
                break
                
        self.logger.info(f"完成Subcategory '{subcategory_node.name}'的User Preference节点生成，共 {len(current_nodes)} 个节点")
        return current_nodes

    def _format_nodes_for_prompt(self, nodes: List[Node], short_name: str) -> str:
        """将节点列表格式化为prompt字符串"""
        formatted = ["<node begin>"]
        for node in nodes:
            formatted.extend([
                f"{short_name}: {node.name}",
                f"Definition: {node.definition}",
                f"Example: {node.example}",
                ""
            ])
        formatted.append("</node end>")
        return "\n".join(formatted)

if __name__ == "__main__":
    async def main():
        # 初始化LLM提供商
        config = config
        kg_path = "/hdd2/lh/agenticrouter_data/kg_data_supp/kg_data.json"
        dir_path = "/hdd2/lh/agenticrouter_data/kg_data_supp"
        llm_provider = LLMProvider(
            provider=ProviderType.OPENAI,
            model="gemini-2.5-pro", # gpt-4.1-2025-04-14 gemini-2.5-pro
            config=config
        )
        generation_stage = "subcategory" # choice from ["domain", "subcategory", "difficulty", "preference"]
        skip_flag = True
        
        # 创建KG构建器
        kg_constructor = KGConstructor(llm_provider)
        
        if generation_stage == "domain":        
        # 生成Domain节点
            
            domain_nodes = await kg_constructor.generate_domain_nodes(max_iterations = 10, stability_threshold = 2)
            
            # 添加初始节点
            initial_nodes = [Node(name=domain['name'], definition=domain['definition'], example=domain['example'])
                            for domain in InitialDomains]
            domain_nodes.extend(initial_nodes)
            
            print("Generated Domain Nodes:")
            for node in domain_nodes:
                print(f"- {node.name}")
                
            # 保存domain nodes
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_data = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "nodes": [{"name": node.name, "definition": node.definition, "example": node.example} for node in domain_nodes]
            }
            with open(kg_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        elif generation_stage == "subcategory":
            # 生成Subcategory节点
            with open(kg_path, 'r', encoding='utf-8') as f:
                domain_nodes_data = json.load(f)
                domain_nodes = [Node(name=node['name'], definition=node['definition'], example=node['example']) 
                                for node in domain_nodes_data["nodes"]]
            
            for i, domain in enumerate(domain_nodes):
                if "subcategory_nodes" in domain_nodes_data["nodes"][i].keys() and skip_flag:
                    continue
                print(f"\nProcessing Domain: {domain.name}")
                subcategory_nodes = await kg_constructor.generate_subcategory_nodes(
                    domain_node=domain,
                    max_iterations=10,
                    stability_threshold=2,
                    max_subcategory_nodes=10
                )
                
                # 先读取当前数据
                with open(kg_path, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                # 更新数据
                save_data["nodes"][i]["subcategory_nodes"] = [
                    {"name": node.name, "definition": node.definition, "example": node.example} 
                    for node in subcategory_nodes
                ]
                
                # 保存更新后的数据
                with open(kg_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                print(f"Added {len(subcategory_nodes)} subcategory nodes for domain: {domain.name}")
                for node in subcategory_nodes:
                    print(f"  - {node.name}")
                
        elif generation_stage == "difficulty":      
            # 生成Difficulty Level节点          
            with open(kg_path, 'r', encoding='utf-8') as f:
                domain_nodes_data = json.load(f)
                domain_nodes = [Node(name=node['name'], definition=node['definition'], example=node['example']) 
                                for node in domain_nodes_data["nodes"]]
            
            for i, domain in enumerate(domain_nodes):
                print(f"\nProcessing Domain: {domain.name}")
                subcategory_nodes = [Node(name=node['name'], definition=node['definition'], example=node['example']) 
                                for node in domain_nodes_data["nodes"][i]["subcategory_nodes"]]
                for j, subcategory in enumerate(subcategory_nodes):
                    print(f"\nProcessing Subcategory: {subcategory.name}")
                    if "difficulty_nodes" in domain_nodes_data["nodes"][i]["subcategory_nodes"][j].keys() and skip_flag:
                        print(f"Skipping difficulty nodes for subcategory: {subcategory.name}")
                        continue
                    difficulty_nodes = await kg_constructor.generate_difficulty_nodes(
                        domain_node=domain,
                        subcategory_node=subcategory,
                        max_iterations=5,
                        stability_threshold=2,
                        max_difficulty_nodes=5
                    )
                    # 先读取当前数据
                    with open(kg_path, 'r', encoding='utf-8') as f:
                        save_data = json.load(f)
                    
                    # 更新数据
                    save_data["nodes"][i]["subcategory_nodes"][j]["difficulty_nodes"] = [
                        {"name": node.name, "definition": node.definition, "example": node.example} 
                        for node in difficulty_nodes
                    ]
                    
                    # 保存更新后的数据
                    with open(kg_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"Added {len(difficulty_nodes)} difficulty nodes for subcategory: {subcategory.name}")
                    for node in difficulty_nodes:
                        print(f"  - {node.name}")

        elif generation_stage == "preference":
            # 生成User Preference节点
            with open(kg_path, 'r', encoding='utf-8') as f:
                domain_nodes_data = json.load(f)
                domain_nodes = [Node(name=node['name'], definition=node['definition'], example=node['example']) 
                                for node in domain_nodes_data["nodes"]]
                
            for i, domain in enumerate(domain_nodes):
                print(f"\nProcessing Domain: {domain.name}")
                subcategory_nodes = [Node(name=node['name'], definition=node['definition'], example=node['example']) 
                                    for node in domain_nodes_data["nodes"][i]["subcategory_nodes"]]
                for j, subcategory in enumerate(subcategory_nodes):
                    print(f"\nProcessing Subcategory: {subcategory.name}")
                    if "preference_nodes" in domain_nodes_data["nodes"][i]["subcategory_nodes"][j].keys() and skip_flag:
                        continue
                    preference_nodes = await kg_constructor.generate_preference_nodes(
                        domain_node=domain,
                        subcategory_node=subcategory,
                        max_iterations=5,
                        stability_threshold=2,
                        max_preference_nodes=6
                    )
                    # 先读取当前数据
                    with open(kg_path, 'r', encoding='utf-8') as f:
                        save_data = json.load(f)
                    
                    # 更新数据
                    save_data["nodes"][i]["subcategory_nodes"][j]["preference_nodes"] = [
                        {"name": node.name, "definition": node.definition, "example": node.example} 
                        for node in preference_nodes
                    ]
                    
                    # 保存更新后的数据
                    with open(kg_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"Added {len(preference_nodes)} preference nodes for subcategory: {subcategory.name}")
                    for node in preference_nodes:
                        print(f"  - {node.name}")
    asyncio.run(main())
