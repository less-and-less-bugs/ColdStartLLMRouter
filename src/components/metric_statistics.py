"""
分析不同模型和数据集上的指标统计信息
"""
import os
import json
import pandas as pd
import numpy as np
import torch
from typing import Dict, List
from pathlib import Path

from src.utils.data_loader import DatasetGen, normalize_metric
from src.components.agenticrouter_multistage import MODEL_NAMES, TASK_NAMES, METRICS

def calculate_metric_statistics(data_dir: str, 
                              task_names: List[str] = TASK_NAMES,
                              model_names: List[str] = MODEL_NAMES,
                              metrics: List[str] = METRICS):
    """
    计算不同模型和数据集上的指标统计信息
    
    Args:
        data_dir: 数据目录
        task_names: 任务名称列表
        model_names: 模型名称列表
        metrics: 指标名称列表
    
    Returns:
        Dict: 包含统计信息的字典
    """
    # 初始化数据加载器
    data_loader = DatasetGen(
        data_dir=data_dir,
        task_names=task_names,
        model_names=model_names
    )
    
    # 加载数据
    train_data, val_data, test_data = data_loader.load_data()
    
    # 获取归一化参数
    data_loader.get_max_min_values_per_task(train_data, val_data, test_data)
    
    # 初始化结果字典
    statistics = {
        "per_task": {},
        "per_model": {},
        "per_task_model": {},  # 新增：按任务和模型的交叉统计
        "overall": {}
    }
    
    # 合并所有数据集以进行整体分析
    all_data = pd.concat([train_data, val_data, test_data])
    
    # 1. 按任务统计
    for task in task_names:
        task_data = all_data[all_data['task_description'] == task]
        if len(task_data) == 0:
            continue
            
        task_stats = {
            "raw": {},
            "normalized": {},
            "sample_count": len(task_data)
        }
        
        for metric in metrics:
            raw_values = task_data[metric].tolist()
            if metric == 'cost':
                norm_values = task_data['normalized_cost'].tolist()
            elif metric == 'latency':
                norm_values = task_data['normalized_latency'].tolist()
            else:
                norm_values = raw_values  # effectiveness 不需要归一化
                
            # 原始值统计
            raw_stats = {
                "mean": np.mean(raw_values),
                "std": np.std(raw_values),
                "min": np.min(raw_values),
                "max": np.max(raw_values),
                "median": np.median(raw_values)
            }
            
            # 归一化值统计
            norm_stats = {
                "mean": np.mean(norm_values),
                "std": np.std(norm_values),
                "min": np.min(norm_values),
                "max": np.max(norm_values),
                "median": np.median(norm_values)
            }
            
            task_stats["raw"][metric] = raw_stats
            task_stats["normalized"][metric] = norm_stats
            
        statistics["per_task"][task] = task_stats
    
    # 2. 按模型统计
    for i, model in enumerate(model_names):
        model_stats = {
            "raw": {},
            "normalized": {},
            "per_task": {}
        }
        
        # 整体统计
        for metric in metrics:
            l = len(all_data)
            raw_values = [all_data[metric].iloc[j][i] for j in range(l)] 
            if metric == 'cost':
                norm_values = [all_data['normalized_cost'].iloc[j][i] for j in range(l)]
            elif metric == 'latency':
                norm_values = [all_data['normalized_latency'].iloc[j][i] for j in range(l)]
            else:
                norm_values = raw_values
                
            raw_stats = {
                "mean": np.mean(raw_values),
                "std": np.std(raw_values),
                "min": np.min(raw_values),
                "max": np.max(raw_values),
                "median": np.median(raw_values)
            }
            
            norm_stats = {
                "mean": np.mean(norm_values),
                "std": np.std(norm_values),
                "min": np.min(norm_values),
                "max": np.max(norm_values),
                "median": np.median(norm_values)
            }
            
            model_stats["raw"][metric] = raw_stats
            model_stats["normalized"][metric] = norm_stats
        
        # 按任务统计
        for task in task_names:
            task_data = all_data[all_data['task_description'] == task]
            if len(task_data) == 0:
                continue
                
            task_model_stats = {
                "raw": {},
                "normalized": {},
                "sample_count": len(task_data)
            }
            
            for metric in metrics:
                raw_values = [task_data[metric].iloc[j][i] for j in range(len(task_data))]
                if metric == 'cost':
                    norm_values = [task_data['normalized_cost'].iloc[j][i] for j in range(len(task_data))]
                elif metric == 'latency':
                    norm_values = [task_data['normalized_latency'].iloc[j][i] for j in range(len(task_data))]
                else:
                    norm_values = raw_values
                    
                raw_stats = {
                    "mean": np.mean(raw_values),
                    "std": np.std(raw_values),
                    "min": np.min(raw_values),
                    "max": np.max(raw_values),
                    "median": np.median(raw_values)
                }
                
                norm_stats = {
                    "mean": np.mean(norm_values),
                    "std": np.std(norm_values),
                    "min": np.min(norm_values),
                    "max": np.max(norm_values),
                    "median": np.median(norm_values)
                }
                
                task_model_stats["raw"][metric] = raw_stats
                task_model_stats["normalized"][metric] = norm_stats
                
            model_stats["per_task"][task] = task_model_stats
            
        statistics["per_model"][model] = model_stats
    
    # 3. 整体统计
    overall_stats = {
        "raw": {},
        "normalized": {},
        "sample_count": len(all_data)
    }
    
    for metric in metrics:
        raw_values = [all_data[metric].iloc[j] for j in range(len(all_data))]
        if metric == 'cost':
            norm_values = [all_data['normalized_cost'].iloc[j] for j in range(len(all_data))]
        elif metric == 'latency':
            norm_values = [all_data['normalized_latency'].iloc[j] for j in range(len(all_data))]
        else:
            norm_values = raw_values
            
        raw_stats = {
            "mean": np.mean(raw_values),
            "std": np.std(raw_values),
            "min": np.min(raw_values),
            "max": np.max(raw_values),
            "median": np.median(raw_values)
        }
        
        norm_stats = {
            "mean": np.mean(norm_values),
            "std": np.std(norm_values),
            "min": np.min(norm_values),
            "max": np.max(norm_values),
            "median": np.median(norm_values)
        }
        
        overall_stats["raw"][metric] = raw_stats
        overall_stats["normalized"][metric] = norm_stats
        
    statistics["overall"] = overall_stats
    
    return statistics

def main():
    """主函数"""
    # 设置数据目录
    data_dir = "/hdd2/lh/agenticrouter_data/data"
    
    # 计算统计信息
    statistics = calculate_metric_statistics(data_dir)
    
    # 保存结果
    output_dir = "/hdd2/lh/agenticrouter/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "metric_statistics.json")
    with open(output_file, "w") as f:
        json.dump(statistics, f, indent=2)
    
    print(f"统计结果已保存到: {output_file}")
    
    # 打印一些关键统计信息
    print("\n整体统计:")
    for metric in METRICS:
        raw_stats = statistics["overall"]["raw"][metric]
        norm_stats = statistics["overall"]["normalized"][metric]
        print(f"\n{metric}指标:")
        print(f"  原始值 - 均值: {raw_stats['mean']:.4f}, 标准差: {raw_stats['std']:.4f}")
        print(f"  归一化 - 均值: {norm_stats['mean']:.4f}, 标准差: {norm_stats['std']:.4f}")
    
    print("\n按模型统计:")
    for model in MODEL_NAMES:
        print(f"\n{model}:")
        for metric in METRICS:
            raw_stats = statistics["per_model"][model]["raw"][metric]
            norm_stats = statistics["per_model"][model]["normalized"][metric]
            print(f"  {metric}指标:")
            print(f"    原始值 - 均值: {raw_stats['mean']:.4f}, 标准差: {raw_stats['std']:.4f}")
            print(f"    归一化 - 均值: {norm_stats['mean']:.4f}, 标准差: {norm_stats['std']:.4f}")
    print("\n按任务和模型交叉统计:")
    for task in TASK_NAMES:
        for model in MODEL_NAMES:
            print(f"\n{task} - {model}:")
            for metric in METRICS:
                raw_stats = statistics["per_model"][model]["per_task"][task]["raw"][metric]
                norm_stats = statistics["per_model"][model]["per_task"][task]["normalized"][metric]
                print(f"  {metric}指标:")
                print(f"    原始值 - 均值: {raw_stats['mean']:.4f}, 标准差: {raw_stats['std']:.4f}")
                print(f"    归一化 - 均值: {norm_stats['mean']:.4f}, 标准差: {norm_stats['std']:.4f}")

    print("\n按任务统计:")
    for task in TASK_NAMES:
        print(f"\n{task}:")
        for metric in METRICS:
            raw_stats = statistics["per_task"][task]["raw"][metric]
            norm_stats = statistics["per_task"][task]["normalized"][metric]
            print(f"  {metric}指标:")
            print(f"    原始值 - 均值: {raw_stats['mean']:.4f}, 标准差: {raw_stats['std']:.4f}")
            print(f"    归一化 - 均值: {norm_stats['mean']:.4f}, 标准差: {norm_stats['std']:.4f}")

if __name__ == "__main__":
    main()
