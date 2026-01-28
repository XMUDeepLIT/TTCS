#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate evaluation results organized by step
Supports step_20, step_40, ... directory structure
Base model results as step 0
"""
import json
import os
import re
import argparse
from pathlib import Path


def load_result_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_all_steps(save_path_dir, base_model_dir=None):
    """
    Get all step directories
    
    Returns:
        steps: [(step_number, step_dir_path), ...], sorted by step
    """
    steps = []
    
    if base_model_dir and os.path.exists(base_model_dir):
        steps.append((0, base_model_dir))
    
    if os.path.exists(save_path_dir):
        for name in os.listdir(save_path_dir):
            path = os.path.join(save_path_dir, name)
            if os.path.isdir(path):
                match = re.match(r'^step_(\d+)$', name)
                if match:
                    step_num = int(match.group(1))
                    steps.append((step_num, path))
    
    steps.sort(key=lambda x: x[0])
    return steps


def aggregate_step_results(step_dir, step_num):
    """
    Aggregate evaluation results for a single step
    
    Args:
        step_dir: Step directory path
        step_num: Step number
    
    Returns:
        Aggregated result dictionary
    """
    result_files = {
        'bbeh': os.path.join(step_dir, 'bbeh_final_results.json'),
        'mmlupro': os.path.join(step_dir, 'mmlupro_final_results.json'),
        'supergpqa': os.path.join(step_dir, 'supergpqa_final_results.json')
    }
    
    results = {}
    for dataset_name, file_path in result_files.items():
        result = load_result_file(file_path)
        if result:
            results[dataset_name] = result
    
    if not results:
        return None
    
    aggregated_result = {
        'step': step_num,
        'step_dir': step_dir,
        'datasets': {}
    }
    
    for dataset_name, result in results.items():
        aggregated_result['datasets'][dataset_name] = {
            'dataset': result.get('dataset', dataset_name),
            'accuracy': result.get('accuracy') or result.get('micro_accuracy'),
            'macro_accuracy': result.get('macro_accuracy'),
            'success': result.get('success', 0),
            'fail': result.get('fail', 0),
            'total': result.get('total', 0),
            'per_category_accuracy': result.get('per_category_accuracy', {})
        }
    
    total_success = sum(r.get('success', 0) for r in aggregated_result['datasets'].values())
    total_fail = sum(r.get('fail', 0) for r in aggregated_result['datasets'].values())
    total_samples = sum(r.get('total', 0) for r in aggregated_result['datasets'].values())
    
    aggregated_result['overall'] = {
        'total_success': total_success,
        'total_fail': total_fail,
        'total_samples': total_samples,
        'overall_accuracy': round(total_success / total_samples * 100, 2) if total_samples > 0 else 0.0
    }
    
    return aggregated_result


def aggregate_all_steps(save_path_dir, base_model_dir=None, output_file=None):
    """
    Aggregate evaluation results from all steps
    
    Args:
        save_path_dir: Evaluation results directory (containing step_* subdirectories)
        base_model_dir: Base model evaluation results directory (as step 0)
        output_file: Output file path (optional)
    """
    steps = get_all_steps(save_path_dir, base_model_dir)
    
    if not steps:
        print(f"Warning: No step directories found in {save_path_dir}")
        return
    
    print(f"Found {len(steps)} steps: {[s[0] for s in steps]}")
    
    all_results = []
    
    for step_num, step_dir in steps:
        print(f"Processing step {step_num}: {step_dir}")
        
        result = aggregate_step_results(step_dir, step_num)
        
        if result:
            step_output_file = os.path.join(step_dir, 'aggregated_eval_results.json')
            with open(step_output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Saved to: {step_output_file}")
            
            all_results.append(result)
        else:
            print(f"  Warning: No results found for step {step_num}")
    
    if output_file is None:
        output_file = os.path.join(save_path_dir, 'all_steps_aggregated_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll steps aggregated results saved to: {output_file}")
    print(f"Total steps processed: {len(all_results)}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for result in all_results:
        step = result['step']
        datasets = result['datasets']
        overall = result['overall']
        
        print(f"\nStep {step}:")
        for name, data in datasets.items():
            acc = data.get('accuracy', 'N/A')
            print(f"  {name}: {acc}%")
        print(f"  Overall: {overall['overall_accuracy']}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate evaluation results organized by step')
    parser.add_argument("--save_path_dir", type=str, required=True,
                        help="Directory containing step_* subdirectories with evaluation results")
    parser.add_argument("--base_model_dir", type=str, default=None,
                        help="Directory containing base model evaluation results (as step 0)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path for aggregated results (optional)")
    
    args = parser.parse_args()
    aggregate_all_steps(args.save_path_dir, args.base_model_dir, args.output_file)
