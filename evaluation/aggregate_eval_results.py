#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate evaluation results from three datasets into a single json file
"""
import json
import os
import argparse
from pathlib import Path


def load_result_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def aggregate_eval_results(save_path_dir, model_name, output_file=None):
    """
    Aggregate evaluation results from three datasets
    
    Args:
        save_path_dir: Base directory for saving results
        model_name: Model name
        output_file: Output file path (optional)
    """
    model_dir = os.path.join(save_path_dir, model_name)
    
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory does not exist: {model_dir}")
        return
    
    result_files = {
        'bbeh': os.path.join(model_dir, 'bbeh_final_results.json'),
        'mmlupro': os.path.join(model_dir, 'mmlupro_final_results.json'),
        'supergpqa': os.path.join(model_dir, 'supergpqa_final_results.json')
    }
    
    results = {}
    for dataset_name, file_path in result_files.items():
        result = load_result_file(file_path)
        if result:
            results[dataset_name] = result
            print(f"Loaded {dataset_name} results from {file_path}")
        else:
            print(f"Warning: {dataset_name} results file not found: {file_path}")
    
    if not results:
        print(f"Warning: No results found for model {model_name}")
        return
    
    aggregated_result = {
        'model_name': model_name,
        'model_path': results.get('bbeh', {}).get('model_path') or 
                      results.get('mmlupro', {}).get('model_path') or 
                      results.get('supergpqa', {}).get('model_path'),
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
    
    if output_file is None:
        output_file = os.path.join(model_dir, 'aggregated_eval_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_result, f, ensure_ascii=False, indent=2)
    
    print(f"Aggregated results saved to: {output_file}")
    print(f"Overall accuracy: {aggregated_result['overall']['overall_accuracy']}%")
    print(f"Total samples: {total_samples}")


def aggregate_all_models(save_path_dir, model_list, output_file=None):
    """
    Aggregate results from all models into a single json file, also generate individual aggregated files for each model
    
    Args:
        save_path_dir: Base directory for saving results
        model_list: List of model names
        output_file: Output file path (optional)
    """
    if output_file is None:
        output_file = os.path.join(save_path_dir, 'all_models_aggregated_results.json')
    
    all_results = []
    for model_name in model_list:
        model_dir = os.path.join(save_path_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory does not exist: {model_dir}")
            continue
        
        result_files = {
            'bbeh': os.path.join(model_dir, 'bbeh_final_results.json'),
            'mmlupro': os.path.join(model_dir, 'mmlupro_final_results.json'),
            'supergpqa': os.path.join(model_dir, 'supergpqa_final_results.json')
        }
        
        results = {}
        for dataset_name, file_path in result_files.items():
            result = load_result_file(file_path)
            if result:
                results[dataset_name] = result
        
        if not results:
            print(f"Warning: No results found for model {model_name}")
            continue
        
        aggregated_result = {
            'model_name': model_name,
            'model_path': results.get('bbeh', {}).get('model_path') or 
                          results.get('mmlupro', {}).get('model_path') or 
                          results.get('supergpqa', {}).get('model_path'),
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
        
        model_output_file = os.path.join(model_dir, 'aggregated_eval_results.json')
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_result, f, ensure_ascii=False, indent=2)
        print(f"Aggregated results for {model_name} saved to: {model_output_file}")
        
        all_results.append(aggregated_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"All aggregated results saved to: {output_file}")
    print(f"Total models processed: {len(all_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate evaluation results from three datasets')
    parser.add_argument("--save_path_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model (optional, if not provided, will aggregate all models)")
    parser.add_argument("--model_list", type=str, nargs='+', default=None, help="List of model names (optional)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (optional)")
    args = parser.parse_args()
    
    if args.model_list:
        aggregate_all_models(args.save_path_dir, args.model_list, args.output_file)
    elif args.model_name:
        aggregate_eval_results(args.save_path_dir, args.model_name, args.output_file)
    else:
        print("Error: Either --model_name or --model_list must be provided")
        parser.print_help()

