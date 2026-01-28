import datasets
import json
import re
import random
import argparse
import os
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_max_position_embeddings(model_path):
    try:
        config = AutoConfig.from_pretrained(model_path)
        return getattr(config, 'max_position_embeddings', None)
    except Exception as e:
        print(f"Warning: Failed to read model config: {e}")
        return None


def extract_last_boxed(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None

def extract_last_final_answer(text):
    pattern1 = r'Final Answer:((?:[^<]|<[^<])*?)\n'
    pattern2 = r'The answer is:((?:[^<]|<[^<])*?)\n'
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    return None

def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return extract_last_final_answer(model_output)

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}\n'
    return option_str

def get_prediction(output):
    solution = extract_solution(output)
    if solution is None:
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    for option in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        if option in solution:
            return option
    return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--save_path_dir", type=str, default="", help="Directory to save results")
    parser.add_argument("--data_path_dir", type=str, default="", help="Directory to save results")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to evaluate per category (0-1, 0 means no subsampling, default 0.1)",
    )
    parser.add_argument("--output_file", type=str, default=None, help="File to save results (optional, will use save_path_dir/model_name if not provided)")
    args = parser.parse_args()
    
    # Create save directory
    save_dir = os.path.join(args.save_path_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    data_path=os.path.join(args.data_path_dir, 'MMLU-Pro')
    if os.path.exists(data_path):
        print(f'load dataset from local: {data_path}')
    else:
        data_path='TIGER-Lab/MMLU-Pro'
        print(f'Warning: local dataset not found, load from huggingface: {data_path}')
    dataset = datasets.load_dataset(data_path)
    # Set output file if not provided
    if args.output_file is None:
        args.output_file = os.path.join(save_dir, "mmlupro_outputs.json")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    max_pos_emb = get_max_position_embeddings(args.model_path)
    
    if max_pos_emb is not None and max_pos_emb <= 4096:
        FILTER_PROMPT = True
        MAX_PROMPT_LEN = 2048
        MAX_TOKENS = 2048
        print(f"max_position_embeddings={max_pos_emb}, enabling prompt filter (>{MAX_PROMPT_LEN} tokens)")
    elif max_pos_emb is not None and max_pos_emb <= 8192:
        FILTER_PROMPT = True
        MAX_PROMPT_LEN = 2048
        MAX_TOKENS = 6144
        print(f"max_position_embeddings={max_pos_emb}, enabling prompt filter (>{MAX_PROMPT_LEN} tokens)")
    else:
        FILTER_PROMPT = False
        MAX_PROMPT_LEN = None
        MAX_TOKENS = 8192
        print(f"max_position_embeddings={max_pos_emb}, no prompt filter")
    
    llm = LLM(model=args.model_path, tensor_parallel_size=1, gpu_memory_utilization=0.85)
    
    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']
    # For each category store [correct_count, incorrect_count]
    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0
    answers = []
    
    print('----------------- Start Answering -------------------')
    # Use a fixed random seed for reproducible subsampling
    random.seed(42)
    sample_ratio = max(0.0, min(1.0, getattr(args, "sample_ratio", 1.0)))

    # Compute and print total number of raw samples and planned samples (after subsampling)
    total_raw_samples = 0
    total_planned_samples = 0
    for category in categories:
        category_entries = [entry for entry in dataset['test'] if entry['category'] == category]
        total_raw_samples += len(category_entries)
        if 0.0 < sample_ratio < 1.0 and len(category_entries) > 0:
            sample_size = max(1, int(len(category_entries) * sample_ratio))
        else:
            sample_size = len(category_entries)
        total_planned_samples += sample_size
    print(f"Total raw samples in dataset (test split): {total_raw_samples}")
    print(f"Total samples planned to evaluate (after sampling): {total_planned_samples}")
    
    total_filtered = 0
    
    print("\n--- Phase 1: Filter long prompts ---")
    category_data = {}
    
    for category in categories:
        all_category_entries = [entry for entry in dataset['test'] if entry['category'] == category]
        
        if 0.0 < sample_ratio < 1.0 and len(all_category_entries) > 0:
            target_size = max(1, int(len(all_category_entries) * sample_ratio))
        else:
            target_size = len(all_category_entries)
        
        valid_entries = []
        valid_prompts = []
        filtered_count = 0
        
        for entry in all_category_entries:
            query = entry['question'] + '\n' + form_options(entry['options']) + '\n'
            messages = [{
                "role": "user",
                "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
            }]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "user: " + query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
            
            if FILTER_PROMPT:
                prompt_len = len(tokenizer.encode(prompt))
                if prompt_len > MAX_PROMPT_LEN:
                    filtered_count += 1
                    continue
            
            valid_entries.append(entry)
            valid_prompts.append(prompt)
        
        if filtered_count > 0:
            print(f"  [{category}] Filtered {filtered_count} long prompts, remaining {len(valid_entries)}/{len(all_category_entries)}")
            total_filtered += filtered_count
        
        category_data[category] = {
            'valid_entries': valid_entries,
            'valid_prompts': valid_prompts,
            'original_size': len(all_category_entries),
            'target': target_size
        }
    
    print("\n--- Phase 2: Allocate sample sizes ---")
    global_target = int(total_raw_samples * sample_ratio)
    total_valid = sum(len(d['valid_entries']) for d in category_data.values())
    sum_of_targets = sum(d['target'] for d in category_data.values())
    
    rounding_deficit = global_target - sum_of_targets
    if rounding_deficit > 0:
        print(f"  Rounding deficit: {rounding_deficit} samples need to be added")
    
    deficit_total = 0
    surplus_categories = []
    
    for category, data in category_data.items():
        valid_count = len(data['valid_entries'])
        target = data['target']
        if valid_count < target:
            deficit_total += target - valid_count
            data['sample_size'] = valid_count
        else:
            surplus = valid_count - target
            data['sample_size'] = target
            if surplus > 0:
                surplus_categories.append((category, surplus))
    
    total_deficit = deficit_total + rounding_deficit
    
    if total_deficit > 0 and surplus_categories:
        print(f"  Need to add {total_deficit} samples from other categories (filter deficit={deficit_total}, rounding deficit={rounding_deficit})")
        surplus_categories.sort(key=lambda x: x[1], reverse=True)
        remaining_deficit = total_deficit
        
        for category, surplus in surplus_categories:
            if remaining_deficit <= 0:
                break
            extra = min(surplus, remaining_deficit)
            category_data[category]['sample_size'] += extra
            remaining_deficit -= extra
            print(f"    [{category}] Extra sampling +{extra}")
    
    print("\n--- Phase 3: Execute evaluation ---")
    for category in categories:
        data = category_data[category]
        valid_entries = data['valid_entries']
        valid_prompts = data['valid_prompts']
        sample_size = data['sample_size']
        target = data['target']
        
        if len(valid_entries) == 0:
            print(f"  [{category}] Warning: All samples filtered, skipping category")
            continue
        
        if sample_size < len(valid_entries):
            indices = random.sample(range(len(valid_entries)), sample_size)
            category_entries = [valid_entries[i] for i in indices]
            prompts = [valid_prompts[i] for i in indices]
        else:
            category_entries = valid_entries
            prompts = valid_prompts
        
        extra_info = f"+{sample_size - target}" if sample_size > target else ""
        print(f"  [{category}] Sampling {len(category_entries)} (target={target}{extra_info}, valid={len(valid_entries)})")
        
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=MAX_TOKENS)
        outputs = llm.generate(prompts, sampling_params)
        
        # Process results concurrently
        def process_entry(entry_output_pair):
            entry, output = entry_output_pair
            answer = output.outputs[0].text
            entry['solution'] = answer
            prediction = get_prediction(answer)
            is_correct = entry["answer"] == prediction
            return entry, is_correct
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_entry, (entry, output)) 
                      for entry, output in zip(category_entries, outputs)]
            
            for future in as_completed(futures):
                entry, is_correct = future.result()
                #answers.append(entry)
                if is_correct:
                    success += 1
                    per_category_accuracy[category][0] += 1
                else:
                    fail += 1
                    per_category_accuracy[category][1] += 1
            
        # Print category accuracy as soon as it's computed
        total_cat = per_category_accuracy[category][0] + per_category_accuracy[category][1]
        cat_accuracy = per_category_accuracy[category][0] / total_cat if total_cat > 0 else 0.0
        print(f"{category}: {cat_accuracy:.4f}")
    
    # Save all the answers in a JSON file
    # with open(args.output_file, 'w') as f:
    #     json.dump(answers, f, indent=2)
    
    # Calculate per-category report, micro average, and macro average
    print("\n----- Accuracy Report -----")
    category_accuracy_report = {}
    for category in categories:
        correct, incorrect = per_category_accuracy[category]
        total = correct + incorrect
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0
        category_accuracy_report[category] = accuracy
        print(f"{category}: {correct}/{total} -> {accuracy*100:.2f}% accuracy")
        
    total_predictions = success + fail
    micro_avg = success / total_predictions if total_predictions > 0 else 0.0
    target_samples = int(total_raw_samples * sample_ratio)
    actual_ratio = total_predictions / total_raw_samples * 100 if total_raw_samples > 0 else 0
    meet_target = "✓" if total_predictions >= target_samples else "✗"
    
    print(f'\n----------------------------------')
    print(f'Sampling Statistics:')
    print(f'  Raw samples: {total_raw_samples}')
    print(f'  Target samples: {target_samples} ({sample_ratio*100:.0f}%)')
    print(f'  Evaluated: {total_predictions} ({actual_ratio:.1f}%)')
    print(f'  Filtered long prompts: {total_filtered}')
    print(f'  Met target: {meet_target}')
    print(f'----------------------------------')
    print(f"Micro Average Accuracy: {micro_avg*100:.2f}%")
    
    valid_categories = [cat for cat in categories if (per_category_accuracy[cat][0] + per_category_accuracy[cat][1] > 0)]
    if valid_categories:
        macro_avg = sum(category_accuracy_report[cat] for cat in valid_categories) / len(valid_categories)
    else:
        macro_avg = 0.0
    print(f"Macro Average Accuracy: {macro_avg*100:.2f}%")
    
    # Save final results to save_dir
    final_results_file = os.path.join(save_dir, "mmlupro_final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            "dataset": "mmlupro",
            "model": args.model_name,
            "model_path": args.model_path,
            "micro_accuracy": round(micro_avg * 100, 2),
            "macro_accuracy": round(macro_avg * 100, 2),
            "success": success,
            "fail": fail,
            "total": total_predictions,
            "sample_ratio": args.sample_ratio,
            "total_filtered": total_filtered,
            "per_category_accuracy": category_accuracy_report
        }, f, indent=2)
    
    print(f"Results saved to: {args.output_file}")
    print(f"Final results saved to: {final_results_file}")
