import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import os, sys
import random
import json
import regex as re
from src.Challenger_dataset import get_prompts
from src.reward_manager import custom_extract_boxed_content


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.8,
        seed=int(args.suffix),
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    dataframe=get_prompts(num_querys=args.num_samples, get_prompts_func=args.get_prompts_func, icl_files=args.train_file)
    prompt = [
            tokenizer.apply_chat_template(
                example['prompt'], 
                tokenize=False,
                add_generation_prompt=True, 
                add_special_tokens=True,
            ) 
            for example in dataframe
        ]
    if random.randint(0,64)==0:
        print(f'{prompt[0]=}')
    completions: List[RequestOutput] = model.generate(prompt, sampling_params=sample_params,use_tqdm=False)
    results=[]
    
    for idx, completion in enumerate(completions):
        response = completion.outputs[0].text
        reference_question = dataframe[idx].get('reference_question','')
        test_item = dataframe[idx].get('test_item',None)
        data_source = dataframe[idx].get('data_source','')
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            #answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if args.get_prompts_func == "R_Zero" or args.get_prompts_func == "ttrl_icl" or args.get_prompts_func == "weakness_icl":
                answers = custom_extract_boxed_content(response)
            elif args.get_prompts_func == "weakness":
                answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
            else:
                answers="None"
                print("Warning: get_prompts_func is not supported extracted answers, default to None")

            if questions and answers:
                question = questions[-1].strip()                
                answer = answers.strip()
                results.append({
                    "idx": idx, 
                    "data_source": data_source,
                    'prompt': prompt[idx],
                    'reference_question': reference_question,
                    'response':response,
                    "question": question,                   
                    'answer': answer,
                    "score": 0,
                    'is_synthetic': True
                })
            else:
                results.append({
                    "idx": idx, 
                    "data_source": data_source,
                    'prompt': prompt[idx],
                    'reference_question': reference_question,
                    'response':response,
                    "question": '', 
                    'answer': '',
                    "score": -1
                })
            if test_item is not None:
                results.append({
                    "example": test_item,
                    "is_synthetic": False,
                    "score": 0,
                })
        except:
            results.append({
                "idx": idx, 
                'prompt': prompt[idx],
                'reference_question': reference_question,
                'response':response,
                "data_source": data_source,
                "question": '', 
                'answer': '',
                "score": -1
            })
    random.shuffle(results)
    os.makedirs(args.storage_path, exist_ok=True)
    with open(f"{args.storage_path}/{args.suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="1", help="Suffix to add to the output file")
    parser.add_argument("--storage_path", type=str, default="", help="")
    parser.add_argument("--get_prompts_func", type=str, default="R_Zero", help="Function to get prompts")
    parser.add_argument("--train_file", type=str, default="", help="Train file")
    #parser.add_argument("--save_name", type=str, default="challenger_generated_question", help="")
    args = parser.parse_args()
    print(f"[train_file]: {args.train_file}")


    main(args) 