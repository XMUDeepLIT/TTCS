import os
import json
import pandas as pd
import random
def data_merge(args):
    data_path_dir = args.data_path_dir
    save_path_dir = args.save_path_dir
    data_list = os.listdir(data_path_dir)
    datas = []
    for data_file in data_list:
        if data_file.endswith(".json"):
            with open(os.path.join(data_path_dir, data_file), "r") as f:
                datas.extend(json.load(f))
    os.makedirs(save_path_dir, exist_ok=True)
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
   
    real_data = []
    synthetic_data = []
    for idx, item in enumerate(datas):
        if item["score"] == 0:
            if item["is_synthetic"]:
                synthetic_data.append({
                    'data_source': f'Challenger_{args.exp_name}',
                    'problem':item["question"],
                    'prompt':[
                        {'role': 'system',   'content': instruction},
                        {
                        "role": "user",
                        'content':item["question"]
                    }],
                    'reward_model':{
                        'style':'rule', 
                    },
                    'ability': 'math',
                    'extra_info':{
                        'idx':idx,
                        'raw_data_source':item["data_source"],
                        'reference_question':item["reference_question"],
                        'question':item["question"],
                        'answer':item["answer"],
                        'score':item["score"],
                        'gen_q_prompt':item['prompt']
                    }
                })
            
            if not item["is_synthetic"] and args.hybrid_data:
                example=item["example"]
                real_data.append(example)
    seen_problems = set()
    for item in real_data:
        problem = item.get('problem')
        seen_problems.add(problem)
    synthetic_cnt_before = len(synthetic_data)
    unique_synthetic_data = []
    for item in synthetic_data:
        problem = item.get('problem')
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_synthetic_data.append(item)
    
    front_synthetic_cnt = min(len(real_data), len(unique_synthetic_data))
    front_synthetic = unique_synthetic_data[:front_synthetic_cnt]
    remain_synthetic = unique_synthetic_data[front_synthetic_cnt:]
    front_data = real_data + front_synthetic
    random.shuffle(front_data)
    
    random.shuffle(remain_synthetic)
    
    final_data = front_data + remain_synthetic
    with open(f'{save_path_dir}/train_data.jsonl', 'w',encoding='utf-8') as f:
        for line in final_data:
            f.write(json.dumps(line,ensure_ascii=False) + "\n")
    df = pd.DataFrame(final_data)
    df.to_parquet(f'{save_path_dir}/train_data.parquet')


            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_dir", type=str, default="", help="")
    parser.add_argument("--save_path_dir", type=str, default="", help="")
    parser.add_argument("--exp_name", type=str, default="test", help="")
    parser.add_argument("--hybrid_data", action="store_true", help="")
    args = parser.parse_args()

    data_merge(args)
    
    