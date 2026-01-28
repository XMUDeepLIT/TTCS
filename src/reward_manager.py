from collections import defaultdict,Counter

import torch
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from typing import Dict, List,Optional, Any
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np



def custom_extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return None


def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
                # sentence_bleu may return float or list[float], ensure we get a float
                score = score[0] if isinstance(score, list) else score
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(storage_path:str, prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    os.makedirs(f"{storage_path}/temp_results", exist_ok=True)
    return f"{storage_path}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=2):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def get_reward_server_config():
    """Get Reward Server config from environment variables"""
    ports_str = os.environ.get("SE_REWARD_PORTS", "5000,5001")
    ports = [int(p.strip()) for p in ports_str.split(",") if p.strip()]
    
    n_servers = int(os.environ.get("SE_N_REWARD_SERVERS", len(ports)))
    
    base_port = int(os.environ.get("SE_REWARD_BASE_PORT", 5000))
    while len(ports) < n_servers:
        ports.append(base_port + len(ports))
    
    return ports[:n_servers]

def fetch(port, filepath, question_reward):
    """Send request to Reward Server at specified port"""
    response = requests.get(f"http://0.0.0.0:{port}/hello?name={filepath}&question_reward={question_reward}")
    print(f"[fetch] port={port}, response={response}")
    return True

def generate_results(data, storage_path: str, question_reward: str):
    """Distribute data to multiple Reward Servers and collect results"""
    ports = get_reward_server_config()
    n_servers = len(ports)
    
    print(f"[generate_results] Using {n_servers} Reward Servers, ports: {ports}")
    
    datas = split_list(data, n_servers)
    random_names = [generate_temp_filename(storage_path=storage_path, prefix=f"temp_{i}") for i in range(n_servers)]
    
    for i in range(n_servers):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=n_servers) as executor:
        futures = [executor.submit(fetch, ports[i], random_names[i], question_reward) for i in range(n_servers)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(n_servers):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(n_servers):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results


@register("challenger")
class ChallengerRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='challenger',
        storage_path:str="",
        question_reward:str="R_Zero",
        gen_question_func:str="R_Zero",
        group_question_repetion_penalty=True
    ) -> None:
        assert storage_path is not None, "storage_path must be provided"
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.question_reward=question_reward
        self.group_question_repetion_penalty=group_question_repetion_penalty
        self.gen_question_func=gen_question_func
        os.makedirs(self.storage_path, exist_ok=True)
    def compute_score(self, predicts, reference_qs, storage_path:str, step=0):
        results = []
        for i in range(len(predicts)):
            questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
            if self.gen_question_func == "R_Zero" or self.gen_question_func == "weakness_icl" or self.gen_question_func == "ttrl_icl":
                answers = custom_extract_boxed_content(predicts[i])
            elif self.gen_question_func == "weakness":
                answers = re.findall(r"<answer>(.*?)</answer>", predicts[i], re.DOTALL)
            else:
                raise ValueError(f"Invalid gen_question_func: {self.gen_question_func}")
            if questions and answers:
                try:
                    question = questions[-1].strip()
                    answer = answers.strip()
                    results.append({"idx":i, "question": question, "answer": answer, "reference_question": reference_qs[i]})
                except:
                    results.append({"idx":i, "question": "", "answer": "", "reference_question": reference_qs[i]})
            else:
                results.append({"idx":i, "question": "", "answer": "", "reference_question": reference_qs[i]})

        final_results = generate_results(results, storage_path=storage_path, question_reward=self.question_reward)
        if self.group_question_repetion_penalty:
            penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
        else:
            penalty = [0] * len(final_results)
        # print(penalty)
        assert len(penalty) == len(final_results), f'{len(penalty)=}\n {len(final_results)=}'
        scores = []
        saved_results = []
        for i in range(len(final_results)):
            # Use uncertrainity_reward from vLLM server response, fallback to 0 if not available
            base_score = final_results[i].get("reward", 0)
            final_score = max(0, base_score - penalty[i]) if final_results[i]['question'] else 0
            scores.append({"score": final_score, "format": 1 if final_results[i]['question'] else 0,"repetition_penalty": penalty[i]})
            saved_results.append(
                {
                    "idx": i,
                    'step': step,
                    'Challenger_rollout': predicts[i],
                    'extracted_question':final_results[i]['question'],
                    'reward_info': final_results[i]['reward_info'],
                    'reward': final_results[i]['reward'],
                    'final_reward':final_score,
                    'error':final_results[i].get('error', None),
                }
            )
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        os.makedirs(reward_info_path_dir, exist_ok=True)
        step_str = str(step).zfill(3)
        with open(f"{reward_info_path_dir}/expdata_step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
            for result in saved_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return scores
    
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        qeurys = []
        reference_qs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            #valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            #prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            qeurys.append(response_str)
            reference_qs.append(data_item.non_tensor_batch["reference_question"])
        results = self.compute_score(qeurys, reference_qs, self.storage_path, step)
        for i in range(len(results)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            

            score: float
            result = results[i]
            
            if isinstance(result, dict):
                score = result["score"]
                
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            # TODO: add reward post-processing
            reward_tensor[i, valid_response_length - 1] = reward
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("solver")
class SolverRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        storage_path:str="",
        filter_lower= 0.0,
        filter_high= 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.low = filter_lower
        self.high = filter_high
        
    def compute_score(
        self,
        solution_str: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Compute the reward score for a solution.

        Args:
            solution_str: The solution string
            ground_truth: The ground truth answer

        Returns:
            Reward score (1.0 for correct, -1.0 for incorrect)
        """
        # Limit solution length for efficiency
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

        # Verify the solution
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        correct = False
        pred = custom_extract_boxed_content(solution_str)
        for gt in ground_truth:
            if pred is None:
                continue
            correct = grade_answer(str(pred), str(gt))
            if correct:
                break

        reward = 1.0 if correct  else 0.0
        acc = reward

        return {
            "score": reward,
            "acc": acc,
            "pred": pred if pred is not None else 'None',
        }
 
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        #topics = data.non_tensor_batch["topic"] if self.num_examine == 0 else data.non_tensor_batch["data_source"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        uids = data.non_tensor_batch["uid"]
        uid2labels = defaultdict(list)
        uid2all_labels = defaultdict(list)
        
        labels = []
        prompts = []
        responses = []
        responses_length = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            responses_length.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            label = custom_extract_boxed_content(response_str[-300:])
            uid2all_labels[uids[i]].append(label if label is not None else 'None') 

            if label is not None:
                uid2labels[uids[i]].append(label)
            
            prompts.append(prompt_str)
            responses.append(response_str)
        uid2ground_truths = defaultdict(lambda:None)   
        ground_truths = []
        for i in range(len(data)):
            if 'ground_truth' in data[i].non_tensor_batch['reward_model']:
                ground_truths.append(data[i].non_tensor_batch['reward_model']['ground_truth'])
            else:
                if uid2ground_truths[uids[i]] is not None:
                    ground_truths.append(uid2ground_truths[uids[i]])
                    continue
                answers_count = {}
                for res in list(uid2labels[uids[i]]):
                    if not res: continue
                    matched = False
                    for exist_ans in list(set(answers_count.keys())):
                        if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                            answers_count[exist_ans] += 1
                            matched = True
                            break
                        try:
                            is_match = False
                            is_match = grade_answer(str(res), str(exist_ans))
                            if is_match:
                                answers_count[exist_ans] += 1
                                matched = True
                                break
                        except Exception as e:
                            print(f"Error comparing '{res}' and '{exist_ans}': {e}")
                            continue
                    if not matched:
                        answers_count[res] = 1
                if not answers_count:
                    majority_ans, max_count = '', 0
                else:
                    majority_ans = max(answers_count, key=answers_count.get)
                    max_count = answers_count[majority_ans]
                uid2ground_truths[uids[i]]=majority_ans
                ground_truths.append(majority_ans)
        reward_infos = []
        uid2group_acc = defaultdict(list)
        for i in range(len(data)):
            if self.num_examine > 0: # val data
                expected_gt = data[i].non_tensor_batch['reward_model'].get('ground_truth', [])
                assert expected_gt == ground_truths[i], f"val data ground_truth is not correct: expected {expected_gt}, got {ground_truths[i]}" 
            
            result = self.compute_score(responses[i], ground_truths[i])
            score: float
            valid_response_length = responses_length[i]
            if isinstance(result, dict):
                score = result["score"]
                uid2group_acc[uids[i]].append(result["acc"])
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                uid2group_acc[uids[i]].append(score)
                reward_extra_info["acc"].append(score)
            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            
            reward_infos.append(
                {
                    'idx':i,
                    "step": step,
                    "style": "val" if self.num_examine > 0 else "exp",
                    "question": prompts[i],
                    'raw_question': data[i].non_tensor_batch['raw_prompt'],
                    "response": responses[i],
                    "pred": result.get("pred", ""),
                    "all_labels": uid2all_labels[uids[i]],
                    'filtered_labels': uid2labels[uids[i]],
                    "ground_truth(majority)": ground_truths[i],
                    "reward": reward,
                }
            )
            
        for i, example in enumerate(reward_infos):
            acc_mean=float(np.mean(uid2group_acc[uids[i]]))
            example.update({
                'uid2group_acc':uid2group_acc[uids[i]],
                'uid2acc_mean': acc_mean,
                'is_kept': bool(acc_mean >= self.low and acc_mean <= self.high)
            })
            #print(f'{i=},{example=}')
        

        reward_extra_info['reward_infos'] = reward_infos
        #reward_infos =[example.update({'uid2group_acc':uid2group_acc[uids[i]]}) for i, example in enumerate(reward_infos)]
        #/root/users/ycy/Self-evolving-Agent/saved_results/Solver/Qwen3-4B-Base-V1
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        step_str = str(step).zfill(3)
        if self.num_examine > 0:
            os.makedirs(f"{reward_info_path_dir}/valdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/valdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        else:
            os.makedirs(f"{reward_info_path_dir}/expdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/expdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

