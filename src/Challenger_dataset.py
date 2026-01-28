import copy
import logging
import os
import random
import re
import json
from collections import defaultdict
from typing import Optional
from string import Template
import math
import pandas as pd
import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


logger = logging.getLogger(__name__)




def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}



def make_json_serializable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'to_dict'):
        return make_json_serializable(obj.to_dict())
    else:
        return obj

        
def load_data(path):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path).to_dict(orient='records')
    elif path.endswith('.jsonl'):
        data = pd.read_json(path, lines=True)
        return data.to_dict(orient='records')
    else:
        raise ValueError(f"Invalid file type: {path}")
def get_prompts_weakness_icl(num_querys: int, icl_files: str):
    template_path =  'prompt_template.txt'
    with open(template_path, 'r', encoding='utf-8') as f:
        raw_prompt = f.read()
    icl_data = load_data(icl_files)

    groups = {}
    for item in icl_data:
        src = item.get("data_source", "unknown")
        groups.setdefault(src, []).append(item)

    if not groups:
        return []

    sources = list(groups.keys())


    weights = {}
    for src in sources:
        base_w = len(groups[src])
        w = float(base_w)
        if "AIME" in str(src).upper():
            w *= 10.0
        elif "AMC" in str(src).upper():
            w *= 5.0
        weights[src] = w

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return []

    alloc = {}
    frac = {}
    for src in sources:
        exact = num_querys * weights[src] / total_weight
        k = int(exact)
        alloc[src] = k
        frac[src] = exact - k

    current_total = sum(alloc.values())
    if current_total < num_querys:
        need = num_querys - current_total
        for src in sorted(sources, key=lambda s: frac[s], reverse=True):
            if need <= 0:
                break
            alloc[src] += 1
            need -= 1
    elif current_total > num_querys:
        need = current_total - num_querys
        for src in sorted(sources, key=lambda s: frac[s]):
            if need <= 0:
                break
            if alloc[src] > 0:
                alloc[src] -= 1
                need -= 1

    selected_items = []
    for src in sources:
        group = groups[src]
        n_samples = alloc.get(src, 0)
        if n_samples <= 0:
            continue
        if n_samples >= len(group):
            selected_items.extend(random.choices(group, k=n_samples))
        else:
            selected_items.extend(random.sample(group, k=n_samples))

    random.shuffle(selected_items)
    
    
    dataframe = []
    for idx, item in enumerate(selected_items):
        reference_question = (
            item.get("problem")
            or item.get("extra_info", {}).get("problem", "")
            or item.get("extra_info", {}).get("question", "")
            or ""
        )

        user_prompt = raw_prompt.format(reference_question=reference_question)
        prompt = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        test_item_clean = make_json_serializable(item)
        
        dataframe.append(
            {
                "idx": idx,
                "data_source": item.get("data_source", "unknown"),
                "reference_question": reference_question,
                "prompt": prompt,
                "ability": "math",
                "test_item": test_item_clean,
            }
        )

    random.shuffle(dataframe)
    return dataframe

def get_prompts(num_querys, get_prompts_func, icl_files=None):
    if icl_files is not None and get_prompts_func == "ttrl_icl":
        return get_prompts_weakness_icl(num_querys, icl_files)
    else:
        if get_prompts_func == "R_Zero":
            return get_prompts_R_zero(num_querys)
        else:
            raise ValueError(f"Invalid get_prompts_func: {get_prompts_func}")


def get_prompts_R_zero(num_querys):
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            )
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            )
        }
    ]
    dataframe=[
        {
            'idx': idx,
            'data_source': 'Challenger',
            'topic':'Challenger',
            'prompt': chat,
            'ability': 'math'
           
        }
        for idx in range(num_querys)
    ]
    return dataframe


class ChallengerTopicDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        
        self.tokenizer = tokenizer
        self.dynamic_topics = config.get('dynamic_topics',False)
        print(f'{self.dynamic_topics=} ,{self.topic_path=}')
        self.num_querys = config.get("num_querys", 1000)
        self.ttrl_icl_files = config.get("ttrl_icl_files", None)
        #assert self.ttrl_icl_files is not None, "ttrl_icl_files must be provided"
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        print(f'{self.max_prompt_length=}')
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self._build_dataset()
     
    def __len__(self):
        return len(self.dataframe)

    def _build_dataset(self):
        self.dataframe = get_prompts(num_querys=self.num_querys, get_prompts_func=self.get_prompts_func, icl_files=self.ttrl_icl_files)
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages: dict = row_dict.pop('prompt')
        #print(f'{messages=}')
        model_inputs = {}
        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                "models like GLM can copy chat_template.jinja from instruct models"
            )
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        if random.randint(0,1000) == 0:
            print(f'{raw_prompt=}')
        #print(f'{raw_prompt=}')
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        
        position_ids = compute_position_id_with_mask(attention_mask)
        #row_dict = {}
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
            row_dict['raw_inputs'] = raw_prompt

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()



    
