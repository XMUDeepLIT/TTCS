#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from collections import defaultdict,Counter
from mathruler.grader import grade_answer,extract_boxed_content
from src.utils import process_single_R_Zero, process_single_TTCS
  # 1. Import the thread-safe 'stopit' library
import random
# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.95,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ------------------------- Global Variables ------------------------ #
# Initialize these in main to avoid multiprocessing issues
tokenizer = None
model = None
sample_params = None

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# Initialize threading events first
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def initialize_model():
    """Initialize vLLM model and related components"""
    global tokenizer, model, sample_params
    print('[init] Loading model...')
    
    # Pause GPU idle worker during model initialization to avoid CUDA graph conflicts
    print('[init] Pausing GPU idle worker for model initialization...')
    pause_event.set()
    torch.cuda.synchronize()
    time.sleep(1)  # Give the idle worker time to pause
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = vllm.LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        
        sample_params = vllm.SamplingParams(
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            top_k=40,
            stop_token_ids=[tokenizer.eos_token_id],
            n=10, # Generate 10 candidate answers for each question
        )
        print('[init] Model loaded successfully!')
    finally:
        # Resume GPU idle worker after model initialization
        print('[init] Resuming GPU idle worker...')
        pause_event.clear()

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

# Initialize the idle thread but don't start it yet
idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)

# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.


# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()
    
    name = request.args.get('name', 'None')
    question_reward = request.args.get('question_reward', 'None')

    print(f'[server] Received request for task file: {name}')
    print(f'[server] Question reward: {question_reward}')

    # ---------- Load Data ----------
    try:
        if not os.path.exists(name):
            print(f'[server] ERROR: Task file {name} does not exist')
            return jsonify({'error': f'Task file {name} does not exist'}), 404
        
        import time
        max_wait = 5  
        wait_time = 0
        while wait_time < max_wait:
            try:
                with open(name, 'r') as f:
                    data = json.load(f)
                break
            except (json.JSONDecodeError, IOError) as e:
                if wait_time < max_wait - 1:
                    print(f'[server] File {name} not ready, waiting... ({wait_time+1}s)')
                    time.sleep(1)
                    wait_time += 1
                else:
                    raise e
        
        if not data:
            print(f'[server] ERROR: Task file {name} is empty')
            return jsonify({'error': f'Task file {name} is empty'}), 400
            
    except Exception as e:
        print(f'[server] ERROR: Failed to load task file {name}: {e}')
        return jsonify({'error': f'Failed to load task file: {str(e)}'}), 500
    
    questions = [item.get('question', '') for item in data]
    reference_qs = [item.get('reference_question', '') for item in data]
    answers = [item.get('answer', '') for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    instructions = 'Please reason step by step, and put your final answer within \\boxed{}.'
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system',   'content': instructions},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # (vLLM generation logic remains unchanged)
    
    if valid_chats:
        print(f'[server] Generating responses for {len(valid_chats)} valid chats')
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True, enable_thinking=False)
                for chat in valid_chats
            ]
            if random.randint(0,64)==0:
                print(f'{prompts[0]=}')
        else:
            prompts = [
                'system: You are a helpful assistant.' + '\n' + 'user: ' + chat[0]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=False)
        print(f'[server] Generated {len(responses)} responses')
    else:
        responses = []
        print('[server] No valid chats, responses list is empty')
    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    
    def process_single(idx, question, answer, response, reference_question):
        if question_reward == 'R_Zero':
            return process_single_R_Zero(idx, question, answer,response)
        elif question_reward == 'TTCS':
            return process_single_TTCS(idx, question, answer, response, reference_question)
        else:
            raise NotImplementedError(f'Question reward {question_reward} not implemented')
    
    results_all = []
    response_idx = 0
    print(f'[server] Processing {len(valid_chats)} questions with {len(responses)} responses')
    
    for idx, (q, a, ref_q) in enumerate(zip(questions, answers, reference_qs)):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(idx, q, a, response, ref_q)
                results_all.append(item)
            else:
                # 'idx':idx, 'question': q, 'answer': a, 'initial_answer':a, 'uncertainty_reward': -1, 'labels_info': {}, 'labels': []
                results_all.append({
                    'idx':idx,
                    'question': q,
                    'answer': a,
                    'reward_info': {},
                    'reward':0.0,  
                    'error': f'no question:{q=}, skipping...'
                })
        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred in files:{name} while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'idx':idx,
                'question': q,
                'answer': a,
                'reward_info': {},
                'reward':0.0,  
                'error':    f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')


    out_path = name.replace('.json', '_results.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(results_all, f, indent=4)
        print(f'[server] Results saved to {out_path}')
        
        
        if os.path.exists(name):
            os.remove(name)
            print(f'[server] Cleaned up task file: {name}')
    except Exception as e:
        print(f'[server] ERROR: Failed to save results to {out_path}: {e}')
        return jsonify({'error': f'Failed to save results: {str(e)}'}), 500

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    torch.cuda.synchronize()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'status': 'success', 'results_count': len(results_all)})

# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        # Start the GPU idle worker thread
        idle_thread.start()
        print('[main] GPU idle worker thread started.')
        
        # Initialize model before starting the server
        initialize_model()
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')