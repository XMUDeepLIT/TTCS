import torch
import re
import json
import numpy as np
from typing import List, Tuple, Dict, Any
import math
from math_verify import verify
import random
from mathruler.grader import grade_answer
from mathruler.math_normalize import normalize_answer, _strip_string
import stopit
import difflib



class MathSimilarityFilter:

    def __init__(
        self, 
        text_threshold: float = 0.6, 
        jaccard_threshold: float = 0.7,  
        skeleton_threshold: float = 0.90  
    ):
        self.text_threshold = text_threshold
        self.jaccard_threshold = jaccard_threshold
        self.skeleton_threshold = skeleton_threshold
        
        self.stopwords = {
            "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
            "that", "this", "is", "are", "be", "as", "at", "by", "from", "it",
            "let", "given", "consider", "determine", "find", "compute", "evaluate",
            "show", "prove", "calculate", "solve", "express", "write", "state",
            "number", "numbers", "integer", "integers", "value", "values",
            "solution", "solutions", "answer", "result", "following", "problem",
        }
        
        self._re_math_inline = re.compile(r'(?<!\\)\$(.*?)(?<!\\)\$', re.DOTALL)
        self._re_math_paren = re.compile(r'\\\((.*?)\\\)', re.DOTALL)
        self._re_math_bracket = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
        self._re_whitespace = re.compile(r'\s+')
        self._re_digits = re.compile(r'\d+')
        self._re_latex_cmd = re.compile(r'\\[a-zA-Z]+')
        self._re_single_var = re.compile(r'(?<!\\)\b[a-zA-Z]\b')
        self._re_greek = re.compile(r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)')
        

        self._unicode_math_map = {
            '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt',
            '∞': '\\infty', '∂': '\\partial', '∇': '\\nabla',
            '≤': '\\leq', '≥': '\\geq', '≠': '\\neq', '≈': '\\approx',
            '×': '\\times', '÷': '\\div', '±': '\\pm', '∓': '\\mp',
            '∈': '\\in', '∉': '\\notin', '⊂': '\\subset', '⊃': '\\supset',
            '∪': '\\cup', '∩': '\\cap', '∅': '\\emptyset',
            'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta',
            'ε': '\\epsilon', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
            'π': '\\pi', 'σ': '\\sigma', 'φ': '\\phi', 'ω': '\\omega',
            '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
            '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
            '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
            '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        }


    
    def _normalize_unicode_math(self, text: str) -> str:
        for unicode_char, latex_cmd in self._unicode_math_map.items():
            text = text.replace(unicode_char, latex_cmd)
        return text
    
    def _normalize_latex(self, text: str) -> str:
        if not text:
            return ""
        # 先转换 Unicode 数学符号
        text = self._normalize_unicode_math(text)
        try:
            return _strip_string(text) or text
        except Exception:
            return text

    def _clean_text(self, text: str) -> str:
        text = self._normalize_unicode_math(text)
        text = text.lower()
        text = self._re_whitespace.sub(' ', text)
        text = re.sub(r'[^\w\s\+\-\*\/\=\<\>\^\|\(\)\{\}\[\]\\\_\,\.]', '', text)
        return text.strip()

    def _get_tokens(self, text: str) -> set:
        tokens = re.findall(
            r'[a-zA-Z]+|\d+|\\[a-zA-Z]+|[\+\-\=\<\>\^\|\/\(\)\{\}\[\]_]', 
            text.lower()
        )
        return {t for t in tokens if not (t.isalpha() and t in self.stopwords)}

    def _adaptive_text_threshold(self, clean_ref: str, clean_syn: str) -> float:
        L = min(len(clean_ref), len(clean_syn))
        if L < 100:
            return max(self.text_threshold, 0.88)
        elif L < 200:
            return max(self.text_threshold, 0.78)
        elif L < 400:
            return max(self.text_threshold, 0.68)
        return self.text_threshold


    def _extract_math_segments(self, text: str) -> str:
        segs = []
        segs += self._re_math_inline.findall(text)
        segs += self._re_math_paren.findall(text)
        segs += self._re_math_bracket.findall(text)
        return " ".join(segs) if segs else ""

    def _skeletonize_math(self, math_text: str) -> str:
        if not math_text:
            return ""
        
        s = self._normalize_latex(math_text)
        s = self._re_greek.sub('<V>', s)
        s = self._re_digits.sub('<N>', s)
        s = self._re_single_var.sub('<V>', s)
        s = s.replace(' ', '')
        
        return s

    def check_text_similarity(self, ref_q: str, syn_q: str) -> tuple:
        clean_ref = self._clean_text(ref_q)
        clean_syn = self._clean_text(syn_q)
        
        score = difflib.SequenceMatcher(None, clean_ref, clean_syn).ratio()
        threshold = self._adaptive_text_threshold(clean_ref, clean_syn)
        
        return score > threshold, score

    def check_jaccard_similarity(self, ref_q: str, syn_q: str) -> tuple:
        tokens_ref = self._get_tokens(self._clean_text(ref_q))
        tokens_syn = self._get_tokens(self._clean_text(syn_q))
        
        if not tokens_ref or not tokens_syn:
            return False, 0.0
        
        intersection = tokens_ref & tokens_syn
        union = tokens_ref | tokens_syn
        
        score = len(intersection) / len(union)
        return score > self.jaccard_threshold, score

    def check_equation_structure(self, ref_q: str, syn_q: str) -> tuple:
        ref_math = self._extract_math_segments(ref_q)
        syn_math = self._extract_math_segments(syn_q)
        
        if not ref_math or not syn_math:
            ref_math = ref_q
            syn_math = syn_q
        
        skel_ref = self._skeletonize_math(ref_math)
        skel_syn = self._skeletonize_math(syn_math)
        
        if not skel_ref or not skel_syn:
            return False, 0.0
        
        score = difflib.SequenceMatcher(None, skel_ref, skel_syn).ratio()
        return score > self.skeleton_threshold, score

    def is_bad_case(self, ref_q: str, syn_q: str) -> tuple:
        
        if not ref_q or not syn_q or len(ref_q.strip()) < 5 or len(syn_q.strip()) < 5:
            return False, "Pass (input too short)", 0.0
        
        
        bad_text, score_text = self.check_text_similarity(ref_q, syn_q)
        if bad_text:
            return True, f"Text Similarity too high: {score_text:.2f}", score_text

        bad_skel, score_skel = self.check_equation_structure(ref_q, syn_q)
        bad_jacc, score_jacc = self.check_jaccard_similarity(ref_q, syn_q)

        if bad_skel:
            if score_text > 0.45 or score_jacc > 0.40:
                return True, f"Equation Skeleton too similar: {score_skel:.2f} (text={score_text:.2f}, jacc={score_jacc:.2f})", score_skel
        if bad_jacc:
            return False, f"[Review] Vocabulary Overlap high: {score_jacc:.2f}", score_jacc

        return False, "Pass", 0.0
    
    def is_bad_case_strict(self, ref_q: str, syn_q: str) -> tuple:
        if not ref_q or not syn_q or len(ref_q.strip()) < 5 or len(syn_q.strip()) < 5:
            return False, "Pass (input too short)", 0.0
            
        bad_text, score_text = self.check_text_similarity(ref_q, syn_q)
        if bad_text:
            return True, f"Text Similarity too high: {score_text:.2f}", score_text

        bad_skel, score_skel = self.check_equation_structure(ref_q, syn_q)
        if bad_skel:
            return True, f"Equation Skeleton too similar: {score_skel:.2f}", score_skel

        bad_jacc, score_jacc = self.check_jaccard_similarity(ref_q, syn_q)
        if bad_jacc:
            return True, f"Vocabulary Overlap too high: {score_jacc:.2f}", score_jacc

        return False, "Pass", 0.0

@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

def process_single_R_Zero(idx, question, answer, response):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    
    results = [str(extract_boxed_content(out.text)) for out in response.outputs]

    answer_counts = {}
    for res in list(results):
        if not res: continue # Skip empty results
        matched = False
        
        for exist_ans in list(set(answer_counts.keys())):
            # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
            if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                answer_counts[exist_ans] += 1
                matched = True
                break # Match found, break from the inner loop over exist_ans
            
            # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
            try:
                is_match = False
                # First direction: res vs exist_ans
                match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                if match_result_1 == 'TIMED_OUT':
                    print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                elif match_result_1:
                    is_match = True

                # Second direction (only if first failed): exist_ans vs res
                if not is_match:
                    match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                    if match_result_2 == 'TIMED_OUT':
                            # Log timeout for the second direction as well
                        print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                    elif match_result_2:
                        is_match = True
                
                if is_match:
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop

            except Exception as e:
                # Catch any other potential errors from the grader function itself.
                print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                continue # Continue to the next comparison in the inner loop
        
        if not matched:
            answer_counts[res] = 1

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    if 'none' == majority_ans.lower():
        score = 0.0
    else:
        score = max_count / len(results) if results else 0.0
        #print(f'[process_single] Question {idx}: No valid labels found')
    uncertainty_reward = 1 - 2 * abs(score - 0.5)
    reward_info={
        'majority_accuracy': score,
        'all_labels': results,
        'answer_counts':answer_counts,
        'majority_accuracy': score,
        'majority_ans': majority_ans,
        'reward':uncertainty_reward,
    }
    if random.randint(0, 64) == 0:
        print(f'Question: {question} \n Answer: {answer} \n results:{results} \n answer_counts: {answer_counts} \n Answer: {majority_ans}\n Score: {score} \n Reward: {uncertainty_reward}')
    return {
        'idx':idx,
        'question': question,
        'answer': answer,
        'reward_info': reward_info,
        'reward':uncertainty_reward
    }

def calculate_bell_reward(score, sharpness=1.2):
    base = 4.0 * score * (1.0 - score)
    base = max(0.0, base)
    
    return base ** sharpness

 
_similarity_filter = None

def get_similarity_filter():
    global _similarity_filter
    if _similarity_filter is None:
        _similarity_filter = MathSimilarityFilter()
    return _similarity_filter


def process_single_TTCS(idx, question, answer, response, reference_question):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    
    results = [str(extract_boxed_content(out.text)) for out in response.outputs]

    answer_counts = {}
    for res in list(results):
        if not res: continue # Skip empty results
        matched = False
        
        for exist_ans in list(set(answer_counts.keys())):
            # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
            if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                answer_counts[exist_ans] += 1
                matched = True
                break # Match found, break from the inner loop over exist_ans
            
            # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
            try:
                is_match = False
                # First direction: res vs exist_ans
                match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                if match_result_1 == 'TIMED_OUT':
                    print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                elif match_result_1:
                    is_match = True

                # Second direction (only if first failed): exist_ans vs res
                if not is_match:
                    match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                    if match_result_2 == 'TIMED_OUT':
                            # Log timeout for the second direction as well
                        print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                    elif match_result_2:
                        is_match = True
                
                if is_match:
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop

            except Exception as e:
                # Catch any other potential errors from the grader function itself.
                print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                continue # Continue to the next comparison in the inner loop
        
        if not matched:
            answer_counts[res] = 1

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    if 'none' == majority_ans.lower():
        score = 0.0
    else:
        score = max_count / len(results) if results else 0.0
    filter = get_similarity_filter()
    is_bad, message, filter_score = filter.is_bad_case(reference_question, question)
        #print(f'[process_single] Question {idx}: No valid labels found')
    
    uncertainty_reward = calculate_bell_reward(score)
    if is_bad:
        novelty_factor = max(0.0, 1.0 - filter_score)
        reward = uncertainty_reward * novelty_factor
    else:
        reward = uncertainty_reward
    reward_info={
        'majority_accuracy': score,
        'all_labels': results,
        'answer_counts':answer_counts,
        'majority_accuracy': score,
        'majority_ans': majority_ans,
        'filter_info':{
            'is_bad': is_bad,
            'message': message,
            'filter_score': filter_score,
        },
        'reward':reward,
    }

    return {
        'idx':idx,
        'question': question,
        'answer': answer,
        'reward_info': reward_info,
        'reward':reward
    }
