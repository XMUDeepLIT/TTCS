import datasets
import json
from tqdm import tqdm
import anthropic
import re
import random

client = anthropic.Anthropic(
    api_key="",
)


def run_one_question(question: str):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        system="You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.",
        messages=[
            {"role": "user", "content": question}
        ],
        temperature = 0.1,
        top_p = 1,
    )
    return message.content[0].text


def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def get_prediction(output):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    else:
        print("extraction failed, do a random guess")
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])


if __name__ == "__main__":
    dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro')

    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']

    prompts = {c: '' for c in categories}
    for d in dataset['validation']:
        prompts[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'

    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0
    answers = []

    output_file = "output.json"
    file = open(output_file, "w")

    print('----------------- Start Answering -------------------')
    i = 0
    for entry in tqdm(dataset['test']):
        prefix = prompts[entry['category']]
        query = prefix + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n'
        answer = run_one_question(query)
        # print(answer)
        entry['solution'] = answer
        answers.append(entry)
        prediction = get_prediction(answer)
        if entry["answer"] == prediction:
            success += 1
            per_category_accuracy[entry['category']][0] += 1
        else:
            fail += 1
            per_category_accuracy[entry['category']][1] += 1

        json_string = json.dumps(entry)
        file.write(json_string + '\n')

        print(success / (success + fail))

    for k, v in per_category_accuracy.items():
        print('accuracy: ', k, v[0] / (v[0] + v[1]))