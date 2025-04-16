import os
import random
from collections import Counter
from datasets import load_dataset
from together import Together

# Initialize Together client
client = Together()

# Set Together AI API Key
os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")

# Load a random subset of 3 questions from the validation set
dataset = load_dataset("commonsense_qa", split="validation")
sampled_dataset = random.sample(list(dataset), 100)

def format_prompt(question, choices):
    choice_str = "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
    return f"Question: {question}\nChoices:\n{choice_str}\nAnswer with the correct letter. Give explaination in the next line."

def get_response(prompt, temperature=0.0):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def self_consistent_answer(question, choices, n_samples=3):
    prompt = format_prompt(question, choices)
    answers = [get_response(prompt, temperature=0.7) for _ in range(n_samples)]
    # print("Prompt:")
    # print(prompt)
    # print("Answers:")
    # for answer in answers:
        # print("------")
        # print(answer)
        # print("------")

    return Counter(answers).most_common(1)[0][0],answers

# Evaluate self-consistency
for idx, item in enumerate(sampled_dataset, 1):
    print(f"\n--- Question {idx} ---")
    # print("Greedy Decode:")
    greedy_answer = get_response(format_prompt(item["question"], item["choices"]), temperature=0.0)
    # print(greedy_answer)

    predictions = []

    n = 4
    # print(f"\nSelf-Consistency with {n} sample(s):")
    prediction, answers = self_consistent_answer(item["question"], item["choices"], n_samples=n)
    # print(f"Most common answer: {prediction}")

    if greedy_answer.split('\n')[0][0] != prediction.split('\n')[0][0]:
        print("Answers Mismatch")
        print("GReedy Decode:")
        print(greedy_answer)
        print("Self consistency with 3 paths")
        print("Three Answers:")
        print(answers[0])
        print(answers[1])
        print(answers[2])
        print("Prediction:")
        print(prediction)

    else:
        print("Answers match")