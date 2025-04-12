import os
from collections import Counter
from datasets import load_dataset
from together import Together

client = Together()

# Set Together AI API Key
os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")

# Load CommonsenseQA subset
dataset = load_dataset("commonsense_qa", split="validation[:10]")

def format_prompt(question, choices):
    choice_str = "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
    return f"Question: {question}\nChoices:\n{choice_str}\nAnswer with the correct letter."

def get_response(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    # return response["choices"][0]["message"]["content"].strip()
    return response.choices[0].message.content

def self_consistent_answer(question, choices, n_samples=5):
    prompt = format_prompt(question, choices)
    answers = [get_response(prompt) for _ in range(n_samples)]
    print("Prompt:")
    print(prompt)
    print("Answers:")
    for answer in answers:
        print("------")
        print(answer)
        print("------")
    
    return Counter(answers).most_common(1)[0][0]

# Evaluate self-consistency
correct = 0
for item in dataset:
    prediction = self_consistent_answer(item["question"], item["choices"])
    gold = item["answerKey"]
    if gold in prediction.upper():
        correct += 1

print(f"Self-consistency accuracy over 10 examples: {correct}/10")
