import os
import random
from collections import Counter
from datasets import load_dataset
from together import Together
# from dotenv import load_dotenv

# load_dotenv()

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
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
self_consistency_accuracy = 0
greedy_accuracy = 0
difference = 0
better = 0
worse = 0

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
    correct_answer = item["answerKey"]
    print("Correct answer:")
    print(correct_answer)

    if correct_answer == greedy_answer[0]:
        print("Greedy Correct")
        greedy_accuracy += 1
    else :
        print("Greedy Incorrect")
        print("Greedy Decode:")
        print(greedy_answer[0])
    
    if correct_answer == prediction[0]:
        print("Self-consistency Correct")
        self_consistency_accuracy += 1
    else :
        print("Self-consistency Incorrect")
        print("Self-consistency predict:")
        print(prediction[0])
    
    if greedy_answer.split('\n')[0][0] != prediction.split('\n')[0][0]:
        print("Answers Mismatch")
        print("Greedy Decode:")
        print(greedy_answer)
        print("Self consistency with 3 paths")
        # print("Three Answers:")
        # print(answers[0])
        # print(answers[1])
        # print(answers[2])
        print("Prediction:")
        print(prediction)
        difference += 1
        if prediction[0] == correct_answer:
            better += 1
        if greedy_answer[0] == correct_answer:
            worse += 1
    else:
        print("Answers match")

print("--- Evaluation ---")
print("self_consistency_accuracy ", self_consistency_accuracy, "%")
print("greedy_accuracy ", greedy_accuracy, "%")
print("difference ", difference, "%")
print("better ", better, "%")
print("worse ", worse, '%')