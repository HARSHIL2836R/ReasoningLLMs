import os
import random
from collections import Counter
from datasets import load_dataset
from together import Together
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

class Metrics :
    def count_unique_paths(paths):
        return len(set(paths))

    def shannon_entropy(paths):
        total = len(paths)
        if total == 0:
            return 0.0

        freq = Counter(paths)
        entropy = 0.0

        for count in freq.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy

    def avg_cosine_distance(paths):
        if len(paths) < 2:
            return 0.0  # No pairs to compare

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(paths)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Extract upper triangle of similarity matrix (excluding diagonal)
        n = len(paths)
        sim_scores = [
            similarity_matrix[i][j]
            for i in range(n) for j in range(i + 1, n)
        ]
        return np.mean(sim_scores)

def setup(total_questions):
    "Returns Dataset, DataFrame to work on, Together AI Client"
    # # Initialize Together client
    # os.environ["TOGETHER_API_KEY"] = API_KEY
    load_dotenv()
    client = Together()

    # Set Together AI API Key
    # os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")

    # Load a random subset of 3 questions from the validation set
    dataset = load_dataset("commonsense_qa", split="validation")
    sampled_dataset = random.sample(list(dataset), total_questions)

    csv_file_path = "llama_8b_instruct.csv"
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame(columns=['Question', 'Answer Match Greedy', 'Answer Match Self-consistency', 'Correct Answer', 'Greedy Answer', 'Self Answer','Approaches Agree'])

    return sampled_dataset, df, client

def format_prompt(question, choices):
    choice_str = "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
    return f"Question: {question}\nChoices:\n{choice_str}\nAnswer with the correct letter"

def get_response(client,prompt, temperature=0.0):
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        # model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def sc_most_common(client,question, choices, n_samples=3, temperature = 0.7):
    prompt = format_prompt(question, choices)
    answers = [get_response(client,prompt, temperature=temperature) for _ in range(n_samples)]

    return Counter(answers).most_common(1)[0][0],answers

def extract_option(answer):
    """Extract the option (e.g., 'A', 'B', etc.) from the answer text."""
    for option in ['A', 'B', 'C', 'D', 'E']:
        if answer.startswith(option) or f"{option}." in answer or f"{option} " in answer:
            return option
    return None  # Return None if no valid option is found

def run_tests(sampled_dataset,df,self_consistent_answer):
    for idx, item in enumerate(sampled_dataset, 1):
        if item["question"] in df['Question'].values:
            print(f"Skipping question {idx} as it is already in the dataframe.")
            continue

        print(f"\n--- Question {idx} ---")
        df.loc[idx, 'Question'] = item["question"]
        greedy_answer = get_response(format_prompt(item["question"], item["choices"]), temperature=0.0)
        df.loc[idx, 'Greedy Answer'] = greedy_answer

        n = 6
        prediction, answers = self_consistent_answer(item["question"], item["choices"], n_samples=n)
        df.loc[idx, 'Self Answer'] = prediction
        correct_answer = item["answerKey"] 
        index = item['choices']['label'].index(item['answerKey'])
        correct_answer = correct_answer + ". " + item['choices']['text'][index]    

        df.loc[idx, 'Correct Answer'] = correct_answer

        # Extract options from greedy_answer and prediction
        greedy_option = extract_option(greedy_answer)
        self_option = extract_option(prediction)

        if correct_answer[0] == greedy_option:
            df.loc[idx, 'Answer Match Greedy'] = True
        else:
            df.loc[idx, 'Answer Match Greedy'] = False
        
        if correct_answer[0] == self_option:
            df.loc[idx, 'Answer Match Self-consistency'] = True
        else :
            df.loc[idx, 'Answer Match Self-consistency'] = False
        
        if greedy_option != self_option:
            df.loc[idx, "Approaches Agree"] = False
        else:
            df.loc[idx, 'Approaches Agree'] = True

        df.to_csv("llama_8b_instruct.csv")

    print("--- Evaluation ---")
    greedy_accuracy = df['Answer Match Greedy'].mean() * 100  # Percentage of correct greedy answers
    self_consistency_accuracy = df['Answer Match Self-consistency'].mean() * 100  # Percentage of correct self-consistent answers

    # Calculate differences
    total_questions = len(df)
    disagreements = df[df['Approaches Agree'] == False].shape[0]  # Count of disagreements
    better_self = df[(df['Approaches Agree'] == False) & (df['Answer Match Self-consistency'] == True)].shape[0]
    worse_self = df[(df['Approaches Agree'] == False) & (df['Answer Match Greedy'] == True)].shape[0]

    # Print results
    print(f"Total Questions: {total_questions}")
    print(f"Greedy Accuracy: {greedy_accuracy:.2f}%")
    print(f"Self-Consistency Accuracy: {self_consistency_accuracy:.2f}%")
    print(f"Disagreements: {disagreements}")
    print(f"Better Self-Consistency: {better_self}")
    print(f"Worse Self-Consistency: {worse_self}")

def cluster_and_majority_vote(answers, threshold=0.7):
    """
    Cluster answers based on cosine similarity and apply majority vote within clusters.
    
    Args:
        answers (list): List of answers generated by the model.
        threshold (float): Cosine similarity threshold to form clusters.
        
    Returns:
        str: The most common answer from the largest cluster.
    """
    # Convert answers to TF-IDF vectors
    vectorizer = TfidfVectorizer().fit_transform(answers)
    similarity_matrix = cosine_similarity(vectorizer)

    # Clustering based on similarity threshold
    clusters = []
    visited = set()
    for i in range(len(answers)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(len(answers)):
            if j not in visited and similarity_matrix[i][j] >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    # Find the largest cluster and apply majority vote
    largest_cluster = max(clusters, key=len)
    clustered_answers = [answers[i] for i in largest_cluster]
    most_common_answer = Counter(clustered_answers).most_common(1)[0][0]

    return most_common_answer

# Update the self-consistency function to use clustering
def sc_with_clustering(client, question, choices, n_samples=3, threshold=0.7):
    """
    Self-consistency with clustering based on cosine similarity.
    t
    Args:
        question (str): The question text.
        choices (dict): The answer choices.
        n_samples (int): Number of samples to generate.
        threshold (float): Cosine similarity threshold for clustering.
        
    Returns:
        str: The final answer after clustering and majority vote.
    """
    prompt = format_prompt(question, choices)
    answers = [get_response(client, prompt, temperature=0.7) for _ in range(n_samples)]
    final_answer = cluster_and_majority_vote(answers, threshold=threshold)
    metrics = {
        "unique_paths" : Metrics.count_unique_paths(answers),
        "path_similarity" : Metrics.avg_cosine_distance(answers),
        "diversity_score": Metrics.shannon_entropy(answers)
    }
    return final_answer, answers, metrics

def sc_with_temperature_sampling(client, question, choices, temperatures=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3]):
    """
    Self-consistency approach using multiple temperature values for sampling.

    Args:
        question (str): The question to ask.
        choices (dict): Dictionary with 'label' and 'text' for the answer choices.
        temperatures (list): List of temperature values to use for sampling.

    Returns:
        str: Most common answer across temperature samples.
        list: List of all generated answers.
    """
    prompt = format_prompt(question, choices)
    answers = [get_response(client, prompt, temperature=temp) for temp in temperatures]
    most_common = Counter(answers).most_common(1)[0][0]
    metrics = {
        "unique_paths" : Metrics.count_unique_paths(answers),
        "path_similarity" : Metrics.avg_cosine_distance(answers),
        "diversity_score": Metrics.shannon_entropy(answers)
    }

    return most_common, answers, metrics
