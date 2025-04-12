from datasets import load_dataset

dataset = load_dataset("commonsense_qa", split="validation[:10]")

print(dataset.info.description)