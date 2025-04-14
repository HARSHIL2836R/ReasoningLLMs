from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  # or use "t5-small" with adjustments
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure model is in evaluation mode and on correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Input prompt
prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Beam search parameters
beam_width = 3
max_length = 30
num_return_sequences = beam_width

# Generate using beam search
outputs = model.generate(
    input_ids,
    max_length=max_length,
    num_beams=beam_width,
    num_return_sequences=num_return_sequences,
    early_stopping=True,
    no_repeat_ngram_size=2  # optional: prevent repetitive phrases
)

# Decode and print each result
print(f"\nBeam Search Results for prompt: \"{prompt}\"\n")
for i, beam_output in enumerate(outputs):
    decoded = tokenizer.decode(beam_output, skip_special_tokens=True)
    print(f"[Beam {i+1}]: {decoded}\n")
