from api import *
import gradio as gr

sampled_dataset, df, client = setup(100)

def process_question(question, choice1, choice2, choice3, choice4, n_samples=6, mode="Clustering"):
    # Ensure question and choices are provided
    if not question.strip() or not choice1.strip() or not choice2.strip() or not choice3.strip() or not choice4.strip():
        return "Missing question or choices", "", ""

    # Apply defaults if sliders are untouched (None)
    n_samples = n_samples if n_samples is not None else 6

    prompt = format_prompt(question, 
        {
            'label': ['A', 'B', 'C', 'D'],
            'text': [choice1, choice2, choice3, choice4]
        })
    greedy_answer = get_response(client, prompt)

    if mode == "Clustering" :
        prediction, answers, metrics = sc_with_clustering(
            client,
            question,
            {
                'label': ['A', 'B', 'C', 'D'],
                'text': [choice1, choice2, choice3, choice4]
            },
            n_samples=n_samples,
        )
    else :
        prediction, answers, metrics = sc_with_temperature_sampling(
            client, 
            question,
            {
                'label': ['A', 'B', 'C', 'D'],
                'text': [choice1, choice2, choice3, choice4]
            }
        )
    output1 = prediction
    combined_answers = '\n'.join(answers)
    output2 = get_response(client, f"Combine these answers into a single answer without modifying the content:\n{combined_answers}")
    output3 = greedy_answer
    metrics_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]) # Convert metrics to string or key-value pairs
    output4 = metrics_str
    return output1, output2, output3, output4

with gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(label="Question", value="What is the capital of France?"),
        gr.Textbox(label="Choice 1", value="Paris"),
        gr.Textbox(label="Choice 2", value="London"),
        gr.Textbox(label="Choice 3", value="Berlin"),
        gr.Textbox(label="Choice 4", value="Madrid"),
        gr.Slider(minimum=1, maximum=20, step=1, value=6, label="Number of Samples (optional)"),
        gr.Radio(choices=["Temperature", "Clustering"], label="Decoding Strategy", value="Clustering")
    ],
    outputs=[
        gr.Textbox(label="Self consistent Prediction"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Greedy Prediction"),
        gr.Textbox(label="Metrics", lines=3)
    ],
    title="Question Processor",
    description="Enter a question and four choices to process.",
    flagging_mode="never"
) as demo:
    demo.launch()