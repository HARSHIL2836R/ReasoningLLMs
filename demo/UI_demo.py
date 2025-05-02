from api import *
import gradio as gr

sampled_dataset, df, client = setup(100,"3116fd3668302432d738187aa87a1ef3d5f89559b73d736b02f31da487baebd0")

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
    greedy_answer = get_response(client, prompt, temperature=0.7)

    prediction, answers = sc_most_common(
        client,
        question,
        {
            'label': ['A', 'B', 'C', 'D'],
            'text': [choice1, choice2, choice3, choice4]
        },
        n_samples=n_samples,
    )
    output1 = prediction
    combined_answers = '\n'.join(answers)
    output2 = get_response(client, f"Combine these explaination into a single explaination without modifying the content:\n{combined_answers}")
    output3 = greedy_answer
    return output1, output2, output3

with gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(label="Question", value="When people are overly drunk how do the seem to others?"),
        gr.Textbox(label="Choice 1", value="disoriented"),
        gr.Textbox(label="Choice 2", value="appear ridiculous"),
        gr.Textbox(label="Choice 3", value="walk off"),
        gr.Textbox(label="Choice 4", value="throw up"),
        gr.Slider(minimum=1, maximum=20, step=2, value=6, label="Number of Samples (optional)"),
        gr.Radio(choices=["Greedy", "Clustering"], label="Decoding Strategy", value="Clustering")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Greedy"),
    ],
    title="Question Processor",
    description="Enter a question and four choices to process.",
) as demo:
    demo.launch()