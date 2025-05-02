from api import *
import gradio as gr

sampled_dataset, df, client = setup(100,"3116fd3668302432d738187aa87a1ef3d5f89559b73d736b02f31da487baebd0")

def process_question(question, choice1, choice2, choice3, choice4):
    # Replace with your logic to process the question and choices
    prediction,answers = sc_most_common(client,question,
                                {
                                'label': ['A','B','C','D'],
                                'text': [choice1,choice2,choice3,choice4]
                                },n_samples=6)
    # output1 = [prediction].append([answer for answer in answers]).join("\n")
    # output1 = '\n'.join([prediction,answers[0]])
    output1 = prediction
    output2 = get_response(client,f"Combine these answers into a single answer without modifying the content {'\n'.join(answers)}")
    output3 = "Processed result here"
    return output1, output2, output3

with gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(label="Question", value="What is the capital of France?"),
        gr.Textbox(label="Choice 1", value="Paris"),
        gr.Textbox(label="Choice 2", value="London"),
        gr.Textbox(label="Choice 3", value="Berlin"),
        gr.Textbox(label="Choice 4", value="Madrid"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Explaination"),
        gr.Textbox(label="Output 3"),
    ],
    title="Question Processor",
    description="Enter a question and four choices to process.",
) as demo:
    demo.launch()