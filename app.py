import discovery_engine
import gradio as gr


def search(question):
    result = discovery_engine.search(question)
    return result["response"]

def main():
    with gr.Blocks() as demo:
        question = gr.TextArea(label="question", lines=8, value="welche indikationen gibt es für Zahnfleischbluten?")
        submit = gr.Button("submit", )
        answer = gr.TextArea(label="answer", lines=8)
        submit.click(fn=search, inputs=[question], outputs=answer)
    demo.launch()


if __name__ == "__main__":
    main()
