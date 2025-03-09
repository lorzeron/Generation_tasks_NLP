#http://127.0.0.1:8001/

import gradio as gr
from inference import get_reply  # Функция для получения ответа

def chat_interface(user_input):
    reply = get_reply(user_input)
    return reply

with gr.Blocks() as demo:
    gr.Markdown("# Чат-бот на основе NLP")
    with gr.Row():
        user_input = gr.Textbox(label="Ваш вопрос")
        submit = gr.Button("Отправить")
    bot_output = gr.Textbox(label="Ответ бота")
    
    submit.click(chat_interface, inputs=user_input, outputs=bot_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001, share=False)
