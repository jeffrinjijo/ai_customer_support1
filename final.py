import os
import gradio as gr
import requests

# ✅ GROQ CLIENT CLASS
class GroqClient:
    def _init_(self, api_key: str, model: str = "llama3-70b-8192"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API Error {response.status_code}: {response.text}")

# ✅ Load your Groq API Key
GROQ_API_KEY = "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"
groq = GroqClient(api_key=GROQ_API_KEY)

# ✅ SYSTEM PROMPT (No stars, no markdown)
SYSTEM_PROMPT = """
You are an AI-powered multi-agent customer support assistant.
For every customer query, do the following:
1.⁠ ⁠Write a summary of the customer's issue.
2.⁠ ⁠List the actions that need to be taken.
3.⁠ ⁠Suggest which department should handle it (Tech Support, Billing, Orders, General).
4.⁠ ⁠Recommend a solution based on past data.
5.⁠ ⁠Give an estimated resolution time.
Do not use any formatting like asterisks, bullet points, or markdown. Return everything in plain, readable text.
"""

# ✅ Function to process input
def process_query(user_input):
    if not user_input.strip():
        return "Please enter a support query."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    try:
        return groq.chat(messages)
    except Exception as e:
        return f"Error: {e}"

# ✅ Custom CSS for clean, modern style
custom_css = """
body {
    background-color: #121212;
}
.gradio-container {
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
textarea, input {
    background-color: #1e1e1e !important;
    color: white !important;
    border: 1px solid #a64dff !important;
}
button {
    background-color: #7f00ff !important;
    color: white !important;
    border-radius: 8px;
    font-weight: bold;
}
button:hover {
    background-color: #a64dff !important;
}
"""

# ✅ Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("""
        <div style="text-align:center; padding: 20px;">
            <h1 style="color:#a64dff;">ZenFlowCX: AI-Powered Multi-Agent System for Seamless Customer Experience</h1>
            <p style="color:gray;">Unified response generation with summary, action steps, routing, resolution & time</p>
        </div>
    """)
    
    with gr.Row():
        user_input = gr.Textbox(label="Describe your issue", placeholder="Ex: I was charged twice on my last invoice...", lines=4)
    submit_btn = gr.Button("Submit", variant="primary")
    result_output = gr.Textbox(label="AI Response", lines=20)

    submit_btn.click(fn=process_query, inputs=user_input, outputs=result_output)

demo.launch()
