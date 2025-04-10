# agents/summary_action_agent.py

import ollama

def get_summary_and_actions(user_text):
    prompt = f"""
You are an AI customer support assistant.

Here is a customer message:
\"\"\"
{user_text}
\"\"\"

1. Summarize the issue in 2 lines.
2. Identify any actions that the support team should take (e.g., follow-up, escalation, ticket assignment).

Return the result as:
Summary: ...
Actions: ...
"""

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
