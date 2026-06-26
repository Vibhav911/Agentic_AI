import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json'
}


history = []


def generate_response(prompt):
    history.append(prompt)
    final_prompt = ''.join(history)
    data = {
        'model': 'codellama',
        'prompt': final_prompt,
        'stream': False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response = response.text
        data = json.loads(response)
        actual_response = data['response']
        return actual_response
    else:
        print("error:", response.text)
        
        
        
        
# Creating Frontend Interface
interface = gr.Interface(
    fn=generate_response,
    inputs = gr.Textbox(lines=4, placeholder="Enter your code related question here..."),
    outputs = gr.Textbox(label="Answer"),
    title="CodeGuru: Your Code Teaching Assistant",
    description="Ask anything about coding and get instant answers"
)

interface.launch(share=True)
    

    
    
    