import ollama

MODEL_NAME="llama3.1"
print('Generating AI response...')
response=ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            'role':"user",
            'content':"Why is the sky blue? Explain in one sentence."

        }
    ]
)

print(response['message']['content'])

print("\n generating AI response..")

response=ollama.chat(
    model=MODEL_NAME,
    messages=[{
        'role':"user",
        'content':"What is the capital of France?"
    }]
)

print(response['message']['content'])