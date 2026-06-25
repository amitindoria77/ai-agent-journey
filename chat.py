import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": user_input}]
    )
    print("Claude:", response.content[0].text)