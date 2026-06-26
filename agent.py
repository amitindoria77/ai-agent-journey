import anthropic
import chromadb
import os
import requests

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Step 1: Define the tool
tools = [
    {
        "name": "calculator",
        "description": "Perform a basic math calculation. Use this whenever the user asks a math question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '47 * 12'"
                }
            },
            "required": ["expression"]
        }
    },
    {
    "name": "get_weather",
    "description": "Get the current weather for a city. Use this whenever the user asks about weather.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
        },
        "required": ["city"]
    }
}
]

# Step 2: The actual function that runs when the tool is called
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
    
def get_weather(city):
    geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()
    temp = weather["current_weather"]["temperature"]
    return f"{temp}°C in {city}"

# Step 3: Interactive loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break

    messages = [{"role": "user", "content": user_input}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        tools=tools,
        messages=messages
    )
    # Step 4: Check if Claude asked to use a tool
    if response.stop_reason == "tool_use":
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        tool_results = []

        for tool_use_block in tool_use_blocks:
            tool_name = tool_use_block.name
            tool_input = tool_use_block.input

            print(f"Claude wants to call: {tool_name} with {tool_input}")

            if tool_name == "calculator":
                result = calculator(tool_input["expression"])
            elif tool_name == "get_weather":
                result = get_weather(tool_input["city"])

            print(f"Local result: {result}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": result
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        final_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )

        print("Claude:", final_response.content[0].text)
    else:
        print("Claude:", response.content[0].text)
