import json
import os

import requests


def pwd() -> str:
    return os.getcwd()


def ls(path: str) -> list[str]:
    return os.listdir(path)


def cat(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


FUNCTIONS = {"pwd": pwd, "ls": ls, "cat": cat}
BACKEND = "ollama"
URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
CONTEXT_SIZE = 32768
CONTEXT_WARING_THRESHOLD = 80


with open("tools.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def tool_def_to_ollama(tool_def: dict) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": definition["description"],
                "parameters": definition["parameters"],
            },
        }
        for name, definition in tool_def.items()
    ]


CONVERTERS = {
    "ollama": tool_def_to_ollama,
}

TOOLS = CONVERTERS[BACKEND](config["tools"])


def execute_command(name: str, args: dict) -> str:
    if name not in FUNCTIONS:
        return f"ERROR: Unknown tool: {name}"

    try:
        result = FUNCTIONS[name](**args)
        return f"SUCCESS: {result}"
    except Exception as e:
        return f"ERROR: {e}"


def show_context_usage(data: dict):
    prompt_tokens = data.get("prompt_eval_count", 0)
    usage = (prompt_tokens / CONTEXT_SIZE) * 100

    print(f"\n[Context: {prompt_tokens}/{CONTEXT_SIZE} tokens ({usage:.1f}%)]")

    if usage >= CONTEXT_WARING_THRESHOLD:
        print("[WARNING] Context window is almost full!")


SYSTEM_PROMPT = (
    config["instructions"]["description"]
    + "\n"
    + "\n".join(config["instructions"]["rules"])
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})

    for _ in range(10):
        response = requests.post(
            URL,
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
                "think": True,
                "options": {"num_ctx": CONTEXT_SIZE},
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()
        show_context_usage(data)
        assistant_message = data["message"]

        messages.append(assistant_message)
        # print(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            return assistant_message.get("content", "")

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"Calling: {name}({args})")

            result = execute_command(name, args)

            messages.append({"role": "tool", "tool_name": name, "content": result})

    return "ERROR: Maximum tool iterations reached."


def clear_history():
    global messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]


if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ("/exit", "/quit"):
            break
        if user_input.lower() == "/clear":
            clear_history()
            continue
        if user_input.lower() == "/model":
            print(MODEL)
            continue
        if user_input.lower() == "/tools":
            print(str(FUNCTIONS.keys()))
            continue

        try:
            print("\nAssistant: ", chat(user_input))
        except requests.RequestException as e:
            print("\nAPI error: ", e)
