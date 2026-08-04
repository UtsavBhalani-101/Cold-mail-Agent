import json
import os
import sys

from ddgs import DDGS
from dotenv import load_dotenv
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_COMPANY = "OpenAI"
MAX_TOOL_ROUNDS = 3


def web_search(query: str) -> list[dict[str, str]]:
    """Search DuckDuckGo and return the top 5 organic results."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

    formatted_results = []
    for result in results:
        formatted_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("href") or result.get("url", ""),
                "snippet": result.get("body") or result.get("snippet", ""),
            }
        )

    return formatted_results


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search DuckDuckGo and return top 5 web results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to run.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOLS_BY_NAME = {
    "web_search": web_search,
}


def build_prompt(company: str) -> str:
    return (
        f"Search for 10 engineers at {company}. "
        "Use the web_search tool if you need current public web results. "
        "After searching, answer with the likely person or people and cite the source URLs you used."
    )


def call_tool(tool_name: str, arguments_json: str) -> str:
    tool = TOOLS_BY_NAME.get(tool_name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
        results = tool(**args)
        return json.dumps({"results": results})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "Missing GROQ_API_KEY. Add it to your .env file, for example:\n"
            "GROQ_API_KEY=your_groq_api_key_here",
            file=sys.stderr,
        )
        return 1

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)

    company = " ".join(sys.argv[1:]).strip() or os.getenv("COMPANY_NAME", DEFAULT_COMPANY)
    prompt = build_prompt(company)

    messages = [
        {"role": "user", "content": prompt}
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            print(response_message.content)
            return 0

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            print(f"Calling tool: {function_name}({function_args})")

            tool_result = call_tool(function_name, function_args)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result,
                }
            )

    print("The model did not produce a final answer after the tool-call limit.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
