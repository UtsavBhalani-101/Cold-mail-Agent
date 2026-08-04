import os
import sys

import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-flash-lite-latest"
DEFAULT_PROMPT = "Find the names of people on the team page of Val Town."
MAX_TOOL_ROUNDS = 5
MAX_PAGE_CHARS = 12000


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


def fetch_page(url: str) -> dict[str, str]:
    """Fetch a webpage and return simplified visible text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    readable_text = "\n".join(line for line in lines if line)

    if len(readable_text) > MAX_PAGE_CHARS:
        readable_text = readable_text[:MAX_PAGE_CHARS] + "\n...[truncated]"

    return {
        "url": response.url,
        "text": readable_text,
    }


web_search_declaration = types.FunctionDeclaration(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo and return the top 5 results. "
        "Use this when current or source-backed public web information is needed."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    response_json_schema={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["title", "url", "snippet"],
                },
            }
        },
        "required": ["results"],
    },
)


fetch_page_declaration = types.FunctionDeclaration(
    name="fetch_page",
    description=(
        "Fetch a web page by URL and return readable visible text with HTML tags removed. "
        "Use this after web_search when a result URL looks likely to contain the answer."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the page to fetch.",
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    response_json_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["url", "text"],
    },
)


TOOLS_BY_NAME = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}


def build_prompt(user_request: str) -> str:
    return (
        f"{user_request}\n\n"
        "Use web_search to find likely pages. Use fetch_page on promising URLs to read the actual page. "
        "Answer only after using the page text when it is available, and cite the source URLs you used."
    )


def call_tool(function_call: types.FunctionCall) -> dict[str, object]:
    tool = TOOLS_BY_NAME.get(function_call.name)
    if tool is None:
        return {"error": f"Unknown tool: {function_call.name}"}

    try:
        args = dict(function_call.args or {})
        result = tool(**args)
        if function_call.name == "web_search":
            return {"results": result}
        if function_call.name == "fetch_page":
            return result
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Missing GEMINI_API_KEY. Add it to your .env file, for example:\n"
            "GEMINI_API_KEY=your_api_key_here",
            file=sys.stderr,
        )
        return 1

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)
    tool = types.Tool(function_declarations=[web_search_declaration, fetch_page_declaration])
    config = types.GenerateContentConfig(tools=[tool])
    user_request = " ".join(sys.argv[1:]).strip() or os.getenv("TEST_PROMPT", DEFAULT_PROMPT)
    prompt = build_prompt(user_request)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        function_calls = response.function_calls or []
        if not function_calls:
            print(response.text)
            return 0

        contents.append(response.candidates[0].content)

        for function_call in function_calls:
            print(f"Calling tool: {function_call.name}({dict(function_call.args or {})})")
            tool_response = call_tool(function_call)
            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response=tool_response,
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[function_response_part],
                )
            )

    print("The model did not produce a final answer after the tool-call limit.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
