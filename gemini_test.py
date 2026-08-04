import json
import os
import sys

import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_PROMPT = "Find the names of people on the team page of Val Town."
MAX_TOOL_ROUNDS = 8
MAX_TOTAL_SEARCHES = 6
MAX_TOTAL_FETCHES = 4
MAX_CONTACT_SEARCHES = 2
MAX_PAGE_CHARS = 12000
CONFIDENCE_VALUES = {"high", "medium", "low"}
PERSON_SCHEMA = {
    "name": "string or null",
    "role": "string or null",
    "email": "string or null",
    "confidence": '"high" | "medium" | "low"',
    "source_url": "string",
    "personalization_hook": "string or null",
}
FINAL_SCHEMA = {
    "people": [PERSON_SCHEMA],
    "notes": "string or null",
}
REQUIRED_FINAL_KEYS = set(FINAL_SCHEMA)
REQUIRED_PERSON_KEYS = set(PERSON_SCHEMA)


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
        "Answer only after using the page text when it is available.\n"
        "IMPORTANT SEARCH CONSTRAINTS:\n"
        "- Do NOT perform repetitive or speculative search queries to guess or locate missing fields such as emails or personal handles.\n"
        "- Only include emails that are explicitly visible in fetched page text or search result snippets. Never infer an address pattern.\n"
        "- If a field such as email is not plainly visible, set it to null immediately.\n"
        f"- You have at most {MAX_TOTAL_SEARCHES} searches, {MAX_TOTAL_FETCHES} page fetches, and {MAX_CONTACT_SEARCHES} total email/contact searches.\n"
        "- Conclude tool usage as soon as the main request is answered.\n\n"
        "Your final answer must be only one valid JSON object matching this schema:\n"
        f"{json.dumps(FINAL_SCHEMA, indent=2)}\n"
        'If you cannot find a field confidently, use null and set that person\'s "confidence" to "low" rather than guessing. '
        'Set "source_url" to the URL that best supports each person, or an empty string if no source supports it. '
        "Use notes for brief uncertainty, including when public emails were not found. "
        "Do not wrap the JSON in markdown or include any extra text."
    )


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def is_contact_search(query: str) -> bool:
    query = normalize_text(query)
    contact_terms = ("email", "e-mail", "@", "contact", "mail", "linkedin")
    return any(term in query for term in contact_terms)


def budget_error(message: str) -> dict[str, object]:
    return {
        "error": message,
        "instruction": (
            "Do not call more tools for this missing information. Return the final JSON now, "
            "using null for unknown fields."
        ),
    }


def call_tool(function_call: types.FunctionCall, tool_state: dict[str, object]) -> dict[str, object]:
    tool = TOOLS_BY_NAME.get(function_call.name)
    if tool is None:
        return {"error": f"Unknown tool: {function_call.name}"}

    try:
        args = dict(function_call.args or {})
        if function_call.name == "web_search":
            query = normalize_text(args.get("query"))
            seen_queries = tool_state["seen_queries"]
            if query in seen_queries:
                return budget_error(f"Duplicate search blocked: {args.get('query')}")

            if tool_state["searches"] >= MAX_TOTAL_SEARCHES:
                return budget_error("Search budget exhausted.")

            if is_contact_search(query):
                if tool_state["contact_searches"] >= MAX_CONTACT_SEARCHES:
                    return budget_error("Email/contact search budget exhausted.")
                tool_state["contact_searches"] += 1

            seen_queries.add(query)
            tool_state["searches"] += 1

        if function_call.name == "fetch_page":
            url = normalize_text(args.get("url")).rstrip("/")
            seen_urls = tool_state["seen_urls"]
            if url in seen_urls:
                return budget_error(f"Duplicate fetch blocked: {args.get('url')}")

            if tool_state["fetches"] >= MAX_TOTAL_FETCHES:
                return budget_error("Fetch budget exhausted.")

            seen_urls.add(url)
            tool_state["fetches"] += 1

        result = tool(**args)
        if function_call.name == "web_search":
            return {"results": result}
        if function_call.name == "fetch_page":
            return result
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


def parse_final_json(text: str) -> dict[str, object]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Final answer must be a JSON object.")

    missing_keys = REQUIRED_FINAL_KEYS - set(data)
    extra_keys = set(data) - REQUIRED_FINAL_KEYS
    if missing_keys:
        raise ValueError(f"Missing required keys: {sorted(missing_keys)}")
    if extra_keys:
        raise ValueError(f"Unexpected keys: {sorted(extra_keys)}")

    if not isinstance(data["people"], list):
        raise ValueError("people must be an array.")

    for index, person in enumerate(data["people"]):
        if not isinstance(person, dict):
            raise ValueError(f"people[{index}] must be an object.")

        missing_person_keys = REQUIRED_PERSON_KEYS - set(person)
        extra_person_keys = set(person) - REQUIRED_PERSON_KEYS
        if missing_person_keys:
            raise ValueError(f"people[{index}] missing required keys: {sorted(missing_person_keys)}")
        if extra_person_keys:
            raise ValueError(f"people[{index}] has unexpected keys: {sorted(extra_person_keys)}")

        nullable_string_fields = ["name", "role", "email", "personalization_hook"]
        for field in nullable_string_fields:
            if person[field] is not None and not isinstance(person[field], str):
                raise ValueError(f"people[{index}].{field} must be a string or null.")

        if person["confidence"] not in CONFIDENCE_VALUES:
            raise ValueError(f'people[{index}].confidence must be "high", "medium", or "low".')

        if not isinstance(person["source_url"], str):
            raise ValueError(f"people[{index}].source_url must be a string.")

    if data["notes"] is not None and not isinstance(data["notes"], str):
        raise ValueError("notes must be a string or null.")

    return data


def print_valid_final_json(text: str) -> bool:
    try:
        data = parse_final_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid JSON final answer: {exc}", file=sys.stderr)
        return False

    print(json.dumps(data, indent=2, ensure_ascii=False))
    return True


def force_final_answer(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
) -> int:
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "Stop using tools. Based only on the evidence already returned by tools, "
                        "produce the final JSON now. Use null for emails or other fields that were "
                        "not explicitly visible in the tool results."
                    )
                )
            ],
        )
    )
    final_response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return 0 if print_valid_final_json(final_response.text or "") else 1


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
    config = types.GenerateContentConfig(
        tools=[tool],
        response_mime_type="application/json",
    )
    user_request = " ".join(sys.argv[1:]).strip() or os.getenv("TEST_PROMPT", DEFAULT_PROMPT)
    prompt = build_prompt(user_request)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    tool_state = {
        "searches": 0,
        "fetches": 0,
        "contact_searches": 0,
        "seen_queries": set(),
        "seen_urls": set(),
    }

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        function_calls = response.function_calls or []
        if not function_calls:
            if print_valid_final_json(response.text or ""):
                return 0

            contents.append(response.candidates[0].content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "Retry once. Return only a valid JSON object with exactly these keys: "
                                f"{', '.join(sorted(REQUIRED_FINAL_KEYS))}. "
                                'Use null for unknown values, and use "low" confidence if anything is uncertain.'
                            )
                        )
                    ],
                )
            )

            retry_response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            if retry_response.function_calls:
                print("Retry produced tool calls instead of final JSON.", file=sys.stderr)
                return 1
            return 0 if print_valid_final_json(retry_response.text or "") else 1

        contents.append(response.candidates[0].content)

        for function_call in function_calls:
            print(
                f"Calling tool: {function_call.name}({dict(function_call.args or {})})",
                file=sys.stderr,
            )
            tool_response = call_tool(function_call, tool_state)
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

        if (
            tool_state["searches"] >= MAX_TOTAL_SEARCHES
            or tool_state["fetches"] >= MAX_TOTAL_FETCHES
            or tool_state["contact_searches"] >= MAX_CONTACT_SEARCHES
        ):
            return force_final_answer(client, model, contents)

    print("Tool round limit reached; forcing a final JSON answer.", file=sys.stderr)
    return force_final_answer(client, model, contents)


if __name__ == "__main__":
    raise SystemExit(main())
