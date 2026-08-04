import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_COMPANY = "Val Town"
DEFAULT_DB_PATH = "company_research.sqlite3"
MAX_TOOL_ROUNDS = 10
MAX_PAGE_CHARS = 12000
CONFIDENCE_VALUES = {"high", "medium", "low"}

STAGE_SIZE_SCHEMA = {
    "stage": "string or null",
    "headcount": "string or null",
    "funding": "string or null",
    "company_domain": "string or null",
    "confidence": '"high" | "medium" | "low"',
    "source_urls": ["string"],
    "notes": "string or null",
}
DECISION_MAKER_SCHEMA = {
    "likely_role": "string",
    "decision_rule": "string",
    "rationale": "string",
    "confidence": '"high" | "medium" | "low"',
}
PERSON_SCHEMA = {
    "name": "string or null",
    "role": "string or null",
    "profile_url": "string or null",
    "source_url": "string",
    "confidence": '"high" | "medium" | "low"',
    "notes": "string or null",
}
EMAIL_SCHEMA = {
    "email": "string or null",
    "is_inferred": "boolean",
    "pattern": "string or null",
    "domain": "string or null",
    "source_url": "string",
    "confidence": '"high" | "medium" | "low"',
    "notes": "string or null",
}
PERSONALIZATION_SCHEMA = {
    "summary": "string or null",
    "activity_type": "string or null",
    "activity_date": "string or null",
    "source_url": "string",
    "confidence": '"high" | "medium" | "low"',
    "notes": "string or null",
}
COMPANY_RESEARCH_SCHEMA = {
    "company_name": "string",
    "researched_at": "ISO-8601 string",
    "stage_size": STAGE_SIZE_SCHEMA,
    "decision_maker": DECISION_MAKER_SCHEMA,
    "person": PERSON_SCHEMA,
    "email": EMAIL_SCHEMA,
    "personalization": PERSONALIZATION_SCHEMA,
}


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


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def is_contact_search(query: str) -> bool:
    query = normalize_text(query)
    contact_terms = ("email", "e-mail", "@", "contact", "mail", "hunter", "apollo", "rocketreach")
    return any(term in query for term in contact_terms)


def budget_error(message: str) -> dict[str, object]:
    return {
        "error": message,
        "instruction": (
            "Do not call more tools for this missing information. Return the final JSON now, "
            "using null for unknown fields or a clearly marked inference when requested."
        ),
    }


def new_tool_state(
    max_searches: int,
    max_fetches: int,
    max_contact_searches: int,
) -> dict[str, Any]:
    return {
        "searches": 0,
        "fetches": 0,
        "contact_searches": 0,
        "max_searches": max_searches,
        "max_fetches": max_fetches,
        "max_contact_searches": max_contact_searches,
        "seen_queries": set(),
        "seen_urls": set(),
    }


def call_tool(function_call: types.FunctionCall, tool_state: dict[str, Any]) -> dict[str, object]:
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

            if tool_state["searches"] >= tool_state["max_searches"]:
                return budget_error("Search budget exhausted.")

            if is_contact_search(query):
                if tool_state["contact_searches"] >= tool_state["max_contact_searches"]:
                    return budget_error("Email/contact search budget exhausted.")
                tool_state["contact_searches"] += 1

            seen_queries.add(query)
            tool_state["searches"] += 1

        if function_call.name == "fetch_page":
            url = normalize_text(args.get("url")).rstrip("/")
            seen_urls = tool_state["seen_urls"]
            if url in seen_urls:
                return budget_error(f"Duplicate fetch blocked: {args.get('url')}")

            if tool_state["fetches"] >= tool_state["max_fetches"]:
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


def parse_json_object(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Final answer must be a JSON object.")
    return data


def build_agent_prompt(task: str, schema: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    context_text = json.dumps(context or {}, indent=2, ensure_ascii=False)
    return (
        f"{task}\n\n"
        "Use web_search for discovery and fetch_page only for URLs likely to contain direct evidence. "
        "Do not repeat equivalent searches. Do not keep searching for fields after the available evidence is exhausted. "
        "Only treat emails as verified when they are explicitly visible in fetched page text or search snippets. "
        "When the task asks for email inference, mark is_inferred=true and keep confidence low unless a source verifies the address.\n\n"
        f"Prior context:\n{context_text}\n\n"
        "Return only one valid JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        'Use null for unknown values and "low" confidence when uncertain. Do not include markdown or extra text.'
    )


def make_client() -> tuple[genai.Client, str]:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it to your .env file, for example: "
            "GEMINI_API_KEY=your_api_key_here"
        )

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    return genai.Client(api_key=api_key), model


def force_final_answer(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
) -> dict[str, Any]:
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "Stop using tools. Based only on the evidence already returned by tools, "
                        "produce the final JSON now. Use null for unknown fields. If an email is only "
                        "inferred, clearly set is_inferred=true."
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
    return parse_json_object(final_response.text or "{}")


def run_agent(
    client: genai.Client,
    model: str,
    task: str,
    schema: dict[str, Any],
    context: dict[str, Any] | None = None,
    max_searches: int = 4,
    max_fetches: int = 2,
    max_contact_searches: int = 1,
) -> dict[str, Any]:
    use_tools = max_searches > 0 or max_fetches > 0
    if use_tools:
        tool = types.Tool(function_declarations=[web_search_declaration, fetch_page_declaration])
        config = types.GenerateContentConfig(
            tools=[tool],
            response_mime_type="application/json",
        )
    else:
        config = types.GenerateContentConfig(response_mime_type="application/json")
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=build_agent_prompt(task, schema, context))],
        )
    ]
    tool_state = new_tool_state(max_searches, max_fetches, max_contact_searches)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        function_calls = response.function_calls or []
        if not function_calls:
            try:
                return parse_json_object(response.text or "{}")
            except (json.JSONDecodeError, ValueError):
                contents.append(response.candidates[0].content)
                return force_final_answer(client, model, contents)

        contents.append(response.candidates[0].content)
        for function_call in function_calls:
            print(
                f"Calling tool: {function_call.name}({dict(function_call.args or {})})",
                file=sys.stderr,
            )
            tool_response = call_tool(function_call, tool_state)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=function_call.name,
                            response=tool_response,
                        )
                    ],
                )
            )

        if (
            tool_state["searches"] >= tool_state["max_searches"]
            or tool_state["fetches"] >= tool_state["max_fetches"]
            or (
                tool_state["max_contact_searches"] > 0
                and tool_state["contact_searches"] >= tool_state["max_contact_searches"]
            )
        ):
            return force_final_answer(client, model, contents)

    return force_final_answer(client, model, contents)


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS company_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                researched_at TEXT NOT NULL,
                stage TEXT,
                headcount TEXT,
                funding TEXT,
                company_domain TEXT,
                decision_maker_role TEXT,
                person_name TEXT,
                person_role TEXT,
                email TEXT,
                email_is_inferred INTEGER NOT NULL,
                personalization_summary TEXT,
                record_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_company_research_company_name
            ON company_research(company_name)
            """
        )


def save_company_research(record: dict[str, Any], db_path: str | None = None) -> None:
    db_path = db_path or os.getenv("RESEARCH_DB_PATH", DEFAULT_DB_PATH)
    init_db(db_path)

    stage_size = record.get("stage_size") or {}
    decision_maker = record.get("decision_maker") or {}
    person = record.get("person") or {}
    email = record.get("email") or {}
    personalization = record.get("personalization") or {}

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO company_research (
                company_name,
                researched_at,
                stage,
                headcount,
                funding,
                company_domain,
                decision_maker_role,
                person_name,
                person_role,
                email,
                email_is_inferred,
                personalization_summary,
                record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("company_name"),
                record.get("researched_at"),
                stage_size.get("stage"),
                stage_size.get("headcount"),
                stage_size.get("funding"),
                stage_size.get("company_domain"),
                decision_maker.get("likely_role"),
                person.get("name"),
                person.get("role"),
                email.get("email"),
                1 if email.get("is_inferred") else 0,
                personalization.get("summary"),
                json.dumps(record, ensure_ascii=False),
            ),
        )


def research_company(company_name: str, db_path: str | None = None) -> dict[str, Any]:
    client, model = make_client()
    company_name = company_name.strip()
    if not company_name:
        raise ValueError("company_name is required.")

    stage_size = run_agent(
        client=client,
        model=model,
        task=(
            f"Determine the stage and size of {company_name}. Search for funding, headcount, "
            "LinkedIn/company size snippets, Crunchbase/Tracxn-style summaries, and the official domain."
        ),
        schema=STAGE_SIZE_SCHEMA,
        max_searches=3,
        max_fetches=2,
        max_contact_searches=0,
    )

    decision_maker = run_agent(
        client=client,
        model=model,
        task=(
            "Infer the likely hiring decision-maker role for outbound recruiting. "
            "Use this rule: if the company appears to have fewer than 20 people, choose founder/co-founder; "
            "otherwise choose hiring manager, talent acquisition, recruiter, head of people, or department leader."
        ),
        schema=DECISION_MAKER_SCHEMA,
        context={
            "company_name": company_name,
            "stage_size": stage_size,
        },
        max_searches=0,
        max_fetches=0,
        max_contact_searches=0,
    )

    person = run_agent(
        client=client,
        model=model,
        task=(
            f"Find the best matching person's name at {company_name} for this target role: "
            f"{decision_maker.get('likely_role')}. Search the official team/about page, GitHub org, "
            "and Google-indexed LinkedIn using site:linkedin.com. Return one best person."
        ),
        schema=PERSON_SCHEMA,
        context={
            "company_name": company_name,
            "stage_size": stage_size,
            "decision_maker": decision_maker,
        },
        max_searches=4,
        max_fetches=2,
        max_contact_searches=1,
    )

    email = run_agent(
        client=client,
        model=model,
        task=(
            "Find or infer the person's email. First search for a public verified email for the person. "
            "If none is visible, infer a likely address from the company domain using common patterns such as "
            "first@domain, first.last@domain, firstinitiallast@domain, or firstlast@domain. "
            "Never mark an inferred address as verified."
        ),
        schema=EMAIL_SCHEMA,
        context={
            "company_name": company_name,
            "stage_size": stage_size,
            "person": person,
        },
        max_searches=2,
        max_fetches=1,
        max_contact_searches=2,
    )

    personalization = run_agent(
        client=client,
        model=model,
        task=(
            "Find one recent public thing this person did for cold-email personalization. "
            "Prefer a blog post, GitHub activity, conference/podcast appearance, public LinkedIn-indexed post, "
            "tweet/X result, funding announcement quote, or company news mention."
        ),
        schema=PERSONALIZATION_SCHEMA,
        context={
            "company_name": company_name,
            "person": person,
            "email": email,
        },
        max_searches=3,
        max_fetches=2,
        max_contact_searches=0,
    )

    record = {
        "company_name": company_name,
        "researched_at": datetime.now(timezone.utc).isoformat(),
        "stage_size": stage_size,
        "decision_maker": decision_maker,
        "person": person,
        "email": email,
        "personalization": personalization,
    }
    save_company_research(record, db_path=db_path)
    return record


def main() -> int:
    company_name = " ".join(sys.argv[1:]).strip() or os.getenv("COMPANY_NAME", DEFAULT_COMPANY)
    try:
        record = research_company(company_name)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
