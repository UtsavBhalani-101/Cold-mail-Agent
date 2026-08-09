import json
import os
import re
import sys
from datetime import datetime, timezone
from time import sleep
from typing import Any

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv


DEFAULT_MODEL = "thinkingmachines/inkling"
DEFAULT_COMPANY = "Val Town"
DEFAULT_REPORT_PATH = "results/company_research.md"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
ABSTRACT_EMAIL_VALIDATION_URL = "https://emailreputation.abstractapi.com/v1/?api_key=4d9e3593f7c84481973f3ee6c6ba0dc8&email=utsavbhalani.tech@gmail.com"
ABSTRACT_RATE_LIMIT_DELAY_SECONDS = 1.1
MAX_TOOL_ROUNDS = 10
MAX_PAGE_CHARS = 12000
CONFIDENCE_VALUES = {"high", "medium", "low", "unverifiable"}
EMAIL_REGEX = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}$",
    re.IGNORECASE,
)
OBFUSCATED_LOCAL_PART_MARKERS = {"protected", "email", "hidden", "obfuscated", "encoded"}

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
    "confidence": '"high" | "medium" | "low" | "unverifiable"',
    "needs_manual_check": "boolean",
    "is_catch_all_domain": "boolean",
    "verification_status": '"valid" | "invalid" | "risky" | "unknown" | "skipped" | null',
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


def abstract_bool(value: object) -> bool | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def classify_abstract_email_result(result: dict[str, Any]) -> str:
    """Normalize Abstract's old and new email API response shapes."""
    status = str(
        result.get("deliverability")
        or (result.get("email_deliverability") or {}).get("status")
        or ""
    ).strip().lower()
    risk_status = str(
        (result.get("email_risk") or {}).get("address_risk_status") or ""
    ).strip().lower()

    if status in {"deliverable", "valid"}:
        return "risky" if risk_status in {"high", "risky"} else "valid"
    if status in {"undeliverable", "invalid"}:
        return "invalid"
    if status in {"risky"}:
        return "risky"
    return "unknown"


def is_abstract_catch_all(result: dict[str, Any]) -> bool:
    legacy_value = abstract_bool(result.get("is_catchall_email"))
    if legacy_value is not None:
        return legacy_value

    email_quality = result.get("email_quality") or {}
    quality_value = abstract_bool(email_quality.get("is_catchall"))
    return bool(quality_value)


def call_abstract_email_validation(email: str, api_key: str) -> dict[str, Any]:
    response = requests.get(
        ABSTRACT_EMAIL_VALIDATION_URL,
        params={"api_key": api_key, "email": email, "auto_correct": "false"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Abstract API returned a non-object response.")
    return data


def is_plausible_email(email_string: object) -> bool:
    email = str(email_string or "").strip()
    if not EMAIL_REGEX.fullmatch(email):
        return False

    local_part = email.rsplit("@", 1)[0].lower()
    return not any(marker in local_part for marker in OBFUSCATED_LOCAL_PART_MARKERS)


def normalize_domain(value: object) -> str:
    domain = str(value or "").strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return domain.removeprefix("www.").strip()


def infer_first_email(person: dict[str, Any] | None, domain: object) -> str | None:
    normalized_domain = normalize_domain(domain)
    if "." not in normalized_domain:
        return None

    name = str((person or {}).get("name") or "").strip()
    first_name_match = re.search(r"[A-Za-z][A-Za-z'-]*", name)
    if not first_name_match:
        return None

    first_name = re.sub(r"[^a-z0-9]", "", first_name_match.group(0).lower())
    candidate = f"{first_name}@{normalized_domain}" if first_name else ""
    return candidate if is_plausible_email(candidate) else None


def guard_scraped_email(email_result: dict[str, Any], person: dict[str, Any] | None = None) -> dict[str, Any]:
    guarded = dict(email_result or {})
    email = str(guarded.get("email") or "").strip()
    if not email or is_plausible_email(email):
        if email:
            guarded["email"] = email
        return guarded

    domain = guarded.get("domain") or (email.rsplit("@", 1)[-1] if "@" in email else None)
    inferred_email = infer_first_email(person, domain)
    obfuscation_note = (
        f"Rejected scraped email '{email}' because it is not a plausible real address "
        "or appears to be an anti-scraping placeholder; the source page likely uses email obfuscation."
    )

    if inferred_email:
        guarded.update(
            {
                "email": inferred_email,
                "is_inferred": True,
                "pattern": "first@domain",
                "domain": normalize_domain(domain),
                "confidence": "low",
                "needs_manual_check": True,
                "verification_status": None,
                "notes": append_note(guarded.get("notes"), obfuscation_note),
            }
        )
        return guarded

    guarded.update(
        {
            "email": None,
            "is_inferred": False,
            "pattern": None,
            "domain": normalize_domain(domain) or guarded.get("domain"),
            "confidence": "low",
            "needs_manual_check": True,
            "verification_status": "skipped",
            "notes": append_note(
                guarded.get("notes"),
                f"{obfuscation_note} No first@domain fallback could be inferred from the available person/domain context.",
            ),
        }
    )
    return guarded


def confidence_for_verification_status(status: str) -> str:
    return "high" if status == "valid" else "low"


def verify_email(email_result: dict[str, Any], person: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify an email with Abstract API and guard inferred emails against catch-all domains."""
    verified = guard_scraped_email(email_result, person)
    email = str(verified.get("email") or "").strip()
    verified.setdefault("needs_manual_check", False)
    verified.setdefault("is_catch_all_domain", False)
    verified.setdefault("verification_status", None)

    if not email:
        verified["verification_status"] = "skipped"
        verified["needs_manual_check"] = True
        return verified

    api_key = os.getenv("ABSTRACT_EMAIL_VALIDATION_API_KEY") or os.getenv("ABSTRACT_API_KEY")
    if not api_key:
        verified["verification_status"] = "skipped"
        verified["needs_manual_check"] = True
        verified["confidence"] = "low"
        notes = verified.get("notes")
        missing_key_note = (
            "Email verification skipped because ABSTRACT_EMAIL_VALIDATION_API_KEY "
            "or ABSTRACT_API_KEY is not configured."
        )
        verified["notes"] = f"{notes} {missing_key_note}".strip() if notes else missing_key_note
        return verified

    try:
        real_result = call_abstract_email_validation(email, api_key)
        status = classify_abstract_email_result(real_result)
        verified["verification_status"] = status
        verified["abstract_quality_score"] = real_result.get("quality_score") or (
            real_result.get("email_quality") or {}
        ).get("score")

        domain = str(verified.get("domain") or email.rsplit("@", 1)[-1]).strip().lower()
        should_check_catch_all = bool(verified.get("is_inferred")) and "@" in email and domain
        if should_check_catch_all:
            catch_all_email = f"zzz-notreal-check@{domain}"
            sleep(ABSTRACT_RATE_LIMIT_DELAY_SECONDS)
            catch_all_result = call_abstract_email_validation(catch_all_email, api_key)
            catch_all_status = classify_abstract_email_result(catch_all_result)
            catch_all_domain = catch_all_status == "valid" or is_abstract_catch_all(catch_all_result)
            verified["is_catch_all_domain"] = catch_all_domain
            verified["catch_all_check_email"] = catch_all_email
            verified["catch_all_verification_status"] = catch_all_status

            if catch_all_domain:
                verified["confidence"] = "unverifiable"
                verified["needs_manual_check"] = True
                verified["notes"] = append_note(
                    verified.get("notes"),
                    "Abstract API catch-all probe was also valid; domain accepts invalid-looking addresses.",
                )
                return verified

        verified["confidence"] = confidence_for_verification_status(status)
        verified["needs_manual_check"] = status == "invalid"
        if status in {"risky", "unknown"}:
            verified["needs_manual_check"] = False
        return verified
    except requests.HTTPError as exc:
        verified["verification_status"] = "unknown"
        verified["confidence"] = "low"
        verified["needs_manual_check"] = True
        verified["notes"] = append_note(verified.get("notes"), f"Abstract API error: {exc}")
        return verified
    except (requests.RequestException, ValueError) as exc:
        verified["verification_status"] = "unknown"
        verified["confidence"] = "low"
        verified["needs_manual_check"] = True
        verified["notes"] = append_note(verified.get("notes"), f"Email verification failed: {exc}")
        return verified


def append_note(existing_note: object, extra_note: str) -> str:
    existing_text = str(existing_note or "").strip()
    if existing_text:
        return f"{existing_text} {extra_note}"
    return extra_note


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo and return the top 5 results. "
                "Use this when current or source-backed public web information is needed."
            ),
            "parameters": {
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch a web page by URL and return readable visible text with HTML tags removed. "
                "Use this after web_search when a result URL looks likely to contain the answer."
            ),
            "parameters": {
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
        },
    },
]


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


def call_tool(tool_name: str, tool_args: dict[str, Any], tool_state: dict[str, Any]) -> dict[str, object]:
    tool = TOOLS_BY_NAME.get(tool_name)
    if tool is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        args = dict(tool_args or {})
        if tool_name == "web_search":
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

        if tool_name == "fetch_page":
            url = normalize_text(args.get("url")).rstrip("/")
            seen_urls = tool_state["seen_urls"]
            if url in seen_urls:
                return budget_error(f"Duplicate fetch blocked: {args.get('url')}")

            if tool_state["fetches"] >= tool_state["max_fetches"]:
                return budget_error("Fetch budget exhausted.")

            seen_urls.add(url)
            tool_state["fetches"] += 1

        result = tool(**args)
        if tool_name == "web_search":
            return {"results": result}
        if tool_name == "fetch_page":
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
        "Reject scraped emails whose local part contains protected, email, hidden, obfuscated, or encoded; "
        "those are anti-scraping placeholders, not found addresses. "
        "When the task asks for email inference, mark is_inferred=true and keep confidence low unless a source verifies the address.\n\n"
        f"Prior context:\n{context_text}\n\n"
        "Return only one valid JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        'Use null for unknown values and "low" confidence when uncertain. Do not include markdown or extra text.'
    )


def make_client() -> tuple[str, str, str]:
    load_dotenv()

    api_key = os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NIM_API_KEY or NVIDIA_API_KEY. Add it to your .env file, for example: "
            "NIM_API_KEY=your_api_key_here"
        )

    model = os.getenv("NIM_MODEL") or os.getenv("NVIDIA_MODEL") or DEFAULT_MODEL
    base_url = (os.getenv("NVIDIA_BASE_URL") or NVIDIA_BASE_URL).rstrip("/")
    return api_key, model, base_url


def create_chat_completion(
    api_key: str,
    base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("NVIDIA API returned a non-object response.")
    return data


def force_final_answer(
    api_key: str,
    model: str,
    base_url: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    messages.append(
        {
            "role": "user",
            "content": (
                "Stop using tools. Based only on the evidence already returned by tools, "
                "produce the final JSON now. Use null for unknown fields. If an email is only "
                "inferred, clearly set is_inferred=true. Do not return anti-scraping placeholder "
                "emails whose local part contains protected, email, hidden, obfuscated, or encoded."
            ),
        }
    )
    final_response = create_chat_completion(
        api_key,
        base_url,
        {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
    )
    content = final_response["choices"][0]["message"].get("content") or "{}"
    return parse_json_object(content)


def run_agent(
    api_key: str,
    model: str,
    base_url: str,
    task: str,
    schema: dict[str, Any],
    context: dict[str, Any] | None = None,
    max_searches: int = 4,
    max_fetches: int = 2,
    max_contact_searches: int = 1,
) -> dict[str, Any]:
    use_tools = max_searches > 0 or max_fetches > 0
    messages = [{"role": "user", "content": build_agent_prompt(task, schema, context)}]
    tool_state = new_tool_state(max_searches, max_fetches, max_contact_searches)

    for _ in range(MAX_TOOL_ROUNDS):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        if use_tools:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        response = create_chat_completion(api_key, base_url, payload)
        response_message = response["choices"][0]["message"]
        tool_calls = response_message.get("tool_calls") or []
        if not tool_calls:
            try:
                return parse_json_object(response_message.get("content") or "{}")
            except (json.JSONDecodeError, ValueError):
                messages.append(response_message)
                return force_final_answer(api_key, model, base_url, messages)

        messages.append(response_message)
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            function_name = function.get("name") or ""
            function_args_text = function.get("arguments") or "{}"
            try:
                function_args = json.loads(function_args_text)
            except json.JSONDecodeError:
                function_args = {}
            print(
                f"Calling tool: {function_name}({function_args})",
                file=sys.stderr,
            )
            tool_response = call_tool(function_name, function_args, tool_state)
            messages.append(
                {
                    "tool_call_id": tool_call.get("id"),
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(tool_response, ensure_ascii=False),
                }
            )

        if (
            tool_state["searches"] >= tool_state["max_searches"]
            or tool_state["fetches"] >= tool_state["max_fetches"]
            or (
                tool_state["max_contact_searches"] > 0
                and tool_state["contact_searches"] >= tool_state["max_contact_searches"]
            )
        ):
            return force_final_answer(api_key, model, base_url, messages)

    return force_final_answer(api_key, model, base_url, messages)


def markdown_value(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "N/A"
    return str(value)


def markdown_link(value: object) -> str:
    text = markdown_value(value)
    if text == "N/A":
        return text
    return f"[{text}]({text})"


def format_company_research_markdown(record: dict[str, Any]) -> str:
    stage_size = record.get("stage_size") or {}
    decision_maker = record.get("decision_maker") or {}
    person = record.get("person") or {}
    email = record.get("email") or {}
    personalization = record.get("personalization") or {}

    source_urls = stage_size.get("source_urls") or []
    source_lines = "\n".join(f"- {markdown_link(url)}" for url in source_urls) or "- N/A"
    raw_json = json.dumps(record, indent=2, ensure_ascii=False)

    return (
        f"## {markdown_value(record.get('company_name'))}\n\n"
        f"- Researched at: {markdown_value(record.get('researched_at'))}\n\n"
        "### Company\n\n"
        f"- Domain: {markdown_value(stage_size.get('company_domain'))}\n"
        f"- Stage: {markdown_value(stage_size.get('stage'))}\n"
        f"- Headcount: {markdown_value(stage_size.get('headcount'))}\n"
        f"- Funding: {markdown_value(stage_size.get('funding'))}\n"
        f"- Confidence: {markdown_value(stage_size.get('confidence'))}\n"
        f"- Notes: {markdown_value(stage_size.get('notes'))}\n\n"
        "### Decision Maker\n\n"
        f"- Likely role: {markdown_value(decision_maker.get('likely_role'))}\n"
        f"- Confidence: {markdown_value(decision_maker.get('confidence'))}\n"
        f"- Rationale: {markdown_value(decision_maker.get('rationale'))}\n\n"
        "### Person\n\n"
        f"- Name: {markdown_value(person.get('name'))}\n"
        f"- Role: {markdown_value(person.get('role'))}\n"
        f"- Profile: {markdown_link(person.get('profile_url'))}\n"
        f"- Source: {markdown_link(person.get('source_url'))}\n"
        f"- Confidence: {markdown_value(person.get('confidence'))}\n"
        f"- Notes: {markdown_value(person.get('notes'))}\n\n"
        "### Email\n\n"
        f"- Email: {markdown_value(email.get('email'))}\n"
        f"- Inferred: {markdown_value(email.get('is_inferred'))}\n"
        f"- Pattern: {markdown_value(email.get('pattern'))}\n"
        f"- Domain: {markdown_value(email.get('domain'))}\n"
        f"- Confidence: {markdown_value(email.get('confidence'))}\n"
        f"- Verification status: {markdown_value(email.get('verification_status'))}\n"
        f"- Needs manual check: {markdown_value(email.get('needs_manual_check'))}\n"
        f"- Catch-all domain: {markdown_value(email.get('is_catch_all_domain'))}\n"
        f"- Catch-all probe: {markdown_value(email.get('catch_all_check_email'))}\n"
        f"- Source: {markdown_link(email.get('source_url'))}\n"
        f"- Notes: {markdown_value(email.get('notes'))}\n\n"
        "### Personalization\n\n"
        f"- Summary: {markdown_value(personalization.get('summary'))}\n"
        f"- Activity type: {markdown_value(personalization.get('activity_type'))}\n"
        f"- Activity date: {markdown_value(personalization.get('activity_date'))}\n"
        f"- Source: {markdown_link(personalization.get('source_url'))}\n"
        f"- Confidence: {markdown_value(personalization.get('confidence'))}\n"
        f"- Notes: {markdown_value(personalization.get('notes'))}\n\n"
        "### Company Sources\n\n"
        f"{source_lines}\n\n"
        "<details>\n"
        "<summary>Raw JSON</summary>\n\n"
        "```json\n"
        f"{raw_json}\n"
        "```\n"
        "</details>\n"
    )


def slugify_filename(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "company"


def save_company_research_report(record: dict[str, Any], report_path: str | None = None) -> str:
    report_path = report_path or os.getenv("RESEARCH_REPORT_PATH", DEFAULT_REPORT_PATH)
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    report_exists = os.path.exists(report_path) and os.path.getsize(report_path) > 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    company_slug = slugify_filename(str(record.get("company_name") or DEFAULT_COMPANY))
    entry = format_company_research_markdown(record)
    header = "" if report_exists else "# Company Research\n\n"

    with open(report_path, "a", encoding="utf-8") as report:
        report.write(f"{header}{entry}\n\n---\n\n")

    snapshot_path = os.path.join(
        report_dir or ".",
        f"company_research_{timestamp}.md",
    )
    with open(snapshot_path, "w", encoding="utf-8") as snapshot:
        snapshot.write(f"# Company Research\n\n{entry}\n")

    return os.path.abspath(snapshot_path)


def research_company(company_name: str, report_path: str | None = None) -> dict[str, Any]:
    api_key, model, base_url = make_client()
    company_name = company_name.strip()
    if not company_name:
        raise ValueError("company_name is required.")

    stage_size = run_agent(
        api_key=api_key,
        model=model,
        base_url=base_url,
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
        api_key=api_key,
        model=model,
        base_url=base_url,
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
        api_key=api_key,
        model=model,
        base_url=base_url,
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
        api_key=api_key,
        model=model,
        base_url=base_url,
        task=(
            "Find or infer the person's email. First search for a public verified email for the person. "
            "If none is visible, infer a likely address from the company domain using common patterns such as "
            "first@domain, first.last@domain, firstinitiallast@domain, or firstlast@domain. "
            "Do not accept scraped placeholder addresses whose local part contains protected, email, hidden, "
            "obfuscated, or encoded; fall back to inference or null with an obfuscation note. "
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
    email = verify_email(email, person)

    personalization = run_agent(
        api_key=api_key,
        model=model,
        base_url=base_url,
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
    record["report_path"] = save_company_research_report(record, report_path=report_path)
    return record


def main() -> int:
    company_name = " ".join(sys.argv[1:]).strip() or os.getenv("COMPANY_NAME", DEFAULT_COMPANY)
    try:
        record = research_company(company_name)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report_path = record.get("report_path")
    if report_path:
        print(f"Saved report: {report_path}", file=sys.stderr)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
