"""Report agent: turns the structured result into a short note.

Uses Claude when ANTHROPIC_API_KEY is set (or REPORT_LLM=on); otherwise, and on any error,
timeout or refusal, a deterministic template is returned. The LLM only receives the
structured facts - never the photo or any patient identity - and never changes the label."""
import asyncio
import json
import logging
import os

import anthropic

log = logging.getLogger("report")

REPORT_MODEL = os.getenv("REPORT_MODEL", "claude-opus-5-5")
REPORT_TIMEOUT = float(os.getenv("REPORT_TIMEOUT", "10"))
LLM_ENABLED = os.getenv("REPORT_LLM", "auto") == "on" or (
    os.getenv("REPORT_LLM", "auto") == "auto" and bool(os.getenv("ANTHROPIC_API_KEY")))

_client = anthropic.AsyncAnthropic(timeout=REPORT_TIMEOUT, max_retries=1) if LLM_ENABLED else None

LANG_NAMES = {"en": "English", "bn": "Bengali", "hi": "Hindi"}

SYSTEM = """You turn a structured diabetic-foot-ulcer classifier result into a short note for a health worker.
Rules:
- Use ONLY the facts given. Never change, question or reinterpret the Texas class, probabilities or flags.
- Say plainly that this is decision support from an AI model, not a diagnosis.
- Mention every flag in plain words, most urgent first.
- At most 90 words, no headings, no markdown.
- Write in the requested language."""

FLAG_TEXT = {
    "deep_ulcer_urgent_referral": "The wound may be deep (grade 2-3): refer urgently to a foot specialist.",
    "possible_infection": "Signs of infection: assess for cellulitis, discharge or fever and treat per protocol.",
    "possible_ischemia_check_pulses_abi": "Possible poor blood supply: check foot pulses and ABI/Doppler.",
    "low_confidence_needs_review": "The model is not confident: a clinician should review this photo.",
}


def template_report(result: dict, flags: list) -> str:
    g, s = result["grade"], result["stage"]
    parts = [f"Texas class {result['ut_class']}: {g['meaning']}, {s['meaning']} "
             f"(grade confidence {round(g['confidence'] * 100)}%)."]
    parts += [FLAG_TEXT[f] for f in flags if f in FLAG_TEXT]
    parts.append("AI decision support only - not a diagnosis. Confirm clinically.")
    return " ".join(parts)


async def write_report(result: dict, flags: list, lang: str = "en") -> str:
    if _client is None:
        return template_report(result, flags)
    facts = {"texas_class": result["ut_class"], "grade": result["grade"], "stage": result["stage"],
             "infection_probability": result["infection"]["prob"],
             "ischemia_probability": result["ischemia"]["prob"], "flags": flags,
             "language": LANG_NAMES.get(lang, "English")}
    try:
        resp = await asyncio.wait_for(
            _client.beta.messages.create(
                model=REPORT_MODEL,
                max_tokens=2000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low"},
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(facts, sort_keys=True)}],
            ),
            timeout=REPORT_TIMEOUT + 2,
        )
    except asyncio.TimeoutError:
        log.warning("report LLM timed out; using template")
        return template_report(result, flags)
    except anthropic.RateLimitError:
        log.warning("report LLM rate limited; using template")
        return template_report(result, flags)
    except anthropic.APIStatusError as e:
        log.warning("report LLM API error %s; using template", e.status_code)
        return template_report(result, flags)
    except anthropic.APIConnectionError:
        log.warning("report LLM unreachable; using template")
        return template_report(result, flags)

    if resp.stop_reason == "refusal":
        log.warning("report LLM refused; using template")
        return template_report(result, flags)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or template_report(result, flags)
