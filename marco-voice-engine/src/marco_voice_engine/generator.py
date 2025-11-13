"""Generador de variantes alineadas con la voz del goldset."""

from __future__ import annotations

from typing import Any, Dict, List
import time

import httpx

from .config import (
    get_generation_model_primary,
    get_openrouter_api_key,
    get_openrouter_chat_endpoint,
)
from .style_retriever import StyleRetriever


class GenerationError(Exception):
    """Errores derivados del proceso de generación."""


STYLE_CONTRACT_OPS = """
You are the ghostwriter for a single specific author.
Your job is to write short posts that sound like a real human operator, not an AI.

Hard rules:
- One idea per piece.
- Direct second person ("you").
- Voice: street-level, sharp, practical, no fluff.
- No emojis. No hashtags.
- No motivational quotes. No LinkedIn-style corporate tone.
- Always contain something concretely useful, uncomfortable, or sharply observant.
- Assume the reader is an intelligent operator/founder, skip basic explanations.
- Never explain what you are doing. Just write the posts.
- You are speaking only to solo founders (Day 1–Year 1), overloaded, doing every function themselves.
- They love the craft, hate ops, and are drowning in chaos.
- Every line must feel like step-zero tactical help or a sharp diagnostic for that person — no generic startup advice, no enterprise context.

Use the examples ONLY to capture:
- tone,
- rhythm range,
- density of meaning,
- level of aggression vs clarity,
- how the author talks to the reader.

Forbidden:
- Do not copy exact sentences from examples.
- Do not reuse fixed templates.
- Do not start every post with the same pattern.
- Do not produce checklist-pattern posts unless the idea itself clearly demands it.
If your output looks like a repeated template, you have failed.
""".strip()

STYLE_CONTRACT_CHAOS = """
You are the ghostwriter for the same author, but in his feral/chaotic mode.

Rules:
- Still intelligent. Still sharp. Still self-aware.
- Allowed: weird metaphors, brainrot references, dark humor, unhinged observations.
- No LinkedIn coach tone. No generic AI sludge. No inspirational posters.
- You can be playful, cynical, or absurd, but it must feel deliberate, not random noise.
- Punchlines > lectures. Show attitude, not advice manuals.
- Same audience: solo operators in early days (Year 0–1), overloaded and doing every function themselves.
- Chaos is allowed, but it stays anchored in their reality: overload, loneliness, internet brainrot, avoiding the real work they know they owe.
- If a line could apply to “everyone on LinkedIn”, it's wrong.

Forbidden:
- Do not explain frameworks.
- Do not sound like a brand or a corporation.
- Do not use "as an AI" or anything that breaks the character.
- No generic listicles or sterile how-to threads.
""".strip()

STYLE_CONTRACT_REPLY = """
Read the tweet. Don't think about it.

Write the first FUCKING thing that comes to your head.
One FUCKING sentence. Ugly is FUCKING good. DO NOT DEBATE OR ATTACK THE ORIGINAL TWEET.
You could be happy for the other person accomplishment, or just want to say something.

Send it.
""".strip()


def _build_examples_block(examples: List[Dict]) -> str:
    lines: List[str] = []
    for ex in examples:
        txt = ex.get("text", "").strip()
        if not txt:
            continue
        lines.append(f"- {txt}")
    return "\n".join(lines)


def _build_prompt(idea: str, examples: List[Dict], mode: str = "ops") -> List[Dict[str, str]]:
    """
    Construye mensajes para un endpoint tipo /chat/completions.
    Pedimos 3 variantes separadas de forma parseable.
    """

    examples_block = _build_examples_block(examples)

    system_content = (
        STYLE_CONTRACT_OPS
        if mode == "ops"
        else STYLE_CONTRACT_CHAOS
        if mode == "chaos"
        else STYLE_CONTRACT_REPLY
    )

    if mode == "reply":
        reply_examples = """
Original: "I built an AI agent but still can't keep my calendar under control."
Reply: "Oof. Agents won't save you if you still treat every ping like gospel. Kill half the invites and the agent suddenly works."

Original: "Revenue is flat so I'm just going to launch 3 new offers."
Reply: "Yep. When focus hurts you make junk food offers. Sit with the boring one that already sells and tighten ops instead."
""".strip()

        user_content = f"""
Original tweet:
"{idea}"

You will write 3 alternative reply tweets in the style above.

Reply examples (absorb tone, do NOT copy):
{reply_examples}

# Constraints for reply-mode generation
# - Each reply must be concise (1 tweet, max ~220 characters), no list, no thread.
# - Never confront directly; do not start with "no", "actually", or similar.
# - Add signal: either a sharp diagnostic, a grounded example, or a tactical nudge.
# - No generic praise without substance. No emojis or hashtags.
# - Each reply must feel like it was written for a solo founder drowning in their own business.
# - Reference at least one idea/word from the original to prove you're replying to *that* tweet.
# - Show your angle clearly: either a personal pattern you've seen, a constraint others ignore, or a missing consequence. If you can't add that, skip the reply.
# - Before outputting, answer silently those checklist questions; only produce replies that pass them.

Output format (strict):
Return ONLY:
1) A line with 'POST 1: ' followed by the first reply.
2) A line with 'POST 2: ' followed by the second reply.
3) A line with 'POST 3: ' followed by the third reply.
No extra commentary.
""".strip()
    else:
        user_content = f"""
You will write 3 alternative posts for X/Twitter.

Context:
- Audience: experienced operators, founders, and solo builders who care about systems, focus, and execution. They hate fluff, cliches, and generic "hustle" talk.
- Maintain the same voice and stance as the examples below, but DO NOT copy their structure or phrases.
- Assume the reader is a solo operator in Year 0–1, overloaded, doing every function themselves, drowning in their own business.

Style examples (DO NOT imitate verbatim, ONLY absorb tone/attitude/precision):
{examples_block}

Task:
- Take this idea and turn it into 3 distinct posts:
  IDEA: "{idea}"

Constraints:
- Each post must be self-contained.
- Each post must express exactly one clear idea.
- Length: about 40 to 220 characters each.
- No emojis. No hashtags. No exclamation spam.
- No meta comments about writing, prompts, AI, or being a bot.
- Vary the rhythm and structure between the 3 posts.
- If you detect yourself repeating the same opening/structure, change it.

Output format (strict):
Return ONLY:
1) A line with 'POST 1: ' followed by the first post.
2) A line with 'POST 2: ' followed by the second post.
3) A line with 'POST 3: ' followed by the third post.
No extra commentary.
""".strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _call_llm(messages: List[Dict[str, str]], model_name: str, mode: str) -> str:
    api_key = get_openrouter_api_key()
    endpoint = get_openrouter_chat_endpoint()

    if not api_key or not model_name:
        raise GenerationError("Missing OpenRouter API key or model_name.")

    temperature = 0.7
    if mode == "chaos":
        temperature = 0.9

    stop_sequences = ["```"]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.0,
        "n": 3,
    }
    if stop_sequences:
        payload["stop"] = stop_sequences

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = httpx.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=40,
            )
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
            break
        except httpx.HTTPStatusError as err:
            last_error = err
            status = err.response.status_code if err.response else None
            if status == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise GenerationError(f"HTTP error from provider: {err}") from err
        except Exception as err:
            last_error = err
            if attempt < 2:
                time.sleep(1 + attempt)
                continue
            raise GenerationError(f"Error calling generation model: {err}") from err
    else:
        raise GenerationError(f"Error calling generation model: {last_error}")

    choices = data.get("choices", [])
    segments: List[str] = []
    for choice in choices:
        content_obj: Any = ""
        message = choice.get("message")
        if isinstance(message, dict):
            content_obj = message.get("content", "")
        elif "text" in choice:
            content_obj = choice.get("text", "")

        if isinstance(content_obj, list):
            # Algunas APIs entregan fragmentos estructurados.
            content = "".join(
                part.get("text", "")
                for part in content_obj
                if isinstance(part, dict)
            )
        else:
            content = str(content_obj or "")

        content = content.strip()
        if content:
            segments.append(content)

    content = "\n\n".join(segments).strip()
    if not content:
        fallback_fields = [
            "output",
            "content",
            "text",
        ]
        for field in fallback_fields:
            raw_field = data.get(field)
            if not raw_field:
                continue
            if isinstance(raw_field, list):
                raw_field = " ".join(
                    str(item)
                    for item in raw_field
                    if isinstance(item, (str, bytes))
                )
            content = str(raw_field).strip()
            if content:
                break

    if not content:
        raise GenerationError(f"Empty response from LLM. Payload: {data}")
    return content


def _parse_variants(raw: str) -> List[str]:
    """
    Parsea el formato:
    POST 1: ...
    POST 2: ...
    POST 3: ...
    """

    variants: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("post 1:"):
            variants.append(line.split(":", 1)[1].strip())
        elif line.lower().startswith("post 2:"):
            variants.append(line.split(":", 1)[1].strip())
        elif line.lower().startswith("post 3:"):
            variants.append(line.split(":", 1)[1].strip())
    return [v for v in variants if v]


class Generator:
    """
    Alto nivel: dado una idea, selecciona ejemplos de estilo y
    pide al modelo 3 variantes en la voz correcta.
    """

    def __init__(self, model_name: str | None = None, mode: str = "ops") -> None:
        self._style = StyleRetriever()
        self._model_name = model_name or get_generation_model_primary()
        self._mode = mode
        if not self._model_name:
            raise GenerationError("No generation model configured for Generator.")

    def generate_variants(
        self,
        idea: str,
        tone: str | None = None,
        formato: str = "tweet",
        n_expected: int = 3,
    ) -> List[str]:
        if not idea or not idea.strip():
            raise GenerationError("Idea cannot be empty.")

        examples = self._style.select_examples(
            idea=idea,
            k=12,
            tone=tone,
            formato=formato,
            style=(
                "ops"
                if self._mode == "ops"
                else "chaos"
                if self._mode == "chaos"
                else "reply"
            ),
        )

        messages = _build_prompt(idea=idea, examples=examples, mode=self._mode)
        raw = _call_llm(messages, model_name=self._model_name, mode=self._mode)
        variants = _parse_variants(raw)

        mode_limits = {
            "ops": (40, 220),
            "chaos": (40, 240),
            "reply": (20, 180),
        }
        min_len, max_len = mode_limits.get(self._mode, (40, 220))

        cleaned: List[str] = []
        seen: set[str] = set()
        for v in variants:
            v = v.strip()
            length = len(v)
            if not (min_len <= length <= max_len):
                continue
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(v)

        return cleaned[:n_expected]
