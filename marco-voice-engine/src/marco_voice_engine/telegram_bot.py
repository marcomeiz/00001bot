"""Private Telegram bot for OPS / CHAOS idea generation."""

from __future__ import annotations

import html
import json
import logging
import os
import random
import re
from typing import List, Set

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import get_topics_filename, get_voice_data_dir
from .pipeline import VoiceEngine


load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_IDS_ENV = os.getenv("ALLOWED_USER_IDS", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")
if not ALLOWED_IDS_ENV:
    raise RuntimeError("ALLOWED_USER_IDS not set; bot must be whitelisted")

ALLOWED_IDS = {
    int(item.strip())
    for item in ALLOWED_IDS_ENV.split(",")
    if item.strip().isdigit()
}
if not ALLOWED_IDS:
    raise RuntimeError("ALLOWED_USER_IDS has no valid numeric entries")


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("marco_voice_engine.telegram_bot")


def load_topics() -> List[str]:
    base_dir = get_voice_data_dir()
    topics_path = base_dir / get_topics_filename()
    if not topics_path.exists():
        raise FileNotFoundError(f"topics.json not found at {topics_path}")

    with topics_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    topics: List[str] = []

    def _walk(payload: object) -> None:
        if isinstance(payload, str):
            val = payload.strip()
            if val:
                topics.append(val)
            return

        if isinstance(payload, dict):
            candidate = (
                payload.get("idea")
                or payload.get("topic")
                or payload.get("text")
                or payload.get("theme")
            )
            if isinstance(candidate, str):
                val = candidate.strip()
                if val:
                    topics.append(val)

            for key in ("themes", "topics", "items", "categories"):
                nested = payload.get(key)
                if nested:
                    _walk(nested)
            return

        if isinstance(payload, list):
            for item in payload:
                _walk(item)

    _walk(data)

    clean_topics = [t for t in topics if t]
    if not clean_topics:
        raise ValueError("No usable topics found in topics.json")
    return clean_topics


TOKEN_PATTERN = re.compile(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{4,}")


def _signature(text: str) -> Set[str]:
    return set(token.lower() for token in TOKEN_PATTERN.findall(text))


TOPICS = load_topics()
TOPIC_SIGNATURES = {topic: _signature(topic) for topic in TOPICS}
KEYBOARD = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("⚙️ OPS", callback_data="mode_ops_random"),
            InlineKeyboardButton("🧪 CHAOS", callback_data="mode_chaos_random"),
        ]
    ]
)


def is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_IDS


async def _reply_not_authorized(update: Update) -> None:
    target = update.message or update.callback_query
    if not target:
        return
    if update.message:
        await update.message.reply_text("No autorizado.")
    elif update.callback_query:
        await update.callback_query.answer("No autorizado", show_alert=True)


def _topic_distance(sig_a: Set[str], sig_b: Set[str]) -> float:
    if not sig_a and not sig_b:
        return 0.0
    union = sig_a | sig_b
    if not union:
        return 0.0
    intersection = sig_a & sig_b
    return 1.0 - (len(intersection) / len(union))


def pick_topic_far_from(last_topic: str | None) -> str:
    if not TOPICS:
        raise ValueError("No topics configured.")

    if not last_topic:
        return random.choice(TOPICS)

    reference_sig = _signature(last_topic)
    scored = []
    for topic, sig in TOPIC_SIGNATURES.items():
        dist = _topic_distance(reference_sig, sig)
        if topic.strip().lower() == last_topic.strip().lower():
            dist = -1.0
        scored.append((dist, topic))

    scored.sort(reverse=True, key=lambda item: item[0])
    best_distance = scored[0][0]
    threshold = max(-1.0, best_distance - 0.15)
    candidates = [topic for dist, topic in scored if dist >= threshold and dist >= 0.0]
    if not candidates:
        candidates = [topic for _, topic in scored if topic.strip().lower() != last_topic.strip().lower()]
    if not candidates:
        candidates = TOPICS
    return random.choice(candidates)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        log.warning("Unauthorized /start from %s (%s)", user.id, user.username)
        await _reply_not_authorized(update)
        return

    await update.message.reply_text(
        "Pulsa un botón y te devuelvo tiros con un topic aleatorio:",
        reply_markup=KEYBOARD,
    )


async def topic_count(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        log.warning("Unauthorized /t from %s (%s)", user.id, user.username)
        await _reply_not_authorized(update)
        return

    total = len(TOPICS)
    await update.message.reply_text(f"Hay {total} topics disponibles en topics.json.")


async def handle_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        log.warning("Unauthorized message from %s (%s)", user.id, user.username)
        await _reply_not_authorized(update)
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    context.user_data["last_original"] = text
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💬 REPLY", callback_data="reply_to_last")],
            [InlineKeyboardButton("⚙️ OPS", callback_data="mode_ops_random")],
            [InlineKeyboardButton("🧪 CHAOS", callback_data="mode_chaos_random")],
        ]
    )

    await update.message.reply_text(
        "Contexto guardado. ¿Qué quieres hacer?",
        reply_markup=keyboard,
    )


def _format_variants_text(mode: str, topic: str, accepted: List[dict]) -> str:
    lines = [
        f"MODE: {mode.upper()}",
        f"Topic: “{html.escape(topic)}”",
        "",
        "Opciones:",
    ]
    for idx, item in enumerate(accepted, start=1):
        txt = html.escape(item["text"].strip())
        sim = item["scores"].get("similarity", 0.0)
        lines.append(f"#{idx} (sim={sim:.3f}):")
        lines.append("<pre>")
        lines.append(txt)
        lines.append("</pre>")
        lines.append("")
    return "\n".join(lines).strip()


async def _generate_topic_response(query, context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    if not TOPICS:
        await query.edit_message_text("No hay topics configurados.")
        return

    last_topic = context.user_data.get("last_topic_random")
    topic = pick_topic_far_from(last_topic)
    context.user_data["last_topic_random"] = topic
    engine = VoiceEngine(mode=mode)
    result = engine.propose(topic, retry_on_empty=True)
    accepted = result.get("accepted", [])[:3]

    if not accepted:
        await query.edit_message_text(
            f"MODE: {mode.upper()}\nTopic: “{topic}”\n\nNada pasó el filtro."
        )
        return

    text = _format_variants_text(mode, topic, accepted)
    await query.edit_message_text(
        text,
        reply_markup=KEYBOARD,
        parse_mode=ParseMode.HTML,
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user = query.from_user
    if not is_allowed(user.id):
        log.warning("Unauthorized callback from %s (%s)", user.id, user.username)
        await query.answer("No autorizado", show_alert=True)
        return

    data = query.data or ""
    if data == "mode_ops_random":
        await _generate_topic_response(query, context, "ops")
        return
    if data == "mode_chaos_random":
        await _generate_topic_response(query, context, "chaos")
        return
    if data == "reply_to_last":
        original = context.user_data.get("last_original")
        if not original:
            await query.edit_message_text(
                "No tengo contexto guardado. Pega el tweet primero.",
                reply_markup=KEYBOARD,
            )
            return

        ve = VoiceEngine()
        result = ve.reply(original)
        accepted = result.get("accepted", [])

        if not accepted:
            await query.edit_message_text(
                f"Original:\n“{html.escape(original)}”\n\nSin reply válido.",
                reply_markup=KEYBOARD,
                parse_mode=ParseMode.HTML,
            )
            return

        lines = ["Replies:", f"Original: “{html.escape(original)}”", ""]
        for idx, item in enumerate(accepted[:3], start=1):
            txt = html.escape(item["text"].strip())
            sim = item["scores"].get("similarity", 0.0)
            lines.append(f"#{idx} (sim={sim:.3f}):")
            lines.append("<pre>")
            lines.append(txt)
            lines.append("</pre>")
            lines.append("")

        await query.edit_message_text(
            "\n".join(lines).strip(),
            reply_markup=KEYBOARD,
            parse_mode=ParseMode.HTML,
        )
        return

    await query.answer("Acción no reconocida", show_alert=True)


def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("t", topic_count))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_context))
    app.add_handler(CallbackQueryHandler(handle_callback))

    log.info("Starting Telegram bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
