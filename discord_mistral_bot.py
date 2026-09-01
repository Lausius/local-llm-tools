#!/usr/bin/env python3
"""
discord_mistral_bot.py

A simple Discord bot that forwards any message it sees in a channel to a
local Ollama model (default: mistral) and replies with the model's response.

Includes two layers of memory:
  1. Rolling chat history per channel (chat_memory.json) -- short-term,
     last few exchanges, gives conversational continuity.
  2. Long-term facts (remembered_facts.json) -- durable facts extracted
     from messages (preferences, setup details, ongoing projects), injected
     into every prompt regardless of channel.

Commands (type these as a message in the channel):
    !forget             Clear rolling chat history for this channel
    !facts              List all remembered long-term facts (numbered)
    !forget-fact <n>    Remove a single fact by its number
    !forget-facts       Clear all long-term facts
    !schedule <text>    Manage stock digest schedules in plain English, e.g.:
                          !schedule list
                          !schedule add AAPL ORSTED.CO at 08:00
                          !schedule remove abc12345
                          !schedule edit abc12345 to TSLA MSFT
                        Only 4 fixed, validated actions exist (see
                        schedule_manager.py) -- the model can never run
                        arbitrary commands, only request one of these.
    !code <question>    Ask about the bot's own implementation. Uses
                        keyword search over the actual source files
                        (code_search.py) so answers are grounded in real
                        code, not guessed.
    !time               Show the container's current date/time/timezone --
                        the ground truth used for all !schedule timing.

A short self-description (SELF_KNOWLEDGE.md) is injected into every prompt
so the model gives accurate answers about what it is/can do, rather than
inventing capabilities it doesn't have.

Scheduling now runs in-process via APScheduler (schedule_manager.py), with
state saved to schedules.json -- no host crontab needed, so this works
identically on bare metal or inside Docker.

Setup:
    pip install discord.py python-dotenv requests apscheduler yfinance

    Create a .env file in the same folder with:
        DISCORD_BOT_TOKEN=your_token_here
        STOCK_WEBHOOK_URL=your_webhook_url_here
        OLLAMA_URL=http://localhost:11434/api/generate   (or http://ollama:11434/api/generate in Docker)

    Make sure Ollama is running (ollama serve) and the model is pulled
    (ollama pull mistral).

Run:
    python3 discord_mistral_bot.py
"""

import os
import re
import json
import asyncio
import requests
import discord
from datetime import datetime
from dotenv import load_dotenv

import schedule_manager as sched
import stock_digest
import code_search

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MAX_DISCORD_LEN = 2000  # Discord's message length limit

# Self-knowledge: a short, accurate description of what this bot is and can
# do, loaded once and injected into every prompt (alongside remembered facts)
# so the model answers questions about ITSELF from real documentation rather
# than guessing. See SELF_KNOWLEDGE.md.
SELF_KNOWLEDGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SELF_KNOWLEDGE.md")


def load_self_knowledge() -> str:
    if not os.path.exists(SELF_KNOWLEDGE_FILE):
        return ""
    with open(SELF_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


SELF_KNOWLEDGE = load_self_knowledge()

# Persistent memory: one JSON file, in a data directory that's kept separate
# from the code directory. Locally this defaults to the script's own folder;
# in Docker it's set to a mounted volume path (see docker-compose.yml) so
# data survives container rebuilds without being shadowed by the code volume.
DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(DATA_DIR, "chat_memory.json")
DEBUG_LOG_FILE = os.path.join(DATA_DIR, "bot_debug.log")
HISTORY_LIMIT = 6  # number of past exchanges (user+assistant pairs) to keep per channel


def log_debug_event(event_type: str, detail: str):
    """Append structured debug events for prompt tuning and regression checks."""
    timestamp = datetime.now().astimezone().isoformat()
    line = f"[{timestamp}] {event_type}: {detail}\n"
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def load_memory() -> dict:
    """Load all channel histories from disk. Returns {} if file doesn't exist yet."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # If the file is corrupted/unreadable, start fresh rather than crash
        print(f"Warning: could not read {MEMORY_FILE}, starting with empty memory.")
        return {}


def save_memory(memory: dict):
    """Write all channel histories to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


channel_history = load_memory()  # loaded once at startup, kept in memory + written after each turn

# Long-term "facts" memory: durable statements extracted from conversation,
# independent of the rolling chat history above. Stored globally (not per
# channel) since facts about you are true regardless of which channel you're in.
FACTS_FILE = os.path.join(DATA_DIR, "remembered_facts.json")
MAX_FACTS = 200  # simple cap so the file/prompt doesn't grow unbounded


def load_facts() -> list:
    """Load the list of remembered facts from disk. Returns [] if none yet."""
    if not os.path.exists(FACTS_FILE):
        return []
    try:
        with open(FACTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read {FACTS_FILE}, starting with empty facts.")
        return []


def save_facts(facts: list):
    """Write the facts list to disk."""
    with open(FACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


remembered_facts = load_facts()


def normalize_fact_text(fact: str) -> str:
    """Normalize durable-fact wording to reduce awkward or reversed facts.
    This is generic: it handles user-naming preferences as a fact, but does not
    hardcode any specific name.
    """
    text = fact.strip().strip('"\'')
    if not text:
        return ""

    patterns = [
        r'^user prefers to be called\s+["\']?(.+?)["\']?\.?$',
        r'^the user prefers to be called\s+["\']?(.+?)["\']?\.?$',
        r'^call me\s+["\']?(.+?)["\']?\.?$',
        r'^can i call you\s+["\']?(.+?)["\']?\.?$',
        r'^can i call me\s+["\']?(.+?)["\']?\.?$',
        r'^i want to call you\s+["\']?(.+?)["\']?\.?$',
        r'^i want to name you\s+["\']?(.+?)["\']?\.?$',
        r'^please call me\s+["\']?(.+?)["\']?\.?$',
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip().strip('"\'')
            name = re.sub(r'[?!.]+$', '', name)
            return f'The user prefers to call the assistant "{name}".'

    return text


def get_preferred_assistant_name() -> str | None:
    """Return the preferred assistant name from stored facts, if any.
    If no preference is stored, fall back to the actual Discord bot name.
    """
    for fact in remembered_facts:
        match = re.search(r'The user prefers to call the assistant "(.+?)"\.?$', fact, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r'The user prefers to call the assistant\s+(.+?)\.?$', fact, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'')
    if 'client' in globals() and getattr(client, 'user', None) is not None:
        return client.user.name
    return None


def extract_fact(user_message: str, model: str) -> str | None:
    """Ask the model whether this message contains a durable fact worth
    remembering long-term (e.g. preferences, ongoing projects, personal
    details). Returns the fact as a short string, or None if there isn't one.
    """
    prompt = f"""Does the following message contain a durable fact about the user worth
remembering for future conversations? Examples of durable facts: their name,
preferences, ongoing projects, their hardware/setup, recurring context,
important dates (birthdays, anniversaries), or explicit requests to remember
something (e.g. "remember this", "husk dette").

Respond in the SAME language as the message.
Respond with ONLY ONE of these two things, nothing else:
  - If there IS a durable fact: the fact itself, as one short third-person sentence.
  - If there is NOT a durable fact: the single word NONE.
Never explain your reasoning. Never say things like "not specified" or
"not mentioned" -- just respond NONE in that case.

Message: "{user_message}"

Fact:"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()["response"].strip()
    cleaned = normalize_fact_text(result)

    if not cleaned:
        return None

    # Catch both the literal NONE sentinel and near-miss phrasings where the
    # model explains there's no fact instead of just saying NONE.
    lowered = cleaned.lower()
    no_fact_markers = [
        "none",
        "not specified",
        "not mentioned",
        "no fact",
        "not provided",
        "not stated",
        "no durable fact",
        "not given",
    ]
    if any(marker in lowered for marker in no_fact_markers):
        return None

    return cleaned


def remember_fact_if_any(user_message: str, model: str):
    """Extract a fact from the message (if any) and store it, avoiding exact duplicates."""
    fact = extract_fact(user_message, model)
    if fact and fact not in remembered_facts:
        remembered_facts.append(fact)
        del remembered_facts[:-MAX_FACTS]  # keep only the most recent MAX_FACTS
        save_facts(remembered_facts)


def should_inject_self_knowledge(user_message: str) -> bool:
    """Only include the bot's self-knowledge when the user explicitly asks
    about the bot's identity, commands, or how it works. This keeps ordinary
    chat natural and avoids repetitive disclaimers.
    """
    lowered = user_message.lower()
    triggers = [
        "what can you do",
        "what are you",
        "who are you",
        "how do you work",
        "what commands",
        "what can i ask you",
        "how do i use",
        "what is your purpose",
        "can you explain yourself",
        "!schedule",
        "!facts",
        "!forget",
        "!code",
    ]
    return any(trigger in lowered for trigger in triggers)


def build_system_note(user_message: str) -> str:
    """Build a compact system prompt with strong tone rules and only optional
    self-knowledge when relevant.
    """
    facts_block = ""
    if remembered_facts:
        facts_block = "Known facts about the user:\n" + "\n".join(f"- {f}" for f in remembered_facts) + "\n\n"

    self_block = f"{SELF_KNOWLEDGE}\n\n" if SELF_KNOWLEDGE and should_inject_self_knowledge(user_message) else ""

    now_aware = datetime.now().astimezone()
    time_block = (
        f"Current date/time (REAL, accurate -- state it directly and "
        f"confidently if asked, do not caveat or call it a guess): "
        f"{now_aware.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)')}\n\n"
    )

    return (
        "You are a helpful assistant chatting in a Discord channel. "
        "Keep replies SHORT -- 1 to 3 sentences for most messages. Only "
        "write more than that if the user explicitly asks for detail, a "
        "list, or an explanation. For casual messages and small talk, just "
        "chat naturally. Do not mention that you are a chatbot, that you don't "
        "have personal experiences, or otherwise disclaim your identity in "
        "ordinary conversation unless the user explicitly asks about it. "
        "Do not mention stock-digest commands or examples in casual chat; only "
        "mention !schedule or other commands if the user explicitly asks about "
        "them. If the user asks about your capabilities or commands, answer "
        "directly and accurately using the provided self-knowledge. Follow any "
        "user preferences in the 'Known facts about the user' section exactly. "
        "If a fact says they prefer not to include a phrase, style, or closing "
        "line, do not include it unless the user explicitly asks for it. Never "
        "append generic help-offers such as 'If you need help' or 'Let me know "
        "if you need help' unless the user explicitly asks for that wording.\n\n"
        + self_block
        + time_block
        + facts_block
    )


def ask_mistral(channel_id: int, user_message: str) -> str:
    """Send the message (plus recent channel history and remembered facts) to the local model."""
    channel_key = str(channel_id)  # JSON keys must be strings
    history = channel_history.get(channel_key, [])

    convo = ""
    for role, text in history:
        convo += f"{role}: {text}\n"
    convo += f"User: {user_message}\nAssistant:"

    system_note = build_system_note(user_message)

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": system_note + convo, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    reply = response.json()["response"].strip()
    reply = apply_user_preferences(reply, user_message)

    bad_disclaimer_markers = [
        "i'm just a chatbot",
        "i am just a chatbot",
        "i don't have personal experiences",
        "i don't have personal experience",
        "i am an ai",
        "i am a language model",
    ]
    if any(marker in reply.lower() for marker in bad_disclaimer_markers):
        log_debug_event("bot_disclaimer_response", f"User: {user_message}\nAssistant: {reply}")

    history.append(("User", user_message))
    history.append(("Assistant", reply))
    channel_history[channel_key] = history[-HISTORY_LIMIT * 2:]
    save_memory(channel_history)

    return reply


def split_for_discord(text: str, limit: int = MAX_DISCORD_LEN):
    """Split long replies into chunks under Discord's message length limit."""
    chunks = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:]
    chunks.append(text)
    return chunks


def apply_user_preferences(reply: str, user_message: str | None = None) -> str:
    """Hard-enforce user preferences and remove common unwanted boilerplate from
    outgoing replies, while preserving normal casual conversation.
    """
    cleaned = reply.strip()

    fact_phrases = []
    for fact in remembered_facts:
        fact_lower = fact.lower()
        if "prefer" in fact_lower and ("include" in fact_lower or "not" in fact_lower):
            for quoted in re.findall(r'["\']([^"\']+)["\']', fact):
                fact_phrases.append(quoted)

    for phrase in fact_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

    for pattern in [
        r"(?is)\s*I['’]m just a chatbot.*?(?:[.!?]|$)",
        r"(?is)\s*I am just a chatbot.*?(?:[.!?]|$)",
        r"(?is)\s*I don't have personal experiences.*?(?:[.!?]|$)",
        r"(?is)\s*I don't have personal experience.*?(?:[.!?]|$)",
        r"(?is)\s*I am an AI.*?(?:[.!?]|$)",
        r"(?is)\s*I am a language model.*?(?:[.!?]|$)",
        r"(?is)\s*if you need help(?: with .*?)?\s*[,;.!]?",
        r"(?is)\s*if you have any other questions or need assistance\s*[,;.!]?",
        r"(?is)\s*feel free to ask(?: if you have any questions)?\s*[,;.!]?",
        r"(?is)\s*let me know if you need help\s*[,;.!]?",
        r"(?is)\s*let me know if you have any other questions\s*[,;.!]?",
        r"(?is)\s*if you need any help\s*[,;.!]?",
        r"(?is)\s*with scheduling stock digests.*?(?:[.!?]|$)",
        r"(?is)\s*for example,.*?will add a new digest for the .*? at a daily interval.*?(?:[.!?]|$)",
        r"(?is)\s*you can also.*?scheduled digests.*?(?:[.!?]|$)",
        r"(?is)\s*you can use the !schedule command.*?(?:[.!?]|$)",
        r"(?is)\s*!schedule\s*(?:command|actions|for example|example)?(?:[.!?]|$)",
    ]:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" \n\t,;:.-")
    if not cleaned:
        cleaned = "Hey there! How can I help?"
    return cleaned

def parse_schedule_intent(instruction: str, model: str) -> dict:
    """Ask the model to translate a natural-language schedule instruction into
    ONE of a fixed set of JSON actions. This is the only interface between the
    model and schedule_manager -- it never gets to run commands directly.
    """
    if not instruction or instruction.lower() == "list":
        return {"action": "list"}

    prompt = f"""Translate the following instruction into EXACTLY ONE JSON object,
and respond with ONLY that JSON object, nothing else -- no explanation, no markdown.

Valid formats (pick the one that matches the instruction):
  {{"action": "list"}}
  {{"action": "add", "tickers": ["AAPL", "ORSTED.CO"], "time": "08:00"}}
  {{"action": "remove", "id": "abc12345"}}
  {{"action": "edit", "id": "abc12345", "tickers": ["AAPL"]}}
  {{"action": "unknown"}}

Rules:
- "time" must be 24-hour HH:MM format.
- "tickers" must be an array of ticker symbol strings, uppercase.
- "id" is an 8-character code the user must have provided (from a prior list).
- If the instruction doesn't clearly match one of these, respond with {{"action": "unknown"}}.

Instruction: "{instruction}"

JSON:"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # Models sometimes wrap JSON in markdown fences -- strip those if present
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"action": "unknown"}

    if not isinstance(parsed, dict) or "action" not in parsed:
        return {"action": "unknown"}

    return parsed


def format_schedule_list(schedules: list) -> str:
    if not schedules:
        return "No schedules set up yet. Try: `!schedule add AAPL ORSTED.CO at 08:00`"
    lines = [f"- `{s['id']}` — {', '.join(s['tickers'])} at {s['time']}" for s in schedules]
    return "Current schedules:\n" + "\n".join(lines)


async def resolve_tickers_for_schedule(raw_tickers: list) -> tuple:
    """Resolve a list of user-provided tickers/company names into real,
    working Alpha Vantage symbols, using stock_digest.resolve_ticker for
    each. Runs in an executor since it makes blocking network calls.
    Returns (resolved_symbols, notes) where notes describes any name ->
    symbol resolutions that happened, for user-facing confirmation.
    """
    loop = asyncio.get_running_loop()
    resolved = []
    notes = []

    for raw in raw_tickers:
        symbol, _data, note = await loop.run_in_executor(None, stock_digest.resolve_ticker, raw, MODEL)
        resolved.append(symbol)
        if note:
            notes.append(f"Resolved {note}")

    return resolved, notes


async def handle_schedule_command(channel, instruction: str):
    """Parse a natural-language schedule instruction and execute it via the
    narrow, validated functions in schedule_manager.py. Every branch here
    calls one specific allowlisted function -- there is no generic execution path.
    """
    async with channel.typing():
        try:
            intent = parse_schedule_intent(instruction, MODEL)
        except requests.exceptions.ConnectionError:
            await channel.send("Couldn't reach Ollama on this machine. Is it running? (`ollama serve`)")
            return
        except Exception as e:
            await channel.send(f"Error understanding that instruction: {e}")
            return

    action = intent.get("action")

    try:
        if action == "list":
            schedules = sched.list_schedules()
            await channel.send(format_schedule_list(schedules))

        elif action == "add":
            resolved_tickers, notes = await resolve_tickers_for_schedule(intent.get("tickers", []))
            result = sched.add_schedule(resolved_tickers, intent.get("time", ""))
            note_text = ("\n" + "\n".join(notes)) if notes else ""
            await channel.send(
                f"Added schedule `{result['id']}`: {', '.join(result['tickers'])} at {result['time']} daily."
                f"{note_text}"
            )

        elif action == "remove":
            job_id = intent.get("id", "")
            removed = sched.remove_schedule(job_id)
            if removed:
                await channel.send(f"Removed schedule `{job_id}`.")
            else:
                await channel.send(f"No schedule found with id `{job_id}`. Try `!schedule list` to see valid ids.")

        elif action == "edit":
            resolved_tickers, notes = await resolve_tickers_for_schedule(intent.get("tickers", []))
            result = sched.edit_tickers(intent.get("id", ""), resolved_tickers)
            note_text = ("\n" + "\n".join(notes)) if notes else ""
            await channel.send(
                f"Updated schedule `{result['id']}`: now {', '.join(result['tickers'])} at {result['time']} daily."
                f"{note_text}"
            )

        else:
            await channel.send(
                "I couldn't understand that as a schedule instruction. Try things like:\n"
                "`!schedule list`\n"
                "`!schedule add ORSTED at 08:00` (company names work too)\n"
                "`!schedule remove abc12345`\n"
                "`!schedule edit abc12345 to TSLA MSFT`"
            )

    except sched.ScheduleError as e:
        await channel.send(f"Couldn't do that: {e}")
    except Exception as e:
        await channel.send(f"Unexpected error managing schedules: {e}")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


async def run_scheduled_digest(tickers: list):
    """Wraps the (blocking) stock_digest.run_digest so it can be awaited by
    the scheduler without blocking the bot's event loop. Posts a failure
    notice to Discord (via the same webhook) if the digest fails, so
    problems are visible instead of only showing up in container logs.
    """
    loop = asyncio.get_running_loop()
    try:
        status = await loop.run_in_executor(None, stock_digest.run_digest, tickers, MODEL)
        print(f"[scheduled digest] {status}")
    except Exception as e:
        error_msg = f"⚠️ Scheduled digest for {', '.join(tickers)} failed: {e}"
        print(f"[scheduled digest] {error_msg}")
        try:
            await loop.run_in_executor(
                None, stock_digest.post_to_discord, stock_digest.WEBHOOK_URL, error_msg
            )
        except Exception:
            pass  # if even posting the error fails, at least it's in the logs above


@client.event
async def on_ready():
    print(f"Logged in as {client.user}. Listening for messages...")
    sched.init_scheduler(run_scheduled_digest)
    print("Schedule manager initialized; restored any saved schedules.")


@client.event
async def on_message(message: discord.Message):
    # Ignore the bot's own messages to avoid loops
    if message.author == client.user:
        return

    # Ignore other bots
    if message.author.bot:
        return

    user_text = message.content.strip()
    if not user_text:
        return

    lower_text = user_text.lower()

    if "how are you" in lower_text:
        await message.channel.send("I’m doing well, thanks — how are you?")
        return

    if any(greeting in lower_text for greeting in ["hi ", "hey", "hello"]):
        await message.channel.send("Hey! How can I help?")
        return

    if "what is your name" in lower_text or "what should i call you" in lower_text or "what should i call it" in lower_text:
        preferred_name = get_preferred_assistant_name()
        if preferred_name:
            await message.channel.send(f"You can call me {preferred_name}.")
        else:
            default_name = os.getenv("BOT_NAME") or (client.user.name if client.user else "Bot")
            await message.channel.send(f"You can call me {default_name}.")
        return

    if "what can you do" in lower_text:
        await message.channel.send(
            "I can chat naturally, remember facts from our conversations, and help with stock-digest scheduling if you ask. I can also answer questions about the bot itself with `!code`."
        )
        return

    # Simple command to wipe this channel's rolling chat history
    if lower_text == "!forget":
        channel_key = str(message.channel.id)
        channel_history.pop(channel_key, None)
        save_memory(channel_history)
        await message.channel.send("Chat history cleared for this channel.")
        return

    # Show what long-term facts are currently remembered
    if lower_text == "!facts":
        if not remembered_facts:
            await message.channel.send("I don't have any long-term facts stored yet.")
        else:
            listing = "\n".join(f"{i+1}. {f}" for i, f in enumerate(remembered_facts))
            for chunk in split_for_discord(
                "Here's what I remember long-term:\n" + listing +
                "\n\nUse `!forget-fact <number>` to remove one."
            ):
                await message.channel.send(chunk)
        return

    # Remove a single fact by its number (as shown in !facts)
    if user_text.lower().startswith("!forget-fact "):
        try:
            index = int(user_text.split(maxsplit=1)[1]) - 1
            removed = remembered_facts.pop(index)
            save_facts(remembered_facts)
            await message.channel.send(f"Removed: {removed}")
        except (ValueError, IndexError):
            await message.channel.send("Usage: `!forget-fact <number>` -- check `!facts` for valid numbers.")
        return

    # Wipe all long-term facts
    if user_text.lower() == "!forget-facts":
        remembered_facts.clear()
        save_facts(remembered_facts)
        await message.channel.send("All long-term facts cleared.")
        return

    # Show the container's own current date/time/timezone -- the ground
    # truth used for all !schedule timing, so you can verify it matches
    # your actual local time rather than assuming.
    if user_text.lower() == "!time":
        now_aware = datetime.now().astimezone()  # attaches the system's local tzinfo
        tz_env = os.getenv("TZ", "not set (likely UTC)")
        await message.channel.send(
            f"Container time: **{now_aware.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)')}**\n"
            f"TZ environment variable: `{tz_env}`\n"
            f"(This is the time scheduled digests are based on -- compare it to your actual local time.)"
        )
        return

    # Schedule management -- explicit prefix only, never triggered by normal chat.
    # The model is only ever allowed to choose one of a fixed set of validated
    # actions (see schedule_manager.py) -- it never gets shell or file access.
    if user_text.lower().startswith("!schedule"):
        instruction = user_text[len("!schedule"):].strip()
        await handle_schedule_command(message.channel, instruction)
        return

    # Answer questions about the bot's own implementation, grounded in the
    # actual source files (simple keyword search, no code execution involved).
    if user_text.lower().startswith("!code"):
        question = user_text[len("!code"):].strip()
        if not question:
            await message.channel.send("Usage: `!code <question>`, e.g. `!code how does fact extraction work`")
            return
        async with message.channel.typing():
            try:
                matches = code_search.search_codebase(question)
                context = code_search.format_snippets_for_prompt(matches)
                prompt = f"""You are answering a question about your own source code, using ONLY
the code snippets below as ground truth. If the snippets don't actually
answer the question, say so honestly rather than guessing.

{context}

Question: {question}

Answer concisely, referencing the relevant file(s) by name."""
                response = requests.post(
                    OLLAMA_URL,
                    json={"model": MODEL, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                response.raise_for_status()
                answer = response.json()["response"].strip()
            except requests.exceptions.ConnectionError:
                await message.channel.send("Couldn't reach Ollama on this machine. Is it running?")
                return
            except Exception as e:
                await message.channel.send(f"Error answering that: {e}")
                return
        for chunk in split_for_discord(answer):
            await message.channel.send(chunk)
        return

    async with message.channel.typing():
        try:
            reply = ask_mistral(message.channel.id, user_text)
            remember_fact_if_any(user_text, MODEL)  # check if this message is worth remembering long-term
        except requests.exceptions.ConnectionError:
            await message.channel.send(
                "Couldn't reach Ollama on this machine. Is it running? (`ollama serve`)"
            )
            return
        except Exception as e:
            await message.channel.send(f"Error talking to the model: {e}")
            return

    for chunk in split_for_discord(reply):
        await message.channel.send(chunk)


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit(
            "DISCORD_BOT_TOKEN not found. Create a .env file with:\n"
            "DISCORD_BOT_TOKEN=your_token_here"
        )
    client.run(TOKEN)