#!/usr/bin/env python3
"""
stock_digest.py

Fetches live data for one or more stock tickers via Alpha Vantage (an
official, licensed API -- see fetch_stock_data() docstring for why this
replaced yfinance), asks a local Ollama model (default: mistral) to
summarize them, and posts the result to a Discord channel via a webhook.

Setup:
    pip install requests python-dotenv

    Get a free API key at https://www.alphavantage.co/support/#api-key
    (free tier: 25 requests/day, but covers 20+ global exchanges --
    unlike Finnhub/Twelve Data's free tiers, which are US-only)

    Create/update .env in the same folder with:
        STOCK_WEBHOOK_URL=your_webhook_url_here
        ALPHA_VANTAGE_API_KEY=your_key_here

    Make sure Ollama is running locally (ollama serve) and the model is
    pulled (ollama pull mistral).

Run manually:
    python3 stock_digest.py ORSTED.CPH AAPL TSLA

Note: non-US tickers may need a different exchange suffix than you're used
to from Yahoo Finance -- e.g. Copenhagen-listed stocks may need ".CPH"
rather than ".CO". Test a new ticker manually before scheduling it.

Note: recurring scheduling is now handled by the Discord bot itself
(schedule_manager.py, via the !schedule command) rather than host cron --
this keeps things working identically whether running on bare metal or
inside Docker. This script can still be run manually anytime, or wrapped
in your own scheduler if you're not using the bot's built-in one.
"""

import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("STOCK_WEBHOOK_URL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
MAX_DISCORD_LEN = 2000
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 5
ALPHA_VANTAGE_RATE_LIMIT_SECONDS = 1.1
_alpha_vantage_lock = threading.Lock()
_alpha_vantage_last_request = 0.0


def _alpha_vantage_get(params: dict, timeout: int = 30):
    """Alpha Vantage enforces a strict 1 request / second limit on free keys.
    Serialize every request and sleep for the remaining interval before each
    outbound call to avoid hitting the rate cap when multiple schedules or
    tickers are processed in quick succession.
    """
    global _alpha_vantage_last_request

    with _alpha_vantage_lock:
        now = time.monotonic()
        elapsed = now - _alpha_vantage_last_request
        if _alpha_vantage_last_request and elapsed < ALPHA_VANTAGE_RATE_LIMIT_SECONDS:
            time.sleep(ALPHA_VANTAGE_RATE_LIMIT_SECONDS - elapsed)
        _alpha_vantage_last_request = time.monotonic()

        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        return response


def fetch_stock_data(ticker: str) -> dict:
    """Pull current price + basic quote data for a ticker via Alpha Vantage's
    GLOBAL_QUOTE endpoint. Switched from yfinance because yfinance scrapes
    Yahoo's unofficial endpoint, which intermittently blocks requests with no
    reliable fix (see git history/commit notes for the failed workarounds
    tried). Alpha Vantage is an official, licensed API -- more limited (only
    1 request per ticker, free tier capped at 25 requests/day, and no
    52-week high/low or P/E ratio on this endpoint) but stable.

    Note: unlike Yahoo's ".CO" suffix, Alpha Vantage may expect a different
    exchange suffix for non-US tickers (e.g. ".CPH" for Copenhagen) --
    verify with a manual test call before relying on this for a specific
    non-US ticker.
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set in environment/.env")

    last_error = None
    delay = RETRY_DELAY_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _alpha_vantage_get(
                {
                    "function": "GLOBAL_QUOTE",
                    "symbol": ticker,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                },
                timeout=30,
            )
            data = response.json()

            # Alpha Vantage returns HTTP 200 even when the daily quota is
            # exceeded or the key/symbol is invalid -- the error shows up
            # inside the JSON body instead of as an HTTP error code.
            if "Note" in data or "Information" in data:
                raise RuntimeError(data.get("Note") or data.get("Information"))

            quote = data.get("Global Quote", {})
            if not quote or "05. price" not in quote:
                raise ValueError(
                    f"Could not find data for ticker '{ticker}'. Check the symbol "
                    f"(non-US tickers may need a different suffix on Alpha Vantage, e.g. '.CPH' not '.CO')."
                )

            return {
                "ticker": ticker.upper(),
                "current_price": float(quote["05. price"]),
                "previous_close": float(quote["08. previous close"]),
                "day_high": float(quote["03. high"]),
                "day_low": float(quote["04. low"]),
                "volume": quote.get("06. volume", "N/A"),
                "change_percent": quote.get("10. change percent", "N/A"),
            }
        except (ValueError, RuntimeError):
            # Invalid symbol or daily quota exceeded -- retrying won't help,
            # don't waste API budget on a guaranteed-to-fail repeat.
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Fetch failed for {ticker} ({e}), retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(delay)
                delay *= 2
                continue
            raise

    raise last_error


def search_symbol(query: str) -> list:
    """Search Alpha Vantage's symbol database for a company name or partial
    ticker. Returns a list of candidate matches (symbol, name, region,
    currency, type). Costs 1 API call, same budget as a quote request.
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set in environment/.env")

    response = _alpha_vantage_get(
        {"function": "SYMBOL_SEARCH", "keywords": query, "apikey": ALPHA_VANTAGE_API_KEY},
        timeout=30,
    )
    data = response.json()

    if "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Note") or data.get("Information"))

    matches = data.get("bestMatches", [])
    return [
        {
            "symbol": m.get("1. symbol", ""),
            "name": m.get("2. name", ""),
            "type": m.get("3. type", ""),
            "region": m.get("4. region", ""),
            "currency": m.get("8. currency", ""),
        }
        for m in matches
    ]


def resolve_ticker(raw: str, model: str) -> tuple:
    """Resolve a company name or possibly-wrong ticker into a real,
    working Alpha Vantage symbol. Tries the input directly first (cheap,
    1 API call, covers the common case where it's already a valid ticker
    like 'AAPL'). If that fails, searches by name and asks Mistral to pick
    the best candidate -- preferring proper exchange listings (symbols with
    a region-specific suffix, e.g. '.LON', '.DEX') over thinly-traded US OTC
    ADR-style symbols, which tend to have wider spreads and less reliable data.

    Returns (resolved_symbol, quote_data, resolution_note). resolution_note
    is None if the input worked as-is, or a short string describing what it
    was resolved to/from, for user-facing confirmation messages.
    """
    try:
        data = fetch_stock_data(raw)
        return raw.upper(), data, None
    except ValueError:
        pass  # raw wasn't a valid ticker directly -- fall through to search

    candidates = search_symbol(raw)
    if not candidates:
        raise ValueError(f"Could not find any stock matching '{raw}'.")

    candidates_desc = "\n".join(
        f"{i+1}. {c['symbol']} -- {c['name']} ({c['region']}, {c['currency']})"
        for i, c in enumerate(candidates)
    )
    prompt = f"""The user wants to track the stock "{raw}". Here are possible matches
from a symbol search:

{candidates_desc}

Pick the BEST match to use for tracking this company's price. Prefer:
- A listing whose currency matches the company's home country when
  identifiable from the name/region (most accurate price tracking).
- Proper exchange listings (symbols with a region suffix like ".LON",
  ".DEX", ".PAR") over plain 5-letter US OTC symbols (these often have
  wider spreads / less reliable data for non-US companies).

Respond with ONLY the number of your choice, nothing else."""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    raw_choice = response.json()["response"].strip()

    try:
        choice_index = int("".join(c for c in raw_choice if c.isdigit())) - 1
        chosen = candidates[choice_index]
    except (ValueError, IndexError):
        chosen = candidates[0]  # fall back to the top search result

    resolved_symbol = chosen["symbol"]
    data = fetch_stock_data(resolved_symbol)
    note = f"'{raw}' \u2192 {resolved_symbol} ({chosen['name']}, {chosen['region']}, {chosen['currency']})"
    return resolved_symbol, data, note


def summarize_with_mistral(all_data: list, model: str) -> str:
    """Ask the local model to write a short digest covering all fetched tickers."""
    prompt = f"""You are a financial data assistant writing a short daily digest message
for Discord. Below is CURRENT, real, live stock data just fetched from a market
data API for one or more tickers. Treat these numbers as ground truth for right now.

Data:
{json.dumps(all_data, indent=2)}

Write a concise daily digest covering each ticker: current price, how it compares
to the previous close (use the provided change_percent, and confirm it matches
the price difference), and the day's high/low range. Use a short heading per
ticker. Do not give buy/sell recommendations. Keep the whole thing readable in a
Discord message -- a few lines per ticker, no long paragraphs.
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def post_to_discord(webhook_url: str, content: str):
    """Post a message to Discord via webhook, splitting if over the length limit."""
    chunks = []
    while len(content) > MAX_DISCORD_LEN:
        split_at = content.rfind("\n", 0, MAX_DISCORD_LEN)
        if split_at == -1:
            split_at = MAX_DISCORD_LEN
        chunks.append(content[:split_at])
        content = content[split_at:]
    chunks.append(content)

    for chunk in chunks:
        resp = requests.post(webhook_url, json={"content": chunk}, timeout=30)
        resp.raise_for_status()


def run_digest(tickers: list, model: str = "mistral") -> str:
    """Fetch data, summarize, and post to Discord for the given tickers.
    Returns a short status string. Raises on unrecoverable errors (e.g. no
    webhook configured, Ollama unreachable) so callers can report failures.
    """
    if not WEBHOOK_URL:
        raise RuntimeError("STOCK_WEBHOOK_URL not set in environment/.env")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_data = []
    failures = []
    for ticker in tickers:
        try:
            all_data.append(fetch_stock_data(ticker))
        except Exception as e:
            failures.append(f"{ticker}: {e}")

    if not all_data:
        raise RuntimeError(f"No data fetched successfully. Failures: {failures}")

    summary = summarize_with_mistral(all_data, model)
    message = f"**Stock Digest — {timestamp}**\n\n{summary}"
    post_to_discord(WEBHOOK_URL, message)

    status = f"Posted digest for {', '.join(t['ticker'] for t in all_data)}."
    if failures:
        status += f" (Failed: {', '.join(failures)})"
    return status


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data, summarize with local LLM, post to Discord.")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols, e.g. AAPL ORSTED.CO TSLA")
    parser.add_argument("--model", default="mistral", help="Ollama model to use (default: mistral)")
    args = parser.parse_args()

    print(f"Fetching data for: {', '.join(args.tickers)}")
    try:
        status = run_digest(args.tickers, args.model)
        print(status)
    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Is it running? Try: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()