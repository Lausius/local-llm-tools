#!/usr/bin/env python3
"""
stock_digest.py

Fetches live data for one or more stock tickers, asks a local Ollama model
(default: mistral) to summarize them, and posts the result to a Discord
channel via a webhook. Designed to be run manually or on a schedule (cron) --
no bot process needs to stay running for this, unlike discord_mistral_bot.py.

Setup:
    pip install yfinance requests python-dotenv

    Create/update .env in the same folder with:
        STOCK_WEBHOOK_URL=your_webhook_url_here

    Make sure Ollama is running locally (ollama serve) and the model is
    pulled (ollama pull mistral).

Run manually:
    python3 stock_digest.py ORSTED.CO AAPL TSLA

Note: recurring scheduling is now handled by the Discord bot itself
(schedule_manager.py, via the !schedule command) rather than host cron --
this keeps things working identically whether running on bare metal or
inside Docker. This script can still be run manually anytime, or wrapped
in your own scheduler if you're not using the bot's built-in one.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("STOCK_WEBHOOK_URL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MAX_DISCORD_LEN = 2000


def fetch_stock_data(ticker: str) -> dict:
    """Pull current price + basic fundamentals for a ticker via yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not info or price is None:
        raise ValueError(f"Could not find data for ticker '{ticker}'. Check the symbol.")

    return {
        "ticker": ticker.upper(),
        "company_name": info.get("longName", "N/A"),
        "current_price": price,
        "currency": info.get("currency", "N/A"),
        "previous_close": info.get("previousClose", "N/A"),
        "day_high": info.get("dayHigh", "N/A"),
        "day_low": info.get("dayLow", "N/A"),
        "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "volume": info.get("volume", "N/A"),
    }


def summarize_with_mistral(all_data: list, model: str) -> str:
    """Ask the local model to write a short digest covering all fetched tickers."""
    prompt = f"""You are a financial data assistant writing a short daily digest message
for Discord. Below is CURRENT, real, live stock data just fetched from a market
data API for one or more tickers. Treat these numbers as ground truth for right now.

Data:
{json.dumps(all_data, indent=2)}

Write a concise daily digest covering each ticker: current price, how it compares
to the previous close (up/down and by how much %), and one line of relevant
context (e.g. proximity to 52-week high/low) if notable. Use a short heading per
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
