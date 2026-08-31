#!/usr/bin/env python3
"""
stock_assistant.py

Fetches live stock data (via yfinance) and passes it to a local Ollama
model (default: mistral) to get a plain-English summary/analysis.

Usage:
    python3 stock_assistant.py AAPL
    python3 stock_assistant.py TSLA --model llama3.1:8b
    python3 stock_assistant.py MSFT --raw   # just print the data, skip the LLM

Requirements:
    pip install yfinance requests
    Ollama must be installed and running (ollama serve), with the model
    already pulled (e.g. `ollama pull mistral`).
"""

import argparse
import json
import sys

import requests
import yfinance as yf

OLLAMA_URL = "http://localhost:11434/api/generate"


def fetch_stock_data(ticker: str) -> dict:
    """Pull current price + basic fundamentals for a ticker via yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        raise ValueError(f"Could not find data for ticker '{ticker}'. Check the symbol.")

    price = info.get("currentPrice") or info.get("regularMarketPrice")

    data = {
        "ticker": ticker.upper(),
        "company_name": info.get("longName", "N/A"),
        "current_price": price,
        "currency": info.get("currency", "N/A"),
        "previous_close": info.get("previousClose", "N/A"),
        "day_high": info.get("dayHigh", "N/A"),
        "day_low": info.get("dayLow", "N/A"),
        "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "volume": info.get("volume", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
    }
    return data


def ask_local_model(data: dict, model: str) -> str:
    """Send the fetched data to a local Ollama model for a plain-English summary."""
    prompt = f"""You are a financial data assistant. Below is CURRENT, real, live stock
data that was just fetched from a market data API. Do not question its accuracy
or claim you lack real-time data -- treat these numbers as ground truth for right now.

Data:
{json.dumps(data, indent=2)}

Give the user a short, clear summary (4-6 sentences) covering:
- Current price and how it compares to the previous close (up/down, and by how much %)
- Where it sits relative to its 52-week range
- Any notable valuation context (P/E ratio) if available
- One sentence of neutral, factual context (no investment advice)

Do not give buy/sell recommendations. Keep it concise and readable.
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_ticker_from_question(question: str, model: str) -> str:
    """Ask the local model to pull a stock ticker out of a natural-language question."""
    prompt = f"""Extract the stock ticker symbol for the company mentioned in this question.
Respond with ONLY the ticker symbol, nothing else -- no explanation, no punctuation.
If the company trades on a non-US exchange, include the correct Yahoo Finance suffix
(e.g. Orsted -> ORSTED.CO, since it trades in Copenhagen).
If you cannot confidently identify a ticker, respond with exactly: UNKNOWN

Question: {question}

Ticker:"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    ticker = response.json()["response"].strip().split()[0].strip(".,!?\"'")
    return ticker


def answer_natural_question(question: str, data: dict, model: str) -> str:
    """Answer the user's original natural-language question using the fetched data."""
    prompt = f"""You are a financial data assistant. Below is CURRENT, real, live stock
data that was just fetched from a market data API. Treat these numbers as ground
truth for right now -- do not claim you lack real-time data.

Data:
{json.dumps(data, indent=2)}

The user asked: "{question}"

Answer their question directly and conversationally using the data above.
Do not give buy/sell recommendations. Keep it concise (a few sentences).
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def interactive_mode(model: str):
    """Chat loop: ask questions in plain English, get live-data-backed answers."""
    print(f"Interactive stock assistant (model: {model})")
    print("Ask about any stock in plain English. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "/bye"):
            print("Bye!")
            break

        try:
            print("Identifying ticker...")
            ticker = extract_ticker_from_question(question, model)

            if ticker.upper() == "UNKNOWN":
                print("Couldn't identify a specific company/ticker from that. Try naming it more directly.\n")
                continue

            print(f"Fetching live data for {ticker}...")
            data = fetch_stock_data(ticker)

            print("Thinking...\n")
            answer = answer_natural_question(question, data, model)
            print(answer)
            print()

        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama. Is it running? Try: ollama serve\n")
        except ValueError as e:
            print(f"{e}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch live stock data and summarize it with a local LLM.")
    parser.add_argument("ticker", nargs="?", help="Stock ticker symbol, e.g. AAPL, TSLA, MSFT")
    parser.add_argument("--model", default="mistral", help="Ollama model to use (default: mistral)")
    parser.add_argument("--raw", action="store_true", help="Only print raw data, skip the LLM summary")
    parser.add_argument("--chat", action="store_true", help="Interactive mode: ask questions in plain English")
    args = parser.parse_args()

    if args.chat or not args.ticker:
        interactive_mode(args.model)
        return

    try:
        print(f"Fetching live data for {args.ticker.upper()}...\n")
        data = fetch_stock_data(args.ticker)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    print("Raw data:")
    for k, v in data.items():
        print(f"  {k}: {v}")
    print()

    if args.raw:
        return

    try:
        print(f"Asking {args.model} to summarize...\n")
        summary = ask_local_model(data, args.model)
        print("Summary:")
        print(summary)
    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Is it running? Try: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"Error getting summary from model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
