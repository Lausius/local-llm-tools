# Dockerfile for the Discord Mistral bot
# Build with: docker compose build
FROM python:3.12-slim

WORKDIR /app

# tzdata is needed for the TZ environment variable (set in docker-compose.yml)
# to actually resolve to a real timezone -- the slim base image doesn't
# include it by default.
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching -- only reinstalls if
# requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and reference docs -- code_search.py's !code command
# searches these files at runtime, so they need to physically exist here.
COPY discord_mistral_bot.py schedule_manager.py stock_digest.py code_search.py ./
COPY SELF_KNOWLEDGE.md README.md docker-compose.yml ./

# Data files (chat_memory.json, remembered_facts.json, schedules.json) live
# in a separate directory from the code, so a volume mounted here doesn't
# hide the code copied above. Created automatically on first run.
ENV DATA_DIR=/app/data
RUN mkdir -p /app/data

CMD ["python3", "discord_mistral_bot.py"]