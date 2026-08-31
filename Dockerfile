# Dockerfile for the Discord Mistral bot
# Build with: docker compose build
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching -- only reinstalls if
# requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY discord_mistral_bot.py schedule_manager.py stock_digest.py ./

# Data files (chat_memory.json, remembered_facts.json, schedules.json) live
# in a separate directory from the code, so a volume mounted here doesn't
# hide the code copied above. Created automatically on first run.
ENV DATA_DIR=/app/data
RUN mkdir -p /app/data

CMD ["python3", "discord_mistral_bot.py"]
