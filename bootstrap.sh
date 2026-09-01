#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable. Check that Docker is running and your user can access the Docker socket." >&2
  exit 1
fi

if [ ! -f .env ]; then
  echo ".env not found. Copy .env.example to .env and fill in the required values first." >&2
  exit 1
fi

echo "Starting Ollama service..."
docker compose up -d ollama

for i in $(seq 1 30); do
  if docker compose ps ollama | grep -q "healthy\|running"; then
    break
  fi
  sleep 2
  if [ "$i" -eq 30 ]; then
    echo "Ollama did not become healthy in time." >&2
    docker compose logs --tail=100 ollama >&2
    exit 1
  fi
done

echo "Pulling Mistral model..."
docker exec ollama ollama pull mistral

echo "Pulling embedding model..."
docker exec ollama ollama pull nomic-embed-text

echo "Starting bot service..."
docker compose up -d bot

echo "Bootstrap complete."
echo "Useful checks:"
echo "  docker compose ps"
echo "  docker compose logs -f ollama"
echo "  docker compose logs -f bot"
