# Local Mistral Discord Bot

A self-hosted Discord bot backed by a local LLM (Mistral, via [Ollama](https://ollama.com)), running entirely on your own hardware — no data leaves your machine. Includes persistent memory, long-term fact recall, and a safe, allowlisted stock-digest scheduler. Fully containerized with Docker for easy setup on any machine.

---

## Features

- **Chat in Discord** — talk to a local Mistral model directly in a Discord channel, no cloud API involved.
- **Two layers of memory:**
  - *Rolling chat history* (`chat_memory.json`) — short-term context per channel.
  - *Long-term facts* (`remembered_facts.json`) — durable facts extracted from conversation (preferences, dates, ongoing projects), remembered across restarts and channels.
- **Live stock digests** — fetches real market data (via `yfinance`) and has Mistral summarize it in plain English, posted to Discord via webhook. Avoids the model ever inventing prices or facts — real data goes in, a summary comes out.
- **Safe, scheduled digests** — `!schedule` lets you manage recurring digests in plain English. The model can only ever trigger one of four fixed, validated actions (list/add/remove/edit) — it never gets shell or file access. See [Safety design](#safety-design) below.
- **Fully containerized** — Ollama + the bot run in Docker with GPU passthrough, making the whole setup portable across machines/distros.

---

## Architecture

```
┌─────────────┐      ┌──────────────────────┐      ┌─────────────┐
│   Discord    │◄────►│  discord_mistral_bot  │◄────►│   Ollama    │
│  (channel)   │      │   (this container)    │      │ (container, │
└─────────────┘      │                        │      │  GPU-accel) │
                      │  - chat_memory.json    │      └─────────────┘
                      │  - remembered_facts    │
                      │  - schedules.json      │
                      │  (APScheduler, runs    │
                      │   stock_digest.py at   │
                      │   scheduled times)     │
                      └───────────┬────────────┘
                                  │
                                  ▼
                         Discord webhook
                        (posts digest msgs)
```

---

## Prerequisites

- A machine with an NVIDIA GPU (tested on an RTX A3000 Laptop GPU, 6GB VRAM)
- Docker Engine + Docker Compose (native Engine, **not** Docker Desktop — see [Troubleshooting](#troubleshooting) for why)
- NVIDIA Container Toolkit (for GPU passthrough into Docker)
- A Discord account with permission to create applications/bots

---

## Setup

### 1. Install Docker Engine (if not already installed)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```
Log out and back in (or reboot) for the group change to take effect.

### 2. Install the NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
```

**Arch/Omarchy:**
```bash
sudo pacman -S nvidia-container-toolkit
```

Then register it with Docker and restart:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it worked:
```bash
docker info | grep -i nvidia
```

### 3. Set up a Discord bot application

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) → **New Application**
2. Go to **Bot** → Reset Token → copy it (this is your `DISCORD_BOT_TOKEN`)
3. Under **Privileged Gateway Intents**, enable **Message Content Intent**
4. Go to **OAuth2 → URL Generator** → check scope `bot` → under Bot Permissions check `Send Messages` and `Read Message History` → copy the generated URL and open it to invite the bot to your server

### 4. Set up a Discord webhook (for stock digests)

In the channel you want digests posted to: **Edit Channel → Integrations → Webhooks → New Webhook** → copy the URL (this is your `STOCK_WEBHOOK_URL`)

### 5. Configure environment variables

```bash
cp .env.example .env
```
Edit `.env` and fill in `DISCORD_BOT_TOKEN` and `STOCK_WEBHOOK_URL`.

### 6. Build and start everything

```bash
docker compose up -d
```

### 7. Pull the model into the containerized Ollama

```bash
docker exec -it ollama ollama pull mistral
```

### 8. Verify

```bash
docker compose ps                    # both services should show healthy/running
docker compose logs -f bot           # should show "Logged in as ... Listening for messages..."
```

Send a message in your Discord channel — the bot should reply.

---

## Usage

### Chat
Just type any message in the channel the bot is in — it replies using Mistral, with access to short-term chat history and long-term remembered facts.

### Commands

| Command | Description |
|---|---|
| `!forget` | Clear rolling chat history for the current channel |
| `!facts` | List all remembered long-term facts (numbered) |
| `!forget-fact <n>` | Remove a single fact by its number |
| `!forget-facts` | Clear all long-term facts |
| `!schedule list` | List all scheduled stock digests |
| `!schedule add AAPL ORSTED.CO at 08:00` | Add a new daily digest schedule |
| `!schedule edit <id> to TSLA MSFT` | Change an existing schedule's tickers |
| `!schedule remove <id>` | Remove a schedule |

### Manual stock digest (without scheduling)
```bash
docker exec -it discord-mistral-bot python3 stock_digest.py ORSTED.CO AAPL
```

### Stopping / starting
```bash
docker compose down     # stop everything
docker compose up -d    # start again (data persists via volumes)
```

---

## Safety design

This project deliberately gives the model **no shell, file, or arbitrary command access** — a design choice made after specifically testing what would happen if it tried (it can't; it's text-in, text-out only).

The one place the model's output *does* trigger an action is `!schedule`: the model translates a plain-English instruction into strict JSON matching one of four fixed shapes (`list`, `add`, `remove`, `edit`). The application code validates every field (ticker format, time format, job count limits) before calling one of four fixed, narrow functions in `schedule_manager.py`. If the model's output doesn't match exactly, nothing executes — there is no generic/fallback execution path. This is the standard "function calling with a fixed toolset" pattern, deliberately kept narrow rather than giving the model raw command execution.

---

## Data persistence

All state is stored in Docker volumes, so it survives container restarts and rebuilds:

- `bot_data` volume → `chat_memory.json`, `remembered_facts.json`, `schedules.json`
- `ollama_data` volume → downloaded models (so you don't re-download Mistral on every rebuild)

To fully reset everything (including downloaded models):
```bash
docker compose down -v
```

---

## Troubleshooting

**`address already in use` on port 11434**
Something else (likely a native Ollama install) is already using that port. Either stop it (`sudo systemctl stop ollama && sudo systemctl disable ollama`) or remove the `ports` mapping for the `ollama` service in `docker-compose.yml` if you don't need host-level access to it.

**`could not select device driver "nvidia"`**
The NVIDIA Container Toolkit isn't installed or registered with Docker. See step 2 above. If you're on **Docker Desktop for Linux**, note that it runs containers inside its own VM with separate config from the native Engine — GPU passthrough is much simpler on the native Engine, which is what this project assumes.

**`permission denied` connecting to the Docker socket**
Your user was added to the `docker` group but the current shell session doesn't know yet. Run `newgrp docker` or open a new terminal.

**`error getting credentials - docker-credential-desktop not found`**
Leftover from a previous Docker Desktop install. Fix: `echo '{}' > ~/.docker/config.json`

**Healthcheck stuck / `curl: not found`**
The official Ollama image doesn't include `curl`. This project's healthcheck uses `ollama list` instead, which is guaranteed to exist in the image.

**`404 Client Error` when chatting**
The containerized Ollama has no models yet — it's a separate instance from any native install you may have used before. Run `docker exec -it ollama ollama pull mistral`.

**Script has `bash: ./script.sh: Permission denied`**
Run `chmod +x script.sh` first.

**Script fails with a garbled filename or `$'\r': command not found`**
The file has Windows-style line endings (CRLF). Fix with `sed -i 's/\r$//' script.sh` or `dos2unix script.sh`.

---

## Hardware notes

Tested on a Lenovo ThinkPad P15 Gen 2i (i7-11850H, RTX A3000 6GB VRAM, 32GB RAM). Mistral 7B runs comfortably GPU-accelerated at this VRAM level; larger models (13B+) will split across GPU/CPU and run slower. Running this continuously generates real heat under sustained load — this project intentionally avoids a 24/7-running bot process for scheduled tasks by keeping the scheduler lightweight and the interactive bot something you start/stop as needed, rather than running a dedicated always-on server on laptop hardware.

---

## Portability

Because everything runs in Docker with configuration via `.env` and `docker-compose.yml`, moving this to a new machine (e.g. switching from Ubuntu to Omarchy) should only require repeating the Setup steps above on the new machine — no manual Python venv or path reconfiguration needed.