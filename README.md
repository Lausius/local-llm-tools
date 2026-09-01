Local Mistral Discord Bot

A self-hosted Discord bot backed by a local LLM (Mistral, via Ollama), running entirely on your own hardware — no data leaves your machine. Includes persistent memory, long-term fact recall, and a safe, allowlisted stock-digest scheduler. Fully containerized with Docker for easy setup on any machine.

Features

Chat in Discord — talk to a local Mistral model directly in a Discord channel, no cloud API involved.

Two layers of memory:

Rolling chat history (chat_memory.json) — short-term context per channel.

Long-term facts (remembered_facts.json) — durable facts extracted from conversation (preferences, dates, ongoing projects), remembered across restarts and channels.

Live stock digests — fetches real market data from Alpha Vantage and has Mistral summarize it in plain English, posted to Discord via webhook. Avoids the model ever inventing prices or facts — real data goes in, a summary comes out.

Safe, scheduled digests — !schedule lets you manage recurring digests in plain English. Ollama is constrained by a JSON Schema, application code validates every field, and add/edit/remove require an explicit !confirm before anything changes.

Code introspection — !code <question> answers questions about the bot’s own implementation using a local keyword search over the project source, so answers are grounded in the actual code rather than guesswork.

Fully containerized — Ollama + the bot run in Docker with GPU passthrough, making the whole setup portable across machines/distros.

Architecture

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

Prerequisites

A machine with an NVIDIA GPU (tested on an RTX A3000 Laptop GPU, 6GB VRAM)

Docker Engine + Docker Compose (native Engine, not Docker Desktop — see Troubleshooting for why)

NVIDIA Container Toolkit (for GPU passthrough into Docker)

A Discord account with permission to create applications/bots

Setup

1. Install Docker Engine (if not already installed)

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

Log out and back in (or reboot) for the group change to take effect.

2. Install the NVIDIA Container Toolkit

Ubuntu/Debian:

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

Arch/Omarchy:

sudo pacman -S nvidia-container-toolkit

Then register it with Docker and restart:

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

Verify it worked:

docker info | grep -i nvidia

3. Set up a Discord bot application

Go to discord.com/developers/applications → New Application

Go to Bot → Reset Token → copy it (this is your DISCORD_BOT_TOKEN)

Under Privileged Gateway Intents, enable Message Content Intent

Go to OAuth2 → URL Generator → check scope bot → under Bot Permissions check Send Messages and Read Message History → copy the generated URL and open it to invite the bot to your server

4. Set up a Discord webhook (for stock digests)

In the channel you want digests posted to: Edit Channel → Integrations → Webhooks → New Webhook → copy the URL (this is your STOCK_WEBHOOK_URL)

5. Configure environment variables

cp .env.example .env

Edit .env and fill in at least:

DISCORD_BOT_TOKEN

STOCK_WEBHOOK_URL

ALPHA_VANTAGE_API_KEY (for live stock digests)

Optional:

OLLAMA_URL (defaults to http://localhost:11434/api/generate or Docker service URL in compose)

OLLAMA_EMBED_MODEL (defaults to nomic-embed-text; used for local repo indexing for !code semantic retrieval)

DATA_DIR (if you want bot data stored outside the repo folder)

ALLOWED_USER_IDS (recommended comma-separated Discord user IDs)

ALLOWED_CHANNEL_IDS (recommended comma-separated Discord channel IDs)

CONFIRMATION_TTL_SECONDS (defaults to 60)

MAX_INJECTED_FACTS (defaults to 20; prevents long-term memory from filling the prompt)

For a private bot, set at least ALLOWED_USER_IDS. Enable Developer Mode in
Discord and use Copy User ID / Copy Channel ID to obtain numeric IDs.

6. Build and start everything

docker compose up -d

7. Start Ollama and pull the required models

docker compose up -d

Once the ollama container is healthy, pull the models you need:

docker exec -it ollama ollama pull mistral
docker exec -it ollama ollama pull nomic-embed-text

8. Verify

docker compose ps                    # both services should show healthy/running
docker compose logs -f bot           # should show "Logged in as ... Listening for messages..."

Send a message in your Discord channel — the bot should reply.

Run the schedule validation tests:

docker compose run --rm bot python -m unittest -v test_schedule_intent.py

Usage

Chat

Just type any message in the channel the bot is in — it replies using Mistral, with access to short-term chat history and long-term remembered facts.

Commands

Command

Description

!forget

Clear rolling chat history for the current channel

!facts

List all remembered long-term facts (numbered)

!forget-fact <n>

Remove a single fact by its number

!forget-facts

Clear all long-term facts

!schedule list

List all scheduled stock digests

!schedule add AAPL ORSTED.CO at 08:00

Add a new daily digest schedule

!schedule edit <id> to TSLA MSFT

Change an existing schedule's tickers

!schedule remove <id>

Remove a schedule

!confirm

Confirm a pending add/edit/remove operation

!cancel

Cancel a pending schedule operation

!code <question>

Ask the bot to explain how the project works, using repo search and source-grounded answers

Manual stock digest (without scheduling)

docker exec -it discord-mistral-bot python3 stock_digest.py ORSTED.CO AAPL

Stopping / starting

docker compose down     # stop everything
docker compose up -d    # start again (data persists via volumes)

Safety design

This project deliberately gives the model no shell, file, or arbitrary command access — a design choice made after specifically testing what would happen if it tried (it can't; it's text-in, text-out only).

The one place model output influences an action is !schedule. Ollama's
structured-output mode constrains the response to a JSON Schema containing only
list, add, remove, edit, or unknown. The application then performs
action-specific validation, ticker resolution, and schedule validation. A
proposed add/edit/remove is stored only in memory for 60 seconds and executes
only after the same user replies !confirm in the same channel. There is no
generic execution path, and restarting the bot discards pending confirmations.

Normal Ollama requests run outside Discord's event loop, so a slow local model
does not freeze command handling. Long-term fact extraction happens after the
visible reply in a background task.

Data persistence

Persistent state survives container restarts and rebuilds:

./data bind mount → chat_memory.json, remembered_facts.json, schedules.json

ollama_data volume → downloaded models (so you don't re-download Mistral on every rebuild)

To fully reset everything (including downloaded models):

docker compose down -v

Troubleshooting

address already in use on port 11434
Something else (likely a native Ollama install) is already using that port. Either stop it (sudo systemctl stop ollama && sudo systemctl disable ollama) or remove the ports mapping for the ollama service in docker-compose.yml if you don't need host-level access to it.

The supplied Compose file binds Ollama to 127.0.0.1, not your LAN. You can
remove ports entirely if only the bot container needs Ollama.

The bot ignores my messages
If ALLOWED_USER_IDS or ALLOWED_CHANNEL_IDS is configured, confirm that the
numeric Discord IDs are correct. Both allowlists are enforced when both are set.

A schedule change was not applied
Add/edit/remove operations require !confirm from the same user in the same
channel within CONFIRMATION_TTL_SECONDS (60 seconds by default).

could not select device driver "nvidia"
The NVIDIA Container Toolkit isn't installed or registered with Docker. See step 2 above. If you're on Docker Desktop for Linux, note that it runs containers inside its own VM with separate config from the native Engine — GPU passthrough is much simpler on the native Engine, which is what this project assumes.

permission denied connecting to the Docker socket
Your user was added to the docker group but the current shell session doesn't know yet. Run newgrp docker or open a new terminal.

error getting credentials - docker-credential-desktop not found
Leftover from a previous Docker Desktop install. Fix: echo '{}' > ~/.docker/config.json

Healthcheck stuck / curl: not found
The official Ollama image doesn't include curl. This project's healthcheck uses ollama list instead, which is guaranteed to exist in the image.

404 Client Error when chatting
The containerized Ollama has no models yet — it's a separate instance from any native install you may have used before. Run docker exec -it ollama ollama pull mistral.

Script has bash: ./script.sh: Permission denied
Run chmod +x script.sh first.

Script fails with a garbled filename or $'\r': command not found
The file has Windows-style line endings (CRLF). Fix with sed -i 's/\r$//' script.sh or dos2unix script.sh.

Hardware notes

Tested on a Lenovo ThinkPad P15 Gen 2i (i7-11850H, RTX A3000 6GB VRAM, 32GB RAM). Mistral 7B runs comfortably GPU-accelerated at this VRAM level; larger models (13B+) will split across GPU/CPU and run slower. Running this continuously generates real heat under sustained load — this project intentionally avoids a 24/7-running bot process for scheduled tasks by keeping the scheduler lightweight and the interactive bot something you start/stop as needed, rather than running a dedicated always-on server on laptop hardware.

Portability

Because everything runs in Docker with configuration via .env and docker-compose.yml, moving this to a new machine (e.g. switching from Ubuntu to Omarchy) should only require repeating the Setup steps above on the new machine — no manual Python venv or path reconfiguration needed.
