"""
schedule_manager.py (Docker-friendly version)

A small, deliberately narrow interface for managing scheduled stock digest
runs. Same four validated operations as before -- the model still never gets
shell or file access, only these:

    list_schedules()
    add_schedule(tickers, time_str)
    remove_schedule(job_id)
    edit_tickers(job_id, new_tickers)

Unlike the original cron-based version, this one uses APScheduler running
inside the same process as the bot, with schedule state persisted to a JSON
file (schedules.json). This works identically whether running directly on a
host or inside a Docker container -- no dependency on the host's crontab,
which containers don't have access to. This module must be initialized once
(via init_scheduler) with the digest function to run when a job fires.

Requires: pip install apscheduler
"""

import os
import re
import json
import uuid

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(DATA_DIR, exist_ok=True)
SCHEDULES_FILE = os.path.join(DATA_DIR, "schedules.json")
MAX_JOBS = 20

TICKER_PATTERN = re.compile(r"^[A-Za-z0-9\.\-]{1,12}$")
TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

_scheduler = AsyncIOScheduler()
_digest_fn = None  # set via init_scheduler; called as await _digest_fn(tickers) when a job fires
_started = False


class ScheduleError(Exception):
    """Raised for any validation or operation failure. Safe to show to the user."""
    pass


def _load_schedules() -> dict:
    if not os.path.exists(SCHEDULES_FILE):
        return {}
    try:
        with open(SCHEDULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read {SCHEDULES_FILE}, starting with empty schedules.")
        return {}


def _save_schedules(schedules: dict):
    with open(SCHEDULES_FILE, "w", encoding="utf-8") as f:
        json.dump(schedules, f, indent=2)


def _validate_tickers(tickers: list) -> list:
    if not tickers:
        raise ScheduleError("At least one ticker is required.")
    if len(tickers) > 10:
        raise ScheduleError("Too many tickers in one schedule (max 10).")
    cleaned = [t.strip().upper() for t in tickers]
    for t in cleaned:
        if not TICKER_PATTERN.match(t):
            raise ScheduleError(f"'{t}' doesn't look like a valid ticker symbol.")
    return cleaned


def _validate_time(time_str: str):
    match = TIME_PATTERN.match(time_str.strip())
    if not match:
        raise ScheduleError(f"'{time_str}' isn't a valid 24-hour time (expected HH:MM, e.g. 08:00).")
    hour, minute = match.groups()
    return int(hour), int(minute)


def validate_schedule_spec(tickers: list, time_str: str | None = None) -> tuple[list, tuple[int, int] | None]:
    """Validate a proposed schedule without changing persisted jobs."""
    cleaned = _validate_tickers(tickers)
    parsed_time = _validate_time(time_str) if time_str is not None else None
    return cleaned, parsed_time


async def _fire_job(job_id: str):
    """Called by APScheduler when a job's time arrives. Looks up the job's
    current tickers from disk (in case they were edited) and runs the digest.
    """
    schedules = _load_schedules()
    entry = schedules.get(job_id)
    if not entry or _digest_fn is None:
        return
    try:
        await _digest_fn(entry["tickers"])
    except Exception as e:
        print(f"Scheduled digest job {job_id} failed: {e}")


def init_scheduler(digest_fn):
    """Call once at bot startup, from within a running asyncio event loop.
    digest_fn must be an async function that accepts a list of tickers and
    runs+posts the digest. Restores any schedules saved from a previous run
    (this is what makes schedules survive a container restart).
    """
    global _digest_fn, _started
    _digest_fn = digest_fn

    if not _started:
        _scheduler.start()
        _started = True

    for job_id, entry in _load_schedules().items():
        _scheduler.add_job(
            _fire_job,
            trigger=CronTrigger(hour=entry["hour"], minute=entry["minute"]),
            args=[job_id],
            id=job_id,
            replace_existing=True,
        )


def list_schedules() -> list:
    schedules = _load_schedules()
    return [
        {"id": job_id, "tickers": entry["tickers"], "time": f"{entry['hour']:02d}:{entry['minute']:02d}"}
        for job_id, entry in schedules.items()
    ]


def add_schedule(tickers: list, time_str: str) -> dict:
    schedules = _load_schedules()
    if len(schedules) >= MAX_JOBS:
        raise ScheduleError(f"Maximum number of schedules ({MAX_JOBS}) reached. Remove one first.")

    tickers = _validate_tickers(tickers)
    hour, minute = _validate_time(time_str)

    job_id = uuid.uuid4().hex[:8]
    schedules[job_id] = {"tickers": tickers, "hour": hour, "minute": minute}
    _save_schedules(schedules)

    _scheduler.add_job(
        _fire_job,
        trigger=CronTrigger(hour=hour, minute=minute),
        args=[job_id],
        id=job_id,
        replace_existing=True,
    )

    return {"id": job_id, "tickers": tickers, "time": f"{hour:02d}:{minute:02d}"}


def remove_schedule(job_id: str) -> bool:
    schedules = _load_schedules()
    if job_id not in schedules:
        return False

    del schedules[job_id]
    _save_schedules(schedules)

    try:
        _scheduler.remove_job(job_id)
    except Exception:
        pass  # job may not be registered yet (e.g. right after a fresh restart)

    return True


def edit_tickers(job_id: str, new_tickers: list) -> dict:
    schedules = _load_schedules()
    if job_id not in schedules:
        raise ScheduleError(f"No schedule found with id '{job_id}'. Use list to see valid ids.")

    new_tickers = _validate_tickers(new_tickers)
    schedules[job_id]["tickers"] = new_tickers
    _save_schedules(schedules)

    entry = schedules[job_id]
    return {"id": job_id, "tickers": new_tickers, "time": f"{entry['hour']:02d}:{entry['minute']:02d}"}
