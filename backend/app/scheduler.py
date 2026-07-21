"""In-process ingestion scheduler (APScheduler). NOT AWS Lambda — runs in-container."""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from .config import get_settings
from .db import session_scope
from .ingest import run_ingest

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _job() -> None:
    logger.info("Scheduled ingestion job starting")
    try:
        with session_scope() as session:
            result = run_ingest(session)
        logger.info("Scheduled ingestion: %s", result.summary())
    except Exception:
        logger.exception("Scheduled ingestion failed")


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler
    settings = get_settings()
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _job,
        "interval",
        seconds=settings.ingest_interval_seconds,
        id="ingest",
        next_run_time=None,  # first run triggered explicitly on startup
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started; ingest interval=%ss", settings.ingest_interval_seconds
    )
    return _scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
