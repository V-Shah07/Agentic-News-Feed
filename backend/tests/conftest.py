"""Shared test fixtures. Unit tests run against an isolated SQLite database."""
from __future__ import annotations

import os

# Force a throwaway SQLite DB and disable the background scheduler before any
# app module imports and reads settings.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_newsfeed.db")
os.environ.setdefault("INGEST_INTERVAL_SECONDS", "0")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import models


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    models.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
