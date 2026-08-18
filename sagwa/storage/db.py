"""Engine/session setup, reading DATABASE_URL from the environment."""
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

_engine = None
_SessionLocal = None


def _init():
    global _engine, _SessionLocal
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL", "sqlite:///./sagwa.db")
        _engine = create_engine(database_url)
        _SessionLocal = sessionmaker(bind=_engine)


@contextmanager
def get_session() -> Session:
    _init()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
