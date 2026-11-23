"""Every test session gets its own throwaway, freshly migrated SQLite database.

Before this, the suite wrote its fixture runs into the developer's real
`sagwa.db` (or whatever `.env`'s DATABASE_URL named): dozens of `stub` and
`trace-test` runs per `pytest`, which buried the real agent runs in the
dashboard's trace viewer (2026-11-23).

Set here, at import time, because `sagwa.storage.db` creates its engine on
first use and `load_dotenv()` never overrides a variable that is already set.
Migrating with Alembic rather than `create_all` means the suite also proves
the migrations build the schema the models expect.
"""
import os
import tempfile
from pathlib import Path

_DB = Path(tempfile.mkdtemp(prefix="sagwa-tests-")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_DB}"


def pytest_sessionstart(session):
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parents[1] / "migrations"))
    command.upgrade(config, "head")
