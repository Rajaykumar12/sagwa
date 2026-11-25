"""Go/no-go for the next live agent run under Groq's free-tier daily cap.

Groq doesn't report daily (TPD) usage in response headers, so this sums the
agent's own token counts from `sagwa.db` over the last 24 hours (the cap is a
rolling window) and adds the next run's worst-case estimate. The cap is set
30K under Groq's 200K/day for `gpt-oss-20b`, as a margin for anything the
database doesn't see (probes, other tools on the same key).

    python -m examples.agent.budget <worst_case_tokens>   # exit 0 = GO, 1 = STOP

Counts runs of `examples.agent.adapter:*` only: the agent is the 20b user.
The judge runs on `gpt-oss-120b` with its own cap and records no tokens.
"""
import sqlite3
import sys

CAP = 170_000


def used_last_24h(db: str = "sagwa.db") -> int:
    con = sqlite3.connect(db)
    return con.execute("""SELECT COALESCE(SUM(r.tokens), 0) FROM results r JOIN runs u ON u.id = r.run_id
                          WHERE u.target_name LIKE 'examples.agent.adapter:%'
                          AND u.created_at >= datetime('now', '-24 hours')""").fetchone()[0]


if __name__ == "__main__":
    used = used_last_24h()
    worst = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ok = used + worst <= CAP
    print(f"20b used in last 24h: {used:,} | next run worst case: {worst:,} | "
          f"total {used + worst:,} / cap {CAP:,} -> {'GO' if ok else 'STOP'}")
    sys.exit(0 if ok else 1)
