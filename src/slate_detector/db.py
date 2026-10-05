"""Read-only SQL against a restored FishSense v1 backup.

The backup (``~/mnt/fishsense_process_work/database_backups/fishsense/*.dump``)
is restored into a throwaway ``postgres:17`` container; see README. Every
query runs inside ``BEGIN READ ONLY`` and is rolled back. The runner shells in
with ``docker exec``, so no password passes through this code.
"""

from __future__ import annotations

import csv
import io
import subprocess

CONTAINER = "slate-db"
DATABASE = "fishsense"


def wrap_read_only(sql: str) -> str:
    return f"BEGIN READ ONLY;\n{sql.rstrip().rstrip(';')};\nROLLBACK;\n"


class DockerPsqlRunner:
    def __init__(self, container: str = CONTAINER, database: str = DATABASE, user: str = "postgres"):
        self.container, self.database, self.user = container, database, user

    def query(self, sql: str) -> list[dict[str, str]]:
        """One SELECT; rows as dicts of strings (psql CSV output)."""
        cmd = [
            "docker", "exec", "-i", self.container,
            "psql", "-U", self.user, "-d", self.database,
            "-q", "--csv", "-v", "ON_ERROR_STOP=1",
        ]
        result = subprocess.run(cmd, input=wrap_read_only(sql), capture_output=True, text=True, check=True)
        return list(csv.DictReader(io.StringIO(result.stdout)))
