# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File log exporter: write OTel log records to a JSONL file.

When OTEL_LOGS_EXPORTER=file and OTEL_LOGS_FILE_PATH is set, log records
(prompts, completions, etc. from OpenAI instrumentation) are written to the file.
The file is general-purpose; evals and other consumers can read it.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Import SDK types used by BatchLogRecordProcessor
try:
    from opentelemetry.sdk._logs.export import (
        LogRecordExportResult,
        LogRecordExporter,
    )
    from opentelemetry.sdk._logs import ReadableLogRecord
    _FILE_EXPORTER_AVAILABLE = True
except ImportError:
    LogRecordExporter = None  # type: ignore[misc, assignment]
    LogRecordExportResult = None
    ReadableLogRecord = None
    _FILE_EXPORTER_AVAILABLE = False


OTEL_LOGS_FILE_PATH = "OTEL_LOGS_FILE_PATH"
_DEFAULT_FILE_PATH = "data/otel_logs.jsonl"


def _default_file_path() -> str:
    return os.getenv(OTEL_LOGS_FILE_PATH, _DEFAULT_FILE_PATH)


class FileLogExporter(LogRecordExporter if LogRecordExporter else object):  # type: ignore[misc]
    """Exports OTel log records to a JSONL file (one JSON object per line).

    Path is read from OTEL_LOGS_FILE_PATH (default: data/otel_logs.jsonl).
    Parent directory is created if it does not exist. Writes are append-only
    and serialized with a lock for thread safety.
    """

    def __init__(self, file_path: str | None = None) -> None:
        if not _FILE_EXPORTER_AVAILABLE:
            raise RuntimeError(
                "FileLogExporter requires opentelemetry-sdk (logs); install opentelemetry-sdk"
            )
        self._path = file_path or _default_file_path()
        self._lock = threading.Lock()
        self._file = None
        self._shutdown = False

    def _ensure_file(self) -> None:
        if self._file is not None and not self._file.closed:
            return
        path = Path(self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if self._shutdown:
            logger.warning("FileLogExporter already shutdown, ignoring batch")
            return LogRecordExportResult.FAILURE
        if not batch:
            return LogRecordExportResult.SUCCESS
        try:
            with self._lock:
                self._ensure_file()
                for record in batch:
                    # Single-line JSON for JSONL (same shape as ReadableLogRecord.to_json())
                    line = record.to_json(indent=None) + "\n"
                    self._file.write(line)
                self._file.flush()
            return LogRecordExportResult.SUCCESS
        except Exception as e:
            logger.exception("FileLogExporter export failed: %s", e)
            return LogRecordExportResult.FAILURE

    def shutdown(self) -> None:
        self._shutdown = True
        with self._lock:
            if self._file is not None and not self._file.closed:
                try:
                    self._file.close()
                except Exception as e:
                    logger.debug("FileLogExporter close: %s", e)
                self._file = None
