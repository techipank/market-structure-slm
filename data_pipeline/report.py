"""Structured validation findings.

Design note
-----------
Validators return *data*, not exceptions. A single corrupt row must not hide
the other nine problems in the file. This is the same shape we will reuse for
dataset quality scoring later: a list of typed findings with a severity,
a machine-readable code, and pointers back to the offending rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from data_pipeline import SCHEMA_VERSION


class Severity(StrEnum):
    """Why two levels and not one.

    ERROR   -> the data is impossible or unusable. Real markets cannot produce
               it, so it is a source/transport bug. The file fails the gate.
    WARNING -> the data is unusual but physically possible (holiday gaps, a
               limit-up move, a zero-volume session). Recorded, not fatal.
    INFO    -> observations worth logging for provenance.
    """

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class Issue(BaseModel):
    code: str
    severity: Severity
    message: str
    #: Positional indices (0-based, into the loaded frame) of offending rows.
    #: Truncated to `sample_limit`; `count` carries the true total.
    row_indices: list[int] = Field(default_factory=list)
    #: Human-readable sample of offending timestamps, same truncation.
    row_timestamps: list[str] = Field(default_factory=list)
    count: int = 0


class CheckResult(BaseModel):
    check: str
    description: str
    passed: bool
    issues: list[Issue] = Field(default_factory=list)


class Verdict(StrEnum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


class DatasetFacts(BaseModel):
    """Provenance facts recorded so the train/test split can be *verified*
    split instead of assuming it."""

    source_path: str
    sha256: str
    rows: int
    columns: list[str]
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    inferred_interval_seconds: int | None = None
    declared_interval: str | None = None
    timezone: str | None = None


class ValidationReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    dataset: DatasetFacts
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(
            i.count for c in self.checks for i in c.issues if i.severity is Severity.ERROR
        )

    @property
    def warning_count(self) -> int:
        return sum(
            i.count for c in self.checks for i in c.issues if i.severity is Severity.WARNING
        )

    @property
    def verdict(self) -> Verdict:
        if self.error_count:
            return Verdict.FAIL
        if self.warning_count:
            return Verdict.PASS_WITH_WARNINGS
        return Verdict.PASS

    # ---------------------------------------------------------------- output

    def to_json(self) -> str:
        payload = self.model_dump(mode="json")
        payload["verdict"] = self.verdict.value
        payload["error_count"] = self.error_count
        payload["warning_count"] = self.warning_count
        return json.dumps(payload, indent=2, sort_keys=False)

    def to_markdown(self) -> str:
        d = self.dataset
        lines: list[str] = [
            f"# OHLCV validation report — `{Path(d.source_path).name}`",
            "",
            f"**Verdict: {self.verdict.value}**  "
            f"({self.error_count} errors, {self.warning_count} warnings)",
            "",
            "> Educational/research use only. Not trading advice.",
            "",
            "## Dataset",
            "",
            "| field | value |",
            "|---|---|",
            f"| source | `{d.source_path}` |",
            f"| sha256 | `{d.sha256}` |",
            f"| rows | {d.rows} |",
            f"| columns | {', '.join(d.columns)} |",
            f"| first timestamp | {d.first_timestamp} |",
            f"| last timestamp | {d.last_timestamp} |",
            f"| declared interval | {d.declared_interval} |",
            f"| inferred interval (s) | {d.inferred_interval_seconds} |",
            f"| timezone | {d.timezone} |",
            "",
            "## Checks",
            "",
            "| check | result | errors | warnings |",
            "|---|---|---|---|",
        ]
        for c in self.checks:
            errs = sum(i.count for i in c.issues if i.severity is Severity.ERROR)
            warns = sum(i.count for i in c.issues if i.severity is Severity.WARNING)
            lines.append(f"| `{c.check}` | {'PASS' if c.passed else 'FAIL'} | {errs} | {warns} |")

        detailed = [c for c in self.checks if c.issues]
        if detailed:
            lines += ["", "## Findings", ""]
            for c in detailed:
                lines.append(f"### `{c.check}` — {c.description}")
                lines.append("")
                for i in c.issues:
                    lines.append(
                        f"- **{i.severity.value}** `{i.code}` "
                        f"(count={i.count}) — {i.message}"
                    )
                    if i.row_timestamps:
                        shown = ", ".join(i.row_timestamps)
                        lines.append(f"  - sample timestamps: {shown}")
                lines.append("")

        lines += [
            "",
            "---",
            "",
            "_No data was modified or repaired to produce this report._",
            "",
        ]
        return "\n".join(lines)
