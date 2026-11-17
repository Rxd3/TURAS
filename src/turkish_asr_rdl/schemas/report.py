"""Validation results shared by every artifact validator."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationReport:
    """Rule violations (errors) and advisory findings (warnings) for one check."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def combine(*reports: ValidationReport) -> ValidationReport:
    """Merge reports, keeping errors and warnings in the order given."""
    return ValidationReport(
        errors=tuple(e for r in reports for e in r.errors),
        warnings=tuple(w for r in reports for w in r.warnings),
    )
