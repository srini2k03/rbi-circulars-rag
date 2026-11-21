"""Which kind of regulated entity a notification is addressed to.

The November 2025 consolidation issued one set of Directions per entity type, titled
``Reserve Bank of India (<Entity> - <Topic>) Directions``. For a commercial bank's policy,
the NBFC, co-operative bank, RRB, ... versions are the wrong authority even when their
text is nearly identical, and sometimes their numbers differ.
"""

from __future__ import annotations

import re

TARGET_ENTITY = "Commercial Banks"
OTHER_ENTITIES: frozenset[str] = frozenset(
    {
        "Non-Banking Financial Companies",
        "Urban Co-operative Banks",
        "Rural Co-operative Banks",
        "Regional Rural Banks",
        "Small Finance Banks",
        "Payments Banks",
        "Local Area Banks",
        "All India Financial Institutions",
        "Asset Reconstruction Companies",
        "Housing Finance Companies",
        "Credit Information Companies",
        "Core Investment Companies",
        "Mortgage Guarantee Companies",
        "Standalone Primary Dealers",
        "Non-Bank Prepaid Payment Instruments Issuers",
        "Non-Operative Financial Holding Companies",
    }
)
_ENTITY_RE = re.compile(r"^Reserve Bank of India \((?P<entity>[^()]+?)\s+-\s")


def entity_of(title: str | None) -> str | None:
    """``Reserve Bank of India (Urban Co-operative Banks - KYC) Directions`` -> the entity;
    ``None`` for notifications not addressed to a single entity type."""
    if not title:
        return None
    m = _ENTITY_RE.match(title)
    return m.group("entity").strip() if m else None


def is_other_entity(title: str | None, target: str = TARGET_ENTITY) -> bool:
    """True when the notification is addressed to a known entity type other than ``target``."""
    entity = entity_of(title)
    return entity is not None and entity != target and entity in OTHER_ENTITIES
