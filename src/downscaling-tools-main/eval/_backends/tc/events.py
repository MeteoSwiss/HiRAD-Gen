"""Pure TC event identity — geographic/temporal only."""
from __future__ import annotations

from dataclasses import dataclass

from .data_types import BoundingBox


@dataclass(frozen=True)
class TCEvent:
    name: str
    year: str
    month: str
    dates: list[str]  # DD strings
    analysis_dates: list[str]  # YYYYMMDD strings
    bbox: BoundingBox  # Evaluation crop box — must NOT overlap with other scoring events
    scoring_eligible: bool = True  # False for combined events like franklin_idalia


EVENTS: dict[str, TCEvent] = {
    "franklin": TCEvent(
        name="franklin",
        year="2023",
        month="08",
        dates=["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"],
        analysis_dates=["20230820", "20230826"],
        bbox=BoundingBox(north=38.0, south=15.0, east=-58.0, west=-78.0),
    ),
    "idalia": TCEvent(
        name="idalia",
        year="2023",
        month="08",
        dates=["26", "27", "28", "29", "30"],
        analysis_dates=["20230826"],
        bbox=BoundingBox(north=40.0, south=10.0, east=-80.0, west=-100.0),
    ),
    # Combined diagnostic region — covers both storms. Not for scoring.
    "franklin_idalia": TCEvent(
        name="franklin_idalia",
        year="2023",
        month="08",
        dates=["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"],
        analysis_dates=["20230820", "20230826"],
        bbox=BoundingBox(north=40.0, south=10.0, east=-58.0, west=-100.0),
        scoring_eligible=False,
    ),
    "hilary": TCEvent(
        name="hilary",
        year="2023",
        month="08",
        dates=["16", "17", "18", "19", "20"],
        analysis_dates=["20230816"],
        bbox=BoundingBox(north=35.0, south=0.0, east=-95.0, west=-125.0),
    ),
    "dora": TCEvent(
        name="dora",
        year="2023",
        month="08",
        dates=[
            "01", "02", "03", "04", "05", "06", "07", "08",
            "09", "10", "11", "12", "13", "14", "15", "16", "17",
        ],
        analysis_dates=["20230801", "20230807", "20230813"],
        bbox=BoundingBox(north=25.0, south=5.0, east=-105.0, west=175.0),  # crosses dateline
    ),
    "fernanda": TCEvent(
        name="fernanda",
        year="2023",
        month="08",
        dates=["12", "13", "14", "15", "16", "17"],
        analysis_dates=["20230812"],
        bbox=BoundingBox(north=30.0, south=0.0, east=-105.0, west=-135.0),
    ),
    "humberto": TCEvent(
        name="humberto",
        year="2025",
        month="09",
        dates=["26", "27", "28", "29", "30"],
        analysis_dates=["20250926"],
        bbox=BoundingBox(north=45.0, south=15.0, east=-50.0, west=-90.0),
    ),
}


def get_event(name: str) -> TCEvent:
    return EVENTS[name]


def get_scoring_events() -> list[TCEvent]:
    return [e for e in EVENTS.values() if e.scoring_eligible]


def events_for_period(year: str, month: str) -> list[TCEvent]:
    return [e for e in EVENTS.values() if e.year == year and e.month == month]
