"""Unit tests for agent demo tools (convert_units, search_knowledge_base, get_current_time)."""

import re

from app.main import (
    _canonical_unit,
    convert_units,
    get_current_time,
    search_knowledge_base,
)


def test_canonical_unit_miles_and_aliases():
    assert _canonical_unit("mi") == "miles"
    assert _canonical_unit("Miles") == "miles"
    assert _canonical_unit("km") == "kilometers"


def test_convert_miles_to_km():
    out = convert_units(1, "miles", "km")
    assert "miles" in out.lower() and "kilometers" in out.lower()
    assert re.search(r"1\.609", out)


def test_convert_f_to_c():
    out = convert_units(32, "fahrenheit", "celsius")
    assert "0" in out.replace(" ", "")


def test_convert_units_invalid_amount():
    out = convert_units("x", "miles", "km")
    assert "invalid" in out.lower()


def test_convert_units_same_unit():
    out = convert_units(5, "km", "km")
    assert "unchanged" in out.lower() or "kilometers" in out.lower()


def test_search_knowledge_base_empty_query():
    assert "no query" in search_knowledge_base("").lower()


def test_get_current_time_utc():
    out = get_current_time("UTC")
    assert re.search(r"\d{4}-\d{2}-\d{2}", out)
    assert "UTC" in out or "+0000" in out or "GMT" in out


def test_get_current_time_bad_tz():
    out = get_current_time("Not/A_Real_Zone")
    assert "unknown" in out.lower()
