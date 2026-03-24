"""Live weather via Open-Meteo (geocoding + forecast); no API key. https://open-meteo.com/"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_TIMEOUT_S = 10.0
_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Tokens that suggest the user means the United States (helps disambiguate "Dallas", "Portland", etc.)
_US_CONTEXT_SNIPPETS = (
    "texas",
    " tx",
    ", tx",
    " tx,",
    "usa",
    "u.s.",
    "u.s.a",
    "united states",
    "america",
)

# Word-boundary names of major Texas cities (for US + Texas disambiguation without explicit "TX")
_TEXAS_CITY_PATTERN = re.compile(
    r"\b("
    r"dallas|houston|san antonio|austin|fort worth|el paso|arlington|corpus christi|"
    r"plano|lubbock|laredo|irving|garland|frisco|mckinney|amarillo|brownsville"
    r")\b",
    re.I,
)


def _location_suggests_texas(q: str) -> bool:
    ql = (q or "").lower()
    if "texas" in ql or re.search(r"\btx\b", ql):
        return True
    return _TEXAS_CITY_PATTERN.search(q or "") is not None


def _location_suggests_us(q: str) -> bool:
    ql = (q or "").lower()
    for snip in _US_CONTEXT_SNIPPETS:
        if snip in ql:
            return True
    return _location_suggests_texas(q)


def _pick_geocode_hit(results: list, *, prefer_texas: bool) -> dict | None:
    if not results or not isinstance(results, list):
        return None
    if prefer_texas:
        for r in results:
            if not isinstance(r, dict):
                continue
            admin1 = (r.get("admin1") or "")
            if "texas" in admin1.lower():
                logger.info(
                    "Open-Meteo geocode: using Texas match name=%s admin1=%s id=%s",
                    r.get("name"),
                    admin1,
                    r.get("id"),
                )
                return r
    hit = results[0]
    if isinstance(hit, dict):
        logger.info(
            "Open-Meteo geocode: using first result name=%s admin1=%s country=%s id=%s",
            hit.get("name"),
            hit.get("admin1"),
            hit.get("country_code"),
            hit.get("id"),
        )
    return hit if isinstance(hit, dict) else None


def _wmo_label(code: int | None) -> str:
    """Short label for WMO weather interpretation codes (Open-Meteo)."""
    if code is None:
        return "Unknown conditions"
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return "Unknown conditions"
    if c == 0:
        return "Clear sky"
    if c in (1, 2):
        return "Mainly clear"
    if c == 3:
        return "Overcast"
    if c in (45, 48):
        return "Foggy"
    if c in (51, 53, 55):
        return "Drizzle"
    if c in (56, 57):
        return "Freezing drizzle"
    if c in (61, 63, 65):
        return "Rain"
    if c in (66, 67):
        return "Freezing rain"
    if c in (71, 73, 75):
        return "Snow"
    if c == 77:
        return "Snow grains"
    if c in (80, 81, 82):
        return "Rain showers"
    if c in (85, 86):
        return "Snow showers"
    if c == 95:
        return "Thunderstorm"
    if c in (96, 99):
        return "Thunderstorm with hail"
    return "Mixed conditions"


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def _geocode(query: str) -> tuple[float, float, str] | None:
    q = (query or "").strip()
    if not q:
        return None
    q = q[:200]
    suggest_us = _location_suggests_us(q)
    suggest_tx = _location_suggests_texas(q)
    count = 10 if (suggest_us or suggest_tx) else 5
    params: dict[str, str] = {
        "name": q,
        "count": str(count),
        "language": "en",
        "format": "json",
    }
    if suggest_us:
        params["countryCode"] = "US"
    enc = urllib.parse.urlencode(params)
    url = f"{_GEO_URL}?{enc}"
    try:
        data = _http_get_json(url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.warning("Open-Meteo geocode failed: %s", e)
        return None
    results = data.get("results")
    if not results or not isinstance(results, list):
        return None
    hit = _pick_geocode_hit(results, prefer_texas=suggest_tx)
    if not hit:
        return None
    lat, lon = hit.get("latitude"), hit.get("longitude")
    if lat is None or lon is None:
        return None
    name = hit.get("name") or q
    admin = hit.get("admin1")
    country = hit.get("country_code") or hit.get("country")
    place_parts = [name]
    if admin:
        place_parts.append(str(admin))
    if country:
        place_parts.append(str(country))
    label = ", ".join(place_parts)
    return float(lat), float(lon), label


def fetch_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """
    Resolve `location` with Open-Meteo geocoding, then fetch current conditions.
    Returns a single user-facing sentence or an error message (no stack traces).
    """
    u = unit if unit in ("celsius", "fahrenheit") else "fahrenheit"
    geo = _geocode(location)
    if geo is None:
        return f'No location match for "{(location or "").strip()[:120] or "unknown"}". Try a city and region or country.'

    lat, lon, place = geo
    wind_unit = "mph" if u == "fahrenheit" else "kmh"
    current_vars = (
        "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m"
    )
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": current_vars,
            "temperature_unit": u,
            "wind_speed_unit": wind_unit,
        }
    )
    url = f"{_FORECAST_URL}?{params}"
    try:
        data = _http_get_json(url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.warning("Open-Meteo forecast failed: %s", e)
        return "Weather service temporarily unavailable. Please try again later."

    cur = data.get("current")
    if not isinstance(cur, dict):
        return "Weather data was incomplete. Please try again later."

    temp = cur.get("temperature_2m")
    feel = cur.get("apparent_temperature")
    code = cur.get("weather_code")
    wind = cur.get("wind_speed_10m")
    rh = cur.get("relative_humidity_2m")

    sym = "°C" if u == "celsius" else "°F"
    wunit = "mph" if u == "fahrenheit" else "km/h"

    parts = [_wmo_label(code)]
    if temp is not None:
        parts.append(f"{round(float(temp))}{sym}")
    if feel is not None and temp is not None and abs(float(feel) - float(temp)) > 0.5:
        parts.append(f"feels like {round(float(feel))}{sym}")
    if wind is not None:
        parts.append(f"wind {round(float(wind))} {wunit}")
    if rh is not None:
        parts.append(f"{round(float(rh))}% humidity")

    summary = ", ".join(parts)
    return f"Current weather (Open-Meteo): {place}: {summary}."
