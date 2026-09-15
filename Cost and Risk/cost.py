"""
Role 6 - Predictive Cost & Risk Engineer
=========================================
Standalone module for the Intelligent Cloud Migration Planner.

Pipeline (per application):
  1. LIVE AWS pricing  - tiered resolution chain, real USD/hour, never hard-fails
  2. MONTE-CARLO cost  - NumPy simulation of utilization + storage variance,
                         reports 5th / mean / 95th percentile monthly cost
  3. RISK SCORE        - explainable weighted model, 0-100 + LOW/MEDIUM/HIGH
  4. JSON OUTPUT       - one dict per app, ready for the React dashboard and
                         the GenAI copilot's FAISS index

Pricing tier chain (first hit wins, later tiers are automatic fallbacks):
  T1  boto3 'pricing' client (AWS Price List Query API, signed credentials)
  T2  boto3 'pricing' client, anonymous (unsigned) - works where the API
      allows unauthenticated queries
  T3  Public AWS Pricing Calculator pricing file (b0.p.awsstatic.com) -
      anonymous HTTPS, ~0.7 MB, all instance types for one region
  T4  Local JSON cache (cache/aws_prices_<region>.json), refreshed at most
      once every CACHE_TTL_HOURS
  T5  Built-in static table - last-resort estimates so demos always run

Tier discipline: T1/T2 are the "official SDK" story for the paper; T3 keeps the
module fully functional with zero AWS account; T4 avoids re-downloading on
every call; T5 guarantees no crash during a live demo.

Usage:
    python cost.py                       # demo portfolio, prints JSON
    python cost.py --input apps.json     # analyze a portfolio file
    python cost.py --app APP_001         # single app by id from the input file
    python cost.py --refresh-pricing     # force-refresh the price cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Module constants
# ----------------------------------------------------------------------------

HOURS_PER_MONTH = 730          # AWS convention: 730 h/month
CACHE_TTL_HOURS = 24           # max age of the local price cache
DEFAULT_ITERATIONS = 2000      # Monte-Carlo iterations (spec: 1000-2000)
RNG_SEED = 42                  # reproducible runs for team demos and tests

# T3 - AWS Pricing Calculator public pricing file, per region / OS
CALCULATOR_URL_TEMPLATE = (
    "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/"
    "ec2-ondemand-without-sec-sel/{location}/Linux/index.json?timestamp={ts}"
)
# Region code -> "Location" label used by the calculator files
REGION_LOCATIONS = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-central-1": "EU (Frankfurt)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "sa-east-1": "South America (Sao Paulo)",
    "ca-central-1": "Canada (Central)",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# T5 - static fallback table, USD/hour on-demand Linux shared tenancy.
# Used only when every live tier fails; keeps demos and tests deterministic.
FALLBACK_PRICING = {
    "t2.micro": 0.0116, "t2.small": 0.023, "t2.medium": 0.0464,
    "t3.nano": 0.0052, "t3.micro": 0.0104, "t3.small": 0.0208,
    "t3.medium": 0.0416, "t3.large": 0.0832, "t3.xlarge": 0.1664,
    "m5.large": 0.096, "m5.xlarge": 0.192, "m5.2xlarge": 0.384,
    "m6i.large": 0.096, "m6i.xlarge": 0.192,
    "c5.large": 0.085, "c5.xlarge": 0.17, "c5.2xlarge": 0.34,
    "r5.large": 0.126, "r5.xlarge": 0.252, "r5.2xlarge": 0.504,
    "db.t3.medium": 0.068, "db.m5.large": 0.171, "db.r5.large": 0.24,
}
DEFAULT_FALLBACK_RATE = 0.05  # sensible guess for an unknown instance type

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _request_json(url: str, timeout: int = 25) -> Optional[dict]:
    """GET a URL and parse JSON (handles gzip). Returns None on any failure."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "cloud-migration-planner-cost-module/1.0",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Referer": "https://calculator.aws/",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _load_fresh_cache(region: str) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"aws_prices_{region}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        # Reject the cache if it is older than the TTL
        if time.time() - float(blob.get("fetched_at", 0)) > CACHE_TTL_HOURS * 3600:
            return None
        return blob
    except Exception:
        return None


def _read_any_cache(region: str) -> Optional[Tuple[Dict[str, float], str]]:
    """Read the cache regardless of age (used when live tiers all fail)."""
    path = os.path.join(CACHE_DIR, f"aws_prices_{region}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return blob["prices_usd_per_hour"], blob.get("source", "unknown")
    except Exception:
        return None


def _save_cache(region: str, prices: Dict[str, float], source: str) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        blob = {
            "fetched_at": time.time(),
            "source": source,
            "region": region,
            "prices_usd_per_hour": prices,
        }
        path = os.path.join(CACHE_DIR, f"aws_prices_{region}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        os.replace(tmp, path)  # atomic write
    except Exception:
        pass  # cache is best-effort


def _extract_hourly_from_query_api_item(item: dict) -> Optional[float]:
    """Pull the hourly USD rate out of a Price List Query API product item."""
    try:
        terms = item["terms"]["OnDemand"]
        offer_key = next(iter(terms))
        dims = terms[offer_key]["priceDimensions"]
        dim = next(iter(dims.values()))
        usd = dim["pricePerUnit"]["USD"]
        if dim.get("unit") == "Hrs":
            return float(usd)
        return None  # explicit: refuse non-hourly units
    except (KeyError, IndexError, StopIteration, TypeError, ValueError):
        return None


def _parse_calculator_file(data: dict) -> Dict[str, float]:
    """Flatten a calculator pricing file to {instance_type: hourly_usd}.

    If several SKU variants exist for one type (e.g. different license
    models) the cheapest on-demand variant is kept, which matches the
    'No License required / Linux shared' selection the rest of the module
    assumes.
    """
    prices: Dict[str, float] = {}
    for region_blob in data.get("regions", {}).values():
        for entry in region_blob.values():
            itype = entry.get("Instance Type")
            price = entry.get("price")
            if not itype or price is None:
                continue
            try:
                p = float(price)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            if itype not in prices or p < prices[itype]:
                prices[itype] = p
    return prices


# ----------------------------------------------------------------------------
# Safe numeric parsing helpers
# ----------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float; return `default` for None, blanks, non-numerics, NaN/inf."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce to int (truncating); `default` for None/non-numeric/NaN/inf.

    Parses through float first so "7.0" is accepted and 7.9 truncates to 7.
    """
    parsed = _safe_float(value, float("nan"))
    return default if math.isnan(parsed) else int(parsed)


# ----------------------------------------------------------------------------
# Pricing resolver
# ----------------------------------------------------------------------------


class PricingResolver:
    """Resolves instance-type -> live hourly USD rate via the tier chain."""

    def __init__(self, region: str = "us-east-1", verbose: bool = False):
        self.region = region
        self.verbose = verbose
        self._bulk_prices: Optional[Dict[str, float]] = None
        self._bulk_source: Optional[str] = None
        self._unsigned_boto3_dead = False  # memoized: stop retrying a tier that cannot work here

    def _log(self, msg: str) -> None:
        if getattr(self, "verbose", False):
            print(f"[pricing] {msg}", file=sys.stderr)

    # -- T1/T2: boto3 -------------------------------------------------------
    def _try_boto3(self, instance_type: str, unsigned: bool) -> Optional[float]:
        try:
            import boto3
            from botocore import UNSIGNED  # sentinel object, not the string "UNSIGNED"
            from botocore.config import Config
        except ImportError:
            return None
        try:
            client = boto3.client(
                "pricing",
                region_name="us-east-1",  # pricing API lives in us-east-1
                config=Config(
                    signature_version=UNSIGNED if unsigned else "s3v4",
                    retries={"max_attempts": 1},
                    connect_timeout=5,
                    read_timeout=15,
                ),
            )
            # Filter by the resolver's region so the returned rate matches the
            # location being planned (the Query API otherwise spans all regions).
            filters = [
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            ]
            location = REGION_LOCATIONS.get(self.region)
            if location:
                filters.append({"Type": "TERM_MATCH", "Field": "location", "Value": location})
            resp = client.get_products(ServiceCode="AmazonEC2", Filters=filters, MaxResults=1)
            if not resp.get("PriceList"):
                return None
            item = json.loads(resp["PriceList"][0])
            rate = _extract_hourly_from_query_api_item(item)
            self._log(f"boto3 (unsigned={unsigned}) {instance_type} -> {rate}")
            return rate
        except Exception as exc:  # noqa: BLE001 - fallback chain by design
            self._log(f"boto3 (unsigned={unsigned}) failed: {type(exc).__name__}")
            if unsigned:
                self._unsigned_boto3_dead = True  # don't retry this tier for every app
            return None

    # -- T3/T4: public bulk file + local cache ------------------------------
    def _get_bulk_prices(self, refresh: bool = False) -> Dict[str, float]:
        """Load (or download) the full region price map. Cached locally."""
        if self._bulk_prices is not None and not refresh:
            return self._bulk_prices

        cache = None if refresh else _load_fresh_cache(self.region)
        if cache:
            self._bulk_prices = cache["prices_usd_per_hour"]
            self._bulk_source = f"cache ({cache.get('source', 'unknown')})"
            self._log(f"cache hit: {len(self._bulk_prices)} prices")
            return self._bulk_prices

        location = REGION_LOCATIONS.get(self.region, "")
        if location:
            url = CALCULATOR_URL_TEMPLATE.format(
                location=urllib.request.quote(location),
                ts=int(time.time() * 1000),
            )
            data = _request_json(url)
            prices = _parse_calculator_file(data) if data else {}
            if prices:
                self._bulk_prices = prices
                self._bulk_source = "aws-pricing-calculator-public-file"
                _save_cache(self.region, prices, self._bulk_source)
                self._log(f"downloaded {len(prices)} prices for {self.region}")
                return prices

        # T4: a stale cache is better than nothing
        stale = _read_any_cache(self.region)
        if stale:
            self._bulk_prices = stale[0]
            self._bulk_source = f"stale cache ({stale[1]})"
            return self._bulk_prices
        return {}

    # -- T5 -----------------------------------------------------------------
    @staticmethod
    def _static_fallback(instance_type: str) -> float:
        if instance_type in FALLBACK_PRICING:
            return FALLBACK_PRICING[instance_type]
        # Size-based guess for unknown types: scale a base rate by size/family
        size = instance_type.split(".")[-1] if "." in instance_type else "medium"
        size_factor = {"nano": 0.125, "micro": 0.25, "small": 0.5, "medium": 1.0,
                       "large": 2.0, "xlarge": 4.0, "2xlarge": 8.0}.get(size, 1.0)
        family = instance_type.split(".")[0] if "." in instance_type else ""
        family_factor = {"t2": 0.9, "t3": 1.0, "c5": 2.04, "m5": 2.3, "r5": 3.0}.get(family, 1.2)
        return round(DEFAULT_FALLBACK_RATE * size_factor * family_factor, 4)

    # -- public API ----------------------------------------------------------
    def get_hourly_rate(self, instance_type: str, refresh: bool = False) -> Tuple[float, str]:
        """Return (hourly_usd, source_tier) trying every tier in order."""
        itype = (instance_type or "").strip()
        if not itype:
            return self._static_fallback("t3.medium"), "fallback-static"

        # T1: signed boto3 - only worth trying when credentials exist
        try:
            from botocore.credentials import EnvProvider
            if EnvProvider().load():
                rate = self._try_boto3(itype, unsigned=False)
                if rate:
                    return rate, "aws-price-list-query-api"
        except Exception:
            pass

        # T2: anonymous boto3 (works in environments that allow it)
        if not self._unsigned_boto3_dead:
            rate = self._try_boto3(itype, unsigned=True)
            if rate:
                return rate, "aws-price-list-query-api-anonymous"

        # T3/T4: bulk public file with local cache
        bulk = self._get_bulk_prices(refresh=refresh)
        if itype in bulk:
            return bulk[itype], self._bulk_source or "aws-pricing-calculator-public-file"

        # T5: static fallback
        return self._static_fallback(itype), "fallback-static"

    def refresh_cache(self) -> int:
        """Force a re-download; returns the number of prices fetched."""
        self._bulk_prices = None
        return len(self._get_bulk_prices(refresh=True))


# ----------------------------------------------------------------------------
# Monte-Carlo cost simulator
# ----------------------------------------------------------------------------

# Amazon EBS gp3 storage list price, USD per GB-month, by AWS region.
# Baseline figures from aws.amazon.com/ebs/pricing; regions not listed fall
# back to DEFAULT_EBS_GB_MONTH.
EBS_GB_MONTH_BY_REGION: Dict[str, float] = {
    "us-east-1": 0.080, "us-east-2": 0.080, "us-west-1": 0.095, "us-west-2": 0.080,
    "eu-west-1": 0.084, "eu-west-2": 0.086, "eu-central-1": 0.086,
    "ap-south-1": 0.088, "ap-southeast-1": 0.088, "ap-southeast-2": 0.096,
    "ap-northeast-1": 0.088, "sa-east-1": 0.110, "ca-central-1": 0.085,
}
DEFAULT_EBS_GB_MONTH = 0.080  # for regions missing from the table


def get_ebs_rate(region: str = "us-east-1") -> float:
    """Region-aware EBS gp3 storage rate, USD per GB-month."""
    return EBS_GB_MONTH_BY_REGION.get(region, DEFAULT_EBS_GB_MONTH)


def run_monte_carlo_simulation(
    hourly_rate: float,
    storage_gb: float,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = RNG_SEED,
    ebs_rate: Optional[float] = None,
    region: str = "us-east-1",
) -> Dict[str, float]:
    """Vectorized Monte-Carlo simulation of the monthly cost.

    Randomized inputs per iteration:
      - compute utilization multiplier ~ Normal(1.0, 0.20), floored at 0.5
        (models fluctuating traffic / burst usage on a fixed instance)
      - storage multiplier ~ Uniform(1.0, 1.30)  (0-30% data growth)

    Storage is priced with the region-aware EBS rate: pass `ebs_rate` directly
    (USD per GB-month) or let it be looked up from `region` via get_ebs_rate().

    Returns 5th percentile / mean / 95th percentile / std-dev in USD.
    """
    rng = np.random.default_rng(seed)

    compute_mult = rng.normal(loc=1.0, scale=0.20, size=iterations)
    compute_mult = np.maximum(compute_mult, 0.5)  # never below 50% utilization
    storage_mult = rng.uniform(low=1.0, high=1.30, size=iterations)

    storage_rate = ebs_rate if ebs_rate is not None else get_ebs_rate(region)
    base_compute = hourly_rate * HOURS_PER_MONTH
    total = base_compute * compute_mult + (storage_gb * storage_rate) * storage_mult

    return {
        "low_5th_percentile_usd": round(float(np.percentile(total, 5)), 2),
        "expected_mean_usd": round(float(np.mean(total)), 2),
        "high_95th_percentile_usd": round(float(np.percentile(total, 95)), 2),
        "std_dev_usd": round(float(np.std(total)), 2),
    }


# ----------------------------------------------------------------------------
# Risk scoring
# ----------------------------------------------------------------------------


def calculate_risk_score(app_data: Dict[str, Any]) -> Dict[str, Any]:
    """Explainable weighted risk model returning a 0-100 score.

    Weights: tech age 30 | dependencies 30 | compliance 25 | deprecated 15
    (+ optional criticality bonus, capped at 100 overall).
    Every contribution is reported so the dashboard/copilot can explain 'why'.
    """
    points = 0
    factors: List[Dict[str, Any]] = []

    def add(category: str, pts: int, detail: str) -> None:
        nonlocal points
        if pts <= 0:
            return
        points += pts
        factors.append({"category": category, "points": pts, "detail": detail})

    # 1) Tech age (max 30) - safe parsing: nulls / "unknown" / NaN -> 0
    age = _safe_float(app_data.get("app_age_years"), 0.0)
    if age > 10:
        add("tech_age", 30, f"Legacy codebase: {age:g} years old (>10)")
    elif age > 5:
        add("tech_age", 15, f"Aging stack: {age:g} years old (5-10)")
    else:
        add("tech_age", 0, f"Modern stack: {age:g} years old")

    # 2) Dependency coupling (max 30) - safe parsing: nulls / "unknown" -> 0
    deps = _safe_int(app_data.get("dependency_count"), 0)
    if deps >= 5:
        add("dependencies", 30, f"High coupling: {deps} direct dependencies (>=5)")
    elif deps >= 2:
        add("dependencies", 15, f"Moderate coupling: {deps} dependencies (2-4)")
    else:
        add("dependencies", 0, f"Loose coupling: {deps} dependencies")

    # 3) Compliance (max 25)
    if app_data.get("has_compliance_data"):
        frameworks = app_data.get("compliance_frameworks") or ["HIPAA/PCI/GDPR"]
        add("compliance", 25, f"Handles regulated data ({', '.join(map(str, frameworks))})")
    else:
        add("compliance", 0, "No regulated-data workload")

    # 4) Deprecated runtime (max 15)
    if app_data.get("is_deprecated_tech"):
        add("deprecated_tech", 15,
            f"Deprecated runtime: {app_data.get('tech_stack') or 'legacy environment'}")
    else:
        add("deprecated_tech", 0, f"Supported runtime: {app_data.get('tech_stack') or 'n/a'}")

    # 5) Optional criticality bonus (score is capped at 100 anyway)
    crit = str(app_data.get("criticality") or "").lower()
    if crit in ("mission-critical", "critical"):
        add("criticality", 10, "Mission-critical workload - outage is high-impact")
    elif crit == "business-critical":
        add("criticality", 5, "Business-critical workload")

    score = int(min(points, 100))
    if score >= 65:
        level, badge = "HIGH", "red"
    elif score >= 35:
        level, badge = "MEDIUM", "amber"
    else:
        level, badge = "LOW", "green"

    return {
        "risk_score": score,
        "risk_level": level,
        "risk_badge_color": badge,
        "risk_factors": factors,
        "weights_used": {
            "tech_age_max": 30, "dependencies_max": 30,
            "compliance_max": 25, "deprecated_tech_max": 15,
            "criticality_bonus_max": 10,
        },
    }


# ----------------------------------------------------------------------------
# Application analyzer + portfolio mode
# ----------------------------------------------------------------------------


def analyze_application(app_data: Dict[str, Any],
                        resolver: Optional[PricingResolver] = None,
                        iterations: int = DEFAULT_ITERATIONS) -> Dict[str, Any]:
    """Full pipeline for one application -> JSON-ready dict."""
    resolver = resolver or PricingResolver()

    instance_type = app_data.get("recommended_instance") or "t3.medium"
    storage_gb = _safe_float(app_data.get("storage_gb", 50), 50.0)

    hourly_rate, price_source = resolver.get_hourly_rate(instance_type)
    cost = run_monte_carlo_simulation(hourly_rate, storage_gb, iterations=iterations,
                                      region=resolver.region)
    risk = calculate_risk_score(app_data)

    low = cost["low_5th_percentile_usd"]
    exp = cost["expected_mean_usd"]
    high = cost["high_95th_percentile_usd"]

    drivers = "; ".join(
        f"{f['category']} +{f['points']}"
        for f in risk["risk_factors"] if f["points"] > 0
    ) or "no significant drivers"

    return {
        "app_id": app_data.get("app_id", "UNKNOWN"),
        "app_name": app_data.get("app_name", "Unnamed Application"),
        "target_aws_instance": instance_type,
        "aws_region": resolver.region,
        "storage_gb": storage_gb,
        "hourly_rate_usd": hourly_rate,
        "price_source": price_source,
        "cost_simulation_monthly": {
            "low_5th_percentile_usd": low,
            "expected_mean_usd": exp,
            "high_95th_percentile_usd": high,
            "std_dev_usd": cost["std_dev_usd"],
            "confidence_interval_percent": 90,
            "iterations": iterations,
            "assumptions": {
                "hours_per_month": HOURS_PER_MONTH,
                "compute_utilization": "Normal(mean=1.0, sd=0.20), floored at 0.5",
                "storage_growth": "Uniform(1.0, 1.30)",
                "ebs_usd_per_gb_month": get_ebs_rate(resolver.region),
                "ebs_rate_source": "region-aware table",
            },
        },
        "risk_assessment": risk,
        "summary_for_copilot": (
            f"{app_data.get('app_name', 'This app')} runs on {instance_type} "
            f"at ${hourly_rate}/hr (price source: {price_source}). Predicted monthly "
            f"cost is ${low}-${high} (90% CI, expected ${exp}) with {storage_gb:g} GB storage. "
            f"Migration risk is {risk['risk_level']} ({risk['risk_score']}/100). "
            f"Drivers: {drivers}."
        ),
    }


def analyze_portfolio(apps: List[Dict[str, Any]],
                      iterations: int = DEFAULT_ITERATIONS,
                      verbose: bool = False) -> List[Dict[str, Any]]:
    """Analyze a whole portfolio, sharing one resolver (one bulk download)."""
    resolver = PricingResolver(verbose=verbose)
    return [analyze_application(app, resolver=resolver, iterations=iterations)
            for app in apps]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _demo_portfolio() -> List[Dict[str, Any]]:
    return [
        {
            "app_id": "APP_001", "app_name": "Legacy Core Billing",
            "recommended_instance": "m5.large", "storage_gb": 250,
            "app_age_years": 12, "dependency_count": 7,
            "has_compliance_data": True, "compliance_frameworks": ["PCI"],
            "is_deprecated_tech": True, "tech_stack": "Java 6 / WebLogic",
            "criticality": "mission-critical",
        },
        {
            "app_id": "APP_002", "app_name": "User Notification Service",
            "recommended_instance": "t3.medium", "storage_gb": 30,
            "app_age_years": 2, "dependency_count": 1,
            "has_compliance_data": False, "is_deprecated_tech": False,
            "tech_stack": "Python 3.11 / FastAPI", "criticality": "low",
        },
        {
            "app_id": "APP_003", "app_name": "Analytics Warehouse",
            "recommended_instance": "r5.2xlarge", "storage_gb": 900,
            "app_age_years": 6, "dependency_count": 3,
            "has_compliance_data": False, "is_deprecated_tech": False,
            "tech_stack": "PostgreSQL 12", "criticality": "business-critical",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive Cost & Risk Simulator (Role 6)")
    parser.add_argument("--input", help="JSON file with a list of app dicts")
    parser.add_argument("--app", help="only analyze the app with this app_id")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--refresh-pricing", action="store_true",
                        help="force-refresh the AWS price cache")
    parser.add_argument("--out", help="write JSON result to this file")
    parser.add_argument("--quiet", action="store_true", help="no progress logs")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        apps = payload if isinstance(payload, list) else payload.get("applications", [])
    else:
        apps = _demo_portfolio()

    if args.app:
        apps = [a for a in apps if a.get("app_id") == args.app]
        if not apps:
            parser.error(f"app_id {args.app!r} not found")

    resolver = PricingResolver(region=args.region, verbose=not args.quiet)
    if args.refresh_pricing:
        n = resolver.refresh_cache()
        print(f"[pricing] refreshed cache with {n} instance prices", file=sys.stderr)

    results = [analyze_application(a, resolver=resolver, iterations=args.iterations)
               for a in apps]

    output = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aws_region": args.region,
        "app_count": len(results),
        "applications": results,
    }

    text = json.dumps(output, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
