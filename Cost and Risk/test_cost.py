"""
Tests for the Predictive Cost & Risk Simulator (Role 6).

Run with:  python -m pytest test_cost.py -v
   or:     python test_cost.py          (plain, no pytest needed)

Network tiers are bypassed via monkeypatched resolver stubs so tests are
fast, deterministic and offline-safe. One optional test exercises the real
public pricing endpoint and is skipped automatically if there is no internet.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import urllib.error

import numpy as np
import pytest

import cost
from cost import (
    FALLBACK_PRICING,
    PricingResolver,
    analyze_application,
    analyze_portfolio,
    calculate_risk_score,
    default_recommended_instance,
    fetch_backend_applications,
    normalize_app_payload,
    parse_csv_portfolio,
    run_monte_carlo_simulation,
)


# ----------------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------------

@pytest.fixture
def offline_resolver(monkeypatch):
    """A resolver whose live tiers are stubbed out -> always static fallback."""
    r = PricingResolver()
    monkeypatch.setattr(r, "_try_boto3", lambda *a, **k: None)
    monkeypatch.setattr(r, "_get_bulk_prices", lambda *a, **k: {})
    return r


@pytest.fixture
def stubbed_resolver(monkeypatch):
    """Resolver returning a fixed rate, as if a live tier answered."""
    r = PricingResolver()
    monkeypatch.setattr(r, "get_hourly_rate", lambda itype, refresh=False: (0.0416, "test-stub"))
    return r


LOW_APP = {
    "app_id": "APP_LOW", "app_name": "Shiny Microservice",
    "recommended_instance": "t3.medium", "storage_gb": 30,
    "app_age_years": 2, "dependency_count": 1,
    "has_compliance_data": False, "is_deprecated_tech": False,
}

HIGH_APP = {
    "app_id": "APP_HIGH", "app_name": "Legacy Core Billing",
    "recommended_instance": "m5.large", "storage_gb": 250,
    "app_age_years": 12, "dependency_count": 7,
    "has_compliance_data": True, "compliance_frameworks": ["PCI"],
    "is_deprecated_tech": True, "tech_stack": "Java 6 / WebLogic",
    "criticality": "mission-critical",
}


# ----------------------------------------------------------------------------
# Pricing resolution
# ----------------------------------------------------------------------------

class TestPricing:
    def test_static_fallback_known_type(self, offline_resolver):
        rate, source = offline_resolver.get_hourly_rate("t3.medium")
        assert rate == FALLBACK_PRICING["t3.medium"] == 0.0416
        assert source == "fallback-static"

    def test_static_fallback_unknown_type_is_positive(self, offline_resolver):
        rate, source = offline_resolver.get_hourly_rate("x9z.superlarge")
        assert rate > 0
        assert source == "fallback-static"

    def test_static_fallback_scales_with_size(self, offline_resolver):
        small, _ = offline_resolver.get_hourly_rate("m5.large")
        big, _ = offline_resolver.get_hourly_rate("m5.2xlarge")
        assert big > small

    def test_blank_instance_type_does_not_crash(self, offline_resolver):
        rate, _ = offline_resolver.get_hourly_rate("")
        assert rate > 0

    def test_live_tier_wins_over_fallback(self, monkeypatch):
        r = PricingResolver()
        monkeypatch.setattr(r, "_try_boto3", lambda it, unsigned: 0.096 if unsigned else None)  # anonymous tier hits
        monkeypatch.setattr(r, "_get_bulk_prices", lambda *a, **k: {})
        monkeypatch.setattr(cost, "PricingResolver", lambda *a, **k: r)  # T1 cred-check constructs its own resolver
        rate, source = r.get_hourly_rate("m5.large")
        assert rate == 0.096
        assert source == "aws-price-list-query-api-anonymous"

    def test_boto3_missing_is_tolerated(self, monkeypatch):
        # Simulate boto3 not installed at all
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("boto3") or name.startswith("botocore"):
                raise ImportError("no boto3 in this sandbox")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        r = PricingResolver()
        # Fully sever the live tiers, including the T1 credential check
        monkeypatch.setattr(r, "_try_boto3", lambda *a, **k: None)
        monkeypatch.setattr(r, "_get_bulk_prices", lambda *a, **k: {})
        monkeypatch.setattr(cost, "PricingResolver", lambda *a, **k: r)
        # Should fall through to the static table without raising
        rate, source = r.get_hourly_rate("t3.medium")
        assert rate == FALLBACK_PRICING["t3.medium"]
        assert source == "fallback-static"

    @pytest.mark.optional
    def test_real_public_pricing_endpoint(self):
        """Optional: hits the real anonymous AWS pricing file (needs internet)."""
        r = PricingResolver(region="us-east-1")
        rate, source = r.get_hourly_rate("t3.medium")
        assert rate > 0
        # If the network worked, we must NOT be on the static fallback tier
        if "fallback" not in source:
            assert abs(rate - 0.0416) < 0.01  # t3.medium on-demand, us-east-1
            assert "aws" in source or "cache" in source

    def test_parse_calculator_file(self):
        data = {"regions": {"US East (N. Virginia)": {
            "a": {"Instance Type": "t3.medium", "price": "0.0416"},
            "b": {"Instance Type": "t3.medium", "price": "0.9999"},  # pricier variant
            "c": {"Instance Type": "bad", "price": "not-a-number"},
            "d": {"Instance Type": "zero", "price": "0"},
        }}}
        prices = cost._parse_calculator_file(data)
        assert prices == {"t3.medium": 0.0416}

    # -- boto3 Query API region filter -------------------------------------

    @staticmethod
    def _fake_boto3(monkeypatch, capture):
        """Install fake boto3/botocore modules; get_products captures Filters."""

        class FakeConfig:
            def __init__(self, **kwargs):
                pass

        class FakeClient:
            def get_products(self, **kwargs):
                capture["filters"] = kwargs["Filters"]
                return {"PriceList": []}  # empty -> resolver falls through

        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = lambda *a, **k: FakeClient()
        fake_botocore = types.ModuleType("botocore")
        fake_botocore.UNSIGNED = object()  # sentinel like the real one
        fake_config_mod = types.ModuleType("botocore.config")
        fake_config_mod.Config = FakeConfig

        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
        monkeypatch.setitem(sys.modules, "botocore.config", fake_config_mod)

    def test_boto3_query_includes_region_location_filter(self, monkeypatch):
        capture = {}
        self._fake_boto3(monkeypatch, capture)
        r = PricingResolver(region="eu-west-1")
        r._try_boto3("m5.large", unsigned=True)  # returns None (empty result), but records the query
        fields = {f["Field"]: f["Value"] for f in capture["filters"]}
        assert fields["location"] == "EU (Ireland)"
        assert fields["instanceType"] == "m5.large"

    def test_boto3_unknown_region_omits_location_filter(self, monkeypatch):
        capture = {}
        self._fake_boto3(monkeypatch, capture)
        r = PricingResolver(region="mars-1")  # not in REGION_LOCATIONS
        r._try_boto3("m5.large", unsigned=True)
        fields = {f["Field"] for f in capture["filters"]}
        assert "location" not in fields  # unknown region -> no bogus filter


# ----------------------------------------------------------------------------
# Monte-Carlo simulation
# ----------------------------------------------------------------------------

class TestMonteCarlo:
    def test_percentile_ordering(self):
        out = run_monte_carlo_simulation(0.0416, 30, iterations=1500)
        assert out["low_5th_percentile_usd"] < out["expected_mean_usd"] < out["high_95th_percentile_usd"]

    def test_reproducible_with_same_seed(self):
        a = run_monte_carlo_simulation(0.10, 100, iterations=800, seed=7)
        b = run_monte_carlo_simulation(0.10, 100, iterations=800, seed=7)
        assert a == b

    def test_costs_scale_with_hourly_rate(self):
        cheap = run_monte_carlo_simulation(0.01, 50, iterations=1000)
        pricey = run_monte_carlo_simulation(1.00, 50, iterations=1000)
        assert pricey["expected_mean_usd"] > cheap["expected_mean_usd"] * 50

    def test_costs_scale_with_storage(self):
        small = run_monte_carlo_simulation(0.05, 10, iterations=1000)
        big = run_monte_carlo_simulation(0.05, 1000, iterations=1000)
        assert big["expected_mean_usd"] > small["expected_mean_usd"]

    def test_zero_rate_gives_storage_only_costs(self):
        out = run_monte_carlo_simulation(0.0, 100, iterations=500)
        assert 0 < out["low_5th_percentile_usd"] <= out["expected_mean_usd"]

    def test_extreme_rate_is_finite(self):
        out = run_monte_carlo_simulation(1e6, 1e6, iterations=200)
        assert np.isfinite(out["expected_mean_usd"])
        assert out["expected_mean_usd"] > 0

    def test_iterations_respected(self):
        # 5 iterations -> degenerate but valid output, no crash
        out = run_monte_carlo_simulation(0.05, 10, iterations=5)
        assert set(out) == {"low_5th_percentile_usd", "expected_mean_usd",
                            "high_95th_percentile_usd", "std_dev_usd"}


# ----------------------------------------------------------------------------
# Risk scoring
# ----------------------------------------------------------------------------

class TestRisk:
    def test_low_risk_app(self):
        r = calculate_risk_score(LOW_APP)
        assert r["risk_score"] <= 34
        assert r["risk_level"] == "LOW"
        assert r["risk_badge_color"] == "green"

    def test_high_risk_app(self):
        r = calculate_risk_score(HIGH_APP)
        assert r["risk_score"] == 100  # 30+30+25+15+10, capped
        assert r["risk_level"] == "HIGH"
        assert r["risk_badge_color"] == "red"

    def test_score_bounds(self):
        for app in (LOW_APP, HIGH_APP, {}):
            r = calculate_risk_score(app)
            assert 0 <= r["risk_score"] <= 100

    def test_empty_app_is_low_risk(self):
        r = calculate_risk_score({})
        assert r["risk_score"] == 0
        assert r["risk_level"] == "LOW"

    def test_explainability_factors_present(self):
        r = calculate_risk_score(HIGH_APP)
        cats = {f["category"] for f in r["risk_factors"] if f["points"] > 0}
        assert cats == {"tech_age", "dependencies", "compliance",
                        "deprecated_tech", "criticality"}

    def test_boundary_values(self):
        # age exactly 10 -> not ">10" -> 15 pts band; deps exactly 5 -> 30 pts
        r = calculate_risk_score({"app_age_years": 10, "dependency_count": 5})
        pts = {f["category"]: f["points"] for f in r["risk_factors"]}
        assert pts["tech_age"] == 15
        assert pts["dependencies"] == 30

    def test_frameworks_flow_into_detail(self):
        r = calculate_risk_score({"has_compliance_data": True,
                                  "compliance_frameworks": ["HIPAA"]})
        detail = next(f for f in r["risk_factors"]
                      if f["category"] == "compliance")["detail"]
        assert "HIPAA" in detail

    # -- safe numeric parsing (regression: nulls / "unknown" must not crash) --

    def test_safe_float_helper(self):
        assert cost._safe_float("12.5") == 12.5
        assert cost._safe_float(7) == 7.0
        assert cost._safe_float(None) == 0.0
        assert cost._safe_float("unknown", 3.0) == 3.0
        assert cost._safe_float("") == 0.0
        assert cost._safe_float(float("nan")) == 0.0
        assert cost._safe_float(float("inf")) == 0.0
        assert cost._safe_float("12,5") == 0.0  # not a valid float

    def test_safe_int_helper(self):
        assert cost._safe_int("7") == 7
        assert cost._safe_int("7.9") == 7  # parses via float, then truncates
        assert cost._safe_int(4.2) == 4
        assert cost._safe_int(None, 5) == 5
        assert cost._safe_int("unknown", 2) == 2
        assert cost._safe_int(float("nan"), 1) == 1
        assert cost._safe_int(float("inf"), 9) == 9

    def test_dirty_numeric_fields_do_not_crash(self):
        dirty = {
            "app_age_years": "unknown",
            "dependency_count": None,
            "has_compliance_data": False,
        }
        r = calculate_risk_score(dirty)  # must not raise
        assert r["risk_score"] == 0
        assert r["risk_level"] == "LOW"
        assert all(f["points"] > 0 for f in r["risk_factors"])  # zero-point factors are omitted

    def test_numeric_strings_are_accepted(self):
        r = calculate_risk_score({"app_age_years": "12", "dependency_count": "7"})
        pts = {f["category"]: f["points"] for f in r["risk_factors"]}
        assert pts["tech_age"] == 30
        assert pts["dependencies"] == 30


# ----------------------------------------------------------------------------
# End-to-end application analysis + JSON contract
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Region-aware storage pricing
# ----------------------------------------------------------------------------

class TestStoragePricing:
    def test_no_hardcoded_ebs_constant_left_behind(self):
        assert not hasattr(cost, "EBS_GB_MONTH"), "EBS_GB_MONTH was supposed to be removed"

    def test_region_aware_rates(self):
        assert cost.get_ebs_rate("us-east-1") == 0.080
        assert cost.get_ebs_rate("sa-east-1") > cost.get_ebs_rate("us-east-1")
        assert cost.get_ebs_rate("not-a-region") == cost.DEFAULT_EBS_GB_MONTH

    def test_simulation_uses_region_rate(self):
        cheap = run_monte_carlo_simulation(0.0, 1000, iterations=500, region="us-east-1")
        pricey = run_monte_carlo_simulation(0.0, 1000, iterations=500, region="sa-east-1")
        assert pricey["expected_mean_usd"] > cheap["expected_mean_usd"]

    def test_explicit_ebs_rate_overrides_region(self):
        a = run_monte_carlo_simulation(0.0, 1000, iterations=500,
                                       region="us-east-1", ebs_rate=0.50)
        b = run_monte_carlo_simulation(0.0, 1000, iterations=500,
                                       region="sa-east-1", ebs_rate=0.50)
        assert a == b  # explicit parameter wins over the region lookup

    def test_analyze_application_reports_region_rate(self, monkeypatch):
        r = PricingResolver(region="sa-east-1")
        monkeypatch.setattr(r, "_try_boto3", lambda *a, **k: None)
        monkeypatch.setattr(r, "_get_bulk_prices", lambda *a, **k: {})
        out = analyze_application({"app_id": "Z"}, resolver=r)
        assumptions = out["cost_simulation_monthly"]["assumptions"]
        assert assumptions["ebs_usd_per_gb_month"] == cost.get_ebs_rate("sa-east-1")
        assert assumptions["ebs_rate_source"] == "region-aware table"


class TestAnalyzeApplication:
    def test_json_serializable(self, stubbed_resolver):
        out = analyze_application(HIGH_APP, resolver=stubbed_resolver)
        blob = json.dumps(out)  # must not raise
        assert isinstance(blob, str)

    def test_required_keys_for_frontend(self, stubbed_resolver):
        out = analyze_application(HIGH_APP, resolver=stubbed_resolver)
        for key in ("app_id", "app_name", "target_aws_instance", "hourly_rate_usd",
                    "price_source", "cost_simulation_monthly", "risk_assessment",
                    "summary_for_copilot"):
            assert key in out, f"missing key: {key}"

    def test_cost_block_shape(self, stubbed_resolver):
        c = analyze_application(HIGH_APP, resolver=stubbed_resolver)["cost_simulation_monthly"]
        assert c["low_5th_percentile_usd"] < c["expected_mean_usd"] < c["high_95th_percentile_usd"]
        assert c["confidence_interval_percent"] == 90
        assert c["iterations"] == cost.DEFAULT_ITERATIONS
        assert "assumptions" in c

    def test_risk_block_shape(self, stubbed_resolver):
        r = analyze_application(HIGH_APP, resolver=stubbed_resolver)["risk_assessment"]
        assert r["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(r["risk_factors"], list) and r["risk_factors"]

    def test_copilot_summary_mentions_key_facts(self, stubbed_resolver):
        s = analyze_application(HIGH_APP, resolver=stubbed_resolver)["summary_for_copilot"]
        assert "m5.large" in s
        assert "risk is HIGH" in s
        assert "$" in s

    def test_defaults_for_sparse_input(self, offline_resolver):
        out = analyze_application({"app_id": "X"}, resolver=offline_resolver)
        assert out["target_aws_instance"] == "t3.medium"
        assert out["storage_gb"] == 50.0
        assert out["app_name"] == "Unnamed Application"


class TestPortfolio:
    def test_portfolio_shapes_and_shared_resolver(self, stubbed_resolver, monkeypatch):
        calls = {"n": 0}
        real_init = PricingResolver.__init__

        def counting_init(self, *a, **k):
            calls["n"] += 1
            real_init(self, *a, **k)

        monkeypatch.setattr(PricingResolver, "__init__", counting_init)
        monkeypatch.setattr(PricingResolver, "get_hourly_rate",
                            lambda self, itype, refresh=False: (0.0416, "test-stub"))
        results = analyze_portfolio([LOW_APP, HIGH_APP])
        assert len(results) == 2
        assert calls["n"] == 1  # one resolver constructed for the whole batch

    def test_portfolio_ids_preserved(self, stubbed_resolver):
        results = analyze_portfolio([LOW_APP, HIGH_APP])
        assert [r["app_id"] for r in results] == ["APP_LOW", "APP_HIGH"]


# ----------------------------------------------------------------------------
# Local price cache
# ----------------------------------------------------------------------------

class TestCache:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cost, "CACHE_DIR", str(tmp_path))
        cost._save_cache("us-east-1", {"t3.medium": 0.0416}, "unit-test")
        blob = cost._load_fresh_cache("us-east-1")
        assert blob["prices_usd_per_hour"]["t3.medium"] == 0.0416
        assert blob["source"] == "unit-test"

    def test_stale_cache_rejected_when_fresh_required(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cost, "CACHE_DIR", str(tmp_path))
        cost._save_cache("us-east-1", {"t3.medium": 0.0416}, "unit-test")
        # Backdate the timestamp beyond the TTL
        path = os.path.join(str(tmp_path), "aws_prices_us-east-1.json")
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        blob["fetched_at"] = time.time() - (cost.CACHE_TTL_HOURS + 1) * 3600
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        assert cost._load_fresh_cache("us-east-1") is None
        # ...but the stale reader can still use it as a last resort
        stale = cost._read_any_cache("us-east-1")
        assert stale is not None and stale[0]["t3.medium"] == 0.0416

    def test_corrupt_cache_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cost, "CACHE_DIR", str(tmp_path))
        path = os.path.join(str(tmp_path), "aws_prices_us-east-1.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        assert cost._load_fresh_cache("us-east-1") is None
        assert cost._read_any_cache("us-east-1") is None

    def test_bulk_tier_used_when_boto3_fails(self, monkeypatch):
        r = PricingResolver()
        monkeypatch.setattr(r, "_try_boto3", lambda *a, **k: None)
        monkeypatch.setattr(r, "_get_bulk_prices", lambda *a, **k: {"m5.large": 0.096})
        rate, source = r.get_hourly_rate("m5.large")
        assert rate == 0.096
        assert "calculator" in source


# ----------------------------------------------------------------------------
# Backend / CSV / fixture integration (live dataset payloads)
# ----------------------------------------------------------------------------

# Record shape produced by backend GET /applications (backend/app/models.py)
BACKEND_APP = {
    "id": "app-004",
    "name": "Legacy Batch Processor",
    "owner": "Operations",
    "technology": "COBOL / Mainframe",
    "criticality": "Low",
    "dependencies": ["app-001", "app-002", "app-003", "app-005"],
}

# Row shape produced by data/processed/application_portfolio_1000.csv
CSV_ROW = {
    "application_id": "APP001",
    "application_name": "incentivize holistic bandwidth",
    "cpu_usage": "0.76",
    "memory_usage": "0.64",
    "age_years": "15",
    "criticality": "Low",
    "compliance_flag": "1",
    "dependency_ids": "APP089,APP635,APP387",
    "dependency_count": "5",
}


class TestPayloadNormalization:
    def test_backend_record_maps_to_canonical_schema(self):
        app = normalize_app_payload(dict(BACKEND_APP))
        assert app["app_id"] == "app-004"
        assert app["app_name"] == "Legacy Batch Processor"
        assert app["tech_stack"] == "COBOL / Mainframe"
        assert app["dependency_count"] == 4  # derived from the dependency list

    def test_backend_record_sniffs_deprecated_cobol(self):
        app = normalize_app_payload(dict(BACKEND_APP))
        assert app["is_deprecated_tech"] is True

    def test_backend_record_without_signal_gets_default_instance(self):
        app = normalize_app_payload({"id": "app-001", "name": "Customer Portal"})
        assert app["recommended_instance"] == "t3.medium"

    def test_csv_row_maps_to_canonical_schema(self):
        app = normalize_app_payload(dict(CSV_ROW))
        assert app["app_id"] == "APP001"
        assert app["app_name"] == "incentivize holistic bandwidth"
        assert app["app_age_years"] == "15"
        assert app["dependency_count"] == "5"
        assert app["has_compliance_data"] is True  # "1" -> True

    def test_csv_row_derives_instance_from_utilization(self):
        app = normalize_app_payload(dict(CSV_ROW))  # cpu 0.76 -> 60-80% band
        assert app["recommended_instance"] == "m5.large"

    def test_very_heavy_row_derives_m5_xlarge(self):
        app = normalize_app_payload({"app_id": "APP100", "cpu_usage": "0.92",
                                     "memory_usage": "0.88"})
        assert app["recommended_instance"] == "m5.xlarge"

    def test_native_schema_passes_through_unchanged(self):
        app = normalize_app_payload(dict(HIGH_APP))
        assert app["recommended_instance"] == "m5.large"
        assert app["app_age_years"] == 12
        assert app["dependency_count"] == 7
        assert app["has_compliance_data"] is True

    def test_string_false_flag_is_not_compliant(self):
        app = normalize_app_payload({"app_id": "X", "compliance_flag": "false"})
        assert app["has_compliance_data"] is False

    def test_non_dict_payload_raises(self):
        with pytest.raises(TypeError):
            normalize_app_payload(["not", "a", "dict"])


class TestInstanceSizing:
    def test_no_signal_defaults_to_t3_medium(self):
        assert default_recommended_instance(None, None) == "t3.medium"

    def test_light_workload_gets_burstable(self):
        assert default_recommended_instance(0.19, 0.10) == "t3.small"

    def test_moderate_workload_gets_t3_medium(self):
        assert default_recommended_instance(0.50, 0.40) == "t3.medium"

    def test_heavy_workload_gets_memory_optimized(self):
        assert default_recommended_instance(0.90, 0.85) == "m5.xlarge"

    def test_percent_strings_are_understood(self):
        assert default_recommended_instance("90%", "10%") == "m5.xlarge"
        assert default_recommended_instance("20%", "5%") == "t3.small"

    def test_cost_module_utilization_bounds(self):
        assert cost._utilization("76%") == 0.76
        assert cost._utilization(0.64) == 0.64
        assert cost._utilization(90) == 0.90
        assert cost._utilization("junk") == 0.0


class TestBackendFetch:
    @staticmethod
    def _fake_urlopen(monkeypatch, body, status_ok=True):
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        def fake_urlopen(req, timeout=0, context=None):
            if not status_ok:
                raise urllib.error.URLError("connection refused")
            return FakeResp()

        monkeypatch.setattr(cost.urllib.request, "urlopen", fake_urlopen)

    def test_fetch_parses_bare_list(self, monkeypatch):
        self._fake_urlopen(monkeypatch, [dict(BACKEND_APP)])
        apps = fetch_backend_applications("http://127.0.0.1:8000")
        assert apps and apps[0]["id"] == "app-004"

    def test_fetch_parses_wrapped_dict(self, monkeypatch):
        self._fake_urlopen(monkeypatch, {"applications": [dict(BACKEND_APP)]})
        apps = fetch_backend_applications("http://127.0.0.1:8000")
        assert apps and apps[0]["name"] == "Legacy Batch Processor"

    def test_fetch_failure_returns_none(self, monkeypatch):
        self._fake_urlopen(monkeypatch, [], status_ok=False)
        assert fetch_backend_applications("http://127.0.0.1:9999") is None

    def test_fetch_bad_json_returns_none(self, monkeypatch):
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"{not json"

        monkeypatch.setattr(cost.urllib.request, "urlopen",
                            lambda req, timeout=0, context=None: FakeResp())
        assert fetch_backend_applications("http://127.0.0.1:8000") is None

    def test_fetch_empty_list_returns_none(self, monkeypatch):
        self._fake_urlopen(monkeypatch, [])
        assert fetch_backend_applications("http://127.0.0.1:8000") is None


class TestRiskBackendPayloads:
    def test_backend_record_scores_dependencies_from_list(self):
        r = calculate_risk_score(dict(BACKEND_APP))  # raw payload, no normalization
        pts = {f["category"]: f["points"] for f in r["risk_factors"] if f["points"] > 0}
        assert pts["dependencies"] == 15  # 4-item list -> moderate coupling (2-4)
        assert pts["deprecated_tech"] == 15  # COBOL sniffed from technology text

    def test_csv_row_scores_via_alias_fields(self):
        r = calculate_risk_score(dict(CSV_ROW))  # age_years / compliance_flag aliases
        pts = {f["category"]: f["points"] for f in r["risk_factors"] if f["points"] > 0}
        assert pts["tech_age"] == 30
        assert pts["dependencies"] == 30
        assert pts["compliance"] == 25

    def test_backend_criticality_low_adds_no_points(self):
        r = calculate_risk_score(dict(BACKEND_APP))
        cats = {f["category"] for f in r["risk_factors"] if f["points"] > 0}
        assert "criticality" not in cats  # "Low" is not a bonus tier


class TestAnalyzeBackendPayloads:
    def test_backend_record_end_to_end(self, offline_resolver):
        out = analyze_application(dict(BACKEND_APP), resolver=offline_resolver)
        assert out["app_id"] == "app-004"
        assert out["app_name"] == "Legacy Batch Processor"
        # Backend records carry no age/compliance fields, so the score reflects
        # only what the data supports: deps 15 + COBOL 15 = 30 -> LOW band
        assert out["risk_assessment"]["risk_score"] == 30
        assert out["risk_assessment"]["risk_level"] == "LOW"
        assert out["cost_simulation_monthly"]["expected_mean_usd"] > 0
        assert "Legacy Batch Processor" in out["summary_for_copilot"]

    def test_csv_row_end_to_end(self, offline_resolver):
        out = analyze_application(dict(CSV_ROW), resolver=offline_resolver)
        assert out["app_id"] == "APP001"
        assert out["target_aws_instance"] == "m5.large"  # cpu 0.76 -> 60-80% band
        assert out["risk_assessment"]["risk_score"] >= 75  # age 30 + deps 30 + compliance 25

    def test_output_stays_json_serializable(self, offline_resolver):
        for payload in (BACKEND_APP, CSV_ROW):
            blob = json.dumps(analyze_application(dict(payload), resolver=offline_resolver))
            assert isinstance(blob, str)


class TestFixtureFallback:
    def test_fixture_loads_and_analyzes(self, offline_resolver):
        apps, source = cost._default_portfolio()
        assert "fixture" in source
        assert [a["app_id"] for a in apps] == ["APP_001", "APP_002", "APP_003"]
        results = [analyze_application(a, resolver=offline_resolver) for a in apps]
        assert all(r["cost_simulation_monthly"]["expected_mean_usd"] > 0 for r in results)
        assert results[0]["risk_assessment"]["risk_level"] == "HIGH"  # APP_001 stays risky

    def test_backend_down_falls_back_to_fixture(self, monkeypatch, offline_resolver):
        # Simulate an unreachable backend, then confirm the pipeline still produces results
        assert fetch_backend_applications("http://127.0.0.1:9999") is None
        apps, source = cost._default_portfolio()
        assert "fixture" in source
        results = analyze_portfolio(apps)  # uses real resolver tiers -> must not crash
        assert len(results) == 3
        assert all(r["app_id"] for r in results)

    def test_missing_fixture_falls_back_to_demo(self, monkeypatch):
        monkeypatch.setattr(cost, "DEFAULT_FIXTURE_PATH", "Z:/definitely/missing.json")
        apps, source = cost._default_portfolio()
        assert "demo" in source
        assert [a["app_id"] for a in apps] == ["APP_001", "APP_002", "APP_003"]

    def test_csv_portfolio_parse(self, tmp_path):
        csv_file = tmp_path / "portfolio.csv"
        header = ("application_id,application_name,cpu_usage,memory_usage,age_years,"
                  "criticality,compliance_flag,dependency_ids,dependency_count")
        csv_file.write_text(
            header + "\n" + "APP001,holistic bandwidth,0.76,0.64,15,Low,1,\"APP089,APP635\",5\n"
            + "APP002,intuitive e-commerce,0.9,0.19,14,Low,0,0,0\n",
            encoding="utf-8")
        apps = parse_csv_portfolio(str(csv_file))
        assert len(apps) == 2
        assert apps[0]["application_id"] == "APP001"
        assert apps[1]["dependency_count"] == "0"
        out = analyze_application(apps[0], resolver=PricingResolver())
        assert out["app_id"] == "APP001"  # full pipeline accepts parsed rows


class TestCLI:
    def test_cli_demo_output_is_valid_json(self, capsys, monkeypatch, offline_resolver):
        # Keep the CLI offline/deterministic
        monkeypatch.setattr(cost, "PricingResolver", lambda region="us-east-1", verbose=False: offline_resolver)
        argv = ["cost.py", "--quiet"]
        monkeypatch.setattr(sys, "argv", argv)
        cost.main()
        captured = capsys.readouterr().out
        payload = json.loads(captured)
        assert payload["app_count"] == 3
        assert payload["applications"][0]["app_id"] == "APP_001"
        assert "fixture" in payload["data_source"]

    def test_cli_app_filter(self, capsys, monkeypatch, offline_resolver):
        monkeypatch.setattr(cost, "PricingResolver", lambda region="us-east-1", verbose=False: offline_resolver)
        monkeypatch.setattr(sys, "argv", ["cost.py", "--quiet", "--app", "APP_002"])
        cost.main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["app_count"] == 1
        assert payload["applications"][0]["app_id"] == "APP_002"

    def test_cli_input_file(self, capsys, monkeypatch, offline_resolver, tmp_path):
        infile = tmp_path / "apps.json"
        infile.write_text(json.dumps([{"app_id": "A1", "app_name": "X"}]),
                          encoding="utf-8")
        monkeypatch.setattr(cost, "PricingResolver", lambda region="us-east-1", verbose=False: offline_resolver)
        monkeypatch.setattr(sys, "argv", ["cost.py", "--quiet", "--input", str(infile)])
        cost.main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["applications"][0]["app_id"] == "A1"

    def test_cli_backend_unreachable_falls_back(self, capsys, monkeypatch, offline_resolver):
        monkeypatch.setattr(cost, "PricingResolver", lambda region="us-east-1", verbose=False: offline_resolver)
        monkeypatch.setattr(sys, "argv",
                            ["cost.py", "--quiet", "--backend", "http://127.0.0.1:9999"])
        cost.main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["app_count"] == 3  # fixture served instead
        assert "fixture" in payload["data_source"]

    def test_cli_csv_input(self, capsys, monkeypatch, offline_resolver, tmp_path):
        csv_file = tmp_path / "portfolio.csv"
        header = ("application_id,application_name,cpu_usage,memory_usage,age_years,"
                  "criticality,compliance_flag,dependency_ids,dependency_count")
        csv_file.write_text(header + "\n"
                            + "APP001,holistic bandwidth,0.76,0.64,15,Low,1,\"APP089,APP635\",5\n",
                            encoding="utf-8")
        monkeypatch.setattr(cost, "PricingResolver", lambda region="us-east-1", verbose=False: offline_resolver)
        monkeypatch.setattr(sys, "argv", ["cost.py", "--quiet", "--csv", str(csv_file)])
        cost.main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["data_source"].startswith("csv")
        assert payload["applications"][0]["app_id"] == "APP001"


# ----------------------------------------------------------------------------
# Plain runner (no pytest required)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--strict-markers", "-m", "not optional"]))
