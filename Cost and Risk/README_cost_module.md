# Cost & Risk Simulator — Role 6 (Predictive Cost & Risk Engineer)

Standalone module for the **Intelligent Cloud Migration Planner**. It turns an
application's technical specs into:

1. **Live AWS pricing** — real on-demand USD/hour for the recommended EC2 instance
2. **Monte-Carlo monthly cost range** — 5th percentile / mean / 95th percentile
3. **Migration risk score** — explainable 0–100 score with LOW / MEDIUM / HIGH
4. **Clean JSON** — ready for the React dashboard and the GenAI copilot's FAISS index

## Quick start

```bash
pip install -r requirements.txt

# Demo portfolio (3 sample apps), prints JSON to stdout
python cost.py

# Analyze your own portfolio file (list, or {"applications": [...]})
python cost.py --input example_app_portfolio.json

# Single app, another region, custom iterations
python cost.py --input example_app_portfolio.json --app APP_001 --region eu-west-1 --iterations 1500

# Force-refresh the local AWS price cache
python cost.py --refresh-pricing

# Save output for the backend to upload
python cost.py --input example_app_portfolio.json --out results.json

# Tests
python -m pytest test_cost.py -v
```

## How to call it from your own code

```python
from cost import analyze_portfolio

results = analyze_portfolio(apps, iterations=2000)
# results is a list of plain dicts -> json.dumps(results) and ship it
```

## 1. Live pricing — the tier chain

The resolver tries sources in order and stops at the first success. **It never
raises** — every failure falls through to the next tier, so demos can't crash:

| Tier | Source | Needs AWS account? | Notes |
|------|--------|--------------------|-------|
| T1 | boto3 `pricing` client (Price List **Query API**) | Yes (signed creds) | The "official SDK" path; auto-detected via env credentials; filtered by region `location` so the rate matches the planned region |
| T2 | boto3 `pricing` client, unsigned | No | Kept for environments where the API allows it |
| T3 | Public AWS Pricing Calculator pricing file | No | Anonymous HTTPS, ~0.7 MB, **all** instance types of a region |
| T4 | Local cache `cache/aws_prices_<region>.json` | No | 24 h TTL; a stale cache is still used if every live tier fails |
| T5 | Built-in static table | No | Last resort; unknown instance types get a size/family-scaled estimate |

Every result carries its `price_source`, so you can always tell which tier
produced the number (great for the paper's "we used REAL AWS pricing" section).

## 2. Monte-Carlo cost model

Monthly cost per iteration `i` (default 2,000 iterations, fixed seed 42):

```
cost_i = hourly_rate * 730 * compute_mult_i  +  storage_gb * ebs_rate(region) * storage_mult_i

compute_mult_i ~ Normal(1.0, 0.20), floored at 0.5     # traffic/burst variance
storage_mult_i ~ Uniform(1.0, 1.30)                    # 0–30% data growth
```

`ebs_rate(region)` is **region-aware** (EBS gp3 list price per GB-month):
$0.080 in us-east-1, $0.084 in eu-west-1, $0.110 in sa-east-1, etc. — see
`EBS_GB_MONTH_BY_REGION` in `cost.py`. Unknown regions fall back to $0.080.
Pass `ebs_rate=...` explicitly to override the lookup.

Output: **low (5th pct) / expected (mean) / high (95th pct) / std-dev**, i.e. a
90% confidence interval — deliberately *not* the single static number that AWS
Migration Hub or the AWS Pricing Calculator gives (that contrast is your paper's
evaluation angle).

## 3. Risk score — weighted and explainable

| Factor | Max points | Rule |
|--------|-----------|------|
| Tech age | 30 | >10 yrs → 30 · 5–10 yrs → 15 |
| Dependencies | 30 | ≥5 deps → 30 · 2–4 deps → 15 |
| Compliance | 25 | regulated data flag → 25 |
| Deprecated runtime | 15 | deprecated flag → 15 |
| Criticality (bonus) | +10 | mission-critical → 10 · business-critical → 5 |

Score is capped at 100. Bands: **≥65 HIGH (red)**, **35–64 MEDIUM (amber)**,
**<35 LOW (green)**. Every contributing factor is returned with its points and a
human-readable detail string, so the dashboard badge and the copilot can answer
*"why is this app risky?"* with no extra work.

## 4. Output JSON schema (the contract)

Top level:

```jsonc
{
  "generated_at_utc": "2026-09-14T10:00:00Z",
  "aws_region": "us-east-1",
  "app_count": 3,
  "applications": [ ...one object per app (below)... ]
}
```

Per application:

```jsonc
{
  "app_id": "APP_001",
  "app_name": "Legacy Core Billing",
  "target_aws_instance": "m5.large",
  "aws_region": "us-east-1",
  "storage_gb": 250.0,
  "hourly_rate_usd": 0.096,                    // real live rate
  "price_source": "aws-pricing-calculator-public-file",
  "cost_simulation_monthly": {
    "low_5th_percentile_usd": 74.2,            // chart lower bound
    "expected_mean_usd": 78.5,                 // chart marker
    "high_95th_percentile_usd": 84.9,          // chart upper bound
    "std_dev_usd": 3.3,
    "confidence_interval_percent": 90,
    "iterations": 2000,
    "assumptions": { "hours_per_month": 730, "ebs_usd_per_gb_month": 0.08, "...": "..." }
  },
  "risk_assessment": {
    "risk_score": 100,                          // 0-100, badge value
    "risk_level": "HIGH",                       // LOW | MEDIUM | HIGH
    "risk_badge_color": "red",                  // green | amber | red
    "risk_factors": [
      { "category": "tech_age", "points": 30, "detail": "Legacy codebase: 12 years old (>10)" },
      { "category": "dependencies", "points": 30, "detail": "High coupling: 7 direct dependencies (>=5)" },
      { "category": "compliance", "points": 25, "detail": "Handles regulated data (PCI)" },
      { "category": "deprecated_tech", "points": 15, "detail": "Deprecated runtime: Java 6 / WebLogic" },
      { "category": "criticality", "points": 10, "detail": "Mission-critical workload - outage is high-impact" }
    ],
    "weights_used": { "tech_age_max": 30, "dependencies_max": 30, "compliance_max": 25,
                      "deprecated_tech_max": 15, "criticality_bonus_max": 10 }
  },
  "summary_for_copilot": "Legacy Core Billing runs on m5.large at $0.096/hr ... Drivers: tech_age +30; dependencies +30; ..."
}
```

### Who consumes what

- **Frontend Engineer** — `cost_simulation_monthly.{low,expected,high}` for the
  range chart; `risk_assessment.risk_score / risk_level / risk_badge_color` for
  the warning badge; `risk_factors[].detail` for tooltips.
- **GenAI Copilot Engineer** — index `summary_for_copilot` verbatim (it is one
  self-contained sentence pack with prices, range, risk and drivers), plus the
  structured blocks for precise numeric answers.
- **Coordinator / backend** — top-level `generated_at_utc` + `price_source`
  makes results auditable and cacheable; the module is import-safe for Lambda.

### Input contract (what the Data Engineer should send)

```jsonc
{
  "app_id": "APP_001",
  "app_name": "Legacy Core Billing",
  "recommended_instance": "m5.large",     // from the 6R engine / right-sizing
  "storage_gb": 250,
  "app_age_years": 12,
  "dependency_count": 7,
  "has_compliance_data": true,
  "compliance_frameworks": ["PCI"],       // optional, enriches the risk detail
  "is_deprecated_tech": true,
  "tech_stack": "Java 6 / WebLogic",      // optional
  "criticality": "mission-critical"       // optional: low | business-critical | mission-critical
}
```

Missing fields are defaulted sensibly (t3.medium, 50 GB, age 0, no compliance) —
the module never crashes on sparse input. Dirty values are safe too: `null`,
blank strings, and non-numeric junk like `"unknown"` in `app_age_years`,
`dependency_count`, or `storage_gb` are parsed through `_safe_float`/`_safe_int`
and fall back to their defaults instead of raising `ValueError`.

## Design decisions & known limits

- **EBS storage is priced with a region-aware gp3 table** (`EBS_GB_MONTH_BY_REGION`,
  ~$0.08/GB-month in US/EU, higher in São Paulo/Mumbai). RDS storage, IOPS and
  data-transfer out are not modeled — fine for a portfolio-level planner; add
  per-GB tiers if the 6R engine starts recommending dedicated DB hosts.
- **Burstable t3 CPU credits** are not priced separately; the compute multiplier
  approximates burst behavior.
- The risk model is intentionally a transparent weighted rule engine (mirrors the
  plan's fallback approach for the 6R engine), so factor-level SHAP-style
  explanations come for free.
- All random draws go through `np.random.default_rng(42)` → identical outputs on
  every machine, which makes demos and tests reproducible.

## Files

| File | Purpose |
|------|---------|
| `cost.py` | The module (pricing tiers + Monte-Carlo + risk + CLI) |
| `test_cost.py` | Offline-safe test suite (9 classes incl. regression tests) |
| `requirements.txt` | numpy, boto3, pytest |
| `example_app_portfolio.json` | Input contract example |
| `cache/aws_prices_<region>.json` | Auto-created price cache (safe to delete) |
