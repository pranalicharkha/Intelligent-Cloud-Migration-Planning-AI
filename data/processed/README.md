# Data Engineering

## Overview

This folder contains the processed datasets prepared for the Intelligent Cloud Migration Planning project.

The datasets provide workload-level and application-level information required for downstream feature engineering and machine learning.

## Datasets

### 1. Workload Dataset

**File:** `task_summary_ml_ready.csv`

- Approximately **179,427 records** and **21 columns**.
- Derived from **Google Cluster Workload Traces / Google Cluster Data**.
- Each row represents a processed workload/task record.

Key features include:

- `task_id` — Unique task identifier
- `machine_id` — Machine identifier
- `avg_cpu` — Average CPU usage
- `max_cpu` — Maximum CPU usage
- `avg_memory` — Average memory usage
- `max_memory` — Maximum memory usage
- `avg_disk_io` — Average disk I/O
- `lifetime_seconds` — Task lifetime
- `cpu_utilization_ratio` — CPU utilization
- `memory_utilization_ratio` — Memory utilization
- `cpu_headroom` — Remaining CPU capacity
- `memory_headroom` — Remaining memory capacity
- `priority` — Task priority
- `scheduling_class` — Scheduling class
- `num_observations` — Number of observations
- `num_events` — Number of events

### 2. Application Portfolio Dataset

**File:** `application_portfolio.csv`

- Approximately **150 synthetic application records**.
- Generated using **Python and Faker**.
- Created for application-level cloud migration analysis and downstream machine learning.

Key features include:

- `application_id` — Unique application identifier
- `application_name` — Application name
- `cpu_usage` — CPU usage
- `memory_usage` — Memory usage
- `age_years` — Application age
- `criticality` — Business criticality
- `compliance_flag` — Compliance status
- `dependency_ids` — Application dependencies
- `dependency_count` — Number of dependencies

## Data Sources

### Workload Dataset

The workload dataset is derived from Google Cluster Workload Traces and contains processed task-level resource information.

### Application Portfolio

The application portfolio is synthetically generated using Python and Faker. It does not represent real production applications.

## Data Quality

The workload dataset was checked for missing values.

**Missing values:** None detected in the final processed dataset.

The application portfolio is generated in a structured format with the required application-level fields.

## Dataset Separation

The datasets are intentionally maintained separately because they represent different types of information:

- `task_summary_ml_ready.csv` → Workload and resource-level data
- `application_portfolio.csv` → Application-level migration data

They can be used independently and combined later during feature engineering if required by the ML pipeline.

The synthetic application data should not be interpreted as a real-world mapping to the Google Cluster workload data.

## Output

```text
data/processed/
├── README.md
├── task_summary_ml_ready.csv
└── application_portfolio.csv