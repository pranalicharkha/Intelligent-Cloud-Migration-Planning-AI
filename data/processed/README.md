# Data Engineering

## Overview

This folder contains the processed workload dataset prepared as part of the data engineering stage of the project.

The main objective is to collect, clean, process, transform, and organize workload data into a structured format.

## Dataset

**File:** `task_summary_ml_ready.csv`

The dataset contains approximately **179,427 records** and **21 columns**.

Each row represents a workload/task record containing information about resource usage, resource capacity, task characteristics, and utilization.

## Data Source

The dataset is derived from **Google Cluster Workload Traces / Google Cluster Data**.

The required workload data was collected from the available cluster trace dataset and processed into a structured task-level dataset.

## Data Processing

The following data engineering operations were performed:

1. Collected workload/task event data.
2. Selected relevant fields from the source data.
3. Processed task-level records.
4. Aggregated resource usage information.
5. Calculated CPU usage statistics.
6. Calculated memory usage statistics.
7. Calculated disk I/O information.
8. Calculated task lifetime.
9. Calculated the number of observations and events.
10. Included machine, priority, and scheduling information.
11. Calculated CPU utilization.
12. Calculated memory utilization.
13. Calculated CPU headroom.
14. Calculated memory headroom.
15. Calculated overall resource utilization.
16. Checked the final dataset for missing values.

## Column Summary

### Task Information

* `task_id` — Unique identifier for each task.
* `machine_id` — Identifier of the machine where the task ran.
* `priority` — Priority assigned to the task.
* `scheduling_class` — Scheduling class assigned to the task.
* `lifetime_seconds` — Total lifetime of the task in seconds.
* `num_observations` — Number of resource observations recorded for the task.
* `num_events` — Number of events associated with the task.

### CPU Information

* `avg_cpu` — Average CPU usage of the task.
* `max_cpu` — Maximum CPU usage recorded for the task.
* `cpu_std` — Standard deviation of CPU usage.
* `cpu_capacity` — CPU capacity available to the task.
* `cpu_utilization_ratio` — CPU usage relative to available CPU capacity.
* `cpu_headroom` — Remaining CPU capacity after resource usage.

### Memory Information

* `avg_memory` — Average memory usage of the task.
* `max_memory` — Maximum memory usage recorded for the task.
* `memory_std` — Standard deviation of memory usage.
* `memory_capacity` — Memory capacity available to the task.
* `memory_utilization_ratio` — Memory usage relative to available memory capacity.
* `memory_headroom` — Remaining memory capacity after resource usage.

### Other Resource Information

* `avg_disk_io` — Average disk I/O usage of the task.
* `overall_resource_utilization` — Overall utilization calculated from the processed resource features.

## Data Quality

The processed dataset was checked for missing values.

**Missing values:** None detected in the final processed dataset.

The dataset was also reviewed to ensure that the processed values were stored in a structured and consistent format.

## Output

The final processed dataset is:

```text
data/processed/task_summary_ml_ready.csv
```

This file represents the final output of the current data processing stage.

## Data Processing Flow

```text
Source Workload Data
        ↓
Data Collection
        ↓
Data Selection
        ↓
Data Cleaning
        ↓
Task-Level Processing
        ↓
Data Aggregation
        ↓
Feature Calculation
        ↓
Data Validation
        ↓
Processed Dataset
```

## Notes

* The original raw dataset is not stored in this folder because of its large size.
* The processed dataset is maintained separately from the raw source data.
* Any future changes to the processing workflow should be documented.
* Dataset changes should be tracked using Git.
