# Local API Contract

Base URL: `http://127.0.0.1:8000`

This contract remains stable for the frontend while the backend delegates real implementation details to service adapters. The local code does not use AWS services.

## `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

## `GET /applications`

Returns the application portfolio loaded from the repository dataset.

## `GET /applications/{app_id}`

Example: `/applications/APP001`

Returns the single application record.

## `POST /recommendation`

Request examples:

```json
{ "application_id": "APP001" }
```

```json
{ "app_id": "APP001" }
```

Response fields:

- `app_id`
- `application_id`
- `recommendation` (`Rehost`, `Replatform`, `Repurchase`, `Refactor`, `Retire`, `Retain`)
- `confidence` (0 to 1)
- `explanation` (object or string, with model details when available)

## `POST /migration-waves`

Request example:

```json
{ "application_ids": ["APP001", "APP002"] }
```

The field is optional; if omitted, all applications are planned.

Response example:

```json
{
  "waves": [
    { "wave": 1, "applications": ["APP002"], "risk": "Low" }
  ]
}
```

## `POST /copilot`

Request example:

```json
{ "question": "How should we sequence migration waves?" }
```

Response fields:

- `question`
- `answer`
- `sources`

## `POST /cost-risk`

Request examples:

```json
{ "application_id": "APP001" }
```

```json
{ "app_id": "APP001" }
```

Response fields:

- `app_id`
- `application_id`
- `monthly_aws_cost`
- `cost_range` (`lower` and `upper`)
- `risk_score` (0 to 100)

## Validation and errors

- Missing or malformed request fields return HTTP 422.
- Unknown application IDs return HTTP 404.
- Data-loading and internal service failures return HTTP 500 with a JSON detail message.

## Local test flow

1. Activate the local backend venv.
2. Run `uvicorn app.main:app --reload` from the `backend` directory.
3. Use the examples above or load the Swagger docs at `/docs`.
4. Validate results with `pytest -q`.
