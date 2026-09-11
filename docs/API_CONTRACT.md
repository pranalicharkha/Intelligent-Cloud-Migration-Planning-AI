# Local Mock API Contract

Base URL: `http://127.0.0.1:8000`

The contract is intentionally independent of the implementation behind it. The frontend can use these routes now and continue using them when the mock services are replaced with the real project modules.

## `GET /applications`

Returns the application portfolio.

## `POST /recommendation`

Request: `{ "application_id": "app-001" }`

Response fields: `application_id`, `recommendation` (one of `Rehost`, `Replatform`, `Repurchase`, `Refactor`, `Retire`, `Retain`), `confidence` (0 to 1), and `explanation`.

## `POST /migration-waves`

Request: `{ "application_ids": ["app-001", "app-002"] }`. The field is optional; when omitted, all mock applications are planned.

Response: `{ "waves": [{ "wave": 1, "applications": ["app-002"], "risk": "Low" }] }`.

## `POST /copilot`

Request: `{ "question": "How should we sequence migration waves?" }`.

Response fields: `question`, `answer`, and `sources`.

## `POST /cost-risk`

Request: `{ "application_id": "app-001" }`.

Response fields: `application_id`, `monthly_aws_cost`, `cost_range` (`lower` and `upper`), and `risk_score` (0 to 100).

Unknown application IDs return HTTP 404. Invalid or missing request fields return HTTP 422 through FastAPI validation.

## Postman

1. Start the server with `uvicorn app.main:app --reload` from `backend`.
2. Create a Postman collection with base URL `http://127.0.0.1:8000`.
3. Send `GET {{baseUrl}}/applications`.
4. For each POST route, select **Body > raw > JSON** and use the examples above.
5. Confirm successful responses are HTTP 200. Try `{ "application_id": "unknown" }` on recommendation or cost-risk to confirm HTTP 404, and `{}` on recommendation to confirm HTTP 422.

## GitHub

From the repository root:

```powershell
git add backend README.md docs/API_CONTRACT.md
git commit -m "Add local FastAPI mock integration layer"
git push origin <your-branch>
```
