# Intelligent Cloud Migration Planning

AI-powered planning for simplified cloud migration.

## Current Phase

The project is being developed locally first with mock data and APIs. The FastAPI backend is ready for a React frontend without requiring AWS. Real ML, GenAI, cost, and wave-planning modules and AWS deployment will be added in later phases.

## Backend Quick Start

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`. Full endpoint payloads and Postman instructions are in [docs/API_CONTRACT.md](docs/API_CONTRACT.md).

## Repository Areas

- `backend/`: local FastAPI integration layer and mock services.
- `src/`: future recommendation, discovery, copilot, cost-risk, and wave-planning modules.
- `frontend/`: future React client.
- `docs/`: architecture and API contracts.

## Future Scope

The API paths and Pydantic contracts are intentionally stable. Real implementations should replace the functions in `backend/app/services.py` while keeping the frontend-facing endpoints unchanged. AWS Lambda, API Gateway, DynamoDB, S3, Random Forest, SHAP, NetworkX, LangChain, and FAISS are not part of the local prototype yet.

## Contributing
We welcome contributions to improve Intelligent Cloud Migration Planning. Please refer to our `CONTRIBUTING.md` file for guidelines on how to get involved.

.
