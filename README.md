Intelligent Cloud Migration Planning
AI-Powered Planning for Simplified Cloud Migration

Description
Intelligent Cloud Migration Planning is an open-source project designed to simplify the complexities of cloud migration. It provides a suite of AI-driven tools to help organizations make informed decisions, optimize migration waves, understand costs and risks, and gain insights into the migration process. This project is ideal for teams looking to leverage AI for a more efficient and less risky cloud migration.

Features
*   **Explainable 6R Recommendation Engine:** Predicts and explains Rehost, Replatform, Repurchase, Refactor, Retire, or Retain strategies for each application using a Random Forest classifier and SHAP for explainability.
*   **Dependency-Aware Migration Wave Planner:** Groups tightly-connected applications and sequences migration waves to minimize risk, utilizing NetworkX for graph analysis and community detection.
*   **GenAI Migration Copilot (RAG Assistant):** A chatbot that answers migration-related questions and explains recommendations, grounded in real documentation using LangChain and a local FAISS vector store.
*   **Predictive Cost & Risk Simulator:** Estimates monthly AWS costs with a confidence range and provides a risk score, querying the AWS Price List API and using Monte-Carlo simulations for uncertainty.

Installation

```bash
# Clone the repository
git clone <repository_url>
cd intelligent-cloud-migration

# Install backend dependencies (example for Lambda)
pip install -r requirements.txt

# Install frontend dependencies (example for React)
cd frontend
npm install
```

Quick Start / Usage

```python
# Example of using the 6R Recommendation Engine
from intelligent.recommendation_engine import Recommender

recommender = Recommender()
app_data = {
    "cpu_memory": "2xCPU, 8GB RAM",
    "age_years": 7,
    "criticality": "High",
    "compliance_flag": False,
    "dependency_count": 5
}
recommendation, explanation = recommender.predict(app_data)
print(f"Recommendation: {recommendation}")
print(f"Explanation: {explanation}")

# Example of planning migration waves (conceptual)
from intelligent.wave_planner import WavePlanner
from intelligent.dependency_graph import DependencyGraph

graph = DependencyGraph("path/to/dependency_data.json")
planner = WavePlanner(graph.get_graph())
migration_waves = planner.plan_waves()
print(f"Migration Waves: {migration_waves}")
```

Configuration
The project's configuration can be managed through environment variables and configuration files located in the `config/` directory. Key configurations include API endpoints, data sources, and model parameters.

Contributing
We welcome contributions to improve Intelligent Cloud Migration Planning. Please refer to our `CONTRIBUTING.md` file for guidelines on how to get involved.

License
This project is licensed under the MIT License. See the `LICENSE` file for details.
