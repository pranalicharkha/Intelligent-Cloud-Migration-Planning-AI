# Intelligent Cloud Migration Planning using AI

## 1. System Overview

The system helps organizations plan cloud migration using AI-driven
recommendations, dependency-aware migration waves, cost prediction,
risk analysis, and a GenAI migration assistant.

## 2. High-Level Architecture

React Frontend
       |
       v
API Gateway
       |
       v
AWS Lambda
       |
       +------------------+
       |                  |
       v                  v
DynamoDB                 S3
       |
       v
Migration Results

AI/ML Components:
- 6R Recommendation Engine
- Dependency & Migration Wave Planner
- Cost & Risk Simulator
- GenAI Migration Copilot

## 3. Main Features

### Feature 1: Explainable 6R Recommendation

Input:
- CPU
- Memory
- Application age
- Criticality
- Compliance
- Dependency count

Processing:
- Random Forest classifier
- SHAP explainability

Output:
- Rehost
- Replatform
- Repurchase
- Refactor
- Retire
- Retain

### Feature 2: Dependency-Aware Migration Wave Planner

Processing:
- Build application dependency graph using NetworkX
- Detect application communities
- Generate migration waves
- Order waves using a greedy heuristic

### Feature 3: GenAI Migration Copilot

Processing:
- AWS migration documentation
- FAISS vector store
- LangChain
- Hugging Face LLM

Output:
- Migration-related answers
- Explanation of recommendations
- Supporting sources

### Feature 4: Predictive Cost & Risk Simulator

Processing:
- AWS Price List API
- Cost calculation
- Monte Carlo simulation

Output:
- Estimated monthly cost
- Lower cost bound
- Upper cost bound
- Risk score

## 4. AWS Services

- Amazon S3
- AWS Lambda
- Amazon API Gateway
- Amazon DynamoDB
- Amazon CloudWatch

## 5. Data Flow

Application Dataset
        |
        v
       S3
        |
        v
Data Processing
        |
        +----------------------+
        |          |           |
        v          v           v
      6R Model   Wave       Cost/Risk
                 Planner
        |          |           |
        +----------+-----------+
                   |
                   v
             Migration Results
                   |
                   v
               DynamoDB
                   |
                   v
              API Gateway
                   |
                   v
             React Dashboard
             