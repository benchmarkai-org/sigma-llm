# Sigma LLM

A machine learning-powered system for generating and evaluating Sigma detection rules for cybersecurity.

## Overview

Sigma LLM is a project that leverages large language models (LLMs) to automatically generate, evaluate, and improve Sigma rules for cybersecurity threat detection. The system can:

1. Generate Sigma rules from natural language descriptions
2. Evaluate the quality of generated rules against reference rules
3. Compare and judge different rule implementations
4. Summarize detection logic and reference materials

## Project Structure

```
sigma_llm/
├── api/                  # API service for rule generation and evaluation
│   ├── app.py            # Flask API endpoints
│   └── Dockerfile        # Container definition for API service
├── evaluation/           # Evaluation results and metrics
│   └── results/          # JSON files with evaluation results
├── scripts/              # Utility scripts for testing and evaluation
│   ├── base_evaluator.py # Base class for rule evaluation
│   ├── create_*.py       # Scripts to create test data
│   ├── evaluate.py       # Main evaluation script
│   ├── run_experiments.py # Script to run multiple model experiments
│   └── Dockerfile        # Container for running experiments
├── sigma_llm/            # Core library
│   ├── base.py           # Abstract base classes
│   └── llm.py            # LLM integration
└── utils/                # Utility functions
    └── populate_vectorstore.py # Script to populate vector database
```

## Key Features

### Rule Generation

The system can generate Sigma rules from natural language descriptions, leveraging LLMs to understand the detection requirements and produce properly formatted YAML rules that follow Sigma syntax.

### Rule Evaluation

The evaluation framework assesses generated rules based on:
- YAML validity
- Required fields presence
- Detection logic similarity to reference rules
- Metadata completeness
- LLM-based judgment of rule quality

### Vectorstore Integration

The system uses Pinecone as a vector database to store and retrieve relevant Sigma rules, enabling semantic search to find similar rules that can inform the generation process.

## Technologies Used

- **LLMs**: Claude, GPT-4, and other models for rule generation and evaluation
- **Vector Database**: Pinecone for storing rule embeddings
- **API Framework**: Flask for the REST API
- **Containerization**: Docker for deployment
- **Cloud Integration**: Google Cloud Storage for storing evaluation results

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- API keys for:
  - OpenAI
  - Anthropic
  - Pinecone

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -e .
   ```

3. Set up environment variables:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=sigma-rules
   SERVICE_API_KEY=your_service_key
   ```

### Running the API

```bash
cd api
python app.py
```

Or using Docker:

```bash
docker build -t sigma-llm-api -f api/Dockerfile .
docker run -p 8090:8090 --env-file .env sigma-llm-api
```

### Running Experiments

```bash
cd scripts
python run_experiments.py
```

Or using Docker:

```bash
docker build -t sigma-experiments -f scripts/Dockerfile .
docker run --env-file .env sigma-experiments
```

## API Endpoints

- `POST /api/v1/rules`: Generate a Sigma rule from a natural language query
- `POST /api/v1/judge`: Compare two Sigma rules and determine which is better
- `POST /api/v1/assess`: Assess the quality of a single Sigma rule
- `POST /api/v1/summarize-detection`: Summarize the detection logic of a rule
- `POST /api/v1/summarize-references`: Summarize reference content for rule creation
- `GET /api/v1/health`: Health check endpoint

## Evaluation Metrics

The system evaluates generated rules using several metrics:

1. **Valid YAML**: Ensures the rule is properly formatted
2. **Required Fields**: Checks for essential Sigma rule components
3. **Detection Logic Similarity**: Compares detection logic to reference rules
4. **Metadata Completeness**: Evaluates the presence of metadata fields
5. **LLM Judgment**: Uses an LLM to assess rule quality based on:
   - Detection effectiveness
   - False positive control

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

