# Privacy-Preserving Federated GraphRAG for Cross-Hospital Rare Disease Diagnosis

A lightweight federated graph-retrieval prototype that models rare-disease knowledge across multiple hospital-specific Neo4j databases and combines the results with a local LLM for explainable diagnosis support.

This repository is intentionally focused on a local, privacy-aware research workflow rather than a production-ready clinical system. The project demonstrates:

- Graph-based rare disease knowledge modeling in Neo4j
- Federated querying across multiple hospital graph shards
- Privacy-preserving aggregation of structural evidence instead of raw patient data
- Local LLM synthesis with Ollama for diagnostic reasoning

---

## Project Overview

Healthcare data is fragmented across institutions, and raw patient-level sharing is often not allowed. This project simulates a multi-hospital setup where each hospital maintains its own Neo4j graph database containing a disease-related knowledge shard.

A central app:

1. Queries each hospital graph in parallel
2. Extracts only aggregated, non-identifying structural evidence
3. Combines those signals into a federated context
4. Sends the merged context to a local LLM for explanation-based differential diagnosis support

---

## Repository Structure

```text
federated_graphRAG/
├── app.py                          # Main federated GraphRAG orchestration logic
├── README.md                      # Project documentation
├── .gitignore                     # Local environment exclusions
├── data/
│   ├── raw/                       # Source Orphanet-like disease CSV files
│   │   ├── rare_diseases_complete.csv
│   │   ├── rare_diseases_genes.csv
│   │   ├── rare_diseases_info.csv
│   │   ├── rare_diseases_natural_history.csv
│   │   └── rare_diseases_prevalence.csv
│   └── federated/
│       ├── hospital_A.csv
│       ├── hospital_B.csv
│       └── hospital_C.csv
└── scripts/
    ├── partition_data.py          # Merges source CSVs and splits them by hospital
    └── ingest_to_neo4j.py        # Loads each hospital partition into Neo4j
```

---

## Current Implementation Details

The repository currently contains three core parts:

- `scripts/partition_data.py`
  - Reads the raw disease datasets in `data/raw/`
  - Merges them on `OrphaCode`
  - Shuffles and partitions the combined data into three hospital CSVs

- `scripts/ingest_to_neo4j.py`
  - Connects to Neo4j
  - Creates or clears graph data per hospital database
  - Loads disease, gene, and age-of-onset relationships into Neo4j

- `app.py`
  - Connects to `hospitalA`, `hospitalB`, and `hospitalC`
  - Executes the same graph query against each database in parallel
  - Aggregates the result into a single federated evidence summary
  - Sends the merged evidence to Ollama using `llama3.2`

---

## Prerequisites

Before running the project, install and configure the following:

- Python 3.10+
- Neo4j Desktop or a local Neo4j instance
- Ollama with a local model available

### Neo4j Setup

Create or enable a local Neo4j instance and use the same credentials in the code.

The current code expects these database names:

- `hospitalA`
- `hospitalB`
- `hospitalC`

The code also sets the auth pair to:

```python
NEO4J_AUTH = ("neo4j", "your_secure_password")
```

Update that value in:

- `app.py`
- `scripts/ingest_to_neo4j.py`

so it matches your Neo4j password.

### Ollama Setup

Install Ollama, then pull and run a model:

```bash
ollama pull llama3.2
ollama run llama3.2
```

The app uses the local model name `llama3.2` by default.

---

## Python Dependencies

Install the required Python packages:

```bash
pip install neo4j pandas ollama pydantic
```

If you are using a virtual environment, activate it before installation.

Example:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install neo4j pandas ollama pydantic
```

---

## Data Preparation Workflow

### 1. Prepare the raw source data

Place the raw CSV files inside:

```text
data/raw/
```

The script expects files such as:

- `rare_diseases_info.csv`
- `rare_diseases_genes.csv`
- `rare_diseases_natural_history.csv`

### 2. Partition the data by hospital

Run:

```bash
python scripts/partition_data.py
```

This creates files in:

```text
data/federated/
```

such as:

- `hospital_A.csv`
- `hospital_B.csv`
- `hospital_C.csv`

### 3. Ingest the partitions into Neo4j

Run:

```bash
python scripts/ingest_to_neo4j.py
```

This script connects to each hospital database and loads the graph relationships.

---

## Running the Federated GraphRAG App

Once the databases are populated, run:

```bash
python app.py
```

The app will:

1. Query all hospital databases in parallel
2. Search for structural symptom-to-disease mappings
3. Aggregate the evidence without exposing raw patient records
4. Send a combined context to the local LLM
5. Print an explainable diagnostic summary

---

## Example Runtime Behavior

The app uses a sample symptom set such as:

```python
["Muscle weakness", "Ptosis", "Fatigability"]
```

It then queries each hospital database for disease matches and synthesizes an explanation-based output.

---

## Notes and Limitations

- This project is a demonstration prototype, not a regulated clinical deployment.
- The current implementation does not yet load values from a `.env` file; configuration is set directly in Python code.
- The privacy-preserving layer is simulated at the graph-query level and should be treated as a research design pattern, not as formal medical privacy compliance.
- The code uses local Neo4j and Ollama services, so your machine must have both available.

---

## Suggested Next Improvements

- Add `.env` loading for Neo4j and Ollama configuration
- Add a proper `requirements.txt`
- Add a CLI for custom symptom input
- Replace the simulated hospital evidence with a more realistic federated learning or secure aggregation layer
- Add health-data validation and safer prompt restrictions for clinical use

---

## Summary

This repository demonstrates a federated GraphRAG workflow for rare-disease diagnosis across multiple hospital shards. It is intentionally designed to be run locally and to keep the workflow understandable and easy to modify for research exploration.
