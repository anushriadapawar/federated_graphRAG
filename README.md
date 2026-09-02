# Privacy-Preserving Federated GraphRAG for Cross-Hospital Rare Disease Diagnosis

An advanced, privacy-preserving distributed knowledge retrieval framework designed for collaborative, multi-institution rare disease diagnostics using Large Language Models (LLMs) and Distributed Knowledge Graphs.

## 🏛️ Project Overview
Healthcare institutions generate massive quantities of data (EHRs, clinical notes, genomic records), but regulatory frameworks (HIPAA, GDPR) and ethical boundaries strictly prevent centralizing or directly exchanging raw patient datasets. This issue is particularly critical for **rare diseases**, where individual hospitals observe very few cases, leading to highly fragmented clinical knowledge.

Our solution implements an architecture-level framework that solves this silo problem:
1. **GraphRAG Topology:** Instead of flat text blocks, it models medical data dynamically into an interconnected Knowledge Graph (Patients, Symptoms, Diseases, Genes) inside Neo4j to execute multi-hop relational searches.
2. **Federated Learning Layer:** Questions move to the data—rather than moving data to a central cloud. Isolated hospital database partitions process queries locally.
3. **Privacy-Preserving Aggregation:** Local nodes filter out unique individual identifiers (Patient IDs) and supply only anonymized statistical data packets to the central aggregator, ensuring zero leakage of protected health information.

---

## 🏗️ System Architecture & Data Path
```text
 🧑‍⚕️ Clinician Input Query (Plain English text)
            │
            ▼
   [ Medical Parsing Agent ] ──► Extracts Clinical Tokens via Local Llama 3.2
            │
            ▼
   [ Distributed Routing Agent ] ──► Parallel Broadcasts out to Firewalled Node Slots
            │
      ┌─────┴────────────────────────┐
      ▼ (Query Local Graph Shard)     ▼ (Query Local Graph Shard)
┌───────────────────────────┐   ┌───────────────────────────┐
│   HOSPITAL INFRASTRUCTURE A│   │   HOSPITAL INFRASTRUCTURE B│
│  - Neo4j Partition Shard  │   │  - Neo4j Partition Shard  │
│  - Data Masking Filter    │   │  - Data Masking Filter    │
└───────────┬───────────────┘   └───────────┬───────────────┘
            │ (Anonymized Statistics)       │ (Anonymized Statistics)
            └───────────────┬───────────────┘
                            ▼
     [ Privacy-Preserved Context Buffer ] (Merged metadata summaries)
                            │
                            ▼
   [ Diagnostic Aggregator Agent ] ──► Synthesizes Final Report via Local Llama 3.2
                            │
                            ▼
 📝 Secure Consolidated Diagnostic Report Displayed to User
```

---

## 🛠️ Technology Stack
* **Language Model Interface:** Ollama running an offline local instance of `llama3.2`
* **Graph Engine DBMS:** Neo4j Desktop (Community / Enterprise Instance Local Node)
* **Federated Routing Graph:** LangGraph Engine Core
* **Development Engine:** Python 3.10+ & Pandas Data Normalization Framework

---

## 📁 Repository Directory Layout
```text
federated_graphRAG/
│
├── .venv/                      # Local Python Virtual Environment (Ignored by Git)
├── data/                       # Grounding Data Repositories (Manually Added)
│   ├── rare_diseases_complete.csv
│   ├── rare_diseases_genes.csv
│   ├── rare_diseases_info.csv
│   ├── rare_diseases_natural_history.csv
│   └── rare_diseases_prevalence.csv
│
├── .env                        # Local Port Connection Parameters (Ignored by Git)
├── .gitignore                  # Git System Exclusions Mapping
└── app.py                      # Core Multi-Agent Orchestration & Ingestion Script
```

---

## 🚀 Setting Up the Development Workspace

Follow this step-by-step workflow to stand up your environment cluster locally.

### 1. Prerequisite Installations
Ensure the following tools are fully up and running on your hardware:
* Download and open [Neo4j Desktop](https://neo4j.com). Create a new active database DBMS instance, and set your access password to match your configuration.
* Download and run [Ollama](https://ollama.com). Pull down your local language processing engine model via your command prompt line:
  ```bash
  ollama run llama3.2
  ```
  *Leave this terminal panel minimized and running continuously in your background window landscape.*

### 2. Project File Initializations
Clone the source directory down to your computer path:
```bash
git clone https://github.com
cd federated_graphRAG
```
* Create a file named `.env` in your root folder layout:
  ```env
  NEO4J_URI=bolt://localhost:7687
  NEO4J_PASSWORD=your_secure_password_here
  OLLAMA_HOST=http://localhost:11434
  ```
* Create a folder subdirectory named `data/` in the root workspace directory, and place the unzipped **Orphadata Kaggle CSV** files directly into that location.

### 3. Initialize Python Environment Packages
Rebuild your local virtual runtime dependencies directly using your terminal lane path:
```powershell
# Create the local isolation footprint
python -m venv .venv

# Activate the local folder execution variables (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Force-compile all mandatory project dependencies locally
pip install langgraph langchain-openai langchain-community neo4j python-dotenv pandas requests
```

---

## 🖥️ Running the Application Engine

Once your virtual environment panel displays the active variable indicator `(.venv)`, execute the multi-agent graph script:
```bash
python app.py
```

### Expected Execution Log Lifecycle:
1. **Schema Initialization:** The backend verifies optimized uniqueness lookup bounds across the local Neo4j layer.
2. **Kaggle Ingestion Processing:** Pandas imports rows, formats data entities case-insensitively, clears historical tables, and assigns data alternately to `Hospital_A` and `Hospital_B`.
3. **Clinical Extraction:** The model processes your text query into clean diagnostic search targets.
4. **Federated Query Extraction:** Both hospitals execute local calculations, mask raw patient rows, and supply aggregated age/gene packets.
5. **Consolidated Response Output:** The terminal prints out an explainable technical rare condition research overview matching the parameters.

---

## 🎯 Sample Presentation Testing Queries

To demonstrate the full execution capabilities of your model to an evaluation panel, use these sample questions:

1. **Multi-Hospital Cross-Query:** `"Patient presenting with chronic signs of Systemic Inflammation and unclassified syndrome indicators."` (Demonstrates parallel query broadcasting and metadata aggregation).
2. **Multi-Hop Traversal Query:** `"Looking for distributed case logs linked to the KIF7 genetic variant mutation."` (Demonstrates Cypher graph path navigation across Symptom ──► Patient ──► Disease ──► Gene).
3. **Failsafe System Integrity Check:** `"Patient has a common fracture or sports injury in the knee cap."` (Demonstrates the empty-buffer safety filter, verifying that the model reports zero data matching parameters instead of creating hallucinations).
