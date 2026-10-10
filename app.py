import asyncio
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase
import ollama
from pydantic import BaseModel

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "your_secure_password")  # Change to your Neo4j password

# Simulated hospital databases running inside your Neo4j instance
HOSPITALS = ["hospitalA", "hospitalB", "hospitalC"]

# Local LLM Parameters
LLM_MODEL = "llama3.2"  # Or 'qwen2.5:7b-instruct' / 'biomistral:7b'
LLM_OPTIONS = {
    "temperature": 0.1,    # Low temperature to minimize medical hallucinations
    "top_p": 0.9,
    "num_ctx": 16384       # Large context window for handling aggregated graph insights
}

# ==========================================
# PRIVACY-PRESERVING GRAPH RETRIEVAL
# ==========================================
async def query_hospital_graph(hospital_db: str, symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    Queries a single hospital database. 
    Enforces privacy by returning ONLY aggregated structural paths and counts.
    """
    # Cypher query designed to look for structural paths matching symptoms without touching patient PII
    cypher_query = """
    MATCH (s:Symptom)-[:INDICATES]->(d:Disease)
    WHERE s.name IN $symptoms
    RETURN d.name AS disease_name, 
           count(s) AS matching_symptoms_count, 
           collect(s.name) AS matched_symptoms
    ORDER BY matching_symptoms_count DESC
    LIMIT 5
    """
    
    evidence = []
    try:
        # Connect asynchronously to the isolated hospital database
        async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
            async with driver.session(database=hospital_db) as session:
                result = await session.run(cypher_query, symptoms=symptoms)
                async for record in result:
                    evidence.append({
                        "disease": record["disease_name"],
                        "match_count": record["matching_symptoms_count"],
                        "symptoms": record["matched_symptoms"]
                    })
    except Exception as e:
        print(f"[-] Secure link to {hospital_db} failed or database offline: {e}")
        
    return evidence

# ==========================================
# CENTRAL GRAPH-RAG ORCHESTRATION
# ==========================================
async def main():
    # Example clinical scenario: Entry symptoms provided by a physician
    input_symptoms = ["Muscle weakness", "Ptosis", "Fatigability"]
    print(f"[+] Initializing Privacy-Preserved Cross-Hospital Search for: {input_symptoms}\n")

    # 1. Fetch graph evidence from all hospitals concurrently (Federation)
    tasks = [query_hospital_graph(hospital, input_symptoms) for hospital in HOSPITALS]
    federated_results = await asyncio.gather(*tasks)

    # 2. Combine and aggregate the privacy-filtered outputs centrally
    aggregated_context = ""
    for hospital_name, records in zip(HOSPITALS, federated_results):
        aggregated_context += f"--- Insights from Simulated Node: {hospital_name} ---\n"
        if not records:
            aggregated_context += "No structural overlap found under current privacy filter.\n"
        for rec in records:
            aggregated_context += f"- Suspected Disease: {rec['disease']} (Matched {rec['match_count']} symptoms: {', '.join(rec['symptoms'])})\n"
        aggregated_context += "\n"

    # 3. Construct the Clinical Synthesis Prompt
    system_prompt = (
        "You are an expert clinical geneticist specializing in rare diseases. "
        "Analyze the following privacy-filtered federated graph database outputs. "
        "Provide an explainable, differential diagnosis listing at least 3 distinct possibilities. "
        "Do not invent medical facts. Base your clinical reasoning on the aggregated insights provided."
    )
    
    user_prompt = f"Federated Graph Context:\n{aggregated_context}\nPatient presenting symptoms: {', '.join(input_symptoms)}"

    print("[+] Consolidating evidence. Querying local Ollama instance...")
    
    # 4. Stream output generation from local Ollama LLM
    try:
        response_stream = ollama.generate(
            model=LLM_MODEL,
            prompt=f"{system_prompt}\n\n{user_prompt}",
            options=LLM_OPTIONS,
            stream=True
        )
        
        print("\n=== FINAL EXPLAINABLE DIAGNOSTIC SUMMARY ===")
        for chunk in response_stream:
            print(chunk['response'], end='', flush=True)
        print("\n============================================")
        
    except Exception as e:
        print(f"\n[-] Failed to connect to Ollama: {e}. Check if 'ollama serve' is running.")

if __name__ == "__main__":
    asyncio.run(main())
