import os
import pandas as pd
from neo4j import GraphDatabase

# Configuration Parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "adapawar")  # Match your actual Neo4j credentials

# Mapping each generated partition file to its designated active database node
DATA_MAPPING = {
    "hospitalA": "data/federated/hospital_A.csv",
    "hospitalB": "data/federated/hospital_B.csv",
    "hospitalC": "data/federated/hospital_C.csv"
}

def clean_data(val):
    """Safely handle missing or empty fields from text files."""
    if pd.isna(val) or str(val).strip().lower() in ['nan', 'null', '']:
        return "Unknown"
    return str(val).strip()

def ingest_hospital_data(db_name: str, file_path: str):
    if not os.path.exists(file_path):
        print(f"[-] Data file missing for {db_name} at: {file_path}")
        return

    print(f"[+] Loading dataset for {db_name}...")
    df = pd.read_csv(file_path)
    
    # Establish connection driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    # Define optimization query block to create nodes and relationships efficiently
    cypher_query = """
    MERGE (d:Disease {orphaCode: $orpha_code})
    ON CREATE SET 
        d.name = $disease_name,
        d.disorderType = $disorder_type,
        d.disorderGroup = $disorder_group
        
    MERGE (g:Gene {symbol: $gene_symbol})
    ON CREATE SET g.name = $gene_name
    
    MERGE (a:AgeGroup {onset: $age_of_onset})
    
    // Form relationships mapping health traits
    MERGE (d)-[:ASSOCIATED_WITH_GENE {type: $gene_relation}]->(g)
    MERGE (d)-[:MANIFESTS_AT]->(a)
    """
    
    success_count = 0
    
    # Process transactional insertion batches within the specific active database scope
    with driver.session(database=db_name) as session:
        # Clear existing structure to prevent duplicate testing conflicts
        session.run("MATCH (n) DETACH DELETE n")
        
        for _, row in df.iterrows():
            # Clean fields dynamically relying on Orphadata file labels
            orpha_code = clean_data(row.get('OrphaCode'))
            if orpha_code == "Unknown":
                continue  # Skip corrupt entries lacking keys
                
            try:
                session.run(
                    cypher_query,
                    orpha_code=orpha_code,
                    disease_name=clean_data(row.get('Name')),
                    disorder_type=clean_data(row.get('DisorderType')),
                    disorder_group=clean_data(row.get('DisorderGroup')),
                    gene_symbol=clean_data(row.get('GeneSymbol')),
                    gene_name=clean_data(row.get('GeneName')),
                    gene_relation=clean_data(row.get('GeneDisorderRelation Type')),
                    age_of_onset=clean_data(row.get('AgeOfOnset'))
                )
                success_count += 1
            except Exception as e:
                # Silently catch exceptions to ensure loop continuity across diverse columns
                continue

    print(f"[✓] {db_name} populated with {success_count} unified disease connections.")
    driver.close()

if __name__ == "__main__":
    print("[+] Starting multi-hospital graph generation schema...")
    for db, path in DATA_MAPPING.items():
        ingest_hospital_data(db, path)
    print("\n[+] All federated nodes are populated and ready for GraphRAG runtime queries!")
