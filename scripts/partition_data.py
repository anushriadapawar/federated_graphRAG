import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/federated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_and_partition_data():
    info_path = os.path.join(RAW_DIR, "rare_diseases_info.csv")
    genes_path = os.path.join(RAW_DIR, "rare_diseases_genes.csv")
    history_path = os.path.join(RAW_DIR, "rare_diseases_natural_history.csv")

    for p in [info_path, genes_path, history_path]:
        if not os.path.exists(p):
            print(f"[-] Missing file: {p}. Make sure files match the Kaggle names perfectly.")
            return

    print("[+] Reading Kaggle Orphadata files...")
    df_info = pd.read_csv(info_path)
    df_genes = pd.read_csv(genes_path)
    df_history = pd.read_csv(history_path)

    # Clean space paddings out of column headers
    for df in [df_info, df_genes, df_history]:
        df.columns = df.columns.str.strip()

    print("[+] Merging data streams on OrphaCode...")
    # Step-by-step merge avoiding column name namespace collisions
    merged = pd.merge(df_info, df_genes, on="OrphaCode", how="inner")
    if "Name" in merged.columns and "DiseaseName" in merged.columns:
        merged.drop(columns=["DiseaseName"], inplace=True)
        
    final_df = pd.merge(merged, df_history, on="OrphaCode", how="inner")
    if "DiseaseName" in final_df.columns:
        final_df.drop(columns=["DiseaseName"], inplace=True)

    print(f"[✓] Unified master dataset contains {len(final_df)} completely mapped rows.")

        # Shuffle and split into your isolated nodes safely using pure Pandas
    df_shuffled = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate clean mathematical split indexes for 3 chunks
    chunk_size = int(np.ceil(len(df_shuffled) / 3))
    
    hospitals = ['hospital_A.csv', 'hospital_B.csv', 'hospital_C.csv']
    
    for i, filename in enumerate(hospitals):
        # Slice the dataframe rows cleanly using pandas tracking
        chunk = df_shuffled.iloc[i * chunk_size : (i + 1) * chunk_size]
        
        out_path = os.path.join(OUTPUT_DIR, filename)
        chunk.to_csv(out_path, index=False)
        print(f"[+] Successfully wrote {len(chunk)} rows to {out_path} (Size: {os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    merge_and_partition_data()
