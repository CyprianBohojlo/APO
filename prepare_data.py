"""
Load the two FinanceBench JSONL files, merge them, keep the
OPEN_SOURCE split and save to a feather/parquet file for fast reuse.
"""
import pathlib, pandas as pd

ROOT = pathlib.Path(
    r"C:\Users\cypri\Desktop\Master Thesis"   # <= change here once
)
DATA = ROOT / "data" / "FinanceBench"
DATA.mkdir(parents=True, exist_ok=True) 
OUT  = DATA / "dataset_prepared.parquet"

def main():
    q  = pd.read_json(DATA / r"C:\Users\cypriDesktop\Master Thesis\data\FinanceBench\financebench_open_source.jsonl", lines=True)
    di = pd.read_json(DATA / r"C:\Users\cypri\Desktop\Master Thesis\data\FinanceBench\financebench_document_information.jsonl", lines=True)
    df = q.merge(di, on="doc_name").sort_values("doc_name")
    df.to_parquet(OUT, index=False)
    print(f"Dataset saved to {OUT}  ({len(df)} rows, {df.doc_name.nunique()} PDFs)")

if __name__ == "__main__":
    main()


"""
TODO
load data from Finder dataset using this chunk: import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/Linq-AI-Research/FinDER/data/train-00000-of-00001.parquet")
"""