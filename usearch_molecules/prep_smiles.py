"""Exports the strings column with SMILES from Parquet files into newline-delimited files for simpler parsing with StringZilla."""

import os

from tqdm import tqdm
from stringzilla import File, Str

from usearch_molecules.dataset import FingerprintedDataset


def export_smiles(data):
    for shard in tqdm(data.shards):
        table = shard.load_table(["smiles"])
        smiles_path = str(shard.table_path)
        smiles_path = smiles_path.replace(".parquet", ".smi")
        smiles_path = smiles_path.replace("/parquet/", "/smiles/")
        if os.path.exists(smiles_path):
            continue

        with open(smiles_path, "w") as f:
            for line in table["smiles"]:
                f.write(str(line) + "\n")

        smiles_file = File(smiles_path)
        reconstructed = smiles_file.splitlines()
        for row, line in enumerate(table["smiles"]):
            assert str(reconstructed[row]) == str(line)
        shard.table_cached = None


if __name__ == "__main__":
    for dataset in ["example", "pubchem", "gdb13", "real"]:
        if not os.path.exists(f"data/{dataset}"):
            continue
        export_smiles(FingerprintedDataset.open(f"data/{dataset}"))
