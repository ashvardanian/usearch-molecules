import streamlit as st
from stmol import showmol
import py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem

from shared import FingerprintedDataset, shape_mixed
from usearch.eval import measure_seconds

st.set_page_config(layout="wide")
st.title("USearch Molecules")

max_results = 12
results_per_row = 4

search_pubchem = st.checkbox("PubChem")
search_gdb13 = st.checkbox("GDB13")
search_real = st.checkbox("REAL")


@st.cache_resource
def get_dataset():
    data = FingerprintedDataset.open(
        "data/pubchem",
        shape=shape_mixed,
        limit=10,
    )
    return data


data = get_dataset()


def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock


def render_mol(xyz):
    xyzview = py3Dmol.view()  # (width=400,height=400)
    xyzview.addModel(xyz, "mol")
    xyzview.setStyle({"stick": {}})
    xyzview.setBackgroundColor("black")
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=500)


molecule = st.text_input("SMILES please", "CC")
render_mol(makeblock(molecule))


seconds, results = measure_seconds(
    lambda: data.search(
        molecule,
        max_results,
        log=st.progress,
    )
)
st.markdown(f"Found {len(results)} in {seconds:.3f} seconds")

for match_idx, match in enumerate(results):
    col_idx = match_idx % results_per_row
    if col_idx == 0:
        st.write("---")
        cols = st.columns(results_per_row, gap="large")

    with cols[col_idx]:
        render_mol(makeblock(match[1]))
        st.write(match[1])
