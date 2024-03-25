from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
except ImportError:
    print("Can't fingerprint molecules without RDKit and JPype")


def molecule_to_maccs(x):
    return MACCSkeys.GenMACCSKeys(x)


def molecule_to_ecfp4(x):
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)


def molecule_to_fcfp4(x):
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True)


def smiles_to_maccs_ecfp4_fcfp4(
    smiles: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses RDKit to simultaneously compute MACCS, ECFP4, and FCFP4 representations."""

    molecule = Chem.MolFromSmiles(smiles)
    return (
        np.packbits(molecule_to_maccs(molecule)),
        np.packbits(molecule_to_ecfp4(molecule)),
        np.packbits(molecule_to_fcfp4(molecule)),
    )


_cdk = None
_cdk_smiles_parser = None
_cdk_fingerprinter = None


def smiles_to_pubchem(smiles: str) -> Tuple[np.ndarray]:
    """Uses Chemistry Development Kit to compute PubChem representations."""
    global _cdk
    global _cdk_smiles_parser
    global _cdk_fingerprinter

    if not isJVMStarted():
        cdk_path = os.path.join(os.getcwd(), "cdk-2.2.jar")
        startJVM(getDefaultJVMPath(), "-Djava.class.path=%s" % cdk_path)
        _cdk = JPackage("org").openscience.cdk

    if _cdk_smiles_parser is None:
        _cdk_smiles_parser = _cdk.smiles.SmilesParser(
            _cdk.DefaultChemObjectBuilder.getInstance()
        )

    if _cdk_fingerprinter is None:
        _cdk_fingerprinter = _cdk.fingerprint.PubchemFingerprinter(
            _cdk.silent.SilentChemObjectBuilder.getInstance()
        )

    molecule = _cdk_smiles_parser.parseSmiles(smiles)
    cdk_fingerprint = _cdk_fingerprinter.getBitFingerprint(molecule)
    cdk_set_bits = list(cdk_fingerprint.getSetbits())
    bitset = np.zeros(881, dtype=np.uint8)
    bitset[cdk_set_bits] = 1
    bitset = np.packbits(bitset)
    return (bitset,)


_coati2_model = None
_coati2_tokenizer = None


def smiles_to_coati2(smiles: List[str], device: str = None) -> np.ndarray:
    """Uses COATI2 embedding model for small molecules."""
    global _coati2_model
    global _coati2_tokenizer

    # COATI dependencies are quite heavy. Lets avoid them until this functioned is called.
    if _coati2_model is None:
        import torch
        from coati.models.simple_coati2.io import load_coati2
        from coati.generative.coati_purifications import embed_smiles_batch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model parameters are pulled from the url and stored in a local models/ dir.
        # Tutorial: https://github.com/terraytherapeutics/COATI/blob/main/examples/coati2/tutorial.ipynb
        encoder, tokenizer = load_coati2(
            freeze=True,
            device=torch.device(device),
            doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl",
        )
        _coati2_model = encoder
        _coati2_tokenizer = tokenizer
    else:
        from coati.generative.coati_purifications import embed_smiles_batch

    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = [Chem.CanonSmiles(smile) for smile in smiles]
    vecs = embed_smiles_batch(smiles, _coati2_model, _coati2_tokenizer)
    vecs = vecs.cpu().numpy()
    return vecs


@dataclass
class FingerprintShape:
    """Represents the shape of a hybrid fingerprint, potentially containing multiple concatenated bit-vectors."""

    include_maccs: bool = False
    include_ecfp4: bool = False
    include_fcfp4: bool = False
    nbytes_padding: int = 0

    @property
    def nbytes(self) -> int:
        return (
            self.include_maccs * 21
            + self.nbytes_padding
            + self.include_ecfp4 * 256
            + self.include_fcfp4 * 256
        )

    @property
    def nbits(self) -> int:
        return self.nbytes * 8

    @property
    def index_name(self) -> str:
        parts = ["index"]
        if self.include_maccs:
            parts.append("maccs")
        if self.include_ecfp4:
            parts.append("ecfp4")
        if self.include_fcfp4:
            parts.append("fcfp4")
        return "-".join(parts) + ".usearch"


shape_maccs = FingerprintShape(
    include_maccs=True,
    nbytes_padding=3,
)

shape_mixed = FingerprintShape(
    include_maccs=True,
    include_ecfp4=True,
    nbytes_padding=3,
)
