#  USearch Molecules

![USearch Molecules 7B datataset thumbnail](USearchMolecules.png)

"USearch Molecules" is a large Chem-Informatics dataset of small molecules under 50 "heavy" atoms.
It includes __7'131'914'291 molecules__ gathered from:

- 115'034'339 molecules from the __pubchem__ dataset.
- 977'468'301 molecules from the __gdb13__ dataset.
- 6'039'411'651 molecules from the __real__ dataset.

All molecules have been encoded using `rdkit` and `cdk`, to produce binary fingerprints of four kinds:

- __MACCS__: Molecular ACCess System keys with __166__ dimensions.
- __PubChem__: Structure Fingerprints with __881__ dimensions.
- __ECFP4__: Extended Connectivity Fingerprint of diameter 4 with __2048__ dimensions.
- __FCFP4__: Functional Class Fingerprint of diameter 4 with __2048__ dimensions.

Those fingerprints have then been indexed using [Unum's USearch](https://github.com/unum-cloud/usearch) to empower real-time search and clustering of moleculear structures for drug discovery and broader chemistry.
The dataset is included into [AWS OpenData platform](https://registry.opendata.aws/usearch-molecules/) and is publicly available in the `s3://usearch-molecules` bucket, accessible even without AWS credentials, entirely anonymously:

```sh
aws s3 ls --no-sign-request s3://usearch-molecules
```

## Dataset Structure

```sh
.
├── data
│   ├── pubchem
│   │   ├── index-maccs.usearch # 18.6 GB
│   │   ├── index-maccs-ecfp4.usearch # 46.1 GB
│   │   └── parquet # 30 GB
│   │       ├── 0000000000-0001000000.parquet # 265 MB
│   │       ├── 0001000000-0002000000.parquet # 265 MB
│   │       ├── ... 
│   │       └── 0115000000-0116000000.parquet # 177 MB
│   ├── gdb13
│   │   ├── index-maccs.usearch # 157.0 GB
│   │   ├── index-maccs-ecfp4.usearch # 390.1 GB
│   │   └── parquet # 189 GB
│   │       ├── 0000000000-0001000000.parquet # 198 MB
│   │       ├── 0001000000-0002000000.parquet # 198 MB
│   │       ├── ... 
│   │       └── 0977000000-0978000000.parquet # 93 MB
│   └── real
│       └── parquet # 477 GB
│           ├── 0000000000-0001000000.parquet # 262 MB
│           ├── 0001000000-0002000000.parquet # 262 MB
│           ├── ... 
│           └── 6039000000-6040000000.parquet # 108 MB
└── README.md
```

Pre-constructed search and clustering indexes for Enamine REAL dataset are much harder to distribute and deploy.
Those are not yet available in the bucket, but are available per request.
To view the dataset structure, one can use Python:

```sh
  $ pip install pyarrow
  $ python
>>> import pyarrow.parquet as pq
>>> pq.read_table('data/real/parquet/0000000000-0001000000.parquet')

pyarrow.Table
smiles: string not null
maccs: fixed_size_binary[21] not null
pubchem: fixed_size_binary[111] not null
ecfp4: fixed_size_binary[256] not null
fcfp4: fixed_size_binary[256] not null
```

In a tabular form that will look like:

|      | `smiles`                                                   |                                      `maccs` |                                                                                                                                                                                                                        `pubchem` |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            `ecfp4` |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            `fcfp4` |
| :--- | :--------------------------------------------------------- | -------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 0    | CNCC(C)NC(=O)C1(C(C)(C)OC)CC1                              | 0x00000200000002002021227C488B9C02100615FFCC | 0x00733000000000000000000000001800000000000000000000000000000000000000001E00100000000E6CC18006020002C004000800011010000000000000000000810800000040160080001400000636008000000000000F80000000000000000000000000000000000000000000 | 0x40000000000000000000800000002400000000000000000000000000000000000000001000000200000000000000000000800000000000000000000000000000000000000002000000000002000000000020000000000100000000000000000000000000010000000040000000000000000000020000000800000000000000000000000048000000000000000000000280200000000000000000020000000000000000000000000000000100000000000000020000000000000000000400000001000000000000000000000000000000010004000000000000000000000800000000000000000000800000000000000400000000000000000010000020000000 | 0xE0001400000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000001000401000000000000000000000000400000000000000000000001000000000000000000000100080000004000000000000000000000000000000000000000000000000000000000000000004000800000000000000000000000000001000000200000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000000004000000000000000001000000000000000000000080020000004000000000000000000000000000000000000000080 |
| 1    | CN(C(=O)C1=CC2=C(F)C=C(F)C=C2N1)C1CN(C(=O)CC2=CC=CN=C2O)C1 | 0x00900000002000004011172DAC534CE55EF3EB7FFC | 0x007BB1800000000000000000000000005801600000003C400000000000000001F000001F00100800000C28C19E0C3EC4F3C99200A8033577540082802037222008D921BC6CDC0866F2C295B394710864D611C8D987BE99809E00000000000200000000000000040000000000000000 | 0x00000000000001000000800000200100000100000000000000000000000000020000000000000000040000000008002000000000000000808000000000000000000200000000000000000001000000000020000000000014000000001000200100000000014040000000000000104000000000020100400000000000000040100000110040000000880000200000000000100000000000000400000000000000000000000000000104040000080000000000000000080000000100000000000000000000000000042000000000004000020000000000014000004200200000000000000000008000002040000000000400800000000000000000004001000000 | 0xBE800000000000000001000000000000000080000000080000000000000000000000000000000000000200000000000000000000000900000000000000010000000000010000000000020000000000000000000000000000000000200000000000000080080000000000000000000000040000008000000000002000000080000000000000400004000000000000000010000000000000000000000000000000000000400000000000000014000000000008000000000000000000000000000000000800000000000000000000000400080000000000001000400000000100000000000000000040004000000000002404000000000000000002020040003180 |

I've also added a tiny sample dataset under `data/example` directory, with only 2 shards totalling at 2 million entries, with pre-constructed indexes to simplify the entry.
Those come handy if you want to test your application without downloading the whole dataset, or visualize a few molecules using the StreamLit app.

```sh
.
└── data
    └── example # 1.8 GB
        ├── index-maccs.usearch # 329 MB
        ├── index-maccs-ecfp4.usearch # 817 MB
        ├── parquet # 30 GB
        │   ├── 0000000000-0001000000.parquet # 265 MB
        │   └── 0001000000-0002000000.parquet # 265 MB
        └── smiles # 30 GB
            ├── 0000000000-0001000000.smi # 58 MB
            └── 0001000000-0002000000.smi # 58 MB
```

## Usage

### Exploring Dataset via Command Line Interface

First, install NumPy, RDKit, and USearch v2, and download the dataset:

```sh
pip3 install git+https://github.com/ashvardanian/usearch-molecules.git@main
mkdir -p data/pubchem data/gdb13 data/real data/example
aws s3 sync --no-sign-request s3://usearch-molecules/data/example data/example/
```

If you need just one of the subsets:

```sh
aws s3 sync --no-sign-request s3://usearch-molecules/data/pubchem/ data/pubchem/
aws s3 sync --no-sign-request s3://usearch-molecules/data/gdb13/ data/gdb13/
aws s3 sync --no-sign-request s3://usearch-molecules/data/real/ data/real/
```

You can immediately check if the indexes are readable:

```sh
  $ python
>>> from usearch.index import Index
>>> Index.metadata("data/pubchem/index-maccs.usearch") # example of reading metadata

{'matrix_included': True,
 'matrix_uses_64_bit_dimensions': False,
 'version': '2.8.10',
 'kind_metric': <MetricKind.Tanimoto: 116>,
 'kind_scalar': <ScalarKind.B1: 1>,
 'kind_key': <ScalarKind.U64: 8>,
 'kind_compressed_slot': <ScalarKind.U32: 9>,
 'count_present': 115627267,
 'count_deleted': 0,
 'dimensions': 192}

>>> Index.restore("data/pubchem/index-maccs-ecfp4.usearch") # example of parsing it

usearch.Index
- config
-- data type: ScalarKind.B1
-- dimensions: 2240
-- metric: MetricKind.Tanimoto
-- connectivity: 16
-- expansion on addition:128 candidates
-- expansion on search: 64 candidates
- binary
-- uses OpenMP: 1
-- uses SimSIMD: 1
-- uses hardware acceleration: avx512+popcnt
- state
-- size: 115,627,267 vectors
-- memory usage: 69,631,939,864 bytes
-- max level: 4
--- 0. 115,627,267 nodes
--- 1. 7,148,410 nodes
--- 2. 461,450 nodes
--- 3. 37,714 nodes
--- 4. 5,152 nodes
```

With those out of the way, you can now query the downloaded files:

```py
smiles = [
  'CC(O)C(CN)=NNCC(C)(C)C',
  'C=C(C#C)C1=C2CN3CC3C2=NO1',
  'CCC1C(=O)NC2(CCCC2)C(=O)N1CCC#C',
]
```

## Exploring Dataset via Graphical Interface

```sh
pip install streamlit stmol ipython_genutils
streamlit run streamlit_app.py
```


## Predict Molecule Properties using BARTSmiles

```py
!git clone https://github.com/YerevaNN/BARTSmiles.git
!pip install -r BARTSmiles/requirements.txt
!pip install fairseq

```

## Methodology

### Dataset Sources

Original data came from:

- __PubChem__: [CID-SMILES](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz).gz
- __GDB13__: [gdb13](https://zenodo.org/record/5172018/files/gdb13.tgz?download=1).tgz
- Enamine __REAL__, split by Heavy Atom Counts:
    - HAC 6-21: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_6_21_420M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 22-23: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_22_23_471M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 24: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 25: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_25_557M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 26:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_26_833M_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_26_833M_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 27:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_27_1.1B_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_27_1.1B_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 28:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_28_1.2B_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_28_1.2B_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 29-38:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_29_38_988M_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_29_38_988M_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2

### Pre-processing Pipeline

1. `prep_schedule.py`: convert and split datasets into standardized Parquet files.
2. `prep_encode.py`: produce MACCS, PubChem, ECFP4, and FCFP4 fingerprints and index those.
3. `prep_smiles.py`: export newline-delimited `.smi` files to simplify navigation with [StringZilla][stringzilla].

Every script is designed to work with bigger-than-memory data.
In other words, processing 1 TB of molecules doesn't require 1 TB of RAM.
Everything happens in a "gliding-window" fashion, with computationally intensive parts split between processes and threads.

```sh
python prep_schedule.py # Prepare Parquet files
python prep_encode.py # Build USearch indexes
python prep_smiles.py # Export SMILES new-line delimited files to simplify serving
```

Once completed, datasets have been uploaded to S3:

```sh
aws s3 sync data/pubchem/parquet/ s3://usearch-molecules/data/pubchem/parquet/
aws s3 sync data/gdb13/parquet/ s3://usearch-molecules/data/gdb13/parquet/
aws s3 sync data/real/parquet/ s3://usearch-molecules/data/real/parquet/
```

[stringzilla]: https://github.com/ashvardanian/stringzilla

