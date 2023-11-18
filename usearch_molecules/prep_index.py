"""Fingerprints the molecules, encoding them with 4 techniques, producing 28 Billions fingerprints for 7 Billion molecules."""

import os
import logging
from typing import List, Callable
from multiprocessing import Process, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind
from usearch.eval import self_recall, SearchStats

from usearch_molecules.metrics_numba import (
    tanimoto_conditional,
    tanimoto_maccs,
)
from usearch_molecules.dataset import (
    write_table,
    FingerprintedDataset,
    FingerprintedEntry,
)
from usearch_molecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    smiles_to_pubchem,
    shape_mixed,
    shape_maccs,
)

logger = logging.getLogger(__name__)


def augment_with_rdkit(parquet_path: os.PathLike):
    meta = pq.read_metadata(parquet_path)
    column_names: List[str] = meta.schema.names
    if "maccs" in column_names and "ecfp4" in column_names and "fcfp4" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    maccs_list = []
    ecfp4_list = []
    fcfp4_list = []
    for smiles in table["smiles"]:
        try:
            fingers = smiles_to_maccs_ecfp4_fcfp4(str(smiles))
            maccs_list.append(fingers[0].tobytes())
            ecfp4_list.append(fingers[1].tobytes())
            fcfp4_list.append(fingers[2].tobytes())
        except Exception:
            maccs_list.append(bytes(bytearray(21)))
            ecfp4_list.append(bytes(bytearray(256)))
            fcfp4_list.append(bytes(bytearray(256)))

    maccs_list = pa.array(maccs_list, pa.binary(21))
    ecfp4_list = pa.array(ecfp4_list, pa.binary(256))
    fcfp4_list = pa.array(fcfp4_list, pa.binary(256))
    maccs_field = pa.field("maccs", pa.binary(21), nullable=False)
    ecfp4_field = pa.field("ecfp4", pa.binary(256), nullable=False)
    fcfp4_field = pa.field("fcfp4", pa.binary(256), nullable=False)

    table = table.append_column(maccs_field, maccs_list)
    table = table.append_column(ecfp4_field, ecfp4_list)
    table = table.append_column(fcfp4_field, fcfp4_list)
    write_table(table, parquet_path)


def augment_with_cdk(parquet_path: os.PathLike):
    meta = pq.read_metadata(parquet_path)
    column_names: List[str] = meta.schema.names
    if "pubchem" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    pubchem_list = []
    for smiles in table["smiles"]:
        try:
            fingers = smiles_to_pubchem(str(smiles))
            pubchem_list.append(fingers[0].tobytes())
        except Exception:
            pubchem_list.append(bytes(bytearray(111)))

    pubchem_list = pa.array(pubchem_list, pa.binary(111))
    pubchem_field = pa.field("pubchem", pa.binary(111), nullable=False)

    table = table.append_column(pubchem_field, pubchem_list)
    write_table(table, parquet_path)


def augment_parquets_shard(
    parquet_dir: os.PathLike,
    augmentation: Callable,
    shard_index: int,
    shards_count: int,
):
    filenames: List[str] = sorted(os.listdir(parquet_dir))
    files_count = len(filenames)
    try:
        for file_idx in range(shard_index, files_count, shards_count):
            try:
                filename = filenames[file_idx]
                augmentation(os.path.join(parquet_dir, filename))
                logger.info(
                    "Augmented shard {}. Process # {} / {}".format(
                        filename, shard_index, shards_count
                    )
                )
            except KeyboardInterrupt as e:
                raise e

    except KeyboardInterrupt as e:
        logger.info(f"Stopping shard {shard_index} / {shards_count}")
        raise e


def augment_parquet_shards(
    parquet_dir: os.PathLike,
    augmentation: Callable,
    processes: int = 1,
):
    if processes > 1:
        process_pool = []
        for i in range(processes):
            p = Process(
                target=augment_parquets_shard,
                args=(parquet_dir, augmentation, i, processes),
            )
            p.start()
            process_pool.append(p)

        for p in process_pool:
            p.join()
    else:
        augment_parquets_shard(parquet_dir, augmentation, 0, 1)


def mono_index_maccs(dataset: FingerprintedDataset):
    index_path_maccs = os.path.join(dataset.dir, "index-maccs.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_maccs = Index(
        ndim=shape_maccs.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_maccs.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_maccs,
    )

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            if shard.first_key in index_maccs:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            table = shard.load_table(["maccs"])
            n = len(table)

            # No need to shuffle the entries as they already are:
            keys = np.arange(shard.first_key, shard.first_key + n)
            maccs_fingerprints = [table["maccs"][i].as_buffer() for i in range(n)]

            # First construct the index just for MACCS representations
            vectors = np.vstack(
                [
                    FingerprintedEntry.from_parts(
                        None,
                        maccs_fingerprints[i],
                        None,
                        None,
                        shape_maccs,
                    ).fingerprint
                    for i in range(n)
                ]
            )

            index_maccs.add(keys, vectors, log=f"Building {index_path_maccs}")

            # Optional self-recall evaluation:
            # stats: SearchStats = self_recall(index_maccs, sample=1000)
            # logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
            # logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
            if shard_idx % 100 == 0:
                index_maccs.save(index_path_maccs)

            # Discard the objects to save some memory
            dataset.shards[shard_idx].table_cached = None
            dataset.shards[shard_idx].index_cached = None

        index_maccs.save(index_path_maccs)
        index_maccs.reset()
    except KeyboardInterrupt:
        pass


def mono_index_mixed(dataset: FingerprintedDataset):
    index_path_mixed = os.path.join(dataset.dir, "index-maccs-ecfp4.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_mixed = Index(
        ndim=shape_mixed.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_conditional.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_mixed,
    )

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            if shard.first_key in index_mixed:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            table = shard.load_table(["maccs", "ecfp4"])
            n = len(table)

            # No need to shuffle the entries as they already are:
            keys = np.arange(shard.first_key, shard.first_key + n)
            maccs_fingerprints = [table["maccs"][i].as_buffer() for i in range(n)]
            ecfp4_fingerprints = [table["ecfp4"][i].as_buffer() for i in range(n)]

            # First construct the index just for MACCS representations
            vectors = np.vstack(
                [
                    FingerprintedEntry.from_parts(
                        None,
                        maccs_fingerprints[i],
                        ecfp4_fingerprints[i],
                        None,
                        shape_mixed,
                    ).fingerprint
                    for i in range(n)
                ]
            )

            index_mixed.add(keys, vectors, log=f"Building {index_path_mixed}")

            # Optional self-recall evaluation:
            # stats: SearchStats = self_recall(index_mixed, sample=1000)
            # logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
            # logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
            if shard_idx % 50 == 0:
                index_mixed.save(index_path_mixed)

            # Discard the objects to save some memory
            dataset.shards[shard_idx].table_cached = None
            dataset.shards[shard_idx].index_cached = None

        index_mixed.save(index_path_mixed)
        index_mixed.reset()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logger.info("Time to encode some molecules!")

    processes = max(cpu_count() - 4, 1)

    mono_index_maccs(FingerprintedDataset.open("data/pubchem"))
    mono_index_mixed(FingerprintedDataset.open("data/pubchem"))

    mono_index_maccs(FingerprintedDataset.open("data/gdb13"))
    mono_index_mixed(FingerprintedDataset.open("data/gdb13"))

    mono_index_maccs(FingerprintedDataset.open("data/real"))
    mono_index_mixed(FingerprintedDataset.open("data/real"))
