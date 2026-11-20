"""Pinned-revision access to WorldSpeech ``tr_tr`` parquet shards (network required).

Only used by the release builder; unit tests never call these functions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator

REPO = "datasets/disco-eth/WorldSpeech"


def _fs():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def list_shards(revision: str, config: str = "tr_tr") -> list[str]:
    fs = _fs()
    paths = fs.ls(f"{REPO}@{revision}/data/{config}", detail=False)
    return sorted(p.rsplit("/", 1)[-1] for p in paths if p.endswith(".parquet"))


def fetch_metadata(revision: str, config: str = "tr_tr", workers: int = 16):
    """All columns except ``audio`` for every shard, plus provenance columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def one(shard: str):
        with _fs().open(f"{REPO}@{revision}/data/{config}/{shard}", "rb", block_size=8 << 20) as f:
            pf = pq.ParquetFile(f)
            t = pf.read(columns=[c for c in pf.schema_arrow.names if c != "audio"])
        n = t.num_rows
        t = t.append_column("upstream_split", pa.array([shard.split("-")[0]] * n))
        t = t.append_column("shard", pa.array([shard] * n))
        return t.append_column("row_in_shard", pa.array(range(n), pa.int32()))

    with ThreadPoolExecutor(workers) as ex:
        tables = list(ex.map(one, list_shards(revision, config)))
    return pa.concat_tables(tables).to_pandas()


def shard_audio_reader(revision: str, config: str = "tr_tr"):
    """Return ``read(shard, rows) -> iterator of (row, opus_bytes)`` for write_audio."""
    import pyarrow.parquet as pq

    def read(shard: str, rows: list[int]) -> Iterator[tuple[int, bytes]]:
        with _fs().open(f"{REPO}@{revision}/data/{config}/{shard}", "rb", block_size=32 << 20) as f:
            col = pq.ParquetFile(f).read(columns=["audio"]).column("audio")
        for r in rows:
            yield r, col[r].as_py()["bytes"]

    return read


def prefetching(read, shards_rows: Iterable[tuple[str, list[int]]], workers: int = 8):
    """Read several shards in parallel; yields (shard, list of (row, bytes))."""
    with ThreadPoolExecutor(workers) as ex:
        futs = [(s, ex.submit(lambda s=s, r=r: list(read(s, r)))) for s, r in shards_rows]
        for s, fut in futs:
            yield s, fut.result()
