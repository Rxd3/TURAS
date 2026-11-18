"""Build a TURAS release from a selection table: IDs, audio, manifests, split files.

Protected outputs (references, exclusion ledger, audio) go under an artifact
root outside Git. Small public outputs (split ID lists, split manifest, source
registry, release summary) go under the repository ``data/`` tree.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCHEMA_VERSION = "0.1"
ANNOTATION_POLICY_VERSION = "turas-verbatim-v0.1"
SPLITS = ("train", "validation", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_id(source_url: str) -> str:
    """Opaque stable source ID from the canonical recording URL."""
    return "src_" + sha256_bytes(f"worldspeech-tr_tr|{source_url}".encode())[:12]


def utterance_id(source_url: str, segment_id: str, start_ms: int, end_ms: int) -> str:
    """Opaque stable utterance ID from source, upstream segment and offsets (never text)."""
    key = f"worldspeech-tr_tr|{source_url}|{segment_id}|{start_ms}|{end_ms}"
    return "utt_" + sha256_bytes(key.encode())[:16]


def to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def assign_ids(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    out["start_ms"] = out["source_start_s"].map(to_ms)
    out["end_ms"] = out["source_end_s"].map(to_ms)
    out["source_id"] = out["source_url"].map(source_id)
    out["utterance_id"] = [
        utterance_id(u, s, a, b)
        for u, s, a, b in zip(out.source_url, out.segment_id, out.start_ms, out.end_ms)
    ]
    if out["utterance_id"].duplicated().any():
        raise ValueError("utterance_id collision")
    return out


def encode_flac_16k_mono(opus_bytes: bytes) -> tuple[bytes, int, int, float]:
    """Decode upstream OGG Opus, downmix (channel mean), resample to 16 kHz, FLAC PCM_16.

    Matches the shared loader's conversion (float32 mean, soxr HQ) so every
    system reads the same signal. Returns (flac_bytes, sample_rate, channels, duration_s).
    """
    import soundfile as sf
    import soxr

    x, sr = sf.read(io.BytesIO(opus_bytes), dtype="float32", always_2d=True)
    mono = x.mean(axis=1, dtype=np.float32)
    if sr != 16000:
        mono = soxr.resample(mono, sr, 16000, quality="HQ")
    mono = np.clip(mono, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, mono, 16000, format="FLAC", subtype="PCM_16")
    data = buf.getvalue()
    y, sr2 = sf.read(io.BytesIO(data), always_2d=True)
    return data, sr2, y.shape[1], y.shape[0] / sr2


def shard_requests(selected: pd.DataFrame) -> list[tuple[str, list[int]]]:
    """(shard, sorted row indexes) pairs needed for the selected rows."""
    return [
        (shard, sorted(int(r) for r in grp["row_in_shard"]))
        for shard, grp in selected.groupby("shard", sort=True)
    ]


def write_audio(
    selected: pd.DataFrame,
    artifact_root: Path,
    shard_payloads: Iterable[tuple[str, list[tuple[int, bytes]]]],
) -> pd.DataFrame:
    """Write ``audio/<utterance_id>.flac`` for every selected row and measure it.

    ``shard_payloads`` yields (shard, [(row_in_shard, opus_bytes), ...]); it is
    injected so tests and offline rebuilds need no network.
    """
    (artifact_root / "audio").mkdir(parents=True, exist_ok=True)
    key = {(s, int(r)): u for s, r, u in zip(selected.shard, selected.row_in_shard, selected.utterance_id)}
    measured: dict[str, tuple[str, str, int, int, float]] = {}
    for shard, payload in shard_payloads:
        for row, opus in payload:
            uid = key[(shard, int(row))]
            flac, sr, ch, dur = encode_flac_16k_mono(opus)
            rel = f"audio/{uid}.flac"
            (artifact_root / rel).write_bytes(flac)
            measured[uid] = (rel, sha256_bytes(flac), sr, ch, dur)
    missing = set(selected["utterance_id"]) - set(measured)
    if missing:
        raise RuntimeError(f"{len(missing)} selected rows produced no audio")
    out = selected.copy()
    for i, c in enumerate(["audio_uri", "audio_sha256", "sample_rate_hz", "channels", "duration_s"]):
        out[c] = out["utterance_id"].map(lambda u, i=i: measured[u][i])
    return out


def reference_rows(rel: pd.DataFrame, config: dict) -> list[dict]:
    up = config["upstream"]
    rows = []
    for r in rel.sort_values("utterance_id").itertuples(index=False):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_release": config["release_id"],
                "utterance_id": r.utterance_id,
                "source_id": r.source_id,
                "source_segment_id": r.segment_id,
                "upstream_dataset": up["dataset"],
                "upstream_config": up["config"],
                "upstream_revision": up["revision"],
                "upstream_split": r.upstream_split,
                "source_url": r.source_url,
                "source_recording_key": r.source_url.rsplit("v=", 1)[-1],
                "start_ms": int(r.start_ms),
                "end_ms": int(r.end_ms),
                "audio_uri": r.audio_uri,
                "audio_sha256": r.audio_sha256,
                "sample_rate_hz": int(r.sample_rate_hz),
                "channels": int(r.channels),
                "duration_s": round(float(r.duration_s), 6),
                "language": "tr_tr",
                "domain_topics": [t for t in r.topic_ids.split("|") if t],
                "domain_topics_origin": "lexicon-screening-unverified",
                "speaker_ids": None,
                "session_id": r.source_id,
                "duplicate_cluster_id": r.utterance_id,
                "leakage_group_id": r.source_id,
                "reference_text": r.human_transcript,
                "reference_origin": "upstream_human",
                "verification_status": "unreviewed",
                "verification_record_id": None,
                "annotation_policy_version": ANNOTATION_POLICY_VERSION,
                "split": r.turas_split,
                "rights_record_id": "rights_worldspeech_tr_tr_tbmm",
            }
        )
    return rows


def inference_rows(rel: pd.DataFrame, config: dict, split: str, split_manifest_sha: str) -> list[dict]:
    """Strict allowlist rows in split-file (sorted ID) order."""
    part = rel[rel.turas_split == split].sort_values("utterance_id")
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_release": config["release_id"],
            "split_manifest_sha256": split_manifest_sha,
            "utterance_id": r.utterance_id,
            "audio_uri": r.audio_uri,
            "audio_sha256": r.audio_sha256,
            "duration_s": round(float(r.duration_s), 6),
            "sample_rate_hz": int(r.sample_rate_hz),
            "channels": int(r.channels),
            "language": "tr_tr",
        }
        for r in part.itertuples(index=False)
    ]


def write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    path.write_bytes(data.encode("utf-8"))
    return sha256_file(path)


def write_split_files(rel: pd.DataFrame, split_dir: Path) -> dict[str, dict]:
    split_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for split in SPLITS:
        part = rel[rel.turas_split == split]
        ids = sorted(part["utterance_id"])
        p = split_dir / f"{split}.txt"
        p.write_bytes(("\n".join(ids) + "\n").encode("utf-8"))
        info[split] = {
            "file": f"{split}.txt",
            "sha256": sha256_file(p),
            "utterances": len(ids),
            "duration_s": round(float(part["duration_s"].sum()), 3),
            "recordings": int(part["source_id"].nunique()),
            "session_date_min": str(part["session_date"].min()),
            "session_date_max": str(part["session_date"].max()),
            "upstream_test_rows": int((part["upstream_split"] == "test").sum()),
            "ordered_id_digest": sha256_bytes("\n".join(ids).encode()),
        }
    all_sets = [set(rel[rel.turas_split == s]["utterance_id"]) for s in SPLITS]
    for i in range(3):
        for j in range(i + 1, 3):
            if all_sets[i] & all_sets[j]:
                raise ValueError("split files overlap")
    return info


def topic_hours(rel: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        c: Counter = Counter()
        for topics, dur in zip(rel[rel.turas_split == split]["topic_ids"], rel[rel.turas_split == split]["duration_s"]):
            for t in filter(None, topics.split("|")):
                c[t] += dur / 3600
        out[split] = {k: round(v, 3) for k, v in sorted(c.items())}
    return out
