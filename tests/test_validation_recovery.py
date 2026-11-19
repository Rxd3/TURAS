"""Synthetic identity/text fixtures; never fetch the real dataset in tests."""
import pytest

from turkish_asr_rdl.data.recover_validation import make_reference, VARIANT
from turkish_asr_rdl.data.release import utterance_id


def fixture_rows():
    source = {"source_url": "https://example.invalid/synthetic-recording", "segment_id": "synthetic-segment",
              "source_start_s": 1.0, "source_end_s": 11.0, "duration": 10.0,
              "human_transcript": "SYNTHETIC human fixture: I \u0130 2026.",
              "asr_transcript": "SYNTHETIC MACHINE TEXT MUST NEVER BECOME GOLD"}
    inference = {"utterance_id": utterance_id(source["source_url"], source["segment_id"], 1000, 11000),
                 "audio_uri": "audio/synthetic.flac", "audio_sha256": "a" * 64,
                 "sample_rate_hz": 16000, "channels": 1, "duration_s": 10.0}
    return source, inference


def test_preserves_actual_human_text_and_never_fabricates_review_status():
    source, inference = fixture_rows()
    row = make_reference(source, inference, shard="train-00000.parquet", row_index=7)
    assert row["reference_text"] == source["human_transcript"]
    assert row["reference_variant"] == VARIANT
    assert row["verification_status"] == "unreviewed" and row["verification_record_id"] is None
    assert row["split"] == "validation" and row["upstream_split"] == "train"
    assert row["audio_sha256"] == inference["audio_sha256"]
    del source["human_transcript"]
    with pytest.raises(ValueError, match="ASR text is never substituted"):
        make_reference(source, inference, shard="train-00000.parquet", row_index=7)


def test_refuses_wrong_identity_and_audio_duration():
    source, inference = fixture_rows()
    with pytest.raises(ValueError, match="identity"):
        make_reference({**source, "segment_id": "wrong"}, inference, shard="train-00000.parquet", row_index=7)
    with pytest.raises(ValueError, match="duration mismatch"):
        make_reference({**source, "duration": 20.0}, inference, shard="train-00000.parquet", row_index=7)
