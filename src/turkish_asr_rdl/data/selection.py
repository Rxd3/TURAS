"""Candidate screening and recording-grouped split selection for TURAS.

Input is the WorldSpeech ``tr_tr`` metadata table (every column except
``audio``, plus ``upstream_split``, ``shard`` and ``row_in_shard``). Every row
gets either a split or an exclusion reason; nothing disappears silently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .screening import Lexicon, count_quantities, screening_tokens, transcript_issues

LEGIS_TOPIC = "T_LEGIS"


def screen(meta: pd.DataFrame, lexicon: Lexicon) -> pd.DataFrame:
    """Add issue codes, matched terms/topics and quantity counts per row."""
    issues, terms, topics, n_econ, n_legis, n_qty, dup_key = [], [], [], [], [], [], []
    for text in meta["human_transcript"]:
        toks = screening_tokens(text)
        matches = lexicon.match(toks)
        q = count_quantities(text, toks, matches, lexicon)
        ids = sorted({m.term_id for m in matches})
        tops = sorted({lexicon.topic_of(t) for t in ids})
        issues.append("|".join(transcript_issues(text)))
        terms.append("|".join(ids))
        topics.append("|".join(tops))
        n_legis.append(sum(lexicon.topic_of(t) == LEGIS_TOPIC for t in ids))
        n_econ.append(len(ids) - n_legis[-1])
        n_qty.append(q["amount"] + q["percentage"])
        dup_key.append(" ".join(toks))
    out = meta.copy()
    out["issue_codes"] = issues
    out["term_ids"] = terms
    out["topic_ids"] = topics
    out["n_econ_terms"] = n_econ
    out["n_legis_terms"] = n_legis
    out["n_quantities"] = n_qty
    out["_dup_key"] = dup_key
    return out


def exclusion_reasons(screened: pd.DataFrame, config: dict) -> pd.Series:
    """First applicable exclusion reason per row ('' = eligible)."""
    f = config["filters"]
    rule = config["domain_rule"]
    reason = pd.Series("", index=screened.index, dtype=object)

    def mark(mask: pd.Series, code: str) -> None:
        reason[(reason == "") & mask] = code

    mark(~screened["source"].isin(f["sources"]), "source_not_selected")
    bad = set(f["exclude_issue_codes"])
    mark(
        screened["issue_codes"].map(lambda s: bool(bad & set(filter(None, s.split("|"))))),
        "non_spoken_or_truncated_text",
    )
    if f.get("exclude_repeated_text", True):
        mark(screened["_dup_key"].duplicated(keep=False), "repeated_text")
    mark(screened["cer"] > f["max_upstream_cer"], "upstream_cer_above_max")
    e, l, q = screened["n_econ_terms"], screened["n_legis_terms"], screened["n_quantities"]
    in_domain = (e >= rule["min_economic_terms_alone"]) | (
        (e >= rule["min_economic_terms_with_support"]) & ((l >= 1) | (q >= 1))
    )
    mark(~in_domain, "not_in_domain")
    return reason


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    score = df["n_econ_terms"] + df["n_legis_terms"].clip(upper=2) + df["n_quantities"].clip(upper=1)
    return df.assign(_score=score).sort_values(["_score", "segment_id"], ascending=[False, True])


def _take(df: pd.DataFrame, cap_s: float) -> list[str]:
    """Best-ranked segments of one recording, stopping before ``cap_s`` seconds."""
    csum = df["duration"].cumsum().to_numpy()
    return df["segment_id"].iloc[: int((csum <= cap_s).sum())].tolist()


def _taken_seconds(df: pd.DataFrame, cap_s: float) -> float:
    csum = df["duration"].cumsum().to_numpy()
    kept = csum[csum <= cap_s]
    return float(kept[-1]) if len(kept) else 0.0


def _waterfill_cap(groups: dict[str, pd.DataFrame], target_s: float) -> float:
    """Smallest per-recording cap whose capped total reaches ``target_s``.

    Spreading one cap over all recordings maximises session diversity. If the
    recordings cannot reach the target, the cap covers everything available.
    """
    lo, hi = 0.0, max((float(g["duration"].sum()) for g in groups.values()), default=0.0)
    if sum(_taken_seconds(g, hi) for g in groups.values()) < target_s:
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2
        if sum(_taken_seconds(g, mid) for g in groups.values()) >= target_s:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class Selection:
    table: pd.DataFrame  # every upstream row with turas_split / exclusion_reason
    caps_s: dict[str, float]


def select(screened: pd.DataFrame, config: dict) -> Selection:
    sp = config["splits"]
    key = sp["group_key"]
    t = screened.copy()
    t["exclusion_reason"] = exclusion_reasons(t, config)
    t["turas_split"] = ""
    eligible = t[t["exclusion_reason"] == ""]
    date = eligible.groupby(key)["session_date"].first()

    def groups_for(rec_ids) -> dict[str, pd.DataFrame]:
        return {r: _rank(eligible[eligible[key] == r]) for r in sorted(rec_ids)}

    test_cfg, val_cfg, train_cfg = sp["test"], sp["validation"], sp["train"]
    test_recs = set(date[date >= test_cfg["session_date_from"]].index)
    window = sorted(
        date[(date >= val_cfg["session_date_from"]) & (date <= val_cfg["session_date_to"])].index
    )
    random.Random(val_cfg["seed"]).shuffle(window)
    cap_val = sp["per_recording_cap_minutes"]["validation"] * 60
    val_recs, acc = [], 0.0
    for r in window:
        if acc >= val_cfg["target_hours"] * 3600:
            break
        val_recs.append(r)
        acc += _taken_seconds(_rank(eligible[eligible[key] == r]), cap_val)
    train_recs = set(date[date <= train_cfg["session_date_to"]].index) - set(val_recs) - test_recs

    caps: dict[str, float] = {}
    chosen: dict[str, str] = {}
    for split, recs, target_h in (
        ("test", test_recs, test_cfg["target_hours"]),
        ("validation", set(val_recs), val_cfg["target_hours"]),
        ("train", train_recs, train_cfg["target_hours"]),
    ):
        groups = groups_for(recs)
        if split == "train":
            # Upstream test rows never become TURAS training data.
            groups = {r: g[g["upstream_split"] != "test"] for r, g in groups.items()}
        cap = _waterfill_cap(groups, target_h * 3600)
        cap = min(cap, sp["per_recording_cap_minutes"][split] * 60)
        caps[split] = cap
        for g in groups.values():
            for sid in _take(g, cap):
                chosen[sid] = split

    t["turas_split"] = t["segment_id"].map(chosen).fillna("")
    rec_split = t[t.turas_split != ""].groupby(key)["turas_split"].agg(lambda s: next(iter(set(s))))
    in_rec = t[key].map(rec_split).fillna("")
    unchosen = (t["exclusion_reason"] == "") & (t["turas_split"] == "")
    quarantine = unchosen & (in_rec == "train") & (t["upstream_split"] == "test")
    t.loc[quarantine, "exclusion_reason"] = "upstream_test_in_train_recording"
    t.loc[unchosen & ~quarantine, "exclusion_reason"] = "eligible_not_sampled"
    return Selection(t.drop(columns=["_dup_key"]), caps)


def leakage_report(table: pd.DataFrame, key: str = "source_url") -> dict:
    """Checks that must all be zero/true before a release is proposed."""
    sel = table[table["turas_split"] != ""]
    per_rec = sel.groupby(key)["turas_split"].nunique()
    train_upstream_test = int(((sel.turas_split == "train") & (sel.upstream_split == "test")).sum())
    text_by_split = sel.assign(k=sel["human_transcript"].map(lambda s: " ".join(screening_tokens(s))))
    shared_text = int(text_by_split.groupby("k")["turas_split"].nunique().gt(1).sum())
    return {
        "recordings_in_more_than_one_split": int((per_rec > 1).sum()),
        "dates_in_more_than_one_split": int(sel.groupby("session_date")["turas_split"].nunique().gt(1).sum()),
        "train_rows_from_upstream_test": train_upstream_test,
        "normalized_texts_shared_across_splits": shared_text,
        "duplicate_segment_ids": int(sel["segment_id"].duplicated().sum()),
        "rows_without_split_or_reason": int(((table.turas_split == "") & (table.exclusion_reason == "")).sum()),
    }
