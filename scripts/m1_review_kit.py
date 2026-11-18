"""Make a self-contained listening-review page for TURAS verification (Member 1).

    python scripts/m1_review_kit.py --artifact-root ../turas-artifacts --pilot \
        --out ../turas-artifacts/review/pilot-review.html

--pilot draws 50 utterances (25 test, 15 validation, 10 train; at most one per
recording; seed 2026) and writes their IDs to data/verification/. Without
--pilot, pass --split and optionally --ids-file to review a whole split.
The page embeds the FLAC audio and the reference text, so it is a protected
file: share it only with reviewers, never commit it.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = "turas-tr-econ-legis-v0.1"
PILOT = {"test": 25, "validation": 15, "train": 10}


def load_refs(root: Path) -> list[dict]:
    path = root / RELEASE / f"{RELEASE}.references.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pilot_sample(refs: list[dict], seed: int = 2026) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for split, n in PILOT.items():
        by_rec: dict[str, list[dict]] = {}
        for r in refs:
            if r["split"] == split:
                by_rec.setdefault(r["source_id"], []).append(r)
        recs = sorted(by_rec)
        rng.shuffle(recs)
        for rec in recs[:n]:
            rows = sorted(by_rec[rec], key=lambda r: r["utterance_id"])
            out.append(rng.choice(rows))
    return out


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TURAS Review</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deded8;--acc:#2563eb;--ok:#15803d;--warn:#b45309;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--fg:#ececea;--mut:#a3a39e;--card:#1f1f1d;--line:#3a3a36;--acc:#60a5fa;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
header{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:2}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.card.done{border-left:5px solid var(--ok)}.card.rej{border-left:5px solid var(--bad)}.card.cor{border-left:5px solid var(--warn)}
.meta{color:var(--mut);font-size:13px;display:flex;gap:12px;flex-wrap:wrap}
textarea{width:100%;box-sizing:border-box;min-height:70px;font:16px/1.5 system-ui;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
audio{width:100%;margin:8px 0}
label{margin-right:12px;white-space:nowrap}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.ar{direction:rtl;text-align:right;color:var(--mut);font-size:14px}
input[type=text]{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
</style></head><body><main>
<header><strong>TURAS listening review</strong> · policy turas-verbatim-v0.1 · <span id="prog"></span><br>
Reviewer ID <input type="text" id="rev" value="rev_m1" size="8"> <button class="primary" onclick="exportJson()">Export decisions (JSON)</button>
<span class="meta">Decisions autosave in this browser.</span></header>
<p class="ar">استمع للمقطع كاملاً وقارن النص كلمة بكلمة. <b>accept</b> إذا النص مطابق، <b>correct</b> إذا الكلام واضح والنص فيه خطأ (صحّح النص في المربع)، <b>reject</b> إذا الكلام غير واضح أو مقطوع أو لا تتأكد. لا تخمّن أبداً. بالنهاية اضغط Export وأرسل الملف.</p>
<p>Rules: write what is spoken; keep digits as written (<code>5 milyar lira</code>, <code>yüzde 42</code>); remove non-spoken labels; never guess — reject instead. Full guide: data/verification/turas-verbatim-v0.1.md</p>
<div id="list"></div></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "turas-review-__KIT__";
const ISSUES = ["boundary_cut","overlap","noise","edited_minutes","wrong_segment","non_speech","other"];
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} prog(); }
function prog(){ const n = Object.values(state).filter(s => s.decision).length;
  document.getElementById("prog").textContent = n + " / " + ITEMS.length + " decided"; }
function esc(s){ const d = document.createElement("div"); d.textContent = s; return d.innerHTML.replace(/"/g, "&quot;"); }
const list = document.getElementById("list");
ITEMS.forEach((it, i) => {
  const sk = it.utterance_id + "|" + it.audio_sha256;  // saved work never carries over to rebuilt audio
  const s = state[sk] || (state[sk] = {text: it.reference_text, issues: []});
  const c = document.createElement("div"); c.className = "card"; c.id = "c" + i;
  c.innerHTML = `<div class="meta"><b>#${i + 1}</b><span>${esc(it.split)}</span><span>${esc(it.session_date)}</span>
    <span>${esc(it.utterance_id)}</span><a href="${esc(it.src)}" target="_blank" rel="noopener">source video at ${esc(String(it.t))}s</a></div>
    <audio controls preload="none" src="data:audio/flac;base64,${it.audio}"></audio>
    <div class="meta">Upstream reference (edit only when choosing correct):</div>
    <textarea>${esc(s.text)}</textarea>
    <div>${["accept","correct","reject"].map(d => `<label><input type="radio" name="d${i}" value="${d}" ${s.decision === d ? "checked" : ""}> ${d}</label>`).join("")}</div>
    <div class="meta">${ISSUES.map(x => `<label><input type="checkbox" value="${x}" ${s.issues.includes(x) ? "checked" : ""}> ${x}</label>`).join("")}</div>
    <input type="text" placeholder="note (optional)" style="width:100%;box-sizing:border-box;margin-top:6px" value="${esc(s.note || "")}">`;
  list.appendChild(c);
  const paint = () => { c.className = "card" + (s.decision === "accept" ? " done" : s.decision === "reject" ? " rej" : s.decision === "correct" ? " cor" : ""); };
  c.querySelector("textarea").oninput = e => { s.text = e.target.value; save(); };
  c.querySelectorAll("input[type=radio]").forEach(r => r.onchange = () => { s.decision = r.value; s.ts = new Date().toISOString(); paint(); save(); });
  c.querySelectorAll("input[type=checkbox]").forEach(b => b.onchange = () => { s.issues = [...c.querySelectorAll("input[type=checkbox]:checked")].map(x => x.value); save(); });
  c.querySelector("input[type=text]").oninput = e => { s.note = e.target.value; save(); };
  paint();
});
prog();
function exportJson(){
  const rev = document.getElementById("rev").value.trim() || "rev_unknown";
  const out = ITEMS.map(it => { const s = state[it.utterance_id + "|" + it.audio_sha256];
    return {utterance_id: it.utterance_id, audio_sha256: it.audio_sha256, reviewer_id: rev,
      review_utc: s.ts || null, annotation_policy_version: "turas-verbatim-v0.1", decision: s.decision || null,
      corrected_text: s.decision === "correct" ? s.text : null, text_changed: s.text !== it.reference_text,
      issue_codes: s.issues, note: s.note || ""}; });
  const blob = new Blob([JSON.stringify({kit: "__KIT__", reviews: out}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "__KIT__-decisions-" + rev + ".json"; a.click();
}
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--split", choices=["validation", "test", "train"])
    ap.add_argument("--ids-file", type=Path)
    args = ap.parse_args()

    refs = load_refs(args.artifact_root)
    if args.pilot:
        items, kit = pilot_sample(refs), "pilot-v0.1"
        ids_path = REPO / "data/verification" / f"{RELEASE}-pilot-ids.txt"
        ids_path.write_text("\n".join(sorted(r["utterance_id"] for r in items)) + "\n", encoding="utf-8")
    elif args.split:
        wanted = None
        if args.ids_file:
            wanted = set(args.ids_file.read_text(encoding="utf-8").split())
        items = [r for r in refs if r["split"] == args.split and (wanted is None or r["utterance_id"] in wanted)]
        items.sort(key=lambda r: r["utterance_id"])
        kit = f"{args.split}-v0.1"
    else:
        print("choose --pilot or --split", file=sys.stderr)
        return 2

    registry = {}
    with open(REPO / "data/metadata" / f"{RELEASE}-source-registry.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            sid, url, date, _ = line.rstrip("\n").split(",")
            registry[sid] = date
    payload = []
    for r in items:
        audio = (args.artifact_root / RELEASE / r["audio_uri"]).read_bytes()
        t = r["start_ms"] // 1000
        payload.append({
            "utterance_id": r["utterance_id"], "split": r["split"], "session_date": registry[r["source_id"]],
            "src": f"{r['source_url']}&t={t}s", "t": t, "audio_sha256": r["audio_sha256"],
            "reference_text": r["reference_text"], "audio": base64.b64encode(audio).decode("ascii"),
        })
    page = PAGE.replace("__ITEMS__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__KIT__", html.escape(kit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"{len(items)} items -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
