# TURAS presentation

Use the [four-slide PowerPoint](TURAS-2-minute.pptx) for the two-minute slot.
Its speaker notes contain a timed talk track.

| Time | Slide |
| --- | --- |
| 0:00–0:20 | Research question |
| 0:20–0:45 | Frozen test set |
| 0:45–1:25 | Word error rate gains |
| 1:25–1:50 | Graph rescoring result and study limits |
| 1:50–2:00 | Optional verification or transition time |

For a live check, pre-open a terminal at the repository root and run:

```text
python scripts/verify_submission.py
```

The verifier uses the Python standard library and checks the frozen protocol,
code and split hashes, and published metric arithmetic. It runs without model
weights, audio, or a GPU.

The [full 12-slide deck](TURAS.pptx) and [speaker outline](SLIDE_OUTLINE.md) cover
the methods, ablations, timing and reproduction details for questions. The
[submission guide](../SUBMISSION.md) links the source reports and instructions
for reproducing the experiment with authorized artifacts.
