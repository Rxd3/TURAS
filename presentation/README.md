# TURAS presentation

Use the [final four-slide PowerPoint](TURAS-final-2-minute.pptx) for the two-minute slot.
Its speaker notes contain a timed talk track.

| Time | Slide |
| --- | --- |
| 0:00–0:15 | Technical methods |
| 0:15–0:40 | Frozen data and evaluation |
| 0:40–1:25 | Word error rate gains and additional baselines |
| 1:25–2:00 | Heterogeneous GNN rescoring and study limits |

Before presenting, verify the public artifacts from the repository root:

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

The updated English deck includes B5 and the supplied single-Gaussian HMM.
Use [the roughly 200-word speaker script](TURAS-two-minute-speaker-script.txt) for the
strict two-minute slot. The older full deck retains its historical M4/M5 scope.
