"""Update the native editable four-slide deck, preserving the original chart."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
ET.register_namespace("p", NS["p"])
ET.register_namespace("a", NS["a"])
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")

TALKS = [
    ("0:00-0:15", "TURAS studies Turkish parliamentary speech recognition. We compare Whisper LoRA and full fine-tuning, XLS-R CTC adaptation, and a train-only relational graph that reranks model-generated transcripts."),
    ("0:15-0:40", "The frozen split contains 4,376 training, 438 validation and 636 test clips, with 28 test recording groups. We select settings on validation. The references are edited parliamentary minutes, and 611 test references remain unreviewed. We apply the same Turkish-aware normalizer and corpus-level word error metric to every system."),
    ("0:40-1:25", "Acoustic adaptation gives the clearest improvement. Whisper-small word error falls from 24.51 percent to 12.37 with LoRA and 10.37 with full tuning. XLS-R improves from 37.15 to 25.15 percent, combining fine-tuning with a train-derived vocabulary extension. Zero-shot Whisper-large-v3 reaches 12.32 percent, level with LoRA, and Qwen3-ASR reaches 16.09. The additional BiLSTM-CTC baseline reaches 47.78 percent. The supplied single-Gaussian character HMM reaches approximately 100 percent, so it provides a weak classical reference."),
    ("1:25-2:00", "A two-layer heterogeneous GNN learns typed bilinear relation scores from train-only graph evidence. We hold N-best lists fixed and select rescoring weights on validation. Learned graph rescoring does not reliably improve any adapted model: all three recording-group confidence intervals include zero. We preserve that negative result. The study uses one training seed per model. Independent review and reference adjudication remain open. The submission includes hashed artifacts and reproduction commands."),
]


def replace(shape, text):
    body = shape.find("p:txBody", NS)
    old = body.findall("a:p", NS)
    template = deepcopy(old[0])
    for paragraph in old:
        body.remove(paragraph)
    for line in text.split("\n"):
        paragraph = deepcopy(template)
        run = paragraph.find("a:r", NS)
        if run is None:
            run = ET.SubElement(paragraph, "{" + NS["a"] + "}r")
            ET.SubElement(run, "{" + NS["a"] + "}t")
        for other in list(paragraph):
            if other.tag not in {"{" + NS["a"] + "}pPr", "{" + NS["a"] + "}endParaRPr"} and other is not run:
                paragraph.remove(other)
        run.find("a:t", NS).text = line
        body.append(paragraph)


def result_table(slide, rows):
    """Add native editable cells for the separate follow-up benchmark."""
    tag = lambda ns, name: "{" + NS[ns] + "}" + name
    frame = ET.SubElement(slide.find("p:cSld/p:spTree", NS), tag("p", "graphicFrame"))
    nv = ET.SubElement(frame, tag("p", "nvGraphicFramePr"))
    ET.SubElement(nv, tag("p", "cNvPr"), id="20", name="Follow-up results table")
    ET.SubElement(nv, tag("p", "cNvGraphicFramePr"))
    ET.SubElement(nv, tag("p", "nvPr"))
    transform = ET.SubElement(frame, tag("p", "xfrm"))
    ET.SubElement(transform, tag("a", "off"), x="6830000", y="2010000")
    ET.SubElement(transform, tag("a", "ext"), cx="4570000", cy="3310000")
    graphic = ET.SubElement(frame, tag("a", "graphic"))
    data = ET.SubElement(graphic, tag("a", "graphicData"), uri="http://schemas.openxmlformats.org/drawingml/2006/table")
    table = ET.SubElement(data, tag("a", "tbl"))
    ET.SubElement(table, tag("a", "tblPr"), firstRow="1", bandRow="1")
    grid = ET.SubElement(table, tag("a", "tblGrid"))
    for width in [3260000, 1310000]:
        ET.SubElement(grid, tag("a", "gridCol"), w=str(width))
    for index, values in enumerate([("Follow-up / baseline", "WER %"), *rows]):
        row = ET.SubElement(table, tag("a", "tr"), h="551666")
        for col, text in enumerate(values):
            cell = ET.SubElement(row, tag("a", "tc"))
            body = ET.SubElement(cell, tag("a", "txBody"))
            ET.SubElement(body, tag("a", "bodyPr"))
            ET.SubElement(body, tag("a", "lstStyle"))
            paragraph = ET.SubElement(body, tag("a", "p"))
            ET.SubElement(paragraph, tag("a", "pPr"), algn="r" if col else "l")
            run = ET.SubElement(paragraph, tag("a", "r"))
            properties = ET.SubElement(run, tag("a", "rPr"), lang="en-US", sz="1700", b="1" if index == 0 else "0")
            fill = ET.SubElement(properties, tag("a", "solidFill"))
            ET.SubElement(fill, tag("a", "srgbClr"), val="FFFFFF" if index == 0 else "183443")
            ET.SubElement(properties, tag("a", "latin"), typeface="Arial")
            ET.SubElement(run, tag("a", "t")).text = text
            properties = ET.SubElement(cell, tag("a", "tcPr"), marL="100000", marR="100000", marT="65000", marB="65000", anchor="ctr")
            fill = ET.SubElement(properties, tag("a", "solidFill"))
            ET.SubElement(fill, tag("a", "srgbClr"), val="087866" if index == 0 else ("EDF4F5" if index % 2 else "FFFFFF"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--completion", action="store_true")
    args = parser.parse_args()
    source = ROOT / "presentation/TURAS-2-minute.pptx"
    target = ROOT / ("presentation/TURAS-final-2-minute-v2.pptx" if args.completion else "presentation/TURAS-final-2-minute.pptx")
    summary = json.loads((ROOT / "artifacts/m3-final-2026-11-22/final-summary.json").read_text())
    metric = lambda name: 100 * summary["systems"][name]["test"]["metrics"]["tr-normalized-v1"]["wer"]
    edits = {
        1: {"subject": "Turkish ASR adaptation and graph rescoring", "research-question": "Whisper LoRA/full tuning, XLS-R CTC and train-only RDL"},
        2: {"heading": "Data and evaluation protocol", "split-title": "Frozen split and selection",
            "split-body": "4,376 train / 438 validation clips.\nModel selection uses validation only.",
            "reference-body": "Edited TBMM minutes.\n611/636 test references unreviewed."},
        3: {"caption": "CTC adaptation includes vocabulary extension (38 to 52 tokens).\n"
                       f"Other baselines: Whisper-large-v3 12.32%; Qwen3-ASR 16.09%; BiLSTM-CTC {metric('b5'):.2f}%; HMM {metric('gaussian-hmm'):.0f}%."},
        4: {"heading": "RDL rescoring and limitations", "key-finding": "All three 95% intervals include zero",
            "comparison-title": "Train-only relational model",
            "comparison-body": "2-layer heterogeneous GNN.\nTyped bilinear relation scores.\nValidation-selected N-best fusion.",
            "limits-title": "Study limits and remaining work",
            "limits-body": "One seed and edited minutes.\nIndependent review pending."},
    }
    talks = TALKS
    table_rows = []
    if args.completion:
        report = json.loads((ROOT / "results/final/completion_test_summary.json").read_text())
        if report["state"] != "completed":
            raise ValueError("Real completed results are required")
        selected = report["b3_selection"]["selected_system"]
        rate = lambda key: 100 * report["systems"][key]["test"]["metrics"]["tr-normalized-v1"]["wer"]
        label = "Qwen 1.7B" if selected == "qwen17" else "Qwen 0.6B"
        table_rows = [("Whisper-large-v3", f"{rate('b2'):.2f}"), (label, f"{rate(selected):.2f}"),
                      ("Seda RNN-T (CPU)", f"{rate('seda'):.2f}"), ("BiLSTM-CTC", f"{metric('b5'):.2f}"),
                      ("4-mixture HMM", f"{rate('gmm'):.2f}")]
        edits[1]["research-question"] = "Acoustic adaptation, broad baselines and train-only graph rescoring"
        edits[3].update(heading="Acoustic adaptation and baseline results",
                        caption="Left: original matched adaptation study. Right: separate follow-up on the same working labels.\n"
                                "CTC adaptation includes vocabulary extension. HMM uses grapheme states and no external LM.")
        edits[4].update({"limits-title": "Limits and reproducibility", "limits-body": "One seed and edited minutes.\nTraining overlap remains uncertain.\nPinned models and restore checks."})
        talks = [
            ("0:00-0:15", "TURAS studies Turkish parliamentary speech recognition. We compare pretrained systems, acoustic adaptation, and a train-only relational graph that reranks model-generated transcripts."),
            ("0:15-0:40", "The frozen split contains 4,376 training, 438 validation and 636 test clips. Model selection uses validation. References are edited parliamentary minutes, with 611 test clips still unreviewed. Every system uses the same Turkish-aware normalizer and corpus-level word error metric."),
            ("0:40-1:25", f"Whisper-small word error falls from 24.51 percent to 12.37 with LoRA and 10.37 with full tuning. XLS-R improves from 37.15 to 25.15 percent, including a train-derived vocabulary extension. In the separate follow-up, Whisper-large reaches {rate('b2'):.2f} percent and validation-selected {label} reaches {rate(selected):.2f}. The CPU transducer reaches {rate('seda'):.2f}, and the genuine four-mixture grapheme HMM reaches {rate('gmm'):.2f} percent."),
            ("1:25-2:00", "A two-layer heterogeneous GNN learns typed bilinear relation scores from train-only evidence. With fixed candidate lists and validation-selected fusion weights, graph rescoring gives no reliable gain. All three recording-group confidence intervals include zero. We preserve that negative result. We provide checked predictions, repeated timing, pinned artifacts and a fresh-environment restoration check. Single-seed training, edited references and unresolved pretrained-data overlap limit the conclusions."),
        ]
    with ZipFile(source) as src, ZipFile(target, "w", ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            for i in range(1, 5):
                if item.filename == f"ppt/slides/slide{i}.xml":
                    slide = ET.fromstring(data)
                    for shape in slide.findall(".//p:sp", NS):
                        name = shape.find("p:nvSpPr/p:cNvPr", NS).get("name")
                        if name in edits.get(i, {}):
                            replace(shape, edits[i][name])
                        if i == 3 and name == "caption":
                            shape.find("p:spPr/a:xfrm/a:ext", NS).set("cy", "510000")
                        if args.completion and i == 4 and name == "limits-body":
                            shape.find("p:spPr/a:xfrm/a:ext", NS).set("cy", "1200000")
                    if args.completion and i == 3:
                        chart = slide.find(".//p:graphicFrame", NS)
                        chart.find("p:xfrm/a:ext", NS).set("cx", "5940000")
                        result_table(slide, table_rows)
                    data = ET.tostring(slide, encoding="utf-8", xml_declaration=True)
                if item.filename == f"ppt/notesSlides/notesSlide{i}.xml":
                    note = ET.fromstring(data)
                    for shape in note.findall(".//p:sp", NS):
                        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
                        if ph is not None and ph.get("type") == "body":
                            sources = "\nSources\nresults/final/M4_M5_FINAL_REPORT.md\nresults/final/m3_test_summary.json\nconfigs/evaluation/m45_final_protocol.json\ndocs/final_review_2026-11-22.md"
                            if args.completion:
                                sources += "\nresults/final/completion_test_summary.json\ndocs/baseline_completion_followup.md"
                            replace(shape, talks[i-1][0] + "\n" + talks[i-1][1] + sources)
                    data = ET.tostring(note, encoding="utf-8", xml_declaration=True)
            dst.writestr(item, data)
    text = "TURAS two-minute technical presentation\n\n" + "\n\n".join(f"Slide {i}: {timing}\n{talk}" for i, (timing, talk) in enumerate(talks, 1)) + "\n"
    (ROOT / "presentation/TURAS-two-minute-speaker-script.txt").write_text(text, encoding="utf-8")
    print(target)
    print("Spoken words:", sum(len(t.split()) for _, t in talks))


if __name__ == "__main__":
    main()
