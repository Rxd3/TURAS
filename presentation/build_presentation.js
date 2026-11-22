// Build the TURAS two-minute deck: title + 4 timed slides (add --backup for 4 extra Q&A slides).
// Every number comes from files in the TURAS repository; see each slide's SOURCES note.
// Usage: npm install pptxgenjs@3.12.0
//        node presentation/build_presentation.js presentation/TURAS-final-2-minute.pptx
//        python presentation/add_error_bars.py presentation/TURAS-final-2-minute.pptx
// Optional: set PPTX_SKILL to a folder with scripts/apply_theme.js to write the TURAS
// theme colors; without it, scheme colors fall back to Office defaults.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
let applyTheme = async () => {};
if (process.env.PPTX_SKILL) ({ applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js"));

const THEME = {
  name: "TURAS",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "182B40", lt1: "FFFFFF", dk2: "5B6B7E", lt2: "EEF2F6",
    accent1: "0E7C6B", accent2: "7B8FA6", accent3: "D35F37", accent4: "2B59C3",
    accent5: "A9B4C2", accent6: "E3A72F", hlink: "0E7C6B", folHlink: "5B6B7E",
  },
};
const HEX = { navy: "182B40", muted: "5B6B7E", tint: "EEF2F6", teal: "0E7C6B", slate: "7B8FA6",
  orange: "D35F37", blue: "2B59C3", light: "A9B4C2", gold: "E3A72F", grid: "DDE3EA", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "TURAS two-minute presentation";
pres.author = "TURAS team";
const C = pres.SchemeColor;
const SRC = "https://github.com/Rxd3/TURAS/blob/main/";
// Backup slides (paired CIs, ablations, entity types, data/limits) only with --backup.
const BACKUP = process.argv.includes("--backup");

// One content layout: title placeholder, footer text and slide number.
pres.defineSlideMaster({
  title: "TURAS_CONTENT",
  background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontSize: 36, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "TURAS", options: { x: 0.6, y: 7.0, w: 2, h: 0.3, fontSize: 11, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 11, color: HEX.muted, align: "right" },
});

// Title layout: dark background, no footer or slide number.
pres.defineSlideMaster({ title: "TURAS_TITLE", background: { color: "13233A" }, objects: [] });

let chartCount = 0;
const errorBars = {}; // chart index -> [[minus], [plus]]
function chart(slide, type, data, opts, bars) {
  chartCount += 1;
  if (bars) errorBars[chartCount] = bars;
  slide.addChart(type, data, opts);
}
function label(slide, text, x, y, w, opts = {}) {
  slide.addText(text, { x, y, w, h: opts.h || 0.4, fontSize: opts.fontSize || 18, bold: opts.bold !== false,
    color: opts.color || C.text1, margin: 0, valign: "top", isTextBox: true, ...opts.extra });
}
function takeaway(slide, text, y = 6.2) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.45, fontSize: 19, bold: true, color: C.text1, margin: 0, isTextBox: true });
}
function footnote(slide, text, y = 6.65) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.32, fontSize: 11, color: C.text2, margin: 0, isTextBox: true });
}
function barOpts(extra) {
  return {
    barDir: "bar", catAxisOrientation: "minMax", barGapWidthPct: 45,
    catAxisLabelColor: HEX.navy, catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: HEX.muted, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: false, valAxisLineShow: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: HEX.navy, dataLabelFontSize: 12,
    dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.00", showLegend: false,
    ...extra,
  };
}
function legend(slide, items, x, y) {
  let cx = x;
  for (const [text, color] of items) {
    slide.addShape(pres.shapes.RECTANGLE, { x: cx, y: y + 0.07, w: 0.16, h: 0.16, fill: { color }, line: { color } });
    const w = 0.095 * text.length + 0.15;
    slide.addText(text, { x: cx + 0.22, y, w, h: 0.3, fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
    cx += w + 0.55;
  }
}

// ---------- Title slide (about 2 seconds) ----------
{
  const s = pres.addSlide({ masterName: "TURAS_TITLE" });
  // Speech waveform motif: rounded bars mirrored around a centre line.
  const heights = [0.35, 0.7, 1.2, 0.8, 1.8, 2.6, 1.6, 3.2, 2.2, 1.3, 2.8, 3.6, 2.4, 1.5, 2.9, 2.0, 1.1, 1.7, 0.9, 1.4, 0.6, 0.9, 0.4];
  heights.forEach((h, i) => {
    const x = 8.25 + i * 0.2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.55 - h / 2, w: 0.13, h, rectRadius: 0.065,
      fill: { color: i === 7 || i === 14 ? HEX.gold : HEX.teal, transparency: i === 7 || i === 14 ? 0 : 10 + (i % 3) * 18 },
      line: { type: "none" }, objectName: `wave-${i + 1}` });
  });
  s.addText("TURAS", { x: 0.7, y: 1.55, w: 6.6, h: 1.4, fontSize: 88, bold: true, color: C.background1, margin: 0, isTextBox: true });
  s.addText("Turkish Unified Recognition Adaptation System", { x: 0.72, y: 2.95, w: 7.3, h: 0.5, fontSize: 20,
    color: HEX.gold, bold: true, margin: 0, isTextBox: true });
  s.addText("Domain-specific Turkish speech recognition for parliamentary debates on the economy and legislation",
    { x: 0.72, y: 3.6, w: 7.0, h: 0.95, fontSize: 18, color: "C9D3DF", margin: 0, valign: "top", isTextBox: true });
  const tags = ["Turkish", "Economy + legislation", "Fine-tuning + graph rescoring"];
  let tx = 0.72;
  tags.forEach((t) => {
    const w = 0.09 * t.length + 0.45;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tx, y: 4.85, w, h: 0.44, rectRadius: 0.22,
      fill: { color: HEX.navy }, line: { color: HEX.teal, width: 1.25 }, objectName: `tag-${t}` });
    s.addText(t, { x: tx, y: 4.85, w, h: 0.44, fontSize: 13, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    tx += w + 0.2;
  });
  s.addText("Project 3", { x: 0.72, y: 6.6, w: 4, h: 0.35, fontSize: 13, color: "8FA0B5", margin: 0, isTextBox: true });
  s.addNotes(`TALK (2 seconds)
This is TURAS, our Turkish speech recognition project.

SOURCES
${SRC}README.md`);
}

// ---------- Slide 1: the domain-specific Turkish dataset ----------
{
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("A domain-specific Turkish dataset we built", { placeholder: "title" });
  s.addText("TURAS: speech recognition for Turkish parliamentary debates on the economy and legislation",
    { x: 0.6, y: 1.2, w: 12.1, h: 0.4, fontSize: 18, color: C.text2, margin: 0, isTextBox: true });

  label(s, "Clips left after each filter (WorldSpeech tr_tr)", 0.6, 1.8, 7.4);
  const steps = ["All Turkish clips", "Parliament (TBMM) only", "Spoken text only", "No duplicate text",
    "Text matches audio", "Economic or legal domain", "No upstream-test overlap", "Kept, capped per recording"];
  const left = [239654, 237203, 220435, 217260, 213374, 28807, 27738, 5450];
  const fill = [HEX.slate, HEX.slate, HEX.slate, HEX.slate, HEX.slate, HEX.navy, HEX.navy, HEX.teal];
  chart(s, pres.charts.BAR, [{ name: "Clips left", labels: [...steps].reverse(), values: [...left].reverse() }],
    barOpts({ x: 0.45, y: 2.15, w: 7.5, h: 3.75, chartColors: [...fill].reverse(), barGapWidthPct: 35,
      valAxisMinVal: 0, valAxisMaxVal: 290000, valAxisHidden: true, valGridLine: { style: "none" },
      dataLabelFormatCode: "#,##0", catAxisLabelFontSize: 13, objectName: "dataset-funnel" }));

  const cards = [
    ["90 terms", "Economic and legal word list. A clip needs 2 economic terms, or 1 plus a law term or a number."],
    ["2026 test", "Split by recording; every 2026 session held out. 4,376 / 438 / 636 clips, 20 / 2 / 3 h. Leakage checks: zero."],
    ["50 clips", "Team listening check of a sample. On its 25 test clips, fine-tuning still cuts WER 22.1% → 10.8%."],
  ];
  cards.forEach(([big, body], i) => {
    const y = 1.85 + i * 1.38;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.3, y, w: 4.5, h: 1.22, rectRadius: 0.08,
      fill: { color: HEX.tint }, line: { color: HEX.tint }, objectName: `dataset-card-${i + 1}` });
    s.addText([{ text: big, options: { fontSize: 24, bold: true, color: C.accent1, breakLine: true } },
      { text: body, options: { fontSize: 13, color: C.text1 } }],
      { x: 8.48, y: y + 0.06, w: 4.2, h: 1.1, valign: "middle", margin: 0, isTextBox: true });
  });
  takeaway(s, "We kept 2% of the source: 5,450 clips of Turkish economic and legislative speech.", 6.1);
  footnote(s, "Then: fine-tune Whisper-small and XLS-R CTC on train; rerank 5-best lists with a train-only graph. Source: WorldSpeech tr_tr (CC BY-NC 4.0).", 6.55);
  s.addNotes(`TALK (30 seconds)
TURAS is Turkish speech recognition for parliamentary debates on the economy and legislation, and we built the dataset ourselves. From about 240 thousand Turkish clips we kept 5,450. We removed speaker headers, duplicates and badly aligned text, and kept only clips matching our 90-term economic and legal word list. We split by recording, held out every 2026 session as the test set, and listened to a 50-clip sample to check the transcripts.

SOURCES
${SRC}docs/dataset.md
${SRC}configs/datasets/turas-domain-lexicon-v0.1.json
${SRC}docs/m1_deadline_handoff.md
${SRC}docs/error_analysis.md
${SRC}data/splits/turas-tr-econ-legis-v0.1/split_manifest.json`);
}

// ---------- Slide 2: benchmark ----------
{
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Test benchmark", { placeholder: "title" });
  s.addText("Normalized word error rate (%) on 636 unseen 2026 clips. Lower is better.",
    { x: 0.6, y: 1.2, w: 12.1, h: 0.4, fontSize: 18, color: C.text2, margin: 0, isTextBox: true });
  const names = ["Whisper-small, full fine-tune", "Whisper-large-v3, zero-shot", "Whisper-small, LoRA",
    "Qwen3-ASR-1.7B, zero-shot", "Whisper-small, pretrained", "XLS-R CTC, fine-tuned", "Seda RNN-T transducer", "XLS-R CTC, pretrained", "BiLSTM-CTC, trained here"];
  const vals = [10.371, 12.319, 12.368, 16.094, 24.512, 25.150, 29.958, 37.152, 47.776];
  const colors = [HEX.teal, HEX.orange, HEX.teal, HEX.orange, HEX.slate, HEX.teal, HEX.orange, HEX.slate, HEX.light];
  label(s, "All systems", 0.6, 1.8, 6);
  chart(s, pres.charts.BAR, [{ name: "Normalized WER", labels: [...names].reverse(), values: [...vals].reverse() }],
    barOpts({ x: 0.45, y: 2.15, w: 7.75, h: 3.55, chartColors: [...colors].reverse(), valAxisMinVal: 0, valAxisMaxVal: 55, valAxisMajorUnit: 10,
      valAxisLabelFormatCode: "0", objectName: "benchmark-chart" }));
  legend(s, [["Fine-tuned", HEX.teal], ["Pretrained parent", HEX.slate], ["Other pretrained", HEX.orange], ["From scratch", HEX.light]], 0.75, 5.75);

  label(s, "Decoding cost, same GPU session", 8.6, 1.8, 4.3);
  chart(s, pres.charts.BAR, [{ name: "End-to-end RTF", labels: ["Whisper-large-v3", "Qwen3-ASR-1.7B", "Whisper-small"],
    values: [0.1754, 0.1384, 0.0576] }],
    barOpts({ x: 8.45, y: 2.15, w: 4.4, h: 1.95, chartColors: [HEX.orange, HEX.orange, HEX.slate], valAxisMinVal: 0, valAxisMaxVal: 0.25,
      valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", dataLabelFormatCode: "0.000", catAxisLabelFontSize: 12,
      objectName: "rtf-chart" }));
  s.addText([{ text: "3.0×", options: { fontSize: 44, bold: true, color: C.accent1, breakLine: true } },
    { text: "slower: zero-shot large-v3 only ties LoRA-tuned Whisper-small (12.32 vs 12.37)", options: { fontSize: 14, color: C.text1 } }],
    { x: 8.6, y: 4.2, w: 4.2, h: 1.6, valign: "top", margin: 0, isTextBox: true });
  takeaway(s, "Full fine-tuning gives the best system: 10.37% WER, down from 24.51% before tuning.");
  footnote(s, "RTF = compute time / audio time (RTX 5070 Ti, 3 repeats). Large-v3, Qwen: M6 scorer, outside the frozen protocol. GMM-HMMs: ~100% WER (not shown).");
  s.addNotes(`TALK (30 seconds)
We fine-tune Whisper-small and XLS-R CTC on the training split. Full fine-tuning cuts Whisper's error from 24.5 to 10.4 percent, the best result; LoRA reaches 12.4, and XLS-R improves from 37.2 to 25.2. Zero-shot Whisper-large-v3 only ties our LoRA model at 12.3 while running three times slower; Qwen3-ASR reaches 16.1 and a Turkish transducer 30.

SOURCES
${SRC}results/final/m45_metrics.csv
${SRC}results/m6/README.md
${SRC}results/m6/timing/c2_timing_summary.json
${SRC}results/final/m3_test_summary.json`);
}

// ---------- Slide 3: where the gains come from ----------
{
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Where the gains come from", { placeholder: "title" });
  label(s, "Word errors by type (% of reference words)", 0.6, 1.4, 6.5);
  chart(s, pres.charts.BAR, [
    { name: "Substitutions", labels: ["Whisper pretrained", "Whisper LoRA", "Whisper full", "XLS-R pretrained", "XLS-R fine-tuned"], values: [16.72, 9.11, 7.61, 27.61, 16.80] },
    { name: "Deletions", labels: ["Whisper pretrained", "Whisper LoRA", "Whisper full", "XLS-R pretrained", "XLS-R fine-tuned"], values: [1.63, 1.51, 1.28, 2.64, 6.57] },
    { name: "Insertions", labels: ["Whisper pretrained", "Whisper LoRA", "Whisper full", "XLS-R pretrained", "XLS-R fine-tuned"], values: [6.16, 1.74, 1.48, 6.90, 1.78] },
  ], barOpts({ barDir: "col", catAxisOrientation: "minMax", x: 0.45, y: 1.8, w: 6.6, h: 3.75, barGapWidthPct: 60,
    chartColors: [HEX.navy, HEX.orange, HEX.gold], valAxisMaxVal: 30, valAxisMajorUnit: 10, valAxisLabelFormatCode: "0",
    dataLabelFormatCode: "0.0", dataLabelFontSize: 10, catAxisLabelFontSize: 12, objectName: "sdi-chart" }));
  legend(s, [["Substitutions", HEX.navy], ["Deletions", HEX.orange], ["Insertions", HEX.gold]], 0.8, 5.6);

  label(s, "Domain-term F1 (%), Whisper-small", 7.5, 1.4, 5.3);
  chart(s, pres.charts.BAR, [
    { name: "Full fine-tune", labels: ["Numbers", "Avro (euro)", "TÜİK", "Bağ-Kur", "SGK"], values: [94.68, 97.1, 96.0, 98.8, 95.3] },
    { name: "Pretrained", labels: ["Numbers", "Avro (euro)", "TÜİK", "Bağ-Kur", "SGK"], values: [32.13, 71.4, 57.1, 46.4, 35.8] },
  ], barOpts({ x: 7.35, y: 1.8, w: 5.5, h: 3.75, barGapWidthPct: 40, chartColors: [HEX.teal, HEX.slate], valAxisMinVal: 0,
    valAxisMaxVal: 100, valAxisMajorUnit: 20, valAxisLabelFormatCode: "0", dataLabelFormatCode: "0.0", dataLabelFontSize: 11,
    dataLabelPosition: "inEnd", dataLabelColor: HEX.white,
    objectName: "terms-chart" }));
  legend(s, [["Pretrained", HEX.slate], ["Full fine-tune", HEX.teal]], 7.65, 5.6);
  takeaway(s, "Fine-tuning removes 76% of inserted words and learns institution names and numbers.");
  footnote(s, "Paired recording-group bootstrap, 2,000 replicates, WER change in points: LoRA −12.14 [−13.46, −10.77]; full −14.14 [−15.42, −12.82]; XLS-R −12.00 [−13.13, −10.90].");
  s.addNotes(`TALK (25 seconds)
Where do the gains come from? Fine-tuning removes three quarters of Whisper's inserted words and halves substitutions. Domain terms improve sharply: the social security agency SGK goes from 36 to 95 percent F1, and spoken numbers from 32 to 95. All three gains have paired 95 percent intervals well below zero.

SOURCES
${SRC}docs/error_analysis.md
${SRC}results/final/M4_M5_FINAL_REPORT.md
${SRC}results/final/m45_test_summary.json`);
}

// ---------- Slide 4: relational rescoring ----------
{
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Relational rescoring: a clean null result", { placeholder: "title" });
  label(s, "WER change from graph rescoring (points, 95% CI)", 0.6, 1.4, 7);
  chart(s, pres.charts.BAR, [{ name: "WER change", labels: ["XLS-R tuned  +0.039", "Whisper full  +0.058", "Whisper LoRA  +0.005"],
    values: [0.039, 0.058, 0.005] }],
    barOpts({ x: 0.45, y: 1.85, w: 7.0, h: 3.2, chartColors: [HEX.navy], barGapWidthPct: 90,
      valAxisMinVal: -0.2, valAxisMaxVal: 0.2, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "+0.0;-0.0;0",
      showValue: false, catAxisLabelPos: "low", objectName: "rdl-chart" }),
    [[0.083, 0.092, 0.113], [0.091, 0.103, 0.107]]);
  s.addText("Zero = no change. Positive = more errors.", { x: 0.6, y: 5.1, w: 6.8, h: 0.3, fontSize: 12, color: C.text2, margin: 0, isTextBox: true });

  label(s, "What the graph does", 7.9, 1.4, 5);
  const pts = [["1", "Train-only graph of entities, aliases, topics and co-mentions"],
    ["2", "2-layer heterogeneous GNN learns typed relation scores"],
    ["3", "Each 5-best hypothesis is linked to the graph; fusion weights set on validation"]];
  pts.forEach(([n, t], i) => {
    const y = 1.9 + i * 0.85;
    s.addShape(pres.shapes.OVAL, { x: 7.9, y, w: 0.5, h: 0.5, fill: { color: HEX.navy }, line: { color: HEX.navy } });
    s.addText(n, { x: 7.9, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: 8.6, y: y - 0.05, w: 4.2, h: 0.65, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.9, y: 4.5, w: 4.9, h: 1.05, rectRadius: 0.08, fill: { color: HEX.tint }, line: { color: HEX.tint } });
  s.addText([{ text: "2.6 pts ", options: { fontSize: 26, bold: true, color: C.accent1 } },
    { text: "headroom exists: the best of five large-v3 candidates scores 9.72 vs 12.32 WER", options: { fontSize: 13, color: C.text1 } }],
    { x: 8.05, y: 4.55, w: 4.65, h: 0.95, valign: "middle", margin: 0, isTextBox: true });
  takeaway(s, "Graph rescoring moves WER by at most 0.06 points; every 95% interval includes zero.", 5.7);
  footnote(s, "Limits: one training seed; 611 of 636 test references are unreviewed edited minutes; speaker overlap unproven.", 6.25);
  s.addNotes(`TALK (25 seconds)
Finally, relational rescoring. A two-layer heterogeneous graph network rescores each five-best list. It changes error rates by at most six hundredths of a point, and every confidence interval includes zero, although better candidates exist. We report this null result, with our limits: one seed and mostly unreviewed references.

SOURCES
${SRC}results/final/M4_M5_FINAL_REPORT.md
${SRC}results/evaluation/m45_rdl_and_ablation_summary.json
${SRC}results/m6/provenance/test_oracle.json`);
}

// ---------- Backup 5: paired gains and CER ----------
if (BACKUP) {
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Backup: paired gains and character errors", { placeholder: "title" });
  label(s, "Paired WER change (points, 95% CI)", 0.6, 1.4, 6);
  chart(s, pres.charts.BAR, [{ name: "WER change", labels: ["Full vs LoRA  −2.00", "XLS-R tuned vs base  −12.00", "Full vs base  −14.14", "LoRA vs base  −12.14"],
    values: [-1.997, -12.003, -14.141, -12.144] }],
    barOpts({ x: 0.45, y: 1.85, w: 6.0, h: 3.9, chartColors: [HEX.teal], valAxisMinVal: -16, valAxisMaxVal: 0, valAxisMajorUnit: 4,
      valAxisLabelFormatCode: "0", showValue: false, catAxisLabelPos: "low",
      catAxisLabelFontSize: 12, objectName: "gains-chart" }),
    [[0.423, 1.126, 1.281, 1.319], [0.455, 1.103, 1.323, 1.370]]);
  label(s, "Test error rates (%)", 6.9, 1.4, 6);
  const head = { bold: true, color: C.background1, fill: { color: HEX.navy } };
  const rows = [["System", "Norm. WER", "Norm. CER", "Raw WER", "Term F1"],
    ["Whisper-small", "24.51", "10.08", "37.21", "92.18"], ["Whisper LoRA", "12.37", "4.15", "21.50", "96.84"],
    ["Whisper full", "10.37", "3.69", "18.88", "98.08"], ["XLS-R pretrained", "37.15", "13.17", "56.57", "83.36"],
    ["XLS-R fine-tuned", "25.15", "7.58", "45.18", "94.23"], ["Whisper-large-v3", "12.32", "5.45", "24.54", "97.76"],
    ["Qwen3-ASR-1.7B", "16.09", "5.60", "27.59", "91.23"], ["BiLSTM-CTC", "47.78", "13.50", "—", "—"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: i === 0 ? { ...head } :
      { color: C.text1, fill: { color: i % 2 ? HEX.tint : HEX.white }, align: j ? "right" : "left" } }))),
    { x: 6.9, y: 1.85, w: 5.9, colW: [2.1, 0.95, 0.95, 0.95, 0.95], fontSize: 13, rowH: 0.42, valign: "middle", margin: [0, 0.08, 0, 0.08],
      border: { type: "none" } });
  footnote(s, "Bootstrap over 28 test recording groups, 2,000 paired replicates, seed 2026. Large-v3 and Qwen rows: M6 scorer; term F1 there is micro F1 × 100.", 6.4);
  s.addNotes(`BACKUP DISCUSSION
All acoustic gains are paired over the same 28 recording groups; the full-versus-LoRA gain of about 2 points also excludes zero. Character error rate falls in the same direction as word error rate. BiLSTM-CTC is trained from scratch on 20 hours and has no terminology diagnostics.

SOURCES
${SRC}results/final/M4_M5_FINAL_REPORT.md
${SRC}results/final/m45_metrics.csv
${SRC}results/m6/test/summary.json
${SRC}results/final/m3_test_summary.json`);
}

// ---------- Backup 6: RDL ablations ----------
if (BACKUP) {
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Backup: graph ablations and controls", { placeholder: "title" });
  label(s, "WER change vs full RDL (points)", 0.6, 1.4, 7);
  const cats = ["Random graph", "No co-mention edges", "No quantity edges"];
  chart(s, pres.charts.BAR, [
    { name: "XLS-R fine-tuned", labels: cats, values: [0.005, 0.005, 0.019] },
    { name: "Whisper full", labels: cats, values: [-0.024, -0.019, -0.010] },
    { name: "Whisper LoRA", labels: cats, values: [-0.024, -0.049, -0.010] },
  ], barOpts({ x: 0.45, y: 1.85, w: 7.2, h: 3.7, chartColors: [HEX.slate, HEX.navy, HEX.teal], barGapWidthPct: 50, catAxisLabelPos: "low",
    valAxisMinVal: -0.06, valAxisMaxVal: 0.04, valAxisMajorUnit: 0.02, valAxisLabelFormatCode: "0.00",
    dataLabelFormatCode: "+0.000;-0.000;0.000", dataLabelFontSize: 11, objectName: "ablation-chart" }));
  legend(s, [["Whisper LoRA", HEX.teal], ["Whisper full", HEX.navy], ["XLS-R fine-tuned", HEX.slate]], 0.8, 5.6);
  label(s, "Reading the controls", 8.1, 1.4, 4.7);
  s.addText([
    { text: "Every ablation moves WER by less than 0.05 points.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "A random graph does as well as the learned one (slightly better for full Whisper).", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "Rescoring with acoustic scores alone is within 0.04 points of RDL.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "The learned graph adds no measurable gain over acoustic scores.", options: { bullet: { indent: 18 } } },
  ], { x: 8.1, y: 1.9, w: 4.7, h: 3.4, fontSize: 15, color: C.text1, paraSpaceAfter: 10, valign: "top", margin: 0, isTextBox: true });
  footnote(s, "Negative = fewer errors than full RDL. Paired 95% intervals are in the final report; most include zero.", 6.4);
  s.addNotes(`BACKUP DISCUSSION
Removing quantity or co-mention relations, or replacing the graph with a random one, changes WER by hundredths of a point. That is consistent with the learned graph not adding information beyond the acoustic scores. For full Whisper the random-graph interval lies just below zero, so the random graph is marginally better than the learned one.

SOURCES
${SRC}results/final/M4_M5_FINAL_REPORT.md
${SRC}results/evaluation/m45_rdl_and_ablation_summary.json`);
}

// ---------- Backup 7: entity types ----------
if (BACKUP) {
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Backup: terminology by entity type", { placeholder: "title" });
  label(s, "Term F1 (%) by entity type", 0.6, 1.4, 8);
  const cats = ["Concept (2,507)", "Currency (502)", "Legal text (161)", "Organization (215)"];
  chart(s, pres.charts.BAR, [
    { name: "Whisper pretrained", labels: cats, values: [93.3, 96.2, 92.1, 64.2] },
    { name: "Whisper full", labels: cats, values: [98.5, 98.3, 95.2, 95.3] },
    { name: "XLS-R pretrained", labels: cats, values: [86.5, 71.0, 83.0, 70.9] },
    { name: "XLS-R fine-tuned", labels: cats, values: [95.5, 93.0, 90.8, 84.2] },
  ], barOpts({ barDir: "col", catAxisOrientation: "minMax", x: 0.45, y: 1.85, w: 8.2, h: 3.7, barGapWidthPct: 50,
    chartColors: [HEX.slate, HEX.teal, HEX.light, HEX.navy], valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 20,
    valAxisLabelFormatCode: "0", dataLabelFormatCode: "0", dataLabelFontSize: 10, objectName: "entity-chart" }));
  legend(s, [["Whisper pretrained", HEX.slate], ["Whisper full", HEX.teal], ["XLS-R pretrained", HEX.light], ["XLS-R fine-tuned", HEX.navy]], 0.8, 5.6);
  label(s, "Spoken numbers (618 quantities)", 9.0, 1.4, 4);
  const rows = [["System", "Quantity F1"], ["Whisper pretrained", "32.13"], ["Whisper LoRA", "93.90"], ["Whisper full", "94.68"],
    ["XLS-R pretrained", "0.00"], ["XLS-R fine-tuned", "0.00"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: i === 0 ? { bold: true, color: C.background1, fill: { color: HEX.navy } } :
      { color: C.text1, fill: { color: i % 2 ? HEX.tint : HEX.white }, align: j ? "right" : "left" } }))),
    { x: 9.0, y: 1.85, w: 3.8, colW: [2.3, 1.5], fontSize: 13, rowH: 0.42, valign: "middle", margin: [0, 0.08, 0, 0.08], border: { type: "none" } });
  s.addText("XLS-R's 38-token head cannot emit digits; the 52-token extension can, but matched 0 of 618 exact quantities.",
    { x: 9.0, y: 4.55, w: 3.8, h: 1.0, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  footnote(s, "Entity-type support in parentheses. Exact (type, unit, value) matching for quantities.", 6.4);
  s.addNotes(`BACKUP DISCUSSION
Organization names are the weakest entity type for pretrained Whisper at 64 percent and rise to 95 after full fine-tuning. Quantities depend on digit output: Whisper learns the minutes' digit style, while XLS-R CTC cannot produce matching digit strings.

SOURCES
${SRC}docs/error_analysis.md`);
}

// ---------- Backup 8: data and protocol ----------
if (BACKUP) {
  const s = pres.addSlide({ masterName: "TURAS_CONTENT" });
  s.addText("Backup: data, protocol and limits", { placeholder: "title" });
  label(s, "Splits (grouped by recording)", 0.6, 1.4, 6);
  const rows = [["Split", "Clips", "Hours", "Recordings", "Sessions"],
    ["Train", "4,376", "20.0", "129", "2017 – Dec 2025"], ["Validation", "438", "2.0", "15", "Oct – Dec 2025"],
    ["Test", "636", "3.0", "28", "Jan – Apr 2026"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: i === 0 ? { bold: true, color: C.background1, fill: { color: HEX.navy } } :
      { color: C.text1, fill: { color: i % 2 ? HEX.tint : HEX.white }, align: j && j < 4 ? "right" : "left" } }))),
    { x: 0.6, y: 1.85, w: 6.3, colW: [1.3, 0.9, 0.85, 1.4, 1.85], fontSize: 14, rowH: 0.45, valign: "middle", margin: [0, 0.08, 0, 0.08], border: { type: "none" } });
  s.addText([
    { text: "Leakage checks all zero: shared recordings, dates, texts or IDs across splits.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "Model and fusion settings chosen on validation; protocol frozen before test access.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "56 pinned code hashes; every reported WER recomputed from stored edit counts.", options: { bullet: { indent: 18 } } },
  ], { x: 0.6, y: 3.9, w: 6.3, h: 1.9, fontSize: 14, color: C.text1, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
  label(s, "Limits we report", 7.4, 1.4, 5.4);
  s.addText([
    { text: "One training seed per adapted model.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "611 of 636 test references are unreviewed edited minutes; 25 have single-reviewer checks.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "Speaker disjointness is not proven.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "Large-v3 and Qwen are scored outside the frozen protocol; no paired test against the fine-tuned systems yet.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "Classical GMM-HMMs without a lexicon or LM recognize almost no words (~100% WER).", options: { bullet: { indent: 18 } } },
  ], { x: 7.4, y: 1.85, w: 5.4, h: 4.0, fontSize: 14, color: C.text1, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
  footnote(s, "Run python scripts/verify_submission.py from a fresh clone to check the frozen protocol, split hashes and metric arithmetic.", 6.4);
  s.addNotes(`BACKUP DISCUSSION
The test set is temporal: every 2026 recording and nothing else. Leakage checks are all zero. Reference quality is the main limitation: most test references are edited parliamentary minutes, which favour systems tuned on that style.

SOURCES
${SRC}docs/dataset.md
${SRC}docs/m1_deadline_handoff.md
${SRC}scripts/verify_submission.py`);
}

(async () => {
  const out = process.argv.slice(2).find((a) => !a.startsWith("--")) || "TURAS-final-2-minute.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  fs.writeFileSync(out + ".errorbars.json", JSON.stringify(errorBars));
  console.log("wrote", out, "charts:", chartCount);
})();
