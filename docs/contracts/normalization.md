# Turkish normalization contract — 0.1

Two mandatory evaluation views are `raw-v1` and `tr-normalized-v1`. Original
references and hypotheses remain immutable. Apply the same versioned transform
to both sides only within the evaluator. Never normalize IDs or overwrite raw text.

## raw-v1

Raw WER splits on Unicode whitespace with no case/punctuation/number transform;
raw CER uses Unicode code points including internal spaces, case and punctuation
as emitted. Reference annotation policy must prohibit accidental leading/trailing
formatting whitespace; do not silently correct model output. Report the CER
space policy alongside the score. Empty references are handled by evaluation
policy, not patched by inserting text.

## tr-normalized-v1

Apply these operations in order:

1. Unicode NFC, retaining Turkish letters/diacritics.
2. Turkish-aware lowercase: map `I` to `ı` and `İ` to `i` before ordinary Unicode
   lowercasing. NFC first also composes `I` followed by combining dot above.
   Default English `lower()`/`casefold()` alone is not the policy.
3. Delete intra-word apostrophes U+0027 and U+2019 only when both adjacent code
   points are Unicode letters (general category L). All other punctuation
   (Unicode category P), including other apostrophes, becomes a space. This
   yields a declared spelling view, not a claim about linguistic equivalence.
4. Collapse every run of Unicode whitespace to one ASCII space and strip ends.
   Preserve digits, decimal grouping content after punctuation removal, currency
   symbols (category S), accents, suffixes and nonpunctuation characters.

Normalized WER splits on the resulting ASCII space. Normalized CER uses Unicode
code points including resulting single internal spaces. Do not expand digits to
words, translate acronyms, remove diacritics, stem, lemmatize, or guess spoken
currency forms. If an additional spoken-number metric is adopted, name and
version it separately and report it alongside the mandatory views.

The policy examples are string transformation rules, **not dataset transcripts**:
`I → ı`, `İ → i`, `ı → ı`, `i → i`; `IŞIK → ışık`, `İZMİR → izmir`.
Keep dotted/dotless families distinct. Planned implementation tests should cover
decomposed dotted I, apostrophes, repeated whitespace, numerals, empty text and
idempotence. Freeze Unicode/runtime and normalizer code hash in experiment
manifests; any policy correction requires a new version and full reevaluation.
