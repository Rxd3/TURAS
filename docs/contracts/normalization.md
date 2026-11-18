# Turkish normalization contract — 0.2

Owner adoption for M4/M5 integration: 2026-11-19; see [D24](../owner_decisions.md).
This records project approval, not independent peer-review sign-off.

Two mandatory evaluation views are `raw-v1` and `tr-normalized-v1`. Original
references and hypotheses remain immutable. Apply the same versioned transform
to both sides only within the evaluator. Never normalize IDs or overwrite raw text.

In both views, whitespace means every character for which Python `str.isspace()`
is true. That set includes U+001C–U+001F, which the Unicode `White_Space`
property does not list.

## raw-v1

Raw WER splits on whitespace with no case/punctuation/number transform;
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
3. Delete every Unicode format character (general category Cf), for example
   U+200B zero width space and U+FEFF zero width no-break space.
4. Delete the intra-word apostrophes U+0027, U+2019 and U+02BC (modifier letter
   apostrophe) only when both adjacent code points in the text from step 3 are
   Unicode letters (general category L). Otherwise each of these three becomes a
   space. U+02BC is itself category Lm, so this step is what removes it. When
   testing an apostrophe's neighbours, these three apostrophe characters never
   count as letters, U+02BC included: `a''b → a b` and
   `a` + U+02BC + U+02BC + `b → a b`.
5. All other punctuation (Unicode category P), including other apostrophes,
   becomes a space, except `%` (U+0025), which is kept like a symbol because it
   carries meaning in quantities such as `%5`. This yields a declared spelling
   view, not a claim about linguistic equivalence.
6. Collapse every run of whitespace to one ASCII space and strip ends.
   Preserve digits, decimal grouping content after punctuation removal, currency
   symbols (category S), accents, suffixes and nonpunctuation characters.
7. Unicode NFC again. Deleting a format character (step 3) or an apostrophe
   (step 4) can leave two code points next to each other that NFC composes,
   for example `e` + U+200B + U+0301 `→ é` (U+00E9).

`tr-normalized-v1` is idempotent: applied to its own output, it returns that
output unchanged, for every input. Step 7 is what makes it so; without it, the
example above would give `e` + U+0301 on the first pass and `é` on the second.

Normalized WER splits on the resulting ASCII space. Normalized CER uses Unicode
code points including resulting single internal spaces. Do not expand digits to
words, translate acronyms, remove diacritics, stem, lemmatize, or guess spoken
currency forms. If an additional spoken-number metric is adopted, name and
version it separately and report it alongside the mandatory views.

Known token-count effects, kept on purpose: digits are not letters, so an
apostrophe after a digit becomes a space (`2026'da → 2026 da`, two tokens), and
grouping and decimal marks are punctuation (`1.250,50 → 1 250 50`, three
tokens). Normalized WER therefore counts such a written number as several tokens.

The policy examples are string transformation rules, **not dataset transcripts**:
`I → ı`, `İ → i`, `ı → ı`, `i → i`; `IŞIK → ışık`, `İZMİR → izmir`;
`I` + U+0307 `→ i`, while `i` + U+0307 stays `i` + U+0307;
`Ankara'da`, `Ankara’da` and `Ankaraʼda` (U+0027, U+2019, U+02BC) `→ ankarada`;
`%5 → %5`, `₺100 → ₺100`, `a+b-c/d → a+b c d`; `a` + U+200B + `b → ab`;
`e` + U+200B + U+0301 `→ é` (U+00E9) in one pass;
`a` + U+00A0 + space + U+2028 + `b → a b`; empty or whitespace-only text `→` empty.
Keep dotted/dotless families distinct. Planned implementation tests should cover
decomposed dotted I, apostrophes, format characters, `%`, repeated whitespace,
numerals, empty text and idempotence. Freeze Unicode/runtime and normalizer code
hash in experiment manifests; any policy correction requires a new version and
full reevaluation.

The label stays `tr-normalized-v1` under contract 0.2, because no evaluation has
used it yet. It freezes at the first real evaluation; after that, any policy
change creates `tr-normalized-v2` and a full reevaluation.

## Alignment tie-break

WER and CER edit counts come from a full Levenshtein alignment with unit costs,
from reference to hypothesis. The backtrace starts at the end of both sequences
and walks back to the start. At each cell it takes the first of these moves that
reaches the cell's optimal cost: the diagonal (a hit when the elements are equal,
otherwise a substitution), then deletion, then insertion. Golden test: reference
`a b` against hypothesis `b a` gives S=2, D=0, I=0.

## Version history

- 0.1 (2026-11-17): initial contract. Steps were NFC, Turkish lowercase,
  apostrophes U+0027/U+2019 only, other punctuation (including `%`) to a space,
  and whitespace collapse; whitespace meant Unicode whitespace.
- 0.2 (proposed by M6 on 2026-11-17, pending review by M1 and model owners
  M2–M5; [D17](../decisions.md)): adds Cf deletion (step 3) and U+02BC (step 4),
  keeps `%`, defines whitespace as `str.isspace()`, documents the digit
  apostrophe and number grouping effects, and states the alignment tie-break with
  its backtrace direction. The label `tr-normalized-v1` is kept until first freeze.
  Amended by M6 on 2026-11-17: adds a final NFC (step 7) and states that
  `tr-normalized-v1` is idempotent for every input.
  Amended by M6 on 2026-11-18, still 0.2 (proposed by M6): in step 4, the
  apostrophe characters U+0027, U+2019 and U+02BC never count as letters when
  testing an apostrophe's neighbours (`a''b → a b`; `a` + U+02BC + U+02BC +
  `b → a b`). The label `tr-normalized-v1` is still unfrozen.
