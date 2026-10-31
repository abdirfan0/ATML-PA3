# Task 3 answer scorer

CPU preparation only. No model weights, inference, images or GPU are required.

## Fixed rules

- MATH uses supplied extracted answers for stored rollouts. Only live responses use the last literal `\boxed{` balanced-brace extractor. An absent, empty or unclosed last box returns no answer, even when an earlier box was valid. There is no numeric fallback.
- Math-Verify 0.9.0, latex2sympy2_extended 1.11.0, ANTLR 4.13.2 and SymPy 1.14.0 are pinned. Extracted strings are parsed as mathematical expressions using LatexExtractionConfig with no text fallback. The equivalence predicate accepts exact nonempty stripped strings or mutually verified strict mathematical equivalence. Parsing and each verification use 5-second timeouts; Math-Verify defaults of float_rounding=6 and numeric_precision=15 are retained. Unparseable nonempty answers match only identical strings; timeouts/parse failures should be counted during the full preparation audit.
- Form MATH groups from **only currently revealed candidates**, without gold answers. Sort unique prediction strings and use the first group for which every member is equivalent. This deterministic all-member rule avoids joining incompatible answers through a numerical-tolerance bridge. Group IDs do not resolve voting ties: the downstream selector must use the earliest supporting candidate in the current order. Recompute groups for each revealed prefix; do not reuse a partition built using unseen answers.
- MathVista follows the released `normalize_extracted_answer` rules at commit `c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb`: parenthesized option letters, exact option letters, otherwise closest choice by Levenshtein distance with first-choice ties; integer conversion via int(float(x)); float rounding to metadata precision; list string comparison. PA3 requires representing multiple-choice choices by option letters. Reference fields use exact choice-text matching first, then decode option letters when needed. This avoids interpreting question 168's literal choice text `(b)` (option E) as an option-B prediction. Model predictions retain the released normalization rules.
- Missing/null/empty predictions are one `NO_ANSWER` group, never correct. They are handled before closest-choice matching, as required by the assignment; invalid numeric normalization also yields NO_ANSWER. Keep all of these predictions and capped outputs in votes, denominators and cost accounting. Do not invoke an LLM or re-extract answers from the full stored responses.
- Candidate normalization never reads metadata's answer field. Reference fields are used separately for grading and preflight consistency checks, not for voting, quality scoring or stopping.

## Provenance

MathVista metadata: https://github.com/lupantech/MathVista/blob/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb/data/testmini.json

Adapted normalization: https://github.com/lupantech/MathVista/blob/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb/evaluation/calculate_score.py

Math-Verify: https://github.com/huggingface/Math-Verify

The original MathVista script is retained for attribution/audit only, not executed. Its dependencies and API calls are not needed. Metadata and the supplied README remain assets outside the eventual git source/results export; preserve their hashes and download locations there instead of committing raw datasets.

The preflight checks the pinned packages, asset/source hashes, all 1000 MathVista IDs and 32000 supplied gold fields against released metadata, and saved 200/800/50 split structure. It saves `configs/scorer_frozen.json` before any held-out metric or adaptive-policy selection is computed. This validation does not calculate performance. Unit tests cover equivalence, grouping, missing values, extraction and answer-type normalization. Token alignment, all ten candidate orders and full analysis will be checked in the next preparation stage.

## Version 2 fix

The original preflight incorrectly passed reference choice text through the prediction parser. Version 2 separates reference decoding from prediction normalization, adds a regression test for released question 168, and keeps the full metadata-consistency assertion with more detailed error reporting. No predictions, gold answers, selected IDs, or scoring thresholds are changed. No performance evaluation had been run.
