# Task 3 offline CPU comparisons

Run after the scorer-v2 and token/cache preparation checks pass. No GPU, model weights, images or new generations are used. The original files and frozen scorer/preparation sources are not modified.

## Fixed selection

At N=1,2,4,8,16,32, reuse every saved candidate order and nested prefix. Compute FirstAns, self-consistency, highest mean logp, highest self-certainty, self-certainty RWC and shortest-rollout selection. MathVista also has vision-only, VisConf and tokenwise-fusion scores under both highest-score and RWC aggregation. Shortest and tokenwise ablations are reported at all budgets, including the required N=32 comparison.

MATH groups are recomputed from only the revealed extracted answers using the frozen equivalence predicate. Cached pairwise parsing/equivalence results do not add unseen answers to these groups. Candidate-only MathVista normalization is unchanged. NO_ANSWER remains in votes, candidate costs, and accuracy denominators.

VisConf calibrates the pooled **token KL values** of the current candidate prefix with linear-interpolation 5th/95th percentiles and denominator max(range,1e-8). Mean calibrated confidence is multiplied by mean normalized visual engagement. Tokenwise fusion instead averages confidence×engagement over aligned tokens; its difference from the product is checked against population token covariance.

Use descending average ranks for tied RWC scores. Integer vote units proportional to the prescribed weights avoid spurious float tie breaking. Score ties and equal answer votes return the earliest supporting candidate in the current saved order. At N=1 all methods must return the only candidate.

FirstAns consumes only its first candidate; all other fixed methods consume the full prefix. `oracle_coverage` and sampling/selection failures refer to the method's consumed candidates. `pool_oracle_coverage` separately records the budget-N opportunity curve (including for FirstAns), and `distinct_groups` describes that N-candidate pool. This keeps the FirstAns cost and the repeated-sampling opportunity distinct. The decomposition is sampling failure = no correct consumed candidate; selection failure = a correct consumed candidate existed but was missed.

## Stopping

Check n=4,8,16, otherwise stop at 32. Agreement requires a unique non-NO_ANSWER SC leader meeting tau. Quality additionally requires the SC and RWC winning answer groups to agree and the normalized RWC vote margin to meet gamma. RWC uses self-certainty for MATH and prefix-calibrated VisConf for MathVista. Both return current SC, not the RWC answer.

Search tau={0.5,0.625,0.75,0.875,1}, gamma={0,0.1,0.25,0.5}, with always-32 fallback. Only the saved 200 development questions per dataset enter threshold selection. Choose lowest mean token cost within one percentage point of development fixed-32 SC; ties higher accuracy, then smaller tau/gamma. Save `configs/adaptive_policies_frozen.json` before producing held-out summaries. Gold/correctness is not an argument to the selection or stopping functions. Tests poison gold labels and unseen KL values to check these boundaries.

## Diagnostics and intervals

Signals use within-question Spearman association with length and correctness. Top/bottom quartiles select eight candidates each, with ties in saved order 0. Confidence–vision Pearson correlations are computed within each rollout, then averaged over defined rollouts in a question, then over defined questions. Constant-vector correlations stay undefined/null and their counts are reported, rather than treating them as zero. These are retrospective associations, not evidence of causal grounding.

The 800 held-out questions per dataset are the summary units. Average each question's ten orders first. Use 1000 paired question-bootstrap resamples (seed 6304) for accuracy and main accuracy differences, sharing resample indices across methods/budgets. Fixed results include differences against same-budget SC and FirstAns. Adaptive results include differences against fixed-32 SC, actual token costs, rollout counts, stopping distributions and recoverable wrong early stops. Full-pool correct-fraction buckets are retrospective only. Preselected signal trajectories keep raw text and aligned token measurements; failure examples use the stated first-qualifying-question rule.

Mathematical parse failures and unparseable gold references are reported for review; they are not removed. The frozen scorer's exact-string fallback and timeouts are preserved. This supplied export has no zero-token rollouts; the runner fails rather than inventing quality scores if that changes.

## Commands and files

```bash
python -m unittest discover -s code -p test_task3_offline_cpu.py -v
python code/task3_offline.py --task-dir /content/drive/MyDrive/ATML/PA3/task3 --workers 2
```

Two CPU worker processes keep symbolic comparisons independent of Python thread/signal limitations. Memory is bounded to one input dataset shard/batch at a time. Complete 50-question result chunks are atomically saved with hashes, and reused after interruptions only when frozen identities match. A partial chunk may need recomputing. Finished reports are under `results/offline/`: per-question compressed results, per-order adaptive decisions, JSON summaries and fixed-method CSVs. Question chunks retain all candidate answers/correctness, selected rollout IDs, candidate IDs/costs and signal diagnostics. Candidate correctness and prefix IDs reproduce each oracle/failure indicator; these labels are stored separately from selector/stopping inputs. Full response/token arrays remain in prepared input caches, with selected raw trajectories copied into result records.

Wall times cover the current session and saved chunk computation separately; symbolic parser cache warmup and CPU type affect speed. These are CPU analysis timings, not deployment inference timings. The live 50-question FP16 experiment, final figures/clean notebook and repository organization still follow.
