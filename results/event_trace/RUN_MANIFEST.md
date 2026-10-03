# Stage P run manifest

## Task-integrity hardening

All smoke runs used one complete five-checkpoint item and both corrupt and
same-length no-overwrite branches. These runs tested pipeline integrity only;
no smoke estimate is used in a headline.

| Revision | Job | Result | Integrity action |
|---|---:|---|---|
| Natural dialogue with numeric other-slot assignments | 20075095 | Capture/no-op passed; later controls returned other-slot values | Removed numeric other-slot candidates; retained other-slot identity markers and context length |
| Natural dialogue with stronger demonstrations | 20075115 | Update semantics improved, but later controls still returned other-slot values | Replaced unrelated assignments with no-update filler; stale traces and current distance unchanged |
| Natural dialogue without numeric distractors | 20075135 | Capture/no-op passed; 160M still failed the no-overwrite behavior gate | Returned to the frozen Stage-N symbolic assignment syntax, as permitted by the spec |
| Stage-N symbolic assignments plus fixed natural diagnostic question | 20075160 | **PASS**: all five no-overwrite branches returned current; corrupt branches returned within-slot stale from tau2 onward; tokenizer and capture no-op exact zero | Froze task syntax for the full run |

The final task uses eight balanced recurring current values, three seeds, three
surface templates, and 16 semantic items per seed (144 items, 1,440 branches).
The stale-trace doses are fixed at 0/1/1/2/4 for tau1/tau2/tau3/tau4/tauQ.
The historical-trace count was not tuned to obtain a target stale rate.

## Formal runs

| Model/stage | Job | Portal | Requested time | Evidence basis | Status |
|---|---:|---|---:|---|---|
| Pythia-160M first full attempt | 20075201 | GPU/A100 | 00:25 | Passing smoke 00:04:19 | **hard stop**: 52/720 control pairs changed token length because `drink`/`sport` split differently at line start |
| Pythia-160M slot-repaired attempt | 20075257 | GPU/A100 | 00:25 | Exact slot signatures passed, but cross-model precheck found numeric values are multi-token outside Pythia | canceled before completion; no result used |
| Pythia-160M common-word diagnostic | 20075313 | GPU/A100 | 00:25 | Cross-model word values passed tokenizer gates | **identity hard stop**: final 144/144 stale and tauQ no-overwrite accuracy 36/144; no matched analysis |
| Pythia-160M common-word analysis | 20075325 | shared CPU | 00:35 | afterok 20075313 | stopped in 14 s because the final correct-vs-stale pool was absent |
| Pythia-160M metadata repair attempt | 20075348 / 20075349 | GPU/A100 + AI/H100 | 00:25 | Frozen numeric values | canceled at startup: no-overwrite foil positions had not yet been stored as matched comparators for R/QK |
| Pythia-160M formal matched capture (distance-definition repair) | 20075365 / 20075366 | GPU/A100 + AI/H100 | 00:25 | Exact same-item overwrite/no-overwrite pairs; foil-old is a metric comparator but never labeled semantic stale | canceled while pending before execution: competitor distance used the farthest rather than nearest old write |
| Pythia-160M formal matched capture | 20075394 / 20075395 | GPU/A100 + AI/H100 | 00:25 | Exact same-item overwrite/no-overwrite pairs; nearest comparator distance is the residualization covariate and farthest distance is retained for audit | canceled while pending; walltime tightened using the completed 00:03:49 full capture |
| Pythia-160M formal matched capture | 20075419 / 20075420 | GPU/A100 + AI/H100 | 00:08 | Completed comparable capture took 00:03:49; request includes >2x measured runtime while remaining backfill-friendly | **PASS**, A100 completed 00:03:28; pending H100 mirror canceled before execution |
| Pythia-160M grouped analysis | 20075428 | shared CPU | 00:12 | 288 matched arms, 5 grouped probes, 1000 shuffles, no model inference | **PASS**, completed 00:02:12; earliest tau2, early-to-late BA 0.979 above shuffle; 121 exact patch pairs |
| Pythia-160M causal patch race | 20075435 / 20075436 | GPU/A100 + AI/H100 | 00:10 | 69 held-out pairs, four paths, about 1,100 small-model forwards; comparable capture completed in 00:03:28 | canceled while pending; tightened to a backfill-friendly six minutes |
| Pythia-160M causal patch race | 20075530 / 20075531 | GPU/A100 + AI/H100 | 00:06 | About 1,100 forwards versus 1,440 in the 00:03:28 capture; 72% runtime buffer | canceled while pending after CPU full completed |
| Pythia-160M CPU patch timing smoke | 20075477 | shared CPU | 00:08 | Four held-out pairs only; isolated output, timing only | completed 00:04:26; identity exact-zero passed, full-runtime estimate about 76 min; no smoke effect is used |
| Pythia-160M full CPU fallback race | 20075565 | shared CPU | 01:35 | Four-pair smoke implied about 76 min for 69 pairs; request added 25% buffer | **PASS**, completed 00:07:56; identity exact-zero; early key/query correction 0.928/0.768, final key/query 0.087/0.058 |
| Cross-architecture one-item capture/no-op race | 20075498 / 20075499 | AI/H100 + GPU/A100 | 00:15 | Qwen-1.5B, Gemma-2-2B, and Llama-8B sequential; formal broad data limited to one item; isolated roots | canceled while pending; split into shorter per-model gates |
| Pythia-1.4B formal descriptive capture race | 20075526 / 20075527 | GPU/A100 + AI/H100 | 00:30 | Same GPT-NeoX hook as passed 160M; full tokenizer gate passed; scale-based runtime buffer | **PASS**, H100 completed 00:04:51; pending A100 canceled; final stale pool 1/144 is mechanism-ineligible |
| Qwen2.5-1.5B one-item capture/no-op race | 20075581 / 20075582 | GPU/A100 + AI/H100 | 00:06 | Tokenizer full gate passed; one item tests GQA QK hook and exact no-op only | canceled while pending after CPU smoke passed |
| Qwen2.5-1.5B one-item CPU capture/no-op | 20075583 | shared CPU | 00:15 | Tokenizer full gate passed; one item tests GQA QK hook and exact no-op only | **PASS**, completed 00:04:31; 12 query / 2 KV heads, no-op exact-zero; sliding-window config is disabled |
| Qwen2.5-1.5B formal capture race | 20075626 / 20075627 | GPU/A100 + AI/H100 | 00:30 | 144 items after tokenizer/GQA/no-op gates; scale-based estimate from 160M capture with buffer | canceled while pending after 1.4B H100 completed in 00:04:51 |
| Gemma-2-2B one-item CPU capture/no-op | 20075628 | shared CPU | 00:20 | Qwen CPU smoke scaled by model size; tokenizer full gate already passed | **PASS**, completed 00:08:39; 8 query / 4 KV heads, no-op exact-zero |
| Llama-3.1-8B one-item CPU capture/no-op | 20075629 | shared CPU | 00:35 | Qwen CPU smoke scaled by model size; tokenizer full gate already passed | **PASS**, completed 00:12:35; 32 query / 8 KV heads, no-op exact-zero |
| Gemma-2-2B formal descriptive capture race | 20075637 / 20075638 | GPU/A100 + AI/H100 | 00:40 | 144 items after tokenizer/GQA/no-op gates; model-scale runtime estimate with buffer | canceled while pending; tightened after 1.4B H100 timing |
| Llama-3.1-8B 72-item confirmation capture race | 20075693 / 20075694 | GPU/A100 + AI/H100 | 00:40 | Fixed first 72 items after tokenizer/GQA/no-op gates; smaller 7-8B confirmation per spec | canceled while pending; tightened after 1.4B H100 timing |
| Qwen2.5-7B one-item CPU capture/no-op | 20075695 | shared CPU | 00:30 | Full tokenizer gate passed; model-specific Qwen GQA/no-op gate before 72-item confirmation | **PASS**, completed 00:12:06; 28 query / 4 KV heads, no-op exact-zero; sliding-window config disabled |
| Qwen2.5-7B 72-item confirmation capture race | 20075722 / 20075724 | GPU/A100 + AI/H100 | 00:40 | Fixed first 72 items after tokenizer/GQA/no-op gates; smaller 7-8B confirmation per spec | canceled while pending; tightened after 1.4B H100 timing |
| Qwen2.5-1.5B formal capture race | 20075824 / 20075825 | GPU/A100 + AI/H100 | 00:12 | 1.4B same-protocol H100 completed 00:04:51; >2x measured-time allowance | **PASS**, H100 completed 00:04:56; A100 canceled; final stale pool 8/144 |
| Gemma-2-2B formal descriptive capture race | 20075826 / 20075827 | GPU/A100 + AI/H100 | 00:15 | 1.4B same-protocol H100 completed 00:04:51; extra architecture/model-size allowance | capture **PASS** in 00:05:46; A100 canceled; no-overwrite behavior control fails badly and gates mechanism |
| Llama-3.1-8B 72-item confirmation capture race | 20075828 / 20075829 | GPU/A100 + AI/H100 | 00:20 | Half item count; parameter-scaled estimate plus model-load buffer | **PASS**, H100 completed 00:05:49; A100 canceled; final stale pool 2/72 |
| Qwen2.5-7B 72-item confirmation capture race | 20075830 / 20075831 | GPU/A100 + AI/H100 | 00:20 | Half item count; parameter-scaled estimate plus model-load buffer | **PASS**, H100 completed 00:05:41; A100 canceled; final stale pool 1/72 |
| Five-model descriptive/grouped analysis | 20076091--20076095 | shared CPU | 00:12 each | Uniform 20-pair mechanism gate; full-pool paired trajectories, 1000 shuffle/sign-flip nulls, grouped retention | **PASS**, all completed 00:01:43--00:03:43; every model mechanism-ineligible on the fixed final-failure pool |
| Pythia-160M final uniform analysis | 20076125 | shared CPU | 00:12 | Adds the same full-pool descriptive fields used by the other five models; matched analysis is deterministically rerun unchanged | **PASS**, completed 00:02:19; original matched-pool readings unchanged and uniform descriptive fields added |

The duplicate H100 capture job 20075202 was canceled while pending after the
A100 copy started. Its output path was isolated and it produced no result.
The blocked analysis dependency 20075210 was canceled after 20075201 failed;
no partial capture was analyzed.

The final cross-model protocol uses tokenizer-adapted value lexicons. Pythia
uses the frozen numeric lexicon that passes its identity control; Qwen, Llama,
and Gemma use a shared eight-word lexicon. Event order, checkpoint definitions,
slot identities, dose counts, seeds, templates, and item count are identical.
Lexical values are therefore not treated as a model-scale variable.
