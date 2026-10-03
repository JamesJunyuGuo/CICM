# Stage L CICM Datasheet

Base: PrefEval-style preference material from `external_data/icf_bench/dynamic_preference/`.
Surface: natural multi-turn preference updates plus filler turns; labels are controlled vocabulary values.
Scoring: program-verifiable normalized matching against controlled values; no LLM judge for headlines.

Seed: `20260722`
Rows: `70`
Dose counts: `{1: 14, 2: 14, 3: 14, 4: 14, 6: 14}`
Slots: `16`
Slot-value assignments: `112`
Unique controlled values: `7`
Current-value count min/max: `10` / `10`
Hardening: `l0b_mild`
Post-final filler turns: `4`
Post-final distractor updates: `2`
System style: `plain`
Query style: `natural`

The default profile uses repeated cross-slot preference values so the current-value probe is identifiable.
With `--n-per-dose 210`, each unique controlled value appears as current exactly 150 times.

## Slots

- `music_genre`: classic, modern, minimal, immersive, social, quiet, budget
- `diet`: classic, modern, minimal, immersive, social, quiet, budget
- `learning_style`: classic, modern, minimal, immersive, social, quiet, budget
- `transport`: classic, modern, minimal, immersive, social, quiet, budget
- `movie_style`: classic, modern, minimal, immersive, social, quiet, budget
- `exercise`: classic, modern, minimal, immersive, social, quiet, budget
- `workspace`: classic, modern, minimal, immersive, social, quiet, budget
- `cuisine`: classic, modern, minimal, immersive, social, quiet, budget
- `reading_format`: classic, modern, minimal, immersive, social, quiet, budget
- `meeting_time`: classic, modern, minimal, immersive, social, quiet, budget
- `vacation`: classic, modern, minimal, immersive, social, quiet, budget
- `pet_preference`: classic, modern, minimal, immersive, social, quiet, budget
- `drink`: classic, modern, minimal, immersive, social, quiet, budget
- `game_style`: classic, modern, minimal, immersive, social, quiet, budget
- `clothing`: classic, modern, minimal, immersive, social, quiet, budget
- `communication`: classic, modern, minimal, immersive, social, quiet, budget
