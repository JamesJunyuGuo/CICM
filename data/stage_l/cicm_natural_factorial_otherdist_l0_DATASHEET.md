# Stage L Natural CICM Datasheet

Purpose: upgrade Stage L data realism while preserving program-verifiable stale-binding labels.

## Construction

- Slot values are natural and slot-specific, not a shared generic value pool.
- A strong model may generate reusable dialogue templates, but the program injects exact controlled values.
- The program records target-slot `value_mentions` with role, message index, and character spans.
- Final queries are programmatically anchored to the target slot to remove query ambiguity.
- Distractor slot updates are retained after the final target binding as recent-other-slot competitors.
- Each row records `competition` metadata for target current, same-slot stale, and recent other-slot values.
- Scoring remains deterministic exact matching against the fixed controlled vocabulary.

Rows: `1200`
Seed: `20260723`
Dose counts: `{1: 240, 2: 240, 3: 240, 4: 240, 6: 240}`
Post-final filler turns: `6`
Post-final distractor updates: `3`
Factorial competition: `True`
Factorial target-current mode: `far`
Other-slot distance bins: `far,mid,near2`
Extra interference turns: `4`
Competition cell counts: `{('far', 'far'): 200, ('far', 'mid'): 200, ('far', 'near2'): 200, ('near', 'far'): 200, ('near', 'mid'): 200, ('near', 'near2'): 200}`
Target-current distance-bin counts: `{'far': 1200}`
Template source: `openai`
OpenAI model for template generation: `gpt-4.1`
Current binding count min/max over slot-value pairs: `57` / `58`

## Natural Value Sets

- `music_genre` (music genre): jazz, rock, classical, electronic, folk, hiphop, blues
- `diet` (meal style): vegan, vegetarian, keto, mediterranean, paleo, pescatarian, gluten free
- `learning_style` (learning style): visual, hands on, lecture, reading, discussion, self paced, tutoring
