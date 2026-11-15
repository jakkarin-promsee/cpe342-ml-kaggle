# Blind-spot hypotheses

These notes revisit whether the labels, feature timing, validation population, and final prediction rules match the competition tasks. Start with the [overall analysis](task-overall-p-note.md) for shared questions, evidence limits, and an experiment priority order. The task notes combine instructor-slide background with current project findings and proposed checks; a plausible mechanism is not a measured improvement.

| Task | Question to investigate |
| --- | --- |
| [Task 1: Anti-Cheat Pre-Filter](task-1-p-note.md) | Do missing labels and report-driven signals affect validation, and does mechanical behavior separate classes at similar performance levels? |
| [Task 2: Player Segment Classification](task-2-p-note.md) | When were tier features computed, how were segments defined, and which ambiguous profiles drive errors? |
| [Task 3: Monthly Spending Prediction](task-3-p-note.md) | Can a zero-versus-positive decision and conditional amount rule improve whole-population MAE? |
| [Task 4: Game Title Detection](task-4-p-note.md) | Could related screenshots cross folds despite exact-hash grouping? |
| [Task 5: Account Security Monitoring](task-5-p-note.md) | What behavior does the selected Tail detector count, and how much can be inferred without anomaly labels? |

Read the [task research notes](../README.md#experiment-records) for the current experiment record and the [competition materials](../README.md#competition-materials) for the official task descriptions and rules. Keep local validation, public overall scores, and unlabeled proxies distinct when evaluating any proposed experiment. Task 5 has no labeled offline F₃ result.
