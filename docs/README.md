# Documentation guide

Start with the competition materials, then read the research note for the task you are working on. The slide hints are ideas to test against the provided data, not confirmed improvements or evidence of Kaggle's hidden implementation.

## Competition materials

| Document | What it covers |
| --- | --- |
| [Competition overview](kaggle-detial/overview.md) | The five tasks, their objectives, and evaluation metrics. |
| [Dataset description](kaggle-detial/data.md) | File layout, train and test data, image folders, data cards, and the sample submission. |
| [Competition rules](kaggle-detial/rule.md) | Allowed data and models, team rules, and evaluation constraints. |
| [Extracted instruction-slide hints](kaggle-detial/hint_canva_extracted.txt) | Instructor examples and suggestions to check before designing a new experiment. |

## Experiment records

Each task note summarizes its data checks, engineered features, model comparisons, current conclusions, and useful next experiments. Scores from local validation, proxies, and the public leaderboard have different meanings; read the notes before comparing them.

| Task | Research note |
| --- | --- |
| Task 1: Anti-Cheat Pre-Filter | [Task 1 research note](research-note-task1.md) |
| Task 2: Player Segment Classification | [Task 2 research note](research-note-task2.md) |
| Task 3: Monthly Spending Prediction | [Task 3 research note](research-note-task3.md) |
| Task 4: Game Title Detection | [Task 4 research note](research-note-task4.md) |
| Task 5: Account Security Monitoring | [Task 5 research note](research-note-task5.md) |

## Blind-spot hypotheses

The [blind-spot hypothesis guide](blind-spot-hypothesis/README.md) links the overall analysis and task-specific notes on whether the prediction objective, available signals, and validation population match the competition task. Treat the proposed checks and experiments as hypotheses rather than established improvements.

| Task | Blind-spot note |
| --- | --- |
| Task 1: Anti-Cheat Pre-Filter | [Task 1 hypotheses](blind-spot-hypothesis/task-1-p-note.md) |
| Task 2: Player Segment Classification | [Task 2 hypotheses](blind-spot-hypothesis/task-2-p-note.md) |
| Task 3: Monthly Spending Prediction | [Task 3 hypotheses](blind-spot-hypothesis/task-3-p-note.md) |
| Task 4: Game Title Detection | [Task 4 hypotheses](blind-spot-hypothesis/task-4-p-note.md) |
| Task 5: Account Security Monitoring | [Task 5 hypotheses](blind-spot-hypothesis/task-5-p-note.md) |

The [Kaggle leaderboard experiment log](kaggle-score-log.md) records user-reported overall scores, submission changes, and controlled comparisons. It does not provide per-task scores or hidden labels. For runnable pipeline examples and the artifact contract, see the [pipeline README](../src/pipeline/README.md).

The [selected-model leakage audit](leakage-audit-top-models.md) records the verified feature and split checks and the raw-feature timing uncertainties. The [submission plan](../submit-plan.md) records the completed Task 5 Tail/LOF and Tail-based behavior comparisons.
