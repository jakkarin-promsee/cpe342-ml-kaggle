# Dataset Description

Part of the **Player Intelligence System — CPE342 ML** competition for CPE342 Machine Learning (1/2026), Computer Engineering, KMUTT.

## Directory structure

```text
CPE342-Hackathon/
├── task1/
│   ├── train.csv          # Labeled training data
│   ├── test.csv           # Test data for predictions
│   └── datacard.txt       # Feature descriptions
├── task2/
│   ├── train.csv
│   ├── test.csv
│   └── datacard.txt
├── task3/
│   ├── train.csv
│   ├── test.csv
│   └── datacard.txt
├── task4/
│   ├── train/             # Training images
│   ├── test/              # Test images
│   ├── train.csv          # Image filenames, labels, and metadata
│   ├── test.csv           # Image filenames and metadata
│   └── datacard.txt
├── task5/
│   ├── test.csv           # Unsupervised test data
│   └── datacard.txt
└── sample_submission.csv  # Submission format template
```

## Data files

### Training data (`train.csv`)

- Tasks 1–4 provide labeled examples for model training, with task-specific features and targets.
- For Task 4, `train.csv` contains image filenames and labels; the images are in `task4/train/`.
- Task 5 has no `train.csv` because it is an unsupervised anomaly detection task.

### Test data (`test.csv`)

- All five tasks provide unlabeled examples for final predictions.
- Test files follow the corresponding task's feature structure.
- For Task 4, `test.csv` contains image filenames; the images are in `task4/test/`.
- The source description does not specify a test sample ID range (it shows `-`).

### Task 4 image folders

- Images are game screenshots at **200 × 100 pixels**.
- Filenames correspond to entries in `train.csv` and `test.csv`.
- Supported formats: PNG and JPG.

### Data cards (`datacard.txt`)

Each task folder includes a data card describing its features, data types, value ranges, and meanings. Use it as the reference when interpreting task data.

### Sample submission (`sample_submission.csv`)

- Contains **25,889 IDs**, from `ANS00001` to `ANS25889`.
- Uses a pipe-separated format: `id|task1|task2|task3|task4|task5`.
- Use it as the template for the final submission file.
