# เรียก Pipeline จาก Notebook ยังไง

ตัวอย่างนี้ใช้ **Task 1** (`src/task1/eda/build_eda_1.ipynb` และ `src/task1/models/build_model.ipynb`) เปลี่ยน `task=1` เป็น task อื่นได้เมื่อข้อมูลและวิธี train พร้อมแล้ว

## จำ 3 อย่างก่อน

| คำ | ใน pipeline นี้หมายถึง | ใครกำหนด |
| --- | --- | --- |
| `src` | ไฟล์ต้นทางที่ notebook อ่าน เช่น `data/raw/train.csv` หรือ artifact EDA ที่ save ไว้ | Notebook เลือกและอ่านเอง |
| `dst` | ตำแหน่งไฟล์ผลลัพธ์ เช่น `data/eda/eda_v1-train.parquet` | Function กำหนดจาก `task` + `name` |
| `card` | ข้อความบอกว่า artifact นี้คืออะไร ทำจากอะไร หรือมีข้อควรระวังอะไร | คนเรียก function กรอกเป็น **string สั้น ๆ** หรือ `dict` |

`save_data` รับ **DataFrame** ไม่ได้รับ path ของ raw CSV โดยตรง ดังนั้นให้อ่าน raw และทำ EDA ใน notebook ก่อน ส่วน `save_model` รับ **โมเดลที่ train แล้ว** ไม่ได้รับ path ของ notebook ทั้งสอง function คืน metadata ที่มี path ของไฟล์ที่เพิ่งเขียน

ทุก `name` เป็นชื่อเฉพาะในชนิด artifact นั้น ใช้ตัวอักษรอังกฤษ ตัวเลข `_` และ `-` เท่านั้น เช่น `eda_v1`, `rf_baseline` ห้ามมีช่องว่างหรือ `/`

## Cell 0: ให้ notebook import `src.pipeline` ได้

วาง cell นี้ไว้ต้น notebook ทั้ง EDA และ model ไม่ว่า Jupyter จะเปิดจาก root หรือจากโฟลเดอร์ notebook

```python
from pathlib import Path
import sys

ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "src" / "pipeline" / "save_data.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"

from src.pipeline.random_seed import set_seed
SEED = set_seed()  # default 42; เปลี่ยนได้ เช่น set_seed(2026)
```

`set_seed()` ตั้งค่าให้ Python `random` และ NumPy แล้วคืนค่า seed กลับมา ใช้ `SEED` ตัวนี้กับ `random_state` ทุกจุดที่รองรับด้วย สำหรับ Task 4 ที่ใช้ PyTorch ให้เรียก `set_seed(include_torch=True)` (ต้องติดตั้ง PyTorch) การตั้ง seed ช่วยให้ทำซ้ำได้ แต่ผลอาจต่างกันได้หาก library, hardware หรือขั้นตอนประมวลผลเปลี่ยน

## 1. ใน `build_eda_1.ipynb`: raw → EDA artifact

ตัวอย่างนี้สมมติว่า raw CSV อยู่ใน `src/task1/data/raw/` เปลี่ยนชื่อไฟล์ตามข้อมูลจริง

```python
import pandas as pd
from src.pipeline.save_data import save_data

raw_train = SRC / "task1" / "data" / "raw" / "train.csv"
raw_test = SRC / "task1" / "data" / "raw" / "test.csv"

train_df = pd.read_csv(raw_train)
test_df = pd.read_csv(raw_test)

# ทำ EDA / feature engineering ตรงนี้
# ตรวจ leakage ก่อน save; fit การแปลงข้อมูลด้วย Train/Validation เท่านั้น

data_info = save_data(
    task=1,
    name="eda_v1",
    train=train_df,
    test=test_df,
    card={
        "source": "task1/data/raw/train.csv + test.csv",
        "note": "EDA รอบแรก; ตัดคอลัมน์ที่ทำให้เกิด leakage แล้ว",
    },
    format="parquet",  # เปลี่ยนเป็น "csv" ได้
)

print(data_info["splits"]["train"]["path"])
print(data_info["splits"]["test"]["path"])
```

จะได้ไฟล์ใน `src/task1/data/eda/`:

```text
eda_v1-train.parquet
eda_v1-test.parquet
eda_v1.json         # เปิดไฟล์นี้ไฟล์เดียวเพื่อดู card, path, จำนวนแถว และคอลัมน์
eda_v1-train.json
eda_v1-test.json
```

ถ้าจะปรับ EDA รอบแรก **ก่อนนำ artifact ไปใช้จริง** แล้วเขียนชื่อเดิมซ้ำ ให้เรียก `save_data(..., force=True)`; ปกติชื่อซ้ำจะได้ `FileExistsError` เมื่อ artifact ถูกใช้เป็นผลจริงแล้ว ให้ตั้งชื่อใหม่เพื่อรักษาผลเก่า

## 2. ใน `build_model.ipynb`: EDA artifact → หลายโมเดล

อ่าน path จาก `eda_v1.json` จึงไม่ต้องจำชื่อไฟล์ Parquet/CSV เอง ตัวอย่างนี้ใช้ Task 1 และสมมติว่า label ชื่อ `target` ให้เปลี่ยนตาม `datacard.txt` จริง

```python
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.pipeline.save_model import save_model
from src.pipeline.evaluate_model import evaluate_model

DATA_NAME = "eda_v1"
TARGET = "target"   # เปลี่ยนตามชื่อ label จริง
ID = "id"           # เปลี่ยนตามชื่อ ID จริง

data_card_path = SRC / "task1" / "data" / "eda" / f"{DATA_NAME}.json"
data_card = json.loads(data_card_path.read_text(encoding="utf-8"))

train_path = SRC / data_card["splits"]["train"]["path"]
test_path = SRC / data_card["splits"]["test"]["path"]
read_data = pd.read_parquet if data_card["format"] == "parquet" else pd.read_csv
train_df = read_data(train_path)
test_df = read_data(test_path)

feature_cols = [column for column in test_df.columns if column != ID]
X = train_df[feature_cols]
y = train_df[TARGET]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = RandomForestClassifier(random_state=SEED)
model.fit(X_train, y_train)
valid_pred = model.predict(X_valid)

model_info = save_model(
    task=1,
    name="rf_v1",
    model=model,
    data_name=DATA_NAME,  # link กลับไปที่ eda_v1.json
    card={"idea": "Random forest baseline", "seed": SEED},
)
eval_info = evaluate_model(
    task=1,
    name="rf_v1",  # ต้องตรงกับ name ตอน save_model
    card={"split": "Validation 20%, stratified", "seed": SEED},
    y_true=y_valid,
    y_pred=valid_pred,
)

print(model_info["model_path"])
print(eval_info["metric"], eval_info["score"])
```

จะได้ `src/task1/save/rf_v1.joblib`, `rf_v1.json`, `rf_v1-evaluation.json` และ `evaluation_index.json` ใน notebook เดียวกัน train โมเดลเพิ่มได้โดยเปลี่ยน `name` และ `card` ของแต่ละโมเดล ไม่ต้องสร้าง notebook ใหม่

ถ้า model ต้องทำ preprocessing (imputation, scaling, encoding) ให้ใส่ขั้นตอนเหล่านั้นใน `sklearn.pipeline.Pipeline` แล้ว save ทั้ง pipeline เพื่อให้ตอน predict test ใช้ขั้นตอนเดียวกัน

## 3. อ่านผลและเลือกโมเดล

```python
from src.pipeline.read_evalutate_summarize import (
    best_model,
    read_evaluation_index,
)

index = read_evaluation_index(1)
print(index["evaluations"])
print(index["best_by_metric"])
print(best_model(1))  # เช่น "rf_v1"
```

`evaluation_index.json` อัปเดตอัตโนมัติทุกครั้งที่บันทึกผลของโมเดลใหม่ ถ้ามีหลาย metric ให้ระบุ เช่น `best_model(5, metric="proxy_metric_name")` เพราะคะแนนคนละชนิดเทียบกันตรง ๆ ไม่ได้

## 4. สร้าง prediction ของ Task 1

```python
from src.pipeline.build_sample_submission import build_sample_submission

submission_info = build_sample_submission(
    task=1,
    name="rf_v1_submit",
    card="Prediction จากโมเดลที่คะแนน validation ดีที่สุดของ Task 1",
    # model_name="rf_v1",  # ใส่เมื่อต้องการเลือกเอง; ไม่ใส่จะใช้ best_model(1)
    id_column="id",
)
print(submission_info["submission_path"])
```

Function นี้อ่าน `test` จาก dataset ที่ระบุใน model card, โหลดโมเดล แล้วเรียก `model.predict()` ได้ `src/task1/submission/rf_v1_submit.csv` รูปแบบ `id|task1` และ card `rf_v1_submit.json` ถ้าโมเดลภาพหรือ framework อื่นไม่ได้ใช้ `joblib`/`.predict()` ตามนี้ ให้ส่ง `loader=` และ `predict_fn=` เอง

## 5. รวมครบ 5 task เป็นไฟล์ Kaggle

ไฟล์ราย task ยังไม่ใช่ไฟล์ส่ง Kaggle ขั้นสุดท้าย ต้องมี prediction ทั้ง 5 task และ `sample_submission.csv` ก่อน

```python
from src.pipeline.build_sample_submission import combine_submissions

final_info = combine_submissions(
    name="final_v1",
    sample_path=ROOT / "CPE342-Hackathon" / "sample_submission.csv",
    task_files={
        1: SRC / "task1/submission/rf_v1_submit.csv",
        2: SRC / "task2/submission/task2_submit.csv",
        3: SRC / "task3/submission/task3_submit.csv",
        4: SRC / "task4/submission/task4_submit.csv",
        5: SRC / "task5/submission/task5_submit.csv",
    },
    card="รวม prediction จากโมเดลที่เลือกของทั้ง 5 task",
)
print(final_info["submission_path"])
```

แก้ `sample_path` ให้ตรงกับที่เก็บไฟล์จริง ผลลัพธ์อยู่ที่ `src/submission/final_v1.csv` และเป็น `id|task1|task2|task3|task4|task5` โดย function จะเช็ก ID กับ sample submission ก่อนเขียน

## Card แบบไหนพอ?

ใช้ string สั้น ๆ ได้เลย เช่น `card="ตัด post-outcome features; validation split 20%"` หากต้องการค้นย้อนหลังง่ายขึ้น ใช้ dict สั้น ๆ เช่น:

```python
card={
    "source": "eda_v1",
    "idea": "Random forest + class_weight balanced",
    "note": "ยังไม่ได้ tune threshold",
}
```

ไม่ต้องกรอก path ปลายทาง วันสร้าง จำนวนแถว หรือ score ซ้ำ เพราะ pipeline เติมให้เอง `force=True` มีในทุกขั้นที่สร้าง artifact; ใช้เฉพาะเวลาตั้งใจเขียนทับชื่อเดิม

## ข้อจำกัดที่ต้องรู้

- Task 3 ต้องส่ง `normalizer=` ให้ `evaluate_model` ตามนิยาม NMAE ของการแข่งขัน เช่น `evaluate_model(task=3, name="reg_v1", card="Validation 20%", y_true=y_valid, y_pred=valid_pred, normalizer=NMAE_DENOMINATOR)` โดย `NMAE_DENOMINATOR` ต้องมาจากกติกาจริง ยังไม่ควรเดาตัวหาร
- Task 5 ไม่มี `train.csv` ตามคำอธิบายข้อมูล แต่ `save_data` เวอร์ชันนี้รับทั้ง `train` และ `test` จึงยังไม่มี workflow แบบเดียวกับ Task 1; อย่านำ Test ไปเป็น Train เพื่อให้ function รันผ่าน ควรเคลียร์กติกา/แหล่ง train ก่อน
- ถ้ามี validation proxy สำหรับ Task 5 ให้บันทึกด้วย `evaluate_model(task=5, name="anomaly_v1", card="Validation proxy", score=proxy_score, metric_name="proxy_name", higher_is_better=True)` ซึ่งไม่ใช่ F₃ ของ leaderboard
- `build_sample_submission` คาดว่า model ที่ save รับ feature columns จาก EDA test ได้โดยตรง หากมี preprocessing ต้อง save ไปพร้อมโมเดล หรือกำหนด `predict_fn=` ให้ครบ
