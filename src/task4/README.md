# Task 4 — Game Title Detection

สถานะตามประกาศเปลี่ยน dataset วันที่ 12-11-2026: **แก้เฉพาะ `test.csv`** ไฟล์ raw ชุดปัจจุบันอยู่ในเครื่องแล้ว แต่ notebook EDA ยังอ่าน test เก่า

| ส่วน | ไฟล์/ชุดข้อมูล | สถานะ |
| --- | --- | --- |
| Raw train | `data/raw/train.csv`, `data/raw/train/` | มีแล้ว; ใช้เป็น raw ชุดปัจจุบัน |
| Raw test เก่า | `data/raw/test-outdate-12-11-2026.csv` | Outdated; เก็บไว้สืบย้อนผลเดิม |
| Raw test ใหม่ | `data/raw/test.csv`, `data/raw/test/` | มีแล้ว; ใช้เป็น raw ชุดปัจจุบัน |
| Data card | `data/raw/datacard.txt` | มีแล้วพร้อม raw ชุดปัจจุบัน |
| EDA เก่า | `data/eda/eda_original_v1.json`, `data/eda/eda_1_v1.json` พร้อม train/test Parquet | test split outdated; ตรวจ train split เทียบ manifest ปัจจุบันก่อนนำไปใช้ |

Notebook `eda/build_eda_original.ipynb`, `eda/build_eda_1.ipynb` ชี้ไปที่ `test-outdate-12-11-2026.csv` และ `models/build_model.ipynb` อ่าน EDA artifact ชุดเก่า ให้ข้ามทั้งสามไฟล์ในการรันชุดใหม่ Prediction/submission จาก test เก่าใช้แทนผลของ CSV ใหม่ไม่ได้ ต้องตรวจการจับคู่ภาพและสร้าง notebook กับ EDA/model/submission artifact ชื่อใหม่
