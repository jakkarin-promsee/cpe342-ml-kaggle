# Task 2 — Player Segment Classification

สถานะตามประกาศเปลี่ยน dataset วันที่ 12-11-2026: **เปลี่ยน dataset ทั้งชุด** ไฟล์ raw ใหม่อยู่ในเครื่องแล้ว แต่ notebook EDA ยังเป็นรุ่นเก่า

| ส่วน | ไฟล์/ชุดข้อมูล | สถานะ |
| --- | --- | --- |
| Raw เก่า | `data/raw/train-outdate-12-11-2026.csv`, `data/raw/test-outdate-12-11-2026.csv` | Outdated; เก็บไว้สืบย้อนผลเดิม |
| Raw ใหม่ | `data/raw/train.csv`, `data/raw/test.csv` | มีแล้ว; ใช้เป็น raw ชุดปัจจุบัน |
| Data card | `data/raw/datacard.txt` | มีแล้วพร้อม raw ชุดปัจจุบัน |
| EDA เก่า | `data/eda/eda_original_v1.json`, `data/eda/eda_combined_v1.json` พร้อม train/test Parquet | Outdated สำหรับ dataset ใหม่ |

Notebook `eda/build_eda_original.ipynb`, `eda/build_eda_1.ipynb` ชี้ไปที่ raw เก่าที่เปลี่ยนชื่อแล้ว และ `models/build_model.ipynb` อ่าน EDA artifact ชุดเก่า ให้ข้ามทั้งสามไฟล์ในการรันชุดใหม่ ไฟล์ EDA/model/submission ที่สร้างไว้ก่อนประกาศยังอ้างอิง dataset เก่า ต้องตรวจ schema/EDA และสร้าง notebook กับ artifact ชื่อใหม่ก่อนฝึกโมเดลจาก raw ใหม่
