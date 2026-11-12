# Task 5 — Account Security Monitoring

สถานะตามประกาศเปลี่ยน dataset วันที่ 12-11-2026: **เปลี่ยน dataset ทั้งชุด** ไฟล์ raw ใหม่อยู่ในเครื่องแล้ว แต่ notebook EDA ยังเป็นรุ่นเก่า

| ส่วน | ไฟล์/ชุดข้อมูล | สถานะ |
| --- | --- | --- |
| Raw เก่า | `data/raw/test-outdate-12-11-2026.csv` | Outdated; เก็บไว้สืบย้อนผลเดิม |
| Raw ใหม่ | `data/raw/test.csv` | มีแล้ว; ใช้เป็น raw ชุดปัจจุบัน |
| Raw train | ไม่มี `train.csv` ในชุดข้อมูลที่มีอยู่ | อย่าสร้างหรือใช้ test แทน train |
| Data card | `data/raw/datacard.txt` | มีแล้วพร้อม raw ชุดปัจจุบัน |
| EDA เก่า | `data/eda/eda_original_v1.json`, `data/eda/eda_1_v1.json` พร้อม test Parquet | Outdated สำหรับ dataset ใหม่ |

Notebook `eda/build_eda_original.ipynb`, `eda/build_eda_1.ipynb` ชี้ไปที่ raw เก่า และ `models/build_model.ipynb` อ่าน EDA artifact ชุดเก่า ให้ข้ามทั้งสามไฟล์ในการรันชุดใหม่ EDA/model/submission เดิมอ้างอิง test ชุดเก่า ต้องยืนยัน workflow ของ Task 5 กับผู้จัดตามข้อกำหนดการแข่งขันก่อนสร้าง artifact ใหม่
