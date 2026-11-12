# Task 5 — Account Security Monitoring

สถานะตามประกาศเปลี่ยน dataset วันที่ 12-11-2026: **เปลี่ยน dataset ทั้งชุด** รอวางไฟล์ใหม่

| ส่วน | ไฟล์/ชุดข้อมูล | สถานะ |
| --- | --- | --- |
| Raw เก่า | `data/raw/test-outdate-12-11-2026.csv` | Outdated; เก็บไว้สืบย้อนผลเดิม |
| Raw ใหม่ | `data/raw/test.csv` | ยังไม่มี; วางไฟล์ที่ดาวน์โหลดใหม่ด้วยชื่อนี้ |
| Raw train | ไม่มี `train.csv` ในชุดข้อมูลที่มีอยู่ | อย่าสร้างหรือใช้ test แทน train |
| Data card | `data/raw/datacard.txt` | ไฟล์เดิม; ต้องตรวจเทียบกับ dataset ใหม่ |
| EDA เก่า | `data/eda/eda_original_v1.json`, `data/eda/eda_1_v1.json` พร้อม test Parquet | Outdated สำหรับ dataset ใหม่ |

Notebook ทั้งสองใน `eda/` ชี้ไปที่ raw เก่าที่เปลี่ยนชื่อแล้วเพื่อสืบย้อนผลเดิม EDA/model/submission เดิมอ้างอิง test ชุดเก่า เมื่อได้ raw ใหม่แล้วต้องยืนยัน workflow ของ Task 5 กับผู้จัดตามข้อกำหนดการแข่งขันก่อนสร้าง artifact ใหม่
