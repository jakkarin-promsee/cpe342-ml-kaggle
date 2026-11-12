# Task 2 — Player Segment Classification

สถานะตามประกาศเปลี่ยน dataset วันที่ 12-11-2026: **เปลี่ยน dataset ทั้งชุด** รอวางไฟล์ใหม่

| ส่วน | ไฟล์/ชุดข้อมูล | สถานะ |
| --- | --- | --- |
| Raw เก่า | `data/raw/train-outdate-12-11-2026.csv`, `data/raw/test-outdate-12-11-2026.csv` | Outdated; เก็บไว้สืบย้อนผลเดิม |
| Raw ใหม่ | `data/raw/train.csv`, `data/raw/test.csv` | ยังไม่มี; วางไฟล์ที่ดาวน์โหลดใหม่ด้วยชื่อนี้ |
| Data card | `data/raw/datacard.txt` | ไฟล์เดิม; ต้องตรวจเทียบกับ dataset ใหม่ |
| EDA เก่า | `data/eda/eda_original_v1.json`, `data/eda/eda_combined_v1.json` พร้อม train/test Parquet | Outdated สำหรับ dataset ใหม่ |

Notebook ทั้งสองใน `eda/` ชี้ไปที่ raw เก่าที่เปลี่ยนชื่อแล้ว เพื่อให้สร้างผลเดิมซ้ำได้ ไฟล์ EDA/model/submission ที่สร้างไว้ก่อนประกาศยังอ้างอิง dataset เก่า ห้ามใช้เป็นผลของ dataset ใหม่ เมื่อได้ raw ใหม่แล้วต้องตรวจ schema/EDA และสร้าง artifact ชื่อใหม่
