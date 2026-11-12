# Task 4 — Game Title Detection

สถานะตามประกาศเปลี่ยน dataset วันที่ 12-11-2026: **แก้เฉพาะ `test.csv`** รอวางไฟล์ใหม่

| ส่วน | ไฟล์/ชุดข้อมูล | สถานะ |
| --- | --- | --- |
| Raw train | `data/raw/train.csv`, `data/raw/train/` | ใช้ได้ตามประกาศ |
| Raw test เก่า | `data/raw/test-outdate-12-11-2026.csv` | Outdated; เก็บไว้สืบย้อนผลเดิม |
| Raw test ใหม่ | `data/raw/test.csv` | ยังไม่มี; วางไฟล์ที่ดาวน์โหลดใหม่ด้วยชื่อนี้ |
| ภาพ test | `data/raw/test/` | โฟลเดอร์เดิม; ประกาศระบุว่าแก้เฉพาะ CSV ควรตรวจชื่อภาพกับ CSV ใหม่เมื่อได้รับ |
| Data card | `data/raw/datacard.txt` | ไฟล์เดิม; ตรวจเทียบกับ CSV ใหม่ |
| EDA เก่า | `data/eda/eda_original_v1.json`, `data/eda/eda_1_v1.json` พร้อม train/test Parquet | train split ยังมาจาก raw ที่ไม่เปลี่ยน; test split outdated และ dataset card ทั้งชุดยังไม่ใช่ชุดใหม่ |

Notebook ทั้งสองใน `eda/` ชี้ไปที่ `test-outdate-12-11-2026.csv` เพื่อสืบย้อนผลเดิม โมเดลที่ฝึกจาก train เดิมอาจยังใช้ได้ แต่ prediction/submission จาก test เก่าใช้แทนผลของ CSV ใหม่ไม่ได้ เมื่อได้ CSV ใหม่แล้วต้องตรวจการจับคู่ภาพและสร้าง EDA test/submission artifact ชื่อใหม่
