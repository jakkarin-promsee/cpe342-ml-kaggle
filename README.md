# CPE342 Player Intelligence System

โปรเจกต์นี้ใช้ notebook สร้าง EDA artifact ก่อน แล้วจึงอ่าน artifact ผ่าน JSON card ไปฝึกโมเดล ดูตัวอย่าง pipeline ที่ [src/pipeline/CALL.md](src/pipeline/CALL.md) และข้อกำหนดข้อมูลที่ [docs/kaggle-detial/data.md](docs/kaggle-detial/data.md)

## เตรียม Python และ Kaggle

ใช้ Python 3.11 ใน virtual environment แล้วติดตั้ง dependency:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
kaggle auth login
```

ต้องเข้าร่วมการแข่งขันและยอมรับกติกาบน Kaggle ก่อนดาวน์โหลด ใช้ slug ที่อยู่ท้าย URL `https://www.kaggle.com/competitions/<slug>`; โปรเจกต์นี้ไม่ได้เก็บ slug หรือ credential ไว้ใน repo

## โหลด raw dataset

```powershell
python src/load-raw-dataset.py --competition <slug>
```

สคริปต์ใช้ Kaggle CLI ดาวน์โหลดไฟล์การแข่งขัน, แตก ZIP และวาง `train.csv`/`test.csv`/`datacard.txt` ใน `src/taskN/data/raw/` รวมถึงภาพ `train/` และ `test/` ของ Task 4 ส่วน `sample_submission.csv` จะอยู่ที่ `src/submission/` Task 5 มีเพียง `test.csv` ตามข้อมูลการแข่งขัน สคริปต์ไม่สร้าง train ปลอม

ถ้าตำแหน่งปลายทางมีไฟล์อยู่แล้ว สคริปต์จะหยุดก่อนคัดลอก ใช้ `--force` เมื่อต้องการแทนไฟล์ raw ชุดปัจจุบัน:

```powershell
python src/load-raw-dataset.py --competition <slug> --force
```

ไฟล์ที่ตั้งชื่อ `*-outdate-12-11-2026.*` จะไม่ถูกเขียนทับ `src/task*/data`, `src/task*/save` และ `src/submission` ถูก Git ignore แต่ยังใช้งานในเครื่องตามปกติ

## รัน notebook

ดูลำดับก่อนรันจริง:

```powershell
python src/re-run-all-ipynb.py --dry-run
python src/re-run-all-ipynb.py --task-n 3 --dry-run
```

รันทุก task หรือเลือก task เดียว:

```powershell
python src/re-run-all-ipynb.py
python src/re-run-all-ipynb.py --task-n 3
```

สคริปต์รัน `src/taskN/eda/*.ipynb` ของทุก task ที่เลือกก่อน แล้วค่อยรัน `src/taskN/models/*.ipynb` เรียงตามชื่อ โดยให้ `build_eda_original.ipynb` มาก่อน `build_eda_1.ipynb` เมื่อสำเร็จจะบันทึก output ลง notebook เดิม ถ้า cell ผิดพลาดจะหยุดทันทีและไม่บันทึก output ของ notebook ที่ล้มเหลว การฝึกโมเดลจริงอาจใช้เวลานาน

รายชื่อ notebook ที่ข้ามกำหนดใน `IGNORE_NOTEBOOKS` ในสคริปต์ ตามสถานะใน `src/taskN/README.md` ปัจจุบัน EDA ของ Task 2, 4, 5 ยังอ้างอิง raw เก่า จึงข้าม EDA และโมเดลของสาม task นี้จนกว่าจะมี notebook รุ่นข้อมูลใหม่ การสั่ง `--task-n` ไปยัง task ที่ไม่มี notebook ปัจจุบันจะแจ้งข้อผิดพลาดชัดเจน

การรันซ้ำไม่ลบ EDA/model artifact เดิม บาง notebook อาจอ่าน artifact ที่มีอยู่แทนการสร้างใหม่ตามกติกา `save_data`/`save_model`; เมื่อทำผลใหม่จริงให้ตั้งชื่อ artifact ใหม่และอัปเดต notebook ให้ใช้ชื่อใหม่นั้น
