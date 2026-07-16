# SDA ServiceDesk Intelligence

خط أنابيب بيانات (data pipeline) لتطبيق Machine Learning على بيانات نظام الدعم الفني
**ManageEngine ServiceDesk Plus** (on-premise, API v3).

المرحلة الحالية = **بناء الـ pipeline فقط**: سحب البيانات → تنظيفها وتسطيحها → تحليل استكشافي
يقرر أي هدف ML نبدأ فيه. بناء النموذج نفسه مؤجل حتى تظهر نتائج الـ EDA.

> ⚠️ كل الاتصال بالـ API **قراءة فقط (GET)** — لا يوجد أي إنشاء/تعديل/حذف على السيرفر.

## المتطلبات والإعداد

```bash
python -m venv venv
venv\Scripts\activate            # PowerShell:  venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env           # ثم عبّئ SDP_BASE_URL و SDP_AUTHTOKEN
```

## الملفات

| ملف | الوظيفة |
|-----|---------|
| `config.py` | تحميل الإعدادات من `.env` |
| `src/sdp_client.py` | عميل HTTP مشترك (auth, pagination, retries) — GET فقط |
| `src/extract.py` | سحب البيانات إلى `data/raw/` (JSON خام، idempotent) |
| `src/preprocess.py` | تسطيح JSON → جداول `data/processed/*.csv/.parquet` |
| `src/eda.py` | تحليل استكشافي + رسومات + `eda_summary.md` |
| `src/dashboard.py` | داشبورد تفاعلي (Streamlit) بثيم نيون + KPIs تشغيلية |
| `src/dashboard_theme.py` | ثيم النيون (ألوان + قالب Plotly + CSS متحرك) |

الـ modules المسحوبة: `requests`, `problems`, `changes`, `projects`, `solutions` (KB)،
بالإضافة إلى جداول مرجعية: `requesters`, `technicians`, `groups`, `categories`, `sites`.

## التشغيل

### 1) السحب (يحتاج وصول للسيرفر)

```bash
# اختبار سريع: تحقق من auth/pagination بدون تفاصيل
python -m src.extract --modules requests,problems --limit 20 --skip-details

# عينة صغيرة مع التفاصيل (تأكد من شكل detail/notes)
python -m src.extract --modules requests --limit 5

# السحب الكامل لكل الـ modules مع التفاصيل والمحادثات
python -m src.extract
```

خيارات: `--modules`، `--limit N`، `--skip-details`، `--skip-notes`.
السحب **idempotent** — ملفات التفاصيل الموجودة تُتخطى، فيمكن إعادة التشغيل بأمان بعد أي انقطاع.

### 2) التجهيز

```bash
python -m src.preprocess                    # يعالج كل ما في data/raw/

# أو تجربة على ملفات العينة مباشرة (بدون سيرفر):
python -m src.preprocess --sample "apiResponse (1).json:requests" --sample "apiResponse.json:problems"
```

### 3) التحليل الاستكشافي

```bash
python -m src.eda        # يكتب data/processed/eda/eda_summary.md + رسومات PNG
```

### 4) الداشبورد التفاعلي (Streamlit)

```bash
streamlit run src/dashboard.py
```

داشبورد بثيم نيون متحرك (خلفية aurora، كروت KPI متوهّجة، أزرار/رسومات بحركة) مقسّم إلى
**تبويبات لكل module**: Requests / Problems / Changes / Projects / Solutions (KB).

- **Requests**: KPIs (الإجمالي/المفتوحة/الإغلاق/خرق SLA/متوسط الحل/عدد الخدمات) + الحجم عبر الزمن +
  التوزيع حسب **الخدمة (template)** / الفريق / **الفني (حمل العمل)** / الحالة / اللغة / الموقع
- **Problems**: KPIs + حسب الفئة/الحالة/الفريق/الفني
- **Changes / Projects / Solutions**: تبويبات تعرض بياناتها تلقائياً بعد سحبها (أو رسالة "شغّل الـ extractor")

مع فلاتر جانبية (تاريخ/فريق/**خدمة**/حالة/موقع) وجدول بيانات قابل للتنزيل في كل تبويب.

> ملاحظة: ألوان الرسومات من palette آمنة لعمى الألوان؛ النيون مُستخدم للزينة فقط (لا يمثّل بيانات).
> متوسط الحل (Avg Resolution) للتذاكر يظهر "—" حتى تُسحب التفاصيل الكاملة (closed_time) عبر `extract.py`.

## المخرجات

- `data/raw/` — JSON خام (git-ignored، قد يحتوي بيانات شخصية)
- `data/processed/*.csv` و `*.parquet` — جداول جاهزة للـ ML
- `data/processed/eda/` — رسومات + `eda_summary.md` مع توصية بأول هدف ML

## الخطوة التالية

بناءً على `eda_summary.md` نبني أول نموذج — الأرجح تصنيف التذاكر (multilingual AR+EN)
أو توقع خرق SLA.
