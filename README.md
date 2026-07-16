# SDA ServiceDesk Intelligence

خط أنابيب بيانات (data pipeline) لتطبيق Machine Learning على بيانات نظام الدعم الفني
**ManageEngine ServiceDesk Plus** (on-premise, API v3).

المراحل: سحب البيانات → تنظيفها وتسطيحها (مع خيار إخفاء الهوية) → تحليل استكشافي → نماذج ML.

> ⚠️ كل الاتصال بالـ API **قراءة فقط (GET)** — لا يوجد أي إنشاء/تعديل/حذف على السيرفر.
>
> 🔒 البيانات تحتوي معلومات شخصية (أسماء/إيميلات/جوالات) — `data/` و `.env` وملفات
> الـ JSON الخام كلها git-ignored. أي ملف يطلع خارج الجهاز يجب أن يكون النسخة `*_anon`.

## هيكل المشروع

```
pipeline/      سكربتات السحب والتجهيز (sdp_client, extract, preprocess, anonymize)
insights/      التحليل الاستكشافي والداشبورد (eda, dashboard) + التقارير الناتجة
models/        نماذج الـ ML (لاحقاً)
config/        الإعدادات (settings.py يقرأ .env)
data/raw/      JSON خام (git-ignored)
data/processed/ جداول CSV/Parquet جاهزة (git-ignored)
```

## الإعداد

```bash
python -m venv venv
venv\Scripts\activate            # PowerShell:  venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env           # ثم عبّئ SDP_BASE_URL و SDP_AUTHTOKEN
```

## التشغيل (من روت المشروع)

### 1) السحب — يحتاج وصول للسيرفر

```bash
# اختبار سريع: تحقق من auth/pagination بدون تفاصيل
python -m pipeline.extract --modules requests,problems --limit 20 --skip-details

# عينة صغيرة مع التفاصيل (تأكد من شكل detail/notes)
python -m pipeline.extract --modules requests --limit 5

# السحب الكامل لكل الـ modules مع التفاصيل والمحادثات
python -m pipeline.extract
```

خيارات: `--modules`، `--limit N`، `--skip-details`، `--skip-notes`.
السحب **idempotent** — ملفات التفاصيل الموجودة تُتخطى، فيمكن إعادة التشغيل بأمان بعد أي انقطاع.

الـ modules: `requests`, `problems`, `changes`, `projects`, `solutions` (KB)
+ جداول مرجعية: `requesters`, `technicians`, `groups`, `categories`, `sites`.

### 2) التجهيز

```bash
python -m pipeline.preprocess               # يعالج كل ما في data/raw/
python -m pipeline.preprocess --anonymize   # + نسخ *_anon (أسماء→IDs، حذف الجوالات)

# أو تجربة على ملفات العينة مباشرة (بدون سيرفر):
python -m pipeline.preprocess --sample "apiResponse (1).json:requests" --sample "apiResponse.json:problems"
```

### 3) التحليل الاستكشافي

```bash
python -m insights.eda      # يكتب insights/eda_summary.md + رسومات PNG
```

### 4) الداشبورد

```bash
streamlit run insights/dashboard.py
```

داشبورد بتصميم enterprise (ثيم فاتح/داكن أصلي عبر `.streamlit/config.toml`، مكونات
Streamlit أصلية بدون CSS مخصص) مقسّم إلى تبويبات:

- **Requests**: KPIs (الإجمالي/المفتوحة/الإغلاق/خرق SLA/متوسط الحل/عدد الخدمات) + الحجم
  عبر الزمن + التوزيع حسب الخدمة/الفريق/الفني/الحالة/اللغة/الموقع
- **Problems**: KPIs + حسب الفئة/الحالة/الفريق/الفني
- **Changes / Projects / Solutions**: تعرض بياناتها تلقائياً بعد سحبها

مع فلاتر جانبية (تاريخ/فريق/خدمة/حالة/موقع) وجدول بيانات قابل للتنزيل في كل تبويب.
ألوان الرسوم من palette متحقق منها لعمى الألوان.

> متوسط الحل (Avg resolution) يظهر "—" حتى تُسحب التفاصيل الكاملة (closed_time) عبر extract.

## المخرجات

- `data/raw/` — JSON خام (git-ignored)
- `data/processed/*.csv|parquet` — جداول جاهزة للـ ML (+ نسخ `*_anon` عند الطلب)
- `insights/eda_summary.md` + رسومات — تحليل استكشافي وتوصية بأول هدف ML
- `insights/summary.md` — تقرير الخلاصة (بعد السحب الفعلي)

## الخطوة التالية

بعد السحب الفعلي والـ EDA: نماذج المرحلة الأولى (clustering للأعطال المتكررة +
تصنيف تلقائي للتذاكر بنموذج multilingual AR+EN)، ثم semantic search للحلول السابقة
وregression لتوقع وقت الحل.
