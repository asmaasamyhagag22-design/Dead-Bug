# HANDOVER — كل اللي محتاجاه تكمّلي على جهاز تاني

> آخر تحديث: ٢٨ يوليو ٢٠٢٦ · المناقشة: السبت ١ أغسطس
> اقري [PROCESS.md](PROCESS.md) للـ **ليه**، والملف ده للـ **إزاي تكمّلي**.

---

## ٠. الهدف الحقيقي — اقري ده الأول

⚠️ **الوثائق التصميمية في `Downloads` بتوصف نظام بحثي offline. ده مش الهدف.**

الهدف: **تطبيق تدريب حي**. المستخدم يفتح، يختار Dead Bug، الكاميرا تشتغل، وياخد لكل عدة
أخضر/أحمر + تعليم على مكان الغلط في الجسم + صوت + عدّاد للصح + تقرير في الآخر.

**الأولوية: يشتغل على أي فيديو YouTube.** الكاميرا الحية ثانوية.

الشغل البحثي (الـ benchmark) **مش ضايع** — هو مخ التطبيق + الدليل إن كود التقييم سليم.
بس لما الاتنين يتعارضوا، **التطبيق بيكسب**.

---

## ١. الحاجات اللي مش في Git — لازم تجيبيها معاكِ

| الحاجة | الحجم | إزاي تجيبيها |
|---|---|---|
| `pose_landmarker_heavy.task` | ٣٠ ميجا | لينك التحميل تحت |
| `data/clips/*.mp4` | ١٠ ملفات، ٥٨ ميجا | **انسخيها من الجهاز القديم — مالهاش مصدر تاني** |
| `venv/` و `venv-a/` | ~١٫٥ جيجا | يتبنوا من جديد، أوامر تحت |
| `data/rehabpile/` | ~٥٠٠ ميجا | بيتحمّل لوحده أول تشغيل |

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

**أهم حاجة تنسخيها: `data/clips/`.** دي الكليبات اللي جمعتيها، ومفيش نسخة تانية منها.

---

## ٢. تجهيز البيئة

```bash
git clone https://github.com/asmaasamyhagag22-design/Dead-Bug.git
cd Dead-Bug
```

### البيئة الأساسية (التطبيق + الإشارات)

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

المتحقق منه شغّال (Python **3.13.5**، ويندوز):

```
mediapipe 0.10.35 · opencv-python 5.0.0.93 · numpy 2.5.1 · scipy 1.18.0
pandas 3.0.5 · matplotlib 3.11.1 · PyYAML 6.0.3 · pytest 9.1.1
pyarrow 25.0.0 · pyttsx3 2.99 (الصوت) · pywin32 312
```

### بيئة الـ benchmark — **منفصلة إجبارياً**

```bash
uv venv venv-a --python 3.13
uv pip install --python venv-a -r requirements-a.txt
```

المتحقق منه شغّال:

```
aeon 1.5.0 · tensorflow 2.21.0 · keras 3.15.0
numpy 2.3.5 · scipy 1.17.1 · pandas 2.3.3 · scikit-learn 1.8.0 · numba 0.63.1
```

⚠️ **ليه منفصلة:** aeon بيحدّ `numpy<2.5` و `pandas<2.4` و `scipy<1.18` — كلهم **أقل** من
اللي mediapipe و opencv شغّالين عليه. تثبيته في نفس البيئة بينزّل ستاك شغّال.

⚠️ **الفخ:** ثبّتي `aeon==1.5.0` **صراحةً**. لو حطيتي قيد زي `numpy<2.5` بدل كده، الـ resolver
بيختار numpy 2.4.6، و numba بيرفضه، وبدل ما يرجع يقلل numpy **بيثبّت `aeon==0.0.0`** — وده
package فاضي على PyPI. والتثبيت **بينجح بـ exit code 0** والـ log بيقول `+ aeon==0.0.0`.
وبعدين `import aeon` بيفشل.

---

## ٣. التأكد إن كل حاجة شغالة

```bash
./venv/Scripts/python.exe -m pytest tests/ -q
```

**١١٣ اختبار، كلهم لازم يعدّوا.** `pythonpath=["src"]` في `pyproject.toml` بيخلي كل حاجة
تشتغل من جذر الريبو من غير أي تثبيت.

لو **بوابة ٠** (`test_normalize.py` + `test_skeleton.py`) وقعت — **وقّفي كل حاجة**. التطبيع
مكسور، وكل رقم بعد كده بلا معنى.

---

## ٤. تشغيل التطبيق

```bash
./venv/Scripts/python.exe scripts/run_live.py --source "data/clips/videoplayback (1).mp4"
```

| اللي عايزاه | الأمر |
|---|---|
| كاميرا | `--source 0` |
| فيديو | `--source "path/to/video.mp4"` |
| من غير صوت | `--no-voice` |
| من غير شاشة (اختبار) | `--headless` |

المفاتيح أثناء التشغيل: `q` خروج + تقرير · `r` إعادة · `space` وقف مؤقت

**مراحل الجلسة:** نام ثابت ٣ ثواني (يحدد الأرض) ← ٣ عدات هادية (خط الأساس بتاعك) ←
كل عدة بعد كده بتتحاكم على خط الأساس ده.

المخرجات: `reports/session_report.json` + `.png`

---

## ٥. تشغيل الـ benchmark

```bash
./venv-a/Scripts/python.exe scripts/run_masar_a.py --all --cheap        # ٣٩ dataset
./venv-a/Scripts/python.exe scripts/run_masar_a.py --models litemv --litemv-classifiers 3 --dataset KIMORE_clf_bn_LA
```

**قابل للاستئناف** — بيقرا اللي خلص من الـ CSV ويتخطاه. وبيكتب بعد كل dataset، فلو وقع
مفيش حاجة بتضيع.

---

## ٦. النتايج اللي عندنا دلوقتي

### Track A — ٣٩ dataset، خلص ✅

| موديل | متوسط الرُتبة | فاز | متوسط macro-F1 | datasets |
|---|---|---|---|---|
| **MiniRocket** | **1.63** | **25/39** | 0.684 | 39 |
| RF (summary) | 2.14 | 12 | 0.610 | 39 |
| RF (flatten) | 2.73 | 6 | 0.528 | 39 |
| majority | 3.58 | 4 | 0.379 | 39 |
| LITEMV | — | 0 | 0.592 ± 0.285 | **1 بس** |

LITEMV أخد **٧٣٦٥ ثانية (ساعتين)** لـ dataset واحد بـ 5 classifiers، وبعدين مات (ذاكرة).
MiniRocket بياخد **ثوانٍ** ويكسبه.

**الصياغة الصادقة:** *"MiniRocket جه الأول في ٢٥ من ٣٩. LITEMV اتقيّم على عيّنة بسبب
التكلفة — ساعتين مقابل ثوانٍ — وماتفوّقش عليه فيها."*

⚠️ متقوليش *"استنسخت نتايج الورقة"* — الورقة نفسها ماقدرناش نجيب إعداداتها. قولي
*"شغّلت على الإعدادات الافتراضية للمكتبة"*.

### Track B — قياسات حقيقية

**MediaPipe مابيفشلش مستلقياً** (عكس أهم خطر في الوثيقة الأصلية):
detection **٩٠–١٠٠٪**، visibility للمفاصل الأساسية **≈0.999**. → RTMPose اتشال من الخطة.

**فرز الكليبات** (`view_score` بالـ pixel space):

| كليب | det% | view | torsoCV | الحكم |
|---|---|---|---|---|
| `videoplayback (3)` | 92.7 | 0.119 | 0.020 | جانبي نضيف — **الوحيد** |
| `videoplayback (6)` | 90.1 | 0.112 | 0.495 | الكاميرا بتزوّم |
| `Video Project 2mk` | 17.3 | 0.103 | 0.020 | detection بينهار |
| `videoplayback (1)` | 97.8 | 0.123 | 0.274 | مائل + حركة |
| `clip` ≡ `videoplayback (4)` | 100 | 0.154 | 0.029 | **نسخة مكررة (md5)** |

**التقسيم التلقائي للتمرين:**

| كليب | نسبة التمرين | عدات |
|---|---|---|
| `clip.mp4` | **١٠٠٪** | 4 |
| `videoplayback (1)` | ٧٠٪ | 6 |
| `videoplayback (3)` | **٠٪** | 0 |

`videoplayback (3)` — ٩٨ ثانية **مافيهاش ولا عدة**. فيديو تعليمي، المدرب بيتكلم.
اتأكدنا بطريقتين مستقلتين (العدّاد الحي + الـ offline segmenter) واتفقوا.

> **فيديوهات YouTube التعليمية مصدر كلام مش مصدر عدات.** وده أهم سبب إن تصويرك مهم.

---

## ٧. الفخاخ اللي اكتشفناها — متقعيش فيها تاني

| المشكلة | التفاصيل |
|---|---|
| **MediaPipe بيقتل الـ process** | قراءة الـ mask بترمي `Check failed: 1 == ChannelSize()` — انهيار native من غير traceback — لما عرض الفريم مش من مضاعفات ٤. عرض 1006 = 4024 بايت/صف. **الحل:** تقريب الأبعاد لمضاعف ١٦ (`ingest.snap_dims_to`) |
| **detector لكل كليب** | `RunningMode.VIDEO` بيحتفظ بحالة tracking وبيطلب timestamps تصاعدية. إعادة استخدام detector عبر كليبات = انهيار |
| **`view_score` بالبكسل مش normalized** | x و y ليهم مقاييس مختلفة. قياس حقيقي: 0.240 غلط مقابل 0.131 صح |
| **`floor_inlier_ratio`** | لازم يتحسب على **أوطى شريحة** من الظل بس. لو اتحسب على كل الأعمدة بيبقى **عكسياً مع الأداء الصحيح** (الأطراف مرفوعة عمداً) |
| **الفلتر السببي** | `zero_phase=True` للـ offline بس. **مستحيل live** — بيبص للأمام |
| **LITEMV import** | جوه الدالة مش على مستوى الملف — بيبني Keras في `__init__` فبيسقط الـ baselines معاه |
| **aeon بيكتب `.keras`** | في `file_path` اللي افتراضيه `./`. مثبّت على `models/` والـ gitignore ماسكه |
| **`class_weight`** | RF و MiniRocket بيقبلوه. **LITEMV لأ** (aeon مش موفّره). ده جزء من النتيجة مش عيب يتخبّى |

---

## ٨. اللي فاضل

### 🔴 العائق الوحيد: التصوير

٤ فيديوهات، ربع ساعة، موبايل على الأرض جنبك بمترين، نايمة على ضهرك والركب ٩٠°:

1. ١٠ عدات و**الضهر مقوّس عمداً**
2. ١٠ عدات **بسرعة من غير وقفة**
3. ١٠ عدات **والحوض بيلف**
4. ١٠ عدات **صح**

\+ ٣ مقاطع قصيرة صح عند **٠° / ٣٠° / ٤٥°** لمعايرة زاوية الكاميرا على أرقام حقيقية.

**من غيرهم:** ادعاء "بيكتشف الأخطاء" يتشال من المناقشة. الـ label بييجي من **نيّتك وقت
التصوير** — مصدر مستقل عن الإشارة، فمشروع منهجياً.

### الباقي

| الحالة | الحاجة |
|---|---|
| ⬜ | بوابة ١ — رسمة `lumbar_gap` صح مقابل مقوّس |
| ⬜ | حساسية C7 على الـ ٤٠ عدة الغلط |
| ⬜ | معايرة `view_score` من فيديوهات ٠/٣٠/٤٥ |
| ⬜ | `data/clips.csv` — تتكتب بإيد من `reports/dedup.csv` |
| ⬜ | `dataset/{build,schema,normative}.py` · `qc/report.py` · `modeling/features.py` · `cli.py` |
| ⬜ | السلايدز · `LIMITATIONS.md` · `MODEL_CARD.md` · `DATASHEET.md` |

---

## ٩. قواعد التقرير اللي متتكسرش

- **الرقم الرسمي = متوسط الـ folds ± std.** الـ confusion matrix المجمّعة للعرض بس.
- **مفيش نسبة عارية.** `"0/48 rep، 95% CI [0، 7.4%]، LOSO على N=6"` — مش `"0% إنذار كاذب"`.
  `evaluate.wilson_interval` موجودة للغرض ده.
- **N في الجملة نفسها**، مش في هامش.
- **قولي الإعداد.** أي رقم من LITEMV مكتوب معاه `n_classifiers` و `n_epochs`.
- **ممنوع** `lumbar_gap` تعمل labels **و** تبقى input في نفس الوقت — دائري.

> **This system detects deviations from a reference movement pattern. It is not a
> diagnostic or clinical tool, was not validated on a patient population, and must
> not be used to guide medical decisions.**

---

## ١٠. خريطة الكود

```
src/deadbug/
├── config.py            كل الثوابت من configs/base.yaml، seeding، manifests
├── pose/
│   ├── skeleton.py      ⭐ MP33/COCO-17، جداول الـ flip، الشجرة، ٨ زوايا (D=109)
│   ├── mediapipe_backbone.py   + extract_clip()
│   └── draw.py          رسم يدوي بـ OpenCV (mp.solutions اتشال من المكتبة)
├── geometry/
│   ├── normalize.py     إزاحة → مقياس (median) → دوران · view_score
│   ├── filters.py       One Euro (سببي + zero-phase) · فجوات · resample
│   └── floor.py         تقدير الأرض بدون معايرة
├── signals/
│   ├── lumbar.py        ⭐ الإشارة الأساسية · perturb_mask (V1)
│   ├── ribcage.py       نفس الصيغة، نطاق [0.45,0.75] · rib_lead
│   ├── rotation.py      انحراف عن خط الأساس، **مش قيمة مطلقة**
│   └── smoothness.py    SPARC (مش LDLJ)
├── segment/
│   ├── reps.py          offline: find_peaks + prominence تلقائي
│   └── activity.py      ⭐ لقطات التمرين تلقائياً — مفيش قص يدوي
├── live/                ⭐ التطبيق
│   ├── counter.py       عدّاد سببي (state machine، من غير lookahead)
│   ├── session.py       المراحل + المعايرة + الحكم
│   ├── feedback.py      صوت في thread منفصل
│   └── ui.py            overlay + تقرير
├── modeling/
│   ├── evaluate.py      ⭐ numpy بس — الكود المشترك بين البيئتين
│   ├── rehabpile.py     تحميل + fallback للأسماء
│   ├── baselines.py     summary_features · RF · MiniRocket
│   └── train.py         fit_predict_* لكل درجة
└── ingest/
    ├── video_source.py  ⭐ snap لمضاعف ١٦ + seek بدل ffmpeg
    └── dedup.py         md5 + dHash + طبقة مراجعة

scripts/run_live.py      ⭐ التطبيق
scripts/run_masar_a.py   الـ benchmark
scripts/run_pose.py      السكربت الأصلي (اتفكّك، متسيبه للمرجع)
```

**الملفات اللي متغيّرش:** `pose/skeleton.py` (جداول الـ flip — نسيان تبديل شمال/يمين =
الموديل يتعلم labels متناقضة والـ loss ينزل عادي) · `modeling/evaluate.py` (numpy بس عن
قصد، عشان البيئتين يستوردوه).
