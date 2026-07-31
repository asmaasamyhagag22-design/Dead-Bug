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

> 🔴 **لو الريبو جوه مجلد متزامن مع السحابة** (زي `G:\Other computers\...` بتاع Google Drive)
> — **ابني البيئة برّة المجلد ده**. تثبيت عشرات الآلاف من الملفات الصغيرة على نظام ملفات
> افتراضي بيفشل بـ `os error 1450: Insufficient system resources`، والـ venv بيطلع نصّه
> مثبّت: `import numpy` بينجح بس `numpy.array` مش موجودة. اتأكدنا من ده على الجهاز ده.
>
> ```bash
> uv venv "C:/Users/<you>/venvs/deadbug" --python 3.13
> uv pip install --python "C:/Users/<you>/venvs/deadbug" -r requirements.txt
> "C:/Users/<you>/venvs/deadbug/Scripts/python.exe" -m pytest tests/ -q
> ```
>
> الكود يقعد على السحابة عادي — الـ venv بس اللي لازم يبقى على قرص محلي.
> الـ `Makefile` بياخد `PY=...` عشان كده.

### البيئة الأساسية (التطبيق + الإشارات)

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

المتحقق منه شغّال — **معاد التأكد ٣١ يوليو** (Python **3.13.13**، ويندوز):

```
mediapipe 1.0.0 · opencv-python 5.0.0 · numpy 2.5.1 · scipy 1.18.0
pandas 3.0.5 · matplotlib · PyYAML · pytest · pyarrow · pyttsx3 (الصوت)
```

> الجدول القديم كان بيقول `mediapipe 0.10.35` و Python `3.13.5` — دول أرقام الجهاز
> القديم. الـ ١٤٢ اختبار بيعدّوا على **mediapipe 1.0.0**، فالكود شغّال على الاتنين.
> ملاحظة: تحذيرات الانهيار في §7 اتقاست على 0.10.35؛ `snap_dims_to: 16` سايبينه شغّال
> على أي حال لأن التكلفة صفر والانهيار كان بيقتل الـ process من غير traceback.

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
| **MiniRocket** | **1.65** | **25/39** | 0.684 | 39 |
| RF (summary) | 2.17 | 11 | 0.610 | 39 |
| LITEMV | 2.67 | 1 | 0.582 | **٣ بس** |
| RF (flatten) | 2.76 | 6 | 0.528 | 39 |
| majority | 3.60 | 4 | 0.379 | 39 |

> الأرقام دي من `reports/masar_a_results.md` (رن ٢٨ يوليو ٧:٢٧م). النسخة الأقدم من الجدول
> كانت بتقول 1.63 و LITEMV على dataset واحد — **اتجاوزت**. الملف هو المصدر.

LITEMV أخد **٧٣٦٥ ثانية (ساعتين)** لـ dataset واحد بـ 5 classifiers. MiniRocket بياخد
**ثوانٍ**. الرن ده **خلص فعلاً وموجود في جدول النتايج** — الانهيار بسبب الذاكرة حصل في رن
تاني، فمتقوليش "مات" عن الرن اللي أرقامه منشورة.

**الصياغة الصادقة:** *"MiniRocket جه الأول في ٢٥ من ٣٩. LITEMV اتقيّم على ٣ datasets بسبب
التكلفة — ساعتين مقابل ثوانٍ — وماتفوّقش عليه فيهم."*

⚠️ متقوليش *"استنسخت نتايج الورقة"* — الورقة نفسها ماقدرناش نجيب إعداداتها. قولي
*"شغّلت على الإعدادات الافتراضية للمكتبة"*.

### Track B — قياسات حقيقية

**MediaPipe مابيفشلش مستلقياً** (عكس أهم خطر في الوثيقة الأصلية): على **٧ من ١٠**
كليبات الـ detection **٩٣–١٠٠٪**، وvisibility للمفاصل الأساسية **0.985–1.000 على العشرة
كلهم**. → RTMPose اتشال من الخطة.

⚠️ التلاتة الباقيين (`Video Project 2` 77.1٪ · `2 (2)` 55.5٪ · `2mk` **17.8٪**) تحت ٩٠٪،
بس ده **مش فشل في وضع الاستلقاء**: الفيديوهات دي فيها شاشات عناوين وتمارين تانية، والـ
detection جواه مقاطع التمرين نفسها بيوصل ٩٦–١٠٠٪. متقوليش "٩٠–١٠٠٪" من غير القيد ده.

**فرز الكليبات** — الأرقام دي معادة القياس **٣١ يوليو** على العشر كليبات كلهم.
المصدر: `reports/triage.csv`. **الجدول القديم هنا كان كل صفوفه قديمة واتشال.**

| كليب | det% | view_score | view | torsoCV | عدات | الحكم |
|---|---|---|---|---|---|---|
| `videoplayback (3)` | 93.0 | 0.117 | **side** | 0.024 | 9 | الجانبي النضيف **الوحيد** |
| `videoplayback (1)` | 99.7 | 0.128 | oblique45 | **0.286** | 5 | عدات حقيقية، بس بيتحرك |
| `clip` ≡ `videoplayback (4)` | 100 | 0.155 | oblique45 | 0.031 | 4 | **مكرر (md5)** |
| `videoplayback (6)` | 95.2 | 0.232 | other | **0.575** | 13 | الكاميرا بتزوّم |
| `Recording …171125` | 100 | 0.138 | oblique45 | 0.060 | 4 | |
| `Recording …171515` | 100 | 0.279 | other | 0.025 | **0** | مافيهوش تمرين فعلاً |
| `Video Project 2` | 77.1 | 0.397 | other | 0.042 | — | مرفوض (floor) |
| `Video Project 2 (2)` | 55.5 | 0.172 | oblique45 | 0.063 | — | مرفوض (floor) |
| `Video Project 2mk` | **17.8** | 0.137 | oblique45 | 0.047 | — | detection بينهار |

⚠️ **الادعاء ده اتسحب:** كان مكتوب إن `videoplayback (3)` ٩٨ ثانية مافيهاش ولا عدة،
واتأكدنا بطريقتين مستقلتين. **الاتنين غلط.** الكود الحالي بيطلّع **٩ كشوفات على ٨٩.٥٪**
من الكليب، والعدّاد الحي والـ segmenter **بيقروا نفس الـ `max_extend_s = 6.0`** فاتفاقهم
ماكانش دليل مستقل أصلاً. اقري [LIMITATIONS.md](LIMITATIONS.md) §9 — فيها جدول الحساسية.

اللي **لسه صحيح**: التسع كشوفات دي مددها 3.6–6.0 ثانية مقابل إيقاع مدرَّب 1–2 ثانية،
وتسلسل التبادل `LRLLRLRLR` بينما الـ Dead Bug بيتبادل بانتظام — يعني مدرّب بيتنقل بين
عروض، مش set. بس "شكلها مش عدات" ادعاء أضعف بكتير من "مافيهاش عدات".

⚠️ **`excursion_peak` بيوصل 10.2 طول جذع على `videoplayback (6)`** — رقم مستحيل فيزيائياً.
مش باج: المقام هو **متوسط طول الجذع للكليب**، ولما الكاميرا تزوّم (torsoCV 0.575) المتوسط
ده مالوش معنى. بوابة `torso_len_cv > 0.10` بتعلّم على الكليب ده وعلى `videoplayback (1)`.

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
| ⬜ | بوابة ١ — رسمة `lumbar_gap` صح مقابل مقوّس · **محتاج تصوير** |
| ⬜ | حساسية C7 على الـ ٤٠ عدة الغلط · **محتاج تصوير** |
| ⬜ | معايرة `view_score` من فيديوهات ٠/٣٠/٤٥ · **محتاج تصوير** |
| ⬜ | السلايدز |
| ⬜ | تراجعي `person_id` في `data/clips.csv` (مبدئي دلوقتي، id لكل dedup group) |
| ✅ | `dataset/{build,schema,normative}.py` · `qc/report.py` · `modeling/features.py` · `cli.py` |
| ✅ | `ingest/{download,clipper}.py` · `scripts/run_triage.py` · `Makefile` · `tests/test_schema.py` |
| ✅ | `LIMITATIONS.md` · `MODEL_CARD.md` · `DATASHEET.md` |

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
