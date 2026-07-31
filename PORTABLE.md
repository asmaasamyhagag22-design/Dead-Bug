# تشغيل المشروع على جهاز تاني

> اقري ده لو هتناقشي من لابتوب غير اللي شغّالة عليه.

---

## ⚠️ حاجة واحدة **مش** على الدرايف — وعمرها ما تقدر تكون

**البيئة (`venv`).** حجمها **٦٠٣ ميجا** وعايشة على `C:\Users\asmaa\venvs\deadbug`.

**ليه مش على الدرايف:**

1. **جرّبناها وفشلت.** تثبيت الـ venv جوه مجلد الدرايف بيقع بـ
   `os error 1450: Insufficient system resources` في نص التثبيت، وبيسيب بيئة
   `import numpy` فيها بينجح و `numpy.array` **مش موجودة**. حصل فعلاً على الجهاز ده.
2. **حتى لو اتنسخت، مش هتشتغل.** الـ venv بيكتب **مسارات مطلقة** جوه `pyvenv.cfg` وجوه كل
   ملف `.exe` في `Scripts/`. الـ venv اللي كان في الريبو أصلاً كان بيشاور على
   `C:\ProgramData\anaconda3` و `C:\Users\Admin\...` — مسارات جهاز قديم، فماكانش بيشتغل.

**الحل: تتبني من جديد على الجهاز الجديد. أمر واحد.**

---

## الخطوات على الجهاز الجديد

### ١. افتحي مجلد المشروع من الدرايف

```
...\dead bug\deadbug-aqa\
```

### ٢. شغّلي ملف واحد

كليك يمين على **`setup.ps1`** ← **Run with PowerShell**

أو من الترمينال:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

بيعمل إيه: يثبّت `uv` لو مش موجود · يبني البيئتين **على القرص المحلي** · يشغّل الاختبارات.
**الوقت: ٥–١٠ دقايق** (حسب النت).

### ٣. اتأكدي إن آخر سطر بيقول

```
All tests passed. Ready.
```

**لو الاختبارات وقعت، متعرضيش.** خصوصاً لو `test_normalize` أو `test_skeleton` وقعوا —
دي **بوابة ٠** وهي بتحجب كل حاجة: لو التطبيع مكسور، كل رقم بعده بلا معنى.

### ٤. شغّلي التطبيق

```powershell
& "$env:USERPROFILE\venvs\deadbug\Scripts\python.exe" scripts\run_app.py
```

وافتحي **http://localhost:8000**

---

## ✅ اللي موجود على الدرايف فعلاً (اتأكدت منه)

| | الحجم | |
|---|---|---|
| `src/` `scripts/` `tests/` `configs/` | ٢٠٠ ملف | الكود كله |
| `pose_landmarker_heavy.task` | **٣٠ م.ب** | موديل MediaPipe — **مش في git**، الدرايف هو النسخة الوحيدة |
| `data/clips/` | **٥٨ م.ب** | ١٠ كليبات — **مش في git، ومفيش نسخة تانية منها في الدنيا** |
| `data/rehabpile/` | **٤.٣ ج.ب** | داتا المسار A — بتتحمّل لوحدها لو ضاعت، بس بتاخد وقت |
| `data/interim/` | ١٥ م.ب | كاش استخراج الـ pose — **بيوفر ٢٥ دقيقة** لو موجود |
| `data/processed/` | ٢١ ك.ب | جدول العدات + النطاق المرجعي |
| `reports/` | ١٢٩ م.ب | نتايج المسار A · triage · QC · فيديوهات المعاينة |
| كل الوثائق | | `VIVA.md` · `LIMITATIONS.md` · `MODEL_CARD.md` · `DATASHEET.md` · `FILMING.md` · `HANDOVER.md` · `PROCESS.md` |

**كل ده موجود.** الفاضي الوحيد هو البيئة، وخطوة ٢ بتبنيها.

---

## 🔴 خطر لازم تتأكدي منه بنفسك

المشروع عايش في:

```
G:\Other computers\My Computer\Desktop\projects\CV\dead bug
```

`Other computers` ده **قسم النسخ الاحتياطي** في Google Drive — بتاع أجهزة تانية. القسم ده
في أغلب الإعدادات **اتجاه واحد** (من الجهاز للسحابة)، ومش مضمون إن التعديلات اللي بتتعمل
منه **ترجع تتزامن**.

**اتأكدي كده، دلوقتي، مش بكرة:**

1. افتحي **drive.google.com** من المتصفح
2. دوّري على `VIVA.md` أو `setup.ps1` (اتعملوا النهاردة)
3. **لو مالقتيهمش → التعديلات مش بتتزامن**

### لو مش بتتزامن، عندك خيارين

**الأول (الأضمن): ادفعي على GitHub**

الريبو موصول بـ `https://github.com/asmaasamyhagag22-design/Dead-Bug.git`، وعندك
**١٥ commit محلي لسه مادفعوش** — كل شغل النهاردة.

```bash
git push origin main
```

وبعدها من أي جهاز: `git clone` + `setup.ps1`.
⚠️ بس **الكليبات والموديل مش في git** (متجاهلين عن قصد، حجم كبير) — دول لازم تنقليهم بإيدك.

**التاني: انسخي المجلد كله على فلاشة**

٤.٦ جيجا. أو من غير `data/rehabpile/` يبقى **٣٠٠ ميجا بس** — وده كفاية تماماً للمناقشة،
لأن نتايج المسار A **متخزّنة أصلاً** في `reports/masar_a_results.md` ومش محتاجة إعادة تشغيل.

---

## أقل نسخة تنفع للمناقشة (٣٠٠ ميجا)

لو الوقت أو المساحة ضيقة، ده اللي **لازم** ينتقل:

```
deadbug-aqa/
├── src/  scripts/  tests/  configs/     ← الكود
├── pose_landmarker_heavy.task           ← ٣٠ م.ب · التطبيق مايشتغلش من غيره
├── data/clips/                          ← ٥٨ م.ب · مفيش نسخة تانية
├── data/interim/                        ← ١٥ م.ب · بيوفر ٢٥ دقيقة استخراج
├── data/processed/                      ← جدول العدات
├── reports/masar_a_results.md + .csv    ← أرقام المسار A
├── reports/qc.csv + qc.html             ← نتايج الجودة
├── setup.ps1  requirements*.txt         ← التجهيز
└── كل ملفات .md                          ← الوثائق والمذاكرة
```

اللي **ينفع** يتساب: `data/rehabpile/` (٤.٣ ج.ب) · `reports/preview/` (فيديوهات معاينة) ·
`data/webapp/` (نتايج مؤقتة).

---

## لو حاجة وقعت يوم المناقشة

| المشكلة | الحل |
|---|---|
| `ModuleNotFoundError` | البيئة مش متبنية → شغّلي `setup.ps1` |
| `numpy has no attribute array` | تثبيت ناقص → `setup.ps1` تاني (بتمسح وتبني من جديد) |
| `pose model not found` | `pose_landmarker_heavy.task` مش منقول → انسخيه، أو حمّليه من اللينك في `HANDOVER.md §1` |
| `cannot open source 0` | مفيش كاميرا → استخدمي تبويب **Upload** أو `--source "data/clips/videoplayback (1).mp4"` |
| الكاميرا مرفوضة في المتصفح | لازم `localhost` أو HTTPS → استخدمي تبويب **Upload** |
| التطبيق بيطلّع **٠ عدات** | **راجعي الإعدادات**: calibration = **3** و baseline = **3**. رقم أعلى بياكل الكليب |
| الاختبارات بطيئة أوي | البيئة على الدرايف → لازم تكون على `C:` |

---

## آخر حاجة: بروفة كاملة قبل المناقشة

على الجهاز اللي هتناقشي منه، وبعد `setup.ps1`:

```powershell
$py = "$env:USERPROFILE\venvs\deadbug\Scripts\python.exe"
& $py -m pytest tests/ -q                    # لازم: 154 passed
& $py scripts\run_app.py                     # افتحي localhost:8000
```

وفي المتصفح: ارفعي `data\clips\videoplayback (1).mp4` من تبويب **Upload** بإعدادات **3 / 3**،
واستني النتيجة تظهر. **لو ده اشتغل، إنتِ جاهزة.**
