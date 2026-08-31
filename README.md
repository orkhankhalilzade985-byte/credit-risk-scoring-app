# 💳 Kredit Risk Skorlaşdırma Sistemi (Credit Risk Scoring System)

Bank müştərilərinin növbəti ay kredit ödənişini buraxma (default) ehtimalını proqnozlaşdıran, izah oluna bilən (explainable) ML sistemi. Layihə tam bir data science pipeline-ı əhatə edir: data təmizləmə → feature engineering → model müqayisəsi → hyperparameter tuning → SHAP izahatlılıq → biznes-dəyəri optimallaşdırması → Streamlit ilə deployment.

🔗 **Canlı demo:** [buraya Streamlit Cloud linkini əlavə et]

## 📊 Dataset

[UCI Credit Card Default Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — 30,000 Tayvan kredit kartı müştərisi, 23 xüsusiyyət (kredit limiti, demoqrafiya, son 6 ayın ödəniş statusu, hesab qalığı, edilən ödənişlər).

## 🎯 Metodologiya

### 1. Data Təmizləmə
- `EDUCATION` və `MARRIAGE` sütunlarında sənədləşdirilməmiş kateqoriyalar (0, 5, 6) "digər" kimi qruplaşdırıldı
- Duplicate və missing value yoxlanıldı (heç biri tapılmadı)

### 2. Feature Engineering
Xam sütunlardan (xüsusilə 6 ayrı `BILL_AMT` sütunundan, aralarında 0.80–0.95 korrelyasiya var idi) yeni, daha informativ göstəricilər yaradıldı:
- `NUM_MONTHS_LATE`, `MAX_DELAY`, `AVG_DELAY_RECENT3` — gecikmə tarixçəsinin icmalı
- `UTILIZATION_RATE` — kredit istifadə nisbəti (borc/limit)
- `PAY_TO_BILL_RATIO`, `BILL_TREND` — ödəniş davranışı göstəriciləri

Nəticə: yaradılan feature-lar (`NUM_MONTHS_LATE`, `MAX_DELAY`) SHAP təhlilində modelin **ən güclü prediktorları** oldu, xam `BILL_AMT` sütunlarını üstələyərək.

### 3. Model Seçimi
Üç model müqayisə edildi:

| Model | ROC-AUC | Recall (Default) | Precision (Default) |
|---|---|---|---|
| Logistic Regression | 0.744 | 0.60 | 0.44 |
| **Random Forest (tuned)** | **0.776** | 0.56–0.80* | 0.34–0.53* |
| PyTorch MLP (custom) | 0.770 | 0.59 | 0.46 |

*Threshold-dan asılı olaraq dəyişir (aşağıya bax)

Random Forest seçildi — həm ən yaxşı ROC-AUC, həm sürətli təlim, həm izah oluna bilən struktur. PyTorch MLP də sınandı və nəticələr müqayisə edildi; kiçik, tabular dataset üzərində tree-based modellərin neyron şəbəkələri üstələməsi gözlənilən və sənədləşdirilmiş bir nəticədir.

### 4. Etibarlılıq Yoxlanışı
- 5-fold Cross-Validation: ROC-AUC = 0.784 ± 0.007 (yüksək sabitlik)
- RandomizedSearchCV ilə hyperparameter tuning (30 kombinasiya × 5 fold)

### 5. İzahatlılıq (SHAP)
Hər proqnoz üçün fərdi izahat generasiya olunur — hansı amillərin riski artırıb/azaldığı göstərilir. Bu, bank tənzimləyicisi/auditoruna qərarın "qara qutu" olmadığını sübut etmək üçün vacibdir.

### 6. Biznes-Dəyəri Optimallaşdırması
Default threshold (0.5) əvəzinə, xərc-əsaslı analiz ilə optimal threshold (0.30) seçildi:
- Yanlış mənfi (buraxılan default) xərci >> yanlış müsbət (lazımsız yoxlama) xərci
- Threshold=0.30-da təxmini ümumi xərc, default (0.50) threshold-a nisbətən ~2 dəfə aşağıdır

## 🖥️ Streamlit Tətbiqi

Üç funksiyalı interaktiv panel:
- **Müştəri Sorğusu** — mövcud müştərini ID ilə axtar, sistem "core banking"dən məlumatı avtomatik gətirmiş kimi simulyasiya edir
- **Dashboard** — risk seqmentlərinə görə ümumi mənzərə
- **Yeni Müştəri** — yeni müraciət üçün əl ilə məlumat daxil et, model kredit qərarını (ver/vermə) versin

> ℹ️ Qeyd: demo sadələşdirilmişdir. Real production mühitində müştəri göstəriciləri core banking sistemindən API vasitəsilə avtomatik alınar, model performansı davamlı monitorinq olunar və planlaşdırılmış dövrlərlə (aylıq–illik, institutun ehtiyacına görə) yenidən öyrədilər.

## 🛠️ Texniki Stack

Python · pandas · scikit-learn · PyTorch · SHAP · Streamlit · matplotlib

## 📁 Repo Strukturu

```
├── app.py                        # Streamlit tətbiqi
├── retrain.py                    # Model təlimi pipeline-ı (sıfırdan işə salına bilər)
├── requirements.txt              # Python asılılıqları
├── best_rf_model.pkl             # Öyrədilmiş model
├── feature_columns.pkl           # Feature sırası (inference üçün)
├── shap_explainer.pkl            # SHAP izahatlılıq obyekti
└── credit_risk_predictions.csv   # Test set nəticələri (dashboard üçün)
```

## 🚀 Lokal İşə Salma

```bash
pip install -r requirements.txt
streamlit run app.py
```
