"""
Retrain script - creditrisk environment üçün
-----------------------------------------------
Bu skript bütün pipeline-ı (data yükləmə -> cleaning -> feature engineering ->
model təlimi -> export) sıfırdan işlədir və bütün .pkl/.csv fayllarını
CƏRİ MÜHİTİN (creditrisk) sklearn/shap versiyalarına uyğun yenidən yaradır.

İşlətmək: python retrain.py
(creditrisk mühiti aktiv olmalıdır: conda activate creditrisk)
"""

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

pd.set_option('display.max_columns', None)

# >>> BURANI ÖZ CSV YOLUNLA DƏYİŞ (əgər fərqlidirsə) <<<
CSV_PATH = r"C:\Users\Hp\Downloads\Butun Folderler\Elave datalar\Credit card\UCI_Credit_Card.csv"

print("1/6 - Data yüklənir...")
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={'default.payment.next.month': 'DEFAULT', 'PAY_0': 'PAY_1'})
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

print("2/6 - Feature engineering...")
df_fe = df.copy()
pay_cols = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

df_fe['NUM_MONTHS_LATE'] = (df[pay_cols] > 0).sum(axis=1)
df_fe['MAX_DELAY'] = df[pay_cols].max(axis=1)
df_fe['AVG_DELAY_RECENT3'] = df[['PAY_1','PAY_2','PAY_3']].mean(axis=1)
df_fe['AVG_BILL'] = df[bill_cols].mean(axis=1)
df_fe['UTILIZATION_RATE'] = (df_fe['AVG_BILL'] / df_fe['LIMIT_BAL']).clip(upper=2)
df_fe['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)
df_fe['PAY_TO_BILL_RATIO'] = (df_fe['AVG_PAY_AMT'] / df_fe['AVG_BILL'].replace(0, np.nan)).fillna(0).clip(upper=5)
df_fe['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
df_fe['AGE_GROUP'] = pd.cut(df['AGE'], bins=[20,30,40,50,60,80],
                              labels=['21-30','31-40','41-50','51-60','60+'])

print("3/6 - Train/test split...")
drop_cols = ['ID', 'DEFAULT', 'AGE_GROUP'] + bill_cols
feature_cols = [c for c in df_fe.columns if c not in drop_cols]

X = df_fe[feature_cols].copy()
y = df_fe['DEFAULT'].copy()
X = pd.concat([X, pd.get_dummies(df_fe['AGE_GROUP'], prefix='AGE', drop_first=True)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("4/6 - Model təlimi (tuned parametrlərlə)...")
best_rf = RandomForestClassifier(
    n_estimators=400, max_depth=12, min_samples_split=5,
    min_samples_leaf=4, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
best_rf.fit(X_train, y_train)

best_rf_proba = best_rf.predict_proba(X_test)[:, 1]
best_rf_pred = best_rf.predict(X_test)
print(classification_report(y_test, best_rf_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, best_rf_proba), 4))

print("5/6 - SHAP explainer və fayllar saxlanılır...")
explainer = shap.TreeExplainer(best_rf)

joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(list(X_train.columns), 'feature_columns.pkl')
joblib.dump(explainer, 'shap_explainer.pkl')

print("6/6 - Dashboard üçün export...")
FINAL_THRESHOLD = 0.30
export_df = df_fe.loc[X_test.index].copy()
export_df['PREDICTED_PROBA'] = best_rf_proba
export_df['PREDICTED_DEFAULT'] = (best_rf_proba >= FINAL_THRESHOLD).astype(int)
export_df['ACTUAL_DEFAULT'] = y_test.values
export_df['RISK_SEGMENT'] = pd.cut(export_df['PREDICTED_PROBA'],
                                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=['Çox Aşağı', 'Aşağı', 'Orta', 'Yüksək', 'Çox Yüksək'])

def result_type(row):
    if row['ACTUAL_DEFAULT']==1 and row['PREDICTED_DEFAULT']==1: return 'True Positive'
    if row['ACTUAL_DEFAULT']==0 and row['PREDICTED_DEFAULT']==0: return 'True Negative'
    if row['ACTUAL_DEFAULT']==0 and row['PREDICTED_DEFAULT']==1: return 'False Positive'
    return 'False Negative'

export_df['RESULT_TYPE'] = export_df.apply(result_type, axis=1)
export_df.to_csv('credit_risk_predictions.csv', index=False)

print("\n✅ TAMAMLANDI! Bütün fayllar bu mühitin sklearn/shap versiyasına uyğun yenidən yaradıldı:")
print("   - best_rf_model.pkl")
print("   - feature_columns.pkl")
print("   - shap_explainer.pkl")
print("   - credit_risk_predictions.csv")
print("\nİndi 'streamlit run app.py' işlədə bilərsən.")