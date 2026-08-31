"""
Credit Risk Scoring App - v2
------------------------------
Bu versiya real bank prosesini simulyasiya edir: işçi əl ilə 24 sahə
doldurmaq əvəzinə, sadəcə Müştəri ID daxil edir. Sistem "core banking"
məlumat bazasından (bu demoda credit_risk_predictions.csv) müştərinin
göstəricilərini avtomatik çəkir, sonra model risk skorunu hesablayır.

İşlətmək üçün:
    1. Bu faylı (app.py) aşağıdakı fayllarla EYNİ qovluğa qoy:
       - best_rf_model.pkl
       - feature_columns.pkl
       - shap_explainer.pkl
       - credit_risk_predictions.csv
    2. Terminalda: streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kredit Risk Skoru v2", page_icon="🏦", layout="wide")

# ============================================
# FAYLLARI YÜKLƏ
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_rf_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    return model, feature_columns, explainer

@st.cache_data
def load_core_banking_db():
    # Bu CSV "core banking sistemi" rolunu oynayır - real mühitdə bu, bir API/database sorğusu olardı
    return pd.read_csv("credit_risk_predictions.csv")

try:
    model, feature_columns, explainer = load_artifacts()
    db = load_core_banking_db()
    FILES_OK = True
except FileNotFoundError as e:
    FILES_OK = False
    missing_file = str(e)

FINAL_THRESHOLD = 0.30

st.title("🏦 Kredit Risk Skoru — Bank İşçisi Paneli")
st.caption("Müştəri ID daxil et, sistem core banking-dən məlumatı avtomatik çəkib risk skorunu hesablasın")

if not FILES_OK:
    st.error(f"Lazımi fayl tapılmadı: {missing_file}\n\n"
             f"app.py-ı best_rf_model.pkl, feature_columns.pkl, shap_explainer.pkl və "
             f"credit_risk_predictions.csv ilə EYNİ qovluğa qoy.")
    st.stop()

# ============================================
# FEATURE ENGINEERING (training ilə eyni məntiq)
# ============================================
def build_features_from_row(row: pd.Series) -> pd.DataFrame:
    pay_vals = [row[f"PAY_{i}"] for i in range(1, 7)]
    bill_vals = [row[f"BILL_AMT{i}"] for i in range(1, 7)]
    pay_amt_vals = [row[f"PAY_AMT{i}"] for i in range(1, 7)]

    feat = {
        "LIMIT_BAL": row["LIMIT_BAL"], "SEX": row["SEX"],
        "EDUCATION": row["EDUCATION"], "MARRIAGE": row["MARRIAGE"], "AGE": row["AGE"],
    }
    for i in range(1, 7):
        feat[f"PAY_{i}"] = row[f"PAY_{i}"]
    for i in range(1, 7):
        feat[f"PAY_AMT{i}"] = row[f"PAY_AMT{i}"]

    feat["NUM_MONTHS_LATE"] = sum(1 for p in pay_vals if p > 0)
    feat["MAX_DELAY"] = max(pay_vals)
    feat["AVG_DELAY_RECENT3"] = np.mean(pay_vals[:3])
    avg_bill = np.mean(bill_vals)
    feat["AVG_BILL"] = avg_bill
    feat["UTILIZATION_RATE"] = min(avg_bill / row["LIMIT_BAL"], 2) if row["LIMIT_BAL"] > 0 else 0
    avg_pay_amt = np.mean(pay_amt_vals)
    feat["AVG_PAY_AMT"] = avg_pay_amt
    feat["PAY_TO_BILL_RATIO"] = min((avg_pay_amt / avg_bill) if avg_bill != 0 else 0, 5)
    feat["BILL_TREND"] = bill_vals[0] - bill_vals[5]

    age = row["AGE"]
    for label, lo, hi in [("31-40", 30, 40), ("41-50", 40, 50), ("51-60", 50, 60), ("60+", 60, 80)]:
        feat[f"AGE_{label}"] = 1 if lo < age <= hi else 0

    df_row = pd.DataFrame([feat])
    return df_row.reindex(columns=feature_columns, fill_value=0)


def build_features_from_inputs(inputs: dict) -> pd.DataFrame:
    """Əl ilə daxil edilən yeni müştəri inputundan feature seti qurur (build_features_from_row ilə eyni məntiq)."""
    fake_row = pd.Series(inputs)
    return build_features_from_row(fake_row)

# ============================================
# TAB-LAR
# ============================================
tab1, tab2, tab3 = st.tabs(["🔍 Müştəri Sorğusu", "📊 Dashboard", "➕ Yeni Müştəri"])

with tab1:
    st.subheader("Müştəri axtarışı")

    col_search, col_random = st.columns([3, 1])
    with col_search:
        customer_id = st.number_input("Müştəri ID daxil et", min_value=int(db["ID"].min()),
                                       max_value=int(db["ID"].max()), value=int(db["ID"].iloc[0]), step=1)
    with col_random:
        st.write("")
        st.write("")
        if st.button("🎲 Təsadüfi müştəri"):
            customer_id = int(db["ID"].sample(1).values[0])
            st.session_state["random_id"] = customer_id

    if "random_id" in st.session_state:
        customer_id = st.session_state["random_id"]

    if st.button("🔎 Axtar və Qiymətləndir", type="primary"):
        match = db[db["ID"] == customer_id]

        if match.empty:
            st.warning("Bu ID core banking sistemində tapılmadı.")
        else:
            row = match.iloc[0]

            with st.spinner("Core banking sistemindən müştəri məlumatı çəkilir..."):
                time.sleep(0.8)  # real API sorğusunu simulyasiya edir

            st.success("✅ Məlumat uğurla alındı")

            st.markdown("**Sistemdən avtomatik gətirilən məlumatlar:**")
            info_cols = st.columns(5)
            info_cols[0].metric("Kredit limiti", f"{row['LIMIT_BAL']:,.0f}")
            info_cols[1].metric("Yaş", int(row["AGE"]))
            info_cols[2].metric("Son ay status (PAY_1)", int(row["PAY_1"]))
            info_cols[3].metric("Orta hesab qalığı", f"{row['AVG_BILL']:,.0f}")
            info_cols[4].metric("Kredit istifadə nisbəti", f"{row['UTILIZATION_RATE']*100:.0f}%")

            with st.expander("Bütün xam göstəriciləri göstər (PAY, BILL_AMT, PAY_AMT)"):
                raw_cols = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] +
                            [f"PAY_{i}" for i in range(1, 7)] +
                            [f"BILL_AMT{i}" for i in range(1, 7)] +
                            [f"PAY_AMT{i}" for i in range(1, 7)])
                st.dataframe(match[raw_cols])

            X_input = build_features_from_row(row)
            proba = model.predict_proba(X_input)[0][1]
            prediction = "RİSKLİ" if proba >= FINAL_THRESHOLD else "TƏHLÜKƏSİZ"

            st.markdown("---")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Default ehtimalı", f"{proba*100:.1f}%")
                if proba >= FINAL_THRESHOLD:
                    st.error(f"⚠️ {prediction}")
                else:
                    st.success(f"✅ {prediction}")
                st.caption(f"Qərar həddi: {FINAL_THRESHOLD*100:.0f}%")

                if "DEFAULT" in row and not pd.isna(row["DEFAULT"]):
                    actual = "Bəli" if row["DEFAULT"] == 1 else "Xeyr"
                    st.caption(f"Faktiki nəticə (tarixi data): {actual}")

            with col_b:
                st.markdown("**Bu qərara ən çox təsir edən amillər (SHAP):**")
                shap_values = explainer.shap_values(X_input)
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                elif len(np.array(shap_values).shape) == 3:
                    sv = np.array(shap_values)[0, :, 1]
                else:
                    sv = shap_values[0]

                impact_df = pd.DataFrame({
                    "Feature": X_input.columns,
                    "Təsir": sv
                }).sort_values("Təsir", key=abs, ascending=False).head(8)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#d62728" if v > 0 else "#2ca02c" for v in impact_df["Təsir"]]
                ax.barh(impact_df["Feature"], impact_df["Təsir"], color=colors)
                ax.set_xlabel("Riskə təsir (müsbət = artırır, mənfi = azaldır)")
                ax.invert_yaxis()
                st.pyplot(fig)

with tab2:
    st.subheader("Test Datası Üzərində Risk Mənzərəsi")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ümumi müştəri", f"{len(db):,}")
    c2.metric("Faktiki default dərəcəsi", f"{db['ACTUAL_DEFAULT'].mean()*100:.1f}%")
    c3.metric("Model dəqiqliyi (ROC-AUC)", "0.776")
    c4.metric("Seçilmiş threshold", f"{FINAL_THRESHOLD*100:.0f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Risk Seqmentlərinə görə Paylanma**")
        st.bar_chart(db["RISK_SEGMENT"].value_counts().sort_index())
    with col2:
        st.markdown("**Model Nəticə Növləri**")
        st.bar_chart(db["RESULT_TYPE"].value_counts())

    st.markdown("---")
    st.markdown("**Ən Yüksək Riskli 20 Müştəri**")
    top_risk = db.sort_values("PREDICTED_PROBA", ascending=False).head(20)
    st.dataframe(top_risk[["ID", "LIMIT_BAL", "AGE", "PREDICTED_PROBA", "RISK_SEGMENT", "RESULT_TYPE"]])

with tab3:
    st.subheader("Yeni müştəri üçün kredit qərarı")
    st.caption("Bu müştəri hələ sistemdə qeydiyyatda deyil — məlumatları əl ilə daxil et, "
               "model dərhal kredit riskini qiymətləndirsin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_limit_bal = st.number_input("Kredit limiti (LIMIT_BAL)", min_value=10000, max_value=1000000,
                                       value=150000, step=10000, key="n_limit")
        n_age = st.number_input("Yaş", min_value=18, max_value=90, value=35, key="n_age")
        n_sex = st.selectbox("Cins", options=[1, 2], format_func=lambda x: "Kişi" if x == 1 else "Qadın", key="n_sex")
    with col2:
        n_education = st.selectbox("Təhsil", options=[1, 2, 3, 4],
                                    format_func=lambda x: {1: "Ali (graduate)", 2: "Universitet",
                                                            3: "Orta məktəb", 4: "Digər"}[x], key="n_edu")
        n_marriage = st.selectbox("Ailə vəziyyəti", options=[1, 2, 3],
                                   format_func=lambda x: {1: "Evli", 2: "Subay", 3: "Digər"}[x], key="n_marr")
    with col3:
        st.markdown("**Ödəniş tarixçəsi məlum deyilsə, defolt (0) dəyərləri saxla.**")

    st.markdown("**Ödəniş statusu (PAY_1 = ən son ay ... PAY_6 = 6 ay əvvəl)**")
    pc = st.columns(6)
    n_pay = {}
    for i, c in enumerate(pc, start=1):
        with c:
            n_pay[f"PAY_{i}"] = st.number_input(f"PAY_{i}", min_value=-2, max_value=9, value=0, key=f"n_pay_{i}")

    st.markdown("**Hesab qalığı (BILL_AMT1 = ən son ay ... BILL_AMT6 = 6 ay əvvəl)**")
    bc = st.columns(6)
    n_bill = {}
    for i, c in enumerate(bc, start=1):
        with c:
            n_bill[f"BILL_AMT{i}"] = st.number_input(f"BILL_AMT{i}", min_value=0, max_value=2000000,
                                                       value=20000, step=1000, key=f"n_bill_{i}")

    st.markdown("**Edilən ödəniş (PAY_AMT1 = ən son ay ... PAY_AMT6 = 6 ay əvvəl)**")
    pac = st.columns(6)
    n_pay_amt = {}
    for i, c in enumerate(pac, start=1):
        with c:
            n_pay_amt[f"PAY_AMT{i}"] = st.number_input(f"PAY_AMT{i}", min_value=0, max_value=1000000,
                                                         value=2000, step=500, key=f"n_payamt_{i}")

    st.markdown("---")

    if st.button("💳 Kredit Qərarını Hesabla", type="primary"):
        inputs = {"LIMIT_BAL": n_limit_bal, "SEX": n_sex, "EDUCATION": n_education,
                  "MARRIAGE": n_marriage, "AGE": n_age, **n_pay, **n_bill, **n_pay_amt}

        X_new = build_features_from_inputs(inputs)
        proba = model.predict_proba(X_new)[0][1]
        decision = "❌ KREDİT VERİLMƏSİN (Yüksək Risk)" if proba >= FINAL_THRESHOLD else "✅ KREDİT VERİLSİN (Aşağı Risk)"

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Default ehtimalı", f"{proba*100:.1f}%")
            if proba >= FINAL_THRESHOLD:
                st.error(decision)
            else:
                st.success(decision)
            st.caption(f"Qərar həddi: {FINAL_THRESHOLD*100:.0f}% (biznes-dəyəri analizinə əsasən seçilib)")

        with col_b:
            st.markdown("**Bu qərara ən çox təsir edən amillər (SHAP):**")
            shap_values = explainer.shap_values(X_new)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif len(np.array(shap_values).shape) == 3:
                sv = np.array(shap_values)[0, :, 1]
            else:
                sv = shap_values[0]

            impact_df = pd.DataFrame({
                "Feature": X_new.columns, "Təsir": sv
            }).sort_values("Təsir", key=abs, ascending=False).head(8)

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in impact_df["Təsir"]]
            ax.barh(impact_df["Feature"], impact_df["Təsir"], color=colors)
            ax.set_xlabel("Riskə təsir (müsbət = artırır, mənfi = azaldır)")
            ax.invert_yaxis()
            st.pyplot(fig)

        st.session_state["last_new_customer"] = {**inputs, "PREDICTED_PROBA": proba}

    if "last_new_customer" in st.session_state:
        if st.button("💾 Bu nəticəni sistemə (CSV) əlavə et"):
            rec = st.session_state["last_new_customer"]
            new_id = int(db["ID"].max()) + 1
            new_row = {"ID": new_id, **{k: v for k, v in rec.items() if k != "PREDICTED_PROBA"},
                       "PREDICTED_PROBA": rec["PREDICTED_PROBA"],
                       "PREDICTED_DEFAULT": int(rec["PREDICTED_PROBA"] >= FINAL_THRESHOLD),
                       "ACTUAL_DEFAULT": np.nan, "RESULT_TYPE": "Yeni (nəticə hələ bilinmir)"}
            new_row["RISK_SEGMENT"] = pd.cut([rec["PREDICTED_PROBA"]], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                              labels=['Çox Aşağı', 'Aşağı', 'Orta', 'Yüksək', 'Çox Yüksək'])[0]
            db_updated = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
            db_updated.to_csv("credit_risk_predictions.csv", index=False)
            st.success(f"✅ Müştəri #{new_id} sistemə əlavə edildi. Dashboard-u yeniləmək üçün səhifəni yenidən yüklə.")
            del st.session_state["last_new_customer"]

st.markdown("---")
st.caption("ℹ️ Qeyd: Bu demoda müştəri məlumatları yerli CSV-dən simulyasiya olunur. "
           "Real production mühitində bu məlumatlar core banking sistemindən API vasitəsilə "
           "avtomatik alınır (bax: API-driven core banking arxitekturası).")