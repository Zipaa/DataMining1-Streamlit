"""
Dashboard Streamlit — Prediksi Kesehatan Mental Remaja
======================================================
Data‑mining dashboard yang mencakup:
  • Gambaran Umum Dataset
  • Visualisasi EDA (distribusi, korelasi)
  • Segmentasi K‑Means (PCA 2‑D)
  • Klasifikasi: Logistic Regression vs Naïve Bayes
  • Prediksi Interaktif (user memasukkan input → prediksi label depresi)

Jalankan:
  streamlit run dashboard_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# ──────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TARGET = "depression_label"
DATA_PATH = "Teen_Mental_Health_Dataset.csv"
MODEL_DIR = "models"

# ──────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Kesehatan Mental Remaja",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# HELPER: load & preprocess data
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load dan preprocessing dataset sesuai dengan notebook BAB 4."""
    df = pd.read_csv(DATA_PATH)

    # --- Drop duplicate ---
    df = df.drop_duplicates().reset_index(drop=True)

    return df


@st.cache_data
def preprocess(df: pd.DataFrame):
    """Label‑encode kolom kategorikal dan scale fitur numerik."""
    df_enc = df.copy()
    cat_cols = ["gender", "platform_usage", "social_interaction_level"]
    mappings = {}
    label_encoders = {}

    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c])
        mappings[c] = dict(zip(le.classes_, le.transform(le.classes_)))
        label_encoders[c] = le

    # Pisah X dan y
    X = df_enc.drop(columns=[TARGET])
    y = df_enc[TARGET]

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_enc, X, y, X_scaled, scaler, mappings, label_encoders


@st.cache_resource
def load_models():
    """Coba muat artefak model yang disimpan dari notebook."""
    models = {}
    try:
        models["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        models["kmeans"] = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
        models["logreg"] = joblib.load(os.path.join(MODEL_DIR, "logreg.pkl"))
        models["naive_bayes"] = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
        models["comparison"] = pd.read_csv(
            os.path.join(MODEL_DIR, "model_comparison.csv"), index_col=0
        )
    except Exception:
        models = None
    return models


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
df_raw = load_data()
df_enc, X, y, X_scaled, scaler, mappings, label_encoders = preprocess(df_raw)

# Coba muat artefak
saved_models = load_models()

# ──────────────────────────────────────────────
# SIDEBAR — Hamburger + Boxed menu items
# ──────────────────────────────────────────────

# Custom CSS: boxed radio items & clean sidebar
st.markdown(
    """
    <style>
    /* ── Sidebar header ── */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.35rem;
        letter-spacing: 0.03em;
    }

    /* ── Boxed radio items ── */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: flex;
        align-items: center;
        border: 1px solid rgba(150, 150, 150, 0.35);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: background 0.2s, border-color 0.2s;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(100, 160, 255, 0.12);
        border-color: rgba(100, 160, 255, 0.5);
    }
    /* Active / checked item */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(100, 160, 255, 0.18);
        border-color: rgba(80, 140, 255, 0.7);
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("📊 Menu Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Gambaran Umum",
        "Visualisasi EDA",
        "Segmentasi K-Means",
        "Klasifikasi",
        "Prediksi Interaktif",
    ],
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════
# HALAMAN 1 — Gambaran Umum Dataset
# ══════════════════════════════════════════════
if page == "Gambaran Umum":
    st.title("Gambaran Umum Dataset")
    st.markdown(
        """
        Dataset yang digunakan: **Teen Mental Health Dataset** (Kaggle).
        Variabel target: `depression_label` (0 = Tidak Depresi, 1 = Depresi).
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df_raw.shape[0])
    col2.metric("Jumlah Kolom", df_raw.shape[1])
    col3.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    st.subheader("5 Data Pertama")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.subheader("Informasi Dataset")
    info_df = pd.DataFrame(
        {
            "Kolom": df_raw.columns,
            "Tipe Data": [str(t) for t in df_raw.dtypes],
            "Non-Null": df_raw.notnull().sum().values,
            "Jumlah Unik": [df_raw[c].nunique() for c in df_raw.columns],
        }
    )
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    desc = df_raw.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    st.dataframe(desc, use_container_width=True)

# ══════════════════════════════════════════════
# HALAMAN 2 — Visualisasi EDA
# ══════════════════════════════════════════════
elif page == "Visualisasi EDA":
    st.title("Visualisasi Exploratory Data Analysis")

    # --- Distribusi Target ---
    st.subheader("Distribusi Status Depresi")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    counts = df_raw[TARGET].value_counts().sort_index()
    bars = ax.bar(
        ["Tidak Depresi (0)", "Depresi (1)"],
        counts.values,
        color=["#4C72B0", "#C44E52"],
        edgecolor="black",
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Depression Label")
    st.pyplot(fig)

    # --- Distribusi numerik ---
    st.subheader("Distribusi Fitur Numerik")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Pilih fitur:", num_cols)

    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    sns.histplot(df_raw[selected_col], kde=True, ax=ax2, color="#4C72B0")
    ax2.set_title(f"Distribusi {selected_col}")
    st.pyplot(fig2)

    # --- Heatmap Korelasi ---
    st.subheader("Heatmap Korelasi (setelah encoding)")
    fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
    corr = df_enc.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax3,
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

# ══════════════════════════════════════════════
# HALAMAN 3 — Segmentasi K‑Means
# ══════════════════════════════════════════════
elif page == "Segmentasi K-Means":
    st.title("Segmentasi K-Means")

    # --- Elbow ---
    st.subheader("Elbow Method")
    inertias = []
    K_range = range(1, 7)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(5, 3))
    ax_elbow.plot(list(K_range), inertias, "o-", color="#4C72B0")
    ax_elbow.axvline(3, color="#C44E52", ls="--", label="K = 3 (titik siku)")
    ax_elbow.set_xlabel("Jumlah Cluster (K)")
    ax_elbow.set_ylabel("Inertia (WCSS)")
    ax_elbow.legend()
    ax_elbow.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    # --- Silhouette ---
    st.subheader("Silhouette Score")
    sil_scores = []
    for k in range(2, 7):
        km_temp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(
            X_scaled
        )
        sil_scores.append(
            {"K": k, "Silhouette Score": round(silhouette_score(X_scaled, km_temp.labels_), 4)}
        )
    sil_df = pd.DataFrame(sil_scores)
    st.dataframe(sil_df, use_container_width=True)

    # --- KMeans K=3 ---
    st.subheader("Hasil Clustering (K = 3)")
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    df_enc_cluster = df_enc.copy()
    df_enc_cluster["Cluster"] = kmeans.labels_

    # PCA 2D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig_pca, ax_pca = plt.subplots(figsize=(5.5, 4))
    scatter = ax_pca.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans.labels_,
        cmap="Set1",
        alpha=0.6,
        edgecolors="k",
        s=30,
    )
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_pca.set_title("Visualisasi K-Means (PCA 2D)")
    plt.colorbar(scatter, ax=ax_pca, label="Cluster")
    st.pyplot(fig_pca)

    # Profil Cluster
    st.subheader("Profil Rata‑Rata per Cluster")
    cluster_profile = df_enc_cluster.groupby("Cluster").mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)

# ══════════════════════════════════════════════
# HALAMAN 4 — Klasifikasi
# ══════════════════════════════════════════════
elif page == "Klasifikasi":
    st.title("Klasifikasi: Logistic Regression vs Naïve Bayes")

    # --- Split (copy agar tidak read-only dari cache) ---
    _X = np.array(X_scaled, copy=True)
    _y = np.array(y, copy=True)
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=RANDOM_STATE, stratify=_y
    )

    # --- Logistic Regression ---
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    pred_lr = logreg.predict(X_test)
    proba_lr = logreg.predict_proba(X_test)[:, 1]

    # --- Naïve Bayes ---
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)
    proba_nb = nb.predict_proba(X_test)[:, 1]

    # Helper metric
    def metric_row(name, y_true, y_pred, y_proba):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        return {
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_true, y_proba), 4),
        }

    eval_lr = metric_row("Logistic Regression", y_test, pred_lr, proba_lr)
    eval_nb = metric_row("Naïve Bayes", y_test, pred_nb, proba_nb)
    comparison = pd.DataFrame([eval_lr, eval_nb]).set_index("Model")

    st.subheader("Tabel Perbandingan Model")
    st.dataframe(comparison, use_container_width=True)

    best = comparison.sort_values(["F1-Score", "ROC-AUC"], ascending=False).index[0]
    st.success(
        f"**Model terbaik:** {best} "
        f"(F1-Score = {comparison.loc[best, 'F1-Score']}, "
        f"ROC-AUC = {comparison.loc[best, 'ROC-AUC']})"
    )

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    col_lr, col_nb = st.columns(2)

    with col_lr:
        st.markdown("**Logistic Regression**")
        fig_cm1, ax_cm1 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test, pred_lr, ax=ax_cm1, cmap="Blues")
        ax_cm1.set_title("Logistic Regression")
        st.pyplot(fig_cm1)

    with col_nb:
        st.markdown("**Naïve Bayes**")
        fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test, pred_nb, ax=ax_cm2, cmap="Oranges")
        ax_cm2.set_title("Naïve Bayes")
        st.pyplot(fig_cm2)

    # --- ROC Curve ---
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    for name, proba in [("Logistic Regression", proba_lr), ("Naïve Bayes", proba_nb)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # --- Classification Report ---
    st.subheader("Classification Report")
    tab1, tab2 = st.tabs(["Logistic Regression", "Naïve Bayes"])
    with tab1:
        st.text(classification_report(y_test, pred_lr, zero_division=0))
    with tab2:
        st.text(classification_report(y_test, pred_nb, zero_division=0))

# ══════════════════════════════════════════════
# HALAMAN 5 — Prediksi Interaktif
# ══════════════════════════════════════════════
elif page == "Prediksi Interaktif":

    # ── Hasil Prediksi di ATAS (jika sudah ada) ──
    if "pred_result" in st.session_state:
        r = st.session_state["pred_result"]
        st.title("Hasil Prediksi")

        res1, res2 = st.columns(2)
        with res1:
            st.markdown("### Logistic Regression")
            st.info(f"**Prediksi:** {r['label_lr']}")
            st.write(f"Probabilitas Tidak Depresi: `{r['prob_lr'][0]:.4f}`")
            st.write(f"Probabilitas Depresi: `{r['prob_lr'][1]:.4f}`")
        with res2:
            st.markdown("### Naïve Bayes")
            st.info(f"**Prediksi:** {r['label_nb']}")
            st.write(f"Probabilitas Tidak Depresi: `{r['prob_nb'][0]:.4f}`")
            st.write(f"Probabilitas Depresi: `{r['prob_nb'][1]:.4f}`")

        # Ringkasan input
        st.subheader("Ringkasan Input")
        st.dataframe(r["input_summary"], use_container_width=True)
        st.markdown("---")

    # ── Form Input di BAWAH ──
    st.title("Prediksi Depresi — Input Manual")
    st.markdown(
        "Masukkan data remaja di bawah ini, lalu klik **Prediksi** untuk melihat "
        "hasil klasifikasi menggunakan model Logistic Regression & Naïve Bayes."
    )

    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=13, max_value=19, value=16)
            gender = st.selectbox("Gender", ["female", "male"])
            daily_sm = st.slider("Daily Social Media Hours", 1.0, 8.0, 4.5, 0.1)
            platform = st.selectbox("Platform Usage", ["Both", "Instagram", "TikTok"])
        with c2:
            sleep = st.slider("Sleep Hours", 4.0, 9.0, 6.5, 0.1)
            screen = st.slider("Screen Time Before Sleep", 0.5, 3.0, 1.5, 0.1)
            academic = st.slider("Academic Performance", 2.0, 4.0, 3.0, 0.01)
            physical = st.slider("Physical Activity", 0.0, 2.0, 1.0, 0.1)
        with c3:
            social = st.selectbox("Social Interaction Level", ["high", "low", "medium"])
            stress = st.slider("Stress Level", 1, 10, 5)
            anxiety = st.slider("Anxiety Level", 1, 10, 5)
            addiction = st.slider("Addiction Level", 1, 10, 5)

        submitted = st.form_submit_button("Prediksi", use_container_width=True)

    if submitted:
        # Encode input
        gender_enc = label_encoders["gender"].transform([gender])[0]
        platform_enc = label_encoders["platform_usage"].transform([platform])[0]
        social_enc = label_encoders["social_interaction_level"].transform([social])[0]

        input_data = np.array(
            [[age, gender_enc, daily_sm, platform_enc, sleep, screen,
              academic, physical, social_enc, stress, anxiety, addiction]]
        )
        input_scaled = scaler.transform(input_data)

        # Train model langsung (copy agar tidak read-only dari cache)
        _X = np.array(X_scaled, copy=True)
        _y = np.array(y, copy=True)
        X_train, X_test, y_train, y_test = train_test_split(
            _X, _y, test_size=0.2, random_state=RANDOM_STATE, stratify=_y
        )
        logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        logreg.fit(X_train, y_train)
        nb = GaussianNB()
        nb.fit(X_train, y_train)

        pred_lr = logreg.predict(input_scaled)[0]
        prob_lr = logreg.predict_proba(input_scaled)[0]
        pred_nb = nb.predict(input_scaled)[0]
        prob_nb = nb.predict_proba(input_scaled)[0]

        label_map = {0: "Tidak Depresi", 1: "Depresi"}

        input_summary = pd.DataFrame(
            {
                "Fitur": [
                    "Age", "Gender", "Daily Social Media Hours",
                    "Platform Usage", "Sleep Hours", "Screen Time Before Sleep",
                    "Academic Performance", "Physical Activity",
                    "Social Interaction Level", "Stress Level",
                    "Anxiety Level", "Addiction Level",
                ],
                "Nilai": [
                    age, gender, daily_sm, platform, sleep, screen,
                    academic, physical, social, stress, anxiety, addiction,
                ],
            }
        )

        # Simpan ke session_state lalu rerun agar muncul di atas
        st.session_state["pred_result"] = {
            "label_lr": label_map[pred_lr],
            "prob_lr": prob_lr,
            "label_nb": label_map[pred_nb],
            "prob_nb": prob_nb,
            "input_summary": input_summary,
        }
        st.rerun()

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Data Mining SI4809 Analisis Kesehatan Mental Remaja")
