# save as app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 1️⃣ Outlier Detection Function
# -------------------------
def real_time_outlier_pipeline(df, num_cols=None, cat_cols=None,
                               contamination=0.03, rare_threshold=0.05, score_threshold=0.5):
    df = df.copy()
    
    # Identify columns
    if num_cols is None:
        num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if cat_cols is None:
        cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Missing value imputation
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Numerical Outliers (IQR)
    iqr_cols = []
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        col_iqr = col + '_iqr'
        df[col_iqr] = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).astype(int)
        iqr_cols.append(col_iqr)
    
    # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['iso'] = iso.fit_predict(df[num_cols])
    df['iso'] = df['iso'].apply(lambda x: 1 if x==-1 else 0)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    df['lof'] = lof.fit_predict(df[num_cols])
    df['lof'] = df['lof'].apply(lambda x: 1 if x==-1 else 0)
    
    # Categorical Rare values
    cat_rare_cols = []
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare_labels = freq[freq < rare_threshold].index.tolist()
        col_rare = col + '_rare'
        df[col_rare] = df[col].apply(lambda x: 1 if x in rare_labels else 0)
        cat_rare_cols.append(col_rare)
    
    # Rare combinations 
    if len(cat_cols) >= 2:
        cat_comb = cat_cols[:2]
        comb_counts = df.groupby(cat_comb).size() / len(df)
        rare_combs = comb_counts[comb_counts < rare_threshold].index.tolist()
        df['rare_comb'] = df.apply(lambda x: 1 if tuple(x[cat_comb]) in rare_combs else 0, axis=1)
    else:
        df['rare_comb'] = 0
    
    # Aggregate Outlier Score
    outlier_cols = iqr_cols + cat_rare_cols + ['iso','lof','rare_comb']
    scaler = MinMaxScaler()
    df['num_outlier_norm'] = scaler.fit_transform(df[['iso','lof'] + iqr_cols]).mean(axis=1)
    df['outlier_score'] = df['num_outlier_norm'] + df[cat_rare_cols + ['rare_comb']].sum(axis=1)
    df['is_outlier'] = df['outlier_score'].apply(lambda x: 1 if x>score_threshold else 0)
    
    # Feature-wise contribution (store original col names instead of *_iqr)
    def get_contrib(row):
        contribs = []
        # numerical outliers
        for col, flag_col in zip(num_cols, iqr_cols):
            if row[flag_col] == 1:
                contribs.append(col)
        # categorical rare
        for col, flag_col in zip(cat_cols, cat_rare_cols):
            if row[flag_col] == 1:
                contribs.append(col)
        # ML methods → general mark
        if row['iso'] == 1: contribs.append("iso")
        if row['lof'] == 1: contribs.append("lof")
        if row['rare_comb'] == 1: contribs.append("rare_comb")
        return contribs
    
    df['feature_contrib'] = df.apply(get_contrib, axis=1)
    
    outliers_df = df[df['is_outlier']==1].copy()
    return outliers_df, df, num_cols, cat_cols


# -------------------------
# 2️⃣ Styling for Highlighting Outlier Values
# -------------------------
def highlight_outliers(row, num_cols, cat_cols):
    styles = []
    contribs = row['feature_contrib']
    for col in row.index:
        if col in contribs:  # highlight original column if flagged
            styles.append('background-color: yellow; color: red; font-weight: bold;')
        else:
            styles.append('')
    return styles

# -------------------------
# 3️⃣ Streamlit UI
# -------------------------
st.set_page_config(page_title="Real-time Outlier Detection", layout="wide")
st.title("📊 Real-time Outlier Detection Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())
    
    # Sidebar controls
    st.sidebar.header("Settings")
    contamination = st.sidebar.slider("Contamination (for ML methods)", 0.01, 0.2, 0.03)
    rare_threshold = st.sidebar.slider("Rare Category Threshold", 0.01, 0.1, 0.05)
    score_threshold = st.sidebar.slider("Outlier Score Threshold", 0.1, 5.0, 0.5)
    
    # Detect outliers
    outliers_df, full_df, num_cols, cat_cols = real_time_outlier_pipeline(
        df, contamination=contamination,
        rare_threshold=rare_threshold,
        score_threshold=score_threshold
    )
    
    st.subheader(f"Total Outliers Detected: {len(outliers_df)}")
    
    # Show all outliers with highlighted cells
    st.subheader("All Outliers (Highlighted Values)")
    display_cols = num_cols + cat_cols
    st.dataframe(
        outliers_df[display_cols + ['outlier_score','feature_contrib']]
        .style.apply(lambda row: highlight_outliers(row, num_cols, cat_cols), axis=1)
    )

    
    # -------------------------
    # Plots
    # -------------------------
    st.subheader("Outlier Score Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(full_df['outlier_score'], bins=20, kde=True, color='tomato', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Numerical Column Boxplots (First 5 Columns)")
    for col in num_cols[:5]:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x=full_df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
