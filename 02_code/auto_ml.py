import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import time

# タイトル
st.title("AutoMLツール（仮）")

# サイドバーでモデルの選択
st.sidebar.title("モデルの選択")
model_choice = st.sidebar.selectbox("モデル", ["ロジスティック回帰", "ランダムフォレスト", "SVM"])

# メイン画面でデータをアップロード
st.header("前処理済みデータのアップロード")
uploaded_file = st.file_uploader("ファイルの選択")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # 目的変数の選択
    target_column = st.selectbox("目的変数の選択", df.columns)
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # モデルの学習
        if st.button("モデルの学習"):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_choice == "ロジスティック回帰":
                model = LogisticRegression()
            elif model_choice == "ランダムフォレスト":
                model = RandomForestClassifier()
            elif model_choice == "SVM":
                model = SVC(probability=True)

            model.fit(X_train, y_train)
            st.info("モデルを学習中・・・")
            time.sleep(3)

            st.success("学習成功！")

            # 学習済みモデルとテストデータをセッション状態に保存
            st.session_state['model'] = model
            st.session_state['X_test_split'] = X_val
            st.session_state['y_test_split'] = y_val

# テストデータのアップロード
if 'model' in st.session_state:
    st.header("テストデータのアップロード")
    test_file = st.file_uploader("テストデータの選択", key="test")
    if st.button("テストデータの評価結果の表示"):
        if test_file is None:
            st.warning("ファイルをアップロードしてください")
        else:
            test_df = pd.read_csv(test_file)
            st.write(test_df.head())

            # テストデータでの予測
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            y_pred = st.session_state['model'].predict(X_test)
            y_proba = st.session_state['model'].predict_proba(X_test)[:, 1]

            # 混同行列の表示
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("混同行列")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # ROC曲線の表示
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            st.subheader("ROC 曲線")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC 曲線 (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('偽陽性率')
            ax.set_ylabel('真陽性率')
            ax.set_title('ROC曲線')
            ax.legend(loc="lower right")
            st.pyplot(fig)

            # PR曲線の表示
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            st.subheader("Precision-Recall 曲線")
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall 曲線')
            ax.legend(loc="lower left")
            st.pyplot(fig)
