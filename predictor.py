# =================================载入数据库================================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# =================================基础配置==================================
#加载训练好的XGBoost模型（确保XGBoost.pkl与脚本同目录）
model = joblib.load('XGBoost.pkl')

#加载测试数据（用于LIME解释器，确保X_test.csv与脚本同目录）
X_test = pd.read_csv('X_test.csv')

#定义特征名称（替换为业务相关列名，与编码规则对应）
feature_names = [
    "STEMI", "TIT", "IRA", "TIMI",
]

# ==========================Streamlit页面配置============================
st.set_page_config(page_title="I/Ra risk prediction calcular", layout="wide")
st.title("I/Ra risk prediction calcular")
st.markdown("Please fill out the following blank to get the result")

# ====================================特征输入组间（按编码规则设计）=======================================
# 1. STEMI（1: STEMI，2：NSTEMI）
STEMI = st.selectbox(
    "What is the diagnosis of AMI?",
    options=[1,2],
    format_func=lambda x: "STEMI" if x == 1 else "NSTEMI"
)


# 2. TIT（1：TIT≤12h，2：TIT>12h）
TIT = st.selectbox(
    "Does the total ischemia time less than 12hours?",
    options=[1, 2],
    format_func=lambda x: "≤12h" if x == 1 else ">12h"
)

# 3. IRA (1: LAD, 2: LCX, 3: RCA, 4: LM)
IRA = st.selectbox(
    "What is the culprit of artery responsible for AMI?",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "LAD",
        2: "LCX",
        3: "RCA",
        4: "LM",
    }[x]
)

# 4. TIMI (1: 70-99%狭窄，2：100%狭窄)
TIMI = st.selectbox(
    "What is the TIMI flow grade?",
    options=[1, 2],
    format_func=lambda x: "70-99%" if x == 1 else "100%"
)

# ======================================4. 数据处理与预测 ==============================================
# 整合用户输入特征
feature_values = [
    STEMI, TIT, IRA, TIMI
]

# 转换为模型输入格式
features = np.array([feature_values])

# 预测按钮逻辑
if st.button("Predict!"):
    # 模型预测
    predicted_class = model.predict(features)[0] #0：低风险，1：高风险
    predicted_proba = model.predict_proba(features)[0]  #概率值

    # 显示预测结果
    st.subheader("Prediction result")
    risk_label = "High risk" if predicted_class == 1 else "Low risk"
    st.write(f"**Level of Risk:{predicted_class}({risk_label})**")
    st.write(f"**Probability of Risk: ** Low risk {predicted_proba[0]:.2%}| High risk {predicted_proba[1]:.2%}")

    #生成个性化建议
    st.subheader("Suggestion")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"This person has a high risk of reperfusion arrhythmia(Probability{probability:.1f}%)."
        )
# ==================================5. LIME解释==================================
st.subheader("LIME特征贡献解释")
lime_explainer = LimeTabularExplainer(
    training_data=X_test.values,
    feature_names=feature_names,
    class_names=['Low I/Ra risk', 'High I/Ra risk'], #适配业务类别
    mode='classification'
)

#生成LIME解释
lime_exp = lime_explainer.explain_instance(
    data_row=features.flatten(),
    predict_fn=model.predict_proba,
    num_features=4  #显示前4个重要特征
)

#显示LIME解释（HTML格式）
lime_html = lime_exp.as_html(show_table=True)
st.components.v1.html(lime_html, height=600, scrolling=True)




