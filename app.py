import streamlit as st  # type: ignore[import]
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="GlucoSense AI",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD DATA ---------------- #

data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ---------------- TRAIN MODEL ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- APP TITLE ---------------- #

st.title("🩺 GlucoSense AI")

st.markdown("""
### AI-Powered Diabetes Risk Assessment

GlucoSense AI uses machine learning to analyze health indicators and estimate possible diabetes risk in a patient-centered and supportive way.

⚠️ This application is designed for educational purposes only and does not replace professional medical diagnosis or treatment.
""")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("GlucoSense AI")

st.sidebar.write("""
AI-powered diabetes risk assessment system designed to support predictive healthcare analysis.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("System Status")
st.sidebar.success("Prediction Model Operational")

st.sidebar.markdown("---")

st.sidebar.subheader("Core Capabilities")

st.sidebar.write("""
• Diabetes Risk Prediction

• Risk Probability Analysis

• Explainable AI Insights

• Patient-Centered Responses

• Preventive Health Recommendations
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset Information")

st.sidebar.write("""
This prediction model was trained using female patient diabetes data for educational healthcare analysis.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Clinical Notice")

st.sidebar.write("""
This application is not intended to replace professional medical diagnosis or clinical decision-making.
""")

st.sidebar.markdown("---")

st.sidebar.caption("""
Digital Health & Predictive AI Prototype
""")

# ---------------- INPUT SECTION ---------------- #

st.subheader("📋 Patient Health Information")

pregnancies = st.number_input("Pregnancies", min_value=0.0)
glucose = st.number_input("Glucose Level", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1.0)

# ---------------- PREDICTION BUTTON ---------------- #

if st.button("Analyze Risk"):

    patient_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]]

    # Prediction
    prediction = model.predict(patient_data)

    # Risk probability
    probability = model.predict_proba(patient_data)
    risk = probability[0][1] * 100

    # ---------------- RESULTS ---------------- #

    st.subheader("📊 Risk Analysis")

    st.progress(int(risk))

    if prediction[0] == 1:

        st.warning(f"""
        ⚠️ GlucoSense AI detected a higher estimated diabetes risk ({risk:.1f}% probability).

        Please remember that this result is not a medical diagnosis. However, the provided health indicators show patterns that may be associated with increased diabetes risk.

        Taking early preventive action and consulting a healthcare professional can make a significant difference.
        """)

        st.markdown("""
        ### Supportive Recommendations

        - Consider speaking with a healthcare professional
        - Monitor blood glucose levels regularly
        - Maintain balanced nutrition
        - Stay physically active
        - Prioritize routine health checkups
        """)

    else:

        st.success(f"""
        ✅ GlucoSense AI detected a lower estimated diabetes risk ({100-risk:.1f}% confidence).

        The provided health indicators do not currently show strong patterns commonly associated with diabetes risk.

        Continuing healthy habits and regular health monitoring is still very important.
        """)

        st.markdown("""
        ### Healthy Lifestyle Tips

        - Continue healthy eating habits
        - Exercise regularly
        - Stay hydrated
        - Maintain consistent sleep patterns
        - Attend regular medical checkups
        """)

    # ---------------- EXPLAINABLE AI ---------------- #

    st.subheader("🧠 AI Insight Summary")

    explanations = []

    if glucose > 140:
        explanations.append("Higher glucose levels contributed strongly to the prediction.")

    if bmi > 30:
        explanations.append("BMI was identified as a significant health factor.")

    if age > 45:
        explanations.append("Age slightly increased the estimated risk level.")

    if dpf > 0.5:
        explanations.append("Family-history-related indicators influenced the prediction.")

    if len(explanations) == 0:
        explanations.append("No major high-risk indicators were strongly detected.")

    for item in explanations:
        st.write("•", item)

    # ---------------- CHART ---------------- #

    st.subheader("📈 Risk Visualization")

    chart_data = pd.DataFrame(
        {"Percentage": [risk, 100 - risk]},
        index=["Estimated Risk", "Remaining"]
    )
    st.bar_chart(chart_data)

    # ---------------- REPORT ---------------- #

    timestamp = datetime.datetime.now()

    report = f"""
GlucoSense AI - Patient Risk Report
-----------------------------------

Date: {timestamp}

Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {blood_pressure}
Skin Thickness: {skin_thickness}
Insulin: {insulin}
BMI: {bmi}
DPF: {dpf}
Age: {age}

Estimated Diabetes Risk: {risk:.1f}%

Note:
This report is generated for educational purposes only and does not replace medical diagnosis.
"""

    st.download_button(
        label="Download Patient Report",
        data=report,
        file_name="glucosense_report.txt",
        mime="text/plain"
    )

    # ---------------- PATIENT SUMMARY ---------------- #

    st.subheader("📝 Patient Summary")

    st.write(f"""
Age: {age}

Glucose Level: {glucose}

BMI: {bmi}

Estimated Diabetes Risk: {risk:.1f}%
""")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption("""
GlucoSense AI • Predictive Healthcare & Digital Health AI Project
""")