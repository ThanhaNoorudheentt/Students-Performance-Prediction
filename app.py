import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, label_encoder


model, label_encoder = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────────────────────────────────────
def predict(input_df: pd.DataFrame):
    pred_encoded = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prediction = label_encoder.inverse_transform([pred_encoded])[0]
    return prediction, probabilities


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📋 Student Information")

with st.sidebar.form("student_form"):

    st.subheader("Personal & Family")

    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)

    address = st.selectbox(
        "Address",
        ["U", "R"],
        help="Urban / Rural"
    )

    famsize = st.selectbox("Family Size", ["LE3", "GT3"])

    Pstatus = st.selectbox(
        "Parent Status",
        ["T", "A"],
        help="Together / Apart"
    )

    # ─────────────────────────────────────────
    st.subheader("Education")

    Medu = st.slider("Mother Education (0-4)", 0, 4, 2)
    Fedu = st.slider("Father Education (0-4)", 0, 4, 2)

    Mjob = st.selectbox(
        "Mother Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

    Fjob = st.selectbox(
        "Father Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

    reason = st.selectbox(
        "Reason for School",
        ["home", "reputation", "course", "other"]
    )

    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"]
    )

    traveltime = st.slider("Travel Time (1-4)", 1, 4, 1)
    studytime = st.slider("Study Time (1-4)", 1, 4, 2)

    # ─────────────────────────────────────────
    st.subheader("Academic History")

    failures = st.slider("Past Failures", 0, 3, 0)
    G1 = st.slider("Grade Period 1 (0-20)", 0, 20, 10)
    G2 = st.slider("Grade Period 2 (0-20)", 0, 20, 10)

    # ─────────────────────────────────────────
    st.subheader("School Support")

    schoolsup = st.selectbox("Extra School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

    activities = st.selectbox(
        "Extra-curricular Activities",
        ["yes", "no"]
    )

    nursery = st.selectbox("Attended Nursery", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Internet at Home", ["yes", "no"])

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )

    # ─────────────────────────────────────────
    st.subheader("Lifestyle")

    famrel = st.slider(
        "Family Relationship Quality (1-5)",
        1, 5, 4
    )

    freetime = st.slider(
        "Free Time After School (1-5)",
        1, 5, 3
    )

    goout = st.slider(
        "Going Out with Friends (1-5)",
        1, 5, 3
    )

    Dalc = st.slider(
        "Workday Alcohol Consumption (1-5)",
        1, 5, 1
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption (1-5)",
        1, 5, 1
    )

    health = st.slider(
        "Current Health Status (1-5)",
        1, 5, 3
    )

    absences = st.slider("Number of Absences", 0, 93, 5)

    submitted = st.form_submit_button(
        "🔮 Predict",
        width="stretch"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎓 Student Pass / Fail Predictor")

st.markdown(
    """
    Fill in the student details in the sidebar and click **Predict**
    to see whether the student is likely to pass or fail.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
if submitted:

    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,
    }])

    prediction, probabilities = predict(input_data)

    col1, col2 = st.columns([1, 2])

    # ─────────────────────────────────────────
    # Result Card
    # ─────────────────────────────────────────
    with col1:

        if prediction == "Pass":
            st.success(f"### ✅ {prediction}")
        else:
            st.error(f"### ❌ {prediction}")

        classes = label_encoder.classes_

        for cls, prob in zip(classes, probabilities):
            st.metric(
                label=f"P({cls})",
                value=f"{prob:.1%}"
            )

    # ─────────────────────────────────────────
    # Probability Chart
    # ─────────────────────────────────────────
    with col2:

        st.subheader("Prediction Confidence")

        fig, ax = plt.subplots(figsize=(6, 3))

        colors = [
            "#ef4444" if c == "Fail" else "#22c55e"
            for c in classes
        ]

        bars = ax.barh(
            classes,
            probabilities,
            color=colors,
            height=0.4
        )

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")

        for bar, prob in zip(bars, probabilities):

            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}",
                va="center",
                fontsize=11
            )

        ax.spines[["top", "right"]].set_visible(False)

        st.pyplot(fig)

        plt.close(fig)

    # ─────────────────────────────────────────
    # Input Summary
    # ─────────────────────────────────────────
    with st.expander("📄 Input Summary"):

        summary_df = (
            input_data.T
            .reset_index()
            .rename(columns={
                "index": "Feature",
                0: "Value"
            })
        )

        # Fix PyArrow mixed datatype issue
        summary_df["Value"] = summary_df["Value"].astype(str)

        st.dataframe(
            summary_df,
            width="stretch"
        )

else:
    st.info(
        "👈 Fill in the student details in the sidebar and click **Predict**."
    )