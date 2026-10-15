import joblib
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR=Path(__file__).parent

st.set_page_config(page_title="Stundent Predictor")

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR/"models"/"student_model.joblib")
#-----------------------------------------------------------------------
st.title("smart student predicetor")
st.write("Enter Student information ")
model_path=BASE_DIR /"models"/"student_model.joblib"
if not model_path.exists():
    st.error("model didnot fit, please run train.py first")
    st.stop()

model= load_model()
#----------------------------------------------------------------------
st.subheader("Student information")

col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider("Study hours per week", 0.0, 20.0, 10.0, step=0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
with col2:
    previous_grade = st.slider("Previous grade", 0, 100, 65)
    assignments_done = st.slider("Assignments submitted (%)", 0, 100, 70)

# ----------------------------------------------------------- التوقع
if st.button("Predict", type="primary"):
    # ترتيب الأعمدة لازم يكون نفس ترتيب التدريب
    student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_grade": [previous_grade],
        "assignments_done": [assignments_done],
    })

    result = model.predict(student)[0]
    chance = model.predict_proba(student)[0][1]

    st.subheader("Result")
    if result == 1:
        st.success("Likely to pass 👍")
    else:
        st.warning("May benefit from extra support 💬")

    st.metric("Estimated chance of passing", f"{chance:.0%}")
    st.progress(float(chance))

    # ------------------------------------------------------- رسم توضيحي
    # نقارن الطالب بمتوسط الطلاب في الداتا (كله على مقياس 0-100)
    data = pd.read_csv(BASE_DIR / "data" / "students.csv")
    average = data.drop("passed", axis=1).mean()

    compare = pd.DataFrame({
        "This student": [
            study_hours / 20 * 100,
            attendance,
            previous_grade,
            assignments_done,
        ],
        "Class average": [
            average["study_hours"] / 20 * 100,
            average["attendance"],
            average["previous_grade"],
            average["assignments_done"],
        ],
    }, index=["Study hours (% of 20h)", "Attendance %", "Previous grade", "Assignments %"])

    st.subheader("How does this student compare?")
    st.bar_chart(compare)

# ----------------------------------------------------------- تنبيه
st.info(
    "This is only an estimate learned from educational (synthetic) data. "
    "It is not a judgment of any student's ability or future."
)
