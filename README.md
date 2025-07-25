# 🧠 Diabetes Predictor Web App

A simple and interactive **Streamlit web app** that predicts whether a person is diabetic or not using a **Support Vector Machine (SVM)** model trained on health data.

[![Open in Streamlit][(https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://akshaanxh-diabetes-predictor.streamlit.app)
](https://diabetes-predictor-vyywp7exwa9972cgaebq5t.streamlit.app/)
---

## 📊 Features

- User-friendly interface with health parameter inputs
- Machine learning model built using:
  - **Python**
  - **Streamlit**
  - **Scikit-learn**
  - **NumPy**
- Model trained to predict diabetes from real-world data
- Deployed on **Streamlit Cloud** with instant live access

---

## 🚀 Live Demo

🔗 [Click here to open the live app][(https://akshaanxh-diabetes-predictor.streamlit.app)](https://diabetes-predictor-vyywp7exwa9972cgaebq5t.streamlit.app/)

---

## 📥 Run the Project Locally

### 1. Clone this repository

```bash
git clone https://github.com/akshaanxh/Diabetes-Predictor.git
cd Diabetes-Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 ML Model Details

- **Algorithm:** Support Vector Machine (SVM)
- **Target Variable:** Diabetic (1) / Not Diabetic (0)
- **Input Features:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

---

## 📁 Project Structure

| File                | Description                            |
|---------------------|----------------------------------------|
| `app.py`            | Streamlit frontend app script          |
| `trained_model.sav` | Pre-trained SVM model for predictions  |
| `requirements.txt`  | Python dependencies                    |
| `README.md`         | Project documentation                  |

---

## ✍️ Author

👤 [Akshaanxh](https://github.com/akshaanxh)
