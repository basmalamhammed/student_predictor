# 🎓 Smart Student Success Predictor

A beginner-friendly, **end-to-end Machine Learning project**. It uses a Random Forest to estimate whether a student is likely to pass, based on study habits, and wraps the model in a simple Streamlit web app.

The goal is not only the final app. The project is built so that learners walk through the **full life cycle of an ML project**, step by step.

> ⚠️ **Disclaimer:** The dataset is synthetic and made for teaching. Predictions are rough estimates, not a judgment of any student's ability or future.

---

## What the app does

1. The user enters: weekly study hours, attendance %, previous grade, and assignments submitted %.
2. They press **Predict**.
3. The app shows the estimated chance of passing and a chart comparing the student to the class average.

## What you will learn

| # | Stage | What you do |
|---|-------|-------------|
| 1 | Understand the data | Load the CSV, inspect rows and columns, choose Features and Target, find missing values |
| 2 | Exploratory Data Analysis | Use Pandas and Matplotlib to see how study hours, attendance and grades relate to the result |
| 3 | Build a Random Forest | Train/test split, handle missing values, train, predict |
| 4 | Evaluate the model | Accuracy, Precision, Recall, F1-score, Confusion Matrix, Feature Importance |
| 5 | Tune and diagnose | Try `n_estimators` and `max_depth`, and spot Overfitting |
| 6 | Save the model | Store and reload it with Joblib |
| 7 | Build the app | Turn the experiment into a Streamlit app |
| 8 | Ship it | Organize files, write `requirements.txt` and README, push to GitHub, deploy |

## Project structure

```
student-success-predictor/
├── data/
│   └── students.csv                 # the dataset
├── models/
│   └── student_model.joblib         # the trained model
├── student_success_tutorial.ipynb   # step-by-step tutorial notebook
├── generate_data.py                 # creates data/students.csv
├── train.py                         # trains and saves the model
├── app.py                           # Streamlit web app
├── requirements.txt
└── README.md
```

## Getting started

**1. Install the libraries**

```bash
pip install -r requirements.txt
```

**2. Create the data** (skip if `data/students.csv` already exists)

```bash
python generate_data.py
```

**3. Follow the tutorial notebook**

```bash
jupyter notebook student_success_tutorial.ipynb
```

**4. Train the model from a script**

```bash
python train.py
```

**5. Run the app**

```bash
streamlit run app.py
```

## Model details

- **Algorithm:** `RandomForestClassifier` (100 trees, `max_depth=5`)
- **Features:** `study_hours`, `attendance`, `previous_grade`, `assignments_done`
- **Target:** `passed` (1 = passed, 0 = failed)
- **Missing values:** filled with the median, computed from the training set only (to avoid data leakage)
- **Test accuracy:** about 83% on a 120-student test set

With a test set this small, differences of 1-2% between settings are mostly noise.

## Limitations

- The data is synthetic, so the model has learned the rules we wrote into the generator, not real student behavior.
- Real student outcomes depend on many factors that are not in this dataset.
- A model like this should never be used to make decisions about real students.

## Deploy the app (optional)

1. Push this folder to a GitHub repository:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Student Success Predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/student-success-predictor.git
   git push -u origin main
   ```

2. Open [Streamlit Community Cloud](https://streamlit.io/cloud), sign in with GitHub, and create a new app from the repository, with `app.py` as the main file.

3. If you see a scikit-learn version warning after deploying, re-run `python train.py` locally with the same library versions and push the updated model file.

## Tech stack

Python · Pandas · Matplotlib · scikit-learn · Joblib · Streamlit

## Ideas to extend the project

- Add more Features (sleep hours, extra classes) and see how the results change
- Compare Random Forest with Logistic Regression or a single Decision Tree
- Use `GridSearchCV` to tune hyperparameters
- Replace the synthetic data with a real public dataset
