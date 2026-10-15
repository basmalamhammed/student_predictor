
# generate_data.py

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 600

# 1. Generate student features
study_hours = np.random.uniform(0, 20, n)
attendance = np.random.uniform(40, 100, n)
previous_grade = np.random.uniform(0, 100, n)
assignments_done = np.random.uniform(0, 100, n)

# 2. Calculate student scores
score = (
    1.5 * study_hours
    + 0.3 * attendance
    + 0.4 * previous_grade
    + 0.2 * assignments_done
    + np.random.normal(0, 8, n)
)

# 3. Generate target labels
passed = (score > np.median(score)).astype(int)

# 4. Create the dataset
df = pd.DataFrame({
    "study_hours": study_hours.round(1),
    "attendance": attendance.round(1),
    "previous_grade": previous_grade.round(1),
    "assignments_done": assignments_done.round(1),
    "passed": passed,
})

# 5. Introduce missing values
for col in ["attendance", "previous_grade"]:
    missing_rows = np.random.choice(n, size=30, replace=False)
    df.loc[missing_rows, col] = np.nan

# 6. Save the dataset
output_path = Path(__file__).resolve().parent.parent / "data" / "students.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Done! Dataset saved to: {output_path}")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nPassed distribution:\n{df['passed'].value_counts()}")
