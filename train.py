import pandas as pd
import joblib 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df= pd.read_csv("data/students.csv")

X=df.drop("passed",axis=1)
Y=df["passed"]

X_train, X_test, Y_train,Y_test= train_test_split(X, Y, test_size=0.3, random_state=42)

medians=X_train.median()

X_train =X_train.fillna(medians)
X_test= X_test.fillna(medians)

model=RandomForestClassifier(n_estimators=90, max_depth=5, random_state=42)
model.fit(X_train, Y_train)

predictions= model.predict(X_test)


# accuracy = accuracy_score(predictions, Y_test)
# print("accuracy : ", accuracy_score)
print(f"Accuracy : {accuracy_score(predictions, Y_test):.2%} ")
print(f"Accuracy : {classification_report(predictions, Y_test)} ")



joblib.dump(model,"models/student_model.joblib")
print("Model saved !")