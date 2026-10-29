import joblib
import pandas as pd

reg = joblib.load("models/model1.pkl")
clf = joblib.load("models/model2.pkl")

reg_input = pd.DataFrame([[5.1, 3.5, 0.2]], columns=["sepal_length", "sepal_width", "petal_width"])
clf_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

print("Predicted petal_length:", reg.predict(reg_input)[0])
print("Predicted species:", clf.predict(clf_input)[0])
