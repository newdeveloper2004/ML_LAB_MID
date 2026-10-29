import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from utils import get_reg_split, get_clf_split

# Model 1: linear regression
X_train, X_test, y_train, y_test = get_reg_split()
reg = LinearRegression().fit(X_train, y_train)
print("R²:", r2_score(y_test, reg.predict(X_test)))
joblib.dump(reg, "models/model1.pkl")

# Model 2: classification
X_train, X_test, y_train, y_test = get_clf_split()
clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
joblib.dump(clf, "models/model2.pkl")
