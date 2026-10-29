from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import get_reg_split, get_clf_split

X, _, y, _ = get_reg_split()
for m in [LinearRegression(), Ridge()]:
    print(type(m).__name__, "R²:", cross_val_score(m, X, y, cv=5, scoring="r2").mean())

X, _, y, _ = get_clf_split()
for m in [RandomForestClassifier(random_state=42), LogisticRegression(max_iter=500)]:
    print(type(m).__name__, "Accuracy:", cross_val_score(m, X, y, cv=5, scoring="accuracy").mean())
