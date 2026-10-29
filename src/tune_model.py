from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils import get_clf_split

X_train, X_test, y_train, y_test = get_clf_split()
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5]},
    cv=5,
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Test accuracy:", grid.score(X_test, y_test))
