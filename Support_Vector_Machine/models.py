# train_model.py
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

from data_utils import load_data
from preprocess import preprocess_split


def train(X, y, encoders):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=10
    )

    svr = SVR()
    svr.fit(X_train, y_train)

    pred = svr.predict(X_test)
    print("Base R2:", r2_score(y_test, pred))
    print("Base MAE:", mean_absolute_error(y_test, pred))

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=1)
    grid.fit(X_train, y_train)

    best_pred = grid.predict(X_test)
    print("Tuned R2:", r2_score(y_test, best_pred))
    print("Tuned MAE:", mean_absolute_error(y_test, best_pred))

    pickle.dump(grid, open("model.pkl", "wb"))
    pickle.dump(encoders, open("encoder.pkl", "wb"))

    return grid
