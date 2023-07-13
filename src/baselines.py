from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


class ModelComparisonExperiment:
    def __init__(self, X_train, X_test, y_train, y_test, groups):
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )
        self.y_train = self.y_train.replace({"other": 0, "Pump adjustment": 1})
        self.y_test = self.y_test.replace({"other": 0, "Pump adjustment": 1})
        self.groups = groups
        self.models = {}

    def train_decision_tree(self):
        model = DecisionTreeClassifier()

        param_grid = {
            "max_depth": [2, 4, 6, 8, 10, 12, 15],
            "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1", verbose=2)
        grid_search.fit(self.X_train, self.y_train, groups=self.groups)
        self.models["decision_tree"] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def train_random_forest(self):
        model = RandomForestClassifier()

        param_grid = {
            "n_estimators": [10, 20, 30, 40, 50],
            "max_depth": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1", verbose=2)
        grid_search.fit(self.X_train, self.y_train, groups=self.groups)
        self.models["random_forest"] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def evaluate_model(self, model):
        X = self.X_test
        y = self.y_test
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        results = {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "confusion_matrix": cm,
        }
        return results

    def select_best_model(self, models):
        # Compare the performance of different models and select the best-performing one
        pass

    def run_experiment(self):
        # Perform the experiment by calling relevant methods
        pass
