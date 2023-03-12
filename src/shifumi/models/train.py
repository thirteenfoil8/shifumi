import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import pathlib
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

from dotenv import dotenv_values
config = dotenv_values(".env")


class Train:
    def __init__(self) -> None:
        self.path_checkpoint = config["MODEL_CHECKPOINT"]
        self.data_path = config["DATA_HAND"]
        self.classes = config["CLASSES"]

    def load_data(self):
        if pathlib.Path(self.data_path + 'store.h5').exists():
            self.data = pd.read_hdf(self.data_path + 'store.h5')
        else:
            data = pd.DataFrame()
            for idx, class_ in enumerate(["rock", "paper", "scissor"]):
                with open(self.data_path + f'{class_}.json', 'r') as file:
                    temp = json.load(file)
                    class_data = []
                    for i in range(len(temp)):
                        class_data.append(temp[i][f"{i}"])
                    temp_df = pd.DataFrame(
                        class_data, columns=list(range(0, 42)))
                    temp_df["42"] = idx
                    data = pd.concat(
                        [data, temp_df], ignore_index=True).reset_index(drop=True)
            self.data = data
            self.transform()

    def transform(self):
        self.data = self.data.apply(lambda r: self.center_points(r), axis=1)
        self.data.to_hdf(self.data_path + 'store.h5', "data")

    def group_points(self, row):
        pairs_list = []
        for i in range(0, len(row), 2):
            if i+1 < len(row):
                pairs_list.append([row[i], row[i+1]])
        return np.array(pairs_list)

    def get_centeroidnp(self, arr):
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length

    def center_points(self, row):
        label = row["42"]
        row = row.drop("42")
        points = self.group_points(row)
        centroid = self.get_centeroidnp(points)
        new_coordinates = points - centroid
        new_coordinates = new_coordinates.flatten()
        new_coordinates = np.append(new_coordinates, label)
        return pd.Series(new_coordinates)

    def train(self):
        X = self.data.drop(42, axis=1)
        y = self.data[42]

        scores = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=12, test_size=30)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

        # for learning_rate in lr_list:
        #     gb_clf = GradientBoostingClassifier(
        #         n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
        #     gb_clf.fit(X_train, y_train)

        #     print("Learning rate: ", learning_rate)
        #     print("Accuracy score (training): {0:.3f}".format(
        #         gb_clf.score(X_train, y_train)))

        # params = {
        #     "colsample_bytree": uniform(0.7, 0.3),
        #     "gamma": uniform(0, 0.5),
        #     "learning_rate": uniform(0.03, 0.3),  # default 0.1
        #     "max_depth": randint(2, 6),  # default 3
        #     "n_estimators": randint(100, 150),  # default 100
        #     "subsample": uniform(0.6, 0.4)
        # }

        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob", random_state=42)
        # search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42,
        #                             n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)
        # search.fit(X, y)
        # self.report_best_scores(search.cv_results_, 1)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        # print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(
            xgb_model.score(X_train, y_train)))
        xgb_model.save_model(config["MODEL_PATH"])

    def pca(self):
        X = self.data.drop(42, axis=1)
        y = self.data[[42]]
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        pca = PCA(n_components=0.95, random_state=2020)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        n_pcs = pca.components_.shape[0]
        new_df = pd.DataFrame(
            X_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"])
        new_df["label"] = self.data[42]
        most_important = [np.abs(pca.components_[i]).argmax()
                          for i in range(n_pcs)]
        self.data = new_df

    def report_best_scores(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def load_model(self):
        model = self.create_model()
        model.load_weights(self.path_checkpoint)
        return model


if __name__ == "__main__":
    train = Train()
    train.load_data()
    # train.pca()
    train.train()
    print("end")
    # train.load_model()
