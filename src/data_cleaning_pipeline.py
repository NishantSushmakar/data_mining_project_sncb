import numpy as np
import pandas as pd
import ast
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

def make_dict(row):
    data = dict()
    for seconds, vehicle_id, event_id, ac_state, dc_state, train_speed in zip(
        row["seconds_to_incident_sequence"],
        row["vehicles_sequence"],
        row["events_sequence"],
        row["dj_ac_state_sequence"],
        row["dj_dc_state_sequence"],
        row["train_kph_sequence"],
    ):
        if seconds not in data.keys():
            data[seconds] = {
                vehicle_id: {
                    event_id: {
                        "train_speed": train_speed,
                        "ac_state": ac_state,
                        "dc_state": dc_state,
                    }
                }
            }
        elif vehicle_id not in data[seconds].keys():
            data[seconds] = {
                vehicle_id: {
                    event_id: {
                        "train_speed": train_speed,
                        "ac_state": ac_state,
                        "dc_state": dc_state,
                    }
                }
            }
        elif event_id not in data[seconds][vehicle_id].keys():
            data[seconds][vehicle_id] = {
                event_id: {
                    "train_speed": train_speed,
                    "ac_state": ac_state,
                    "dc_state": dc_state,
                }
            }
        else:
            print("problem")
    return data

class ConvertSequencesToLists(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_columns):
        self.sequence_columns = sequence_columns
        self.transformed_data_ = None

    def fit(self, X, y=None):
        for col in self.sequence_columns:
            X[col] = X[col].apply(ast.literal_eval)
        self.transformed_data_ = X
        return self

    def transform(self, X, y=None):
        if self.transformed_data_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        return self.transformed_data_

class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self, lat_extreme, lon_extreme, n_neighbors=2, model_file='knn_imputer.pkl'):
        self.lat_extreme = lat_extreme
        self.lon_extreme = lon_extreme
        self.n_neighbors = n_neighbors
        self.model_file = model_file
        self.knn_imputer = None

    def mark_outliers(self, X):
        #outliers := NaN
        X.loc[(X["approx_lat"] < self.lat_extreme[0]) | (X["approx_lat"] > self.lat_extreme[1]), "approx_lat"] = None
        X.loc[(X["approx_lon"] < self.lon_extreme[0]) | (X["approx_lon"] > self.lon_extreme[1]), "approx_lon"] = None
        return X

    def apply_imputer(self, group):
        features = group[["approx_lat", "approx_lon"]]
        imputed = self.knn_imputer.transform(features)
        group[["approx_lat", "approx_lon"]] = imputed
        return group

    def fit(self, X, y=None):
        X = self.mark_outliers(X)
        features = X[["approx_lat", "approx_lon"]]
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.knn_imputer.fit(features)

        joblib.dump(self.knn_imputer, self.model_file)

        return self

    def transform(self, X, y=None):
        #outliers := NaN
        X = self.mark_outliers(X)
        if self.knn_imputer is None:
            self.knn_imputer = joblib.load(self.model_file)

        X = X.groupby("incident_type").apply(self.apply_imputer)
        X = X.reset_index(drop=True)
        X = X.sort_values(by=X.columns[0]).reset_index(drop=True)

        return X

class AddIndexSequence(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformed_data_ = None

    def get_idx(self, ls):
        for idx in range(0, len(ls) - 1):
            if int(ls[idx + 1]) > 0:
                return idx
        return idx + 1

    def fit(self, X, y=None):
        X["index_sequence"] = X["seconds_to_incident_sequence"].apply(self.get_idx)
        self.transformed_data_ = X
        return self

    def transform(self, X, y=None):
        if self.transformed_data_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        return self.transformed_data_

class AddWindowIndices(BaseEstimator, TransformerMixin):
    def __init__(self, window_start=3600, window_end=600):
        self.window_start = window_start
        self.window_end = window_end
        self.transformed_data_ = None

    def get_window(self, lst, idx):
        # Pre-incident window
        pre_incident_idx = idx
        while abs(int(lst[pre_incident_idx])) <= self.window_start and pre_incident_idx >= 0:
            pre_incident_idx -= 1
        pre_incident_idx += 1

        # Post-incident window
        post_incident_idx = idx
        while post_incident_idx < len(lst) and int(lst[post_incident_idx]) <= self.window_end:
            post_incident_idx += 1
        post_incident_idx -= 1

        return pre_incident_idx, post_incident_idx

    def fit(self, X, y=None):
        X[["window_min_idx", "window_max_idx"]] = X[
            ["seconds_to_incident_sequence", "index_sequence"]
        ].apply(
            lambda row: self.get_window(row["seconds_to_incident_sequence"], row["index_sequence"]),
            axis=1,
        ).apply(pd.Series)
        self.transformed_data_ = X
        return self

    def transform(self, X, y=None):
        if self.transformed_data_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        return self.transformed_data_

class AddWindowedSequences(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.transformed_data_ = None

    def get_data_sequence_within_windows(self, row, column):
        min_idx = row["window_min_idx"]
        max_idx = row["window_max_idx"]
        return row[column][min_idx : max_idx + 1]

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            X[f"window_{col}"] = X.apply(
                lambda row: self.get_data_sequence_within_windows(row, col), axis=1
            )
        self.transformed_data_ = X
        return self

    def transform(self, X, y=None):
        if self.transformed_data_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        return self.transformed_data_


class UniqueAndHighestFrequencyPercentageSequences(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformed_data_ = None

    def fit(self, X, y=None):
        X["unique_vehicle_sequence"] = X["vehicles_sequence"].apply(lambda x: len(set(x)))
        X["unique_events_sequence"] = X["events_sequence"].apply(lambda x: len(set(x)))
        X["unique_ac_state_sequence"] = X["dj_ac_state_sequence"].apply(lambda x: len(set(x)))
        X["unique_dc_state_sequence"] = X["dj_dc_state_sequence"].apply(lambda x: len(set(x)))
        X["highest_frequency_percentage_vehicle_sequence"] = X["vehicles_sequence"].apply(lambda x: max(x.count(vehicle) / len(x) for vehicle in set(x)))
        X["highest_frequency_percentage_events_sequence"] = X["events_sequence"].apply(lambda x: max(x.count(event) / len(x) for event in set(x)))
        self.transformed_data_ = X
        return self

    def transform(self, X, y=None):
        if self.transformed_data_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        return self.transformed_data_


def main():
    df = pd.read_csv("sncb_data_challenge.csv", delimiter=";")

    sequence_columns = [
        "vehicles_sequence",
        "events_sequence",
        "seconds_to_incident_sequence",
        "train_kph_sequence",
        "dj_ac_state_sequence",
        "dj_dc_state_sequence",
    ]

    cleaning_pipeline = Pipeline(
        steps=[
            ("convert_sequences", ConvertSequencesToLists(sequence_columns=sequence_columns)),
            # assuming this rectangular mask would be enough to identify the obvious outliers
            ("outlier_imputer", OutlierImputer(lat_extreme=[49.5072, 51.4978], lon_extreme=[2.5833, 6.3667], n_neighbors=2)),
            ("add_index_sequence", AddIndexSequence()),
            ("add_window_indices", AddWindowIndices(window_start=3600, window_end=600)),
            ("add_windowed_sequences", AddWindowedSequences(columns=sequence_columns)),
        ]
    )

    feature_pipeline = Pipeline(
        steps=[
            ("first_six_features", UniqueAndHighestFrequencyPercentageSequences()),
    ])

    cleaning_pipeline.fit(df)
    df_cleaned = cleaning_pipeline.transform(df)

    feature_pipeline.fit(df_cleaned)
    df_new_features = feature_pipeline.transform(df_cleaned)

    output_file = "sncb_data_transformed.csv"
    df_new_features.to_csv(output_file, index=False, sep=";")
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    main()
