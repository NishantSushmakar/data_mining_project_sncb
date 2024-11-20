import numpy as np
import pandas as pd
import ast
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

    def transform(self, X, y=None):
        for col in self.sequence_columns:
            X[col] = X[col].apply(ast.literal_eval)
        return X

    def fit(self, X, y=None):
        return self

class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self, lat_extreme, lon_extreme, n_neighbors=2):
        self.lat_extreme = lat_extreme
        self.lon_extreme = lon_extreme
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.loc[
            (X["approx_lat"] < self.lat_extreme[0]) | (X["approx_lat"] > self.lat_extreme[1]),
            "approx_lat",
        ] = None
        X.loc[
            (X["approx_lon"] < self.lon_extreme[0]) | (X["approx_lon"] > self.lon_extreme[1]),
            "approx_lon",
        ] = None

        def impute_group(group):
            features = group[["approx_lat", "approx_lon"]]
            imputer = KNNImputer(n_neighbors=self.n_neighbors)
            imputed = imputer.fit_transform(features)
            group[["approx_lat", "approx_lon"]] = imputed
            return group

        X = X.groupby("incident_type").apply(impute_group)
        X = X.reset_index(drop=True)
        X = X.sort_values(by=X.columns[0]).reset_index(drop=True)

        return X

class AddIndexSequence(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_idx(self, ls):
        for idx in range(0, len(ls) - 1):
            if int(ls[idx + 1]) > 0:
                return idx
        return idx + 1

    def transform(self, X, y=None):
        X["index_sequence"] = X["seconds_to_incident_sequence"].apply(self.get_idx)
        return X

    def fit(self, X, y=None):
        return self

class AddWindowIndices(BaseEstimator, TransformerMixin):
    def __init__(self, window_start=3600, window_end=600):
        self.window_start = window_start
        self.window_end = window_end

    def get_window(self, lst, idx):
        pre_incident_idx = idx
        while abs(int(lst[pre_incident_idx])) <= self.window_start and pre_incident_idx >= 0:
            pre_incident_idx -= 1
        pre_incident_idx += 1

        post_incident_idx = idx
        while post_incident_idx < len(lst) and int(lst[post_incident_idx]) <= self.window_end:
            post_incident_idx += 1
        post_incident_idx -= 1

        return pre_incident_idx, post_incident_idx

    def transform(self, X, y=None):
        X[["window_min_idx", "window_max_idx"]] = X[
            ["seconds_to_incident_sequence", "index_sequence"]
        ].apply(
            lambda row: self.get_window(row["seconds_to_incident_sequence"], row["index_sequence"]),
            axis=1,
        ).apply(pd.Series)
        return X

    def fit(self, X, y=None):
        return self

class AddWindowedSequences(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def get_data_sequence_within_windows(self, row, column):
        min_idx = row["window_min_idx"]
        max_idx = row["window_max_idx"]
        return row[column][min_idx : max_idx + 1]

    def transform(self, X, y=None):
        for col in self.columns:
            X[f"window_{col}"] = X.apply(
                lambda row: self.get_data_sequence_within_windows(row, col), axis=1
            )
        return X

    def fit(self, X, y=None):
        return self

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

    pipeline = Pipeline(
        steps=[
            ("convert_sequences", ConvertSequencesToLists(sequence_columns=sequence_columns)),
            ("outlier_imputer", OutlierImputer(lat_extreme=[49.5072, 51.4978], lon_extreme=[2.5833, 6.3667], n_neighbors=2)),
            ("add_index_sequence", AddIndexSequence()),
            ("add_window_indices", AddWindowIndices(window_start=3600, window_end=600)),
            ("add_windowed_sequences", AddWindowedSequences(columns=sequence_columns)),
        ]
    )

    df_transformed = pipeline.fit_transform(df)
    output_file = "sncb_data_transformed.csv"
    df_transformed.to_csv(output_file, index=False, sep=";")
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    main()
