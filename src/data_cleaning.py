import pandas as pd
import ast
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import config
import os
import numpy as np

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
        self.output_column = "sequence_dict"

    def process_data(self, X):
        for col in self.sequence_columns:
            X[col] = X[col].apply(ast.literal_eval)
        ## Creating dictionary with the column    
        X[self.output_column] = X.apply(lambda row: make_dict(row), axis=1)
        return X

    def fit(self, X, y=None):
        return self
    
    def fit_transform(self,X,y=None):
        
        return self.process_data(X)


    def transform(self, X, y=None):
        
        return self.process_data(X)
    

class RemoveNoise(BaseEstimator,TransformerMixin):

    def __init__(self):
        self.frequency_dict = {}
        self.classes_frequency_dict = {}
        self.r_event_dict = {}
        
    
    def fit(self,X,y=None):
        return self
    
    def get_frequency_dict(self,X):
        
        lst_event_sequence = X['events_sequence'].tolist()

        frequency_dict = {}

        for lst in lst_event_sequence:

            for event_id in lst: 
                if event_id not in frequency_dict.keys():
                    frequency_dict[event_id] = 1
                else:
                    frequency_dict[event_id] += 1


        return frequency_dict
    

    def get_classes_frequency_dict(self,X):

        classes_frequency_dict = {}

        for incident_type in list(X['incident_type'].value_counts().keys()): 

            sub_lst_event_sequence = X[X['incident_type']==incident_type]['events_sequence'].tolist()   

            if incident_type not in classes_frequency_dict.keys():
                classes_frequency_dict[incident_type] = {}
            
            for lst in sub_lst_event_sequence:
                for event_id in lst: 
                    if event_id not in classes_frequency_dict[incident_type].keys():
                        classes_frequency_dict[incident_type][event_id] = 1
                    else:
                        classes_frequency_dict[incident_type][event_id] += 1


        return classes_frequency_dict
    

    def get_relevance(self):

        r_event_dict = {}

        for key, nested_dict in self.classes_frequency_dict.items():

            if key not in r_event_dict.keys():
                r_event_dict[key] = {}

            for event_id, value in nested_dict.items():
                r_event_dict[key][event_id] = value/self.frequency_dict[event_id]

        return r_event_dict
    

    def get_index_to_drop(self,x,incident_type,threshold=0.1):
        index_lst = []

        for i,value in enumerate(x):
            
            if value in self.r_event_dict[incident_type].keys():
                if self.r_event_dict[incident_type][value] > threshold:
                    index_lst.append(i)


        return index_lst
    
    def get_selected_data(seld,selected_index_lst,col_lst):

        if len(selected_index_lst)==0:
            return np.nan
        
        new_lst = [col_lst[i] for i in selected_index_lst]

        return new_lst
    

    
    def fit_transform(self,X,y=None):

        self.frequency_dict = self.get_frequency_dict(X)
        self.classes_frequency_dict = self.get_classes_frequency_dict(X)
        self.r_event_dict = self.get_relevance()

        joblib.dump(self.r_event_dict,config.R_EVENT_DICT_PATH)

        X['index_to_select'] = X[['incident_type','events_sequence']].apply(lambda row:self.get_index_to_drop(row['events_sequence'],row['incident_type']),axis=1)
        for col in config.SEQUENCE_COLUMNS:
            X[f'{col}'] = X[[col,'index_to_select']].apply(lambda row:self.get_selected_data(row['index_to_select'],row[col]),axis=1)



        X = X.dropna().reset_index(drop=True)
        X = X.drop(columns =['index_to_select'])

        return X 
    
    def transform(self,X,y=None):
        self.r_event_dict = joblib.load(config.R_EVENT_DICT_PATH)

        X['index_to_select'] = X[['incident_type','events_sequence']].apply(lambda row:self.get_index_to_drop(row['events_sequence'],row['incident_type']),axis=1)
        for col in config.SEQUENCE_COLUMNS:
            X[f'{col}'] = X[[col,'index_to_select']].apply(lambda row:self.get_selected_data(row['index_to_select'],row[col]),axis=1)
    

        X = X.dropna().reset_index(drop=True)
        X = X.drop(columns =['index_to_select'])

        return X 

class OutlierImputer(BaseEstimator, TransformerMixin):

    def __init__(self, lat_extreme, lon_extreme, n_neighbors=2, model_file=os.path.join(config.ENCODER_PATH,"knn_imputer.pkl")):
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
        return self
    
    def fit_transform(self,X,y=None):
        X = self.mark_outliers(X)
        features = X[["approx_lat", "approx_lon"]]
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.knn_imputer.fit(features)

        joblib.dump(self.knn_imputer, self.model_file)
        #we still need to fill in the outliers' fields even for train data
        X = X.groupby("incident_type").apply(self.apply_imputer)
        X = X.reset_index(drop=True)
        X = X.sort_values(by=X.columns[0]).reset_index(drop=True)

        return X

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
        pass

    def get_idx(self, ls):

        return_idx = len(ls)-1
           
        for idx in range(0, len(ls) - 1):
            if (int(ls[idx + 1]) > 0):
                return idx
            
        return return_idx

    def process_data(self, X):
        X["index_sequence"] = X["seconds_to_incident_sequence"].apply(self.get_idx)
        return X
    
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.process_data(X)

    def transform(self, X, y=None):
        return self.process_data(X)

class AddWindowIndicesModified(BaseEstimator, TransformerMixin):
    def __init__(self, window_start=-3600, window_end=0):
        self.window_start = window_start
        self.window_end = window_end

    def find_range_indexes(self,numbers, start_integer, end_integer):
        
        # Find the first valid index
        start_index = None
        for i in range(len(numbers)):
            if start_integer <= numbers[i] <= end_integer:
                start_index = i
                break
        
        # If no start index found, return None
        if start_index is None:
            return np.nan,np.nan
        
        # Find the last valid index
        end_index = start_index
        for j in range(start_index, len(numbers)):
            if start_integer <= numbers[j] <= end_integer:
                end_index = j
            else:
                break
        
        return int(start_index), int(end_index)

    def process_data(self, X):
        
        X[["window_min_idx", "window_max_idx"]] = X["seconds_to_incident_sequence"].apply(
            lambda x: self.find_range_indexes(x,self.window_start,self.window_end)
        ).apply(pd.Series)


        X = X.dropna(subset=["window_min_idx"]).reset_index(drop=True)

        return X
    
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.process_data(X)

    def transform(self, X, y=None):
        return self.process_data(X)




class AddWindowIndices(BaseEstimator, TransformerMixin):
    def __init__(self, window_start=3600, window_end=600):
        self.window_start = window_start
        self.window_end = window_end

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

    def process_data(self, X):
        X[["window_min_idx", "window_max_idx"]] = X[
            ["seconds_to_incident_sequence", "index_sequence"]
        ].apply(
            lambda row: self.get_window(row["seconds_to_incident_sequence"], row["index_sequence"]),
            axis=1,
        ).apply(pd.Series)
        return X
    
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.process_data(X)

    def transform(self, X, y=None):
        return self.process_data(X)

class AddWindowedSequences(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def get_data_sequence_within_windows(self, row, column):
        
        min_idx = int(row["window_min_idx"])
        max_idx = int(row["window_max_idx"])
        return row[column][min_idx : max_idx + 1]

    def process_data(self, X):
        for col in self.columns:
            X[f"window_{col}"] = X.apply(
                lambda row: self.get_data_sequence_within_windows(row, col), axis=1
            )
            
        return X
    
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        
        return self.process_data(X)

    def transform(self, X, y=None):
        return self.process_data(X)



data_cleaning_pipeline = Pipeline(steps=[
                   ("convert_sequences", ConvertSequencesToLists(sequence_columns=config.SEQUENCE_COLUMNS)),
                   ("remove_noise",RemoveNoise()),
                   ("impute_outlier",OutlierImputer(lat_extreme=[49.5072, 51.4978], lon_extreme=[2.5833, 6.3667], n_neighbors=2)),
                   ("find_index",AddIndexSequence()),
                   ("find_window_index",AddWindowIndices(window_start=3600, window_end=200)),
                   ("windowed_sequences_creation",AddWindowedSequences(columns=config.SEQUENCE_COLUMNS))
    ])

    