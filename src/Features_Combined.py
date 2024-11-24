#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

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
        
class DropRowsByLength(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X.drop(
            X[X['window_seconds_to_incident_sequence'].apply(lambda x: len(x) == 1 or len(x) == 0)].index, 
            inplace=True
        )
        
        return X

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
            ("drop_zero_length", DropRowsByLength()),
        ]
    )

    df_transformed = pipeline.fit_transform(df)
    output_file = "sncb_data_transformed.csv"
    df_transformed.to_csv(output_file, index=False, sep=";")
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    main()


# In[2]:


class AddDictionaryColumn(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_columns, output_column="dictionary_column"):
        self.sequence_columns = sequence_columns
        self.output_column = output_column

    def transform(self, X, y=None):
        X = X.copy()
        X[self.output_column] = X.apply(lambda row: make_dict(row), axis=1)
        return X

    def fit(self, X, y=None):
        return self


class CalculateMeanMedianSpeed(BaseEstimator, TransformerMixin):
    def __init__(self, dict_column="dictionary_column"):
        self.dict_column = dict_column

    def compute_mean_and_median_speed(self, data_dict):
        speeds = []
        for seconds_data in data_dict.values():
            for vehicle_data in seconds_data.values():
                for event_data in vehicle_data.values():
                    speeds.append(event_data["train_speed"])
        if speeds:
            return np.mean(speeds), np.median(speeds)
        else:
            return np.nan, np.nan

    def transform(self, X, y=None):
        X[["mean_train_speed", "median_train_speed"]] = X[self.dict_column].apply(
            self.compute_mean_and_median_speed
        ).apply(pd.Series)
        return X

    def fit(self, X, y=None):
        return self

class EventSequenceNGrams(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_column='window_events_sequence', max_ngram=3):
        self.sequence_column = sequence_column
        self.max_ngram = max_ngram
        self.vectorizers = {}
        self.top_ngrams = {}
    
    def prepare_sequences(self, sequences):
        return [' '.join(map(str, seq)) for seq in sequences]
    
    def fit(self, X, y=None):
        sequences = self.prepare_sequences(X[self.sequence_column])
        
        for n in range(1, self.max_ngram + 1):
            vectorizer = TfidfVectorizer(
                ngram_range=(n, n),
                lowercase=False, 
                token_pattern=r'(?u)\b\w+\b'  
            )
            self.vectorizers[n] = vectorizer
            self.vectorizers[n].fit(sequences)
            
        return self
    
    def get_top_ngrams(self, sequence, n):
        sequence_str = [' '.join(map(str, sequence))]        
        tfidf_matrix = self.vectorizers[n].transform(sequence_str)
        feature_names = self.vectorizers[n].get_feature_names_out()
        nonzero = tfidf_matrix.nonzero()
        scores = zip(nonzero[1], tfidf_matrix.data)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if sorted_scores:
            top_idx = sorted_scores[0][0]
            return feature_names[top_idx]
        return None
    
    def transform(self, X, y=None):
        X = X.copy()        
        for n in range(1, self.max_ngram + 1):
            X[f'top_{n}_gram'] = X[self.sequence_column].apply(
                lambda seq: self.get_top_ngrams(seq, n)
            )
        
        return X

class StatesCombinationCounter(BaseEstimator, TransformerMixin):
    def __init__(self, ac_column='window_dj_ac_state_sequence', dc_column='window_dj_dc_state_sequence'):
        self.ac_column = ac_column
        self.dc_column = dc_column
    
    def count_state_combinations(self, row):
        ac_states = row[self.ac_column]
        dc_states = row[self.dc_column]        
        counts = {
            'true_true_count': 0,
            'true_false_count': 0,
            'false_true_count': 0,
            'false_false_count': 0
        }
        for ac, dc in zip(ac_states, dc_states):
            if ac and dc:
                counts['true_true_count'] += 1
            elif ac and not dc:
                counts['true_false_count'] += 1
            elif not ac and dc:
                counts['false_true_count'] += 1
            else:
                counts['false_false_count'] += 1
        
        return pd.Series(counts)

    def transform(self, X, y=None):
        X = X.copy()
        combination_counts = X.apply(self.count_state_combinations, axis=1)
        
        for column in ['true_true_count', 'true_false_count', 'false_true_count', 'false_false_count']:
            X[f'ac_dc_{column}'] = combination_counts[column]
        
        return X

    def fit(self, X, y=None):
        return self

class MostFrequentIncidentLocationWithClustering(BaseEstimator, TransformerMixin):
    def __init__(self, lat_column='approx_lat', lon_column='approx_lon', incident_type_column='incident_type', eps=0.01, min_samples=5):
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.incident_type_column = incident_type_column
        self.eps = eps  
        self.min_samples = min_samples 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        lat_lon = X_copy[[self.lat_column, self.lon_column]].values

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='haversine')
        X_copy['cluster'] = db.fit_predict(np.radians(lat_lon))  # DBSCAN expects radians for geographical coordinates

        def get_most_frequent_cluster(group):
            most_frequent_cluster = group['cluster'].value_counts().idxmax()
            cluster_points = group[group['cluster'] == most_frequent_cluster]
            centroid_lat = np.mean(cluster_points[self.lat_column])
            centroid_lon = np.mean(cluster_points[self.lon_column])
            return centroid_lat, centroid_lon

        most_frequent_locations = X_copy.groupby(self.incident_type_column).apply(get_most_frequent_cluster)

        X_copy[['most_frequent_lat', 'most_frequent_lon']] = X_copy[self.incident_type_column].map(most_frequent_locations).apply(pd.Series)

        return X_copy

class MeanDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['mean_diff_wstis'] = X['window_seconds_to_incident_sequence'].apply(lambda x: np.diff(x).mean())
        return X

class MedianDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['median_diff_wstis'] = X['window_seconds_to_incident_sequence'].apply(lambda x: np.median(np.diff(x)))
        return X
        
class MaxDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def getaccs(self, seconds, speed):
        speed_accs = []
        
        for i in range(1,len(seconds)):
            speed_diff=(speed[i]-speed[i-1])*1000/3600
            time_diff=seconds[i]-seconds[i-1]
            if time_diff>0:
                speed_accs.append(speed_diff/time_diff)
        return speed_accs

    def max_ac_dc(self, seconds, speed):
        return max(self.getaccs(seconds, speed), key=abs)
        
    def transform(self, X, y=None):
        X['max_diff_ad'] = X.apply(lambda row: self.max_ac_dc(row['window_seconds_to_incident_sequence'],row['window_train_kph_sequence']),axis=1)
        return X

class IncidentCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def count_negatives(self,lst):
        return len(list(filter(lambda x: x < 0, lst)))
        
    def count_positives(self,lst):
        return len(list(filter(lambda x: x > 0, lst)))
        
    def transform(self, X, y=None):
        def categorize(row):
            positives = self.count_positives(row)
            negatives = self.count_negatives(row)
            
            if positives > negatives:
                return 'Positive'
            elif negatives > positives:
                return 'Negative'
            else:
                return 'Equal'
        #X['neg_count'] = X['window_seconds_to_incident_sequence'].apply(self.count_negatives)
        #X['pos_count'] = X['window_seconds_to_incident_sequence'].apply(self.count_positives)
        X['incident_category'] = X['window_seconds_to_incident_sequence'].apply(categorize)
        return X

class SpeedsFreq(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def getdiffs(self, seconds, speed):
        speed_diffs = []
        index_start = 0
        
        for i in range(1,len(seconds)):
            
            speed_diffs.append(speed[i]-speed[index_start])
            index_start=i
            
        if index_start < len(seconds) - 1:
            speed_diffs.append(speed[len(seconds) - 1]-speed[index_start])
        return speed_diffs

    def freq(self, speed_diffs):
        
        total_count=len(speed_diffs)
        count_pos=0
        count_neg=0
        count_const=0
        
        for diff in speed_diffs:
            if diff>0:
                count_pos+=1
            elif diff<0:
                count_neg+=1
            elif diff==0:
                count_const+=1
        
        lst=[count_pos/total_count*100, count_neg/total_count*100, count_const/total_count*100]
        return lst
        
    def transform(self, X, y=None):
        speed_diffs=X.apply(lambda row: self.getdiffs(row['window_seconds_to_incident_sequence'],row['window_train_kph_sequence']), axis=1)
                
        X['frequency_train_kph_acceleration']=speed_diffs.apply(lambda row: self.freq(row)[0])
        X['frequency_train_kph_deceleration']=speed_diffs.apply(lambda row: self.freq(row)[1])
        X['frequency_train_kph_constant']=speed_diffs.apply(lambda row: self.freq(row)[2])
        
        return X
        
class TenSecSpeeds(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def getcounts(self, seconds, speeds):

        start_elem = seconds[0]
        end_elem = start_elem+100
        prev=0
        
        count_acc=0
        count_dec=0
        count_total=0
        count_unknown=0
        
        if len(set(speeds)) == 1:
            count_constant = 1
            count_total = 1
            return [0.0, 0.0, 0.0, 1.0]
        
        for i in range(math.ceil((seconds[-1]-seconds[0])/100)):
            speed_list=[]
            for j in range(prev+1, len(seconds)):
                if prev==0:
                    speed_list.append(speeds[0])
                if seconds[j]>end_elem:
                    prev=j
                    break
                if start_elem<=seconds[j]<=end_elem:
                    if seconds[j]!=seconds[prev]:
                        speed_list.append(speeds[j])
                prev=j
                    
            if speed_list:
                mean=np.mean(speed_list)
                med=np.median(speed_list)
                count_total+=1
                
                if mean>med:
                    count_acc+=1
                elif mean<med:
                    count_dec+=1
                else:
                    count_unknown+=1

            start_elem=end_elem
            end_elem=start_elem+100

        if count_total > 0:
            freq_acc = count_acc / count_total
            freq_dec = count_dec / count_total
            freq_unknown = count_unknown / count_total
        else:
            freq_acc = freq_dec = freq_unknown = 0
        
        return [freq_acc,freq_dec,freq_unknown,0.0]
    
    def transform(self, X, y=None):
        X['tensec_freq_acc']=X.apply(lambda row: self.getcounts(row['window_seconds_to_incident_sequence'], row['window_train_kph_sequence'])[0], axis=1)
        X['tensec_freq_dec']=X.apply(lambda row: self.getcounts(row['window_seconds_to_incident_sequence'], row['window_train_kph_sequence'])[1], axis=1)
        X['tensec_freq_unknown']=X.apply(lambda row: self.getcounts(row['window_seconds_to_incident_sequence'], row['window_train_kph_sequence'])[2], axis=1)
        X['tensec_freq_const']=X.apply(lambda row: self.getcounts(row['window_seconds_to_incident_sequence'], row['window_train_kph_sequence'])[3], axis=1)
        
        return X


# In[5]:


def main():
    df = pd.read_csv("sncb_data_transformed.csv", delimiter=";")
    
    sequence_columns = [
            "vehicles_sequence",
            "events_sequence",
            "seconds_to_incident_sequence",
            "train_kph_sequence",
            "dj_ac_state_sequence",
            "dj_dc_state_sequence",
            "window_seconds_to_incident_sequence",
            "window_train_kph_sequence",
        ]
    
    pipeline = Pipeline(
            steps=[("convert_sequences", ConvertSequencesToLists(sequence_columns=sequence_columns)),
                    ("add_dictionary_and_mean_median_speed", Pipeline([
                    ("add_dictionary", AddDictionaryColumn(sequence_columns=sequence_columns)),
                    ("compute_mean_and_median_speed", CalculateMeanMedianSpeed(dict_column="dictionary_column")),
                    ])),
                    ("event_sequence_ngrams", EventSequenceNGrams(sequence_column="window_events_sequence", max_ngram=3)),
                    ("state_combinations", StatesCombinationCounter()),
                    ("most_frequent_event_location", MostFrequentIncidentLocationWithClustering()),
                   ("mean_time_difference", MeanDiffTransformer()),
                   ("median_time_difference", MedianDiffTransformer()),
                   ("max_kph_difference", MaxDiffTransformer()),
                   ("inc_categ", IncidentCategory()),
                   ("freq_speed_diffs", SpeedsFreq()),
                   ("window_speed_freqs", TenSecSpeeds()),
                   
            ]
    )
    df_transformed = pipeline.fit_transform(df)
    output_file = "sncb_data_with_features.csv"
    df_transformed.to_csv(output_file, index=False, sep=";")
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    main()


# In[ ]:




