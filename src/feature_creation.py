import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
# import data_cleaning
import config
import numpy as np
import os
import joblib


class DropRowsByLength(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def process_data(self,X):
        
        X.drop(X[X['window_seconds_to_incident_sequence'].apply(lambda x: len(x) == 0)].index, inplace=True)
        return X

    def fit_transform(self,X,y=None):
        return self.process_data(X)
    
    def transform(self, X,y=None):
        return self.process_data(X)
        
        
class UniqueAndHighestFrequencyPercentageSequences(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def process_data(self, X):
        X["unique_vehicle_sequence"] = X["window_vehicles_sequence"].apply(lambda x: len(set(x)))
        X["unique_events_sequence"] = X["window_events_sequence"].apply(lambda x: len(set(x)))
        X["unique_ac_state_sequence"] = X["window_dj_ac_state_sequence"].apply(lambda x: len(set(x)))
        X["unique_dc_state_sequence"] = X["window_dj_dc_state_sequence"].apply(lambda x: len(set(x)))
        X["highest_frequency_percentage_vehicle_sequence"] = X["window_vehicles_sequence"].apply(lambda x: max(x.count(vehicle) / len(x) for vehicle in set(x)))
        X["highest_frequency_percentage_events_sequence"] = X["window_events_sequence"].apply(lambda x: max(x.count(event) / len(x) for event in set(x)))
        return X
    
    def fit(self,X,y=None):
        return self

    def fit_transform(self, X, y=None):
    
        return self.process_data(X)

    def transform(self, X, y=None):
        return self.process_data(X)
    

class CalculateMeanMedianSpeed(BaseEstimator, TransformerMixin):

    def __init__(self, dict_column="sequence_dict"):
        self.dict_column = dict_column

    def fit(self, X, y=None):
        return self
    
    def compute_mean_and_median_speed(self, data_dict):
        
        
        speeds = []
        for seconds_data in data_dict.values():
            for vehicle_data in seconds_data.values():
                for event_data in vehicle_data.values():
                    speeds.append(event_data["train_speed"])
                    break
                break

        
        if len(speeds)>0:
            return np.mean(speeds), np.median(speeds)
        else:
            return np.nan, np.nan

    def get_mean_and_median_speed(self,X):
        X[["mean_train_speed", "median_train_speed"]] = X[self.dict_column].apply(
            self.compute_mean_and_median_speed
        ).apply(pd.Series)
        return X

    def fit_transform(self,X,y=None):
        
        return self.get_mean_and_median_speed(X)

    def transform(self, X, y=None):
        
        return self.get_mean_and_median_speed(X)
    
class StatesCombinationCounter(BaseEstimator, TransformerMixin):
    def __init__(self, ac_column='window_dj_ac_state_sequence', dc_column='window_dj_dc_state_sequence'):
        self.ac_column = ac_column
        self.dc_column = dc_column

    def fit(self, X, y=None):
        return self
    
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
            elif dc and not ac:
                counts['false_true_count'] += 1
            else:
                counts['false_false_count'] += 1
        
        return counts['true_true_count'],counts['true_false_count'],counts['false_true_count'],counts['false_false_count']
    
    def get_combination_counts(self,X):

        X[['ac_dc_true_true_count', 'ac_dc_true_false_count', 'ac_dc_false_true_count', 'ac_dc_false_false_count']]= \
        X.apply(lambda row:self.count_state_combinations(row),axis=1).apply(pd.Series)
        
        return X

    def transform(self, X, y=None):
        return self.get_combination_counts(X)
        

    def fit_transform(self, X, y=None):
        return self.get_combination_counts(X)

    
class EventSequenceNGrams(BaseEstimator, TransformerMixin):

    def __init__(self, sequence_column='window_events_sequence', max_ngram=3,vectoriser_path="tfidf_vectoriser.pkl"):
        self.max_ngram = max_ngram  
        self.model_file = os.path.join(config.ENCODER_PATH,vectoriser_path)  
    
    
    def fit(self, X, y=None):
        return self
    
    
    def fit_transform(self, X, y=None):
        
        X = X.reset_index(drop=True)
        X['cnv_window_event_sequences'] = X['window_events_sequence'].apply(lambda x:" ".join(str(id) for id in x ))
        vectorizer = TfidfVectorizer(ngram_range=(1,self.max_ngram))
        arr = vectorizer.fit_transform(X['cnv_window_event_sequences'].to_list())
        tffidf_feature = pd.DataFrame(arr.toarray(),columns=vectorizer.get_feature_names_out())
        X = pd.concat([X,tffidf_feature],axis=1)
        X.drop(columns=['cnv_window_event_sequences'],inplace=True)

        joblib.dump(vectorizer,self.model_file)
        return X
    
    def transform(self, X, y=None):
        vectorizer = joblib.load(self.model_file)

        X = X.reset_index(drop=True)
        X['cnv_window_event_sequences'] = X['window_events_sequence'].apply(lambda x:" ".join(str(id) for id in x ))
        arr = vectorizer.transform(X['cnv_window_event_sequences'].to_list())
        tffidf_feature = pd.DataFrame(arr.toarray(),columns=vectorizer.get_feature_names_out())
        X = pd.concat([X,tffidf_feature],axis=1)
        X.drop(columns=['cnv_window_event_sequences'],inplace=True)
        
        return X
    

class MeanMedianDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def process_data(self,X):
        X['mean_diff_wstis'] = X['sequence_dict'].apply(lambda x: np.diff(list(x.keys())).mean())
        X['median_diff_wstis'] = X['sequence_dict'].apply(lambda x: np.median(np.diff(list(x.keys()))))
        return X

    def fit_transform(self, X, y=None):
        return self.process_data(X)
    
    
    def transform(self,X,y=None):
        return self.process_data(X)
    

class IncidentCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def count_negatives(self,lst):
        return len(list(filter(lambda x: x < 0, lst)))
        
    def count_positives(self,lst):
        return len(list(filter(lambda x: x > 0, lst)))
    
    def categorize(self,row):
            positives = self.count_positives(row)
            negatives = self.count_negatives(row)
            
            if positives > negatives:
                return 1
            elif negatives > positives:
                return -1
            else:
                return 0
        
    def process_data(self, X, y=None):
        X['incident_category'] = X['window_seconds_to_incident_sequence'].apply(self.categorize)
        return X 
    
    def fit_transform(self,X,y=None):
        return self.process_data(X)

    def transform(self,X,y=None):
        return self.process_data(X)




class SpeedsFreq(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def form_speed_lst(self,value_dict,start,end):

        speed_lst = []
        time_lst = []

        for i in range(start,end+1):

            if i in value_dict.keys():
                time_lst.append(i)
                for item in value_dict[i].values():
                    for i in item.values():
                        speed_lst.append(i['train_speed'])
                        break
                    break
        
        return speed_lst,time_lst
    
    def get_count(self,lst):

        diff_list = list(np.diff(np.array(lst)))

        positive_count = 0
        constant_count = 0
        negative_count = 0 


        if len(diff_list)<=1:
            return 0,0, 1


        for i in range(len(diff_list)):

            if diff_list[i]>0:
                positive_count += 1
            elif diff_list[i]<0:
                negative_count += 1
            else:
                constant_count += 1

        return positive_count, negative_count,constant_count

    def get_acc(self,lst):

        if len(lst)<=1:
            return 0
        
        max_acc = -9999
        max_dcc = 9999

        for i in range(1,len(lst)):
            
            if lst[i-1]!=0:     
                diff = (lst[i] - lst[i-1])/lst[i-1]
            else:
                diff = (lst[i] - lst[i-1])/(lst[i-1]+1)

            if diff > max_acc:
                max_acc = diff
            
            if diff < max_dcc:
                max_dcc = diff

        if abs(max_acc) > abs(max_dcc):
            return max_acc

        return max_dcc 
    
    def get_count_n_seconds_window(self,speed_lst,n,time_lst):
        


        speed_arr = np.array(speed_lst)
        time_arr = np.array(time_lst)
        
        # Use numpy's vectorized operations
        positive_count = 0
        negative_count = 0
        constant_count = 0
        
        # Iterate through time windows efficiently
        i = 0
        while i < len(time_arr):
            # Find the end of the current time window
            end_time = time_arr[i] + n
            
            # Use boolean indexing to find elements in the current window
            window_mask = (time_arr >= time_arr[i]) & (time_arr < end_time)
            window_speeds = speed_arr[window_mask]
            
            # Skip empty windows
            if len(window_speeds) > 0:
                # Use numpy's mean and median calculations
                mean_speed = np.mean(window_speeds)
                median_speed = np.median(window_speeds)
                
                # Count distribution characteristics
                if mean_speed > median_speed:
                    positive_count += 1
                elif mean_speed < median_speed:
                    negative_count += 1
                else:
                    constant_count += 1
            
            # Move to the next window
            i += np.sum(window_mask)
        
        return positive_count, negative_count, constant_count

    
    def process_data(self,X):

        X[["start_time","end_time"]] = X['window_seconds_to_incident_sequence'].apply(lambda lst: (lst[0], lst[-1])).apply(pd.Series)
        X[["unique_speed_list","unique_time_list"]] = X[['sequence_dict','start_time','end_time']].apply(lambda row: self.form_speed_lst(row['sequence_dict'],row['start_time'],row['end_time']),axis=1).apply(pd.Series)
        X[['frequency_train_kph_acceleration','frequency_train_kph_deceleration','frequency_train_kph_constant']]=X['unique_speed_list'].apply(self.get_count).apply(pd.Series)
        X["max_acceleration_and_deceleration"] = X['unique_speed_list'].apply(lambda x:self.get_acc(x))
        X[["freq_tensec_acc","freq_tensec_dcc","freq_tensec_constant"]] = X[['unique_speed_list','unique_time_list']].apply(lambda row:self.get_count_n_seconds_window(row["unique_speed_list"],10,row["unique_time_list"]),axis=1).apply(pd.Series) 

        X.drop(columns=["start_time","end_time","unique_speed_list","unique_time_list"],inplace=True)
        X.drop(columns=config.COL_TO_DROP,inplace=True)
        return X


    def fit_transform(self,X,y=None):
        return self.process_data(X)


    def transform(self,X,y=None):
        return self.process_data(X)


feature_pipeline = Pipeline(steps=[
        ("drop_empty_rows",DropRowsByLength()),
        ("sequence_stats",UniqueAndHighestFrequencyPercentageSequences()),
        ("sequence_mean_and_median_speed",CalculateMeanMedianSpeed()),
        ("energy_state_combination",StatesCombinationCounter()),
        ("tfidf_features",EventSequenceNGrams()),
        ("mean_median_between_seconds",MeanMedianDiffTransformer()),
        ("incident_category",IncidentCategory()),
        ("speed_freq",SpeedsFreq())
  ])



# if __name__ == "__main__":
  
#   df = pd.read_csv(config.DATA_PATH,sep=";")
#   print(df.shape)
#   print("data cleaning started")
#   df = data_cleaning.data_cleaning_pipeline.fit_transform(df)
#   print("data cleaning done")

#   feature_pipeline = Pipeline(steps=[
#         ("drop_empty_rows",DropRowsByLength()),
#         ("sequence_stats",UniqueAndHighestFrequencyPercentageSequences()),
#         ("sequence_mean_and_median_speed",CalculateMeanMedianSpeed()),
#         ("energy_state_combination",StatesCombinationCounter()),
#         ("tfidf_features",EventSequenceNGrams()),
#         ("mean_median_between_seconds",MeanMedianDiffTransformer()),
#         ("incident_category",IncidentCategory()),
#         ("speed_freq",SpeedsFreq())
#   ])

  
#   df = feature_pipeline.fit_transform(df)
#   df.drop(columns=config.COL_TO_DROP,inplace=True)
#   print(df.columns)

  
#   df.to_csv("/Users/nishantsushmakar/Documents/projects_ulb/data-mining/data/data.csv",index=False)
#   print(df.shape)
#   print(df.head())
    

