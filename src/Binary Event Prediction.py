#!/usr/bin/env python
# coding: utf-8

# In[62]:


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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

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
        self.model_file = vectoriser_path 
    
    
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
        #X.drop(columns=COL_TO_DROP,inplace=True)

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
        #X.drop(columns=COL_TO_DROP,inplace=True)
        
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

    def get_math_acc(self, seconds, speed):

        
        speed_accs = []
        for i in range(1,len(seconds)):
            speed_diff=(speed[i]-speed[i-1])*1000/3600
            time_diff=seconds[i]-seconds[i-1]
            if time_diff>0:
                speed_accs.append(speed_diff/time_diff)
        return speed_accs

    def max_ac_dc(self, seconds, speed):
        acc_list = self.get_math_acc(seconds, speed)
    
        if not acc_list:
            return 0
            
        return max(acc_list, key=abs)
    
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
        X[["freq_tensec_acc","freq_tensec_dcc","freq_tensec_constant"]] = X[['unique_speed_list','unique_time_list']].apply(lambda row:self.get_count_n_seconds_window(row["unique_speed_list"],50,row["unique_time_list"]),axis=1).apply(pd.Series) 
        X['max_math_acc_dcc'] = X.apply(lambda row: self.max_ac_dc(row['window_seconds_to_incident_sequence'],row['window_train_kph_sequence']),axis=1).apply(pd.Series)
        X.drop(columns=["start_time","end_time","unique_speed_list","unique_time_list"],inplace=True)
        X.drop(columns=COL_TO_DROP,inplace=True)
        return X


    def fit_transform(self,X,y=None):
        return self.process_data(X)


    def transform(self,X,y=None):
        return self.process_data(X)


class AnomalyFeat(BaseEstimator,TransformerMixin):

    def __init___(self):
        return self
    
    def fit(self,X,y=None):
        return self
    
    def fit_transform(self,X,y=None):
        
        clf = IsolationForest(random_state=42)
        y_pred = clf.fit_predict(X)
        X['anomaly_detection_feat'] = (y_pred == -1).astype(int)
        
        joblib.dump(clf,"anomaly.pkl")
        return X 

    def transform(self,X,y=None):

        clf = joblib.load("anomaly.pkl")
        y_pred = clf.predict(X)
        X['anomaly_detection_feat'] = (y_pred == -1).astype(int)

        return X

    


# In[45]:


COL_TO_DROP = ['Unnamed: 0', 'vehicles_sequence', 'events_sequence',
       'seconds_to_incident_sequence', 'approx_lat', 'approx_lon',
       'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence',
        'window_vehicles_sequence',
        'window_events_sequence',
        'window_seconds_to_incident_sequence',
        'window_train_kph_sequence',
        'window_dj_ac_state_sequence',
        'window_dj_dc_state_sequence','sequence_dict','index_sequence','window_min_idx','window_max_idx',
        ]

SEQUENCE_COLUMNS = [
        "vehicles_sequence",
        "events_sequence",
        "seconds_to_incident_sequence",
        "train_kph_sequence",
        "dj_ac_state_sequence",
        "dj_dc_state_sequence",
    ]

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

# df = pd.read_pickle('main_dataset_incident_occured.pkl')
# print(df.shape)
# df = feature_pipeline.fit_transform(df)
# print(df.shape)
# df.to_csv("featured_data.csv",index=False)


# In[63]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def create_folds(df,n_folds):
    
    df['kfold'] = -1
    y = df.incident_occured
    
    kf = StratifiedKFold(n_splits=n_folds)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f
        
    return df

def run_folds(df,fold,model):
    
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    df_train = df_train.drop(columns=['kfold'])
    df_valid = df_valid.drop(columns=['kfold'])

    #data_transform_pipeline = Pipeline(feature_pipeline.steps)
    #df_train = data_transform_pipeline.fit_transform(df_train)
    #df_valid = data_transform_pipeline.transform(df_valid)

    df_train = feature_pipeline.fit_transform(df_train)
    df_valid = feature_pipeline.transform(df_valid)
    sample_df = feature_pipeline.transform(df)
    print(sample_df.shape)

    print("Unique Labels Training:" , df_train.incident_occured.nunique())
    print("Unique Labels Validation:" ,df_valid.incident_occured.nunique())
    print(df_train.shape)

    x_train = df_train.drop(columns=['incident_occured'],axis=1).values
    y_train = df_train.incident_occured.values
    
    x_valid = df_valid.drop(columns=['incident_occured'],axis=1).values
    y_valid = df_valid.incident_occured.values

    # x_test = test_df.drop(columns=['incident_type','incident_id'],axis=1).values
    # y_test = test_df.incident_type.values

    if model in ['mnb','lr']:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
    
    clf = models[model]

    # if model == 'lgbm_over':
    #     clf.fit(
    #         x_train, y_train,
    #         eval_set=[(x_valid, y_valid)],
    #         eval_metric='multi_logloss'
    #         )
    # else:
    clf.fit(x_train,y_train)
    joblib.dump(clf,f"{model}{fold}.pkl")
    
    y_train_pred = clf.predict(x_train)
    y_valid_pred = clf.predict(x_valid)
    # y_test_pred = clf.predict(x_test)
    
    train_precision = precision_score(y_train, y_train_pred)  
    train_recall = recall_score(y_train, y_train_pred)
    train_f1_score = f1_score(y_train, y_train_pred)
    train_conf_matrix=confusion_matrix(y_train, y_train_pred)


    valid_precision = precision_score(y_valid, y_valid_pred)  
    valid_recall = recall_score(y_valid, y_valid_pred)
    valid_f1_score = f1_score(y_valid, y_valid_pred)
    valid_conf_matrix=confusion_matrix(y_valid, y_valid_pred)


    print(f'Fold{fold}')
    print(f'Train F1 Score:{train_f1_score}, Train Precision:{train_precision}, Train Recall: {train_recall}, Confusion Matrix:\n {train_conf_matrix}')
    print(f'Validation F1 Score:{valid_f1_score}, Validation Precision:{valid_precision}, Validation Recall: {valid_recall},Confusion Matrix:\n {valid_conf_matrix}')
    # print(f'Test F1 Score:{test_f1_score}, Test Precision:{test_precision}, Test Recall: {test_recall}')
    print('*'*50)
    
    
    return train_precision, train_recall, train_f1_score , train_conf_matrix, valid_precision, valid_recall, valid_f1_score, valid_conf_matrix


# In[64]:


def train_model(df, model_name,n_folds):

    # df = pd.read_pickle(config.MAIN_DATA_PATH)
    # test_df = pd.read_pickle(config.TEST_DATA_PATH)
    # df = pd.read_pickle(config.MAIN_DATA_PATH)

    df = create_folds(df,n_folds)
    
    train_p = []
    train_r = []
    train_f1 = []
    train_cm = np.zeros((2,2))
    val_p = []
    val_r = []
    val_f1 = []
    val_cm = np.zeros((2,2))
    
    for i in range(n_folds):

        tp, tr, tf1, tcm, vp, vr, vf1, vcm = run_folds(df,i,model_name)
        train_p.append(tp)
        train_r.append(tr)
        train_f1.append(tf1)
        train_cm += tcm
        val_p.append(vp)
        val_r.append(vr)
        val_f1.append(vf1)
        val_cm += vcm
    
    print("*"*15,"Final Summary","*"*15)

    print(f"average training precision:{np.array(train_p).mean()}")
    print(f"average training recall:{np.array(train_r).mean()}")
    print(f"average training f1:{np.array(train_f1).mean()}")
    print("\noverall training confusion matrix:\n")
    print(train_cm)
    #train_ov_cm_disp = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=data.target_names)
    #train_ov_cm_disp.plot()

    train_cm_norm = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
    #train_norm_cm_disp = ConfusionMatrixDisplay(confusion_matrix=train_cm_norm, display_labels=data.target_names)
    #train_norm_cm_disp.plot()
    print("\nnormalized training confusion matrix:\n")
    print(train_cm_norm)


    print(f"average validation precision:{np.array(val_p).mean()}")
    print(f"average validation recall:{np.array(val_r).mean()}")
    print(f"average validation f1:{np.array(val_f1).mean()}")
    print("\noverall validation confusion matrix: \n")
    print(val_cm)
    #val_ov_cm_disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=data.target_names)
    #val_ov_cm_disp.plot()

    val_cm_norm = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]
    #val_norm_cm_disp = ConfusionMatrixDisplay(confusion_matrix=val_cm_norm, display_labels=data.target_names)
    #val_norm_cm_disp.plot()
    print("\nnormalized validation confusion matrix: \n")
    print(val_cm_norm)


# In[65]:


from sklearn.model_selection import train_test_split

df = pd.read_pickle('main_dataset_incident_occured.pkl')
X = df.drop(columns=['incident_occured'],axis=1)
y = df['incident_occured']
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# train_data = X_train_val.copy()
# train_data['incident_occured'] = y_train_val
# test_data = X_test.copy()
# test_data['incident_occured'] = y_test
# train_data = train_data.dropna().reset_index(drop=True)
# test_data = test_data.dropna().reset_index(drop=True)


# In[86]:


from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


models = {
    # 'mnb':MultinomialNB(),
    # 'gnb':GaussianNB(),
    # 'rf':RandomForestClassifier(n_estimators=1000),
    'lgbm':lgb.LGBMClassifier(n_estimator=10000, verbose=-1, feature_name=X.columns.tolist())
    # 'xgb': XGBClassifier(n_estimators=100, learning_rate=0.1)
    # 'lr': LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=1000)
    
}

''' lgbm_over': lgb.LGBMClassifier(
                # Regularization parameters
                n_estimators=1000,  # High number of trees
                learning_rate=0.01,  # Lower learning rate to prevent overfitting
                
                # Regularization techniques
                regularization_factor=0.1,  # L2 regularization
                max_depth=7,  # Limit tree depth
                min_child_samples=20,  # Minimum number of samples in leaf
                
                # Prevent overfitting
                early_stopping_rounds=50,  # Stop if no improvement
                subsample=0.8,  # Take 80% of data for each tree
                colsample_bytree=0.8,  # Take 80% of features for each tree
                
                # Reduce model complexity
                num_leaves=31,  # Limit number of leaves
                
                # Prevent learning noise
                feature_fraction=0.7,  # Randomly select features
                bagging_fraction=0.7,  # Randomly select data
                bagging_freq=1,  # How often to perform bagging
                
                # Reduce verbosity
                verbose=-1
                ) '''


# In[67]:


df['incident_occured'].value_counts()


# In[87]:


keys = models.keys()
keys_list = list(keys)

for model in keys_list:
    print(f"--------------- MODEL : {model} -----------------")
    train_model(df, model, 4) 


# In[15]:


72,5 rf 71,3
73,2 lgbm 71,8
72,9 xgb 72,5


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt

copy_df=df[['incident_occured',
       'unique_vehicle_sequence',
       'unique_events_sequence', 'unique_ac_state_sequence',
       'unique_dc_state_sequence',
       'highest_frequency_percentage_vehicle_sequence',
       'highest_frequency_percentage_events_sequence', 'mean_train_speed', 'median_train_speed','ac_dc_true_true_count',
        'ac_dc_true_false_count','ac_dc_false_true_count','ac_dc_false_false_count',
       'mean_diff_wstis', 'median_diff_wstis', 'incident_category',
       'frequency_train_kph_acceleration', 'frequency_train_kph_deceleration',
       'frequency_train_kph_constant', 'max_acceleration_and_deceleration',
       'freq_tensec_acc', 'freq_tensec_dcc', 'freq_tensec_constant','max_math_acc_dcc']].copy()

corr_matrix = copy_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 5})
plt.figure(figsize=(20, 10)) 
plt.show()


# In[104]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_top_20_feature_importance(model, feature_names_from_model, feature_names_to_include):
    # Get feature importance values from the model
    importance_values = model.feature_importances_

    # Get all feature names used in the model (from the model's booster)
    # feature_names_from_model = model.booster_.feature_name()

    # # Ensure the number of feature names matches the number of importance values
    # if len(feature_names_from_model) != len(importance_values):
    #     raise ValueError("The number of feature names does not match the number of features in the model.")

    # Create importance DataFrame with model's features and importance values
    importance_df = pd.DataFrame({
        'feature': feature_names_from_model,
        'importance': importance_values
    })
    print("Feature names from model:", feature_names_from_model)
    print("Feature names to include:", feature_names_to_include)

    # If `feature_names_to_include` is provided, filter the DataFrame to include only those features
    if feature_names_to_include:
        importance_df = importance_df[importance_df['feature'].isin(feature_names_to_include)]

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Select top 20 features (after filtering if necessary)
    top_20_features = importance_df.head(20)

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_20_features, palette='viridis')
    plt.title(f'Top 20 Features by Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    # Print top 20 features
    print(top_20_features)

    return top_20_features


# In[112]:


def get_top_20_feature_importance(model,feature_names):
    # Get feature importance values
    importance_values = model.feature_importances_
    #feature_names = model.booster_.feature_name()
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    # Select top 20 features
    top_20_features = importance_df.head(20)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_20_features, palette='viridis')
    plt.title(f'Top 20 Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Print top 20 features
    print(top_20_features)
    
    
    return


# In[113]:


model=joblib.load("lgbm2.pkl")
get_top_20_feature_importance(model,X.columns.tolist())
       #                        X.columns.tolist(), ['unique_vehicle_sequence',
       # 'unique_events_sequence', 'unique_ac_state_sequence',
       # 'unique_dc_state_sequence',
       # 'highest_frequency_percentage_vehicle_sequence',
       # 'highest_frequency_percentage_events_sequence', 'mean_train_speed', 'median_train_speed','ac_dc_true_true_count',
       #  'ac_dc_true_false_count','ac_dc_false_true_count','ac_dc_false_false_count',
       # 'mean_diff_wstis', 'median_diff_wstis', 'incident_category',
       # 'frequency_train_kph_acceleration', 'frequency_train_kph_deceleration',
       # 'frequency_train_kph_constant', 'max_acceleration_and_deceleration',
       # 'freq_tensec_acc', 'freq_tensec_dcc', 'freq_tensec_constant','max_math_acc_dcc'])


# In[ ]:


based on collinearity: remove median_speed, ac_dc_true_false, freq_tensec_acc, freq_tensec_dcc, frequency_train_kph_decelarition

