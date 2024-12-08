from data_cleaning import *
from feature_creation import *
import config
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from model_dispatcher import models
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE,RandomOverSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from remove_data_leakage import *


def create_folds(df,n_folds):
    
    df['kfold'] = -1
    y = df.incident_type
    
    kf = StratifiedKFold(n_splits=n_folds)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f
        
    return df

# def remove_data_leakage(train_df,valid_df):

#     train_df = train_df.drop_duplicates(keep='first').reset_index(drop=True)
#     valid_df = valid_df.drop_duplicates(keep='first').reset_index(drop=True)

#     valid_df = valid_df[~valid_df.apply(tuple, axis=1).isin(train_df.apply(tuple, axis=1))].reset_index(drop=True)

#     return train_df,valid_df

def data_augmentation(df):

    
    
    df_lst = []

    for start_time in range(0, -10801, -600):
        
        end_time = start_time + 3600

        
        print(start_time,"-",end_time)


        data_cleaning_pipeline_custom = Pipeline(steps=[
                ("convert_sequences", ConvertSequencesToLists(sequence_columns=config.SEQUENCE_COLUMNS)),
                ("remove_noise",RemoveNoise()),
                ("impute_outlier",OutlierImputer(lat_extreme=[49.5072, 51.4978], lon_extreme=[2.5833, 6.3667], n_neighbors=2)),
                ("find_index",AddIndexSequence()),
                ("find_window_index",AddWindowIndicesModified(window_start=start_time, window_end=end_time)),
                ("windowed_sequences_creation",AddWindowedSequences(columns=config.SEQUENCE_COLUMNS))
                ])
        
        

        transform_df = data_cleaning_pipeline_custom.fit_transform(df.copy())

        df_lst.append(transform_df)

    final_df = pd.concat(df_lst,axis=0).reset_index(drop=True)
    final_df['len_window'] = final_df['window_seconds_to_incident_sequence'].apply(lambda x:len(x))
    final_df['combination'] = final_df[['incident_id','len_window']].apply(lambda row:str(row['incident_id'])+str(row['len_window']),axis=1)
    final_df = final_df.drop_duplicates(subset = ['combination'],keep='first').reset_index(drop=True)
    final_df.drop(columns=['len_window','combination'],inplace=True)

    return final_df





def run_folds(df,fold,model):
    
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    df_train = df_train.drop(columns=['kfold'])
    df_valid = df_valid.drop(columns=['kfold'])

    data_transform_pipeline = Pipeline(data_cleaning_pipeline.steps + feature_pipeline.steps)
    df_train = data_transform_pipeline.fit_transform(df_train)
    df_valid = data_transform_pipeline.transform(df_valid)

    sample_df = data_transform_pipeline.transform(df)
    print(sample_df.shape)

    # df_train = data_augmentation(df_train)
    # df_valid = data_augmentation(df_valid)
    

    # df_train = feature_pipeline.fit_transform(df_train)
    # df_valid = feature_pipeline.transform(df_valid)
    # test_df = feature_creation.feature_pipeline.transform(test_df)

    # print(df_train.shape,df_valid.shape)
    # df_train,df_valid = remove_data_leakage(df_train,df_valid)    
    # df_valid = remove_cross_dataset_duplicates(train_df=df_train,val_df=df_valid)
    # print(df_train.shape,df_valid.shape)
    print("Unique Labels Training:" , df_train.incident_type.nunique())

    print("Unique Labels Validation:" ,df_valid.incident_type.nunique())
    print(df_train.shape)

    x_train = df_train.drop(columns=['incident_type','incident_id'],axis=1).values
    y_train = df_train.incident_type.values
    
    x_valid = df_valid.drop(columns=['incident_type','incident_id'],axis=1).values
    y_valid = df_valid.incident_type.values

    # x_test = test_df.drop(columns=['incident_type','incident_id'],axis=1).values
    # y_test = test_df.incident_type.values

    # if model == 'xgb':
    #     le = LabelEncoder()
    #     y_train = le.fit_transform(y_train)
    #     y_valid = le.transform(y_valid)

    # if model in ['mnb','lr']:
    #     scaler = MinMaxScaler()
    #     x_train = scaler.fit_transform(x_train)
    #     x_valid = scaler.transform(x_valid)
    
    clf = models[model]

    # if model == 'lgbm_over':
    #     clf.fit(
    #         x_train, y_train,
    #         eval_set=[(x_valid, y_valid)],
    #         eval_metric='multi_logloss'
    #         )
    # else:
    
    clf.fit(x_train,y_train)

    y_train_pred = clf.predict(x_train)
    y_valid_pred = clf.predict(x_valid)
    # y_test_pred = clf.predict(x_test)
    
    train_precision = precision_score(y_train, y_train_pred, average='macro')  
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    train_f1_score = f1_score(y_train, y_train_pred, average='macro')


    valid_precision = precision_score(y_valid, y_valid_pred, average='macro')  
    valid_recall = recall_score(y_valid, y_valid_pred, average='macro')
    valid_f1_score = f1_score(y_valid, y_valid_pred, average='macro')

    # test_precision = precision_score(y_test, y_test_pred, average='macro')  
    # test_recall = recall_score(y_test, y_test_pred, average='macro')
    # test_f1_score = f1_score(y_test, y_test_pred, average='macro')




    print(f'Fold{fold}')
    print(f'Train F1 Score:{train_f1_score}, Train Precision:{train_precision}, Train Recall: {train_recall}')
    print(f'Validation F1 Score:{valid_f1_score}, Validation Precision:{valid_precision}, Validation Recall: {valid_recall}')
    # print(f'Test F1 Score:{test_f1_score}, Test Precision:{test_precision}, Test Recall: {test_recall}')
    print('*'*50)
    
    
    return train_precision, train_recall, train_f1_score , valid_precision, valid_recall, valid_f1_score



def train_model(model_name,n_folds):

    # df = pd.read_pickle(config.MAIN_DATA_PATH)
    # test_df = pd.read_pickle(config.TEST_DATA_PATH)

    df = pd.read_csv(config.DATA_PATH,sep=";")

    # df = pd.read_pickle(config.MAIN_DATA_PATH)
    print(df.shape)

    df = create_folds(df,n_folds)
    
    train_p = []
    train_r = []
    train_f1 = []
    val_p = []
    val_r = []
    val_f1 = []
    
    for i in range(n_folds):

        tp, tr, tf1, vp, vr, vf1 = run_folds(df,i,model_name)
        train_p.append(tp)
        train_r.append(tr)
        train_f1.append(tf1)
        val_p.append(vp)
        val_r.append(vr)
        val_f1.append(vf1)
    
    
    

    print("*"*15,"Final Summary","*"*15)

    print(f"average training precision:{np.array(train_p).mean()}")
    print(f"average training recall:{np.array(train_r).mean()}")
    print(f"average training f1:{np.array(train_f1).mean()}")


    print(f"average validation precision:{np.array(val_p).mean()}")
    print(f"average validation recall:{np.array(val_r).mean()}")
    print(f"average validation f1:{np.array(val_f1).mean()}")
   
    

    


if __name__ == "__main__":

    train_model("lgbm",4) 




