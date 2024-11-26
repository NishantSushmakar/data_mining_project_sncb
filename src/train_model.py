import data_cleaning
import feature_creation
import config
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from model_dispatcher import models
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE,RandomOverSampler
import numpy as np



def create_folds(df,n_folds):
    
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.incident_type
    
    kf = StratifiedKFold(n_splits=n_folds)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f
        
    return df

def run_folds(df,fold,model):
    
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    data_transform_pipeline = Pipeline(data_cleaning.data_cleaning_pipeline.steps + feature_creation.feature_pipeline.steps)
    df_train = data_transform_pipeline.fit_transform(df_train)
    df_valid = data_transform_pipeline.transform(df_valid)

    
    x_train = df_train.drop(columns=['incident_type','kfold'],axis=1).values
    y_train = df_train.incident_type.values
    
    x_valid = df_valid.drop(columns=['incident_type','kfold'],axis=1).values
    y_valid = df_valid.incident_type.values

    # smote = SMOTE(k_neighbors=3,random_state=42)
    # x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    smote = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    # x_train_smote, y_train_smote = x_train,y_train
   
    
    clf = models[model]
    
    clf.fit(x_train_smote,y_train_smote)
    y_train_pred = clf.predict(x_train_smote)
    y_valid_pred = clf.predict(x_valid)
    


    train_precision = precision_score(y_train_smote, y_train_pred, average='macro')  
    train_recall = recall_score(y_train_smote, y_train_pred, average='macro')
    train_f1_score = f1_score(y_train_smote, y_train_pred, average='macro')


    valid_precision = precision_score(y_valid, y_valid_pred, average='macro')  
    valid_recall = recall_score(y_valid, y_valid_pred, average='macro')
    valid_f1_score = f1_score(y_valid, y_valid_pred, average='macro')



    print(f'Fold{fold}')
    print(f'Train F1 Score:{train_f1_score}, Train Precision:{train_precision}, Train Recall: {train_recall}')
    print(f'Validation F1 Score:{valid_f1_score}, Validation Precision:{valid_precision}, Validation Recall: {valid_recall}')
    print('*'*50)
    
    
    return train_precision, train_recall, train_f1_score , valid_precision, valid_recall, valid_f1_score



def train_model(model_name,n_folds):

    df = pd.read_csv(config.DATA_PATH,sep=";")
    
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

    train_model("lgbm",10)




