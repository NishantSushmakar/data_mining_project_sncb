from data_cleaning import *
from feature_creation import *
import config
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np



if __name__ == "__main__":

    df = pd.read_csv(config.DATA_PATH,sep=";")


    print(type(df.loc[0,'vehicles_sequence'][0]))


    df_lst = []

    for start_time in range(0, -14401, -3600):
        
        end_time = start_time + 3600

        
        print(start_time,"-",end_time)


        data_cleaning_pipeline_custom = Pipeline(steps=[
                ("convert_sequences", ConvertSequencesToLists(sequence_columns=config.SEQUENCE_COLUMNS)),
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

    
    
    print(final_df.shape)
    print(final_df.incident_type.value_counts())

    final_df.to_pickle(config.MAIN_DATA_PATH)

