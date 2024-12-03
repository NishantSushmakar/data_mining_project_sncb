import pandas as pd
import numpy as np

def remove_duplicates_and_near_duplicates(df, id_column='incident_id', similarity_threshold=0.95):
    
    # Step 1: Remove exact duplicates
    df_unique = df.drop_duplicates()
    
    # Step 2: Group by incident_id
    grouped = df_unique.groupby(id_column)
    
    # List to store rows to keep
    rows_to_keep = []
    
    # Iterate through each incident_id group
    for incident, group in grouped:
        # If only one row for this incident, keep it
        if len(group) == 1:
            rows_to_keep.append(group)
            continue
        
        # Compare rows within the group
        unique_rows = []
        for idx, row in group.iterrows():
            is_unique = True
            for existing_row in unique_rows:
                # Calculate similarity between rows
                similarity = calculate_row_similarity(row, existing_row)
                
                # If similarity is above threshold, consider it a duplicate
                if similarity >= similarity_threshold:
                    is_unique = False
                    break
            
            # If row is unique, add to unique rows
            if is_unique:
                unique_rows.append(row)
        
        # Convert unique rows to DataFrame and add to results
        if unique_rows:
            rows_to_keep.append(pd.DataFrame(unique_rows))
    
    # Combine the rows to keep
    return pd.concat(rows_to_keep, ignore_index=True)

def remove_cross_dataset_duplicates(train_df, val_df, id_column='incident_id', similarity_threshold=0.95):
    
    # Create a copy to avoid modifying original dataframes
    val_df_cleaned = val_df.copy()
    
    # Remove exact duplicates within validation dataset
    val_df_cleaned = val_df_cleaned.drop_duplicates()
    
    # List to track indices to drop
    indices_to_drop = []
    
    # Get unique incident IDs present in both training and validation datasets
    common_incident_ids = set(train_df[id_column]) & set(val_df_cleaned[id_column])
    
    # Iterate through common incident IDs
    for incident_id in common_incident_ids:
        # Get rows for this incident ID in training and validation datasets
        train_incident_rows = train_df[train_df[id_column] == incident_id]
        val_incident_rows = val_df_cleaned[val_df_cleaned[id_column] == incident_id]
        
        # If no rows found in either dataset, skip
        if len(train_incident_rows) == 0 or len(val_incident_rows) == 0:
            continue
        
        # Compare each validation row with training rows for the SAME incident ID
        for val_idx, val_row in val_incident_rows.iterrows():
            for train_idx, train_row in train_incident_rows.iterrows():
                # Calculate similarity between rows
                similarity = calculate_row_similarity(val_row, train_row)
                
                # If similarity is above threshold, mark for removal
                if similarity >= similarity_threshold:
                    indices_to_drop.append(val_idx)
                    break  # Stop comparing once a similar row is found
    
    # Remove similar rows
    val_df_cleaned = val_df_cleaned.drop(indices_to_drop)
    
    return val_df_cleaned


def calculate_row_similarity(row1, row2, method='jaccard'):
    
    # Remove the ID column for comparison
    r1 = row1.drop('incident_id') if 'incident_id' in row1.index else row1
    r2 = row2.drop('incident_id') if 'incident_id' in row2.index else row2
    
    if method == 'jaccard':
        # Jaccard similarity for mixed data types
        total_features = len(r1)
        matching_features = sum(r1 == r2)
        return matching_features / total_features
    
    elif method == 'cosine':
        # Cosine similarity for numerical features
        numeric_cols = r1.select_dtypes(include=[np.number]).index
        if len(numeric_cols) > 0:
            vec1 = r1[numeric_cols]
            vec2 = r2[numeric_cols]
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return 0
