from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def undersample(df, ratio, args) : 
    patient_df = df[['patient_num', 'gender', 'age', 'label']].drop_duplicates().reset_index(drop=True)
    group0_df = patient_df.query('label == 0')
    group1_df = patient_df.query('label == 1')
    
    group1_age_sex_dist = group1_df.groupby(['gender', 'age']).count().reset_index().sort_values(by = ['age', 'gender'])
    group1_age_sex_dist['patient_num'] = group1_age_sex_dist['patient_num'] * ratio
    
    undersampled_mild_df, undersampled_mild_patient_ids = [], []
    for gender, age, severe_patient_num, _ in group1_age_sex_dist.values : 
        mild_df = group0_df.query('patient_num not in @undersampled_mild_patient_ids').reset_index(drop = True)
        mild_age_sex_dist = mild_df.groupby(['gender','age'])[['patient_num']].nunique().reset_index().sort_values(by = ['age', 'gender'])
        
        mild_patient_num, age_diff = 0, 0
        while mild_patient_num < severe_patient_num : # loop until the number of selected TD patients is greater than or equal to the number of ADHD patients
            mild_patient_num = mild_age_sex_dist.query(f'gender == @gender and {age - age_diff} <= age <= {age + age_diff}').patient_num.sum()
            if age + age_diff - 10 > patient_df['age'].max() : 
                return None, False
            age_diff += 1
        age_diff -= 1
        matched_df = mild_df.query(f"gender == @gender and {age - age_diff} <= age <= {age + age_diff} and patient_num not in @undersampled_mild_patient_ids")
        matched_patient_df = matched_df.drop_duplicates().reset_index(drop = True)
        selected_mild_patient_ids = matched_patient_df.sample(n = severe_patient_num, random_state = args.seed, replace = False).patient_num.values
        undersampled_mild_patient_ids.extend(selected_mild_patient_ids)

        if len(selected_mild_patient_ids) > 0 :
            for selected_mild_patient_id in selected_mild_patient_ids : 
                selected_mild_df = group0_df.query('patient_num == @selected_mild_patient_id')
                undersampled_mild_df.append(selected_mild_df)
    undersampled_mild_df = pd.concat(undersampled_mild_df)
    
    severe_df = df.query('label == 1')
    mild_df = df.query('label == 0 and patient_num in @undersampled_mild_patient_ids')
    df = pd.concat([severe_df, mild_df], axis = 0).reset_index(drop = True)
    return df, True

def fold_split(df, args) : 
    # 6 : 2 : 2
    patient_df = df[['patient_num', 'label']].drop_duplicates().reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits = args.num_folds,  random_state = args.seed, shuffle = True)
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(patient_df['patient_num'], patient_df['label'])) : 
        train_patient_ids = patient_df.loc[train_idx, 'patient_num'].values
        test_patient_ids = patient_df.loc[test_idx, 'patient_num'].values
        
        train_patient_ids, val_patient_ids = train_test_split(train_patient_ids, 
                                                              test_size = 0.25, # 0.8 * 0.25 = 0.2 --> 60% train, 20% val, 20% test 
                                                              random_state = args.seed, 
                                                              stratify = patient_df.query('patient_num in @train_patient_ids')['label']
                                                              )
        
        df[f'fold_{fold_num}'] = df['patient_num'].apply(lambda x : 
                                                        0 if x in train_patient_ids else 
                                                        1 if x in val_patient_ids else
                                                        2)
    return df

def rest_split(args, df, fold_df) : 
    print(df.shape, fold_df.shape)
    for fold_num in range(5) : 
        train_df = fold_df.query(f'fold_{fold_num} == 0').reset_index(drop = True)
        valid_df = df.query(f"patient_num not in @train_df['patient_num'].values").reset_index(drop = True)
        patient_df = valid_df[['patient_num', 'label']].drop_duplicates().reset_index(drop=True)
        valid_idx, test_idx = train_test_split(patient_df.index, test_size = 0.5, random_state = args.seed, stratify = patient_df['label'])
        train_patient_ids = train_df['patient_num'].unique()
        valid_patient_ids = patient_df.loc[valid_idx, 'patient_num'].values
        test_patient_ids = patient_df.loc[test_idx, 'patient_num'].values
        df[f'fold_{fold_num}'] = df['patient_num'].apply(lambda x : 
                                                        0 if x in train_patient_ids else
                                                        1 if x in valid_patient_ids else
                                                        2)
        print(f"Fold {fold_num}")
        print(f"By Patient | Train : {len(train_patient_ids)} | Val : {len(valid_patient_ids)} | Test : {len(test_patient_ids)}")
        print(f"By Images  | Train : {df.query('patient_num in @train_patient_ids').shape[0]} | Val : {df.query('patient_num in @valid_patient_ids').shape[0]} | Test : {df.query('patient_num in @test_patient_ids').shape[0]}")
    return df
        

def split_data(args) :
    df = pd.read_csv(args.data_path)
    df = df.rename(columns = {args.target_column : 'label'})
    print(df.sort_values(by = 'patient_num').reset_index(drop = True).head())
    
    under_df, is_success = undersample(df, args.undersampling_ratio, args)
    if is_success == False : 
        print("Failed")
        return
    under_df = under_df.sort_values(by = 'patient_num').reset_index(drop = True)
    print(under_df.head())
    print("By Patient", under_df[['patient_num', 'label']].drop_duplicates()['label'].value_counts().to_numpy().ravel())
    print("By Image", under_df['label'].value_counts().to_numpy().ravel())
    fold_df = fold_split(under_df, args)
    df = rest_split(args, df, fold_df)
    
    print('='*50)
    print(f"Sample Fold 0")
    print(f"Train : {df.query('fold_0 == 0').shape[0]} | {np.bincount(df.query('fold_0 == 0')['label']).ravel()}")
    print(f"Valid : {df.query('fold_0 == 1').shape[0]} | {np.bincount(df.query('fold_0 == 1')['label']).ravel()}")
    print(f"Test : {df.query('fold_0 == 2').shape[0]} | {np.bincount(df.query('fold_0 == 2')['label']).ravel()}")
    df.to_csv(args.data_save_path, index = False)