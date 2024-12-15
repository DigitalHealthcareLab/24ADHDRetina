from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

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
        print(f"Fold {fold_num}")
        print(f"By Patient | Train : {len(train_patient_ids)} | Val : {len(val_patient_ids)} | Test : {len(test_patient_ids)}")
        print(f"By Images  | Train : {df.query('patient_num in @train_patient_ids').shape[0]} | Val : {df.query('patient_num in @val_patient_ids').shape[0]} | Test : {df.query('patient_num in @test_patient_ids').shape[0]}")
    
    return df


def split_data(args) :
    df = pd.read_csv(args.data_path)
    df = df.rename(columns = {args.target_column : 'label'})
    print(df.sort_values(by = 'patient_num').reset_index(drop = True).head())
    
    df = df.sort_values(by = 'patient_num').reset_index(drop = True)
    print(df.head())
    print("By Patient", df[['patient_num', 'label']].drop_duplicates()['label'].value_counts().to_numpy().ravel())
    print("By Image", df['label'].value_counts().to_numpy().ravel())
    df = fold_split(df, args)
    print('='*50)
    print(f"Sample Fold 0")
    print(f"Train : {df.query('fold_0 == 0').shape[0]} | {np.bincount(df.query('fold_0 == 0')['label']).ravel()}")
    print(f"Valid : {df.query('fold_0 == 1').shape[0]} | {np.bincount(df.query('fold_0 == 1')['label']).ravel()}")
    print(f"Test : {df.query('fold_0 == 2').shape[0]} | {np.bincount(df.query('fold_0 == 2')['label']).ravel()}")
 
    df.to_csv(args.data_save_path, index = False)