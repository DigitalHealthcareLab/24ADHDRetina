import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def get_fold_df(df, args, mode) : 
    fold_num = args.fold_num
    fold_df = df.query(f"fold_{fold_num} == {mode}")
    drop_columns = list(fold_df.filter(regex = 'fold').columns) + ['label', 'image_name', 'patient_num', 'eye_direction', 'age', 'gender']
    X = fold_df.drop(columns = drop_columns)
    y = fold_df['label'].values
    names = fold_df['image_name'].values
    return X, y, names

def load_Normalized_Data(args) : 
    if not args.data_pkl_path.exists() :
        
        df = pd.read_csv(args.data_path)
        train_X, train_y, train_names = get_fold_df(df, args, 0)
        valid_X, valid_y, valid_names = get_fold_df(df, args, 1)
        test_X, test_y, test_names = get_fold_df(df, args, 2)
        
        scaler = StandardScaler()
        scaler.fit(pd.concat([train_X, valid_X]).values)
        train_X, valid_X, test_X = scaler.transform(train_X.values), scaler.transform(valid_X.values), scaler.transform(test_X.values)
        
        data = {
            'train' : (train_X, train_y, train_names),
            'valid' : (valid_X, valid_y, valid_names),
            'test' : (test_X, test_y, test_names),
            'scaler' : scaler
        }
        pickle.dump(data, open(args.data_pkl_path, 'wb'))
    
    data = pickle.load(open(args.data_pkl_path, 'rb'))
    return data
        
        