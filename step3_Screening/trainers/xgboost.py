
from xgboost import XGBClassifier
import pandas as pd
import pickle
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from utils.performance_calculator import calculate_scores, calculate_youden_index

def train_xgboost(args, data) : 
    train_X, train_y, train_names = data['train']
    valid_X, valid_y, valid_names = data['valid']
    test_X, test_y, test_names = data['test']
    
    model = XGBClassifier(n_estimators = 1000, 
                           random_state= args.seed, 
                           n_jobs = 8, 
                           verbose = -1,
                           )
    model.fit(train_X, train_y, eval_set = [(valid_X, valid_y)], verbose = 1000)
    
    train_preds, valid_preds, test_preds = model.predict_proba(train_X), model.predict_proba(valid_X), model.predict_proba(test_X)
    threshold = calculate_youden_index(valid_preds, valid_y)
    train_scores = calculate_scores(train_preds, train_y, threshold)
    valid_scores = calculate_scores(valid_preds, valid_y, threshold)
    test_scores = calculate_scores(test_preds, test_y, threshold)
    
    score_df = pd.DataFrame([train_scores, valid_scores, test_scores], index = ['train', 'valid', 'test']).T
    
    results = {
        'train_preds' : train_preds,
        'train_labels' : train_y,
        'train_names' : train_names,
        'valid_preds' : valid_preds,
        'valid_labels' : valid_y,
        'valid_names' : valid_names,
        'test_preds' : test_preds,
        'test_labels' : test_y,
        'test_names' : test_names,
        'threshold' : threshold,
        'model' : model,
    }
    score_df.to_csv(args.xgb_result_path, index = False)
    pickle.dump(results, open(args.xgb_result_path.with_suffix('.pkl'), 'wb'))
    print(score_df)