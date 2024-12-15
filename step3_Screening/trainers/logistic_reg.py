from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

from utils.performance_calculator import calculate_scores, calculate_youden_index


def train_logreg(args, data) :     
    train_X, train_y, train_names = data['train']
    valid_X, valid_y, valid_names = data['valid']
    test_X, test_y, test_names = data['test']
    
    logreg = LogisticRegression(random_state= args.seed, n_jobs = 8, verbose = 0, max_iter = 10)
    
    logreg.fit(train_X, train_y)
    
    train_preds, valid_preds, test_preds = logreg.predict_proba(train_X), logreg.predict_proba(valid_X), logreg.predict_proba(test_X)
    
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
        'model' : logreg,
    }
    score_df.to_csv(args.logreg_result_path, index = False)
    pickle.dump(results, open(args.logreg_result_path.with_suffix('.pkl'), 'wb'))
    print(score_df)