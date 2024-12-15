import argparse
from omegaconf import OmegaConf
from pathlib import Path


from utils.get_data import load_Normalized_Data
from trainers.random_forest import train_rf
from trainers.lgbm import train_lgbm
from trainers.xgboost import train_xgboost
from trainers.extra_clf import train_ext
from trainers.logistic_reg import train_logreg

def main(args) : 
    args.base_dir = Path(args.save_dir) / "ADHD_Screening"
    args.data_path = args.base_dir / 'data' / args.fold_df_name
    if not args.data_path.exists() :
        print(  f"{args.data_path} does not exist.")
        return
    
    
    args.result_save_dir = args.base_dir / f'fold_{args.fold_num}'
    args.data_pkl_path = args.result_save_dir / 'data.pkl'
    args.result_save_dir.mkdir(parents = True, exist_ok = True)
    
    data = load_Normalized_Data(args)
    
    # Random Forest
    args.rf_result_path = args.result_save_dir / 'random_forest' / 'fold_result.csv'
    if args.rf_result_path.exists() : 
        print(f"{args.rf_result_path} already exists.")
    else : 
        args.rf_result_path.parent.mkdir(parents = True, exist_ok = True)
        train_rf(args, data)
    
    args.xgb_result_path = args.result_save_dir / 'xgboost' / 'fold_result.csv'
    if args.xgb_result_path.exists() : 
        print(f"{args.xgb_result_path} already exists.")
    else : 
        args.xgb_result_path.parent.mkdir(parents = True, exist_ok = True)
        train_xgboost(args, data)
        
    args.ext_result_path = args.result_save_dir / 'extra_clf' / 'fold_result.csv'
    if args.ext_result_path.exists() : 
        print(f"{args.ext_result_path} already exists.")
    else : 
        args.ext_result_path.parent.mkdir(parents = True, exist_ok = True)
        train_ext(args, data)
        
    args.logreg_result_path = args.result_save_dir / 'logistic_reg' / 'fold_result.csv'
    if args.logreg_result_path.exists() : 
        print(f"{args.logreg_result_path} already exists.")
    else : 
        args.logreg_result_path.parent.mkdir(parents = True, exist_ok = True)
        train_logreg(args, data)
    return

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--undersampling_ratio', type = int, default=1)
    parser.add_argument('--fold_num', type = int, default=0)
    
    parser_args = parser.parse_args()
    
    config_path = "configs/adhd_ml_config.yaml"
    args = OmegaConf.load(config_path)
    args.undersampling_ratio = parser_args.undersampling_ratio
    args.fold_num =  parser_args.fold_num
    main(args)