from omegaconf import DictConfig, open_dict, OmegaConf
import argparse
from pathlib import Path

from utils.split_data import split_data
# from utils.split_data2 import split_data


def run(args : DictConfig):  
    
    args.save_data_dir = Path(args.save_dir) / "ADHD_Screening" / args.data_save_dir
    args.data_path = args.save_data_dir / args.preprocess_df_name
    args.data_save_path = Path(args.save_dir) / "ADHD_Screening" / "data" / args.fold_df_name

    if args.data_save_path.exists() :
        print(f"{args.data_save_path} already exists.")
        return
    args.data_save_path.parent.mkdir(parents=True, exist_ok=True)    
    split_data(args)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser_args = parser.parse_args()
    
    config_path = "configs/adhd_ml_config.yaml"
    args = OmegaConf.load(config_path)
    run(args)