from pathlib import Path
from omegaconf import DictConfig, open_dict, OmegaConf
import argparse

from utils.preprocess_data import process

def main(args) : 
    process(args)    
    
    return


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser_args = parser.parse_args()
    
    config_path = Path('configs/adhd_ml_config.yaml')
    args = OmegaConf.load(config_path)
    main(args)
    