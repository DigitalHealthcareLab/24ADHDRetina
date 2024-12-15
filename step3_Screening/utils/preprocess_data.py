from pathlib import Path
import pandas as pd
import numpy as np




def process(args) : 
    raw_data_dir = Path(args.raw_data_dir)
    automorph_df = pd.read_csv(raw_data_dir / args.automorph_data_name)
    adhd_df = pd.read_csv(raw_data_dir / args.adhd_info_data_name)
    
    save_data_dir = Path(args.save_dir) / "ADHD_Screening" / args.data_save_dir
    save_data_dir.mkdir(exist_ok=True, parents=True)
    
    target_image_names = adhd_df['image_name'].values
    new_automorph_df = automorph_df[automorph_df['Name'].isin(target_image_names)]
    
    new_automorph_df = new_automorph_df.rename(columns = {'Name' : 'image_name'})
    new_automorph_df['patient_num'] = new_automorph_df['image_name'].apply(lambda x : adhd_df.set_index('image_name')['patient_num'].to_dict()[x])
    new_automorph_df['eye_direction'] = new_automorph_df['image_name'].apply(lambda x : adhd_df.set_index('image_name')['eye_direction'].to_dict()[x])
    new_automorph_df['age'] = new_automorph_df['image_name'].apply(lambda x : adhd_df.set_index('image_name')['age'].to_dict()[x])
    new_automorph_df['gender'] = new_automorph_df['image_name'].apply(lambda x : adhd_df.set_index('image_name')['gender'].to_dict()[x])
    new_automorph_df[args.target_column] = new_automorph_df['image_name'].apply(lambda x : 0 if x[0] in ['P', 'F', 'M'] else 1)
    
    new_automorph_df.to_csv(save_data_dir / args.preprocess_df_name, index = False)