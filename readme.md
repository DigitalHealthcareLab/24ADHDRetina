## Retinal Fundus Imaging as a Biomarker for Attention-Deficit/Hyperactivity Disorder: Machine Learning for Screening and Visual Attention Stratification

This repository contains code for the pipeline used in this study.



### Step 0 : Environment setting
Our pipeline was built with Python version 3.9.15, and other detailed versions are listed on `reuquirements.txt`

Please make virtual environment and run `pip install -r requirements.txt`

### Step 1 : Preprocess of the data
For standardizing different image sizes into single size, we resized images into (600, 600).

Please modify the directory in the step1_preprocess/preprocess.py and run `python step1_preprocess/preprocess.py`

### Step 2 : Extracting Feature values using AutoMorph
We adapted same pipeline in AutoMorph pipeline.

Please follow the directions in the repository.
`
@article{zhou2022automorph,
  title={AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline},
  author={Zhou, Yukun and Wagner, Siegfried K and Chia, Mark A and Zhao, An and Xu, Moucheng and Struyven, Robbert and Alexander, Daniel C and Keane, Pearse A and others},
  journal={Translational vision science \& technology},
  volume={11},
  number={7},
  pages={12--12},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}
`

### Step 3 : Developing Machine Learning models
Please modify config.yaml file in configs appropriately, sepcifically "raw_data_dir", "automorph_data_name", "adhd_info_data_name".
- raw_data_dir : directory which contains automorph feature values and information of participants
- automorph_data_name : Dataframe extracted using AutoMorph package which contains automorph feature values
- adhd_info_data_name : Dataframe contains information of participants including age, sex, eye direction and others.

Then, run 
`bash total_train.sh`

The results will be automatically generated in "save_dir" notated in configs.yaml

### sample data
- This directory contains the sample data