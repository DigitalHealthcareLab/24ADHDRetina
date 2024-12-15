from pathlib import Path
import pandas as pd
import cv2
from joblib import Parallel, delayed

image_dir = Path('PATH_TO_IMAGE_DIR')
absolute_dir = Path(__file__).resolve().parent

def process_single_image(image_path, new_image_save_dir) : 
    image_save_path = new_image_save_dir / image_path.name
    print(image_save_path)
    if image_save_path.exists() :
        return [image_save_path, image_path.name, 1]
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (600, 600))
    cv2.imwrite(str(image_save_path), image)

def main(new_image_save_dir) : 
    image_paths = image_dir.rglob('*.png')
    Parallel(n_jobs=8)(delayed(process_single_image)(image_path, new_image_save_dir) for image_path in image_paths)

if __name__ == '__main__' :
    new_image_save_dir = absolute_dir / 'resized_images'
    new_image_save_dir.mkdir(exist_ok=True)
    
    main(new_image_save_dir)