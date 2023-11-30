import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csv_path, dest_folder):
    df = pd.read_csv(csv_path)
    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    train, test = train_test_split(df, test_size=1-train_ratio, random_state=44, stratify=df['label'])
    val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=44, stratify=test['label'])
    
    # Write all to CSV file
    train.to_csv(os.path.join(dest_folder, 'train.csv'), index=False)
    val.to_csv(os.path.join(dest_folder, 'val.csv'), index=False)
    test.to_csv(os.path.join(dest_folder, 'test.csv'), index=False)
    

if __name__ == '__main__':
    csv_path = '/mnt/data/code/EDA_apple_sorting/assets/v2/dataset_v2.csv'
    dest_folder = '/mnt/data/code/EDA_apple_sorting/assets/v2'
    split_data(csv_path, dest_folder)
    

    
