import os
from config import BASE_DIR
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == "__main__":

    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/dataset.csv'))
    except FileNotFoundError:
        raise ('dataset.csv file not found - label your data first')

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, temp_index in split.split(df['image'], df['label']):
        train_set = df.loc[train_index]
        temp_set = df.loc[temp_index]

    for val_index, test_index in split.split(temp_set['image'], temp_set['label']):
        validation_set = df.loc[val_index]
        test_set = df.loc[test_index]

    train_set.to_csv(os.path.join(BASE_DIR, 'data/processed/train.csv'), index=False)
    validation_set.to_csv(os.path.join(BASE_DIR, 'data/processed/validation.csv'), index=False)
    test_set.to_csv(os.path.join(BASE_DIR, 'data/processed/test.csv'), index=False)
