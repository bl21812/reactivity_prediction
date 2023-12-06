import pandas as pd
import os


def process_csv(file_path):
    df = pd.read_csv(file_path)
    filtered_df = df[
        df['sequence_id'].apply(lambda sid: os.path.exists(f"/pub5/howard/reactivity_prediction/data/ribo_bpp_np/{sid}.npy"))
    ]
    filtered_df.to_csv("/pub5/howard/reactivity_prediction/data/train_data_60000_wc.csv", index=False)


if __name__ == "__main__":
    process_csv("/pub5/howard/reactivity_prediction/data/train_data_60000.csv", )
