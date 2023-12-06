import numpy as np
import pandas as pd
from pathlib import Path
import os

wc_dir = Path("/pub5/howard/reactivity_prediction/data/Ribonanza_bpp_files")

def get_corresponding_wc_matrix(sequence_id):
    filename = f"/pub5/howard/reactivity_prediction/data/ribo_bpp_np/{sequence_id}.npy"
    if not os.path.exists(filename): # skip if file exists
        print(".", end="", flush=True)
        filename = next(wc_dir.rglob(f"{sequence_id}.txt"))
        wc_matrix = np.zeros((1, 512, 512))

        with open(filename, 'r') as file:
            for line in file:
                if not line.strip():
                    break
                line = line.split(' ')
                i1, i2, val = int(line[0]), int(line[1]), float(line[2])
                wc_matrix[0, i1, i2] = val

        np.save(filename, wc_matrix)
    else:
        print("!", end="", flush=True)

    return str(filename)


def process_csv(file_path, function_to_apply):
    df = pd.read_csv(file_path)
    df['wc_matrix_path'] = df['sequence_id'].apply(function_to_apply)
    df.to_csv("/pub5/howard/reactivity_prediction/data/curr.csv")


if __name__ == "__main__":
    process_csv("/pub5/howard/reactivity_prediction/data/train_data_60000.csv", get_corresponding_wc_matrix)
