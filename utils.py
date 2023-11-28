import os
import pandas as pd
import numpy as np

def load_df_with_secondary_struct(df, secondary_struct_df):
    # Hardcoding the list of eligible secondary soures
    secondary_struct_types = np.array([
        'eterna_nupack',
        'eterna_eternafold+threshknot',
        'vienna2_mfe', 
        'contrafold2_mfe',
        'eternafold_mfe', 
        'e2efold_mfe', 
        'hotknots_mfe', 
        'ipknots_mfe',
        'knotty_mfe', 
        'pknots_mfe', 
        'spotrna_mfe', 
        'vienna[threshknot]_mfe',
        'vienna[hungarian]_mfe', 
        'eternafold[threshknot]_mfe',
        'eternafold[hungarian]_mfe', 
        'contrafold[threshknot]_mfe',
        'contrafold[hungarian]_mfe', 
        'nupack[threshknot]_mfe',
        'nupack[hungarian]_mfe', 
        'shapify_mfe', 
        'eternafold+hfold_1',
        'eternafold+hfold_2', 
        'eternafold+hfold_3', 
        'eternafold+hfold_4',
        'eternafold+hfold_final', 
        'nupack_mfe-pk', 
        'nupack-pk.threshknot',
        'nupack-pk.hungarian',
        'nupack.threshknot',
        'nupack.hungarian', 
        'hotknots', 
        'ipknots', 
        'knotty', 
        'spotrna', 
        'nupack_pk', 
        'vienna_2[threshknot]',
        'vienna_2[hungarian]', 
        'eternafold[threshknot]',
        'eternafold[hungarian]',
        'contrafold_2[threshknot]',
        'contrafold_2[hungarian]',
        'nupack[threshknot]',
        'nupack[hungarian]',
        'nupack-pk[threshknot]',
        'nupack-pk[hungarian]',
        'shapify-hfold', 
    ])

    secondary_struct = []
    for idx, row in df.iterrows():
        seq = df['sequence']
        sub_df = secondary_struct_df.loc[secondary_struct_df['sequence'] == seq]
        if len(sub_df) > 0:
            secondary_struct.append(sub_df.iloc[0][np.random.choice(secondary_struct_types)])
        else:
            secondary_struct.append(None)
    df['secondary_struct'] = secondary_struct

    return df
