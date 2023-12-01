import os
import pandas as pd


def load_df_with_secondary_struct(df, secondary_struct_df, secondary_struct_col='eterna_nupack'):
    secondary_struct = []
    for idx, row in df.iterrows():
        seq = df['sequence']
        sub_df = secondary_struct_df.loc[secondary_struct_df['sequence'] == seq]
        if len(sub_df) > 0:
            secondary_struct.append(sub_df.iloc[0][secondary_struct_col])
        else:
            secondary_struct.append(None)
    df['secondary_struct'] = secondary_struct

    return df
