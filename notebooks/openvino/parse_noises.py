import pandas as pd
from pathlib import Path
s='/home/nlyaly/projects/optimum-intel/notebooks/openvino/nncf_debug_X32_3iter/lora/noises.csv'
df = pd.read_csv(s)
df.columns
# df[0]

# s2 = '/home/nlyaly/projects/optimum-intel/notebooks/openvino/nncf_debug_no_X32/lora/noises.csv'
# df2 = pd.read_csv(s2)
# df2.columns

