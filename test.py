import pandas as pd
df = pd.read_csv('data/raw/hetionet_subset_edges.csv')
all_types = set(df['source_type'].unique()) | set(df['target_type'].unique())
print(sorted(all_types))