import urllib.request
import gzip
import pandas as pd
from pathlib import Path

Path('data/raw').mkdir(parents=True, exist_ok=True)

url = 'https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz'
print('Downloading...')
urllib.request.urlretrieve(url, 'data/raw/hetionet-v1.0-edges.sif.gz')

print('Converting...')
with gzip.open('data/raw/hetionet-v1.0-edges.sif.gz', 'rt') as f:
    df = pd.read_csv(f, sep='\t')

print('Columns:', df.columns.tolist())
print(df.head(3))
df.to_csv('data/raw/hetionet_subset_edges.csv', index=False)
print(f'Done. {len(df)} edges saved.')