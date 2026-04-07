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


url = 'https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-nodes.tsv'
print('Downloading nodes...')
urllib.request.urlretrieve(url, 'data/raw/hetionet_nodes.tsv')

nodes = pd.read_csv('data/raw/hetionet_nodes.tsv', sep='\t')
print(nodes.head(5).to_string(index=False))
print('\nColumns:', nodes.columns.tolist())
print('\nNode types:', nodes['kind'].unique().tolist())