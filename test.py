import pandas as pd

train = pd.read_csv('data/splits/train.csv')
val   = pd.read_csv('data/splits/val.csv')
test  = pd.read_csv('data/splits/test.csv')

val_pos  = set(zip(val[val['label']==1]['disease_local_id'],   val[val['label']==1]['gene_local_id']))
test_pos = set(zip(test[test['label']==1]['disease_local_id'], test[test['label']==1]['gene_local_id']))
train_neg = set(zip(train[train['label']==0]['disease_local_id'], train[train['label']==0]['gene_local_id']))

print("Val positives appearing as train negatives:",  len(val_pos  & train_neg))
print("Test positives appearing as train negatives:", len(test_pos & train_neg))