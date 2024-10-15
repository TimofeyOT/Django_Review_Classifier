import os
import pandas as pd

train = {'review': [], 'rating': []}
files = os.listdir('aclImdb 2/train/neg')

for i in files:
    train["rating"].append(int(i[-5:-4]))
    path = 'aclImdb 2/train/neg/' + i
    train['review'].append(open(path, "r").read())

files = os.listdir('aclImdb 2/train/pos')
for i in files:
    if int(i[-5:-4]) == 0:
        train["rating"].append(10)
    else:
        train["rating"].append(int(i[-5:-4]))
    path = 'aclImdb 2/train/pos/' + i
    train['review'].append(open(path, "r").read())

df_train = pd.DataFrame(data=train)

df_train.to_csv('imdb_train.csv')

test = {'review': [], 'rating': []}
files = os.listdir('aclImdb 2/test/neg')
for i in files:
    test["rating"].append(int(i[-5:-4]))
    path = 'aclImdb 2/test/neg/' + i
    test['review'].append(open(path, "r").read())

files = os.listdir('aclImdb 2/test/pos')
for i in files:
    if int(i[-5:-4]) == 0:
        test["rating"].append(10)
    else:
        test["rating"].append(int(i[-5:-4]))
    path = 'aclImdb 2/test/pos/' + i
    test['review'].append(open(path, "r").read())

df_test = pd.DataFrame(data=test)

df_test.to_csv('imdb_test.csv')


