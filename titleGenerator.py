# %%
#===============================================================================
# CONFIGURATIONS AND DATA LOADING

# load modules
import json
import random
import pandas as pd

# Set pandas to display entire value on column
pd.set_option('display.max_colwidth', 1000)

# transform json to pandas dataframe
df = pd.read_json('VitaLITy-1.0.0.json')

# %%
#===============================================================================
# CLEANING AND PREPARING DATA

# drop columns Authors, Source, Year, CitationCounts, AbstractLength, glove_embedding, glove_umap, specter_embedding, glove_kmeans_cluster, specter_kmeans, ID, _id, specter_umap
df = df.drop(['Authors', 'Source', 'Year', 'CitationCounts', 'AbstractLength', 'glove_embedding', 'glove_umap', 'specter_embedding', 'glove_kmeans_cluster', 'specter_kmeans', 'ID', '_id', 'specter_umap'], axis=1)

# Transform abstract to lowercase
df['Abstract'] = df['Abstract'].str.lower()
# Transform title to lowercase
df['Title'] = df['Title'].str.lower()
# Remove all special characters from title
df['Title'] = df['Title'].str.replace('[^\w\s]', '')

# Transform Title string to list
df['Title'] = df['Title'].apply(lambda x: x.split())

# Transform keywords array values to lowercase if not empty
df['Keywords'] = df['Keywords'].apply(lambda x: [i.lower() for i in x] if x else [])

# list of common english words and special characters to remove from abstract
words = ['is', 'are', 'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', '.', ',', '!', '?', ':', ';', '-', '_', '"', "'", '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', '[', ']', '\\', '/', '|', '~', '`', '=', '+', '<', '>', '@', '`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# Parse abstract removing common english words
df['Abstract'] = df['Abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in words]))

# Remove all special characters from abstract
df['Abstract'] = df['Abstract'].str.replace('[^\w\s]', '')

# Separate abstract words into list
df['Abstract'] = df['Abstract'].apply(lambda x: x.split())

# Join abstract and keywords lists
df['Keywords'] = df['Abstract'] + df['Keywords']

# Remove duplicates from keywords list
df['Keywords'] = df['Keywords'].apply(lambda x: list(set(x)))

# Drop abstract column
df = df.drop(['Abstract'], axis=1)

# %%
#===============================================================================
# CREATING DICTIONARY

# Create a list of all words in the dataset
dictionary = []

# Iterate through the keywords of each row and adds each word to the list
for i in df['Keywords']:
    for j in i:
        # Append only if not on the list
        dictionary.append(j)

# Iterate through the titles of each row and adds each word to the list
for i in df['Title']:
    for j in i:
        # Append only if not on the list
        dictionary.append(j)

# Remove duplicated values from dictionary
dictionary = list(set(dictionary))

# Save dictionary list to dic.txt file
with open('dic.txt', 'w', encoding="utf-8") as f:
    for item in dictionary:
        f.write("%s\n" % item)

# %%
#===============================================================================
# CREATING BAG OF WORDS

# Load the dic.txt file into dictionary variable
with open('dic.txt', 'r', encoding="utf-8") as f:
    dictionary = f.read().splitlines()

# Transform keywords list to string
df['Keywords'] = df['Keywords'].apply(lambda x: ' '.join(x))
# Transform titles list to string
df['Title'] = df['Title'].apply(lambda x: ' '.join(x))

# Import keras tokenizer
from keras.preprocessing.text import Tokenizer

# Create the tokenizer
tokenizer = Tokenizer()
# Fit the tokenizer on each keywords row with lambda function
tokenizer.fit_on_texts(df['Keywords'].apply(lambda x: x))
# Fit the tokenizer on each titles row with lambda function
tokenizer.fit_on_texts(df['Title'].apply(lambda x: x))

# Separate dataframe into training and test sets
train, test = df.iloc[:int(len(df)*0.8)], df.iloc[int(len(df)*0.8):]

# Separate train set into x and y, with y being the Title
x_train, y_train = train.drop(['Title'], axis=1), train['Title']

# Separate test set into x and y, with y being the Title
x_test, y_test = test.drop(['Title'], axis=1), test['Title']

# Encode training data set
x_train = tokenizer.texts_to_matrix(x_train, mode='freq')
print(x_train.shape)

# print full first row
print(df.head(5))

# %%
#===============================================================================
# INSTANTIATING AI 

# Import Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# define network
model = Sequential()
model.add(Dense(50, input_shape=(len(dictionary),), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(x_train, y_train, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))