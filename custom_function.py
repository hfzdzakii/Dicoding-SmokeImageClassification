# Cleaning Text Data
import pandas as pd
import re
import unicodedata
import contractions
import nltk
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model # type: ignore
from tensorflow.data import Dataset # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Lambda, Embedding, GRU, Bidirectional # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.utils import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint # type: ignore

# nltk.download('words')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords') 
# nltk.download('averaged_perceptron_tagger_eng')

class DicodingProject1:
    def __init__(self, TEST_SIZE, EPOCHS, MODE, CLASS, DATASET):
        self.TEST_SIZE = TEST_SIZE
        self.BATCH_SIZE = 64
        self.EPOCHS = EPOCHS
        self.MODE = MODE
        self.MAX_LENGTH = 128
        self.CLASS = CLASS
        self.DATASET = DATASET
        
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    english_words = set(words.words())

    def loadCustomDict(self, path):
        with open(path, 'r') as file:
            return set(line.strip().lower() for line in file if line.strip())

    def normalizeWhitespace(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = contractions.fix(text)
        text = re.sub(r'[\t\r]+', ' ', text) # Menghapus tab
        text = re.sub(r'\b\d+\b', '', text) # Menghilangkan angka
        text = re.sub(r'[-‐‑‒–—―]+', '', text)
        text = re.sub(r'[_﹍﹎＿]', '', text)
        text = re.sub(r'[^\w\s]', '', text) # Hilangkan symbol punctuation
        text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def removeNonEnglish(self, text_series, custom_dict):
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in custom_dict) + r')\b'
        temp_series = text_series.str.replace(pattern, '', case=False, regex=True)
        split_words = temp_series.str.split()
        exploded = split_words.explode()
        exploded = exploded[exploded.str.lower().isin(self.english_words)]
        filtered = exploded[~exploded.str.lower().isin(self.stop_words)]
        lemmatized = filtered.apply(lambda word: self.lemmatizer.lemmatize(word.lower()))
        cleaned_text_series = lemmatized.groupby(level=0).agg(' '.join)
        pattern2 = r'\b(\w+)(?:\s+\1\b)+' #, r'\1', text)
        ser = cleaned_text_series.reindex(text_series.index, fill_value='')
        text = ser.str.replace(pattern2, r'\1', case=False, regex=True)
        return text

    def removeOtherLanguage(self, text):
        phrase = ' translated'
        pos = text.find(phrase)
        if pos != -1:
            text = text[:pos].rstrip()
        text = re.sub(r'\b\w*[^\x00-\x7F]\w*\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def resamplingOversampling(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def prepareDf(self, df):
        custom_dict = self.loadCustomDict('custom_vocab.txt')
        encoder = LabelEncoder()
        df['poem'] = df['poem'].apply(self.normalizeWhitespace)
        df['poem'] = df['poem'].apply(self.removeOtherLanguage)
        df['poem'] = self.removeNonEnglish(df['poem'], custom_dict)
        X = df[['poem']]
        y = df['label']
        y = encoder.fit_transform(y)
        X, y = self.resamplingOversampling(X, y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.TEST_SIZE, random_state=42, stratify=y)
        # X_train, y_train = resamplingOversampling(X_train, y_train)
        X_train = X_train['poem'].values.flatten().tolist() #
        X_val = X_val['poem'].values.flatten().tolist() #
        return X_train, X_val, y_train, y_val

    def buildModel(self, mode, word_counts=256):
        input_layer = Input(shape=(self.MAX_LENGTH,), name='input_layer')
        output = Embedding(input_dim=word_counts, output_dim=word_counts//2, name='embedding_layer')(input_layer)
        if mode=='lstm':
            x = LSTM(64, return_sequences=False)(output)
        if mode=='gru':
            x = GRU(64, return_sequences=False)(output)
        x = Dropout(0.5)(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(self.CLASS, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def createDataset(self, data, labels):
        dataset = Dataset.from_tensor_slices((data, labels))
        dataset = dataset.batch(self.BATCH_SIZE)
        return dataset

    def earlyStopping(self):
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1,
        )
        return early_stopping

    def checkpointCallback(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=f'./model/best_model_{self.DATASET}_{self.TEST_SIZE}_{self.MODE}.keras', 
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
        return checkpoint_callback

    def kerasTokenizer(self, X_train, X_val):
        tokenizer = Tokenizer(oov_token='<OOV>')
        word_counts = tokenizer.word_counts
        tokenizer.fit_on_texts(X_train)
        X_train_sequence = tokenizer.texts_to_sequences(X_train)
        X_val_sequence = tokenizer.texts_to_sequences(X_val)
        X_train_padded = pad_sequences(X_train_sequence, maxlen=self.MAX_LENGTH)
        X_val_padded = pad_sequences(X_val_sequence, maxlen=self.MAX_LENGTH)
        with open(f"./tokenizer/tokenizer_{self.DATASET}_{self.TEST_SIZE}_{self.MODE}.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        return X_train_padded, X_val_padded, len(word_counts)

    def kerasTokenizer2(self, text):
        with open(f"./tokenizer/tokenizer_{self.DATASET}_{self.TEST_SIZE}_{self.MODE}.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        text_sequence = tokenizer.texts_to_sequences(text)
        text_padded = pad_sequences(text_sequence, maxlen=self.MAX_LENGTH)
        return text_padded
    
    def cleanInference(self, df):
        custom_dict = self.loadCustomDict('custom_vocab.txt')
        df['poem'] = df['poem'].apply(self.normalizeWhitespace)
        df['poem'] = df['poem'].apply(self.removeOtherLanguage)
        df['poem'] = self.removeNonEnglish(df['poem'], custom_dict)
        return df
    
    @staticmethod
    def getLabelEncoder(name):
        hartmann = ['sadness', 'fear', 'anger', 'joy']
        savani = ['joy', 'sadness', 'anger']
        if name=='hartmann':
            return {i : label for i, label in enumerate(sorted(hartmann))}
        if name=='savani':
            return {i : label for i, label in enumerate(sorted(savani))}
        
        
class DicodingProject2:
    
    @staticmethod
    def showImages(images, label_mapping):
        plt.figure(figsize=(10,10))
        for idx, img in enumerate(images):
            plt.subplot(int(math.sqrt(len(images))), int(math.sqrt(len(images))), idx+1)
            plt.imshow(img[0])
            plt.title(f'Label : {label_mapping[img[1]]}\nResolution : {img[0].size}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def showImages2(images, label_mapping):
        plt.figure(figsize=(10,10))
        for idx, img in enumerate(images):
            plt.subplot(int(math.sqrt(len(images))), int(math.sqrt(len(images))), idx+1)
            plt.imshow(img[0])
            plt.title(f'Label : {label_mapping[img[1]]}\nResolution : ({np.array(img[0]).shape[0]}, {np.array(img[0]).shape[1]})')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def earlyStopping():
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
        )
        return early_stopping

    @staticmethod
    def checkpointCallback():
        checkpoint_callback = ModelCheckpoint(
            filepath=f'./saved_model/best_model.keras', 
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
        return checkpoint_callback
    
    @staticmethod
    def preprocessImage(dataset, target_size=(128, 128)):
        dataset['resized_image'] = [image.resize(target_size) for image in dataset['image']]
        return dataset
    
    @staticmethod
    def convert2Numpy(dataset):
        data_temp = []
        label_temp = []
        for idx, example in enumerate(dataset):
            img = np.array(example['resized_image'])
            data_temp.append(img)
            label = example['label']
            label_temp.append(label)
        data_temp = np.array(data_temp)
        label_temp = np.array(label_temp)
        return data_temp, label_temp
    
    @staticmethod
    def resampling(data, label):
        num_samples, height, width, channels = data.shape
        data_reshaped = data.reshape(num_samples, -1)
        ros = RandomOverSampler(random_state=42)
        data_resampled, labels_resampled = ros.fit_resample(data_reshaped, label)
        data_resampled = data_resampled.reshape(-1, height, width, channels)
        return data_resampled, labels_resampled

    @staticmethod
    def standarize(image, label):
        image = tf.image.per_image_standardization (image)
        return image, label

    @staticmethod
    def toTfDataset(data, label, batch_size=64, shuffle=True):
        if shuffle:
            return tf.data.Dataset.from_tensor_slices((data, label))\
                .map(DicodingProject2.standarize)\
                .shuffle(len(data))\
                .batch(batch_size)\
                .prefetch(tf.data.experimental.AUTOTUNE)
        if not shuffle:
            return tf.data.Dataset.from_tensor_slices((data, label))\
                .map(DicodingProject2.standarize)\
                .batch(batch_size)\
                .prefetch(tf.data.experimental.AUTOTUNE)