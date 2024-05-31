import json
import gzip
import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from xgboost import XGBRegressor

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

class RTVSlo:

    def __init__(self):
        self.model = None # Inicializacija modela
        self.preprocessor = None # Inicializacija predprocesorja

    def extract_subtopic(self, url): # Funkcija za izluscenje podteme iz url-ja
        match = re.search(r'/(\w+(-\w+)*)/(\w+(-\w+)*)/', url) # Iskanje ujemanja z regularnim izrazom
        if match: # Ce je ujemanje najdeno
            return match.group(3) # Vrnemo tretjo skupino
        return None # Vrnemo None, ce ujemanje ni najdeno

    def data_preprocess(self, data: list) -> pd.DataFrame: # Funkcija za predprocesiranje podatkov
        def preprocess_text(text): # Funkcija za predprocesiranje besedila
            words = word_tokenize(text.lower()) # Tokenizacija besedila
            words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words] # Lematizacija in odstranjevanje zaustavitvenih besed	
            return ' '.join(words) # Povezovanje besed v string
        df = pd.DataFrame(data) # Pretvorba seznama v dataframe
        df['date'] = pd.to_datetime(df['date']) # Pretvorba stolpca z datumom v tip datetime
        df['day_of_week'] = df['date'].dt.dayofweek # Dodajanje stolpca z dnevom v tednu
        df['hour_of_day'] = df['date'].dt.hour # Dodajanje stolpca z uro v dnevu
        df['n_figures'] = df['figures'].apply(len) # Dodajanje stolpca z stevilom slik v clankih
        df['title_len'] = df['title'].apply(len) # Dodajanje stolpca z dolzinami naslovov clankov
        df['lead_len'] = df['lead'].apply(len) # Dodajanje stolpca z dolzinami uvodnega odstavka v clankih
        lemmatizer = WordNetLemmatizer() # Inicializiramo objekt za lematizacijo besed
        stop_words = set(stopwords.words('slovene')) # Uvozimo mnozico zaustavitvenih besed za slovenski jezik
        df['authors_info'] = df['authors'].apply(lambda authors: ' '.join([preprocess_text(author) for author in authors])) # Predprocesiranje avtorjev clankov
        df['title_info'] = df['title'].apply(preprocess_text) # Predprocesiranje naslovov clankov
        df['lead_info'] = df['lead'].apply(preprocess_text) # Predprocesiranje uvodnih odstavkov clankov
        df['paragraphs_info'] = df['paragraphs'].apply(lambda paragraphs: ' '.join([preprocess_text(paragraph) for paragraph in paragraphs])) # Predprocesiranje odstavkov clankov
        df['figures_info'] = df['figures'].apply(lambda figures: ' '.join([preprocess_text(figure['caption']) for figure in figures if 'caption' in figure])) # Predprocesiranje podnapisov slik v clankih
        df['subtopics'] = df['url'].apply(self.extract_subtopic) # Izluscimo podtemo iz url-ja
        return df # Vrnemo dataframe predprocesiranih podatkov

    def fit(self, train_data: list):
        df = self.data_preprocess(train_data) # Predprocesiranje podatkov
        X = df[['day_of_week', 'hour_of_day', 'n_figures', 'title_len', 'lead_len', 'topics', 'subtopics', 'authors_info', 'title_info', 'lead_info', 'paragraphs_info', 'figures_info']] # Izbor znacilk (neodvisnih spremenljivk) za model
        y = np.sqrt(df['n_comments']) # Izbor ciljne spremenljivke (odvisne spremenljivke) za model (koren stevila komentarjev za boljse rezultate)
        onehot_features = ['day_of_week', 'hour_of_day', 'topics', 'subtopics'] # Stolpci za one-hot kodiranje
        minmax_features = ['n_figures', 'title_len', 'lead_len'] # Stolpci za min-max skaliranje
        tfidf_features = ['authors_info', 'title_info', 'lead_info', 'paragraphs_info', 'figures_info'] # Stolpci z besedilnimi podatki za vektorsko predstavitev
        text_tfidf = [('authors', TfidfVectorizer(max_features=10), 'authors_info'),
                        ('title', TfidfVectorizer(max_features=50), 'title_info'),
                        ('lead', TfidfVectorizer(max_features=1000), 'lead_info'),
                        ('paragraphs', TfidfVectorizer(max_features=1000), 'paragraphs_info'),
                        ('figures', TfidfVectorizer(max_features=500), 'figures_info')] # Seznam terk za konfiguracijo TfidfVectorizer na besedilnih stolpcih
        tfidf_transformer = ColumnTransformer(text_tfidf) # Zdruzimo vse konfigurirane TfidfVectorizer transformacije za razlicna besedila
        preprocessor = ColumnTransformer( # Definiranje cevovoda za predprocesiranje podatkov
            transformers=[ # Seznam transformacij
                ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_features), # One-hot kodiranje kategoricnih spremenljivk
                ('minmax', MinMaxScaler(), minmax_features), # Min-max skaliranje numericnih spremenljivk
                ('tfidf', tfidf_transformer, tfidf_features) # Vektorska predstavitev besedilnih spremenljivk
            ],
            remainder='drop' # Preostale stolpce zavrzemo
        )
        self.preprocessor = Pipeline(steps=[ # Definiranje cevovoda za predprocesiranje podatkov
            ('preprocessor', preprocessor), # Dodamo predprocesor
            ('imputer', SimpleImputer(strategy='mean')) # Dodamo imputer za manjkajoce vrednosti
        ])
        X_piped = self.preprocessor.fit_transform(X) # Podatki predprocesirani z uporabo celotnega cevovoda
        """
        self.model = XGBRegressor()
        param_grid = {
            'max_depth': [5],
            'n_estimators': [250],
            'learning_rate': [0.08],
            # 'reg_alpha': [0.1],
            'reg_lambda': [10]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
        grid_search.fit(X_piped, y)
        self.model = grid_search.best_estimator_
        print("Najboljši parametri:", grid_search.best_params_)
        print("Najboljši MAE:", -grid_search.best_score_)
        """
        self.model = XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.08, reg_lambda=10) # Inicializacija modela
        self.model.fit(X_piped, y) # Ucenje modela

    def predict(self, test_data: list) -> np.array:
        df = self.data_preprocess(test_data) # Predprocesiranje podatkov
        X_preprocessed = self.preprocessor.transform(df) # Predprocesiranje podatkov z uporabo cevovoda
        return np.square(self.model.predict(X_preprocessed)) # Napovedovanje stevila komentarjev (kvadrat napovedi, ker smo pri ucenju uporabili koren stevila komentarjev)

def main():
    # Koda za oddajo
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()
    train_data = read_json(args.train_data_path)
    test_data =  read_json(args.test_data_path)
    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)
    predictions = np.maximum(predictions, 0) # Napovedi, ki so manjše od 0, spremenimo v 0
    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')
    np.savetxt('predictions.txt', predictions, fmt='%d')

    """
    # Koda za testiranje
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    args = parser.parse_args()
    train_data = read_json(args.train_data_path)

    rtv = RTVSlo()
    X_train, X_test = train_test_split(train_data, test_size=0.2, random_state=42)
    rtv.fit(X_train)
    y_test_pred = rtv.predict(X_test)
    y_test = np.array([article['n_comments'] for article in X_test])
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print(f"MAE: {np.mean(mae)}")
    print(f"R2: {np.mean(r2)}")
    """

if __name__ == '__main__':
    main()
