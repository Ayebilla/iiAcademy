# !pip install gensim
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api

class JobTitleEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.word2vec = None
        self.vector_size = 50

    def fit(self, X, y=None):
        if self.word2vec is None:
            self.word2vec = api.load("glove-wiki-gigaword-50")
        return self

    def transform(self, X):
        def get_embedding(text):
            if pd.isna(text) or not isinstance(text, str):
                return np.zeros(self.vector_size)
            words = text.lower().strip().split()
            vectors = [self.word2vec[word] for word in words if word in self.word2vec]
            return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

        job_titles = pd.Series(X.iloc[:, 0]) if isinstance(X, pd.DataFrame) else pd.Series(X)
        embeddings = job_titles.apply(get_embedding)
        return np.vstack(embeddings.values)


class SalaryPreprocessor:
    def __init__(self):
        self.country_salary_index = {
            'USA': 9, 'UK': 8, 'Canada': 7, 'Australia': 6, 'China': 5,
            'South Africa': 4, 'Ghana': 3, 'Kenya': 2, 'Uganda': 1
        }
        self.education_mapping = {
            'high school': 1, "bachelor's": 2, "master's": 3, 'phd': 4
        }

    def transform(self, df):
        data = df.copy().bfill()
        data['Education Level'] = data['Education Level'].str.lower().str.strip().replace({
            "bachelor's degree": "bachelor's",
            "master's degree": "master's",
            "phd": "phd"
        })
        data['Country'] = data['Country'].map(self.country_salary_index)
        data['Education'] = data['Education Level'].map(self.education_mapping)
        data['Job Title'] = data['Job Title'].str.lower().str.strip()
        return data.drop(columns=['Education Level'], errors='ignore')
