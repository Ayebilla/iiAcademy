import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import gensim.downloader as api
import joblib
from custom_preprocessor import JobTitleEmbedder, SalaryPreprocessor

def create_salary_pipeline():
    numeric_features = ['Age', 'Years of Experience', 'Education', 'Country']
    categorical_features = ['Gender', 'Race']
    job_title_feature = ['Job Title']

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    job_title_transformer = Pipeline([('embedder', JobTitleEmbedder())])

    preprocessor = FeatureUnion([
        ('numeric', ColumnTransformer([('num', numeric_transformer, numeric_features)])),
        ('categorical', ColumnTransformer([('cat', categorical_transformer, categorical_features)])),
        ('job_title', ColumnTransformer([('job', job_title_transformer, job_title_feature)]))
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=120, max_depth=18, random_state=42))
    ])

# Main routine
def run_training_pipeline():
    df = pd.read_csv('data.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    preprocessor = SalaryPreprocessor()
    df_processed = preprocessor.transform(df)
    joblib.dump(preprocessor, 'salary_preprocessor.pkl')

    X = df_processed.drop(columns=['Salary'])
    y = df_processed['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_salary_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("R² score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    joblib.dump(model, 'salary_predictor_pipeline.pkl')
    print("Pipeline saved successfully!")

run_training_pipeline()
