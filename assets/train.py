import joblib
import pandas as pd  

preprocessor = joblib.load("salary_preprocessor.pkl")
model = joblib.load("salary_predictor_pipeline.pkl")


new_input = pd.DataFrame([{
    'Age': 30,
    'Years of Experience': 5,
    'Education Level': "Master's Degree",  # Will be converted to "master's"
    'Country': 'Kenya',  # Will be mapped to 2
    'Gender': 'Female',
    'Race': 'Black',
    'Job Title': 'data scientist'  # Will be lowercased and embedded
}])


processed_input = preprocessor.transform(new_input)


predicted_salary = model.predict(processed_input)

print(f"Predicted Salary: {predicted_salary[0]:,.2f}")