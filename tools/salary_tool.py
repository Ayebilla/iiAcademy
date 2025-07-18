# tools/salary_tool.py
import pandas as pd
import joblib
import json
import os
from custom_preprocessor import SalaryPreprocessor  # only import what's used

def salary_prediction_tool(json_input: str) -> str:
    try:
        # Load input parameters
        input_dict = json.loads(json_input)
        df = pd.DataFrame([{
            "Age": int(input_dict["age"]),
            "Gender": input_dict["gender"],
            "Education Level": input_dict["education"],
            "Job Title": input_dict["job_title"],
            "Years of Experience": float(input_dict["experience"]),
            "Country": input_dict["country"],
            "Race": input_dict["race"]
        }])

        # Load preprocessor and model from correct paths
        preprocessor = joblib.load("assets/salary_preprocessor.pkl")
        df = preprocessor.transform(df)

        model = joblib.load("assets/salary_predictor_pipeline.pkl")
        prediction = model.predict(df)[0]

        return f"Estimated salary: ${prediction:,.2f}"

    except Exception as e:
        return f"Error during prediction: {str(e)}"
