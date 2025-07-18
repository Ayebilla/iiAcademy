# Career Advisor AI

This project provides an intelligent career advising assistant powered by machine learning and Chainlit. 
It takes user input (age, education level, country, job title, years of experience, etc.) and predicts expected salary while offering career guidance.

---

## Quick Start (Recommended Branch: `working_branch`)

To get started with the working version:

### 1. Clone the Repository and Switch to `working_branch`

```bash
git clone https://github.com/Ayebilla/iiAcademy.git
cd iiAcademy
git checkout working_branch
```

### 2. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Step 1: Generate Model and Preprocessing Pipeline

Go into the `assets` folder and run the ML script:

```bash
cd assets
python ML-modeling.py
```

This will save the model and data processing pipeline needed for inference.

### Step 2: Launch Chainlit App

Navigate back to the root directory and start the Chainlit interface:

```bash
cd ..
chainlit run main.py -w
```

---

## Project Structure

```
iiAcademy/
├── assets/
│   └── ML-modeling.py            # Model training and saving script
│
├── tools/
│   └── salary_tool.py            # Handles preprocessing and prediction
│
├── custom_preprocessor.py        # Custom transformers for data pipeline
├── main.py                       # Chainlit app entry point
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Tech Stack

- Python
- Scikit-learn
- Gensim (for embeddings)
- Chainlit (for conversational UI)
- Joblib (for saving model and pipeline)

---

## Contributing

Feel free to fork the repository, open issues, and submit pull requests — especially on the `working_branch`.

