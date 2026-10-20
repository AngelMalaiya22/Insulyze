# 🩸 GlycoLens: Diabetes Risk Prediction Web App

GlycoLens is an end-to-end machine-learning project that predicts the likelihood of diabetes from eight clinical measurements. It includes the full ML pipeline (EDA, preprocessing, model comparison, Optuna tuning, SHAP explainability) and a Flask website with an input form, deployable for free.

**Live demo:** `https://your-app-name.onrender.com` *(replace after deploying)*

## Features
- Web form with input validation and a clear High / Low risk result with probability
- JSON API endpoint (`POST /api/predict`)
- Leak-free scikit-learn pipeline: median imputation, outlier clipping, feature engineering, one-hot encoding, scaling
- 8 baseline models compared, 3 tuned with Optuna, threshold chosen from out-of-fold predictions
- Responsive layout with automatic dark mode

## Results
| Item | Value |
|---|---|
| Dataset | 768 patients, 8 features (Pima Indians diabetes) |
| Final model | Tuned Logistic Regression |
| 5-fold CV ROC-AUC | about 0.85 |
| Test accuracy / recall / ROC-AUC | 0.75 / 0.76 / 0.82 |
| Top features | Glucose, BMI, Pregnancies, Age |

## Project structure
```
glycolens-web/
├── app.py                    # Flask app (form + API)
├── glycolens_utils.py        # custom preprocessing steps used by the saved model
├── glycolens_model.joblib    # trained pipeline
├── glycolens_config.json     # decision threshold
├── templates/index.html      # web form
├── static/style.css
├── requirements.txt
├── Procfile / render.yaml    # deployment config
└── GlycoLens_Diabetes_Prediction.ipynb   # full ML notebook (add it to the repo)
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # open http://127.0.0.1:5000
```

## API example
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":150,"BloodPressure":80,"SkinThickness":30,
       "Insulin":0,"BMI":34,"DiabetesPedigreeFunction":0.6,"Age":45}'
```
Response: `{"high_risk": true, "label": "High risk", "percent": 73.9, "probability": 0.7394}`

## Deploy on Render (free)
1. Push this folder to a GitHub repository.
2. On [render.com](https://render.com), choose **New → Web Service** and connect the repository (or **New → Blueprint** to use `render.yaml` automatically).
3. Settings if configuring manually: Runtime `Python 3`, Build command `pip install -r requirements.txt`, Start command `gunicorn app:app`, plus environment variable `PYTHON_VERSION = 3.12.3`.
4. Click **Create Web Service**. After the build finishes, your site is live at the `onrender.com` URL.

Note: the free plan sleeps after about 15 minutes of inactivity, so the first visit afterwards can take around a minute.

## Limitations
This is an educational project, not a medical device. The data covers one population (Pima Indian women aged 21+), is small, and has heavy hidden missingness in `Insulin` and `SkinThickness`. It must not be used for diagnosis.

## Tech stack
Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, Flask, Gunicorn.
