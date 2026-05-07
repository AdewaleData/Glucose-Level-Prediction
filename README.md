# Glucose-Level-Prediction

Regression models that estimate **blood glucose** from **non-invasive physiological inputs** (demographics, anthropometrics, and PPG-related features). This repository bundles exploratory analysis, trained **scikit-learn** pipelines, tabular data, and a small **FastAPI** service for local or hosted inference.

**Disclaimer:** This is research and educational software. It is **not** a medical device and must **not** be used for clinical treatment decisions without proper validation, regulatory review, and supervision by qualified health professionals.

---

## Repository layout

| Path | Description |
|------|-------------|
| `model/` | Jupyter notebook (`glucose_prediction_analysis.ipynb`) and serialized pipelines (`*.pkl`). |
| `dataset/` | Training and analysis CSV (`augmented_dataset_new.csv`). |
| `api/` | FastAPI application: loads the tuned model, applies the same row-wise feature engineering, exposes JSON prediction endpoints. |
| `run_api.ps1` | Convenience script to install API dependencies and start Uvicorn (Windows). |

---

## Requirements

- **Python** 3.10+ (3.11–3.13 tested in development).
- **Git** (to clone this repository).

The saved pipelines were produced with **scikit-learn 1.6.1**. The API pins that version so `joblib`/`pickle` deserialization stays reliable. Using a different minor version may trigger load warnings or failures.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/AdewaleData/Glucose-Level-Prediction.git
cd Glucose-Level-Prediction
```

### 2. Notebook (EDA and training)

1. Create a virtual environment and install packages as needed by the notebook (at minimum: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `jupyter`).
2. Open `model/glucose_prediction_analysis.ipynb`.
3. Set the Jupyter **working directory** to the **repository root** (the folder that contains `dataset/` and `model/`). The notebook resolves `PROJECT_ROOT` from `Path.cwd()` and steps up one level when the kernel cwd is `model/`.

Data path used in the notebook: `dataset/augmented_dataset_new.csv`.

### 3. REST API (inference)

From the **repository root**:

```bash
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

On Windows you can instead run:

```powershell
.\run_api.ps1
```

- Interactive OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: `GET /health`

**Default model file:** `model/best_glucose_model_tuned.pkl`  
Override with environment variable:

```text
GLUCOSE_MODEL_PATH=/absolute/path/to/your_model.pkl
```

#### Prediction API

The service accepts **twelve** raw feature fields per record (names and types must match the schema). Engineered columns are computed server-side before inference.

| Field | Role |
|-------|------|
| `Gender`, `Age`, `Height`, `Weight`, `BMI`, `Peak_Diff`, `index` | Subject and session context |
| `PPG_Signal`, `Heart_Rate`, `Pulse_Area`, `Systolic_Peak`, `Diastolic_Peak` | PPG / pulse morphology |

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness and resolved model path. |
| `POST` | `/v1/predict` | Single JSON object → `{ "glucose_level": number }`. |
| `POST` | `/v1/predict_batch` | `{ "rows": [ { ... }, ... ] }` → `{ "glucose_level": [ ... ] }`. |

**Example (`curl`)**

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"Gender\":1,\"Age\":23,\"Height\":175,\"Weight\":56,\"BMI\":18.29,\"Peak_Diff\":13.19,\"index\":0,\"PPG_Signal\":513,\"Heart_Rate\":80,\"Pulse_Area\":418,\"Systolic_Peak\":520.77,\"Diastolic_Peak\":507.54}"
```

*(On bash/macOS, replace `^` line continuations with `\`.)*

**Production notes:** The API enables permissive CORS for prototyping. For public deployment, restrict origins, add authentication, rate limiting, HTTPS, and monitoring.

---

## Data and Git

The primary CSV under `dataset/` is large (~60 MB). GitHub accepts it under the per-file hard limit but may warn above 50 MB. For heavier datasets or frequent updates, consider **Git LFS**, a release asset, or external object storage and document the download step here.

---

## Author

Maintained in connection with the [AdewaleData/Glucose-Level-Prediction](https://github.com/AdewaleData/Glucose-Level-Prediction) GitHub repository.
