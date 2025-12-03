# LLM-Based Short-Term Utility Load Forecaster

## Overview

This project implements a **short term utility load forecasting pipeline** using a **Local/LoRA-augmented GPT-2 model**. The solution is fully reproducible, runs locally, and produces **hourly forecasts for the next 24 hours** for multiple utilities.

---

## Directory Structure
Documents/ # Necessary documents  
src/  
├── trainPipeline.py # Training pipeline using GPT-2 + LoRA  
├── inference.py # Inference pipeline for forecasts  
├── configs.py # Configuration (window size, forecast horizon, data folder)  
├── Data/ # Input CSV files for each utility  
├── trainedModels/ # Saved models and scalers  
├── trainOutputs/ # Training metrics and plots  
├── inferenceOutputs/ # Forecast results and plots  
├── requirements.txt # Python dependencies  
└── venv/ # Virtual environment  


---

## Methodology

1. **Data Preparation**
   - Input CSVs contain hourly load measurements for 3 months.
   - Columns: `utility_name, timestamp, load`.
   - Data is **sorted by timestamp** and scaled using `StandardScaler`.
   - **Out-of-sample evaluation**: last 24 hours are reserved for holdout testing.

2. **Model**
   - **GPT-2** is used as the core modeling component.
   - **LoRA (Low-Rank Adaptation)** is applied for efficient fine-tuning.
   - Model input: sliding window of past load values (`WINDOW_SIZE`).
   - Model output: multi step forecast (`FORECAST_HORIZON = 24`).

3. **Training**
   - Split training data 80/20 for train/validation.
   - Early stopping based on validation loss.
   - Training is **parallelized** for multiple utilities using `ProcessPoolExecutor`.

4. **Inference**
   - Generates **hourly point forecasts** for the next 24 hours.
   - Saves results as **CSV, JSON, and plots**.
   - Parallelized across utilities for faster execution.
   - Assumes sufficient historical data (`>= WINDOW_SIZE` rows).

5. **Evaluation**
   - Metrics reported on holdout:  
     - MAE (Mean Absolute Error)  
     - RMSE (Root Mean Squared Error)  
     - MAPE (Mean Absolute Percentage Error)  
   - Plots of forecast vs actual load are saved for visual inspection.

---

## Assumptions & Notes

- Only **local/open-source tools** used; no external APIs.
- Scaler is fitted on training data only to **avoid data leakage**.
- Forecast horizon is fixed at 24 hours.
- Parallel processing is used for both training and inference.
- GPU is optional; CPU fallback is available.
- Training and inference are **fully scriptable** and reproducible.

---

## How to Run

### Clone the repository
```bash
   git clone https://github.com/gkrm2/Nexus3Capital.git
   cd src
```

### Locally (Python Virtual Environment)
```bash
    # Activate virtual environment
    source venv/Scripts/activate   # Windows
    # or
    source venv/bin/activate       # Linux/macOS

    # Install dependencies
    pip install -r requirements.txt

    # Run training
    python src/trainPipeline.py

    # Run inference
    python src/inference.py
```

### Dockerized Execution (Recommended)
```bash
    # Build Docker image (arm64)
    docker build -t utility-forecast:latest .

    # Run the container
    docker run --rm utility-forecast
```

## Outputs
### Training
    trainOutputs/metrics/results.json → training metrics for each utility
    trainOutputs/plots/ → training loss and validation plots

### Inference
    inferenceOutputs/results/ → forecast CSVs and JSONs for each utility
    inferenceOutputs/plots/ → forecast plots


## References
    GPT-2: HuggingFace Transformers

    LoRA: Low-Rank Adaptation
