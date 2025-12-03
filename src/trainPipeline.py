# trainPipeline.py
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
from peft import LoraConfig, TaskType, get_peft_model
from configs import WINDOW_SIZE, FORECAST_HORIZON, DATA_FOLDER
from concurrent.futures import ProcessPoolExecutor, as_completed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("trainOutputs/metrics", exist_ok=True)
os.makedirs("trainOutputs/plots", exist_ok=True)
os.makedirs("trainedModels", exist_ok=True)

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=WINDOW_SIZE, multi_step=FORECAST_HORIZON):
        self.X, self.y = [], []
        for i in range(len(data) - window_size - multi_step):
            x = torch.tensor(data[i:i+window_size, 0:1], dtype=torch.float32)
            y = torch.tensor(data[i+window_size:i+window_size+multi_step, 0], dtype=torch.float32)
            self.X.append(x)
            self.y.append(y)

        self.X = torch.stack(self.X) if len(self.X) > 0 else torch.empty(0)
        self.y = torch.stack(self.y) if len(self.y) > 0 else torch.empty(0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# GPT-2 + LoRA MODEL
class GPT2ForRegression(nn.Module):
    def __init__(self, input_dim=1, hidden_size=64, window_size=WINDOW_SIZE, multi_step=FORECAST_HORIZON):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)

        config = GPT2Config(
            n_positions=window_size,
            n_embd=hidden_size,
            n_layer=2,
            n_head=4
        )

        self.gpt2 = GPT2Model(config)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )

        self.gpt2 = get_peft_model(self.gpt2, lora_cfg)
        self.regressor = nn.Linear(hidden_size, multi_step)

    def forward(self, x):
        x_embed = self.proj(x)
        out = self.gpt2(inputs_embeds=x_embed)
        last_token = out.last_hidden_state[:, -1, :]
        return self.regressor(last_token)


# TRAINING PIPELINE
def train_model_for_utility(args):
    csv_path, utility_name = args
    print(f"Training LLM (GPT-2 + LoRA) → {utility_name}")

    # Load & sort time series
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fit scaler ONLY on training part (avoids leakage)
    scaler = StandardScaler()
    load_scaled = scaler.fit_transform(df[["load"]].astype(float).values)

    # OUT OF SAMPLE HOLDOUT → last 24 hours
    test_data = load_scaled[-FORECAST_HORIZON:]
    train_val = load_scaled[:-FORECAST_HORIZON]

    # 80/20 split
    split = int(len(train_val) * 0.8)
    train_data = train_val[:split]
    val_data = train_val[split:]

    # Windows
    train_loader = DataLoader(TimeSeriesDataset(train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(val_data), batch_size=16)

    # Model
    model = GPT2ForRegression().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    patience = 10
    no_improve = 0
    epochs = 60

    hist = {"train_loss": [], "val_loss": []}

    start = time.time()

    # TRAINING LOOP
    for epoch in range(epochs):
        model.train()
        t_losses = []

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            t_losses.append(loss.item())

        # Validation
        model.eval()
        v_losses = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                v_losses.append(loss_fn(pred, y).item())

        train_loss = float(np.mean(t_losses)) if t_losses else 0.0
        val_loss = float(np.mean(v_losses)) if v_losses else 0.0

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train={train_loss:.6f} | Val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"trainedModels/{utility_name}_best_model.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Save scaler
    joblib.dump(scaler, f"trainedModels/{utility_name}_scaler.pkl")

    # Holdout evaluation (next 24h)
    model = GPT2ForRegression().to(DEVICE)
    model.load_state_dict(torch.load(f"trainedModels/{utility_name}_best_model.pth"))
    model.eval()

    # Last window
    last_window = train_val[-WINDOW_SIZE:]
    inp = torch.tensor(last_window[:, 0:1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pred_scaled = model(inp).cpu().detach().numpy().reshape(-1, 1)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = df["load"].values[-FORECAST_HORIZON:]

    mae = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

    # Save plots
    plt.figure(figsize=(8, 4))
    plt.plot(range(FORECAST_HORIZON), actual, label="Actual")
    plt.plot(range(FORECAST_HORIZON), pred, label="Forecast")
    plt.legend()
    plt.title(f"{utility_name} Holdout Forecast (24h)")
    plt.tight_layout()
    plt.savefig(f"trainOutputs/plots/{utility_name}_holdout.png")
    plt.close()

    return {
        "Utility": utility_name,
        "Model": "GPT2 + LoRA",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Model_Path": f"trainedModels/{utility_name}_best_model.pth",
        "Scaler_Path": f"trainedModels/{utility_name}_scaler.pkl"
    }


# MAIN → Train ALL utilities in parallel
if __name__ == "__main__":
    csv_files = [(os.path.join(DATA_FOLDER, f), f.replace(".csv", ""))
                 for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

    results = {}

    with ProcessPoolExecutor(max_workers=min(len(csv_files), os.cpu_count())) as executor:
        futures = [executor.submit(train_model_for_utility, args) for args in csv_files]
        for future in as_completed(futures):
            res = future.result()
            results[res["Utility"]] = res

    # Save all metrics
    with open("trainOutputs/metrics/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nTraining complete. Metrics saved.")
