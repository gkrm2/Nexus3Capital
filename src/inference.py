# inference.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import GPT2Model, GPT2Config
from peft import LoraConfig, TaskType, get_peft_model
from configs import WINDOW_SIZE
from concurrent.futures import ProcessPoolExecutor, as_completed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("inferenceOutputs/results", exist_ok=True)
os.makedirs("inferenceOutputs/plots", exist_ok=True)

# MODEL
class GPT2ForRegression(nn.Module):
    def __init__(self, input_dim=1, hidden_size=64, window_size=WINDOW_SIZE, multi_step=24):
        super().__init__()
        self.window_size = window_size
        self.multi_step = multi_step
        self.proj = nn.Linear(input_dim, hidden_size)

        config = GPT2Config(n_positions=window_size, n_embd=hidden_size, n_layer=2, n_head=4)
        self.gpt2 = GPT2Model(config)

        lora_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                              r=8, lora_alpha=16, lora_dropout=0.1,
                              target_modules=["c_attn"])
        self.gpt2 = get_peft_model(self.gpt2, lora_cfg)
        self.regressor = nn.Linear(hidden_size, multi_step)

    def forward(self, x):
        x_embed = self.proj(x)
        out = self.gpt2(inputs_embeds=x_embed)
        last_token = out.last_hidden_state[:, -1, :]
        return self.regressor(last_token)

# FORECAST FUNCTION
def forecast_utility(args):
    utility_name, horizon, frequency = args
    print(f"\nForecasting {utility_name} | Horizon={horizon} | Freq={frequency}")

    csv_path = f"Data/{utility_name}.csv"
    model_path = f"trainedModels/{utility_name}_best_model.pth"
    scaler_path = f"trainedModels/{utility_name}_scaler.pkl"

    # Validate files
    for p in [csv_path, model_path, scaler_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found!")

    # Load data
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Load scaler & scale
    scaler = joblib.load(scaler_path)
    load_scaled = scaler.transform(df[["load"]].astype(float).values)

    # Load checkpoint & determine horizon
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if "regressor.weight" in checkpoint:
        saved_horizon = checkpoint["regressor.weight"].shape[0]
    elif "regressor.bias" in checkpoint:
        saved_horizon = checkpoint["regressor.bias"].shape[0]
    else:
        raise RuntimeError("Cannot infer model output size")

    # Build model
    model = GPT2ForRegression(window_size=WINDOW_SIZE, multi_step=saved_horizon).to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # Input window
    if len(load_scaled) < WINDOW_SIZE:
        raise RuntimeError(f"Not enough data (need {WINDOW_SIZE} rows)")

    inp = torch.tensor(load_scaled[-WINDOW_SIZE:, 0:1], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Forecast
    with torch.no_grad():
        pred_scaled = model(inp).cpu().numpy().reshape(-1, 1)
    pred = scaler.inverse_transform(pred_scaled).flatten()

    # Build timestamps
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    freq_map = {"hourly": "h", "daily": "D", "weekly": "W"}
    ts_future = pd.date_range(last_ts, periods=saved_horizon + 1, freq=freq_map[frequency])[1:]

    # Results
    result_df = pd.DataFrame({"timestamp": ts_future, "forecast_load": pred})

    # Save outputs
    csv_out = f"inferenceOutputs/results/{utility_name}_{frequency}_{saved_horizon}h.csv"
    json_out = f"inferenceOutputs/results/{utility_name}_{frequency}_{saved_horizon}h.json"
    plot_out = f"inferenceOutputs/plots/{utility_name}_{frequency}_{saved_horizon}h.png"

    result_df.to_csv(csv_out, index=False)
    result_df.to_json(json_out, orient="records", indent=4)

    # Plot recent + forecast
    plt.figure(figsize=(10, 4))
    n_recent = min(48, len(df))
    plt.plot(df["timestamp"].values[-n_recent:], df["load"].values[-n_recent:], label="Recent Load")
    plt.plot(ts_future, pred, label="Forecast", marker="o")
    plt.xlabel("Timestamp")
    plt.ylabel("Load")
    plt.title(f"{utility_name} Forecast ({frequency})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out)
    plt.close()

    print(f"Saved: {csv_out}, {json_out}, {plot_out}")
    return result_df

# RUN ALL UTILITIES IN PARALLEL
if __name__ == "__main__":
    csv_files = [f.replace(".csv", "") for f in os.listdir("Data") if f.endswith(".csv")]
    horizon = 24
    frequency = "hourly"

    results = {}
    args_list = [(u, horizon, frequency) for u in csv_files]

    # Parallel processing
    with ProcessPoolExecutor(max_workers=min(len(args_list), os.cpu_count())) as executor:
        futures = [executor.submit(forecast_utility, args) for args in args_list]
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                print(f"Error during inference: {e}")

    print("\nInference completed.")
