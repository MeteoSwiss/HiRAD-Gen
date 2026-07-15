import os
import mlflow
import matplotlib.pyplot as plt
from anemoi.utils.mlflow.auth import TokenAuth

auth = TokenAuth("https://mlflow.ecmwf.int")
auth.login()
auth.authenticate()
mlflow.set_tracking_uri("https://mlflow.ecmwf.int")
client = mlflow.tracking.MlflowClient()

# --- Training loss from the original b24 run ---
B24_RUN_ID = "b24ed243e0504f98b9cf48396f871eac"

# Find the parent run, then collect all child runs which hold the actual training metrics
# Search all runs with the same name — resumed runs share the run name
all_runs = client.search_runs(
    experiment_ids=["642"],
    filter_string="tags.`mlflow.runName` = 'deterministic_medium_training_set_all_vars_era_to_cosmo_downscaling_diffusion_training'",
    max_results=200,
)
print(f"Found {len(all_runs)} runs with this name")
for r in sorted(all_runs, key=lambda r: r.info.start_time):
    loss_keys = [k for k in r.data.metrics if "loss" in k.lower() and "system" not in k.lower()]
    tags_of_interest = {k: v for k, v in r.data.tags.items() if any(x in k.lower() for x in ["resumed", "forked", "parent"])}
    print(f"  {r.info.run_id[:8]}  start={r.info.start_time}  loss_metrics={loss_keys[:3]}  tags={tags_of_interest}")

# Use runs that have training loss metrics, not forked from b24
training_runs = [r for r in all_runs
                 if r.data.tags.get("forkedRunId") is None
                 and any("loss" in k for k in r.data.metrics)]
print(f"\n{len(training_runs)} runs have loss metrics")
sample = training_runs[0] if training_runs else all_runs[0]
print(f"Sample loss metrics: {[k for k in sample.data.metrics if 'loss' in k.lower()]}")

# Fetch full training loss history (paginated)
def get_metric_history(run_id, metric_key):
    history = client.get_metric_history(run_id, metric_key)
    return sorted(history, key=lambda m: m.step)

train_metric_key = "train/weighted_mse_loss_epoch"
print(f"Fetching training loss history across {len(training_runs)} training runs...")
train_history = []
for r in sorted(training_runs, key=lambda r: r.info.start_time):
    train_history.extend(get_metric_history(r.info.run_id, train_metric_key))
train_history = sorted(train_history, key=lambda m: m.step)

if not train_history:
    available = [k for k in sample.data.metrics if "loss" in k.lower()]
    print(f"Key '{train_metric_key}' not found. Available loss metrics: {available}")
    if available:
        train_metric_key = available[0]
        train_history = []
        for r in sorted(training_runs, key=lambda r: r.info.start_time):
            train_history.extend(get_metric_history(r.info.run_id, train_metric_key))
        train_history = sorted(train_history, key=lambda m: m.step)

train_steps = [m.step for m in train_history]
train_values = [m.value for m in train_history]

# --- Validation loss from the per-checkpoint runs ---
val_runs = [
    ("019", "2e288d9719194e598fc22dc5c244a696"),
    ("039", "2b588683b427466699fd991e987eab59"),
    ("059", "9122e20d2ba5427080b16668fe6113ab"),
    ("079", "39f2ba5e00b2447da539830642443ba6"),
    ("099", "cf87b8e54cdd46c39fa29e69639b836f"),
    ("119", "3bf1568b68314e9d81b86d79d407a191"),
    ("139", "dc27d11323f345f1b9c16d1c50394d7c"),
    ("159", "414c9a29aa32479e883a54e341e6aad6"),
    ("179", "9eed09aa16184beb918ad04dca90467a"),
    ("199", "74431e9e76ca4712bf2cf7e1abf0a846"),
    ("219", "e172adabe329494380d4cd7286a448f7"),
    ("239", "7278b5ffd9f145cbb0acba8ad6acf80d"),
    ("259", "6762b1e73d6a4fafb8a64163c7b5765a"),
    ("279", "c6a35b8c3203449c8002f1cf716185e2"),
    ("299", "d3adb8e3fed14316ad664bcbaaf9e9fb"),
    ("319", "e1bf64efb0834c52a8365afa0bf875c6"),
    ("339", "eaa3ce4bac964a399254ae0632dc16ec"),
    ("359", "14a5cefd64d54999b4b19e3febd63b83"),
    ("379", "42c3e8cb10a5403ba35ffe64dfaf7135"),
    ("399", "1a3b156e34574f218745d5df9a362257"),
    ("419", "60ed80c141054b33a2015bd0e8a96150"),
    ("439", "2d2b2c5cd2904960bee24b04e55b17d6"),
    ("459", "6a1553f690a04fb5ac542937fa75277c"),
    ("479", "b3ed85b97fdd43329a51906ffcc55288"),
    ("499", "71e99e141b0a4afd9f1775e578b5b907"),
    ("519", "6d6ac42669ec4f29a5b242a5117b667d"),
    ("539", "8d1a4cf1159f4a6e95adb3fa5a9d7ba9"),
    ("559", "23339c07547d4ed9aa3debf1d2acad99"),
    ("579", "a1bda3dc21e24fc99cc22a3203c6c9c6"),
    ("599", "06a2f96632244e71b7a590669ae32010"),
    ("619", "4074ab31ce854d85ad76f0f82f42a455"),
    ("639", "bafdc460ba9845e19c79b7a09404fb93"),
    ("659", "dd3fe792382c48d2b160a9d6d5008c8f"),
    ("679", "e0879b1f78b54c52b7fdf136624770fe"),
    ("699", "6d7b4a3f67d34da6a385dbadca8ac4cc"),
    ("719", "10c98392eb284c178631b3b7fe18a0ac"),
    ("739", "bb5944c7979f4575accb9a9f6ab93367"),
    ("759", "2179bbf894ed4751a79da86316211dcb"),
    ("779", "5dfc58c024be43218c5c0ef0c6ffe4d3"),
    ("799", "f259f93b89f3439085240168e79a04e2"),
]

val_epochs, val_values = [], []
val_metric_keys = [
    "val/weighted_mse_loss_epoch",
    "val_weighted_mse_loss_epoch",
    "val/loss_epoch",
    "val/loss",
]

print("Fetching validation losses...")
for epoch_str, run_id in val_runs:
    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics
        val = None
        for key in val_metric_keys:
            if key in metrics:
                val = metrics[key]
                break
        if val is None:
            available = [k for k in metrics if "val" in k.lower() and "loss" in k.lower()]
            if available:
                val = metrics[available[0]]
                print(f"  epoch {epoch_str}: using key '{available[0]}'")
        if val is not None:
            val_epochs.append(int(epoch_str))
            val_values.append(val)
        else:
            print(f"  epoch {epoch_str}: no val loss metric found. Available: {list(metrics.keys())[:10]}")
    except Exception as e:
        print(f"  epoch {epoch_str}: ERROR {e}")

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(12, 5))

if train_steps and train_values:
    ax1.plot(train_steps, train_values, color="steelblue", alpha=0.7, label="train loss (by step)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Train loss", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

if val_epochs and val_values:
    ax2 = ax1.twinx()
    # Convert epoch numbers to approximate steps (500 steps/epoch based on config)
    val_steps = [e * 500 for e in val_epochs]
    ax2.plot(val_steps, val_values, "o-", color="tomato", label="val loss (per checkpoint)")
    ax2.set_ylabel("Val loss", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

ax1.set_title("Training vs Validation Loss — b24 run")
fig.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "train_val_loss.png")
plt.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")
plt.show()
