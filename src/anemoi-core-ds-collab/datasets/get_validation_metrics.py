import json, os, mlflow
from anemoi.utils.mlflow.auth import TokenAuth


# Load cached ECMWF token
#with open(os.path.expanduser("~/.config/anemoi/mlflow-token.json")) as f:
#    token_data = json.load(f)

#token = token_data.get("access_token") or token_data.get("token")
#os.environ["MLFLOW_TRACKING_TOKEN"] = token

auth = TokenAuth("https://mlflow.ecmwf.int")
auth.login()
auth.authenticate()
mlflow.set_tracking_uri("https://mlflow.ecmwf.int")
client = mlflow.tracking.MlflowClient()

runs = [
    ("099", "e3c6bf1fa249477699c6d74769ef162c"),
    ("199", "7d1595695bb1413795525958e2a97d3f"),
    ("299", "39804b6a133e49629e49ddae4fe753f3"),
    ("399", "2de1c0bc73d642be88d00230a61f591f"),
    ("499", "0de9946c171a47a6ad79f745fbe37c26"),
    ("599", "c8010fc197d249e3bcd48bad824b5e56"),
    ("699", "89caf825ad534a4e8eb8dc6ee28da769"),
    ("799", "1e2ddd6cbddc419aac74fdc4519e4b53"),
]

print(f"{'epoch':<8} {'val_weighted_mse_loss_epoch'}")
print("-" * 40)
for epoch, run_id in runs:
    run = client.get_run(run_id)
    metrics = run.data.metrics
    val_keys = [k for k in metrics if "val" in k.lower() and "system" not in k.lower()]
    if epoch == runs[0][0]:
        print(f"  [available val keys: {val_keys}]")
    val = (metrics.get("val/weighted_mse_loss_epoch") or
           metrics.get("val_weighted_mse_loss_epoch") or
           metrics.get("val/loss_epoch") or
           (metrics.get(val_keys[0]) if val_keys else None) or
           "NOT FOUND")
    print(f"{epoch:<8} {val}")