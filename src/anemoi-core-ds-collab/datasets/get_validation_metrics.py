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

print(f"{'epoch':<8} {'val_weighted_mse_loss_epoch'}")
print("-" * 40)
for epoch, run_id in runs:
    run = client.get_run(run_id)
    metrics = run.data.metrics
    # Try likely key names
    val = (metrics.get("val/weighted_mse_loss_epoch") or
           metrics.get("val_weighted_mse_loss_epoch") or
           metrics.get("val/loss_epoch") or
           "NOT FOUND")
    print(f"{val}")