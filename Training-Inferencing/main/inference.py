import io
import pandas as pd
from flask import Flask, request, jsonify
from model_loading import load, load_blob_client
from datetime import datetime 
import argparse
import os
import logging
from preprocessing import Preprocess


st = datetime.now()
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "t")

def get_args():
    parse = argparse.ArgumentParser()

    parse.add_argument("--StorageAccount", type=str, default="churnprediction1")
    parse.add_argument("--Container", type=str, default="trained-models")
    parse.add_argument("--LoadLatestModel", type=str2bool, default=True)
    parse.add_argument("--ModelName", type=str, default="chrun-classifier.pkl")
    parse.add_argument("--Origin", default="CI-CD", type=str)

    args = parse.parse_args()
    return args

args = get_args()
StorageAccount = args.StorageAccount
Container = args.Container 
LoadLatestModel = args.LoadLatestModel
ModelName = args.ModelName
origin = args.Origin

logging.info(f"-------Given Args-------: \n \
                    {StorageAccount = }\n \
                    {Container = }\n \
                    {LoadLatestModel = }\n \
                    {ModelName = }\n \
                    {origin = }\n------------------------------------")

logging.info("loading model...")

model, model_info = load(StorageAccount ,Container, LoadLatestModel, ModelName, origin)

logging.info("Loading logs...")
log_client, features = load_blob_client()


logging.info("finished loading model...")
en = datetime.now()

print(f"=====================Total time taken.... {(en-st)}s============================")
# @app.route("/")
# def home():
#     # return {"status":"okay"}
@app.route("/")
def home():
    if model is None:
        return {"status": "model failed to load new"}, 500
    return {"status": "okay new"}


@app.route("/model_info")
def returnmodel_info():
    return jsonify({"model info":model_info})


@app.route("/predict", methods=["POST"])
def score():
    try:
        # Expect JSON input with features, e.g. {"data": [[5.1, 3.5, 1.4, 0.2]]}
        input_json = request.get_json(force=True)
        data = input_json.get("data")

        # Validate input
        if data is None:
            return jsonify({"error": "No data provided"}), 400

        data = pd.DataFrame(data)
        # Run prediction
        predictions = model.predict(data)

        log_df = pd.DataFrame(columns=features)

        for i in range(len(data)):
            values_row = [val.item() if hasattr(val, "item") else val for val in data.iloc[i].values]
            values_row.insert(0, datetime.now())
            values_row.append(predictions[i])
            log_df.loc[len(log_df)] = values_row


        exist_data = log_client.download_blob().readall()
        exist_df = pd.read_csv(io.StringIO(exist_data.decode("utf-8")), sep=",")

        whole_df = pd.concat([exist_df, log_df], ignore_index=True)
        buffer_log = io.StringIO()
        whole_df.to_csv(buffer_log, index=False)
        log_client.upload_blob(buffer_log.getvalue(), overwrite=True)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # Run the Flask app on port 5001 (matching Dockerfile EXPOSE)
    app.run(host="0.0.0.0", port= 80)
