# import json
import io
import pandas as pd
from flask import Flask, request, jsonify
from model_loading import load, load_blob_client
from datetime import datetime 
import argparse
# import os
import logging
from preprocessing import Preprocess
from column_transforms_drift_data import get_drift_handling_pipeline

st = datetime.now()
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "t")

def get_args():
    parse = argparse.ArgumentParser()

    parse.add_argument("--StorageAccount", type=str, default="churnprediction1")
    parse.add_argument("--Container", type=str, default="retrained-models")
    parse.add_argument("--LoadLatestModel", type=str2bool, default=True)
    parse.add_argument("--ModelName", type=str, default="churn_classififer.pkl")
    parse.add_argument("--Origin", default="Retraining", type=str)
    parse.add_argument("--AlterLog", default="logData.json", type=str)

    args = parse.parse_args()
    return args

args = get_args()
StorageAccount = args.StorageAccount
Container = args.Container 
LoadLatestModel = args.LoadLatestModel
ModelName = args.ModelName
origin = args.Origin
AlterLog = args.AlterLog


logging.info(f"-------Given Args-------: \n \
                    {StorageAccount = }\n \
                    {Container = }\n \
                    {LoadLatestModel = }\n \
                    {ModelName = }\n \
                    {origin = }\n------------------------------------")

logging.info("loading model...")

model, model_info = load(StorageAccount ,Container, LoadLatestModel, ModelName, origin)
print(f"model info : {model_info}")


model_version = model_info['version id']
input_features = model['Features']
model = model['Model']

# if AlterLog.endswith("json"):
#     from model_loading import load_blob_client
#     json_blob_client, _ = load_blob_client(AlterLog)


logging.info("Loading logs...")
log_client, features = load_blob_client(model_version+".csv", input_features)


logging.info("finished loading model...")
en = datetime.now()

print(f"=====================Total time taken.... {(en-st)}s============================")

@app.route("/")
def home():
    if model is None:
        return {"status": "model failed to load new"}, 500
    return {"status": f"okay", "model-version":model_version}


@app.route("/model_info")
def returnmodel_info():
    return jsonify({"model info":model_info})


@app.route("/predict", methods=["POST"])
def score():
    try:

        input_json = request.get_json(force=True)
        data = input_json.get("data")


        if data is None:
            return jsonify({"error": "No data provided"}), 400

        data = pd.DataFrame(data)

        predictions = model.predict(data)

        log_df = pd.DataFrame(columns=features)

        try:
            for i in range(len(data)):
                values_row = [val.item() if hasattr(val, "item") else val for val in data.iloc[i].values]
                values_row.insert(0, datetime.now())
                values_row.append(predictions[i])
                log_df.loc[len(log_df)] = values_row
                
                exist_data = log_client.download_blob().readall()
                exist_df = pd.read_csv(io.StringIO(exist_data.decode("utf-8")), sep=",")

                whole_df = pd.concat([exist_df, log_df], ignore_index=True)
                whole_df.drop_duplicates(inplace=True)
                buffer_log = io.StringIO()
                whole_df.to_csv(buffer_log, index=False)
                log_client.upload_blob(buffer_log.getvalue(), max_concurrency=-1,overwrite=True)
        except Exception as e:
            logging.info(f"Error updating log {e}")
            
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":

    app.run(host="0.0.0.0", port= 80, debug=False)
