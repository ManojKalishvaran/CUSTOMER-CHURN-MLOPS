from detect_drift_get_features_to_handle import detect_save_extract_drift, get_features_to_handle
from column_transforms_drift_data import get_drift_handling_pipeline
from retrain_with_oldandnew import train_model_with_both_data
from copy import deepcopy

import requests
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from preprocessing import Preprocess

import io
import pickle
import pprint
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s \
                    - %(levelname)s - %(message)s")


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "t")

def get_args():
    parse = argparse.ArgumentParser()

    parse.add_argument("--PassValidation", default=False, type=str2bool, help="Pass validation of model")
    parse.add_argument("--BaseLine", default=69, type=int, help="Base Line Accuracy")
    parse.add_argument("--DataContainer", default="retraining-data", type=str, help="Container of data" )
    parse.add_argument("--TrainingDataContainer", default="training-data", type=str, help="Container of data" )
    parse.add_argument("--ModelContainer", default="retrained-models", type=str, help="Container of model storage" )
    parse.add_argument("--StoreModel", default=True, type=str2bool, help="Whether to store model in blob")
    parse.add_argument("--SelectLatestData", default=False, type=str2bool, help="whether to select data based on timestamp not name")
    parse.add_argument("--StorageAccount", default="churnprediction1", type=str, help="Storage account name" )

    args = parse.parse_args()
    return args

def read_data():
    """----Read the data----"""

    credential = DefaultAzureCredential()
    # storage_account = "churnprediction1"
    # container_name = "training-data"
    container_name = data_container

    account_url = f"https://{storage_account}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(account_url=account_url \
                                            , credential=credential)

    container_client = blob_service_client.get_container_client(container_name)

    if select_latest_data: 
        logging.info("Getting latest data >>>")
        blobs = container_client.list_blobs()
        latest_blob = max(blobs, key=lambda b:b.last_modified)
        blob_client = container_client.get_blob_client(latest_blob.name)
        blob_data = blob_client.download_blob().readall()
    else:
        blob_name = "customer_churn.csv"
        logging.info(f"Getting defined client {blob_name}>>>")
        blob_client = container_client.get_blob_client(blob_name)

        blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(io.StringIO(blob_data.decode("utf-8")), sep=",")
    return df


def split_data(data):
    """----Split the data----"""

    if "Churn" in data.columns and "Churn" in data.columns:
        X = data.drop(columns=['Churn', 'CustomerID'], axis=1)
    else: 
        X = data.drop(columns=['Churn'], axis=1)

    y = data['Churn']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=87, \
                         shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test



def store_model_blob(pipe, meta):
    credential = DefaultAzureCredential()

    # storage_account = "churnprediction1"
    # model_container_name = "trained-models"

    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(model_container_name)
    try: 
        container_client.create_container()
    except Exception as e: 
        logging.info(f"Container already exists.")

    blob_name = "churn_classififer.pkl"
    blob_client = container_client.get_blob_client(blob_name)

    model_bytes = pickle.dumps(pipe)
    blob_client.upload_blob(model_bytes, overwrite=True, metadata=meta, max_concurrency=-1)

    logging.info(f'fininshed uploading')
    logging.info("Model training uploading completed !!!")

def register_data(data):
    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(training_container)
    try:
        container_client.create_container()
    except Exception as e:
        logging.info("Training data container exists")
    
    data_blob_name = "customer_churn.csv"
    blob_client = container_client.get_blob_client(data_blob_name)

    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    try: 
        blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True, max_concurrency=-1)
        logging.info("Training Data has been registered!")
    except Exception as e:
        logging.info(f"Failed to upload training data...{e}")



def validate_model(model, test_x, test_y):
    scores = {}
    y_pred = model.predict(test_x)
    metricses = [accuracy_score, f1_score, precision_score, \
                    roc_auc_score, recall_score]
    for metric_ in metricses: 
        namen = metric_.__name__
        score = metric_(test_y, y_pred)
        scores[namen]=score
    return scores

def train_model(data, meta):
    """----Train----"""

    x_train, x_test, y_train, y_test = data
    pipe = Pipeline(steps=[
        ("Preprocess", Preprocess()), 
        ("Model", RandomForestClassifier(
            n_estimators=125, \
            n_jobs=-1, random_state=39))
    ])

    empty_model = Pipeline(steps=[
        ("Preprocess", Preprocess()), 
        ("Model", RandomForestClassifier(
            n_estimators=125, \
            n_jobs=-1, random_state=39))
    ])

    pipe.fit(x_train, y_train)
    logging.info("Training Complete")

    """----Validate----"""


    scores = validate_model(pipe, x_test, y_test)
    pprint.pprint(scores)

    return pipe, scores, empty_model

def store(pipe, meta, scores, input_data):
    meta['scores']=str([(k,v) for k,v in scores.items()])
    # print(f"Input features:\n {input_data}")
    input_features =list(input_data.columns)
    model_with_features = {"Model":pipe, "Features":input_features}

    if pass_validation: 
        if store_model: 
            store_model_blob(model_with_features, meta)
    else:

        if store_model:
                logging.info(f"Model got approved and being stored-✔ | Accuracy: {scores['accuracy_score']*100:.0f}%")
                store_model_blob(model_with_features, meta)
        else:
                logging.info(f"Model got approved and not stored-✔ | Accuracy: {scores['accuracy_score']*100:.0f}%")


def is_model_okay(scores):
    base_accuracy = base_line/100
    if scores['accuracy_score']<base_accuracy:
        logging.info("Model's accuracy is not enough- Look up is needed!!!")
        return False
    else: 
        logging.info("Model's accuracy is enough so deployment is next...")
        return True
    

def analyse_handle_drift_train(df, model, xtrain, ytrain):
    dirft_score_by_column, full_drift_metrices = detect_save_extract_drift(storage_account, training_container, data_container)
    features_to_transform = get_features_to_handle(model.named_steps['Model'], xtrain, ytrain, dirft_score_by_column, df)
    
    updated_preprocessing = get_drift_handling_pipeline(features_to_transform)
    
    updated_pipe = Pipeline(
        steps = [
            ("Preprocess", updated_preprocessing),
            ("Model", RandomForestClassifier(
            n_estimators=125, \
            n_jobs=-1, random_state=39))
        ]
    )

    df_data = df.copy()
    X_train, X_test, y_train, y_test = split_data(df_data)

    updated_pipe.fit(X_train, y_train)

    scores_updated = validate_model(updated_pipe, X_test, y_test)
    pprint.pprint(f"Score Drift Handled model: \n {scores_updated}")

    decision_updated = make_decision(updated_pipe, scores_updated, X_train, y_train)
    return updated_pipe, decision_updated, scores_updated

def evaluate_model_with_cv(model, X, y, cv_folds=5, base_accuracy=0.8, scoring='accuracy'):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    logging.info(f"Cross-Validation Scores for each fold: {scores}")
    mean_score = scores.mean()
    std_score = scores.std()
    logging.info(f"Mean {scoring}: {mean_score:.4f}")
    logging.info(f"Standard Deviation of {scoring}: {std_score:.4f}")

    if mean_score >= base_accuracy:
        logging.info(f"Decision: KEEP the model (mean {scoring} >= {base_accuracy})")
        return True, mean_score
    else:
        logging.info(f"Decision: REJECT the model (mean {scoring} < {base_accuracy})")
        return False, mean_score

def make_decision(model, scores, x, y):
    is_okay = is_model_okay(scores)
    if is_okay: 
        keep, mean_score = evaluate_model_with_cv(model, x, y)
        return keep
    else: 
        return False

def main():
    logging.info("Parsing Args >>>")
    args = get_args()

    global pass_validation
    pass_validation = args.PassValidation 
    global base_line
    base_line = args.BaseLine
    global data_container
    data_container = args.DataContainer
    global store_model
    store_model = args.StoreModel 
    global select_latest_data
    select_latest_data = args.SelectLatestData
    global storage_account
    storage_account = args.StorageAccount
    global model_container_name
    model_container_name = args.ModelContainer
    global training_container
    training_container = args.TrainingDataContainer

    meta_data = {
        "base_line":str(base_line), 
        "pass_validation":str(pass_validation), 
        "data_container":str(data_container), 
        "store_model":str(store_model), 
        "select_latest_data":str(select_latest_data)
    }   

    logging.info(f"-------Given Args-------: \n \
                {pass_validation = }\n \
                {base_line = }\n \
                {data_container = }\n \
                {store_model = }\n \
                {select_latest_data = }\n------------------------------------")

    logging.info("Provisioning Data >>>")
    df = read_data()
    logging.info(f"There {df.duplicated().sum()} number of duplicated samples removing....")
    df.drop_duplicates(inplace=True)
    print(df.head())

    logging.info("Splitting data >>>")
    X_train, X_test, y_train, y_test = split_data(df)

    logging.info("Training Model >>>")
    pipe, scores, empty_model = train_model([X_train, X_test, y_train, y_test], meta_data)
    empty_model_train_both = deepcopy(empty_model)

    logging.info("Making Decision >>>")
    decision = make_decision(empty_model, scores, X_train, y_train)

    logging.info(f"Decision was {decision} hardcoding to False")

    # decision = False
    if decision: 
        logging.info("Model can be deployed !!! Saving... >>>")
        meta_data["Retrained_on"] = "Original new data"
        store(pipe, meta_data, scores, X_train)
        logging.info("Training has been successful: Default training. Registering data...")
        register_data(df)
        logging.info("From default retraining...")

    else: 
        logging.info("===============================Model failed to prove... Analysis is getting kicked off >>>===============================")
        updated_pipe, decision_updated, scores_updated = analyse_handle_drift_train(df, pipe, X_train, y_train)
        if decision_updated:
            logging.info("Drift handled Model is okay to be deployed. Storing !!!")
            meta_data["Retrained_on"] = "Drift handled new data"
            store(updated_pipe, meta_data, scores_updated, X_train)
            
            logging.info("Training has been successful: Drift handling. Registering data...")
            register_data(df)
            
            logging.info("From Drift Handiling...")
        else:
            logging.info("===============================Model failed in data drift handling... Training on both data is getting kicked off >>>===============================")
            model_both, scores_both, empty_model_both, X_train_both, y_train_both = train_model_with_both_data(empty_model_train_both, training_container, data_container, storage_account)
            
            decision_both = make_decision(empty_model_both, scores_both, X_train_both, y_train_both)
            if decision_both:
                meta_data["Retrained_on"] = "Trained on both data"
                store(model_both, meta_data, scores_both, X_train_both)

                logging.info("Training has been successful: Both data. Registering data...")
                register_data(df)

                logging.info("From Training on both data...")

            else: 
                logging.info("Even Drift Handled model and trained on both data is not good. Raising a mail...")

                url = "https://prod-20.northcentralus.logic.azure.com:443/workflows/297542e75c3d439d9c7c0ed19418424e/triggers/When_a_HTTP_request_is_received/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2FWhen_a_HTTP_request_is_received%2Frun&sv=1.0&sig=ld_1JlFuje2QQZIGD40xNFHmZE7adLiVTTvxaEJr8jE"

                table_html = "<table border='1' style='border-collapse: collapse;'>"
                table_html += "<tr><th>Metric</th><th>Value</th></tr>"

                for key, value in scores_updated.items():
                    table_html += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"

                table_html += "</table>"

                body = f"""
                Hello,<br><br>
                This is an automated report from ML model Retraining pipeline. <h2>The model is not good even after handling drift and trained with both old and new data.</h2><br><br>
                <b>Model Evaluation Metrics:</b><br>
                {table_html}<br>
                Please review and take any necessary action.<br><br>
                """

                data = {
                    "to": "manojcolan18@gmail.com",
                    "subject": "⚠️ Model Performace Alert - Retraining",
                    "body": body
                }


                res = requests.post(url, json=data)
                logging.info(res.status_code)
                logging.info("Mail has beed sent....")
                
    return pipe, scores, decision, X_train, y_train

if __name__=="__main__":
    try:
        main()
        logging.info(f"Retraining is completed successfully. Raising a mail...")

        url = "https://prod-20.northcentralus.logic.azure.com:443/workflows/297542e75c3d439d9c7c0ed19418424e/triggers/When_a_HTTP_request_is_received/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2FWhen_a_HTTP_request_is_received%2Frun&sv=1.0&sig=ld_1JlFuje2QQZIGD40xNFHmZE7adLiVTTvxaEJr8jE"

        body = f"""
        Hello,<br><br>
        This is an automated report from ML model Retraining pipeline. <h2>Model Retraining unsuccessful. The updated model has been deployed</h2><br>
        """

        data = {
            "to": "manojcolan18@gmail.com",
            "subject": "⚠️ Model Retraining Alert - Success 👍🏼",
            "body": body
        }
        res = requests.post(url, json=data)
        logging.info(res.status_code)
        logging.info("Mail has beed sent....")
    except Exception as e:
            logging.info(f"There is an error in retraining. Raising a mail...\n Error: {e}")

            url = "https://prod-20.northcentralus.logic.azure.com:443/workflows/297542e75c3d439d9c7c0ed19418424e/triggers/When_a_HTTP_request_is_received/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2FWhen_a_HTTP_request_is_received%2Frun&sv=1.0&sig=ld_1JlFuje2QQZIGD40xNFHmZE7adLiVTTvxaEJr8jE"

            body = f"""
            Hello,<br><br>
            This is an automated report from ML model Retraining pipeline. <h2>Model Retraining was unsuccessful. </h2><br>
            <br><b>Error:</b>
            {e}<br><br>
            Please review and take any necessary action.<br><br>
            """

            data = {
                "to": "manojcolan18@gmail.com",
                "subject": "⚠️ Model Retraining Alert - Failed 👎🏼",
                "body": body
            }
            res = requests.post(url, json=data)
            logging.info(res.status_code)
            logging.info("Mail has beed sent....")

