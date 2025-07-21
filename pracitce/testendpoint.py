import logging.config
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import Preprocess

import io
import pickle
import pprint
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
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
    blob_client.upload_blob(model_bytes, overwrite=True, metadata=meta)

    logging.info(f'fininshed uploading')
    logging.info("Model training uploading completed !!!")


def train_model(data, meta):
    """----Train----"""

    x_train, x_test, y_train, y_test = data
    pipe = Pipeline(steps=[
        ("Preprocess", Preprocess()), 
        ("Train", RandomForestClassifier(
            n_estimators=125, \
            n_jobs=-1, random_state=39))
    ])
    pipe.fit(x_train, y_train)
    logging.info("Training Complete")

    """----Validate----"""

    scores = {}
    y_pred = pipe.predict(x_test)
    metricses = [accuracy_score, f1_score, precision_score, \
                  roc_auc_score, recall_score]
    for metric_ in metricses: 
        namen = metric_.__name__
        score = metric_(y_test, y_pred)
        scores[namen]=score
    
    pprint.pprint(scores)

    return pipe, scores

def store(pipe, meta, scores):
    meta['scores']=str([(k,v) for k,v in scores.items()])

    if pass_validation: 
        if store_model: 
            store_model_blob(pipe, meta)
    else:

        if store_model:
                logging.info(f"Model got approved and being stored-✔ | Accuracy: {scores['accuracy_score']*100:.0f}%")
                # meta['scores']=scores
                store_model_blob(pipe, meta)
        else:
                logging.info(f"Model got approved and not stored-✔ | Accuracy: {scores['accuracy_score']*100:.0f}%")


def is_model_okay(scores):
    base_accuracy = base_line/100
    if scores['accuracy_score']<base_accuracy:
        # logging.info("Model's accuracy is not enough- Look up is needed!!!")
        return False
    else: 
        # logging.info("Model's accuracy is enough so deployment is next...")
        return True
    

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
    print(df.head())

    logging.info("Splitting data >>>")
    X_train, X_test, y_train, y_test = split_data(df)

    logging.info("Training Model >>>")
    pipe, scores = train_model([X_train, X_test, y_train, y_test], meta_data)
    return pipe, X_train, y_train