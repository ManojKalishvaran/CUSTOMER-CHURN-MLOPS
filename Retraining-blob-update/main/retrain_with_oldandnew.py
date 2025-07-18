import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split 
import logging
import pprint
from azure.identity import DefaultAzureCredential
# from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
# from sklearn.metrics import accuracy_score, precision_score, \
#     recall_score, f1_score, roc_auc_score
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s \
                    - %(levelname)s - %(message)s")

def get_weights(old_data, new_data):
    weight_old = 1.0
    weight_new = 2.0

    weights_old = np.full(len(old_data), weight_old)
    weights_new = np.full(len(new_data), weight_new)

    sample_weights = np.concatenate([weights_old, weights_new])

    return sample_weights

def read_data(container_name, blob_service_client):
    container_client = blob_service_client.get_container_client(container_name)

    blobs = container_client.list_blobs()
    latest_blob = max(blobs, key=lambda b:b.last_modified)
    blob_client = container_client.get_blob_client(latest_blob.name)
    blob_data = blob_client.download_blob().readall()

    df = pd.read_csv(io.StringIO(blob_data.decode("utf-8")), sep=",")
    return df, container_client

def load_both_data():
        try:
            credential = DefaultAzureCredential()

            account_url = f"https://{storage_account}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
            
            reference_data, _ = read_data(train_container, blob_service_client)
            current_data, current_blob_client = read_data(current_data_container, blob_service_client)

            # reference_data.drop(columns=['Potability'], inplace=True)
            # current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
            # one_week_ago = datetime.utcnow()-timedelta(days=7)
            # current_data_last_week = current_data[current_data['timestamp']>=one_week_ago]
            logging.info(f'Total # Current Data {len(current_data)}\n Total # Reference Data {len(reference_data)}')

            # current_data_last_week.drop(columns=['timestamp', 'prediction'], inplace=True)
            return reference_data, current_data, blob_service_client, current_data_container
            
        except Exception as e: 
            logging.error(f"Error loading data : {e}")
            return None, None, None, None


def make_data_ready():
    reference_data, current_data, blob_service_client, current_data_container = load_both_data()

    reference_data.drop_duplicates(inplace=True)
    current_data.drop_duplicates(inplace=True)

    whole_data = pd.concat([reference_data, current_data], ignore_index=True)

    sample_weights = get_weights(reference_data, current_data)

    shuffled_indices = np.random.permutation(len(whole_data))

    whole_data = whole_data.iloc[shuffled_indices].reset_index(drop=True)
    sample_weights = sample_weights[shuffled_indices]
    
    return whole_data, sample_weights

def train_model_with_both_data(model, train_container_, current_data_container_, storage_account_):
    from retraining import validate_model

    global train_container
    global current_data_container
    global storage_account
    train_container, current_data_container, storage_account = train_container_, current_data_container_, storage_account_

    logging.info("Provisioning training with both old and new data...")
    empty_model = deepcopy(model)
    all_data, sample_weights = make_data_ready()

    if "Churn" in all_data.columns and "Churn" in all_data.columns:
        X = all_data.drop(columns=['Churn', 'CustomerID'], axis=1)
    else: 
        X = all_data.drop(columns=['Churn'], axis=1)

    y = all_data['Churn']

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X,y,sample_weights, test_size=0.3, random_state=43, stratify=y
    )

    logging.info("Training model...")

    model.fit(X_train, y_train, Model__sample_weight=sw_train)
    scores = validate_model(model, X_test, y_test)
    pprint.pprint(scores)

    logging.info("Training with both data has been completed...")
    
    return model, scores, empty_model, X_train, y_train