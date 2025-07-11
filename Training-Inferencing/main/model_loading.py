from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import pickle
import pandas as pd
import io
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load(StorageAccount ,Container, LoadLatestModel, ModelName, origin):

    credential = DefaultAzureCredential()
    
    storage_account = StorageAccount
    container_name = Container

    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    def load_model_(latest_blob):
        model_info = f"Loading latest model: {latest_blob.name}, Last Modified: {latest_blob.last_modified}"
        logging.info(f"Loading latest model: {latest_blob.name}, Last Modified: {latest_blob.last_modified}")

        blob_client = container_client.get_blob_client(latest_blob.name)
        blob_data = blob_client.download_blob().readall()

        model_properties = blob_client.get_blob_properties()
        model_info = {
            "name":latest_blob.name, 
            "Last Modified":latest_blob.last_modified, 
            "is current version":model_properties.get("is_current_version"), 
            "container": model_properties.get("container"),
            "version id": model_properties.get("version_id"), 
            "Pipeline":origin
        }
        return blob_data, model_info

    if LoadLatestModel: 
        logging.info("Loading last modified model ❕")
        blobs = container_client.list_blobs()

        latest_blob = max(blobs, key=lambda b: b.last_modified)
        blob_data, model_info = load_model_(latest_blob)

    else:
        logging.info(f"Loading given model {ModelName} 💀")
        blob_client = container_client.get_blob_client(ModelName)
        properties = blob_client.get_blob_properties()
        meta = properties.metadata
        blob_data, model_info = load_model_(meta)

    model = pickle.loads(blob_data)
    return model, model_info

def load_blob_client():
    feature = ["CustomerID","Age","Gender","Tenure","Usage Frequency","Support Calls","Payment Delay","Subscription Type","Contract Length","Total Spend","Last Interaction","Churn", "prediction"]

    credential = DefaultAzureCredential()

    storage_account_name = "churnprediction1"
    container_name = "data-log-inference"

    account_url = f"https://{storage_account_name}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)

    try: 
        container_client.create_container()
    except Exception as e: 
        logging.info(f"Container already exists.")

    blob_name = "logs.csv"
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():

        df = pd.DataFrame(columns= feature)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
    
    return blob_client, feature

