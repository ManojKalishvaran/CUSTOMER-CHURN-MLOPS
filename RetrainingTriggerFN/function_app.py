import azure.functions as func
import logging
import requests
from requests.auth import HTTPBasicAuth

# Constants (store secrets like PAT in Azure Key Vault or Application Settings securely)
PAT = "pat token"
ORG = "yusufds25280402"
PROJECT = "customer-churn-prediction"
PIPELINE_ID = 41

def run_devops_pipeline():
    url = (
        f"https://dev.azure.com/{ORG}/{PROJECT}/_apis/pipelines/{PIPELINE_ID}/runs?api-version=7.1-preview.1"
    )
    response = requests.post(
        url,
        auth=HTTPBasicAuth('', PAT),
        headers={"Content-Type": "application/json"},
        json={}
    )
    if response.status_code in (200, 201):
        logging.info(f"Pipeline {PIPELINE_ID} triggered successfully.")
    else:
        logging.error(f"Pipeline trigger failed: {response.status_code} - {response.text}")

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.blob_trigger(
    arg_name="myblob",
    path="retraining-data/{name}",  # container/folder path
    connection="AzureWebJobsStorage"
)
def blob_trigger_function(myblob: func.InputStream):
    logging.info(
        f"Blob trigger: '{myblob.name}' size: {myblob.length} bytes"
    )
    run_devops_pipeline()
