import io 
import pandas as pd 
from azure.storage.blob import BlobServiceClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os 
from datetime import datetime, timedelta
from azure.identity import DefaultAzureCredential
import logging
import tempfile
logging.basicConfig(level=logging.INFO, format="%(asctime)s \
                    - %(levelname)s - %(message)s")


def detect_drift(storage_account, train_container, current_data_container):
    
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

    def get_drift_report(referenceData, currentData, blob_service_client, current_data_container):
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=referenceData, current_data=currentData)

        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(tmp_dir, "drift_report.html")
        report.save_html(tmp_file_path)

        container_client = blob_service_client.get_container_client("Drift-report")
        try: 
            container_client.create_container()
        except Exception: 
            logging.info("Report container Already exists")

        report_blob_client = container_client.get_blob_client("data_drift_report.html")

        with open(tmp_file_path, "rb") as f:
            # report_blob_path = "reports/drift_report.html"
            # report_blob = report_blob_client.get_blob_client(container="Drift-report", blob=report_blob_path)
            report_blob_client.upload_blob(f.read(), overwrite=True)

        drift_results = report.as_dict()
        drift_score = drift_results['metrics'][0]['result']['drift_share']
        logging.info(f"Drift Score: {drift_score}......")
        return drift_score

    def main():
        logging.info("Loading current and reference data...")
        reference, target, blob_service_client, current_container = load_both_data()
        if reference is None or target is None: 
            logging.error("Data Loading failed. Skipping drift detection❌")
            return
        logging.info("Determining Drift ")
        drift_value = get_drift_report(reference, target, blob_service_client, current_container)

        if drift_value>0.1:
            logging.info(f"Drift is high... Retraining Needed ❗❗❗ {drift_value}")
            trigger_azure_pipeline()
        else: 
            logging.info("Drift is tolerable... No need for retraining 🔹")
