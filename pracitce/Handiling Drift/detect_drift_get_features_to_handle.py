""" Detect drift get features to handle
Usage: 
    1. Get drift score by columns by passing the location of both data in azure - detect_save_extract_drift(storage_account, train_container, current_data_container)
    2. Get features to handle by passing the drift score along with model and df - get_features_to_handle(model, xtrain, ytrain, score_by_column, df)

"""

import json
import io 
import pandas as pd 
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os 
from datetime import datetime, timedelta
import logging
import tempfile
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
logging.basicConfig(level=logging.INFO, format="%(asctime)s \
                    - %(levelname)s - %(message)s")



def detect_save_extract_drift(storage_account, train_container, current_data_container):
    
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

        container_client = blob_service_client.get_container_client("drift-report")
        try: 
            container_client.create_container()
        except Exception: 
            logging.info("Report container Already exists")

        report_blob_client = container_client.get_blob_client("datadriftreport.html")


        with open(tmp_file_path, "rb") as f:
            print(f"{f = }")
            # report_blob_path = "reports/drift_report.html"
            # report_blob = report_blob_client.get_blob_client(container="Drift-report", blob=report_blob_path)
            report_blob_client.upload_blob(f.read(), overwrite=True, max_concurrency=-1)

        drift_results = report.as_dict()
        # drift_score = drift_results['metrics'][0]['result']['drift_share']
        # logging.info(f"Drift Score: {drift_score}......")
        # return drift_score
        return drift_results

    def extract_score_result(full_metric):
        drift_by_column = []
        for k, v in full_metric['metrics'][1]['result']['drift_by_columns'].items():
            # print(f"Column: {k} --- Score: {v['drift_score']} --- IsDrift: {v['drift_detected']}")
            drift_by_column.append({k:{"Score":v['drift_score'], "IsDrift":v['drift_detected']}})   
        return drift_by_column

    def main():
        logging.info("Loading current and reference data...")
        reference, target, blob_service_client, current_container = load_both_data()
        if reference is None or target is None: 
            logging.error("Data Loading failed. Skipping drift detection❌")
            return
        logging.info("Determining Drift ")
        full_drift_metrices = get_drift_report(reference, target, blob_service_client, current_container)

        # if drift_value>0.1:
        #     logging.info(f"Drift is high... Retraining Needed ❗❗❗ {drift_value}")
        #     trigger_azure_pipeline()
        # else: 
        #     logging.info("Drift is tolerable... No need for retraining 🔹")
        dirft_score_column = extract_score_result(full_drift_metrices)

        return dirft_score_column, full_drift_metrices
    return main()
 

def get_feature_importance(model, X, y, n_repeats=10, random_state=42, scoring=None):
    
    """
    Extract feature importance from any sklearn supervised model.

    Parameters:
    - model: fitted sklearn estimator
    - X: pd.DataFrame or np.ndarray, feature matrix used for evaluation
    - y: array-like, target vector
    - n_repeats: int, number of repeats for permutation importance (default=10)
    - random_state: int, random seed for permutation importance
    - scoring: str or callable, scoring metric for permutation importance (default=None uses model.score)

    Returns:
    - pd.DataFrame with columns: ['feature', 'importance_mean', 'importance_std']
      sorted by descending importance_mean
    """

    # Get feature names if possible
    if hasattr(X, 'columns'):
        feature_names = X.columns
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    # Case 1: Tree-based models with feature_importances_
    if hasattr(model, 'feature_importances_'):
        logging.info("Given model tree based...")
        importances = model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importances,
            'importance_std': np.zeros_like(importances)
        })
        df = df.sort_values(by='importance_mean', ascending=False)
        return df.reset_index(drop=True)

    # Case 2: Linear models with coef_
    elif hasattr(model, 'coef_'):
        # coef_ can be 1D or 2D (multi-class)
        logging.info("Given model is linear model...")

        coefs = model.coef_
        if coefs.ndim == 1:
            importances = np.abs(coefs)
        else:
            # For multi-class, take mean absolute coef across classes
            importances = np.mean(np.abs(coefs), axis=0)
        df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importances,
            'importance_std': np.zeros_like(importances)
        })
        df = df.sort_values(by='importance_mean', ascending=False)
        return df.reset_index(drop=True)

    # Case 3: Use permutation importance (model-agnostic)
    else:
        logging.info("Given model agnostic...")

        # Run permutation importance on given data
        perm_result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_result.importances_mean,
            'importance_std': perm_result.importances_std
        })
        df = df.sort_values(by='importance_mean', ascending=False)
        return df.reset_index(drop=True)

def get_significant_features_drift(importance, drift):
    significant_features = list(importance[importance['importance_mean']>0.01]['feature'].values)    
    
    significant_features_drift = []
    for i in drift: 
        feature = list(i.keys())[0]
        if feature in significant_features: 
            significant_features_drift.append((feature, i[feature]['Score']))
    return significant_features_drift    


def get_columns_based_on_type(sig_drift, df):
    filtered_features = []

    for i in sig_drift:
        filtered_features.append(i[0])

    dtype_filtered_features = {k:v for k,v in df.dtypes.to_dict().items() if k in filtered_features}
    dtype_filtered_features_forcategorical = {k:v for k,v in df.dtypes.to_dict().items()}

    # dtype_filtered_features['Age']== "int64"
    categorical_features = [k for k, v in dtype_filtered_features_forcategorical.items() if v=='O']
    discrete_features = [k for k,v in dtype_filtered_features.items() if 'int' in str(v)]
    continuous_features = [k for k,v in dtype_filtered_features.items() if 'float' in str(v)]

    return {"categorical":categorical_features, "continuous":continuous_features, "discrete":discrete_features}

def get_features_to_handle(model, xtrain, ytrain, score_by_column, df):

    logging.info("Getting the feature importance score....")
    importance = get_feature_importance(model, xtrain, ytrain)

    logging.info("Collecting features with significant drift....")
    features_with_significant_drift = get_significant_features_drift(importance, score_by_column)

    logging.info("Getting columns to be handled...")
    return get_columns_based_on_type(features_with_significant_drift, df)

# if name = "__main__":
#     score_by_column , stats = detect_save_extract_drift("churnprediction1", "training-data", "retraining-data")

#     importance = get_feature_importance(model.named_steps['Train'], X_train, y_train)

#     get_significant_features_drift(importance, score_by_column)

