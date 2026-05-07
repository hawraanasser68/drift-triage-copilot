import os
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "bank-marketing-classifier")
DATABASE_URL        = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")
AGENT_WEBHOOK_URL   = os.getenv("AGENT_WEBHOOK_URL", "http://agent:8001/drift-event")
DRIFT_WINDOW        = int(os.getenv("DRIFT_WINDOW", "500"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
