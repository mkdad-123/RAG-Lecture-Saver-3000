import mlflow
from pathlib import Path
import faiss
from contextlib import contextmanager


def start_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()



def init_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name)

@contextmanager
def start_rag_run(run_name: str = "rag_query"):
    with mlflow.start_run(run_name=run_name):
        yield

def log_params_dict(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics_dict(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_text_artifact(text: str, artifact_path: str):
    mlflow.log_text(text, artifact_path)

def log_faiss_index(index, path: Path):
    faiss.write_index(index, str(path))
    mlflow.log_artifact(str(path))

def log_artifact_metadatas(metadatas: Path ="metadata.json"): 
    mlflow.log_artifact(str(metadatas))


def end_run():
    mlflow.end_run()

