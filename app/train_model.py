import os
import json
import joblib

import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

EXPERIMENT_NAME = "iris-model-zoo"
REGISTERED_MODEL_NAME = "IrisModel"
APP_VERSION = "v0.1.0"

MODEL_PATH = "./app/model.joblib"
MODEL_META_PATH = "./app/model_meta.json"


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=200, multi_class="auto"
        ),
        "SVM": SVC(
            probability=True, kernel="rbf", random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

def log_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    cm_path = "artifacts/confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    mlflow.log_artifact(cm_path, artifact_path="artifacts")

def log_classification_report(y_true, y_pred, labels):
    report_str = classification_report(
        y_true, y_pred, target_names=labels
    )
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    mlflow.log_artifact(report_path, artifact_path="artifacts")

def train_and_log_models(X_train, X_test, y_train, y_test, labels):
    models = get_models()

    best = {
        "name": None,
        "model": None,
        "f1_macro": -1.0,
        "accuracy": 0.0,
        "run_id": None,
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            try:
                y_proba = model.predict_proba(X_test)
                auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr"
                )
            except Exception:
                auc = None

            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            precision_macro = precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            recall_macro = recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )

            mlflow.log_param("model_name", model_name)
            try:
                mlflow.log_params(model.get_params())
            except Exception:
                pass

            mlflow.set_tag("app_version", APP_VERSION)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("precision_macro", precision_macro)
            mlflow.log_metric("recall_macro", recall_macro)
            if auc is not None:
                mlflow.log_metric("auc_ovr", auc)

            log_confusion_matrix(y_test, y_pred, labels)
            log_classification_report(y_test, y_pred, labels)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None,
            )

            run_id = run.info.run_id
            print(
                f"Finished run for {model_name} "
                f"(run_id={run_id}, f1_macro={f1_macro:.4f})"
            )

            if f1_macro > best["f1_macro"]:
                best.update(
                    {
                        "name": model_name,
                        "model": model,
                        "f1_macro": f1_macro,
                        "accuracy": accuracy,
                        "run_id": run_id,
                    }
                )

    return best

def save_best_model_locally(best_info):
    os.makedirs("app", exist_ok=True)

    joblib.dump(best_info["model"], MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")

    meta = {
        "best_model": best_info["name"],
        "metrics": {
            "accuracy": best_info["accuracy"],
            "f1_macro": best_info["f1_macro"],
        },
        "mlflow_run_id": best_info["run_id"],
        "version": APP_VERSION,
    }

    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Model metadata saved to {MODEL_META_PATH}")

def register_best_model_in_registry(best_info):
    mlflow.set_tracking_uri("file:./mlruns")

    model_uri = f"runs:/{best_info['run_id']}/model"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print(
        f"Registered model '{REGISTERED_MODEL_NAME}' as version "
        f"{result.version} (run_id={best_info['run_id']})"
    )

def main():
    X, y = load_data()
    iris = datasets.load_iris()
    labels = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best = train_and_log_models(X_train, X_test, y_train, y_test, labels)

    print(
        f"Best model: {best['name']} "
        f"(f1_macro={best['f1_macro']:.4f}, accuracy={best['accuracy']:.4f})"
    )

    save_best_model_locally(best)

    register_best_model_in_registry(best)


if __name__ == "__main__":
    main()

