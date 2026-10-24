import mlflow
import joblib
import os
from sklearn.metrics import accuracy_score

class ModelEvaluator:
    def __init__(self, trained_pipelines: dict, X_test, y_test, experiment_name="Credit_Score_Orchestration"):
        self.trained_pipelines = trained_pipelines
        self.X_test = X_test
        self.y_test = y_test
        self.experiment_name = experiment_name
        self.best_model_name = ""

    def evaluate_and_log(self):
        print("--- Step 4: Model Evaluation ---")
        best_accuracy = 0

        for name, (pipeline, run_id) in self.trained_pipelines.items():
            y_pred = pipeline.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f" {name} Accuracy: {accuracy:.4f}")
            
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("accuracy", accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model_name = name

        print(f"\nWINNING MODEL: {self.best_model_name} with Accuracy: {best_accuracy:.4f}")
        return self.best_model_name

    def fetch_best_and_save(self, output_filename="artifacts/best_credit_model.pkl"):
        print("--- Step 5: Saving Best Model Directly to Disk ---")
        
        
        winning_pipeline = self.trained_pipelines[self.best_model_name][0]
        
        # Save it directly to the artifacts folder with compression (compress=3)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        joblib.dump(winning_pipeline, output_filename, compress=3)
        
       