from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from train import ModelTrainer
from evaluate import ModelEvaluator

class MLEndToEndPipeline:
    def __init__(self, raw_source_path: str, ingestion_dir: str):
        self.raw_source_path = raw_source_path
        self.ingestion_dir = ingestion_dir

    def run_pipeline(self):
        print("Starting End-to-End ML Pipeline...")
        
        # 1. Ingestion
        ingestion = DataIngestion(input_path=self.raw_source_path, output_dir=self.ingestion_dir)
        ingested_file_path = ingestion.run()
        
        # 2. Preprocessing
        cleaned_df = DataPreprocessor.clean(ingested_file_path)
        
        # 3. Training
        print("--- Step 3: Model Training ---")
        trainer = ModelTrainer(cleaned_df)
        X_train, X_test, y_train, y_test = trainer.split_data()
        trained_pipelines = trainer.train_models()
        
        # 4. Evaluation & Export
        print("--- Step 4: Evaluation & Export ---")
        evaluator = ModelEvaluator(trained_pipelines, X_test, y_test)
        evaluator.evaluate_and_log()
        evaluator.fetch_best_and_save("artifacts/best_credit_model.pkl")
        
        print("Pipeline execution complete!")

if __name__ == "__main__":
    pipeline = MLEndToEndPipeline(
        raw_source_path="data_A.csv", 
        ingestion_dir="artifacts/data"
    ) 
    pipeline.run_pipeline()