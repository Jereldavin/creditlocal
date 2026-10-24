import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mlflow

class ModelTrainer:
    def __init__(self, cleaned_df: pd.DataFrame, target_col: str = 'Credit_Score'):
        self.df = cleaned_df
        self.target_col = target_col
        self.skewed_numeric_features = [
            'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
            'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
        ]
        self.symmetric_numeric_features = [
            'Age', 'Delay_from_due_date', 'Changed_Credit_Limit', 'Credit_Utilization_Ratio'
        ]
        self.categorical_features = [
            'Month', 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
        ]

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        print("--- Step 3: Data Splitting ---")
        X = self.df.drop(columns=[self.target_col], errors='ignore')
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _build_preprocessor(self) -> ColumnTransformer:
        print("--- Step 3.1: Preprocessor Building ---")
        skewed_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        symmetric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        
        return ColumnTransformer(transformers=[
            ('num_skewed', skewed_transformer, self.skewed_numeric_features),
            ('num_symmetric', symmetric_transformer, self.symmetric_numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])

    def train_models(self) -> dict:
        print("--- Step 3.2: Model Training ---")
        preprocessor = self._build_preprocessor()
        
        
        models = {
            'Logistic_Regression': LogisticRegression(random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        trained_pipelines = {}
        mlflow.set_experiment("Credit_Score_Orchestration")

        for name, model_instance in models.items():
            print(f" Training {name}...")
            with mlflow.start_run(run_name=name) as run:
                full_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model_instance)
                ])
                full_pipeline.fit(self.X_train, self.y_train)
                trained_pipelines[name] = (full_pipeline, run.info.run_id)
                
                mlflow.log_params(model_instance.get_params())
                mlflow.sklearn.log_model(full_pipeline, artifact_path="model", serialization_format="pickle")
                
        return trained_pipelines