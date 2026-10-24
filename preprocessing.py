# preprocess.py
import numpy as np
import pandas as pd

class DataPreprocessor:
    @staticmethod
    def clean(file_path: str) -> pd.DataFrame:
        print("--- Step 2: Data Preprocessing ---")
        df = pd.read_csv(file_path)
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)
        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
        df["SSN"] = df["SSN"].replace("#F%$D@*&8", np.nan)

        # 2. List semua kolom numerik yang terkontaminasi teks/simbol
        contaminated_cols = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]

        
        for col in contaminated_cols:
            df[col] = df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 4. Mask nilai tidak logis / outlier
        df["Age"] = df["Age"].mask((df["Age"] < 0) | (df["Age"] > 110))

        df["Annual_Income"] = df["Annual_Income"].mask(df["Annual_Income"] < 0)

        df["Monthly_Inhand_Salary"] = df["Monthly_Inhand_Salary"].mask(
            df["Monthly_Inhand_Salary"] < 0
        )

        df["Num_Bank_Accounts"] = df["Num_Bank_Accounts"].mask(
            (df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20)
        )

        df["Num_Credit_Card"] = df["Num_Credit_Card"].mask(
    (df["Num_Credit_Card"] < 0) | (df["Num_Credit_Card"] > 20)
)

        df["Interest_Rate"] = df["Interest_Rate"].mask(
        (df["Interest_Rate"] < 0) | (df["Interest_Rate"] > 100)
)

        df["Num_of_Loan"] = df["Num_of_Loan"].mask(
        (df["Num_of_Loan"] < 0) | (df["Num_of_Loan"] > 20)
)

        df["Delay_from_due_date"] = df["Delay_from_due_date"].mask(
        df["Delay_from_due_date"] < 0, 0
)

        df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].mask(
        (df["Num_of_Delayed_Payment"] < 0) | (df["Num_of_Delayed_Payment"] > 100)
)

        df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].mask(
        df["Changed_Credit_Limit"] < -100
)

        df["Num_Credit_Inquiries"] = df["Num_Credit_Inquiries"].mask(
        (df["Num_Credit_Inquiries"] < 0) | (df["Num_Credit_Inquiries"] > 50)
)

        df["Outstanding_Debt"] = df["Outstanding_Debt"].mask(
        df["Outstanding_Debt"] < 0
)

        df["Credit_Utilization_Ratio"] = df["Credit_Utilization_Ratio"].mask(
        (df["Credit_Utilization_Ratio"] < 0) | (df["Credit_Utilization_Ratio"] > 100)
)

        df["Total_EMI_per_month"] = df["Total_EMI_per_month"].mask(
        df["Total_EMI_per_month"] < 0
)

        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].mask(
        df["Amount_invested_monthly"] < 0
)

        df["Monthly_Balance"] = df["Monthly_Balance"].mask(
        df["Monthly_Balance"] < -100000
)
        pii_and_ignored_cols = [
            'Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN',
            'Type_of_Loan', 'Credit_History_Age'
        ]

        df = df.drop(columns=pii_and_ignored_cols)
        return df