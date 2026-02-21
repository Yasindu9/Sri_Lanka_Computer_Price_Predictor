import pandas as pd

def analyze_data():
    df = pd.read_csv('computers_data_withoutprocessed.csv')
    print("--- DataFrame Info ---")
    df.info()
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Sample Values ---")
    print(df.head(3))
    
if __name__ == "__main__":
    analyze_data()
