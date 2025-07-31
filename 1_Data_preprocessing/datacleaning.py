import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
 
  
#   step2 load dataset   
# Replace 'your_dataset.csv' with the  actual file name
# C:\Users\shivam\OneDrive\Desktop\ML-visuara\ML_All_model\Data Set For Task\1) iris.csv
def preprocess_species_dataset(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    for col in df.select_dtypes(include='object').columns:
        if col != 'species':
            df[col] = le.fit_transform(df[col])

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('species')

    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    X = df.drop('species', axis=1)
    y = df['species']

    return train_test_split(X, y, test_size=0.2, random_state=42)

result = preprocess_species_dataset('C:/Users/shivam/OneDrive/Desktop/ML-visuara/ML_All_model/DataSet/1_iris.csv')
print(result)
