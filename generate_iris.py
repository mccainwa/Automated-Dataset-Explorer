from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame
df.to_csv("sample_datasets/iris.csv", index=False)
print("iris.csv created successfully")
