from pca import PCA  # Import the PCA class
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.getcwd()), "Data")
    returns = pd.read_excel(
        os.path.join(path, "Data.xlsx"), sheet_name="RETURNS"
    ).rename(columns={"Unnamed: 0": "Date"})
    returns = returns.set_index("Date")
    returns.head()
    stocks = returns.columns.tolist()

    # Def the PCA model
    model = PCA(returns, stocks)

    # Retrive the attribute
    my_data = model.returns
    cov = model.cov_matrix
    eigen_values = model.eigenvalues
    variance_explained = model.variance_explained
    cumulative_variance_explained = variance_explained.cumsum()
    cumulative_variance_explained.plot.bar(figsize=(10, 6), rot=0)
