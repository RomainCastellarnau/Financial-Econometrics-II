from CorePtf import CorePtf

if __name__ == "__main__":
    # Def the PCA model
    model = CorePtf()

    # Retrieve the attribute
    my_data = model.returns
    cov = model.cov_matrix
    eigen_values = model.eigenvalues
    variance_explained = model.variance_explained
    cumulative_variance_explained = variance_explained.cumsum()
    cumulative_variance_explained.plot.bar(figsize=(10, 6), rot=0)
