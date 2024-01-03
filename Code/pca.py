import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


class PCA(object):
    """

    This class derives the principal components of a given data matrix. It first computes the covariance matrix of the dataset.
    Eigenvectors are then computed, and we use the eigenvector to construct the PC scores. We retain only k principal components
    and back-transform the values to their original unit (returns in %) using only a k x number of stocks dimensional space.

    """

    def __init__(self, returns, stocks, k):
        """

        Initializes the PCA object.

        Takes as input:

            - returns (pd.DataFrame): Pandas DataFrame containing the spot values of the yields on which PCA is performed;
            - stocks (List): The list of tenors on which PCA is performed;
            - k (int): The number of Principal Components retained in the final model;

        Variables list:

            - self.sc (sklearn.preprocessing.StandardScaler()): sklearn instance used to standardize the stocks returns;
            - self.mapper (sklearn_pandas.DataFrameMapper()): sklearn_pandas instance used to standardize the stocks returns dataframe by columns;

        """

        self.sc = StandardScaler()
        df = returns
        self.mapper = DataFrameMapper([(list(df.columns), self.sc)], df_out=True)
        self.returns = self.mapper.fit_transform(df.copy())
        self.stocks = stocks
        self.k = k

        self.compute_covariance_matrix()
        self.compute_eigenvectors()
        self.compute_pc_scores()
        self.compute_backtransformed_returns()

    def select_pc_number(self, threshold):
        """

            Function that selects the number of Principal Components to retain in the final model.
            The number of Principal Components is selected by the user, and the function returns the number of Principal Components
            that explain at least the specified threshold of the variance of the data.

        Takes as input:

            threshold (float): The threshold of variance explained by the Principal Components retained in the final model;

        Output:

            k (int): The number of Principal Components retained in the final model;

        """

        k = 0
        for i in range(self.eigenvalues.shape[0]):
            if self.eigenvalues["eigenvalue_absolute"].iloc[i] >= threshold:
                k += 1
        return k

    def compute_covariance_matrix(self):
        """

            Calculates the covariance matrix of the stocks returns across the whole time horizon.

        Takes as input:

            None;

        Output:

            None;

        """

        self.cov_matrix = np.array(self.returns).T
        self.cov_matrix = np.cov(self.cov_matrix, bias=True)
        self.cov_matrix = pd.DataFrame(
            data=self.cov_matrix, columns=self.stocks, index=self.stocks
        )

    def compute_eigenvectors(self):
        """

            Functions that computes the Eigenvectors i.e. the vectors that maximize the variance of the data;
            This function calculates the Eigenvectors. By definition these are the vectors that capture the maximum variance of the
            underlying data, and can be found by minimizing the sum of projection length to the respective vector.

        Takes as input:

            None;

        Output:

            None;

        """

        eig = np.linalg.eig(self.cov_matrix)
        self.pc_indices = [f"PC_{i}" for i in range(1, eig[0].shape[0] + 1)]

        self.eigenvalues = pd.DataFrame(
            eig[0].real, columns=["eigenvalue"], index=self.pc_indices
        )
        self.eigenvalues["eigenvalue_relative"] = self.eigenvalues["eigenvalue"].apply(
            lambda x: x / self.eigenvalues["eigenvalue"].sum()
        )
        self.eigenvalues["eigenvalue_absolute"] = self.eigenvalues[
            "eigenvalue_relative"
        ].cumsum()

        self.eigenvectors = pd.DataFrame(
            eig[1].real, index=self.stocks, columns=self.pc_indices
        )
        self.reduced_eigenvectors = self.eigenvectors.iloc[:, : self.k]

    def compute_pc_scores(self):
        """

            This function transforms the underlying data into the new dimensionality formed by the eigenvectors.

        Takes as input:

            None;

        Output:

            None;

        """

        self.pc_scores = np.matrix(self.returns) * np.matrix(self.eigenvectors)
        self.pc_scores = pd.DataFrame(
            data=self.pc_scores,
            columns=self.pc_indices,
            index=pd.to_datetime(self.returns.index),
        )
        self.reduced_pc_scores = self.pc_scores.iloc[:, : self.k]

    def compute_backtransformed_returns(self):
        """

            This function performs the inverse transformation to obtain back-transformed returns from the reduced number of pc scores.

        Takes as input:

            None;

        Output:

            None;

        """

        self.inv_eigenvectors = pd.DataFrame(
            data=np.linalg.inv(np.matrix(self.eigenvectors)),
            columns=self.stocks,
            index=self.pc_indices,
        )
        self.inv_reduced_eigenvectors = self.inv_eigenvectors.iloc[: self.k, :]

        self.backtransformed_yields = np.matrix(self.reduced_pc_scores) * np.matrix(
            self.inv_reduced_eigenvectors
        )
        self.backtransformed_yields = pd.DataFrame(
            data=self.backtransformed_yields,
            columns=self.stocks,
            index=self.reduced_pc_scores.index,
        )

        inverse_scaled_values = self.sc.inverse_transform(self.backtransformed_yields)
        self.backtransformed_yields = pd.DataFrame(
            inverse_scaled_values,
            index=self.backtransformed_yields.index,
            columns=self.backtransformed_yields.columns,
        )

    def out_of_sample_projection(self, train_eigenvectors, test_yields):
        """

            Function that applies dimension reduction on the test returns using the eigenvectors computed on the train dataset.
            Back-transforms the test returns into a lower-dimensional space.

        Takes as input:

            train_eigenvectors (pd.DataFrame): Pandas DataFrame containing the values of the eigenvectors computed on the train dataset;
            test_yields (pd.DataFrame): Pandas DataFrame containing the values of the returns of the test;

        Output:

            backtransformed_yields (pd.DataFrame): Pandas DataFrame containing the values of the back-transformed returns.

        """

        scores_oos = np.matrix(test_yields) * np.matrix(train_eigenvectors)
        scores_oos = pd.DataFrame(
            data=scores_oos,
            columns=self.pc_indices,
            index=pd.to_datetime(test_yields.index),
        )
        reduced_scores_oos = scores_oos.iloc[:, : self.k]

        inv_eigenvectors_oos = pd.DataFrame(
            data=np.linalg.inv(np.matrix(train_eigenvectors)),
            columns=self.stocks,
            index=self.pc_indices,
        )
        inv_reduced_eigenvectors_oos = inv_eigenvectors_oos.iloc[: self.k, :]

        backtransformed_yields = np.matrix(reduced_scores_oos) * np.matrix(
            inv_reduced_eigenvectors_oos
        )
        backtransformed_yields = pd.DataFrame(
            data=backtransformed_yields,
            columns=self.stocks,
            index=reduced_scores_oos.index,
        )

        inverse_scaled_values = self.sc.inverse_transform(backtransformed_yields)
        backtransformed_yields = pd.DataFrame(
            inverse_scaled_values,
            index=backtransformed_yields.index,
            columns=backtransformed_yields.columns,
        )
        return backtransformed_yields
