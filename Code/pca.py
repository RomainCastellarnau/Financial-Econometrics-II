import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.optimize import minimize


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

        self.scalers = {}  # Dictionary to store scalers
        self.stocks = stocks
        self.k = k

        # Standard scale each column of the returns DataFrame and save scalers
        for col in returns.columns:
            if any(ticker in col for ticker in stocks):
                scaler = StandardScaler()
                returns[col] = scaler.fit_transform(returns[col].values.reshape(-1, 1))
                self.scalers[col] = scaler

        self.returns = returns
        self.compute_covariance_matrix()
        self.compute_eigenvectors()
        self.compute_pc_scores()

        # self.compute_backtransformed_returns()
        # self.pca_model()

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

        self.pc_scores = np.matrix(self.returns) * np.matrix(self.eigenvectors)  # type: ignore
        self.pc_scores = pd.DataFrame(
            data=self.pc_scores,
            columns=self.pc_indices,
        )
        self.pc_scores.index = pd.to_datetime(self.returns.index)  # type: ignore
        self.reduced_pc_scores = self.pc_scores.iloc[:, : self.k]

    def pca_model(self):
        """
        Function that returns a dictionary of linear models associated with each of the i stocks.

        Takes as input:
            None;

        Output:
            pca_model (dict): Dictionary containing the PCA model.
        """

        self.pca_models = {}

        for i in self.stocks:
            y = np.array(self.returns[i])
            X = np.array(self.reduced_pc_scores)
            X = add_constant(X)  # Add a constant term for the intercept

            try:
                model_i = OLS(y, X, hasconst=True).fit()
                self.pca_models[i] = {
                    "model_result": model_i,
                    "alpha": model_i.params[0],
                    "beta": model_i.params[1:],
                    "residuals": model_i.resid,
                }
            except KeyError as e:
                print(f"Error for stock {i}: {e}")
                print("Columns in reduced_pc_scores:")
                print(self.reduced_pc_scores.columns.tolist())
                print("Columns in returns:")
                print(self.returns.columns.tolist())
                raise

        return self.pca_models

    def optimisation_routine(self):
        """
        Function that performs the optimisation routine to compute the weights of the core equity portfolio.

        Takes as input:
            None;

        Output:
            None;
        """

        # Compute the covariance matrix of the residuals
        self.residuals_cov_matrix = np.cov(
            np.array([self.pca_models[i]["residuals"] for i in self.stocks]), bias=True
        )
        self.residuals_cov_matrix = pd.DataFrame(
            data=self.residuals_cov_matrix,
            columns=self.stocks,
            index=self.stocks,
        )

        # # Compute the weights of the core equity portfolio
        # self.core_equity_portfolio = (
        #     np.linalg.inv(self.residuals_cov_matrix)
        #     @ self.eigenvectors.iloc[:, : self.k]
        # )
        # self.core_equity_portfolio = pd.DataFrame(
        #     data=self.core_equity_portfolio,
        #     columns=self.pc_indices,
        #     index=self.stocks,
        # )

    def core_equity_portfolio(self):
        """
        Compute the weights of the equity portfolio designed to replicate the first core equity factor,
        defined as:

        - 𝐴𝑟𝑔𝑚𝑖𝑛𝑤𝑘(

        subject to:
            - ∑ 𝑤1 
            - 𝑤𝑘,0 
            - ∑𝑤𝑘*𝑖𝑏̂𝑖 = 1

        with:
        - Ω̂ the sample covariance matrix of the stock returns,
        - 𝑏̂𝑖,1 the estimated sensitivity of stock 𝑖 to the 1st core equity factor.

        """

    # # def compute_backtransformed_returns(self):
    #     """

    #         This function performs the inverse transformation to obtain back-transformed returns from the reduced number of pc scores.

    #     Takes as input:

    #         None;

    #     Output:

    #         None;

    #     """

    #     self.inv_eigenvectors = pd.DataFrame(
    #         data=np.linalg.inv(np.matrix(self.eigenvectors)),
    #         columns=self.stocks,
    #         index=self.pc_indices,
    #     )
    #     self.inv_reduced_eigenvectors = self.inv_eigenvectors.iloc[: self.k, :]

    #     self.backtransformed_yields = np.matrix(self.reduced_pc_scores) * np.matrix(
    #         self.inv_reduced_eigenvectors
    #     )
    #     self.backtransformed_yields = pd.DataFrame(
    #         data=self.backtransformed_yields,
    #         columns=self.stocks,
    #         index=self.reduced_pc_scores.index,
    #     )

    #     inverse_scaled_values = self.sc.inverse_transform(self.backtransformed_yields)
    #     self.backtransformed_yields = pd.DataFrame(
    #         inverse_scaled_values,
    #         index=self.backtransformed_yields.index,
    #         columns=self.backtransformed_yields.columns,
    #     )

    # def out_of_sample_projection(self, train_eigenvectors, test_yields):
    #     """

    #         Function that applies dimension reduction on the test returns using the eigenvectors computed on the train dataset.
    #         Back-transforms the test returns into a lower-dimensional space.

    #     Takes as input:

    #         train_eigenvectors (pd.DataFrame): Pandas DataFrame containing the values of the eigenvectors computed on the train dataset;
    #         test_yields (pd.DataFrame): Pandas DataFrame containing the values of the returns of the test;

    #     Output:

    #         backtransformed_yields (pd.DataFrame): Pandas DataFrame containing the values of the back-transformed returns.

    #     """

    #     scores_oos = np.matrix(test_yields) * np.matrix(train_eigenvectors)
    #     scores_oos = pd.DataFrame(
    #         data=scores_oos,
    #         columns=self.pc_indices,
    #         index=pd.to_datetime(test_yields.index),
    #     )
    #     reduced_scores_oos = scores_oos.iloc[:, : self.k]

    #     inv_eigenvectors_oos = pd.DataFrame(
    #         data=np.linalg.inv(np.matrix(train_eigenvectors)),
    #         columns=self.stocks,
    #         index=self.pc_indices,
    #     )
    #     inv_reduced_eigenvectors_oos = inv_eigenvectors_oos.iloc[: self.k, :]

    #     backtransformed_yields = np.matrix(reduced_scores_oos) * np.matrix(
    #         inv_reduced_eigenvectors_oos
    #     )
    #     backtransformed_yields = pd.DataFrame(
    #         data=backtransformed_yields,
    #         columns=self.stocks,
    #         index=reduced_scores_oos.index,
    #     )

    #     inverse_scaled_values = self.sc.inverse_transform(backtransformed_yields)
    #     backtransformed_yields = pd.DataFrame(
    #         inverse_scaled_values,
    #         index=backtransformed_yields.index,
    #         columns=backtransformed_yields.columns,
    #     )
    #     return backtransformed_yields
    # def core_portfolio_weights(self )
