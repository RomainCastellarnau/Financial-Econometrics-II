import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import norm
from scipy.optimize import minimize


class PCA(object):
    """

    This class derives the principal components of a given data matrix. It first computes the covariance matrix of the dataset.
    Eigenvectors are then computed, and we use the eigenvector to construct the PC scores. We retain only k principal components
    and back-transform the values to their original unit (returns in %) using only a k x number of stocks dimensional space.
    We define the core equity factors as the main principal components, rescaled to the same volatility as that of the benchmark.

    """

    def __init__(self, returns, stocks, k=None):
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

        self.benchmark = returns.iloc[
            :, 0
        ]  # pd.DataFrame(returns[:, 0], index=returns[:, 0], columns=["benchmark"])
        self.returns = returns.iloc[
            :, 1:
        ]  # pd.DataFrame(returns[:, 1:], index=returns[:, 0], columns=stocks)
        self.scalers = {}  # Dictionary to store scalers
        self.stocks = stocks  # List of stocks tickers on which PCA is performed
        self.k = k  # Number of Principal Components retained in the final model

        # Standard scale each column of the returns DataFrame and save scalers

        for col in returns.columns:
            if any(ticker in col for ticker in stocks):
                scaler = StandardScaler()
                returns[col] = scaler.fit_transform(returns[col].values.reshape(-1, 1))
                self.scalers[col] = scaler

        self.std_returns = returns
        self.std_returns = self.std_returns.iloc[:, 1:]
        self.benchmark_vol = self.benchmark.std()
        self.compute_covariance_matrix()
        self.compute_eigenvectors()
        self.compute_pc_scores()
        self.rescale_pc()

    def select_pc_number(self, threshold):
        """

            Function that selects the number of Principal Components to retain in the final model.
            The number of Principal Components is selected by the user, and the function returns the number of Principal Components
            that explain at least the specified threshold of the variance of the data.
            Kaiser Criterion / Elbow Method / Scree Plot can be used to select the number of Principal Components.

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

            Calculates the covariance matrix of the standardized stocks returns across the whole time horizon.

        Takes as input:

            None;

        Output:

            None;

        """

        self.std_cov_matrix = np.array(self.std_returns).T
        self.cov_matrix = np.array(self.returns).T
        self.std_cov_matrix = np.cov(self.std_cov_matrix, bias=True)
        self.cov_matrix = np.cov(self.cov_matrix, bias=True)
        self.std_cov_matrix = pd.DataFrame(
            data=self.std_cov_matrix, columns=self.stocks[1:], index=self.stocks[1:]
        )
        self.cov_matrix = pd.DataFrame(
            data=self.cov_matrix, columns=self.stocks[1:], index=self.stocks[1:]
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

        eig = np.linalg.eig(self.std_cov_matrix)
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

    def rescale_pc(self):
        """
        Function that rescales the Principal Components to the same volatility as that of the benchmark.

        Takes as input:
            None;

        Output:
            None;
        """

        # Rescale the PC scores to the same volatility as that of the benchmark
        self.rescaled_pc_scores = (
            self.pc_scores * self.benchmark_vol / self.pc_scores.std()
        )

    def pca_model(self):
        """
        Function that returns a dictionary of linear models associated with each of the i stocks.

        Takes as input:
            None;

        Output:
            pca_model (dict): Dictionary containing the PCA model.
        """

        self.pca_models = {}
        # length index of the returns of the benchmark
        self.core_eq_1 = np.zeros(len(self.stocks))
        for i in self.stocks:
            y = np.array(self.returns[i])
            X = np.array(self.rescaled_pc_scores)
            X = add_constant(X)  # Add a constant term for the intercept

            try:
                model_i = OLS(y, X, hasconst=True).fit()
                self.pca_models[i] = {
                    "model_result": model_i,
                    "alpha": model_i.params[0],
                    "beta": model_i.params[1:],
                    "residuals": model_i.resid,
                }
                # Add Core equity factor 1 (the first beta of the regression) to the vector
                self.core_eq_1[i] = self.pca_models[i]["beta"][0]

            except KeyError as e:
                print(f"Error for stock {i}: {e}")
                print("Columns in reduced_pc_scores:")
                print(self.reduced_pc_scores.columns.tolist())
                print("Columns in returns:")
                print(self.returns.columns.tolist())
                raise

        return self.pca_models

    def optim_routine(self):
        core_eq_1 = self.core_eq_1

        def objective(W):
            # calculate mean/variance of the portfolio
            util = np.dot(np.dot(W.T, self.cov_matrix), W)
            # objective: min variance
            return util

        n = len(self.cov_matrix)
        # initial conditions: equal weights
        W = np.ones([n]) / n
        # weights between 0%..100%: no shorts
        b_ = [(0.0, 1.0) for i in range(n)]
        # No leverage: unitary constraint (sum weights = 100%)
        c_ = [
            {"type": "eq", "fun": lambda W: sum(W) - 1.0},
            {"type": "ineq", "fun": lambda W: W.T @ core_eq_1 - 1.0 == 0.0},
        ]

        optimized = minimize(
            objective,
            W,
            (core_eq_1, self.cov_matrix),
            method="SLSQP",
            constraints=c_,
            bounds=b_,
            options={"maxiter": 100, "ftol": 1e-08},
        )
        return optimized.x

    # def core_equity_ptf(self):
    #     """
    #     Function that returns the weights of the core equity portfolio.

    #     Takes as input:
    #         None;

    #     Output:
    #         core_equity_ptf (dict): Dictionary containing the weights of the core equity portfolio.
    #     """
    #     self.core_equity_ptfs = {}
    #     self.core_equity_ptfs = self.optim_routine()

    #     # Reformat the dictionary to a pandas dataframe with the columns being the stocks and the row being the weight
    #     self.core_equity_ptfs = pd.DataFrame(
    #         data=self.core_equity_ptfs, columns=self.stocks
    #     )
    #     self.core_equity_ptfs = self.core_equity_ptfs.T
    #     self.core_equity_ptfs.columns = ["weights"]

    #     return self.core_equity_ptfs

    # def alpha_core_ptf(self):
    #     """
    #     Calculate an estimation of the alpha of the core equity portfolio.

    #     Returns:
    #         float: Alpha of the core equity portfolio.
    #     """
    #     # Ensure core equity portfolio weights are already computed
    #     if not hasattr(self, "core_equity_ptfs"):
    #         self.core_equity_ptf()

    #     # Compute the return of the core equity portfolio
    #     return_core_ptf = np.mean(self.core_equity_ptfs["weights"].T @ self.returns)
    #     # Compute the return of the benchmark
    #     return_benchmark = np.mean(self.benchmark)
    #     # Compute the alpha of the core equity portfolio
    #     alpha_core_ptf = return_core_ptf - return_benchmark

    #     return alpha_core_ptf

    # def simulate_alpha_impact(self, num_simulations=1000):
    #     """
    #     Simulate the impact of estimation errors in the covariance matrix on the alpha of the replicating portfolio.

    #     Args:
    #         num_simulations (int): Number of simulations to perform.

    #     Returns:
    #         tuple: Mean and 95% confidence interval of the estimated alpha.
    #     """
    #     alphas = []

    #     # Ensure core equity portfolio weights are already computed
    #     if not hasattr(self, "core_equity_ptfs"):
    #         self.core_equity_ptf()

    #     for _ in range(num_simulations):
    #         # Perturb the covariance matrix within a confidence region (e.g., using a multivariate normal distribution)
    #         perturbed_cov_matrix = self.perturb_cov_matrix()

    #         # Re-compute the core equity portfolio weights using the perturbed covariance matrix
    #         perturbed_weights = self.optim_routine(cov=perturbed_cov_matrix)

    #         # Compute the alpha for the perturbed weights
    #         return_core_ptf = np.mean(perturbed_weights.T @ self.returns)
    #         return_benchmark = np.mean(self.benchmark)
    #         alpha_core_ptf = return_core_ptf - return_benchmark

    #         alphas.append(alpha_core_ptf)

    #     # Calculate mean and confidence interval of alpha
    #     mean_alpha = np.mean(alphas)
    #     alpha_std = np.std(alphas)
    #     confidence_interval = norm.interval(
    #         0.95, loc=mean_alpha, scale=alpha_std / np.sqrt(num_simulations)
    #     )

    #     return mean_alpha, confidence_interval

    # def perturb_cov_matrix(self):
    #     """
    #     Perturb the covariance matrix within a confidence region.

    #     Returns:
    #         numpy.ndarray: Perturbed covariance matrix.
    #     """
    #     # You can implement a method to perturb the covariance matrix here
    #     # For example, you can use a multivariate normal distribution with mean=original covariance and some covariance matrix representing estimation errors
    #     # This is a simplified example; you may want to adjust it based on your specific needs
    #     estimation_errors = np.random.normal(
    #         loc=0, scale=0.01, size=self.cov_matrix.shape
    #     )
    #     perturbed_cov_matrix = self.cov_matrix + estimation_errors
    #     return perturbed_cov_matrix
