import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import norm
from scipy.optimize import minimize
from statsmodels.multivariate.pca import PCA as smPCA


class PCA(object):
    """

    This class derives the principal components on a given data matrix containing stocks returns.
    First the returns are standardized and then the PCA is performed on the standardized returns.

    """

    def __init__(self, returns, stocks):
        """

        Initializes the PCA object.

        Takes as input:

            - returns (pd.DataFrame): Pandas DataFrame containing the spot values of the yields on which PCA is performed;
            - stocks (List): The list of tenors on which PCA is performed;

        Variables list:

            - benchmark (pd.DataFrame): Pandas DataFrame containing the returns of the benchmark (stock index);
            - returns (pd.DataFrame): Pandas DataFrame containing the returns of the stocks universe on which PCA is performed;
            - scalers (Dict): Dictionary containing the scalers used to standardize the returns of each stock;
            - stocks (List): The list of stocks tickers on which PCA is performed;
            - std_returns (pd.DataFrame): Pandas DataFrame containing the standardized returns of the stocks universe on which PCA is performed;
            - benchmark_vol (float): The volatility of the benchmark;
            - cov_matrix (pd.DataFrame): Pandas DataFrame containing the covariance matrix of the returns;
            - full_model (statsmodels.multivariate.pca.PCA): PCA model computed on the standardized returns;
            - eigenvalues (np.ndarray): Eigenvalues of the standardized returns covariance matrix;
            - eigenvectors (np.ndarray): Eigenvectors of the standardized returns covariance matrix;
            - pc_scores (pd.DataFrame): Pandas DataFrame containing the PC scores of the standardized returns;
            - pc_loadings (pd.DataFrame): Pandas DataFrame containing the PC loadings of the standardized returns;
            - rescaled_eigenvalues (np.ndarray): Rescaled eigenvalues of the standardized returns covariance matrix;
            - variance_explained (np.ndarray): Variance explained by each PC;
            - rescaled_pc_scores (pd.DataFrame): Pandas DataFrame containing the rescaled PC scores of the standardized returns to the same volatility as that of the benchmark;
            - pca_models (Dict): Dictionary containing the OLS regression results of each stock returns on the K selected Principal Components;
            - core_eq_1_exp (np.ndarray): Vector containing the exposure of each stock to the core equity factor 1;
            - core_equity_w (np.ndarray): Vector containing the weights of the core equity portfolio;
            - core_equity_ptf (pd.DataFrame): Pandas DataFrame containing the weights of the core equity portfolio (Equity Portfolio replicating the first equity factor);
        """

        self.benchmark = returns.iloc[
            :, 0
        ]  # pd.DataFrame(returns[:, 0], index=returns[:, 0], columns=["benchmark"])
        self.returns = returns.iloc[
            :, 1:
        ]  # pd.DataFrame(returns[:, 1:], index=returns[:, 0], columns=stocks)
        self.scalers = {}  # Dictionary to store scalers
        self.stocks = stocks[1:]  # List of stocks tickers on which PCA is performed

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
        self.compute_model()  # Compute the PCA model
        self.rescale_pc()  # Rescale the PC scores to the same volatility as that of the benchmark
        self.pca_model()  # Run an OLS regression of each stocks returns on the K selected Principal Components
        self.compute_core_equity_ptf()  # Compute the weights of the core equity portfolio
        # self.alpha_core_ptf()  # Calculate an estimation of the alpha of the core equity portfolio and its sharpe ratio

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

    def compute_model(self):
        """
        Function that computes the Principal Components model on the standardized stocks returns.

        Takes as input:
            None;

        Output:
            None;
        """

        self.full_model = smPCA(self.std_returns)
        self.eigenvalues = self.full_model.eigenvals
        self.eigenvectors = self.full_model.eigenvecs
        self.pc_scores = (
            self.full_model.scores
        )  # Dataframe of PC scores (n x len(stocks))
        # rename columns
        self.pc_scores.columns = ["PC" + str(i) for i in range(1, len(self.stocks) + 1)]  # type: ignore
        self.pc_loadings = self.full_model.loadings  # PC loadings
        self.pc_loadings.columns = ["PC" + str(i) for i in range(1, len(self.stocks) + 1)]  # type: ignore
        # Rescaled Eigenvalues
        self.rescaled_eigenvalues = self.eigenvalues / np.mean(self.eigenvalues)  # type: ignore
        self.variance_explained = self.rescaled_eigenvalues / self.rescaled_eigenvalues.sum()  # type: ignore

    def rescale_pc(self):
        """
        Function that rescales the Principal Components to the same volatility as that of the benchmark.
        First PC is the core equity factor 1 will capture the most variance in the data. (2007 - 2008 crisis)

        Takes as input:
            None;

        Output:
            None;
        """

        # Rescale the PC scores to the same volatility as that of the benchmark
        self.rescaled_pc_scores = (
            self.pc_scores * self.benchmark_vol / self.pc_scores.std()  # type: ignore
        )

    # def select_pc_number(self):
    #     """
    #     Function that selects the number of Principal Components to retain in the final model.
    #     The number of Principal Components is selected based on the Kaiser Criterion and Bai-Ng (2002) Criterion.

    #     Takes as input:
    #         None;
    #     Output:
    #         k (int): The number of Principal Components retained in the final model.
    #     """

    #     threshold = 0.80  # Threshold for the variance explained

    #     # Select the number of Principal Components based on the variance explained

    #     # Select the number of Principal Components based on the Kaiser Criterion;
    #     k_kaiser = self.full_model.ic.shape[1]  # type: ignore

    #     # Bai-Ng (2002) Criterion: Select based on BIC
    #     bic_values = self.full_model.ic

    #     # Find the index corresponding to the minimum BIC value
    #     k_bic = np.argmin(bic_values) + 1  # Adding 1 because the loop starts from 1

    #     # Select the minimum between Kaiser and Bai-Ng
    #     k = min(k_kaiser, k_bic)

    #     # Ensure that k is at least 1
    #     k = max(k, 1)

    #     return k

    def pca_model(self):
        """
        Function that runs an OLS regression of each stocks returns on the K selected Principal Components.
        The first Principal Component is the core equity factor. The first beta of the regression is the exposure of the stock to the core equity factor.
        The exposure of the stock to the core equity factor is stored in the core_eq_1_exp vector and will be used to compute the weights of the core equity portfolio.

        Takes as input:
            None;

        Output:
            None;
        """

        self.pca_models = {}
        self.core_eq_1_exp = np.zeros(len(self.stocks))
        for i , stock in enumerate(self.stocks):
            y = np.array(self.returns[stock])
            X = np.array(self.rescaled_pc_scores)
            X = add_constant(X)  # Add a constant term for the intercept

            try:
                model_i = OLS(y, X, hasconst=True).fit()
                self.pca_models[stock] = {
                    "model_result": model_i,
                    "alpha": model_i.params[0],
                    "beta": model_i.params[1:],
                    "residuals": model_i.resid,
                }
                # Add Core equity factor 1 (the first beta of the regression) to the vector
                self.core_eq_1_exp[i] = self.pca_models[stock]["beta"][0]

            except KeyError as e:
                print(f"Error for stock {i}: {e}")
                print("Columns in reduced_pc_scores:")
                print(self.rescaled_pc_scores.columns.tolist())
                print("Columns in returns:")
                print(self.returns.columns.tolist())
                raise

    def optim_routine(self, covariance_matrix):
        core_eq_1_exp = self.core_eq_1_exp

        def objective(W):
            # calculate mean/variance of the portfolio
            util = np.dot(np.dot(W.T, covariance_matrix), W)
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
            {"type": "ineq", "fun": lambda W: W.T @ core_eq_1_exp - 1.0 == 0.0},
        ]

        optimized = minimize(
            objective,
            W,
            (core_eq_1_exp, covariance_matrix),
            method="SLSQP",
            constraints=c_,
            bounds=b_,
            options={"maxiter": 100, "ftol": 1e-08},
        )
        return optimized.x

    def compute_core_equity_ptf(self):
        """
        Function that returns the weights of the core equity portfolio.

        Takes as input:
            None;

        Output:
            core_equity_ptf (pd.DataFrame): Pandas DataFrame containing the weights of the core equity portfolio (First factor replacing portfolio);
        """

        self.core_equity_w = np.array(self.optim_routine(self.cov_matrix))

        # Reformat the dictionary to a pandas dataframe with the columns being the stocks and the row being the weight
        self.core_equity_ptf = pd.DataFrame(
            data=self.core_equity_w, columns=self.stocks
        )

        self.core_equity_ptf = self.core_equity_ptf.T
        self.core_equity_ptf.columns = ["weights"]

        return self.core_equity_ptf

    def alpha_core_ptf(self):
        """
        Calculate an estimation of the alpha of the core equity portfolio.

        Takes as input:
            None;

        Output:
            alpha_core_ptf (float): The alpha of the core equity portfolio;
            sharpe_ratio (float): The sharpe ratio of the core equity portfolio;
        """
        # Ensure core equity portfolio weights are already computed
        if not hasattr(self, "core_equity_ptf"):
            self.compute_core_equity_ptf()

        # Compute the return of the core equity portfolio
        return_core_ptf = np.mean(self.core_equity_ptf["weights"].T @ self.returns)
        # Compute the return of the benchmark
        return_benchmark = np.mean(self.benchmark)
        # Compute the alpha of the core equity portfolio
        alpha_core_ptf = return_core_ptf - return_benchmark
        sharpe_ratio = alpha_core_ptf / self.benchmark_vol
        return alpha_core_ptf, sharpe_ratio

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
