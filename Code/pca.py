import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.optimize import minimize
from statsmodels.multivariate.pca import PCA as smPCA

sns.set_style("darkgrid")


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
            - pc_scores (pd.DataFrame): Pandas DataFrame containing the PC scores of the standardized returns;
            - pc_loadings (pd.DataFrame): Pandas DataFrame containing the PC loadings of the standardized returns;
            - rescaled_eigenvalues (np.ndarray): Rescaled eigenvalues of the standardized returns covariance matrix;
            - variance_explained (np.ndarray): Variance explained by each PC;
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
        returns_std = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            if any(ticker in col for ticker in stocks):
                scaler = StandardScaler()
                returns_std[col] = scaler.fit_transform(
                    returns[col].values.reshape(-1, 1)
                )
                self.scalers[col] = scaler

        self.std_returns = returns_std
        self.std_returns = self.std_returns.iloc[:, 1:]
        self.benchmark_vol = self.benchmark.std()
        self.compute_covariance_matrix()  # Compute the covariance matrix of the returns not standardized
        self.compute_full_model()  # Compute the PCA model with len(stocks) Principal Components
        self.select_pc_number()  # Select the number of Principal Components to retain in the final model
        self.check_loading_sign()  # Check the sign of the loadings of the first PC
        self.rescale_pc()  # Rescale the PC scores to the same volatility as that of the benchmark
        self.pca_model()  # Run an OLS regression of each stocks returns on the K selected Principal Components
        self.compute_core_equity_ptf()  # Compute the weights of the core equity portfolio

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

    def compute_full_model(self):
        """
        Function that computes the Principal Components model on the standardized stocks returns.

        Takes as input:
            None;

        Output:
            None;
        """

        self.full_model = smPCA(self.std_returns)
        self.eigenvalues = self.full_model.eigenvals
        self.pc_scores = self.full_model.scores
        # rename columns
        self.pc_scores.columns = ["PC" + str(i) for i in range(1, len(self.stocks) + 1)]  # type: ignore
        self.pc_loadings = self.full_model.loadings  # PC loadings
        self.pc_loadings.columns = ["PC" + str(i) for i in range(1, len(self.stocks) + 1)]  # type: ignore
        # Rescaled Eigenvalues
        self.rescaled_eigenvalues = self.eigenvalues / np.mean(self.eigenvalues)  # type: ignore
        self.variance_explained = self.rescaled_eigenvalues / self.rescaled_eigenvalues.sum()  # type: ignore

    def check_loading_sign(self):
        """

        Function that checks the sign of the loadings of the first PC. If F1 the loading vector of the first PC
        contains more than 50% of negative values, the sign of the first loading is flipped.

        Takes as input:
            None;

        Output:
            None;
        """

        pc_1_loading = self.pc_loadings["PC1"]  # type: ignore
        if (
            pc_1_loading[pc_1_loading < 0].count()
            > pc_1_loading[pc_1_loading > 0].count()
        ):
            self.pc_loadings["PC1"] = -self.pc_loadings["PC1"]  # type: ignore
        else:
            pass

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
        self.pc_scores = (
            self.pc_scores * self.benchmark_vol / self.pc_scores.std()  # type: ignore
        )

    def select_pc_number(self):
        """
        Function that selects the number of Principal Components to retain in the final model.
        The number of Principal Components is selected based on the Bai-Ng (2002) Criterion.
        Then the full model is adapted so that it only contains the selected Principal Components.

        Takes as input:
            None;
        Output:
            None;
        """
        # Bai-Ng (2002) Criterion: Select based on the first 20 information criteria
        ic_values = self.full_model.ic.iloc[0:20, 0]  # type: ignore
        # Get the index corresponding to the minimum BIC value
        k = np.argmin(ic_values) + 1  # type: ignore

        # Adapt the full model so that it only contains the selected Principal Components
        self.eigenvalues = self.eigenvalues[:k]  # type: ignore
        self.pc_scores = self.pc_scores.iloc[:, :k]  # type: ignore
        self.pc_loadings = self.pc_loadings.iloc[:, :k]  # type: ignore
        self.rescaled_eigenvalues = self.rescaled_eigenvalues[:k]  # type: ignore
        self.variance_explained = self.variance_explained[:k]  # type: ignore

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
        for i, stock in enumerate(self.stocks):
            y = np.array(self.returns[stock])
            X = np.array(self.pc_scores)
            X = add_constant(X)  # Add a constant term for the intercept

            try:
                model_i = OLS(y, X, hasconst=True).fit()
                self.pca_models[stock] = {
                    "model_result": model_i,
                    "alpha": model_i.params[0],
                    "beta": model_i.params[1:],
                    "residuals": model_i.resid,
                    "R2": model_i.rsquared,
                }
                # Add Core equity factor 1 (the first beta of the regression) to the vector
                self.core_eq_1_exp[i] = self.pca_models[stock]["beta"][0]

            except KeyError as e:
                print(f"Error for stock {i}: {e}")
                print("Columns in reduced_pc_scores:")
                print(self.pc_scores.columns.tolist())  # type: ignore
                print("Columns in returns:")
                print(self.returns.columns.tolist())
                raise

    def optim_routine(self, covariance_matrix):
        core_eq_1_exp = self.core_eq_1_exp

        def objective(W, R, C):
            # calculate mean/variance of the portfolio
            util = np.dot(np.dot(W.T, covariance_matrix), W)
            return util

        n = len(self.stocks)
        # initial conditions: equal weights
        W = np.ones([n]) / n
        # weights between 0%..100%: no shorts
        b_ = [(0.0, 1.0) for i in range(n)]
        # No leverage: unitary constraint (sum weights = 100%)
        c_ = [
            {"type": "eq", "fun": lambda W: sum(W) - 1.0},
            {"type": "eq", "fun": lambda W: W.T @ core_eq_1_exp - 1.0},
        ]

        optimized = minimize(
            objective,
            W,
            args=(
                core_eq_1_exp,
                covariance_matrix,
            ),  # Use args to pass additional arguments
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
            data=self.core_equity_w, columns=["weights"], index=self.stocks
        )

        return self.core_equity_ptf

    def alpha_core_ptf(self):
        """
        Calculate an estimation of the alpha of the core equity portfolio.

        Takes as input:
            None;

        Output:
            alpha_core_ptf (float): The alpha of the core equity portfolio;
            comparative_perf (float): The comparative performance of the core equity portfolio to the benchmark;
            return_core_ptf (pd.DataFrame): Pandas DataFrame containing the returns of the core equity portfolio;
            rmse (float): The root mean squared error of the regression;
        """
        # Ensure core equity portfolio weights are already computed
        if not hasattr(self, "core_equity_ptf"):
            self.compute_core_equity_ptf()

        # Compute the total return of the core equity portfolio since inception
        self.total_return_core_ptf = (
            np.cumprod(1 + self.returns @ self.core_equity_ptf["weights"]) - 1
        )

        self.return_core_ptf = self.returns @ self.core_equity_ptf["weights"]
        self.return_benchmark = self.benchmark

        # Compute the alpha and the beta of the core equity portfolio using a simple OLS regression
        model = OLS(self.return_core_ptf, self.return_benchmark, hasconst=True).fit()
        self.alpha_core = model.params[0]
        self.beta_core = model.params[1]

        r_predicted = model.predict()
        r_residuals = self.return_core_ptf - r_predicted
        rmse = np.sqrt(np.mean(r_residuals**2))

        # Compute the total return of the index since inception (benchmark)
        self.total_return_benchmark = np.cumprod(1 + self.benchmark) - 1

        # Compute the volatility of the core equity portfolio
        self.core_ptf_vol = np.sqrt(
            self.core_equity_ptf["weights"].T
            @ self.cov_matrix
            @ self.core_equity_ptf["weights"]
        )

        # Compute the alpha of the core equity portfolio & its sharpe ratio
        comparative_perf = np.mean(
            self.total_return_core_ptf - self.total_return_benchmark
        )

        self.comparative_perf = comparative_perf

        return self.alpha_core, self.comparative_perf, self.return_core_ptf, rmse

    def plot_compared_performance(self):
        """
        Plot the performance of the core equity portfolio and the benchmark.

        Takes as input:
            None;

        Output:
            None;
        """
        # Ensure core equity portfolio weights are already computed
        if not hasattr(self, "total_return_core_ptf"):
            self.alpha_core_ptf()

        # Plot the performance of the core equity portfolio and the benchmark
        plt.figure(figsize=(10, 6))
        plt.plot(self.total_return_core_ptf, label="Core Equity Portfolio Total Return")
        plt.plot(self.total_return_benchmark, label="Benchmark Total Return")
        plt.legend()
        plt.title("Performance of the Core Equity Portfolio vs. Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.show()

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
