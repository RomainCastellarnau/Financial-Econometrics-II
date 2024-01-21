import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.multivariate.pca import PCA as smPCA
from scipy.stats import norm
from scipy.optimize import minimize
import os

sns.set_style("darkgrid")


class PCA(object):
    """

    This class derives the principal components on a given data matrix containing stocks returns.
    First the returns are standardized and then the PCA is performed on the standardized returns.

    """

    def __init__(self):
        """

        Initializes the PCA object and runs all the relevant methods to compute the core equity portfolio.

        Takes as input:

            - None;

        Methods:

            - compute_covariance_matrix: Calculates the covariance matrix of the stocks returns across the whole time horizon;
            - compute_full_model: Computes the Principal Components model on the standardized stocks returns;
            - select_pc_number: Selects the number of Principal Components to retain in the final model;
            - check_loading_sign: Checks the sign of the loadings of the first PC. If F1 the loading vector of the first PC contains more than 50% of negative values, the sign of the first factor is flipped;
            - rescale_pc: Rescales the Principal Components to the same volatility as that of the benchmark;
            - pca_model: Runs an OLS regression of each stocks returns on the K selected Principal Components an extracts the senstivity of each stock to the core equity factor 1;
            - compute_core_equity_ptf: Computes the weights of the core equity portfolio;
            - alpha_core_ptf: Computes the alpha of the core equity portfolio as well as other performance metrics;
            - plot_compared_performance: Plots the performance of the core equity portfolio and the benchmark;
            - simulate_alpha_impact(n_sim): Simulates the impact of estimation errors in the covariance matrix on the alpha of the replicating portfolio;

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
            - alpha_core (float): Alpha of the core equity portfolio;
            - beta_core (float): Beta of the core equity portfolio;
            - total_return_core_ptf (np.ndarray): Vector containing the total return of the core equity portfolio since inception;
            - total_return_benchmark (np.ndarray): Vector containing the total return of the benchmark since inception;
            - core_ptf_vol (float): Volatility of the core equity portfolio;
            - comparative_perf (float): Comparative performance of the core equity portfolio relative to the benchmark;
            - ptf_stats (Dict): Dictionary containing the performance metrics of the core equity portfolio;
            - mean_alpha (float): Mean alpha of the replicating portfolio;
            - alpha_std (float): Standard deviation of the alpha of the replicating portfolio;
            - alpha_confidence_interval (Tuple): Tuple containing the confidence interval of the alpha of the replicating portfolio;
        """
        path = os.path.join(os.path.dirname(os.getcwd()), "Data")

        # Load the data
        returns = (
            pd.read_excel(os.path.join(path, "Data.xlsx"), sheet_name="RETURNS")
            .rename(columns={"Unnamed: 0": "Date"})
            .set_index("Date")
        )

        self.rf = (
            pd.read_excel(
                os.path.join(path, "Risk_free_rate.xlsx"),
                sheet_name="FRED Graph",
                skiprows=10,
            )
            .set_index("observation_date")
            .rename(columns={"IRLTLT01FRM156N": "10Y OAT"})
            .rename_axis("Date")
        ).iloc[:, 0]
        self.rf.index = returns.index

        stocks = returns.columns.tolist()

        # Initialize the variables
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
                    returns[col].values.reshape(-1, 1)  # type: ignore
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
        self.alpha_core_ptf()  # Compute the alpha of the core equity portfolio

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
        contains more than 50% of negative values, the sign of the first factor is flipped.

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
            self.pc_scores.iloc[:, 0] = self.pc_scores.iloc[:, 0].apply(lambda x: -x)  # type: ignore
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
            X = add_constant(X)

            try:
                model_i = OLS(y, X).fit()
                self.pca_models[stock] = {
                    "model_result": model_i,
                    "alpha": model_i.params[0],
                    "beta": model_i.params[1:],
                    "residuals": model_i.resid,
                    "R2": model_i.rsquared,
                }
                # Add the core equity factor 1 (the first beta of the regression) to the vector of sensitivities
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
            ),
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
        Function that computes the alpha of the core equity portfolio as well as other performance metrics.
        The alpha is computed using a simple OLS regression of the core equity portfolio returns on the benchmark returns.

        Takes as input:
            None;

        Output:
            None;
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
        model = OLS(
            self.return_core_ptf, add_constant(self.return_benchmark), hasconst=True
        ).fit()
        self.alpha_core = model.params.iloc[0] * 12  # Annualized alpha
        self.beta_core = model.params.iloc[1]

        r_predicted = model.predict()
        r_residuals = self.return_core_ptf - r_predicted
        rmse = np.sqrt(np.mean(r_residuals**2))

        # Compute the total return of the index since inception (benchmark)
        self.total_return_benchmark = np.cumprod(1 + self.benchmark) - 1

        # Compute the volatility of the core equity portfolio
        self.core_ptf_vol = (
            np.sqrt(
                self.core_equity_ptf["weights"].T
                @ self.cov_matrix
                @ self.core_equity_ptf["weights"]
                * 12
            )
            ** 0.5
        )

        # Compute the alpha of the core equity portfolio & its sharpe ratio
        self.comparative_perf = np.mean(self.return_core_ptf) * 12 - np.mean(self.rf)

        # Store the results in a dictionary
        self.ptf_stats = {
            "average return (annualized)": np.mean(self.return_core_ptf) * 12,
            "total return": self.total_return_core_ptf.iloc[-1],
            "volatility (annualized)": self.core_ptf_vol,
            "alpha": self.alpha_core,
            "beta": self.beta_core,
            "sharpe": self.comparative_perf / self.core_ptf_vol,
            "rmse": rmse,
        }

        # Round the results to 4 decimals
        for key in self.ptf_stats:
            self.ptf_stats[key] = round(self.ptf_stats[key], 4)

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

    def simulate_alpha_impact(self, num_simulations=1000):
        """
        Simulate the impact of estimation errors in the covariance matrix on the alpha of the replicating portfolio.

        Takes as input:
            num_simulations (int): Number of simulations to perform;

        Output:
            alpha_stats (Dict): Dictionary containing the mean, standard deviation and confidence interval of the alpha of the replicating portfolio;
        """

        # Ensure core equity portfolio weights are already computed
        if not hasattr(self, "core_eq_1_exp"):
            self.pca_model()

        alphas = []
        perfs = []

        for _ in range(num_simulations):
            # Perturb the covariance matrix
            sample = np.random.permutation(self.returns)[int(len(self.returns) / 2) :]
            perturbed_cov_matrix = np.cov(sample.T, bias=True)

            # Re-compute the core equity portfolio weights using the perturbed covariance matrix
            perturbed_weights = self.optim_routine(perturbed_cov_matrix)

            # Compute the alpha for the perturbed weights
            return_core_ptf = self.returns @ perturbed_weights
            sim_perf = np.mean(return_core_ptf) * 12
            result = OLS(return_core_ptf, add_constant(self.return_benchmark)).fit()
            alpha_sim = result.params.iloc[0] * 12

            alphas.append(alpha_sim)
            perfs.append(sim_perf)

        # Calculate mean and confidence interval of alpha
        self.mean_alpha = np.mean(alphas)
        self.alpha_std = np.std(alphas)
        self.alpha_confidence_interval = norm.interval(
            0.95, loc=self.mean_alpha, scale=self.alpha_std / np.sqrt(num_simulations)
        )

        alpha_stats = {
            "mean": self.mean_alpha,
            "std": self.alpha_std,
            "confidence interval": self.alpha_confidence_interval,
        }

        return alpha_stats
