import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.multivariate.pca import PCA as smPCA
from scipy.stats import norm
from scipy.optimize import minimize
import os
import warnings

sns.set_style("darkgrid")


class CorePtf(object):
    """

    Class that computes the core equity portfolio and answers the questions of the assignment.
    It also contains methods to plot the results.

    """

    def __init__(self, dpi=300):
        """

        Initializes the CorePtf object and runs all the relevant methods to compute the core equity portfolio and answer the questions of the assignment.

        Takes as input:

            - dpi (int): Dots per inch of the plots (default: 300);

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
            - compute_higher_factor_ptf: Computes the weights of the portfolios replicating the second and third core equity factors;
            - plot_cumulative_variance_explained: Plots the cumulative variance explained by the Principal Components (Full model);
            - plot_variance_explained: Plots the variance explained by the select Principal Components;
            - plot_compared_performance: Plots the cumulative performance of the first core equity replicating portfolio and the benchmark;
            - plot_core_ptf_comp: Plots the composition of the first core equity portfolio;
            - plot_alpha_distribution: Plots the distribution of the alpha of the first core equity factor replicating portfolio;
            - plot_mean_vol_sim: Plots the mean return against the volatility of the simulated portfolios;
        """
        self.dpi = dpi

        # Set the path to the data
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
        #Question I - 1

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
        self.cumulative_variance_explained = np.cumsum(self.variance_explained)  # type: ignore

    def check_loading_sign(self):
        """
        Question I - 3

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
        Question I - 2

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
        Question II - 1

        Function that selects the number of Principal Components to retain in the final model.
        The number of Principal Components is selected based on the Bai-Ng (2002) Criterion.
        Then the full model is adapted so that it only contains the selected Principal Components.

        Takes as input:
            None;
        Output:
            None;
        """
        # Bai-Ng (2002) Criterion: Selection based on the information criteria provided by reduced form models with up to 20 factors
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
        Question II - 2

        Function that runs an OLS regression of each stocks returns on the K selected Principal Components.
        The first Principal Component is the 1st core equity factor. The first beta of the regression is the exposure of the stock to the core equity factor.
        The exposure of the stock to the core equity factor is stored in the core_eq_1_exp vector and will be used to compute the weights of the core equity portfolio.

        Takes as input:
            None;

        Output:
            None;
        """

        self.pca_models = {}
        self.core_eq_1_exp = np.zeros(len(self.stocks))
        self.core_eq_2_exp = np.zeros(len(self.stocks))
        self.core_eq_3_exp = np.zeros(len(self.stocks))

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
                self.core_eq_2_exp[i] = self.pca_models[stock]["beta"][1]
                self.core_eq_3_exp[i] = self.pca_models[stock]["beta"][2]

            except KeyError as e:
                print(f"Error for stock {i}: {e}")
                print("Columns in reduced_pc_scores:")
                print(self.pc_scores.columns.tolist())  # type: ignore
                print("Columns in returns:")
                print(self.returns.columns.tolist())
                raise

    def optim_routine(self, covariance_matrix, factor=1):
        core_eq_exp = (
            self.core_eq_1_exp
            if factor == 1
            else (
                self.core_eq_2_exp
                if factor == 2
                else (self.core_eq_3_exp if factor == 3 else None)
            )
        )

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
            {"type": "eq", "fun": lambda W: W.T @ core_eq_exp - 1.0},
        ]

        optimized = minimize(
            objective,
            W,
            args=(
                core_eq_exp,
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
        Question III

        Function that returns the weights of the core equity portfolio, these weights are computed using the optimization routine defined above.

        Takes as input:
            None;

        Output:
            core_equity_ptf (pd.DataFrame): Pandas DataFrame containing the weights of the core equity portfolio (First factor replacing portfolio);
        """

        # Ensure the PCA model is already computed
        if not hasattr(self, "core_eq_1_exp"):
            self.pca_model()

        self.core_equity_w = np.array(self.optim_routine(self.cov_matrix))

        # Reformat the dictionary to a pandas dataframe with the columns being the stocks and the row being the weight
        self.core_equity_ptf = pd.DataFrame(
            data=self.core_equity_w, columns=["weights"], index=self.stocks
        )

        return self.core_equity_ptf

    def alpha_core_ptf(self):
        """
        Question IV

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

        # Compute the alpha and the beta of the core equity portfolio using a simple OLS regression
        model = OLS(
            self.return_core_ptf, add_constant(self.benchmark), hasconst=True
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

        # Store the results in a dictionary
        self.ptf_stats = {
            "Average return (annualized)": np.mean(self.return_core_ptf) * 12,
            "Total return": self.total_return_core_ptf.iloc[-1],
            "Volatility (annualized)": self.core_ptf_vol,
            "Alpha": self.alpha_core,
            "Beta": self.beta_core,
            "Sharpe": np.mean(self.return_core_ptf) * 12
            - np.mean(self.rf) / self.core_ptf_vol,
            "RMSE": rmse,
        }

        # Round the results to 4 decimals
        for key in self.ptf_stats:
            self.ptf_stats[key] = round(self.ptf_stats[key], 4)

    def simulate_alpha_impact(self, num_simulations=1000):
        """
        Question V

        Simulate the impact of estimation errors in the covariance matrix on the alpha of the replicating portfolio.

        Takes as input:
            num_simulations (int): Number of simulations to perform (default: 1000);

        Output:
            self.alpha_stats (Dict): Dictionary containing the mean, standard deviation and confidence interval of the alpha of the replicating portfolio;
        """

        # Ensure core equity portfolio weights are already computed
        if not hasattr(self, "core_eq_1_exp"):
            self.pca_model()

        self.alphas = []
        perfs = []
        self.sim_ptf_returns = []
        self.sim_ptf_vol = []

        for _ in range(num_simulations):
            # Sample half of the returns to compute the covariance matrix (perturbed)
            sample_size = int(len(self.returns) * 0.95)
            sample = np.random.permutation(self.returns)[:sample_size]
            perturbed_cov_matrix = np.cov(sample.T, bias=True)

            # Re-compute the core equity portfolio weights using the perturbed covariance matrix
            perturbed_weights = self.optim_routine(perturbed_cov_matrix)

            # Compute the alpha for the perturbed weights
            return_core_ptf = self.returns @ perturbed_weights
            sim_perf = np.mean(return_core_ptf) * 12
            result = OLS(return_core_ptf, add_constant(self.benchmark)).fit()
            alpha_sim = result.params.iloc[0] * 12

            self.sim_ptf_returns.append(np.mean(return_core_ptf) * 12)
            self.sim_ptf_vol.append(
                np.sqrt(
                    perturbed_weights.T @ perturbed_cov_matrix @ perturbed_weights * 12
                )
                ** 0.5
            )
            self.alphas.append(alpha_sim)
            perfs.append(sim_perf)

        # Calculate mean and confidence interval of alpha
        self.mean_alpha = np.mean(self.alphas)
        self.alpha_std = np.std(self.alphas) * np.sqrt(12)
        self.alpha_confidence_interval = norm.interval(
            0.95, loc=self.mean_alpha, scale=self.alpha_std / np.sqrt(num_simulations)
        )

        self.alpha_stats = {
            "mean": self.mean_alpha,
            "std": self.alpha_std,
            "confidence interval": self.alpha_confidence_interval,
        }
        # Round the results to 4 decimals
        self.alpha_stats = {
            key: round(value, 4)
            if not isinstance(value, tuple)
            else tuple(round(v, 4) for v in value)
            for key, value in self.alpha_stats.items()
        }

        return self.alpha_stats

    def compute_higher_factor_ptf(self):
        """

        Function that returns the weights of the portfolios replicating the second and third equity factors. These weights are computed using the optimization routine defined above.

        Takes as input:
            None;

        Output:
            higher_factor_ptfs (Dict): Dictionary containing the weights of the 2nd and third factor replicating portfolios;
        """

        # Ensure the PCA model is already computed
        if not hasattr(self, "core_eq_1_exp"):
            self.pca_model()

        core_equity_w_2 = np.array(self.optim_routine(self.cov_matrix, factor=2))
        core_equity_w_3 = np.array(self.optim_routine(self.cov_matrix, factor=3))

        self.core_equity_ptf_2 = pd.DataFrame(
            data=core_equity_w_2, columns=["weights"], index=self.stocks
        )

        self.core_equity_ptf_3 = pd.DataFrame(
            data=core_equity_w_3, columns=["weights"], index=self.stocks
        )

        perf_ptf_2 = np.mean(self.returns @ self.core_equity_ptf_2["weights"]) * 12
        perf_ptf_3 = np.mean(self.returns @ self.core_equity_ptf_3["weights"]) * 12
        vol_ptf_2 = (
            np.sqrt(
                self.core_equity_ptf_2["weights"].T
                @ self.cov_matrix
                @ self.core_equity_ptf_2["weights"]
                * 12
            )
            ** 0.5
        )

        vol_ptf_3 = (
            np.sqrt(
                self.core_equity_ptf_3["weights"].T
                @ self.cov_matrix
                @ self.core_equity_ptf_3["weights"]
                * 12
            )
            ** 0.5
        )
        higher_factor_ptfs = {
            "Average return (annualized)": [perf_ptf_2, perf_ptf_3],
            "Volatility (annualized)": [vol_ptf_2, vol_ptf_3],
        }

        # Round the results to 4 decimals
        for key in higher_factor_ptfs:
            higher_factor_ptfs[key] = [
                round(higher_factor_ptfs[key][0], 4),
                round(higher_factor_ptfs[key][1], 4),
            ]

        return higher_factor_ptfs

    def format_axis_percentage(self, ax, axis="both", precision=1):
        """
        Formats the axis ticks in percentage with a specified precision.

        Parameters:
            ax (Matplotlib Axes): The Axes on which to apply the formatting.
            axis (str): The axis to format. Can be 'both', 'x', or 'y'.
            precision (int): The number of decimals to display.

        Returns:
            None
        """
        percentage_format = "{:." + str(precision) + "%}"
        if axis in ["both", "x"]:
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: percentage_format.format(x))
            )
        if axis in ["both", "y"]:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda y, _: percentage_format.format(y))
            )

    def plot_cumulative_variance_explained(self, savefig=False):
        """
        Plot the cumulative variance explained by the Principal Components (Full model)

        Takes as input:
            - savefig (bool): Whether to save the figure or not (default: False);

        Output:
            Matplotlib.plt plot;
        """
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.plot(
            range(1, len(self.cumulative_variance_explained) + 1),
            self.cumulative_variance_explained,
        )
        plt.xlabel("Principal Components")
        plt.ylabel("Cumulative Variance Explained")
        plt.title("Cumulative Variance Explained by the Principal Components")
        self.format_axis_percentage(plt.gca(), axis="y")
        plt.show()

        if savefig:
            plt.savefig("Cumulative Variance Explained by the Principal Components.png")

    def plot_variance_explained(self, savefig=False):
        """
        Plot the variance explained by each Principal Component

        Takes as input:
            - savefig (bool): Whether to save the figure or not (default: False);

        Output:
            Matplotlib.plt bar plot;
        """
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.bar(
            x=range(1, len(self.variance_explained) + 1),
            height=self.variance_explained,
        )
        plt.xticks(range(1, len(self.variance_explained) + 1), ["PC1", "PC2", "PC3"])
        plt.xlabel("Principal Components")
        plt.ylabel("Variance Explained")
        plt.title("Variance Explained by each Principal Component")
        self.format_axis_percentage(plt.gca(), axis="y")
        plt.show()

        if savefig:
            plt.savefig("Variance Explained by the selected Principal Components.png")

    def plot_compared_performance(self, factor=1, savefig=False):
        """
        Plot the performance of the core equity portfolio and the benchmark.

        Takes as input:
            - factor (int): The factor replicating portfolio to plot (default: 1);
            - savefig (bool): Whether to save the figure or not (default: False);

        Output:
            Matplotlib.plt plot;
        """
        # Ensure core equity portfolio weights are already computed
        if not hasattr(self, "total_return_core_ptf"):
            self.alpha_core_ptf()

        if factor != 1:
            if not hasattr(self, "core_equity_ptf_2"):
                self.compute_higher_factor_ptf()

            ptf_2_total_return = (
                np.cumprod(1 + self.returns @ self.core_equity_ptf_2["weights"]) - 1
            )
            ptf_3_total_return = (
                np.cumprod(1 + self.returns @ self.core_equity_ptf_3["weights"]) - 1
            )
            # Plot the performance of the factors replicating portfolios and the benchmark
            plt.figure(figsize=(10, 6), dpi=self.dpi)
            plt.plot(
                self.total_return_core_ptf,
                label="1st Factor Replicating Portfolio Total Return",
            )
            plt.plot(
                ptf_2_total_return,
                label="2nd Factor Replicating Portfolio Total Return",
            )
            plt.plot(
                ptf_3_total_return,
                label="3rd Factor Replicating Portfolio Total Return",
            )
            plt.plot(self.total_return_benchmark, label="Benchmark Total Return (SX5E)")
            plt.legend()
            plt.title(
                "Total Performance of the Factors Replicating Portfolios vs. Benchmark"
            )
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            self.format_axis_percentage(plt.gca(), axis="y")
            plt.show()

            if savefig:
                plt.savefig(
                    "Total Performance of the Factors Replicating Portfolios vs. Benchmark.png"
                )
        else:
            # Plot the performance of the first core equity portfolio and the benchmark
            plt.figure(figsize=(10, 6), dpi=self.dpi)
            plt.plot(
                self.total_return_core_ptf,
                label="First Core Equity Portfolio Total Return",
            )
            plt.plot(self.total_return_benchmark, label="Benchmark Total Return (SX5E)")
            plt.fill_between(
                self.total_return_core_ptf.index,
                self.total_return_core_ptf,
                self.total_return_benchmark,  # type: ignore
                color="firebrick",
                alpha=0.1,
            )
            plt.legend()
            plt.title("Performance of the Core Equity Portfolio vs. Benchmark")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            self.format_axis_percentage(plt.gca(), axis="y")
            plt.show()

            if savefig:
                plt.savefig(
                    "Performance of the First Core Equity Portfolio vs. Benchmark.png"
                )

    def plot_core_ptf_comp(self, savefig=False):
        """
        Pie chart showing the composition of the first core equity portfolio

        Takes as input:
            - savefig (bool): Whether to save the figure or not (default: False);

        Output:
            Matplotlib.plt pie chart;
        """
        if not hasattr(self, "core_equity_ptf"):
            self.compute_core_equity_ptf()

        warnings.filterwarnings("ignore")

        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)

        wedges, texts, autotexts = ax.pie(
            self.core_equity_ptf["weights"],
            labels=self.stocks,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )

        for text in texts + autotexts:
            text.set_fontsize(8)

        plt.title("Composition of the First Core Equity Portfolio")
        plt.show()

        if savefig:
            plt.savefig("Composition of the First Core Equity Portfolio.png")

        warnings.filterwarnings("default")

    def plot_alpha_distribution(self, savefig=False):
        """
        Plot the distribution of the alpha of the first core equity factor replicating portfolio

        Takes as input:
            - savefig (bool): Whether to save the figure or not (default: False);

        Output:
            Matplotlib.plt histogram;
        """
        if not hasattr(self, "alphas"):
            self.simulate_alpha_impact()

        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.hist(self.alphas, bins=50)
        plt.xlabel("Alpha of the First Factor Replicating Portfolio")
        plt.ylabel("Frequency")
        plt.title("Distribution of the Alpha of the First Factor Replicating Portfolio")
        self.format_axis_percentage(plt.gca(), axis="x")
        legend_text = (
            r"$\mu_{\hat{\alpha}} = $"
            + f"{self.mean_alpha:.{2}%}\n"
            + r"$\sigma_{\hat{\alpha}} = $"
            + f"{self.alpha_std:.{2}%}"
        )
        plt.legend([legend_text], loc="upper right")
        plt.show()

        if savefig:
            plt.savefig(
                "Distribution of the Alpha of the First Factor Replicating Portfolio.png"
            )

    def plot_mean_vol_sim(self, savefig=False):
        """
        Plot the mean return and volatility of the simulated portfolios

        Takes as input:
            - savefig (bool): Whether to save the figure or not (default: False);

        Output:
            Matplotlib.plt scatter plot;
        """
        if not hasattr(self, "sim_ptf_returns"):
            self.simulate_alpha_impact()

        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.scatter(self.sim_ptf_vol, self.sim_ptf_returns)
        plt.xlabel("Volatility (Annualized)")
        plt.ylabel("Mean Return (Annualized)")
        plt.title("Mean Return vs. Volatility of the Simulated Portfolios")
        self.format_axis_percentage(plt.gca(), axis="both")
        plt.show()

        if savefig:
            plt.savefig("Mean Return vs. Volatility of the Simulated Portfolios.png")
