from CorePtf import CorePtf

if __name__ == "__main__":

    ############################################
    
    # Initialization of the class
    CPtf = CorePtf()

    # Question 1 - We first compute the PCA model of the returns of the 47 stocks returns
    # - The covariance matrix is computed using the stocks returns (not standardized) and stored in the class
    # - The PCA full model is computed by calling the function compute_full_model on the standardized stocks returns
    # - First PC loading sign is checked and pc scores 1 sign are flipped if needed
    # - The reduced form model is then built by selecting the first 3 PC scores (Bai and Ng 2002 criteria)
    # - PC scores are being rescaled to the volatility of the benchmark (SX5E)

    CPtf.compute_covariance_matrix()  # Compute the covariance matrix of the returns not standardized
    CPtf.compute_full_model()  # Compute the PCA model with 47 Principal Components
    CPtf.select_pc_number()  # Select the number of Principal Components to retain in the final model
    CPtf.check_loading_sign()  # Check the sign of the loadings of the first PC
    CPtf.rescale_pc()  # Rescale the PC scores to the same volatility as that of the benchmark
    
    # We plot the cumulative variance explained by the PC
    CPtf.plot_cumulative_variance_explained()

    # We then plot the variance explained by the selected PC with criterion
    CPtf.plot_variance_explained()

    ############################################

    # Question 2 - Each stocks returns are regressed on all the selected PC scores using an OLS regression
    # - The regression coefficients are then stocked and will be used to build the factor replicating portfolios;
    # These coefficients represent the exposure of each stock returns to the K core equity factors.

    CPtf.pca_model()

    ############################################

    # Question 3 - The first core equity replicating portfolios is then built using the regression coefficients and
    # an optimization routine (minimize the variance of the portfolio)

    CPtf.compute_core_equity_ptf()
    core_eq_composition = CPtf.core_equity_ptf

    ############################################

    # Question 4 - We can now compute the returns of the first core equity factor and estimate its alpha against the benchmark (SX5E)
    # - The alpha is computed using a simple OLS regression of the replicating portfolio returns against the benchmark returns
    # - The relevant performance metrics are also computed and stored in a dictionary (see CorePtf.py)

    CPtf.alpha_core_ptf()
    core_eq_1_ptf_stat = CPtf.ptf_stats

    # We can now plot the cumulative returns of the first core equity factor against the benchmark (SX5E)
    CPtf.plot_compared_performance()

    # And analyze its composition
    CPtf.plot_core_ptf_comp()

    ############################################

    # Question 5 - We are now interested in the impact of the estimation error of the variance covariance matrix on the performance of the core equity factor
    # - We randomly choose a sample containing 95% of the returns / compute the associated variance covariance matrix
    # - We then build for each different variance-covariance matrix the core equity factor replicating portfolio and compute its alpha
    # - Having obtained the distribution of the alpha, we can compute it's mean and standard deviation of the alpha and build a confidence interval
    # This process is done by the function simulate_alpha_impact() (this takes a minute and a half to run)
    # alpha_stats is a dictionary containing the mean, the standard deviation of the estimated alphas and a 95% confidence interval

    alpha_stats = CPtf.simulate_alpha_impact()

    # We can now plot the distribution of the alpha of the first core equity factor replicating portfolio
    CPtf.plot_alpha_distribution()

    # And the Mean-Volatility space of the first core equity factor replicating portfolio simulated
    CPtf.plot_mean_vol_sim()

    ############################################
