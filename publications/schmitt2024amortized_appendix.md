# Amortized Bayesian Workflow (Extended Abstract) - Appendix

---

#### Page 7

## A Closed-world diagnostics

In the following, let $\hat{\theta}_{1}^{(j)}, \ldots, \hat{\theta}_{S}^{(j)} \sim q_{\phi}\left(\theta \mid y^{(j)}\right)$ be $S$ draws from the amortized posterior $q_{\phi}(\cdot)$.

## A. 1 Normalized root mean-squared error

As a measure of posterior bias and variance, we assess the recovery of the ground-truth parameters, for example via the average normalized root mean squared error (RMSE) over the test set,

$$
\operatorname{NRMSE}=\frac{1}{J} \sum_{j=1}^{J} \frac{1}{\operatorname{range}\left(\theta_{*}\right)} \sqrt{\frac{1}{S} \sum_{s=1}^{S}\left(\theta_{s}^{(j)}-\hat{\theta}_{s}^{(j)}\right)^{2}}
$$

where range $\left(\theta_{*}\right)=\max _{k}\left(\theta_{*}^{(k)}\right)-\min _{k}\left(\theta_{*}^{(k)}\right)$.

## A. 2 Simulation-based calibration checking

Simulation-based calibration (SBC; [19, 21]) checking evaluates the uncertainty calibration of the amortized posterior. For the true posterior $p(\theta \mid y)$, all intervals $U_{q}(\theta \mid y)$ are well-calibrated for any quantile $q \in(0,1)[1]$,

$$
q=\iint \mathbf{I}\left[\theta_{*} \in U_{q}(\boldsymbol{\theta} \mid y)\right] p\left(y \mid \theta_{*}\right) p\left(\theta_{*}\right) \mathrm{d} \theta_{*} \mathrm{~d} y
$$

with indicator function $\mathbf{I}[\cdot]$. Insufficient calibration of the posterior manifests itself as violations of Eq. 3. To quantify these violations, we report the expected calibration error of the amortized posterior, computed as median SBC error of 20 posterior credible intervals with increasing centered quantiles from $0.5 \%$ to $99.5 \%$, averaged across the $J$ examples in the test set.

## B Testing for atypicality in step 1

Inspired by an out-of-distribution checking method for amortized inference under model misspecification [20], we use a sampling-based hypothesis test to flag atypical data sets where the trustworthiness of amortized inference might be impeded. Concretely, we use the sampling-based estimator for the maximum mean discrepancy (MMD; [8]),

$$
\operatorname{MMD}^{2}(p \| q)=\mathbb{E}_{x, x^{\prime} \sim p(x)}\left[\kappa\left(x, x^{\prime}\right)\right]+\mathbb{E}_{x, x^{\prime} \sim q(x)}\left[\kappa\left(x, x^{\prime}\right)\right]-2 \mathbb{E}_{x \sim p(x), x^{\prime} \sim q(x)}\left[\kappa\left(x, x^{\prime}\right)\right]
$$

where $\kappa(\cdot, \cdot)$ is a positive definite kernel and we aim to quantify the distance between the distributions $p, q$ based on samples.
In our case of atypicality detection in step $1, p$ is the distribution of training data $y$ used during simulation-based training, and $q$ is the opaque distribution behind the observed test data sets. We construct a hypothesis test, where the null hypothesis states that $p=q$. For $M$ training data sets $\left\{y^{(m)}\right\}_{m=1}^{M}$ and $K$ test data sets $\left\{y^{(k)}\right\}_{k=1}^{K}$, we first compute the sampling distribution of MMDs from $M$ MMD estimates based on training samples $y$ vs. $y^{(m)}$. This quantifies the natural sampling distribution for $M$-vs.-1 MMD estimates where both samples stem from the training set. We then compute the $\alpha=95 \%$ percentile, which marks the cut-

> **Image description.** A histogram displays the distribution of MMD² values for training and test samples, along with a cut-off threshold.
>
> The x-axis is labeled "MMD²" and ranges from approximately 0.4 to 1.2. The y-axis is labeled "Density" and ranges from 0.0 to 3.0.
>
> The histogram features two sets of bars:
>
> - Yellow/tan bars representing "Training samples (null distribution)". These bars are more prominent on the left side of the graph, peaking around an MMD² value of 0.6. A smoothed curve of the same color overlays the histogram.
> - Blue bars representing "Test samples". These bars are more prominent on the right side of the graph, peaking around an MMD² value of 1.1. A smoothed curve of the same color overlays the histogram.
>
> A vertical red line labeled "MMD²α cut-off" is positioned at approximately MMD² = 1.15.

Figure 5: Illustration of our sampling-based hypothesis test that flags atypical data sets where amortized inference has no accuracy guarantees.

off for the $5 \%$ most atypical training examples, and denote this threshold as $\mathrm{MMD}_{\alpha}^{2}$. For the $K$ data sets in the test sample, we then compute the MMD estimate of all $M$ training samples against each of the $k=1, \ldots, K$ test samples, here denoted as $\mathrm{MMD}_{k}^{2}$. Then, we put it all together and flag data sets as atypical when $\mathrm{MMD}_{k}^{2} \geq \mathrm{MMD}_{\alpha}^{2}$. The type-I error rate of this test can be set relatively high to obtain a conservative test that will flag many data sets for detailed investigation in further steps of our workflow.

---

#### Page 8

> **Image description.** The image contains three scatter plots arranged horizontally, each displaying the relationship between "Ground truth" and "Estimated" values for different parameters, namely μ (mu), σ (sigma), and ξ (xi).
>
> Each plot shares a similar structure:
>
> - **Axes:** The x-axis is labeled "Ground truth," and the y-axis is labeled "Estimated." Each axis has numerical tick marks and labels.
> - **Data Points:** Each plot contains numerous data points, represented by small, reddish-brown dots. Each dot also has a short vertical line segment associated with it, presumably representing error bars. The data points cluster around a dashed black diagonal line.
> - **Diagonal Line:** A dashed black line runs diagonally from the bottom-left to the top-right corner of each plot, representing the ideal scenario where the estimated value perfectly matches the ground truth.
> - **R² and r Values:** In the upper-left corner of each plot, there are two values displayed: "R² =" followed by a decimal number and "r =" followed by a decimal number. These likely represent the R-squared value and the Pearson correlation coefficient, respectively, indicating the goodness of fit and correlation between the ground truth and estimated values.
>
> Specific details for each plot:
>
> 1.  **Plot 1 (μ):**
>     - Title: "μ"
>     - X-axis range: approximately 3.4 to 4.2
>     - Y-axis range: approximately 3.4 to 4.2
>     - R² = 0.961
>     - r = 0.981
> 2.  **Plot 2 (σ):**
>     - Title: "σ"
>     - X-axis range: 0.0 to 1.0
>     - Y-axis range: 0.0 to 1.0
>     - R² = 0.984
>     - r = 0.992
> 3.  **Plot 3 (ξ):**
>     - Title: "ξ"
>     - X-axis range: approximately -0.4 to 0.4
>     - Y-axis range: approximately -0.4 to 0.4
>     - R² = 0.724
>     - r = 0.853
>
> The plots collectively visualize the parameter recovery performance, with each plot representing a different parameter. The clustering of data points around the diagonal line, along with the R² and r values, indicates the accuracy of the estimation process.

(a) The parameter recovery is excellent for the parameters $\mu, \sigma$ and good for the shape parameter $\xi$.

> **Image description.** The image presents three plots side-by-side, each depicting the rank ECDF (Empirical Cumulative Distribution Function) difference versus the fractional rank statistic for different parameters: mu (µ), sigma (σ), and xi (ξ). These plots are used for simulation-based calibration checking.
>
> Each plot shares a similar structure:
>
> - **Axes:** The x-axis represents the "Fractional rank statistic," ranging from 0.0 to 1.0. The y-axis represents the "ECDF difference," ranging from -0.15 to 0.15.
> - **Rank ECDF:** A red-brown line represents the "Rank ECDF." This line fluctuates around zero, indicating the difference between the empirical CDF of the ranks and the uniform CDF.
> - **Confidence Bands:** A shaded gray area represents the "95% Confidence Bands." This area provides a visual representation of the expected range of variation under ideal calibration. The confidence bands are shaped like a circle, with the ECDF difference plot contained within the circle.
> - **Titles:** Above each plot is the parameter it represents: "µ" (mu), "σ" (sigma), and "ξ" (xi).
>
> The plots visually assess the calibration of the parameters. If the Rank ECDF line stays within the 95% Confidence Bands, it suggests good calibration.

(b) Simulation-based calibration checking indicates excellent calibration for all parameters.
Figure 6: The closed-world diagnostics indicate acceptable convergence of the amortized posterior.

Note. In the case study of this paper, we perform the above test in the summary space, that is, we replace all occurences of $y$ with the learned neural summary statistics $h_{\psi}(y)$, where $h_{\psi}$ is a DeepSet that learns an 8-dimensional representation of the data (see below for details).

## C Experiment details

In this section, we provide experiment details for parameter inference of the generalized extreme value (GEV) distribution.

## C. 1 Problem description

Following Caprani et al. [2], the prior distribution is defined as:

$$
\begin{aligned}
& \mu \sim \mathcal{N}(3.8,0.04) \\
& \sigma \sim \text { Half-Normal }(0,0.09) \\
& \xi \sim \text { Truncated-Normal }(0,0.04) \text { with bounds }[-0.6,0.6]
\end{aligned}
$$

## C. 2 Simulation-based training

For the simulation-based training stage, we simulate 10000 tuples of parameters and observations from the parameter priors and the corresponding GEV distributions. Each data set contains 65 i.i.d. observations from the GEV distribution. The validation set, generated in the same manner, consists of 1000 samples from the joint model. The neural density estimator uses flow matching [13] as a generative neural network backbone. The internal network is a multilayer perception (MLP) with 5 layers of 128 units each, residual connections, and 5\% dropout. Before entering the flow matching network as conditioning variables, we pre-process the observations $y=\left(y_{1}, \ldots, y_{65}\right)$ with a DeepSet [25] that jointly learns an 8-dimensional embedding of the observations while accounting for the permutation-invariant structure of the data. The DeepSet has a depth of 1, uses a mish activation, max inner pooling layers, 64 units in the equivariant and invariant modules, and 5\% dropout. In accordance with common practice in computational Bayesian statistics (e.g., PyMC or Stan), the amortized neural approximator learns to estimate the parameters in an unconstrained parameter space.
Optimization. The neural network is optimized via the Adam optimizer [12], with a cosine decay applied to the learning rate (initial learning rate of $10^{-4}$, a warmup target of $10^{-3}, \alpha=10^{-3}$ ) as

---

#### Page 9

well as a global clipnorm of 1.0 . The batch size is set to 512 and the number of training epochs is 300 .

Diagnostics. The closed-world recovery (Figure 6a) and simulation-based calibration (Figure 6b) indicate that the neural network training has successfully converged to a trustworthy posterior approximator within the scope of the training set.
Inference data sets In order to emulate distribution shifts that arise in real-world applications while preserving the controlled experimental environment, we simulate the "observed" data sets from a joint model with a prior that has $4 \times$ the dispersion of the model used during training. More specifically, the prior is specified as:

$$
\begin{aligned}
& \mu \sim \mathcal{N}(3.8,0.16) \\
& \sigma \sim \text { Half-Normal }(0,0.36) \\
& \xi \sim \text { Truncated-Normal }(0,0.16) \text { with bounds }[-1.2,1.2]
\end{aligned}
$$

# C. 3 ChEES-HMC

We use $S=16$ superchains and $L=128$ subchains, resulting in a total number of $S \cdot L=2048$ chains. The initial step size is set to 0.1 . The number of warmup iterations is set to 200 . The number of sampling iterations is 1 , resulting in a total number of 2048 post-warmup MCMC draws.
