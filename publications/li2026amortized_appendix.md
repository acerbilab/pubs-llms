# Amortized Bayesian Workflow - Appendix

---

## Appendix

This appendix provides additional details and analyses to complement the main text, included in the following sections:

Background, Appendix A Best practices for training amortized estimators, Appendix B Experiment details, Appendix C Additional experimental study of the OOD diagnostic in Step 1, Appendix D Amortized initialization for NUTS, Appendix E

## A Background

This section provides a concise overview of the diagnostics and algorithms used in our workflow, including simulation- based calibration checking, parameter recovery checking, out- of- distribution diagnostic with Mahalanobis distance, Pareto- smoothed importance sampling, the ChEES- HMC algorithm, and the nested \(\hat{R}\) convergence diagnostic. Pseudocodes are also given for reference.

Simulation- based calibration checking. Simulation- based calibration (SBC; Talts et al., 2018; Modrak et al., 2025) is a principled technique for assessing the calibration of posterior distributions estimated by Bayesian inference procedures, particularly useful in simulation- based amortized inference settings. SBC is based on the idea that if the posterior \(p(\theta \mid y)\) is correctly specified, then the rank of the true parameter \(\theta_{\star}\) among posterior draws should follow a uniform distribution. Formally, SBC defines a test statistic \(f:\Theta \times Y\to \mathbb{R}\) (e.g., a component of \(\theta\) , or the log- likelihood \(p(y|\theta)\) ). For each simulated dataset \(y^{(j)}\) generated from the joint model \(p(\theta ,y)\) , the test statistic is evaluated at the ground- truth parameter \(\theta_{\star}^{(j)}\) and compared to the same statistic evaluated over posterior samples \(\{\theta_{s}^{(j)}\} \sim q_{\phi}(\theta |y^{(j)})\) . The rank of \(f(\theta_{\star}^{(j)},y^{(j)})\) among \(\{f(\theta_{s}^{(j)},y^{(j)})\}\) is recorded. Repeating this process for all simulated datasets yields a distribution of rank statistics, which should be uniform under well- calibrated inference. Deviations from uniformity signal systematic bias (e.g., over/under- dispersion) in the posterior approximation. We use the graphical approach proposed by Sailyhoja et al. (2022) to assess the uniformity of the rank statistics in SBC. This method provides visual diagnostics for identifying systematic biases or miscalibrations in the posterior approximation by plotting the empirical cumulative distribution function (ECDF) and confidence bands (95%). The pseudocode of SBC is provided in Algorithm 1. Examples of SBC checking results using this approach are provided in Appendix C.

## Algorithm 1 SBC diagnostic

Require: Joint model \(p(\theta ,y)\) ; amortized posterior \(q_{\phi}(\theta |y)\) ; scalar test function \(f\) ; number of simulated datasets \(J\) (e.g., 200); posterior draws \(S\) (e.g., 1000)

1: for \(j = 1,\ldots ,J\) do

2: Sample \(\theta_{\star}^{(j)}\sim p(\theta)\) , \(y^{(j)}\sim p(y|\theta_{\star}^{(j)})\)

3: Draw \(\theta_{s}^{(j)}\sim q_{\phi}(\theta |y^{(j)})\) for \(s = 1,\ldots ,S\)

4: Compute \(T_{\star}^{(j)} = f(\theta_{\star}^{(j)},y^{(j)})\) , \(T_{s}^{(j)} = f(\theta_{s}^{(j)},y^{(j)})\)

5: Compute rank \(r^{(j)} = \sum_{s = 1}^{S}\mathbb{I}[T_{s}^{(j)}< T_{\star}^{(j)}] + \mathrm{uniform}(0,\sum_{s = 1}^{S}\mathbb{I}[T_{s}^{(j)} = T_{\star}^{(j)})]\)

6: end for

7: Compare empirical ranks \(r^{(j)}\) against the uniform(0, \(S\) ) distribution using the graphical approach of Sailyhoja et al. (2022) to identify miscalibration patterns. Alternatively, the uniformity test based on the scalar statistic in Eq. 7 of Modrak et al. (2025) can also be applied.

Parameter recovery checking. Parameter recovery is a complementary diagnostic to SBC and provides a direct visualization of posterior approximation in recovering true generative parameters (Radev et al., 2020; 2023). The idea is to simulate a collection of datasets \(\{y^{(j)}\}\) along with their corresponding ground- truth parameters \(\{\theta_{*}^{(j)}\}\) from the joint model, and assess whether the posterior distributions \(q_{\phi}(\theta | y^{(j)})\) effectively recover these known values. In our workflow, we compare the posterior median extracted from each posterior to the corresponding ground- truth values, along with the median absolute deviation to indicate uncertainty. These comparisons are visualized using scatter plots, with correlation coefficients quantifying the strength of recovery. While not a direct measure of posterior calibration or correctness, parameter recovery provides important practical insight into whether the learned inverse mapping from \(y\) to \(\theta\) is effective. The pseudocode of the parameter recovery diagnostic is provided in Algorithm 2. Examples of parameter recovery checking results using this approach are provided in Appendix C.

# Algorithm 2 Parameter recovery diagnostic

Require: Joint model \(p(\theta ,y)\) ; amortized posterior \(q_{\phi}(\theta | y)\) ; number of datasets \(J\) ; posterior draws \(S\)

1: for \(j = 1,\ldots ,J\) do

2: Sample \(\theta_{*}^{(j)}\sim p(\theta)\) \(y^{(j)}\sim p(y|\theta_{*}^{(j)})\)

3: Draw \(\theta_{s}^{(j)}\sim q_{\phi}(\theta | y^{(j)})\) for \(s = 1,\ldots ,S\)

4: for each parameter component \(k\) do

5: Compute posterior median \(\hat{\theta}_{k}^{(j)} = \mathrm{median}_{s}[\theta_{s,k}^{(j)}]\)

6: Optionally compute a dispersion measure (e.g., median absolute deviation) for \(\theta_{s,k}^{(j)}\)

7: end for

8: end for

9: For each \(k\) , plot \(\hat{\theta}_{k}^{(j)}\) vs. \(\theta_{*,k}^{(j)}\) and report correlation

OOD diagnostic with Mahalanobis distance. The out- of- distribution (OOD) diagnostic used in Step 1 tests whether an observed dataset \(y_{\mathrm{obs}}\) falls outside the support of the prior predictive distribution (i.e., the training distribution for the amortized estimator). We work with low- dimensional summary statistics \(s(y) \in \mathbb{R}^{d}\) , which are either learned (e.g., via a summary network \(h_{\psi}\) ) or hand- crafted with domain knowledge. In the latter case, the amortized estimator \(q_{\phi}\) must be trained using these same hand- crafted statistics. The pseudocode for computing the Mahalanobis distance and checking whether an observed dataset \(y_{\mathrm{obs}}\) is OOD is provided in Algorithm 3.

# Algorithm 3 OOD diagnostic with Mahalanobis distance (Step 1)

Require: Training datasets \(\{y^{(m)}\}_{m = 1}^{M}\) (e.g., \(M = 10000\) ); summary statistic function \(s(\cdot)\) ; rejection level \(\alpha\) (e.g., \(\alpha = 0.05\) )

1: Compute \(s^{(m)} = s(y^{(m)})\) for all \(m\)

2: Compute empirical mean \(\mu_{s} = \frac{1}{M}\sum_{m = 1}^{M}s^{(m)}\) and covariance \(\Sigma_{s} = \frac{1}{M}\sum_{m = 1}^{M}(s^{(m)} - \mu_{s})(s^{(m)} - \mu_{s})^{\top}\)

3: For each \(m\) , compute Mahalanobis distance \(D_{M}(y^{(m)}) = \sqrt{(s^{(m)} - \mu_{s})^{\top}\Sigma_{s}^{- 1}(s^{(m)} - \mu_{s})}\)

4: Let \(T_{\alpha}\) be the empirical \((1 - \alpha)\) - quantile of \(\{D_{M}(y^{(m)})\}_{m = 1}^{M}\)

5: procedure TESTOOD( \(y_{\mathrm{obs}}\) )

6: Compute \(s_{\mathrm{obs}} = s(y_{\mathrm{obs}})\) and \(D_{M}(y_{\mathrm{obs}})\)

7: return \(\mathbb{I}_{D_{M}(y_{\mathrm{obs}})} > T_{\alpha}\)

8: end procedure

Pareto- smoothed importance sampling. Pareto- smoothed importance sampling (PSIS; Vehtari et al., 2024) is a robust method for improving the stability and reliability of importance sampling (IS) estimates. Given a target distribution \(p(y | \theta) p(\theta)\) and a proposal distribution \(q_{\phi}(\theta)\) , with samples \(\hat{\theta}_{s} \sim q_{\phi}(\theta)\) , PSIS

stabilizes the raw importance weights \(w_{s} = p(y|\hat{\theta}_{s})p(\hat{\theta}_{s}) / q_{\phi}(\hat{\theta}_{s}|y)\) by modeling the tail behavior of the importance weights. Specifically, the distribution of extreme importance weights can be approximated by a generalized Pareto distribution (GPD):

\[p(t|u,\sigma ,k) = \left\{ \begin{array}{l l}{\frac{1}{\sigma}\left(1 + k\left(\frac{t - u}{\sigma}\right)\right)^{-\frac{1}{k} -1},} & {k\neq 0}\\ {\frac{1}{\sigma}\exp \left(\frac{t - u}{\sigma}\right),} & {k = 0,} \end{array} \right. \quad (9)\]

where \(k\) is the shape parameter, \(u\) is the location parameter and \(\sigma\) is the scale parameter. The number of finite fractional moments of the importance weight distribution depends on \(k\) : a generalized Pareto distribution has \(1 / k\) finite moments when \(k > 0\) . To stabilize the importance sampling estimate, the extreme importance weights are replaced with well- spaced order statistics drawn from the fitted generalized Pareto distribution, leading to a more stable and efficient IS estimator. The estimated shape parameter \(\hat{k}\) serves as a diagnostic for the reliability of the importance sampling estimate. The pseudocode of PSIS is provided in Algorithm 4.

Algorithm 4 PSIS weights and Pareto- \(\hat{k}\) diagnostic (Step 2)

Require: Observed data \(y_{\mathrm{obs}}\) ; log- likelihood \(\log p(y|\theta)\) ; log- prior \(\log p(\theta)\) ; amortized posterior \(q_{\phi}(\theta |y)\) ; draws

\(\theta_{s}\sim q_{\phi}\) , \(s = 1,\ldots ,S\)

1: for \(s = 1,\ldots ,S\) do

2: \(\ell_{s} = \log p(y_{\mathrm{obs}}|\theta_{s}) + \log p(\theta_{s}) - \log q_{\phi}(\theta_{s}|y_{\mathrm{obs}})\)

3: end for

4: \(\hat{\ell}_{s} = \ell_{s} - \max_{s}\ell_{s}\)

5: \(w_{s} = \exp (\hat{\ell}_{s})\)

6: Sort \(w_{s}\) to obtain \(w(1)\leq \dots \leq w(S)\)

7: Choose tail size \(M = \lfloor \min (0.2S,3\sqrt{S})\rfloor\) and define tail \(w_{(S - M + 1)},\ldots ,w_{(S)}\)

8: Fit a GPD to the \(M\) largest importance weights \(w_{(S - M + 1)},\ldots ,w_{(S)}\) and obtain shape estimate \(\hat{k}\)

9: Replace the \(M\) largest weights with smoothed values from the fitted GPD to obtain stabilized weights \(\tilde{w}_{s}\)

10: Normalize: \(\bar{w}_{s} = \tilde{w}_{s} / \sum_{r = 1}^{S}\tilde{w}_{r}\)

11: Use \(\theta_{s},\bar{w}_{s}\) as weighted PSIS- corrected posterior draws; treat them as reliable if \(\hat{k}\leq \min (1 - 1 / \log_{10}(S),0.7)\)

Given the PSIS- stabilized weights \(\tilde{w}_{s}\) from Algorithm 4, one can either compute weighted Monte Carlo estimates directly (Vehtari et al., 2024) or apply the SIR procedure in Algorithm 5 to obtain approximately unweighted posterior draws for downstream use (e.g., visualization or MCMC initialization).

Algorithm 5 Sampling importance resampling (SIR) using PSIS weights

Require: PSIS- corrected weighted sample \(\{(\theta_{s},\tilde{w}_{s})\}_{s = 1}^{S}\) ; desired number of resampled draws \(S^{\prime}\) (e.g., \(S^{\prime} = S\)

1: Define a categorical distribution on indices \(s\in \{1,\ldots ,S\}\) with probabilities \(\tilde{w}_{1},\ldots ,\tilde{w}_{S}\)

2: for \(j = 1,\ldots ,S^{\prime}\) do

3: Sample index \(I_{j}\sim \mathrm{Categorical}(\tilde{w}_{1},\ldots ,\tilde{w}_{S})\) by typically with replacement; weighted sampling without replacement is useful for generating unique initializations for MCMC chains

4: Set \(\tilde{\theta}_{j} = \theta_{I_{j}}\)

5: end for

6: Return unweighted PSIS- corrected posterior draws \(\{\hat{\theta}_{j}\}_{j = 1}^{S^{\prime}}\)

ChEES- HMC algorithm. The ChEES- HMC algorithm (Hoffman et al., 2021) is a massively parallel and adaptive extension of Hamiltonian Monte Carlo (HMC) designed to leverage single- instruction multiple- data (SIMD) hardware accelerators such as GPUs. This enables rapid generation of posterior draws following an initial warm- up phase. During warm- up, ChEES- HMC adaptively tunes the trajectory length \(T\) and step size \(\epsilon\) by maximizing the "Change in the Estimator of the Expected Square" (ChEES), a heuristic that serves as a proxy for reducing autocorrelation in the second moments of the Markov chain. ChEES- HMC is particularly suitable for our amortized workflow, as we can easily generate a large number of good starting points (amortized draws) to launch many short MCMC chains. For our experiments, we used ChEES- HMC

to run 2048 parallel chains, organized into 16 superchains with 128 subchains each. This configuration is essential for computing the nested \(\widehat{R}\) diagnostic (Margossian et al., 2024), which assesses convergence across a large number of short chains. The pseudocode explaining the use of ChEES- HMC in Step 3 is provided in Algorithm 6.

## Algorithm 6 Use of ChEES-HMC in Step 3

Require: Log- posterior density \(\log p(\theta ,y_{\mathrm{obs}}) = \log p(y_{\mathrm{obs}}\mid \theta) + \log p(\theta)\) and its gradient w.r.t. \(\theta\) ; number of superchains \(K\) ; number of subchains per superchain \(M\) ; warm- up length \(N_{\mathrm{warmup}}\) (e.g., \(N_{\mathrm{warmup}} = 200\) ); after warm- up sampling length \(N\) (e.g., \(N = 1\) )

1: Collect \(K\) unique amortized posterior draws or PSIS- corrected draws for chain initialization; each of these \(K\) draws must have a finite log- posterior density value.

2: For each superchain group \(k = 1,\ldots ,K\) , initialize \(M\) subchains at the same initial state (total \(K\times M\) chains)

3: Run ChEES- HMC warm- up for \(N_{\mathrm{warmup}}\) iterations to adapt step size \(\epsilon\) and trajectory length \(L\)

4: Fix \((\epsilon ,L)\) and run \(N\) iterations to collect draws from \(KM\) parallel chains

5: Compute nested \(\widehat{R}\) for each parameter component of \(\theta\)

6: If nested \(\widehat{R}\) is below a chosen threshold (e.g. \(< 1.01\) ), accept the combined draws as Step 3 posterior draws; otherwise increase warmup length, use alternative MCMC algorithm (e.g., NUTS- HMC) or revise the model

Nested \(\widehat{R}\) diagnostic. The potential scale reduction factor \(\widehat{R}\) (Gelman & Rubin, 1992; Vehtari et al., 2021) is arguably the most popular diagnostic for assessing the convergence of MCMC chains. The basic idea is that multiple MCMC chains starting from overdispersed initial points should produce similar Monte Carlo estimators if they have converged, i.e., the impact of initialization vanishes as the chains converge to the stationary distribution. Nested \(\widehat{R}\) diagnostic (Margossian et al., 2024) extends the classical \(\widehat{R}\) diagnostic for monitoring convergence of many- short- chain MCMC samplers such as the ChEES- HMC algorithm. It requires organizing chains into \(K\) superchains, each consisting of \(M\) subchains that share the same initial point. Thus, one can assess convergence through comparing the variability between superchains and the variability within superchains, similar to the standard \(\widehat{R}\) diagnostic. We provide the pseudocode for computing the nested \(\widehat{R}\) diagnostic in Algorithm 7 for reference.

## Algorithm 7 Nested \(\widehat{R}\) diagnostic

Require: Posterior draws \(\{\theta^{(nmk)}\}\) after warm- up, where \(k\in \{1,\ldots ,K\}\) (superchains), \(m\in \{1,\ldots ,M\}\) (subchains), \(n\in \{1,\ldots ,N\}\) (draws); scalar function of interest \(f\)

1: Compute scalar values \(f^{(nmk)}\gets f(\theta^{(nmk)})\) for all \(n,m,k\)

2: Compute subchain means: \(\begin{array}{r}{\bar{f}^{(\cdot m k)}\gets \frac{1}{N}\sum_{n = 1}^{N}f^{(n m k)}} \end{array}\)

3: Compute superchain means \(\begin{array}{r}{\bar{f}^{(\cdot ,k)}\gets \frac{1}{M}\sum_{m = 1}^{M}\bar{f}^{(\cdot ,m k)}} \end{array}\) and overall mean \(\begin{array}{r}{\bar{f}^{(\cdot ,\cdot)}\gets \frac{1}{K}\sum_{k = 1}^{K}\bar{f}^{(\cdot ,k)}} \end{array}\)

4: for each superchain \(k = 1,\ldots ,K\) do

5: Compute between- chain variance \(\hat{B}_{k}\)

6: If \(M > 1\) , \(\begin{array}{r}{\hat{B}_{k}\gets \frac{1}{M - 1}\sum_{m = 1}^{M}(\bar{f}^{(\cdot ,m k)} - \bar{f}^{(\cdot ,k)})^{2}} \end{array}\) ; else \(\hat{B}_{k}\gets 0\)

7: Compute within- chain variance \(\hat{W}_{k}\)

8: If \(N > 1\) , \(\begin{array}{r}{\hat{W}_{k}\gets \frac{1}{M}\sum_{m = 1}^{M}\left(\frac{1}{N - 1}\sum_{n = 1}^{N}(f^{(n m k)} - \bar{f}^{(\cdot ,m k)})^{2}\right)} \end{array}\) ; else \(\hat{W}_{k}\gets 0\)

9: end for

10: Compute between- superchain variance: \(\begin{array}{r}{\widehat{B}_{\nu}\gets \frac{1}{K - 1}\sum_{k = 1}^{K}(\bar{f}^{(\cdot ,k)} - \bar{f}^{(\cdot ,\cdot)})^{2}} \end{array}\)

11: Compute within- superchain variance: \(\begin{array}{r}{\widehat{W}_{\nu}\gets \frac{1}{K}\sum_{k = 1}^{K}(\widehat{B}_{k} + \widehat{W}_{k})} \end{array}\)

12: Return nested \(\begin{array}{r}{\widehat{R}\gets \sqrt{\frac{\widehat{W}_{\nu} + \widehat{B}_{\nu}}{\widehat{W}_{\nu}}}} \end{array}\)

## B Best practices for training amortized estimators

Amortized inference approaches problems of Bayesian modeling with methods from deep learning. While the precise training setup naturally depends on the concrete problem at hand, some general principles have proven help across a wide range of amortized inference applications. We summarize these here as initial guidance for applied practitioners.

Rely on established tooling. Modern libraries for amortized inference such as sbi (Boelts et al., 2025) and BayesFlow (Radev et al., 2023) provide well- tested neural density estimators, training loops, and data pipelines. In many cases, their default architectures and optimization settings already yield strong performance without manual tuning. First and foremost, we strongly recommend starting from these defaults and only introducing additional complexity if the diagnostics indicate deficiencies.

Monitor the training process. In many amortized inference settings, data are generated on the fly by a forward simulation program (see Eq. 2), and training does not rely on a fixed dataset. In this case, classical data splits into training and validation set are less meaningful, since each minibatch effectively constitutes fresh data from the joint model. Nonetheless, it is still important to monitor training progress with multiple signals, such as the training loss, calibration diagnostics, or summary statistics of posterior samples. These checks help assess whether the model continues to improve or has already reached a performance plateau after a short period. In other settings, amortized inference may be trained on a fixed set of simulated data, for example due to an expensive simulator or precomputed datasets. In such cases, holding out a validation set is strongly recommended to detect overfitting and guide selection of the amortized estimator.

Track multiple performance signals. Regardless of whether data are simulated on the fly or fixed, we recommend monitoring multiple indicators during training (e.g., after each epoch). Loss curves provide a coarse signal of optimization progress but are often insufficient on their own. For example, normalizing flows are usually trained with a negative log- likelihood loss, which does not account for mode coverage in multi- modal posterior distributions. Complementary diagnostics such as simulation- based calibration, parameter recovery, or posterior predictive checks on held- out datasets offer more direct insight into posterior quality and failure modes.

Assess and adjust model expressiveness. When training stagnates or diagnostics indicate systematic errors, the limiting factor is often model expressiveness rather than optimization details. Underexpressive models may show symptoms such as poor parameter recovery, persistent miscalibration, or posterior collapse toward the prior (i.e., data insensitivity), even when the training loss decreases. A pragmatic strategy is to begin with an overly expressive architecture (i.e., many trainable weights) to establish a performance baseline, and then gradually simplify the model. Conversely, if diagnostics remain unsatisfactory, increasing model capacity is often more effective than tuning hyperparameters of the optimizer.

Accept that training is iterative. Even with modern tooling, amortized inference training may require many iterations during development, especially for complex or weakly identifiable models. The objective is not to find a universally optimal neural architecture, but to reach a regime where the amortized posterior is reliable enough to enter the adaptive workflow proposed in this paper, where subsequent diagnostics and correction steps can take over.

## C Experiment details

In this section, we provide additional experimental details omitted from the main text for brevity.

Evaluation metrics. To assess the quality of posterior approximations produced by each step of the workflow, we compare them against reference posterior draws obtained via a well- tuned No- U- Turn Sampler (NUTS). Specifically, we precomputed NUTS- based posterior samples for a subset of 5000 test datasets, which

serve as a ground- truth reference for evaluation. \(^{8}\) We then evaluate the 1- Wasserstein distance (W1) and the mean marginal total variation (MMTV) distance on up to 100 datasets from each inference step: Step 1 (amortized inference), Step 2 (amortized + PSIS), and Step 3 (ChEES- HMC with amortized initializations). These metrics are reported in the main text (Figure 5).

Neural network architecture for amortized inference. For all experiments, we use a coupling- based normalizing flow implemented in BayesFlow (Radev et al., 2023). The flow consists of 6 transformation layers, each comprising an invertible normalization, two affine coupling transformations, and a random permutation between elements. Before entering the coupling flow network as conditioning variables, the observed dataset \(y\) is encoded into a lower- dimensional summary statistic \(h_{\psi}(y)\) via a summary network \(h_{\psi}\) . This summary network is implemented either as a DeepSet architecture (Zaheer et al., 2017) or a SetTransformer (Lee et al., 2019), depending on the problem setting. Both architectures are designed to handle permutation- invariant data structures. For the Bernoulli GLM, we bypass the summary network and directly use the known 10- dimensional sufficient statistics (Lueckmann et al., 2021). The specific choice of summary network for each application is described in the respective problem descriptions below.

Training- phase optimization. For all problems, the neural network is optimized via the AdamW optimizer (Loshchilov & Hutter, 2019) with weight decay of \(10^{- 3}\) and a cosine decay learning rate schedule (initial learning rate of \(2.5 \times 10^{- 4}\) , a warmup target of \(5 \times 10^{- 4}\) , \(\alpha = 0\) ) as implemented in Keras (Chollet et al., 2015). A global gradient clip norm of 1.5 is applied. Training is performed with a batch size of 512 for 300 epochs, \(^{9}\) with cosine decay steps set to the product of batch size and epochs. A held- out validation set is used to monitor optimization and select the best- performing model checkpoint.

Space transformation. Following standard practice in Bayesian computation (e.g., PyMC; Oriol et al., 2023, Stan; Carpenter et al., 2017), we transform parameters to an unconstrained space for inference. The amortized neural estimator is trained to estimate parameters in this unconstrained space. PSIS operates independently of the parameterization and thus remains unaffected by this transformation. ChEES- HMC also performs inference in the unconstrained space. All evaluation metrics (W1 and MMTV distances) are computed in this space. However, parameter recovery and simulation- based calibration plots are shown in the original constrained space for better interpretability.

Computing infrastructure and software. For all applications, the full workflow—including amortized training, inference, diagnostics, Pareto- smoothed importance sampling, and ChEES- HMC sampling—was conducted on a single NVIDIA V100 GPU (32GB), 8 cores of an AMD EPYC 7452 processor, and 8- 16GB RAM. For runtime details across experiments, refer to Table 1 in the main text. The core code base was built using BayesFlow (Radev et al., 2023) (MIT license), PyMC (Oriol et al., 2023) (Apache- 2.0 license), ArviZ (Kumar et al., 2019) (Apache- 2.0 license) and JAX (Bradbury et al., 2018) (Apache- 2.0 license). We used the implementation of ChEES- HMC provided by TensorFlow Probability (Dillon et al., 2017) (Apache- 2.0 license).

Below, we provide details for each problem to complement the main text.

## C.1 Generalized extreme value distribution

Problem description. Following Caprani (2021), the prior distribution for the parameters of the generalized extreme value distribution (GEV) is defined as:

\[\begin{array}{r l} & {\mu \sim \mathcal{N}(3.8,0.04)}\\ & {\sigma \sim \mathrm{Half - Normal}(0,0.09)}\\ & {\xi \sim \mathrm{Truncated - Normal}(0,0.04)\mathrm{with~bounds}[-0.6,0.6].} \end{array} \quad (10)\]

> **Image description.** This image is a composite figure consisting of two main sections, (a) and (b), which display diagnostic plots for the Generalized Extreme Value (GEV) problem. Section (a) shows parameter recovery using scatter plots, while section (b) shows simulation-based calibration checking using line graphs.
>
> **Section (a): Parameter Recovery**
> This section contains three side-by-side scatter plots, each assessing the correlation between the estimated parameter and the ground truth value. All plots feature the "Ground truth" on the x-axis and the "Estimate" on the y-axis, with a diagonal line representing perfect correlation.
>
> 1.  **$\mu$ Plot:** This plot shows a strong positive correlation, indicated by the coefficient $r = 0.974$. The data points are tightly clustered around the diagonal line. The axes range from 3.0 to 4.4.
> 2.  **$\sigma$ Plot:** This plot shows the strongest correlation, with $r = 0.985$. The data points are extremely tightly clustered around the diagonal line. The axes range from 0.0 to 0.8.
> 3.  **$\xi$ Plot:** This plot shows a positive correlation, though the data points are more scattered than in the other two plots, with $r = 0.817$. The axes range from -0.6 to 0.6.
>
> **Section (b): Simulation-based Calibration Checking**
> This section contains three side-by-side line graphs, which assess the calibration of the model using the "Ecdf Difference" (Empirical Cumulative Distribution Function Difference) against the "Fractional rank statistic."
>
> 1.  **$\mu$ Plot:** The blue line, labeled "Rank Ecdf," hovers very close to the zero line (Ecdf Difference = 0), indicating good calibration. The plot includes a shaded area representing the "95% Confidence Bands," which are narrow and centered around zero. The axes range from 0.0 to 1.0 on both the x and y axes.
> 2.  **$\sigma$ Plot:** Similar to the $\mu$ plot, the blue line for "Rank Ecdf" remains very close to the zero line, suggesting good calibration. The 95% Confidence Bands are narrow and centered around zero. The axes range from 0.0 to 1.0.
> 3.  **$\xi$ Plot:** The blue line for "Rank Ecdf" shows more fluctuation compared to the other two plots, but it generally remains within the boundaries of the "95% Confidence Bands," which are centered around zero. The axes range from 0.0 to 1.0.
>
> In summary, the figure visually demonstrates that the parameter recovery is strong for $\mu$ and $\sigma$, and good for $\xi$, while the simulation-based calibration checking indicates acceptable convergence and good calibration for all three parameters.

<center>Figure 7: Training-phase diagnostics for the GEV problem. The parameter recovery is strong for the parameters \(\mu ,\sigma\) , and good for the shape parameter \(\xi\) . Simulation-based calibration checking indicates good calibration for all parameters. Parameter recovery and simulation-based calibration checking indicate acceptable convergence of the amortized posterior estimator. </center>

Simulation budgets. We use 10,000 simulated parameter- observation pairs for training the amortized estimator, 1000 for validation, and 200 for training- phase diagnostics, including parameter recovery and simulation- based calibration.

Summary network. We use a DeepSet as the summary network. The dimensionality of the learned summary statistics is 16. The DeepSet has a depth of 1, uses a \(mish\) activation, max inner pooling layers, 64 units in the equivariant and invariant modules, and \(5\%\) dropout.

Training- phase diagnostics. The closed- world diagnostics (parameter recovery and simulation- based calibration checking) in Figure 7 indicate that the neural network training has successfully converged to an acceptable posterior estimator within the scope of the training set.

Test datasets. In order to emulate distribution shifts that arise in real- world applications while preserving the controlled experimental environment, we simulate the "observed" datasets from a joint model whose prior is \(2\times\) wider (i.e., with \(4\times\) the variance) than the model used during training. More specifically, the prior is specified as:

\[\begin{array}{r l} & {\mu \sim \mathcal{N}(3.8,0.16)}\\ & {\sigma \sim \mathrm{Half - Normal}(0,0.36)}\\ & {\xi \sim \mathrm{Truncated - Normal}(0,0.16)\mathrm{~with~bounds~}[-1.2,1.2].} \end{array} \quad (11)\]

## C.2 Bernoulli GLM

Problem description. Following Lueckmann et al. (2021), we set the prior for \(\theta\) as:

\[\theta \sim \mathcal{N}\left(0,\left[2 \begin{array}{c}{0}\\ {(F^{\top}F)^{-1}} \end{array} \right]\right), \quad (12)\]

where the matrix \(F\) is defined such that \(F_{i,i - 2} = 1\) , \(F_{i,i - 1} = - 2\) , \(F_{i,i} = 1 + \sqrt{\frac{i - 1}{9}}\) , and \(F_{i,j} = 0\) otherwise, for \(1 \leq i, j \leq 9\) . The task duration is set to \(T = 100\) , with fixed input vectors \(\{v_{i}\}_{i = 1}^{100}\) , where each \(v_{i} \in \mathbb{R}^{10}\) . Corresponding observations are denoted by \(\{y_{i}\}_{i = 1}^{100}\) . Further details can be found in Lueckmann et al. (2021); Gonçalves et al. (2020).

Simulation budgets. We use 10,000 simulated parameter- observation pairs for training the amortized estimator, 1000 for validation, and 200 for training- phase diagnostics, including parameter recovery and simulation- based calibration.

Summary network. For Bernoulli GLM, the 10- dimensional sufficient summary statistic for each dataset can be computed as \(V y^{\top}\) where \(y = [y_{1},\dots ,y_{100}]\) and \(V = [v_{1},\dots ,v_{100}]\) . We therefore use this summary statistic for amortized training directly without relying on a separate summary neural network.

Training- phase diagnostics. The closed- world diagnostics (parameter recovery and simulation- based calibration checking) in Figure 8 indicate that the neural network training has successfully converged to an acceptable posterior estimator within the scope of the training set.

Test datasets. We generate \(K = 10,000\) in- distribution test datasets by sampling parameters from the model prior and simulating corresponding observations \(\{y_{i}\}_{i = 1}^{100}\) from the Bernoulli distribution.

## C.3 Psychometric curve fitting

Problem description. We adopt an overdispersed psychometric model (Schütt et al., 2016) with the error function (erf) as the sigmoid function in the psychometric function:

\[\psi (x;m,w,\lambda ,\gamma) = \gamma +(1 - \lambda -\gamma) \mathrm{erf}(x;m,w), \quad (13)\]

where \(m\) is the threshold, \(w\) is the width, \(\lambda\) is the lapse rate, and \(\gamma\) is the guess rate.

The full probabilistic model is defined as follows:

\[\begin{array}{r l} & {\tilde{m}\sim \mathrm{Beta}(2,2),}\\ & {w\sim \mathrm{Half - Normal}(0,1),}\\ & {\lambda ,\gamma ,\eta \sim \mathrm{Beta}(1,10),}\\ & {m = 2\tilde{m} -1,}\\ & {\tilde{p}_{i} = \psi (x_{i};m,w,\lambda ,\gamma),}\\ & {p_{i}\sim \mathrm{Beta}\left(\left(\frac{1}{\eta^{2}} -1\right)\bar{p}_{i},\left(\frac{1}{\eta^{2}} -1\right)(1 - \bar{p}_{i})\right),}\\ & {y_{i}\sim \mathrm{Binomial}(n_{i},p_{i}),} \end{array} \quad (14)\]

where \(n_{i}\) denotes the number of trials, and \(x_{i}\) is the stimulus level. Stimuli are presented at nine fixed levels: \(x_{i} \in \{- 100.0, - 25.0, - 12.5, - 6.25, 0.0, 6.25, 12.5, 25.0, 100.0\}\) and each value is further normalized by dividing by 100.

Simulation budgets. We use 50,000 simulated parameter- observation pairs for training the amortized estimator, 1000 for validation, and 200 for training- phase diagnostics, including parameter recovery and simulation- based calibration.

Summary network. We use a DeepSet as the summary network, which maps the input dataset to a 16- dimensional summary statistic. The DeepSet has a depth of 2, uses a gelu activation, mean inner pooling layers, 64 units in the equivariant and invariant modules, and 5% dropout.

Training- phase diagnostics. The closed- world diagnostics (parameter recovery and simulation- based calibration checking) in Figure 9 indicate that the neural network training has successfully converged to an acceptable posterior estimator within the scope of the training set.

> **Image description.** This image is a complex technical figure consisting of two main sections, (a) and (b), which display diagnostic plots for a Bernoulli Generalized Linear Model (GLM) problem. Both sections utilize a grid layout of ten individual scatter plots, labeled $\theta_1$ through $\theta_{10}$.
>
> **Section (a): Parameter recovery checking**
> This upper section, labeled "Parameter recovery checking," consists of ten scatter plots arranged in two rows of five. These plots assess how well the estimated parameters match the ground truth.
> *   **Axes:** The horizontal axis for all plots is labeled "Ground truth," and the vertical axis is labeled "Estimate."
> *   **Data Pattern:** In every subplot, the data points form a tight, strong linear relationship, clustering closely around a solid diagonal line, which represents perfect recovery ($y=x$).
> *   **Labels and Correlation:** Each plot is labeled with a Greek letter ($\theta_1$ to $\theta_{10}$) and a corresponding correlation coefficient ($r$), indicating the strength of the linear relationship. The visible $r$ values are:
>     *   $\theta_1$: $r=0.958$
>     *   $\theta_2$: $r=0.947$
>     *   $\theta_3$: $r=0.971$
>     *   $\theta_4$: $r=0.971$
>     *   $\theta_5$: $r=0.972$
>     *   $\theta_6$: $r=0.958$
>     *   $\theta_7$: $r=0.947$
>     *   $\theta_8$: $r=0.937$
>     *   $\theta_9$: $r=0.934$
>     *   $\theta_{10}$: $r=0.938$
>
> **Section (b): Simulation-based calibration checking**
> This lower section, labeled "Simulation-based calibration checking," also contains ten plots arranged in two rows of five. These plots evaluate the calibration of the model.
> *   **Axes:** The horizontal axis for all plots is labeled "Fractional statistic," and the vertical axis is labeled "Ecdf Difference."
> *   **Data Pattern:** Each plot displays a scatter of data points, generally centered around the zero line, indicating good calibration.
> *   **Visual Elements:** A shaded region, labeled "95% Confidence Bands," is visible in each plot, representing the statistical confidence interval for the Ecdf Difference. The data points consistently fall within or very close to these bands.
>
> Overall, the figure visually demonstrates strong parameter recovery (as evidenced by the tight linear fit in Section a) and good calibration (as evidenced by the clustering of data points around zero within the confidence bands in Section b) across all ten parameters ($\theta_1$ through $\theta_{10}$).

<center>Figure 8: Training-phase diagnostics for the Bernoulli GLM problem. The parameter recovery is strong for all parameters. Simulation-based calibration checking indicates good calibration for all parameters. Parameter recovery and simulation-based calibration checking indicate acceptable convergence of the amortized posterior estimator. </center>

Test datasets. Our empirical evaluation uses 8,526 mouse behavioral datasets from the International Brain Laboratory public database (The International Brain Laboratory et al., 2021). We retrieve the data using the provided API with the argument task="biasedChoiceWorld", which corresponds to behavioral data collected after the mice have completed training. Each dataset is processed into an observation tensor of shape (9,3), where each row contains the number of correct trials \(y_{i}\) , the total number of trials \(n_{i}\) , and the stimulus level \(x_{i}\) .

> **Image description.** A composite technical figure, labeled "Figure 9," presenting training-phase diagnostics for a psychometric curve fitting problem. The figure is divided into two main horizontal sections: (a) Parameter recovery and (b) Simulation-based calibration checking.
>
> **Figure Caption:**
> The caption reads: "Figure 9: Training-phase diagnostics for the psychometric curve fitting problem. Recovery is good for $\tilde{m}$ and $w$, while the other parameters exhibit weaker recoverability. Simulation-based calibration checking indicates excellent calibration for all parameters. Parameter recovery and simulation-based calibration checking indicate acceptable convergence of the amortized posterior estimator."
>
> **Section (a): Parameter recovery**
> This section consists of five scatter plots arranged in a single row. Each plot compares an "Estimate" (Y-axis) against the "Ground truth" (X-axis) for a specific parameter. The plots are labeled with the parameter symbol and a correlation coefficient ($r$).
> *   **$\tilde{m}$:** $r = 0.888$
> *   **$w$:** $r = 0.827$
> *   **$\gamma$:** $r = 0.482$
> *   **$\lambda$:** $r = 0.580$
> *   **$\eta$:** $r = 0.581$
> In all five plots, the data points generally cluster along the diagonal line (where Estimate equals Ground truth), indicating a strong positive correlation between the estimated and true values, with $\tilde{m}$ and $w$ showing the highest correlation.
>
> **Section (b): Simulation-based calibration checking**
> This section consists of five line graphs arranged in a single row, positioned directly beneath the corresponding plots in section (a). These plots assess calibration using the ECDF Difference.
> *   **Axes:** The Y-axis is labeled "ECDF Difference," and the X-axis is labeled "Fractional rank statistic."
> *   **Visual Elements:** Each plot displays a blue line representing the "Rank ECDF" and a shaded light blue area representing the "95% Confidence Bands."
> *   **Data Pattern:** In all five plots ($m, w, \gamma, \lambda, \eta$), the blue line remains close to the zero line, and the 95% Confidence Bands are relatively narrow, visually suggesting that the model is well-calibrated across the range of the fractional rank statistic.
>
> The overall arrangement allows for a direct comparison between the parameter recovery performance (scatter plots) and the calibration performance (line graphs) for each of the five parameters ($m, w, \gamma, \lambda, \eta$).

<center>Figure 9: Training-phase diagnostics for the psychometric curve fitting problem. Recovery is good for \(\tilde{m}\) and \(w\) , while the other parameters exhibit weaker recoverability. Simulation-based calibration checking indicates excellent calibration for all parameters. Parameter recovery and simulation-based calibration checking indicate acceptable convergence of the amortized posterior estimator. </center>

## C.4 Decision model

Problem description. Following von Krause et al. (2022), we specify the prior distributions for the drift- diffusion model parameters as:

\[\begin{array}{r l} & {v_{1},v_{2}\sim \mathrm{Gamma}(2,1),}\\ & {a_{1},a_{2}\sim \mathrm{Gamma}(6,0.15),}\\ & {\tau_{c}\sim \mathrm{Gamma}(3,0.15),}\\ & {\tau_{n}\sim \mathrm{Gamma}(3,0.5),} \end{array} \quad (15)\]

where all Gamma distributions use the shape- scale parametrization. \(^{10}\) We implement the drift- diffusion model likelihood using the hssm package (Fengler et al., 2025) and PyMC.

Simulation budgets. We use 100,000 simulated parameter- observation pairs for training the amortized estimator, 1000 for validation, and 200 for training- phase diagnostics, including parameter recovery and simulation- based calibration.

Summary network. We use a SetTransformer as the summary network, which maps the input dataset to a 16- dimensional summary statistic. The SetTransformer has two set attention blocks, followed by a pooling multi- head attention block and a fully connected output layer. Each multilayer perceptron (MLP) in the set blocks has two hidden layers of width 128, with gelu activation and 5% dropout.

Training- phase diagnostics. The closed- world diagnostics (parameter recovery and simulation- based calibration checking) in Figure 10 indicate that the neural network training has successfully converged to an acceptable posterior estimator within the scope of the training set.

Test datasets. The test datasets consist of 15,000 participants pre- processed from the online implicit association test (IAT) database (Xu et al., 2014; von Krause et al., 2022). Each test dataset is a tensor of

> **Image description.** This image is a composite figure, Figure 10, presenting training-phase diagnostics for a decision model, organized into two rows of six panels each. The figure assesses parameter recovery and simulation-based calibration for six different parameters: $v_1, v_2, a_1, a_2, \tau_c$, and $\tau_n$.
>
> The top row, labeled (a) "Parameter recovery," consists of six scatter plots. These plots compare the "Ground truth" (on the x-axis) against the "Estimate" (on the y-axis) for each parameter. In all six panels, the data points cluster tightly around the diagonal line of perfect agreement, indicating strong parameter recovery. The Pearson correlation coefficient ($r$) is displayed in the top left corner of each plot:
> *   $v_1$: $r = 0.952$
> *   $v_2$: $r = 0.973$
> *   $a_1$: $r = 0.953$
> *   $a_2$: $r = 0.956$
> *   $\tau_c$: $r = 0.997$
> *   $\tau_n$: $r = 0.938$
>
> The bottom row, labeled (b) "Simulation-based calibration checking," consists of six Ecdf (Empirical Cumulative Distribution Function) plots. These plots visualize the "Ecdf Difference" (on the y-axis) against the "Fractional statistic" (on the x-axis). Each plot includes a shaded area representing the "95% Confidence Bands." In all six panels, the Ecdf difference curve generally remains close to the zero line, and the curve is contained within the 95% Confidence Bands. This visual pattern suggests that the model exhibits acceptable calibration for all parameters.
>
> The overall layout is a grid of twelve plots (two rows of six columns), with consistent labeling and visual styles across all panels to facilitate comparison between the different parameters.

<center>Figure 10: Training-phase diagnostics for the decision model. Parameter recovery is strong for all parameters. Simulation-based calibration checking indicates good calibration for all parameters except \(\tau_{n}\) , which shows mild deviations, suggesting occasional overestimation by the amortized estimator for this parameter. Parameter recovery and simulation-based calibration checking indicate acceptable convergence of the amortized posterior estimator. </center>

shape (120, 4), where each row corresponds to a single trial and contains the response time, missing data mask, experiment condition type, and stimulus type.

## D Additional experimental study of the OOD diagnostic in Step 1

To further investigate the relationship between the Mahalanobis distance in the OOD diagnostic and the quality of the amortized posterior, we visualize this relationship using scatter plots in Figure 11a for the four tasks considered in the main text. For each task, we use around 1000 test datasets and compute the Pearson correlation coefficient \(r\) . The Mahalanobis distance is positively correlated with the two posterior quality metrics (W1 and MMTV) for the GEV, psychometric curve, and decision model tasks, where out- of- distribution test datasets are present. For the Bernoulli GLM, the correlation is negative; here, all test datasets were generated from the same distribution (prior simulations) as the training datasets and the Mahalanobis distance is not informative. From Figure 11a, we see a key limitation of the Step- 1 OOD diagnostic: the Mahalanobis distance is clearly not a perfect proxy for the posterior quality. In particular, the amortized estimator may still yield low- quality posterior draws on a dataset with a smaller Mahalanobis distance, as also observed in Figure 5.

We next check the impact of the threshold \(\alpha\) in the Step- 1 OOD diagnostic by varying it from 0.01 to 0.5, specifically over the set \([0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]\) , as shown in Figure 11b. As \(\alpha\) increases, more test datasets are rejected, and the overall posterior quality of accepted amortized posterior draws generally improves as measured by the median and IQR of the posterior metrics (lower posterior metric values indicate higher quality). The quality of rejected amortized posterior draws also improves as \(\alpha\) increases, while remaining consistently worse than that of the accepted amortized draws. \(^{11}\) Overall, the posterior quality of the accepted amortized draws, in terms of median and IQR of W1 and MMTV, is not very sensitive to the threshold \(\alpha\) , and \(\alpha = 0.05\) appears to be a reasonable default choice.

These results support the use of the Mahalanobis- distance- based OOD test as a lightweight first- line diagnostic in Step 1: it tends to flag the most problematic datasets and thereby improves the quality of accepted amortized posterior draws at negligible additional cost. At the same time, the residual low- quality posteriors at

> **Image description.** A composite figure consisting of two rows of four panels each, illustrating the relationship between posterior quality metrics (W1 and MMTV), Mahalanobis distance, and the effect of an OOD rejection threshold ($\alpha$) across four benchmark tasks: GEV, Bernoulli GLM, Psychometric curve, and Decision.
>
> The figure is divided into two main sections:
>
> **Top Row (Panel (a)): Scatter Plots of Posterior Quality Metrics vs. Mahalanobis Distance**
> This row contains eight scatter plots arranged in two columns (W1 and MMTV) and four rows (the four benchmark tasks). The X-axis for all plots represents the Mahalanobis distance.
>
> *   **W1 Metric (Top Row of Scatter Plots):**
>     *   **GEV:** Shows a positive correlation, with a reported Pearson correlation coefficient of $r = 0.47$.
>     *   **Bernoulli GLM:** Shows a negative correlation, with a reported Pearson correlation coefficient of $r = -0.46$.
>     *   **Psychometric curve:** Shows a positive correlation, with a reported Pearson correlation coefficient of $r = 0.37$.
>     *   **Decision:** Shows a positive correlation, with a reported Pearson correlation coefficient of $r = 0.42$.
> *   **MMTV Metric (Bottom Row of Scatter Plots):**
>     *   **GEV:** Shows a strong positive correlation, with a reported Pearson correlation coefficient of $r = 0.64$.
>     *   **Bernoulli GLM:** Shows a weak negative correlation, with a reported Pearson correlation coefficient of $r = -0.13$.
>     *   **Psychometric curve:** Shows a positive correlation, with a reported Pearson correlation coefficient of $r = 0.52$.
>     *   **Decision:** Shows a positive correlation, with a reported Pearson correlation coefficient of $r = 0.35$.
>
> **Bottom Row (Panel (b)): Effect of Rejection Threshold ($\alpha$) on Posterior Quality**
> This row contains four panels, each showing two sub-plots (W1 and MMTV) for a single benchmark task. The X-axis for all plots represents the Threshold $\alpha$, ranging from 0.01 to 0.5.
>
> *   **Legend:** A legend is present at the bottom center of the figure, indicating that the blue solid line represents "Accepted" data and the orange dashed line represents "Rejected" data.
> *   **GEV:** Both W1 and MMTV plots show a trend where the quality metric for "Accepted" data (blue) decreases as $\alpha$ increases, while the metric for "Rejected" data (orange) increases.
> *   **Bernoulli GLM:** Both W1 and MMTV plots show the opposite trend; the quality metric for "Accepted" data (blue) increases as $\alpha$ increases, while the metric for "Rejected" data (orange) decreases.
> *   **Psychometric curve:** Both W1 and MMTV plots show a trend where the quality metric for "Accepted" data (blue) decreases as $\alpha$ increases, while the metric for "Rejected" data (orange) increases.
> *   **Decision:** Both W1 and MMTV plots show a trend where the quality metric for "Accepted" data (blue) decreases as $\alpha$ increases, while the metric for "Rejected" data (orange) increases.

(b) Sensitivity of the rejection threshold \(\alpha\) in the OOD test. The median \(\pm\) IQR (shaded area) of the posterior quality metrics is shown separately for accepted and rejected datasets.

Figure 11: Relationship between amortized posterior quality metrics, Mahalanobis distance, and the OOD rejection threshold in Step 1. (a) Scatter plots of W1 and MMTV versus Mahalanobis distance, with Pearson correlation coefficient \(r\) reported in each panel. (b) Effect of varying the threshold \(\alpha\) on the posterior quality of accepted and rejected amortized posterior draws. See text in Appendix D for details.

small Mahalanobis distances underscore that this diagnostic cannot guarantee accuracy. For applications that require tighter accuracy guarantees, it is therefore natural to enforce escalation to Step 2 (PSIS) irrespective of the OOD outcome, trading additional computation for a more robust posterior approximation.

## E Amortized initialization for NUTS

In addition to ChEES- HMC, we evaluate the effectiveness of amortized posterior draws as initializations for the NUTS sampler. The experimental settings mirror those used for ChEES- HMC (Section 3.4), except that we launch only four chains, which is the typical configuration for NUTS. As shown in Figure 12, amortized initializations reduce the number of required warm- up iterations for both the GEV problem and the decision model. For the psychometric curve and Bernoulli GLM problems, all three initialization methods (amortized, PSIS- refined, and random) yield similar convergence behavior according to the \(\hat{R}\) diagnostic (Vehtari et al., 2021).

Notably, NUTS generally requires fewer warm- up iterations than ChEES- HMC across the evaluated problems, suggesting that while amortized initializations are still beneficial, the relative gain is more pronounced for ChEES- HMC, which runs many short chains in parallel.

> **Image description.** A multi-panel line graph, arranged in a 2x2 grid, illustrating the effect of different initialization methods on the $\hat{R}$ diagnostic over a range of "Warmup iterations" for four distinct statistical models.
>
> The overall figure shares common axes across all four panels. The Y-axis, labeled $\hat{R} - 1$, uses a logarithmic scale ranging from $10^{-3}$ to $10^{-1}$. The X-axis, labeled "Warmup iterations," ranges from 10 to 500, with major tick marks at intervals of 100.
>
> A legend located in the top right corner identifies three initialization methods, each represented by a distinct color and marker:
> *   Amortized: Blue line with triangle markers.
> *   Amortized + PSIS: Yellow line with diamond markers.
> *   Random Init.: Red line with square markers.
>
> The four panels are titled: GEV (top left), Psychometric curve (bottom left), Bernoulli GLM (top right), and Decision model (bottom right). Each data point is accompanied by vertical error bars, representing the median $\pm$ IQR across 20 test datasets.
>
> **Panel-Specific Observations:**
>
> 1.  **GEV (Top Left):** This panel shows the most pronounced difference between initialization methods. The blue (Amortized) and yellow (Amortized + PSIS) lines exhibit a rapid initial decrease in $\hat{R}$ within the first 100 iterations, indicating fast convergence. The red (Random Init.) line decreases much more slowly and remains significantly higher than the other two methods for a longer duration.
> 2.  **Psychometric curve (Bottom Left):** In contrast to the GEV panel, the three lines are tightly clustered and show very similar convergence behavior. All three methods start near $10^{-2}$ and stabilize quickly, with minimal visual distinction between the Amortized, Amortized + PSIS, and Random Init. methods.
> 3.  **Bernoulli GLM (Top Right):** Similar to the Psychometric curve, the lines for all three initialization methods are closely grouped. They show a slight initial drop and quickly stabilize around the $10^{-2}$ mark, suggesting comparable performance across the methods.
> 4.  **Decision model (Bottom Right):** This panel shows a trend similar to the GEV panel. The blue and yellow lines drop more steeply and quickly than the red line, indicating that the Amortized and Amortized + PSIS initializations lead to faster convergence in this model compared to Random Init.
>
> In summary, the visual data suggests that while Amortized initializations provide a clear benefit in the GEV and Decision model tasks by reducing the required warmup iterations, the performance of the three initialization methods is nearly indistinguishable for the Psychometric curve and Bernoulli GLM problems.

<center>Figure 12: The effect of initialization for NUTS. The figure shows median±IQR across 20 test datasets. Using amortized posterior draws as initializations for NUTS reduces the required warmup in the GEV and decision model tasks. </center>

---

*Transcribed with OCR and VLMs; text, equations, and figure descriptions may contain mistakes.*
