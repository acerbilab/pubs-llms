# Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations - Appendix

---

#### Page 18

# Appendix A. 

This appendix provides additional details and analyses to complement the main text, included in the following sections:

- Normalizing flow regression algorithm details, A. 1
- Metrics description, A. 2
- Real-world problems description, A. 3
- Additional experiment results, A. 4
- Black-box variational inference implementation, A. 5
- Limitations, A. 6
- Ablation study, A. 7
- Diagnostics, A. 8
- Visualization of posteriors, A. 9

## A.1. Normalizing flow regression algorithm details

Inference space. NFR, VSBQ, Laplace approximation, and BBVI all operate in an unbounded parameter space, which we call the inference space. Originally bounded parameters are first mapped to the inference space and then rescaled and shifted based on user-specified plausible ranges, such as the $68.2 \%$ percentile interval of the prior. After transformation, the plausible ranges in the inference space are standardized to $[-0.5,0.5]$. An appropriate Jacobian correction is applied to the log-density values in the inference space. Similar transformations are commonly used in probabilistic inference software (Carpenter et al., 2017; Huggins et al., 2023). The approximate posterior samples are transformed back to the original space via the inverse transform for performance evaluation against the ground truth posterior samples.

Noise shaping hyperparameter choice for NFR. The function $s(\cdot)$ in Eq. 8 acts as a noise shaping mechanism that increases observation uncertainty for lower-density regions, further preventing overfitting to low-density observations (Li et al., 2024). We define $s(\cdot)$ as a piecewise linear function,

$$
s\left(f_{\max }-f_{n}\right)= \begin{cases}0 & \text { if } f_{\max }-f_{n}<\delta_{1} \\ \lambda\left(f_{\max }-f_{n}-\delta_{1}\right) & \text { if } \delta_{1} \leq f_{\max }-f_{n} \leq \delta_{2} \\ \lambda\left(\delta_{2}-\delta_{1}\right) & \text { if } f_{\max }-f_{n}>\delta_{2}\end{cases}
$$

Here, $\delta_{1}$ and $\delta_{2}$ define the thresholds for moderate and extremely low log-density values, respectively. In practice, we approximate the unknown difference $f_{\max }-f_{n}$ with $y_{\max }-y_{n}$, where $y_{\max }=\max _{n} y_{n}$ is the maximum observed log-density value. We set $y_{\text {low }}=\max _{n}\left(y_{n}-\right.$ $1.96 \sigma_{n})-\delta_{2}$ for Eq. 8. For all problems, we set $\lambda=0.05$ following Li et al. (2024). The thresholds for moderate density and extremely low density are defined as $\delta_{1}=10 D$,

---

#### Page 19

$\delta_{2}=50 D$, where $D$ is the target posterior dimension. ${ }^{6}$ The extremely low-density value is computed as $y_{\text {low }}=\max _{n}\left(y_{n}-1.96 \sigma_{n}\right)-\delta_{2}$.

Normalizing flow architecture specifications. For all experiments, we use the masked autoregressive flow (MAF; Papamakarios et al., 2017) with the original implementation from Durkan et al. (2020). The flow consists of 11 transformation layers, each comprising an affine autoregressive transform followed by a reverse permutation transform. As described in Section 3.3, the flow's base distribution is a diagonal multivariate Gaussian estimated from observations with sufficiently high log-density values. Specifically, we select observations satisfying $y_{n}-1.96 \sigma_{n} \geq \delta_{1}$ and compute the mean and covariance directly from these selected points $\mathbf{x}_{n}$. The maximum scaling factor $\alpha_{\max }$ and $\mu_{\max }$ are chosen such that the normalizing flow exhibits controlled flexibility from the base distribution, as illustrated in Section 4.1. We set $\alpha_{\max }=1.5$ and $\mu_{\max }=1$ (Eq. 9) across the experiments.

Initialization of regression model parameters. The parameter set for the normalizing flow regression model is $\boldsymbol{\psi}=(\boldsymbol{\phi}, C)$, where $\boldsymbol{\phi}$ represents the flow parameters, i.e., the parameters of the neural networks. We initialize $\boldsymbol{\phi}$ by multiplying the default PyTorch initialization (Paszke et al., 2019) by $10^{-3}$ to ensure the flow starts close to its base distribution. The parameter $C$ is initialized to zero.

Termination criteria for normalizing flow regression. For all problems, we set the number of annealed steps $t_{\text {end }}=20$ and the maximum number of training iterations $T_{\max }=$ 30. At each training iteration, the L-BFGS optimizer is run with a maximum of 500 iterations and up to 2000 function evaluations. The L-BFGS optimization terminates if the directional derivative falls below a threshold of $10^{-5}$ or if the maximum absolute change in the loss function over five consecutive iterations is less than $10^{-5}$.

Training dataset. For each benchmark problem, MAP estimation is performed to find the target posterior mode. We launch MAP optimization runs from random initial points and collect multiple optimization traces as the training dataset for NFR and VSBQ. The total number of target density evaluations is fixed to $3000 D$. It is worth noting that the MAP estimate depends on the choice of parameterization. We align with the practical usage scenario where optimization is performed in the original parameter space and the parameter bounds are dealt with by the optimizers (in our case, CMA-ES and BADS).

# A.2. Metrics description 

Following Acerbi (2020); Li et al. (2024), we use three metrics: the absolute difference $\Delta$ LML between the true and estimated log normalizing constant (log marginal likelihood); the mean marginal total variation distance (MMTV); and the "Gaussianized" symmetrized KL divergence (GsKL) between the approximate and true posterior. For each problem, ground-truth posterior samples are obtained through rejection sampling, extensive MCMC, or analytical/numerical methods. The ground-truth log normalizing constant is computed

[^0]
[^0]:    6. El Gammal et al. (2023); Li et al. (2024) set the low-density thresholds by referring to the log-density range of a standard $D$-dimensional multivariate Gaussian distribution, which requires computing an inverse CDF of a chi-squared distribution. However, this computation for determining the extremely low-density threshold can numerically overflow to $\infty$. Therefore, we use a linear approximation in $D$, similar to Huggins et al. (2023).

---

#### Page 20

analytically, using numerical quadrature methods, or estimated from posterior samples via Geyer's reverse logistic regression (Geyer, 1994). For completeness, we describe below the metrics and desired thresholds in detail, largely following Li et al. (2024):

- $\Delta$ LML measures the absolute difference between true and estimated log marginal likelihood. We aim for an LML loss $<1$, as differences in log model evidence $\ll 1$ are considered negligible for model selection (Burnham and Anderson, 2003).
- The MMTV quantifies the (lack of) overlap between true and approximate posterior marginals, defined as

$$
\operatorname{MMTV}(p, q)=\sum_{d=1}^{D} \int_{-\infty}^{\infty} \frac{\left|p_{d}^{\mathrm{M}}\left(x_{d}\right)-q_{d}^{\mathrm{M}}\left(x_{d}\right)\right|}{2 D} d x_{d}
$$

where $p_{d}^{\mathrm{M}}$ and $q_{d}^{\mathrm{M}}$ denote the marginal densities of $p$ and $q$ along the $d$-th dimension. An MMTV metric of 0.2 indicates that, on average across dimensions, the posterior marginals have an $80 \%$ overlap. As a rule of thumb, we consider this level of overlap (MMTV $<0.2$ ) as the threshold for a reasonable posterior approximation.

- The (averaged) GsKL metric evaluates differences in means and covariances:

$$
\operatorname{GsKL}(p, q)=\frac{1}{2 D}\left[D_{\mathrm{KL}}\left(\mathcal{N}[p] \|\mathcal{N}[q]\right)+D_{\mathrm{KL}}\left(\mathcal{N}[q] \|\mathcal{N}[p]\right)\right]
$$

where $D_{\mathrm{KL}}(p \| q)$ is the Kullback-Leibler divergence between distributions $p$ and $q$ and $\mathcal{N}[p]$ denotes a multivariate Gaussian with the same mean and covariance as $p$ (similarly for $q$ ). This metric has a closed-form expression in terms of means and covariance matrices. For reference, two Gaussians with unit variance whose means differ by $\sqrt{2}$ (resp. $\frac{1}{2}$ ) yield GsKL values of 1 (resp. $\frac{1}{8}$ ). As a rule of thumb, we consider $\mathrm{GsKL}<\frac{1}{8}$ to indicate a sufficiently accurate posterior approximation.

# A.3. Real-world problems description 

Bayesian timing model $(D=5)$. We analyze data from a sensorimotor timing experiment in which participants were asked to reproduce time intervals $\tau$ between a mouse click and screen flash, with $\tau \sim$ Uniform[0.6, 0.975] s (Acerbi et al., 2012). The model assumes participants receive noisy sensory measurements $t_{\mathrm{s}} \sim \mathcal{N}\left(\tau, w_{\mathrm{s}}^{2} \tau^{2}\right)$ and they generate an estimate $\tau_{\star}$ by combining this sensory evidence with a Gaussian prior $\mathcal{N}\left(\tau ; \mu_{\mathrm{p}}, \sigma_{\mathrm{p}}^{2}\right)$ and taking the posterior mean. Their reproduced times then include motor noise, $t_{\mathrm{m}} \sim \mathcal{N}\left(\tau_{\star}, w_{\mathrm{m}}^{2} \tau_{\star}^{2}\right)$, and each trial has probability $\lambda$ of a "lapse" (e.g., misclick) yielding instead $t_{\mathrm{m}} \sim$ Uniform[0, 2] s. The model has five parameters $\boldsymbol{\theta}=\left(w_{\mathrm{s}}, w_{\mathrm{m}}, \mu_{\mathrm{p}}, \sigma_{\mathrm{p}}, \lambda\right)$, where $w_{\mathrm{s}}$ and $w_{\mathrm{m}}$ are Weber fractions quantifying perceptual and motor variability. We adopt a spline-trapezoidal prior for all parameters. The spline-trapezoidal prior is uniform between the plausible ranges of the parameter while falling smoothly as a cubic spline to zero toward the parameter bounds. We infer the posterior for a representative participant from Acerbi et al. (2012). As explained in the main text, we make the inference scenario more challenging and realistic by including log-likelihood estimation noise with $\sigma_{n}=3$. This noise magnitude is analogous to what practitioners would find by estimating the log-likelihood via Monte Carlo instead of using numerical integration methods (van Opheusden et al., 2020).

---

#### Page 21

Lotka-Volterra model $(D=8)$. The model describes population dynamics through coupled differential equations:

$$
\frac{\mathrm{d} u}{\mathrm{~d} t}=\alpha u-\beta u v ; \quad \frac{\mathrm{d} v}{\mathrm{~d} t}=-\gamma v+\delta u v
$$

where $u(t)$ and $v(t)$ represent prey and predator populations at time $t$, respectively. Using data from Howard (2009), we infer eight parameters: four rate constants $(\alpha, \beta, \gamma, \delta)$, initial conditions $(u(0), v(0))$, and observation noise intensities $\left(\sigma_{u}, \sigma_{v}\right)$. The likelihood is computed by solving the equations numerically using the Runge-Kutta method. See Carpenter (2018) for further details of priors and model implementations.

Bayesian causal inference in multisensory perception $(D=12)$. In the experiment, participants seated in a moving chair judged whether the direction of their motion $s_{\text {vest }}$ matched that of a visual stimulus $s_{\text {vis }}$ ('same' or 'different'). The model assumes participants receive noisy measurements $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right)$ and $z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}$ is vestibular noise and $\sigma_{\text {vis }}(c)$ represents visual noise under three different coherence levels $c$. Each sensory noise parameter includes both a base standard deviation and a Weber fraction scaling factor. The Bayesian causal inference observer model also incorporates a Gaussian spatial prior, probability of common cause, and lapse rate for random responses, totaling 12 parameters. The model's likelihood is mildly expensive ( $\sim 3 \mathrm{~s}$ per evaluation), due to numerical integration used to compute the observer's posterior over causes, which would determine their response in each trial ('same' or 'different'). We adopt a spline-trapezoidal prior for all parameters, which remains uniform within the plausible parameter range and falls smoothly to zero near the bounds using a cubic spline. We fit the data of representative subject S11 from Acerbi et al. (2018).

# A.4. Additional experiment results 

Lumpy distribution $(D=10)$. Table A. 1 presents the results for the ten-dimensional lumpy distribution, omitted from the main text due to space constraints. All methods, except Laplace, achieve metrics below the target thresholds, with NFR performing best. While the Laplace approximation provides reasonable estimates of the normalizing constant and marginal distributions, it struggles with the full joint distribution.

Table A.1: Lumpy $(D=10)$.

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :-- | :--: | :--: | :--: |
| Laplace | 0.81 | 0.15 | 0.22 |
| BBVI $(1 \times)$ | $0.42[0.40,0.51]$ | $0.065[0.061,0.079]$ | $0.029[0.023,0.035]$ |
| BBVI $(10 \times)$ | $0.32[0.28,0.41]$ | $0.046[0.041,0.051]$ | $0.013[0.0095,0.015]$ |
| VSBQ | $0.11[0.097,0.15]$ | $0.033[0.031,0.038]$ | $0.0070[0.0066,0.0090]$ |
| NFR | $\mathbf{0 . 0 2 6}[0.016,0.040]$ | $\mathbf{0 . 0 2 2}[0.022,0.024]$ | $\mathbf{0 . 0 0 2 0}[0.0018,0.0023]$ |

---

#### Page 22

Results from MAP runs with BADS optimizer. We present here the results of applying NFR and VSBQ to the MAP optimization traces from the BADS optimizer (Acerbi and Ma, 2017), instead of CMA-ES used in the main text. BADS is an efficient hybrid Bayesian optimization method that also deals with noisy observations like CMA-ES. The results for the other baselines (BBVI, Laplace) are the same as those reported in the main text, since these methods do not reuse existing (offline) optimization traces, but we repeat them here for ease of comparison.

The full results are shown in Table A.2, A.3, A.4, A.5, and A.6. From the tables, we can see that NFR still achieves the best performance for all problems. For the challenging 12D multisensory problem, the metrics $\Delta$ LML and GsKL slightly exceed the desired thresholds. Additionally, as shown by comparing Table 4 in the main text and Table A.6, NFR performs slightly worse when using evaluations from BADS, compared to CMA-ES. We hypothesize that this is because BADS converges rapidly to the posterior mode, resulting in less evaluation coverage on the posterior log-density function, as also noted by Li et al. (2024). In sum, our results about the accuracy of NFR qualitatively hold regardless of the optimizer.

Table A.2: Multivariate Rosenbrock-Gaussian $(D=6)$. (BADS)

|  | $\Delta$ LML $(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :-- | :--: | :--: | :--: |
| Laplace | 1.3 | 0.24 | 0.91 |
| BBVI $(1 \times)$ | $1.3[1.2,1.4]$ | $0.23[0.22,0.24]$ | $0.54[0.52,0.56]$ |
| BBVI $(10 \times)$ | $1.0[0.72,1.2]$ | $0.24[0.19,0.25]$ | $0.46[0.34,0.59]$ |
| VSBQ | $0.19[0.19,0.20]$ | $0.038[0.037,0.039]$ | $0.018[0.017,0.018]$ |
| NFR | $\mathbf{0 . 0 0 6 7}[0.0031,0.012]$ | $\mathbf{0 . 0 2 8}[0.026,0.031]$ | $\mathbf{0 . 0 0 5 3}[0.0032,0.0060]$ |

Table A.3: Lumpy. (BADS)

|  | $\Delta$ LML $(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :-- | :--: | :--: | :--: |
| Laplace | 0.81 | 0.15 | 0.22 |
| BBVI $(1 \times)$ | $0.42[0.40,0.51]$ | $0.065[0.061,0.079]$ | $0.029[0.023,0.035]$ |
| BBVI $(10 \times)$ | $0.32[0.28,0.41]$ | $0.046[0.041,0.051]$ | $0.013[0.0095,0.015]$ |
| VSBQ | $\mathbf{0 . 0 2 9}[0.0099,0.043]$ | $0.034[0.033,0.037]$ | $0.0065[0.0060,0.0073]$ |
| NFR | $0.072[0.057,0.087]$ | $\mathbf{0 . 0 2 9}[0.028,0.031]$ | $\mathbf{0 . 0 0 2 1}[0.0017,0.0026]$ |

# A.5. Black-box variational inference implementation 

Normalizing flow architecture specifications and initialization. For BBVI, we use the same normalizing flow architecture as in NFR. The base distribution of the normalizing flow is set to a learnable diagonal multivariate Gaussian, unlike in NFR where the means

---

#### Page 23

Table A.4: Bayesian timing model. (BADS)

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :--: | :--: | :--: | :--: |
| BBVI $(1 \times)$ | $1.6[1.1,2.5]$ | $0.29[0.27,0.34]$ | $0.77[0.67,1.0]$ |
| BBVI $(10 \times)$ | $\mathbf{0 . 3 2}[0.036,0.66]$ | $0.11[0.088,0.15]$ | $0.13[0.052,0.23]$ |
| VSBQ | $\mathbf{0 . 2 2}[0.18,0.42]$ | $\mathbf{0 . 0 5 7}[0.045,0.074]$ | $\mathbf{0 . 0 1 0}[0.0070,0.14]$ |
| NFR | $\mathbf{0 . 2 4}[0.21,0.27]$ | $\mathbf{0 . 0 6 0}[0.052,0.076]$ | $\mathbf{0 . 0 1 4}[0.0088,0.017]$ |

Table A.5: Lotka-volterra model. (BADS)

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :--: | :--: | :--: | :--: |
| Laplace | 0.62 | 0.11 | 0.14 |
| BBVI $(1 \times)$ | $0.47[0.42,0.59]$ | $0.055[0.048,0.063]$ | $0.029[0.025,0.034]$ |
| BBVI $(10 \times)$ | $0.24[0.23,0.36]$ | $0.029[0.025,0.039]$ | $0.0087[0.0052,0.014]$ |
| VSBQ | $1.0[1.0,1.0]$ | $0.084[0.081,0.087]$ | $0.063[0.061,0.064]$ |
| NFR | $\mathbf{0 . 1 8}[0.17,0.18]$ | $\mathbf{0 . 0 1 5}[0.014,0.016]$ | $\mathbf{0 . 0 0 0 7 4}[0.00057,0.00092]$ |

Table A.6: Multisensory. (BADS)

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :--: | :--: | :--: | :--: |
| VSBQ | $1.5 \mathrm{e}+3[6.2 \mathrm{e}+2,2.1 \mathrm{e}+3]$ | $0.87[0.81,0.90]$ | $1.2 \mathrm{e}+4[2.0 \mathrm{e}+2,1.4 \mathrm{e}+8]$ |
| NFR | $\mathbf{1 . 1}[0.95,1.3]$ | $\mathbf{0 . 1 5}[0.13,0.19]$ | $\mathbf{0 . 2 2}[0.15,0.94]$ |

and variances can be estimated from the MAP optimization runs. The base distribution is initialized as a multivariate Gaussian with mean zero and standard deviations set to onetenth of the plausible ranges. The transformation layers, parameterized by neural networks, are initialized using the same procedure as in NFR (see Appendix A.1).

Stochastic optimization. As described in the main text, BBVI is performed by optimizing the ELBO using the Adam optimizer. To give BBVI the best chance of performing well, for each problem we conducted a grid search over the learning rate $\{0.01,0.001\}$ and the number of Monte Carlo samples for gradient estimation $\{1,10,100\}$, selecting the bestperforming configuration based on the estimated ELBO value and reporting the performance metrics accordingly. Following Li et al. (2024), we further apply a control variate technique to reduce the variance of the ELBO gradient estimator.

---

#### Page 24

# A.6. Limitations 

In this work, we leverage normalizing flows as a regression surrogate to approximate the log-density function of a probability distribution. This methodology inherits the limitations of surrogate modeling approaches. Regardless of the source, the training dataset needs to sufficiently cover regions of non-negligible probability mass. In high-dimensional settings, this implies that the required number of training points grows exponentially, eventually becoming impractical (Li et al., 2024). In practice, similarly to other surrogate-based methods, we expect our method to be applicable to models with up to 10-15 parameters, as demonstrated by the 12-dimensional example in the main text. Scalability beyond $D \approx 20$ remains to be investigated.

In the paper, we focus on obtaining training data from MAP optimization traces. In this case, care must be taken to ensure the MAP estimate does not fall exactly on parameter bounds; otherwise, transformations into inference space (Appendix A.1) could push logdensity observations to infinity, rendering them uninformative for constructing the normalizing flow surrogate. This issue is an old and well-known problem in approximate Bayesian inference (e.g., for the Laplace approximation, MacKay, 1998) and can be mitigated by imposing priors that vanish at the bounds (Gelman et al., 2013, Chapter 13), such as the spline-trapezoidal prior as in Appendix A.3). Additionally, fitting a regression model to pointwise log-density observations may become less meaningful in certain scenarios, e.g., when the likelihood is unbounded or highly non-smooth.

Our proposed technique, normalizing flow regression, jointly estimates both the flow parameters and the normalizing constant. The latter is a notoriously challenging quantity to infer even when distribution samples are available (Geyer, 1994; Gutmann and Hyvärinen, 2010; Gronau et al., 2017). We impose a prior over the flow's neural network parameters for mitigating the non-identifiability issue (Section 3.3) and further apply an annealed optimization technique (Section 3.4), which we empirically find improves the posterior approximation and normalizing constant estimation (Section 4, Appendix A.7). However, these are not silver bullets, and we recommend performing diagnostic checks on the flow approximation (Appendix A.8) whenever possible.

## A.7. Ablation study

To validate our key design choices, we conducted ablation studies examining three components of NFR: the likelihood function (Section 3.2), flow priors (Section 3.3), and annealed optimization (Section 3.4). We tested these using two problems from our benchmark: the Bayesian timing model $(D=5)$ and the challenging multisensory perception model $(D=12)$. As shown in Table A.7, our proposed combination of Tobit likelihood, flow prior settings, and annealed optimization achieves the best overall performance. The progression of results reveals several insights.

First, noise shaping in the regression likelihood proves crucial. The basic Gaussian observation noise without noise shaping, as defined in Eq. 7, yields poor approximations of the true target posterior. Adding noise shaping to the regression likelihood significantly improves performance. Switching then to our Tobit likelihood (Eq. 8) provides marginally further benefits. Indeed, the Gaussian likelihood with noise shaping is a special case of the Tobit likelihood where the low-density threshold $y_{\text {low }}$ approaches negative infinity.

---

#### Page 25

Second, the importance of annealing depends on problem complexity. While the lowdimensional timing model performs adequately without annealing, the 12-dimensional multisensory model requires it for stable optimization. This suggests annealing becomes crucial as dimensionality increases.

Finally, flow priors prove essential for numerical stability and performance. Without them, many optimization runs fail due to numerical errors (marked with asterisks in Table A.7), and even successful runs show substantially degraded performance.

Table A.7: Ablation experiments. The abbreviation 'ns' refers to noise shaping (Eq. A.1). Results marked with $*$ indicate that multiple runs failed due to numerical errors.

| Ablation settings |  |  |  | Bayesian timing model $(D=5)$ |  |  | Multisensory $(D=12)$ |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| likelihood | with flow priors | annealing |  | $\Delta$ LML | MMTV | GsKL | $\Delta$ LML | MMTV | GsKL |
| Gaussian w/o ns | $\checkmark$ | $\checkmark$ |  | 0.16 | 0.21 | 0.42 | 4.0 | 0.44 | 5.9 |
| Gaussian w/ ns | $\checkmark$ | $\checkmark$ |  | $[0.089,0.29]$ | $[0.18,0.30]$ | $[0.24,0.83]$ | $[1.9,7.1]$ | $[0.40,0.51]$ | $[3.4,9.3]$ |
| Gaussian w/ ns | $\checkmark$ | $\checkmark$ |  | 0.20 | 0.055 | 0.0096 | 0.87 | 0.13 | 0.12 |
|  |  |  |  | $[0.18,0.23]$ | $[0.043,0.059]$ | $[0.0074,0.013]$ | $[0.69,1.0]$ | $[0.11,0.15]$ | $[0.086,0.17]$ |
| Tobit | $\checkmark$ | $\boldsymbol{x}$ |  | 0.20 | 0.048 | 0.0098 | 24. | 0.82 | $2.8 \mathrm{e}+2$ |
|  |  |  |  | $[0.17,0.23]$ | $[0.044,0.052]$ | $[0.0062,0.011]$ | $[18 . .42]$ | $[0.76,0.84]$ | $[62 . .90 e+3]$ |
| Tobit | $\boldsymbol{x}$ | $\checkmark$ |  | $6.7^{*}$ | $0.99^{*}$ | $2.6 \mathrm{e}+3^{*}$ | $0.86^{*}$ | $0.14^{*}$ | $0.25^{*}$ |
|  |  |  |  | $[0.0,7.9]$ | $[0.99,1.0]$ | $[1.6 \mathrm{e}+3,4.6 \mathrm{e}+3]$ | $[0.73,0.96]$ | $[0.13,0.17]$ | $[0.14,3.6]$ |
| Tobit | $\checkmark$ | $\checkmark$ |  | 0.18 | 0.049 | 0.0086 | 0.82 | 0.13 | 0.11 |
|  |  |  |  | $[0.17,0.24]$ | $[0.041,0.052]$ | $[0.0053,0.011]$ | $[0.75,0.90]$ | $[0.12,0.14]$ | $[0.091,0.16]$ |

# A.8. Diagnostics 

When approximating a posterior through regression on a set of log-density evaluations, several issues can lead to poor-quality approximations. The training points may inadequately cover the true target posterior, and while the normalizing flow can extrapolate to missing regions, its accuracy in these areas is not guaranteed. Additionally, since we treat the unknown log normalizing constant $C$ as an optimization parameter, biased estimates can cause problems: overestimation leads to a hallucination of probability mass in low-density regions, while underestimation results in overly concentrated, mode-seeking behavior.

Given these potential issues, we recommend two complementary diagnostic approaches to practitioners to assess the quality of the flow approximation.

1. When additional noiseless target posterior density evaluations are available, we can use the fitted flow as a proposal distribution for Pareto smoothed importance sampling (PSIS; Vehtari et al., 2022). PSIS computes a Pareto $\hat{k}$ statistic that quantifies how well the proposal (the flow) approximates the target posterior. A value of $\hat{k} \leq 0.7$ indicates a good approximation, while $\hat{k}>0.7$ suggests poor alignment with the

---

#### Page 26

Table A.8: PSIS diagnostics. Both the median and the $95 \%$ confidence interval (CI) of the median are provided. We show the PSIS- $\hat{k}$ statistic computed with 100, 1000, and 2000 proposal samples. $\hat{k}>0.7$ indicates potential issues and is reported in red. (CMA-ES)

| Problem | PSIS- $\hat{k}$ (100) | PSIS- $\hat{k}$ (1000) | PSIS- $\hat{k}$ (2000) |
| :--: | :--: | :--: | :--: |
| Multivariate Rosenbrock-Gaussian | 0.64 [0.38,0.87] | 0.88 [0.63,1.2] | 0.91 [0.75,1.0] |
| Lumpy | 0.36 [0.26,0.45] | 0.34 [0.30,0.42] | 0.39 [0.26,0.45] |
| Lotka-Volterra model | 0.50 [0.24,0.57] | 0.41 [0.27,0.52] | 0.39 [0.28,0.56] |
| Multisensory | 0.23 [0.15,0.50] | 0.37 [0.31,0.50] | 0.53 [0.43,0.56] |

posterior (Yao et al., 2018; Dhaka et al., 2021; Vehtari et al., 2022). ${ }^{7}$ The target log density evaluations needed for this diagnostic can be computed in parallel for efficiency.
2. A simple yet effective complementary diagnostic approach uses corner plots (ForemanMackey, 2016) to visualize flow samples with pairwise two-dimensional marginal densities, alongside log-density observation points $\mathbf{X}$. This visualization can reveal a common failure mode known as hallucination (De Souza et al., 2022; Li et al., 2024), where the surrogate model, the flow in our case, erroneously places significant probability mass in regions far from the training points.

We illustrate these two diagnostics in detail with examples below. For PSIS, we use the normalizing flow $q_{\boldsymbol{\phi}}$ as the proposal distribution for importance sampling and compute the importance weights,

$$
r_{s}=\frac{p_{\text {target }}\left(\mathbf{x}_{s}\right)}{q_{\boldsymbol{\phi}}\left(\mathbf{x}_{s}\right)}, \quad \mathbf{x}_{s} \sim q_{\boldsymbol{\phi}}(\mathbf{x})
$$

PSIS fits a generalized Pareto distribution using the importance ratios $r_{s}$ and returns the estimated shape parameter $\hat{k}$ which serves as a diagnostic for indicating the discrepancy between the proposal distribution and the target distribution. $\hat{k}<0.7$ indicates that the normalizing flow approximation is close to the target distribution. Values of $\hat{k}$ above the 0.7 threshold are indicative of potential issues and reported in red. As shown in Table A. 8 and A.9, PSIS- $\hat{k}$ diagnostics is below the threshold 0.7 for all problems except the multivariate Rosenbrock-Gaussian posterior. However, as we see from the metrics $\Delta$ LML, MMTV, and GsKL in Table 1 and Figure A.2(d), the normalizing flow approximation matches the ground truth target posterior well. We hypothesize that the alarm raised by PSIS- $\hat{k}$ is due to the long tail in multivariate Rosenbrock-Gaussian distribution and PSIS- $\hat{k}$ is sensitive to tail underestimation in the normalizing flow approximation.

In the case of noisy likelihood evaluations or additional likelihood evaluations not available, PSIS could not be applied. Instead, we can use corner plots (Foreman-Mackey, 2016)

[^0]
[^0]:    7. Apart from being a diagnostic, importance sampling can help refine the approximate posterior when $\hat{k}<0.7$.

---

#### Page 27

Table A.9: PSIS diagnostics. Both the median and the $95 \%$ confidence interval (CI) of the median are provided. We show the PSIS- $\hat{k}$ statistic computed with 100, 1000, and 2000 proposal samples. $\hat{k}>0.7$ indicates potential issues and is reported in red. (BADS)

| Problem | PSIS- $\hat{k}$ (100) | PSIS- $\hat{k}$ (1000) | PSIS- $\hat{k}$ (2000) |
| :--: | :--: | :--: | :--: |
| Multivariate Rosenbrock-Gaussian | 0.54 [0.33,0.79] | 0.61 [0.46,0.97] | 0.77 [0.44,1.1] |
| Lumpy | 0.41 [0.23,0.46] | 0.37 [0.32,0.45] | 0.32 [0.25,0.45] |
| Lotka-Volterra model | 0.41 [0.22,0.58] | 0.35 [0.27,0.38] | 0.35 [0.26,0.45] |
| Multisensory | 0.52 [0.20,0.58] | 0.34 [0.27,0.41] | 0.36 [0.26,0.47] |

to detect algorithm failures. Corner plots visualize posterior samples using pairwise twodimensional marginal density contours, along with one-dimensional histograms for marginal distributions. For diagnostics purposes, we could overlay training points $\mathbf{X}$ for NFR onto the corner plots to check whether high-probability regions are adequately supported by training data (Li et al., 2024). A common failure mode, known as hallucination (De Souza et al., 2022; Li et al., 2024), occurs when flow-generated samples lie far from the training points, indicating that the flow predictions cannot be trusted. Figure A. 1 provides an example of such a diagnostic plot. The failure case shown in Figure A.1(a) was obtained by omitting the flow priors as done in the ablation study (Appendix A.7).

---

#### Page 28

> **Image description.** This image contains two corner plots, labeled (a) and (b), arranged side-by-side. Each plot displays a matrix of scatter plots and histograms, visualizing the relationships between five variables: ws, wm, μp, σp, and λ.
> 
> In both plots:
> *   The diagonal elements of the matrix are histograms, showing the distribution of each individual variable. These histograms are represented by orange lines.
> *   The off-diagonal elements are scatter plots, showing the pairwise relationships between the variables. These are represented by blue points.
> 
> Differences between plot (a) and (b):
> *   In plot (a), the blue points in the scatter plots are more dispersed, indicating weaker correlations between the variables.
> *   In plot (b), the blue points are more concentrated, forming distinct clusters and elongated shapes, indicating stronger correlations. Orange density contours are overlaid on the scatter plots in plot (b), highlighting the regions of highest density.
> *   The scales of the axes differ between the two plots, reflecting different ranges of values for the variables. For example, the y-axis for wm in plot (a) extends to 0.4, while in plot (b) it extends to 0.06. Similarly, the x-axis for λ in plot (a) extends to 0.16, while in plot (b) it extends to 0.020.
> 
> The x-axis labels are ws, wm, μp, σp, and λ. The y-axis labels are wm, μp, σp, and λ.

Figure A.1: Diagnostics using corner plots. The orange density contours represent the flow posterior samples, while the blue points indicate training data for flow regression. (a) The flow's probability mass escapes into regions with few or no training points, highlighting an unreliable flow approximation. (b) The highprobability region of the flow is well supported by training points, indicating that the qualitative diagnostic check is passed.

---

#### Page 29

# A.9. Visualization of posteriors 

As an illustration of our results, we use corner plots (Foreman-Mackey, 2016) to visualize posterior samples with pairwise two-dimensional marginal density contours, as well as the 1D marginals histograms. In the following pages, we report example solutions obtained from a run for each problem and algorithm (Laplace, BBVI, VSBQ, NFR). ${ }^{8}$ The ground-truth posterior samples are in black and the approximate posterior samples from the algorithm are in orange (see Figure A.2, A.3, A.4, A.5, and A.6).
8. Both VSBQ and NFR use the log-density evaluations from CMA-ES, as described in the main text.

---

#### Page 30

> **Image description.** The image contains two triangular grid plots, each displaying a matrix of scatter plots and histograms.
> 
> The first plot, labeled "(a) Laplace" at the bottom, shows the relationships between variables X1 through X6. The diagonal elements are histograms, each showing the distribution of a single variable. These histograms have two overlaid lines, one in black and one in orange. The off-diagonal elements are scatter plots, showing the relationship between two variables. These scatter plots are created with gray points, and overlaid with orange points and black contour lines.
> 
> The second plot, labeled "(b) BBVI (10x)" at the bottom, is structured identically to the first plot. It also shows relationships between variables X1 through X6, with histograms on the diagonal and scatter plots off the diagonal. The visual style (colors, point density, contour lines) is the same as in the first plot. The axes labels are visible along the left and bottom edges of each plot.

---

#### Page 31

> **Image description.** This image shows two sets of plots arranged in a matrix format, visualizing multivariate data. The top set is labeled "(c) VSBQ" and the bottom set is labeled "(d) NFR". Each set consists of 6 rows and 6 columns of subplots, with the diagonal subplots displaying histograms and the off-diagonal subplots displaying scatter plots with density contours.
> 
> *   **Arrangement:** The plots are arranged in a triangular matrix format. The top row has one plot, the second row has two plots, and so on, until the sixth row which has six plots.
> *   **Histograms (Diagonal Plots):** The diagonal plots from top-left to bottom-right are histograms. Each histogram displays the distribution of a single variable (x1 to x6). The histograms are represented by orange step functions with black outlines.
> *   **Scatter Plots with Density Contours (Off-Diagonal Plots):** The off-diagonal plots are scatter plots with density contours. The x and y axes of each scatter plot correspond to two different variables (e.g., x1 vs x2, x1 vs x3, etc.). The scatter plots show a cloud of points, with orange density contours overlaid to indicate regions of higher data density. The contours are surrounded by a black outline.
> *   **Axes Labels:** The x-axis labels are present on the bottom row of plots and are labeled x1, x2, x3, x4, x5, and x6, respectively. The y-axis labels are present on the left-most column of plots and are labeled x2, x3, x4, x5, and x6, respectively.
> *   **Text Labels:** Below each set of plots, there is a text label. The top set is labeled "(c) VSBQ" and the bottom set is labeled "(d) NFR".
> *   **Color:** The histograms and density contours are orange with black outlines. The scatter plot points are a lighter shade of orange. The background is white.

Figure A.2: Multivariate Rosenbrock-Gaussian $(D=6)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 32

> **Image description.** The image shows two triangular grids of plots, one above the other. Each grid contains a series of histograms and scatter plots arranged in a lower triangular matrix format.
> 
> The top grid is labeled "(a) Laplace" at the bottom. The bottom grid is labeled "(b) BBVI (10x)" at the bottom.
> 
> Each plot in the grid is enclosed in a square frame. The diagonal plots in each grid are histograms, showing the distribution of a single variable. The off-diagonal plots are scatter plots showing the relationship between two variables.
> 
> The x-axis of each plot is labeled with "x1", "x2", ..., "x10" from left to right. The y-axis of each plot is labeled with "x2", "x3", ..., "x10" from top to bottom.
> 
> The histograms show a distribution, with a orange line representing the estimated distribution and a dark line representing the true distribution. The scatter plots show a cloud of points, with orange contours representing the estimated distribution and dark contours representing the true distribution. The orange contours are generally more concentrated than the dark contours.

---

#### Page 33

> **Image description.** The image shows two sets of plots arranged in a triangular grid, visualizing posterior distributions. Each set represents results from a different algorithm: VSBQ (top) and NFR (bottom).
> 
> Each triangular grid consists of 10 rows and 10 columns, with the plots only appearing in the lower triangle. The diagonal plots in each grid show histograms, while the off-diagonal plots show 2D density contours.
> 
> The axes of the histograms are not explicitly labeled with numerical values, but the x-axis of each histogram is labeled with variables X1 to X10. The density contours are colored in orange, and they are overlaid on a background of black points. The plots are arranged such that the x-axis of each plot corresponds to the variable listed along the bottom row (X1 to X10), and the y-axis corresponds to the variable listed along the leftmost column (X2 to X10).
> 
> Below each triangular grid, the name of the algorithm is written: "(c) VSBQ" for the top grid, and "(d) NFR" for the bottom grid.

Figure A.3: Lumpy $(D=10)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 34

> **Image description.** This image presents two sets of plots, each arranged as a matrix, showing the relationships between different parameters. The top matrix is labeled "(a) BBVI (10x)" and the bottom matrix is labeled "(b) VSBQ". Each matrix consists of a triangular arrangement of plots.
> 
> In each matrix:
> *   The diagonal plots are histograms, showing the distribution of individual parameters. Each histogram has a black and an orange outline. The parameters are, from top to bottom: *w*sub{m}, *μ*sub{p}, *σ*sub{p}, and *λ*.
> *   The off-diagonal plots are density plots, showing the joint distribution of pairs of parameters. These plots have an orange filled contour, surrounded by black contour lines. The parameters on the x-axis are *w*sub{s}, *w*sub{m}, *μ*sub{p}, and *σ*sub{p}. The parameters on the y-axis are *w*sub{m}, *μ*sub{p}, *σ*sub{p}, and *λ*.
> 
> The axes of the plots are labeled with numerical values.

---

#### Page 35

> **Image description.** The image is a visualization of a Bayesian timing model, specifically a corner plot displaying the posterior distributions and relationships between different parameters. The plot consists of a grid of sub-figures.
> 
> *   **Arrangement:** The parameters are arranged along the diagonal, with the marginal distributions of each parameter shown as histograms on the diagonal. The off-diagonal plots show the joint distributions of pairs of parameters as scatter plots with density contours.
> 
> *   **Parameters:** The parameters included in the plot are:
>     *   `ws` (horizontal axis of the bottom row, vertical axis of the first column)
>     *   `wm` (horizontal axis of the second row, vertical axis of the second column)
>     *   `μp` (horizontal axis of the third row, vertical axis of the third column)
>     *   `σp` (horizontal axis of the fourth row, vertical axis of the fourth column)
>     *   `λ` (horizontal axis of the fifth row, vertical axis of the fifth column)
> 
> *   **Histograms (Diagonal Plots):** Each diagonal plot shows a histogram. The histograms are represented by orange lines with square markers. There are also black lines, which are slightly offset from the orange lines.
> 
> *   **Scatter Plots and Contours (Off-Diagonal Plots):** The off-diagonal plots show scatter plots with density contours. The scatter points are small and have an orange hue. The density contours are represented by multiple black lines, indicating regions of higher density.
> 
> *   **Text:**
>     *   Parameter labels are present on the axes of the plots (e.g., `ws`, `wm`, `μp`, `σp`, `λ`).
>     *   Numerical values are present on the axes, indicating the range of values for each parameter.
>     *   The label "(c) NFR" is present below the bottom row of plots.
> 
> *   **Colors:** The primary colors used are orange and black. Orange is used for the histograms and scatter points, while black is used for the contours and axis labels.
> 
> In summary, the image presents a corner plot visualizing the posterior distributions and relationships between several parameters in a Bayesian timing model. The plot includes histograms of the marginal distributions on the diagonal and scatter plots with density contours for the joint distributions off the diagonal.

Figure A.4: Bayesian timing model $(D=5)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 36

> **Image description.** This is a correlation plot matrix, also known as a pair plot. It displays the relationships between multiple variables. The matrix is arranged in a triangular format, with each cell representing the relationship between two variables.
> 
> *   **Diagonal Plots:** The diagonal cells show histograms of individual variables. The variables are: beta, gamma, delta, u(0), v(0), sigma_u, and sigma_v. Each histogram shows the distribution of the variable, with the x-axis representing the variable's value and the y-axis representing the frequency. Two histograms are overlaid in each diagonal plot, one in black and one in orange.
> 
> *   **Off-Diagonal Plots:** The off-diagonal cells show scatter plots or contour plots representing the joint distribution of two variables. The x-axis of each plot corresponds to the variable in the column, and the y-axis corresponds to the variable in the row. These plots show the correlation or dependence between the variables. The plots contain a scatter plot of points, with higher density areas indicated by darker shading and contour lines. The scatter plots show the relationships between all pairs of variables, such as alpha vs. beta, alpha vs. gamma, beta vs. gamma, and so on.
> 
> *   **Variables:** The variables are labeled along the x and y axes. The x-axis labels are: alpha, beta, gamma, delta, u(0), v(0), sigma_u, and sigma_v. The y-axis labels are the same, but vertically oriented. Numerical values are shown along the axes to indicate the range of each variable.
> 
> *   **Overall Structure:** The matrix is arranged in a lower triangular format, meaning that only the cells below and to the left of the diagonal are populated. The upper triangular part is empty.
> 
> *   **Text:** The plot includes labels for each variable along the axes. The labels are "alpha", "beta", "gamma", "delta", "u(0)", "v(0)", "sigma_u", and "sigma_v".
(a) Laplace

> **Image description.** The image is a correlation plot, also known as a scatterplot matrix, displaying the relationships between multiple variables. It's arranged as a grid of plots, with each variable represented along the diagonal.
> 
> *   **Arrangement:** The plot is arranged in a triangular matrix format. The diagonal elements are histograms, while the off-diagonal elements are scatter plots or contour plots showing the joint distribution of pairs of variables.
> *   **Variables:** The variables along the axes are labeled as α, β, γ, δ, u(0), v(0), σu, and σv. These labels are located along the bottom row and the leftmost column of the matrix.
> *   **Histograms:** The diagonal plots are histograms, showing the marginal distribution of each variable. They are represented by orange step-like lines.
> *   **Scatter/Contour Plots:** The off-diagonal plots show the joint distribution of pairs of variables. They appear to be contour plots, with concentric lines indicating regions of higher density. The contours are dark brown/black, and the filled areas within the contours are a lighter orange color.
> *   **Text:** The text "(b) BBVI (10×)" is located at the bottom of the image, likely indicating the method used to generate the plots.
> *   **Axes:** The axes of the plots are labeled with numerical values. For example, the x-axis of the histogram for α ranges from approximately 0.4 to 0.8. The y-axes of the histograms represent the frequency or density of the variables.
> *   **Overall Impression:** The plot shows the correlations and dependencies between the variables. The shapes and orientations of the contour plots reveal the nature and strength of these relationships.

---

#### Page 37

> **Image description.** The image contains two triangular grids of plots, each representing a posterior visualization. The top grid is labeled "(c) VSBQ" and the bottom grid is labeled "(d) NFR". Each grid consists of 21 subplots arranged in a lower triangular matrix format.
> 
> *   **Structure:** The grids are structured such that the diagonal plots show histograms, while the off-diagonal plots show scatter density plots.
> 
> *   **Axes Labels:** The variables represented are α, β, γ, δ, u(0), v(0), σ<sub>u</sub>, and σ<sub>v</sub>. These labels appear along the bottom row and leftmost column of each grid.
> 
> *   **Histograms:** The diagonal plots display histograms in both black and orange. The orange histograms represent the posterior samples from different algorithms, while the black histograms denote ground truth samples.
> 
> *   **Scatter Density Plots:** The off-diagonal plots display scatter density plots, showing the relationship between pairs of variables. These plots contain orange density contours and orange points, representing posterior samples from different algorithms, and black contours and points, denoting ground truth samples.
> 
> *   **Color and Style:** The plots use a combination of black and orange colors. Black is used for ground truth data, while orange represents the data from the algorithms being visualized. Density contours are used to represent the concentration of data points in the scatter plots.

Figure A.5: Lotka-Volterra mode $(D=8)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 38

> **Image description.** The image is a correlation plot, specifically a matrix of scatter plots and histograms, arranged in a triangular format.
> 
> *   **Arrangement:** The plot is structured as a lower triangular matrix. Each cell in the matrix represents the relationship between two variables. The diagonal cells contain histograms of individual variables. The off-diagonal cells contain scatter plots showing the joint distribution of two variables.
> 
> *   **Variables:** The variables are labeled X1 through X12 along the axes. The x-axis labels are at the bottom, and the y-axis labels are on the left.
> 
> *   **Scatter Plots:** The scatter plots in the off-diagonal cells show the relationship between the corresponding variables. The density of points is represented by gray contours. There are also orange areas that seem to highlight regions of higher density.
> 
> *   **Histograms:** The histograms on the diagonal show the distribution of each individual variable. Each histogram has two overlaid distributions: one in black and one in orange.
> 
> *   **Text:** The plot is labeled "(a) VSBQ" at the bottom. The axes are labeled with the variable names (X1 to X12) and numerical values.

---

#### Page 39

> **Image description.** The image is a triangular matrix of plots, displaying pairwise relationships between variables X1 through X12. The plots on the diagonal are histograms, while the off-diagonal plots are scatter plots with density contours. Each plot contains two sets of data, represented by orange and black lines/points.
> 
> *   **Arrangement:** The plots are arranged in a lower triangular matrix format. The variable names X1 to X12 are displayed along the bottom row and the left column, corresponding to the variables being compared in each plot.
> 
> *   **Diagonal Plots (Histograms):** The plots on the diagonal (from top-left to bottom-right) are histograms. Each histogram shows the distribution of a single variable (X1 to X12). Two histograms are overlaid in each plot, one in orange and one in black. The x-axis of each histogram represents the variable's value, and the y-axis represents the frequency or density.
> 
> *   **Off-Diagonal Plots (Scatter Plots with Density Contours):** The off-diagonal plots are scatter plots that show the relationship between two variables. Each point represents a sample. Density contours are overlaid on the scatter plots, indicating regions of higher point density. Two sets of density contours and points are present in each plot, one in orange and one in black.
> 
> *   **Colors:** The image primarily uses black and orange. The data is represented by orange and black lines/points.
> 
> *   **Text:**
>     *   The x-axis labels are "X1" through "X12" along the bottom.
>     *   The y-axis labels are "X2" through "X12" along the left side.
>     *   There are numerical values on the axes of each plot, indicating the range of values for each variable.
>     *   The text "(b) NFR" is located at the bottom center of the image.
> 
> *   **Interpretation:** The image visualizes the posterior distributions of 12 variables and their pairwise relationships. The orange and black data represent different algorithms or datasets being compared. The histograms show the marginal distributions of each variable, and the scatter plots with density contours show the joint distributions of each pair of variables. The differences between the orange and black data indicate differences in the posterior distributions obtained by the different algorithms or datasets.

Figure A.6: Multisensory $(D=12)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.