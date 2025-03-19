```
@article{acerbi2020variational,
  title={Variational Bayesian Monte Carlo with Noisy Likelihoods},
  author={Luigi Acerbi},
  year={2020},
  journal={The Thirty-fourth Annual Conference on Neural Information Processing Systems (NeurIPS 2020)}
}
```

---

#### Page 1

# Variational Bayesian Monte Carlo with Noisy Likelihoods

Luigi Acerbi\*<br>Department of Computer Science<br>University of Helsinki<br>luigi.acerbi@helsinki.fi

#### Abstract

Variational Bayesian Monte Carlo (VBMC) is a recently introduced framework that uses Gaussian process surrogates to perform approximate Bayesian inference in models with black-box, non-cheap likelihoods. In this work, we extend VBMC to deal with noisy log-likelihood evaluations, such as those arising from simulationbased models. We introduce new 'global' acquisition functions, such as expected information gain (EIG) and variational interquantile range (VIQR), which are robust to noise and can be efficiently evaluated within the VBMC setting. In a novel, challenging, noisy-inference benchmark comprising of a variety of models with real datasets from computational and cognitive neuroscience, VBMC +VIQR achieves state-of-the-art performance in recovering the ground-truth posteriors and model evidence. In particular, our method vastly outperforms 'local' acquisition functions and other surrogate-based inference methods while keeping a small algorithmic cost. Our benchmark corroborates VBMC as a general-purpose technique for sample-efficient black-box Bayesian inference also with noisy models.

## 1 Introduction

Bayesian inference provides a principled framework for uncertainty quantification and model selection via computation of the posterior distribution over model parameters and of the model evidence $[1,2]$. However, for many black-box models of interest in fields such as computational biology and neuroscience, (log-)likelihood evaluations are computationally expensive (thus limited in number) and noisy due to, e.g., simulation-based approximations [3, 4]. These features make standard techniques for approximate Bayesian inference such as Markov Chain Monte Carlo (MCMC) ineffective.

Variational Bayesian Monte Carlo (VBMC) is a recently proposed framework for Bayesian inference with non-cheap models [5,6]. VBMC performs variational inference using a Gaussian process (GP [7]) as a statistical surrogate model for the expensive log posterior distribution. The GP model is refined via active sampling, guided by a 'smart' acquisition function that exploits uncertainty and other features of the surrogate. VBMC is particularly efficient thanks to a representation that affords fast integration via Bayesian quadrature [8,9], and unlike other surrogate-based techniques it performs both posterior and model inference [5]. However, the original formulation of VBMC does not support noisy model evaluations, and recent work has shown that surrogate-based approaches that work well in the noiseless case may fail in the presence of even small amounts of noise [10].

In this work, we extend VBMC to deal robustly and effectively with noisy log-likelihood evaluations, broadening the class of models that can be estimated via the method. With our novel contributions, VBMC outperforms other state-of-the-art surrogate-based techniques for black-box Bayesian inference in the presence of noisy evaluations - in terms of speed, robustness and quality of solutions.

[^0]
[^0]: \* Previous affiliation: Department of Basic Neuroscience, University of Geneva.

---

#### Page 2

Contributions We make the following contributions: (1) we introduce several new acquisition functions for VBMC that explicitly account for noisy log-likelihood evaluations, and leverage the variational representation to achieve much faster evaluation than competing methods; (2) we introduce variational whitening, a technique to deal with non-axis aligned posteriors, which are otherwise potentially problematic for VBMC (and GP surrogates more in general) in the presence of noise; (3) we build a novel and challenging noisy-inference benchmark that includes five different models from computational and cognitive neuroscience, ranging from 3 to 9 parameters, and applied to real datasets, in which we test VBMC and other state-of-the-art surrogate-based inference techniques. The new features have been implemented in VBMC: https://github.com/acerbilab/vbmc.

Related work Our paper extends the VBMC framework [5,6] by building on recent informationtheoretical approaches to adaptive Bayesian quadrature [11], and on recent theoretical and empirical results for GP-surrogate Bayesian inference for simulation-based models [10, 12, 13]. For noiseless evaluations, previous work has used GP surrogates for estimation of posterior distributions [14-16] and Bayesian quadrature for calculation of the model evidence [9, 17-20]. Our method is also closely related to (noisy) Bayesian optimization [21-27]. A completely different approach, but worth mentioning for the similar goal, trains deep networks on simulated data to reconstruct approximate Bayesian posteriors from data or summary statistics thereof [28-31].

# 2 Variational Bayesian Monte Carlo (VBMC)

We summarize here the Variational Bayesian Monte Carlo (VBMC) framework [5]. If needed, we refer the reader to the Supplement for a recap of key concepts in variational inference, GPs and Bayesian quadrature. Let $f=\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})$ be the target log joint probability (unnormalized posterior), where $p(\mathcal{D} \mid \boldsymbol{\theta})$ is the model likelihood for dataset $\mathcal{D}$ and parameter vector $\boldsymbol{\theta} \in \mathcal{X} \subseteq \mathbb{R}^{D}$, and $p(\boldsymbol{\theta})$ the prior. We assume that only a limited number of log-likelihood evaluations are available, up to several hundreds. VBMC works by iteratively improving a variational approximation $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$, indexed by $\boldsymbol{\phi}$, of the true posterior density. In each iteration $t$, the algorithm:

1. Actively samples sequentially $n_{\text {active }}$ 'promising' new points, by iteratively maximizing a given acquisition function $a(\boldsymbol{\theta}): \mathcal{X} \rightarrow \mathbb{R}$; for each selected point $\boldsymbol{\theta}_{\star}$ evaluates the target $\boldsymbol{y}_{\star} \equiv f\left(\boldsymbol{\theta}_{\star}\right)\left(n_{\text {active }}=5\right.$ by default).
2. Trains a GP surrogate model of the log joint $f$, given the training set $\boldsymbol{\Xi}_{t}=\left\{\boldsymbol{\Theta}_{t}, \boldsymbol{y}_{t}\right\}$ of input points and their associated observed values so far.
3. Updates the variational posterior parameters $\boldsymbol{\phi}_{t}$ by optimizing the surrogate ELBO (variational lower bound on the model evidence) calculated via Bayesian quadrature.

This loop repeats until reaching a termination criterion (e.g., budget of function evaluations or lack of improvement over several iterations), and the algorithm returns both the variational posterior and posterior mean and variance of the ELBO. VBMC includes an initial warm-up stage to converge faster to regions of high posterior probability, before starting to refine the variational solution (see [5]).

### 2.1 Basic features

We briefly describe here basic features of the original VBMC framework [5] (see also Supplement).
Variational posterior The variational posterior is a flexible mixture of $K$ multivariate Gaussians, $q(\boldsymbol{\theta}) \equiv q_{\boldsymbol{\phi}}(\boldsymbol{\theta})=\sum_{k=1}^{K} w_{k} \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)$, where $w_{k}, \boldsymbol{\mu}_{k}$, and $\sigma_{k}$ are, respectively, the mixture weight, mean, and scale of the $k$-th component; and $\boldsymbol{\Sigma}$ is a common diagonal covariance matrix $\boldsymbol{\Sigma} \equiv \operatorname{diag}\left[\lambda^{(1)^{2}}, \ldots, \lambda^{(D)^{2}}\right]$. For a given $K$, the variational parameter vector is $\boldsymbol{\phi} \equiv\left(w_{1}, \ldots, w_{K}\right.$, $\left.\boldsymbol{\mu}_{1}, \ldots, \boldsymbol{\mu}_{K}, \sigma_{1}, \ldots, \sigma_{K}, \boldsymbol{\lambda}\right) . \quad K$ is set adaptively; fixed to $K=2$ during warm-up, and then increasing each iteration if it leads to an improvement of the ELBO.

Gaussian process model In VBMC, the log joint $f$ is approximated by a GP surrogate model with a squared exponential (rescaled Gaussian) kernel, a Gaussian likelihood, and a negative quadratic mean function which ensures finiteness of the variational objective [5,6]. In the original formulation, observations are assumed to be exact (non-noisy), so the GP likelihood only included a small observation noise $\sigma_{\text {obs }}^{2}$ for numerical stability [32]. GP hyperparameters are estimated initially via MCMC sampling [33], when there is larger uncertainty about the GP model, and later via a maximum-a-posteriori (MAP) estimate using gradient-based optimization (see [5] for details).

---

#### Page 3

The Evidence Lower Bound (ELBO) Using the GP surrogate model $f$, and for a given variational posterior $q_{\phi}$, the posterior mean of the surrogate ELBO can be estimated as

$$
\mathbb{E}_{f \mid \mathbb{E}}[\operatorname{ELBO}(\phi)]=\mathbb{E}_{f \mid \mathbb{E}}\left[\mathbb{E}_{\phi}[f]\right]+\mathcal{H}\left[q_{\phi}\right]
$$

where $\mathbb{E}_{f \mid \mathbb{E}}\left[\mathbb{E}_{\phi}[f]\right]$ is the posterior mean of the expected log joint under the GP model, and $\mathcal{H}\left[q_{\phi}\right]$ is the entropy of the variational posterior. In particular, the expected $\log$ joint $\mathcal{G}$ takes the form

$$
\mathcal{G}\left[q_{\phi} \mid f\right] \equiv \mathbb{E}_{\phi}[f]=\int q_{\phi}(\boldsymbol{\theta}) f(\boldsymbol{\theta}) d \boldsymbol{\theta}
$$

Crucially, the choice of variational family and GP representation affords closed-form solutions for the posterior mean and variance of Eq. 2 (and of their gradients) by means of Bayesian quadrature [8, 9]. The entropy of $q_{\phi}$ and its gradient are estimated via simple Monte Carlo and the reparameterization trick $[34,35]$, such that Eq. 1 can be optimized via stochastic gradient ascent [36].

Acquisition function During the active sampling stage, new points to evaluate are chosen sequentially by maximizing a given acquisition function $a(\boldsymbol{\theta}): \mathcal{X} \rightarrow \mathbb{R}$ constructed to represent useful search heuristics [37]. The VBMC paper introduced prospective uncertainty sampling [5],

$$
a_{\mathrm{pro}}(\boldsymbol{\theta})=s_{\mathbb{E}}^{2}(\boldsymbol{\theta}) q_{\phi}(\boldsymbol{\theta}) \exp \left(\bar{f}_{\mathbb{E}}(\boldsymbol{\theta})\right)
$$

where $\bar{f}_{\mathbb{E}}(\boldsymbol{\theta})$ and $s_{\mathbb{E}}^{2}(\boldsymbol{\theta})$ are, respectively, the GP posterior latent mean and variance at $\boldsymbol{\theta}$ given the current training set $\mathbb{E}$. Effectively, $a_{\text {pro }}$ promotes selection of new points from regions of high probability density, as represented by the variational posterior and (exponentiated) posterior mean of the surrogate log-joint, for which we are also highly uncertain (high variance of the GP surrogate).

Inference space The variational posterior and GP surrogate in VBMC are defined in an unbounded inference space equal to $\mathbb{R}^{D}$. Parameters that are subject to bound constraints are mapped to the inference space via a shifted and rescaled logit transform, with an appropriate Jacobian correction to the log-joint. Solutions are transformed back to the original space via a matched inverse transform, e.g., a shifted and rescaled logistic function for bound parameters (see $[5,38]$ ).

# 2.2 Variational whitening

One issue of the standard VBMC representation of both the variational posterior and GP surrogate is that it is axis-aligned, which makes it ill-suited to deal with highly correlated posteriors. As a simple and inexpensive solution, we introduce here variational whitening, which consists of a linear transformation $\mathbf{W}$ of the inference space (a rotation and rescaling) such that the variational posterior $q_{\phi}$ obtains unit diagonal covariance matrix. Since $q_{\phi}$ is a mixture of Gaussians in inference space, its covariance matrix $\mathbf{C}_{\phi}$ is available in closed form and we can calculate the whitening transform $\mathbf{W}$ by performing a singular value decomposition (SVD) of $\mathbf{C}_{\phi}$. We start performing variational whitening a few iterations after the end of warm-up, and then at increasingly more distant intervals. By default we use variational whitening with all variants of VBMC tested in this paper; see the Supplement for an ablation study demonstrating its usefulness and for further implementation details.

## 3 VBMC with noisy likelihood evaluations

Extending the framework described in Section 2, we now assume that evaluations of the log-likelihood $y_{n}$ can be noisy, that is

$$
y_{n}=f\left(\boldsymbol{\theta}_{n}\right)+\sigma_{\mathrm{obs}}\left(\boldsymbol{\theta}_{n}\right) \varepsilon_{n}, \quad \varepsilon_{n} \stackrel{\text { i.i.d. }}{\sim} \mathcal{N}(0,1)
$$

where $\sigma_{\text {obs }}: \mathcal{X} \rightarrow\left[\sigma_{\min }, \infty\right)$ is a function of the input space that determines the standard deviation (SD) of the observation noise. For this work, we use $\sigma_{\min }^{2}=10^{-5}$ and we assume that the evaluation of the log-likelihood at $\boldsymbol{\theta}_{n}$ returns both $y_{n}$ and a reasonable estimate $\left(\hat{\sigma}_{\text {obs }}\right)_{n}$ of $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{n}\right)$. Here we estimate $\sigma_{\text {obs }}(\boldsymbol{\theta})$ outside the training set via a nearest-neighbor approximation (see Supplement), but more sophisticated methods could be used (e.g., by training a GP model on $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{n}\right)$ [39]).
The synthetic likelihood (SL) technique [3,4] and inverse binomial sampling (IBS) [40,41] are examples of log-likelihood estimation methods for simulation-based models that satisfy the assumptions of our observation model (Eq. 4). Recent work demonstrated empirically that log-SL estimates are approximately normally distributed, and their SD can be estimated accurately via bootstrap [10].

---

#### Page 4

> **Image description.** The image consists of three panels, labeled A, B, and C, presenting data related to VBMC (Variational Bayesian Monte Carlo) with noisy likelihoods.
>
> Panel A: This panel displays a contour plot labeled "True posterior." The plot shows a V-shaped distribution in a two-dimensional space. The x-axis is labeled "θ₁" and the y-axis is labeled "θ₂". The contours are filled with colors ranging from blue on the outside to yellow in the center, indicating increasing probability density.
>
> Panel B: This panel contains two sub-panels, both labeled "VBMC+NPRO" (left) and "VBMC+VIQR" (right). Both sub-panels show scatter plots with contour lines overlaid. The x-axis is labeled "θ₁" and the y-axis is labeled "θ₂". Black dots are scattered across the plot, representing training samples. Red crosses are also present, seemingly indicating the centers of variational mixture components. The contour lines, similar to Panel A, represent probability densities. The distribution of points and contours in each sub-panel is different, reflecting the different acquisition functions (NPRO and VIQR).
>
> Panel C: This panel presents a line graph labeled "Model evidence". The x-axis is labeled "Likelihood evaluations" and ranges from approximately 0 to 100. The y-axis ranges from -4 to -1. Two lines are plotted: a black line labeled "VIQR" and a green line labeled "NPRO". Shaded areas around each line represent the 95% confidence interval (CI) of the ELBO (Evidence Lower Bound) estimated via Bayesian quadrature. A dashed horizontal line is drawn at y = -2.27, representing the true log marginal likelihood (LML). The VIQR line appears to converge closer to the true LML than the NPRO line.

Figure 1: VBMC with noisy likelihoods. A. True target pdf $(D=2)$. We assume noisy loglikelihood evaluations with $\sigma_{\text {obs }}=1$. B. Contour plots of the variational posterior after 100 likelihood evaluations, with the noise-adjusted $a_{\text {npro }}$ acquisition function (left) and the newly proposed $a_{\text {VIQR }}$ (right). Red crosses indicate the centers of the variational mixture components, black dots are the training samples. C. ELBO as a function of likelihood evaluations. Shaded area is $95 \%$ CI of the ELBO estimated via Bayesian quadrature. Dashed line is the true log marginal likelihood (LML).

IBS is a recently reintroduced statistical technique that produces both normally-distributed, unbiased estimates of the log-likelihood and calibrated estimates of their variance [41].
In the rest of this section, we describe several new acquisition functions for VBMC specifically designed to deal with noisy log-likelihood evaluations. Figure 1 shows VBMC at work in a toy noisy scenario (a 'banana' 2D posterior), for two acquisition functions introduced in this section.
Predictions with noisy evaluations A useful quantity for this section is $s_{\mathbb{E}_{i}, \boldsymbol{\theta}_{*}}^{2}(\boldsymbol{\theta})$, the predicted posterior GP variance at $\boldsymbol{\theta}$ if we make a function evaluation at $\boldsymbol{\theta}_{\star}$, with $y_{\star}$ distributed according to the posterior predictive distribution (that is, inclusive of observation noise $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\star}\right)$ ), given training data $\boldsymbol{\Xi}$. Conveniently, $s_{\mathbb{E}_{i}, \boldsymbol{\theta}_{\star}}^{2}(\boldsymbol{\theta})$ can be expressed in closed form as

$$
s_{\mathbb{E}_{i}, \boldsymbol{\theta}_{\star}}^{2}(\boldsymbol{\theta})=s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta})-\frac{C_{\boldsymbol{\Xi}}^{2}\left(\boldsymbol{\theta}_{*} \boldsymbol{\theta}_{\star}\right)}{C_{\boldsymbol{\Xi}}\left(\boldsymbol{\theta}_{\star}, \boldsymbol{\theta}_{\star}\right)+\sigma_{\text {obs }}^{2}\left(\boldsymbol{\theta}_{\star}\right)}
$$

where $C_{\boldsymbol{\Xi}}(\cdot, \cdot)$ denotes the GP posterior covariance (see [10, Lemma 5.1], and also [13,42]).

# 3.1 Noisy prospective uncertainty sampling

The rationale behind $a_{\text {pro }}$ (Eq. 3) and similar heuristic 'uncertainty sampling' acquisition functions $[6,18]$ is to evaluate the log joint where the pointwise variance of the integrand in the expected log joint (as per Eq. 2, or variants thereof) is maximum. For noiseless evaluations, this choice is equivalent to maximizing the variance reduction of the integrand after an observation. Considering the GP posterior variance reduction, $\Delta s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta}) \equiv s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta})-s_{\mathbb{E}_{i}, \boldsymbol{\theta}}^{2}(\boldsymbol{\theta})$, we see that, in the absence of observation noise, $s_{\mathbb{E}_{i}, \boldsymbol{\theta}}^{2}(\boldsymbol{\theta})=0$ and $\Delta s^{2}(\boldsymbol{\theta})_{\boldsymbol{\Xi}}=s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta})$. Thus, a natural generalization of uncertainty sampling to the noisy case is obtained by switching the GP posterior variance in Eq. 3 to the GP posterior variance reduction. Improving over the original uncertainty sampling, this generalization accounts for potential observation noise at the candidate location.
Following this reasoning, we generalize uncertainty sampling to noisy observations by defining the noise-adjusted prospective uncertainty sampling acquisition function,

$$
a_{\text {npro }}(\boldsymbol{\theta})=\Delta s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta}) q_{\boldsymbol{\phi}}(\boldsymbol{\theta}) \exp \left(\bar{f}_{\boldsymbol{\Xi}}(\boldsymbol{\theta})\right)=\left(\frac{s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta})}{s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta})+\sigma_{\text {obs }}^{2}(\boldsymbol{\theta})}\right) s_{\boldsymbol{\Xi}}^{2}(\boldsymbol{\theta}) q_{\boldsymbol{\phi}}(\boldsymbol{\theta}) \exp \left(\bar{f}_{\boldsymbol{\Xi}}(\boldsymbol{\theta})\right)
$$

where we used Eq. 5 to calculate $s_{\mathbb{E}_{i}, \boldsymbol{\theta}}^{2}(\boldsymbol{\theta})$. Comparing Eq. 6 to Eq. 3, we see that $a_{\text {npro }}$ has an additional multiplicative term that accounts for the residual variance due to a potentially noisy observation. As expected, it is easy to see that $a_{\text {npro }}(\boldsymbol{\theta}) \rightarrow a_{\text {pro }}(\boldsymbol{\theta})$ for $\sigma_{\text {obs }}(\boldsymbol{\theta}) \rightarrow 0$.
While $a_{\text {npro }}$ and other forms of uncertainty sampling operate pointwise on the posterior density, we consider next global (integrated) acquisition functions that account for non-local changes in the GP surrogate model when making a new observation, thus driven by uncertanty in posterior mass.

### 3.2 Expected information gain (EIG)

A principled information-theoretical approach suggests to sample points that maximize the expected information gain (EIG) about the integral of interest (Eq. 2). Following recent work on multi-source

---

#### Page 5

active-sampling Bayesian quadrature [11], we can do so by choosing the next location $\boldsymbol{\theta}_{\star}$ that maximizes the mutual information $I\left[\mathcal{G} ; y_{\star}\right]$ between the expected log joint $\mathcal{G}$ and a new (unknown) observation $y_{\star}$. Since all involved quantities are jointly Gaussian, we obtain

$$
a_{\mathrm{EIG}}(\boldsymbol{\theta})=-\frac{1}{2} \log \left(1-\rho^{2}(\boldsymbol{\theta})\right), \quad \text { with } \rho(\boldsymbol{\theta}) \equiv \frac{\mathbb{E}_{\boldsymbol{\phi}}\left[C_{\mathbb{E}}(f(\cdot), f(\boldsymbol{\theta}))\right]}{\sqrt{v_{\mathbb{E}}(\boldsymbol{\theta}) \mathbb{V}_{f(\mathbb{E}}[\mathcal{G}]}}
$$

where $\rho(\cdot)$ is the scalar correlation [11], $v_{\mathbb{E}}(\cdot)$ the GP posterior predictive variance (including observation noise), and $\mathbb{V}_{f(\mathbb{E}}[\mathcal{G}]$ the posterior variance of the expected log joint - all given the current training set $\boldsymbol{\Xi}$. The scalar correlation in Eq. 7 has a closed-form solution thanks to Bayesian quadrature (see Supplement for derivations).

# 3.3 Integrated median / variational interquantile range (IMIQR/ VIQR)

Järvenpää and colleagues [10] recently proposed the interquantile range (IQR) as a robust estimate of the uncertainty of the unnormalized posterior, as opposed to the variance, and derived the integrated median interquantile range (IMIQR) acquisition function from Bayesian decision theory,

$$
a_{\mathrm{IMIQR}}(\boldsymbol{\theta})=-2 \int_{\mathcal{X}} \exp \left(\bar{f}_{\mathbb{E}}\left(\boldsymbol{\theta}^{\prime}\right)\right) \sinh \left(u s_{\mathbb{E} \cup \boldsymbol{\theta}}\left(\boldsymbol{\theta}^{\prime}\right)\right) d \boldsymbol{\theta}^{\prime}
$$

where $u \equiv \Phi^{-1}\left(p_{u}\right)$, with $\Phi$ the standard normal CDF and $p_{u} \in(0.5,1)$ a chosen quantile (we use $p_{u}=0.75$ as in [10]); $\sinh (z)=(\exp (z)-\exp (-z)) / 2$ for $z \in \mathbb{R}$ is the hyperbolic sine; and $s_{\mathbb{E} \cup \boldsymbol{\theta}}\left(\boldsymbol{\theta}^{\prime}\right)$ denotes the predicted posterior standard deviation after observing the function at $\boldsymbol{\theta}^{\prime}$, as per Eq. 5. However, the integral in Eq. 8 is intractable, and thus needs to be approximated at a significant computational cost (e.g., via MCMC and importance sampling [10]).
Instead, we note that the term $\exp \left(\bar{f}_{\mathbb{E}}\right)$ in Eq. 8 represents the joint distribution as modeled via the GP surrogate, which VBMC further approximates with the variational posterior $q_{\boldsymbol{\phi}}$ (up to a normalization constant). Thus, we exploit the variational approximation of VBMC to propose here the variational (integrated median) interquantile range (VIQR) acquisition function,

$$
a_{\mathrm{VIQR}}(\boldsymbol{\theta})=-2 \int_{\mathcal{X}} q_{\boldsymbol{\phi}}\left(\boldsymbol{\theta}^{\prime}\right) \sinh \left(u s_{\mathbb{E} \cup \boldsymbol{\theta}}\left(\boldsymbol{\theta}^{\prime}\right)\right) d \boldsymbol{\theta}^{\prime}
$$

where we replaced the surrogate posterior in Eq. 8 with its corresponding variational posterior. Crucially, Eq. 9 can be approximated very cheaply via simple Monte Carlo by drawing $N_{\text {vigr }}$ samples from $q_{\phi}$ (we use $N_{\text {vigr }}=100$ ). In brief, $a_{\text {VIQR }}$ obtains a computational advantage over $a_{\text {IMIQR }}$ at the cost of adding a layer of approximation in the acquisition function $\left(q_{\phi} \approx \exp \left(\bar{f}_{\mathbb{E}}\right)\right.$ ), but it otherwise follows from the same principles. Whether this approximation is effective in practice is an empirical question that we address in the next section.

## 4 Experiments

We tested different versions of VBMC and other surrogate-based inference algorithms on a novel benchmark problem set consisting of a variety of computational models applied to real data (see Section 4.1). For each problem, the goal of inference is to approximate the posterior distribution and the log marginal likelihood (LML) with a fixed budget of likelihood evaluations.

Algorithms In this work, we focus on comparing new acquisition functions for VBMC which support noisy likelihood evaluations, that is $a_{\text {npro }}, a_{\text {EIG }}, a_{\text {IMIQR }}$ and $a_{\text {VIQR }}$ as described in Section 3 (denoted as VBMC plus, respectively, NPRO, EIG, IMIQR or VIQR). As a strong baseline for posterior estimation, we test a state-of-the-art technique for Bayesian inference via GP surrogates, which also uses $a_{\text {IMIQR }}$ [10] (GP-IMIQR). GP-IMIQR was recently shown to decisively outperform several other surrogate-based methods for posterior estimation in the presence of noisy likelihoods [10]. For model evidence evaluation, to our knowledge no previous surrogate-based technique explicitly supports noisy evaluations. We test as a baseline warped sequential active Bayesian integration (WSABI [18]), a competitive method in a previous noiseless comparison [5], adapted here for our benchmark (see Supplement). For each algorithm, we use the same default settings across problems. We do not consider here non-surrogate based methods, such as Monte Carlo and importance sampling, which performed poorly with a limited budget of likelihood evaluations already in the noiseless case [5].

---

#### Page 6

Procedure For each problem, we allow a budget of $50 \times(D+2)$ likelihood evaluations. For each algorithm, we performed 100 runs per problem with random starting points, and we evaluated performance with several metrics (see Section 4.2). For each metric, we report as a function of likelihood evaluations the median and $95 \%$ CI of the median calculated by bootstrap (see Supplement for a 'worse-case' analysis of performance). For algorithms other than VBMC, we only report metrics they were designed for (posterior estimation for GP-IMIQR, model evidence for WSABI).

Noisy log-likelihoods For a given data set, model and parameter vector $\boldsymbol{\theta}$, we obtain noisy evaluations of the log-likelihood through different methods, depending on the problem. In the synthetic likelihood (SL) approach, we run $N_{\text {sim }}$ simulations for each evaluation, and estimate the log-likelihood of summary statistics of the data under a multivariate normal assumption $[3,4,10]$. With inverse binomial sampling (IBS), we obtain unbiased estimates of the log-likelihood of an entire data set by sampling from the model until we obtain a 'hit' for each data point [40,41]; we repeat the process $N_{\text {rep }}$ times and average the estimates for higher precision. Finally, for a few analyses we 'emulate' noisy evaluations by adding i.i.d. Gaussian noise to deterministic log-likelihoods. Despite its simplicity, the 'emulated noise' approach is statistically similar to IBS, as IBS estimates are unbiased, normally-distributed, and with near-constant variance across the parameter space [41].

# 4.1 Benchmark problems

The benchmark problem set consists of a common test simulation model (the Ricker model [3]) and five models with real data from various branches of computational and cognitive neuroscience. Some models are applied to multiple datasets, for a total of nine inference problems with $3 \leq D \leq 9$ parameters. Each problem provides a target noisy log-likelihood, and for simplicity we assume a uniform prior over a bounded interval for each parameter. For the purpose of this benchmark, we chose tractable models so that we could compute ground-truth posteriors and model evidence via extensive MCMC sampling. We now briefly describe each model; see Supplement for more details.

Ricker The Ricker model is a classic population model used in computational ecology [3]. The population size $N_{t}$ evolves according to a discrete-time stochastic process $N_{t+1}=r N_{t} \exp \left(-N_{t}+\varepsilon_{t}\right)$, for $t=1, \ldots, T$, with $\varepsilon_{t} \stackrel{\text { i.i.d. }}{\sim} \mathcal{N}\left(0, \sigma_{\varepsilon}^{2}\right)$ and $N_{0}=1$. At each time point, we have access to a noisy measurement $z_{t}$ of the population size $N_{t}$ with Poisson observation model $z_{t} \sim \operatorname{Poisson}\left(\phi N_{t}\right)$. The model parameters are $\boldsymbol{\theta}=\left(\log (r), \phi, \sigma_{\varepsilon}\right)$. We generate a dataset of observations $\boldsymbol{z}=\left(z_{t}\right)_{t=1}^{T}$ using the "true" parameter vector $\boldsymbol{\theta}_{\text {ime }}=(3.8,10,0.3)$ with $T=50$, as in [10]. We estimate the log-likelihood via the log-SL approach using the same 13 summary statistics as in [3, 4, 10, 25], with $N_{\text {sim }}=100$ simulations per evaluation, which yields $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\text {MAP }}\right) \approx 1.3$, where $\boldsymbol{\theta}_{\text {MAP }}$ is the maximum-a-posteriori (MAP) parameter estimate found via optimization.

Attentional drift-diffusion model (aDDM) The attentional drift-diffusion model (aDDM) is a seminal model for value-based decision making between two items with ratings $r_{\mathrm{A}}$ and $r_{\mathrm{B}}$ [43]. At each time step $t$, the decision variable $z_{t}$ is assumed to follow a stochastic diffusion process

$$
z_{0}=0, \quad z_{t+\delta t}=z_{t}+d\left(\beta^{a_{t}} r_{\mathrm{A}}-\beta^{\left(1-a_{t}\right)} r_{\mathrm{B}}\right) \delta t+\varepsilon_{t}, \quad \varepsilon_{t} \stackrel{\text { i.i.d. }}{\sim} \mathcal{N}\left(0, \sigma_{\varepsilon}^{2} \delta t\right)
$$

where $\varepsilon_{t}$ is the diffusion noise; $d$ is the drift rate; $\beta \in[0,1]$ is the attentional bias factor; and $a_{t}=1$ (resp., $a_{t}=0$ ) if the subject is fixating item A (resp., item B) at time $t$. Diffusion continues until the decision variable hits the boundary $\left|z_{t}\right| \geq 1$, which induces a choice (A for +1 , B for -1 ). We include a lapse probability $\lambda$ of a random choice at a uniformly random time over the maximum trial duration, and set $\delta t=0.1 \mathrm{~s}$. The model has parameters $\boldsymbol{\theta}=\left(d, \beta, \sigma_{\varepsilon}, \lambda\right)$. We fit choices and reaction times of two subjects (S1 and S2) from [43] using IBS with $N_{\text {rep }}=500$, which produces $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\text {MAP }}\right) \approx 2.8$.

Bayesian timing We consider a popular model of Bayesian time perception [44, 45]. In each trial of a sensorimotor timing task, human subjects had to reproduce the time interval $\tau$ between a click and a flash, with $\tau \sim$ Uniform[0.6, 0.975] s [45]. We assume subjects had only access to a noisy sensory measurement $t_{\mathrm{s}} \sim \mathcal{N}\left(\tau, w_{\mathrm{s}}^{2} \tau^{2}\right)$, and their reproduced time $t_{\mathrm{m}}$ was affected by motor noise, $t_{\mathrm{m}} \sim \mathcal{N}\left(\tau_{\star}, w_{\mathrm{m}}^{2} \tau_{\star}^{2}\right)$, where $w_{\mathrm{s}}$ and $w_{\mathrm{m}}$ are Weber's fractions. We assume subjects estimated $\tau_{\star}$ by combining their sensory likelihood with an approximate Gaussian prior over time intervals, $\mathcal{N}\left(\tau ; \mu_{\mathrm{p}}, \sigma_{\mathrm{p}}^{2}\right)$, and took the mean of the resulting Bayesian posterior. For each trial we also consider a probability $\lambda$ of a 'lapse' (e.g., a misclick) producing a response $t_{\mathrm{m}} \sim$ Uniform[0, 2] s. Model parameters are $\boldsymbol{\theta}=\left(w_{\mathrm{s}}, w_{\mathrm{m}}, \mu_{\mathrm{p}}, \sigma_{\mathrm{p}}, \lambda\right)$. We fit timing responses (discretized with $\delta t_{\mathrm{m}}=0.02 \mathrm{~s}$ ) of a representative subject from [45] using IBS with $N_{\text {rep }}=500$, which yields $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\text {MAP }}\right) \approx 2.2$.

---

#### Page 7

Multisensory causal inference (CI) Causal inference (CI) in multisensory perception denotes the problem the brain faces when deciding whether distinct sensory cues come from the same source [46]. We model a visuo-vestibular CI experiment in which human subjects, sitting in a moving chair, were asked in each trial whether the direction of movement $s_{\text {vest }}$ matched the direction $s_{\text {vis }}$ of a looming visual field [47]. We assume subjects only have access to noisy sensory measurements $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right), z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}$ is the vestibular noise and $\sigma_{\text {vis }}(c)$ is the visual noise, with $c \in\left\{c_{\text {low }}, c_{\text {med }}, c_{\text {high }}\right\}$ distinct levels of visual coherence adopted in the experiment. We model subjects' responses with a heuristic 'Fixed' rule that judges the source to be the same if $\left|z_{\text {vis }}-z_{\text {vest }}\right|<\kappa$, plus a probability $\lambda$ of giving a random response (lapse) [47]. Model parameters are $\boldsymbol{\theta}=\left(\sigma_{\text {vest }}, \sigma_{\text {vis }}\left(c_{\text {low }}\right), \sigma_{\text {vis }}\left(c_{\text {med }}\right), \sigma_{\text {vis }}\left(c_{\text {high }}\right), \kappa, \lambda\right)$. We fit datasets from two subjects (S1 and S2) from [47] using IBS with $N_{\text {rep }}=200$ repeats, which yields $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\text {MAP }}\right) \approx 1.3$ for both datasets.
Neuronal selectivity We consider a computational model of neuronal orientation selectivity in visual cortex [48] used in previous optimization and inference benchmarks [5, 6, 26]. It is a linear-nonlinear-linear-nonlinear (LN-LN) cascade model which combines effects of filtering, suppression, and response nonlinearity whose output drives the firing rate of an inhomogeneous Poisson process (details in [48]). The restricted model has $D=7$ free parameters which determine features such as the neuron's preferred direction of motion and spatial frequency. We fit the neural recordings of one V1 and one V2 cell from [48]. For the purpose of this 'noisy' benchmark, we compute the log-likelihood exactly and add i.i.d. Gaussian noise to each log-likelihood evaluation with $\sigma_{\text {obs }}(\boldsymbol{\theta})=2$.
Rodent 2AFC We consider a sensory-history-dependent model of rodent decision making in a two-alternative forced choice (2AFC) task. In each trial, rats had to discriminate the amplitudes $s_{\mathrm{L}}$ and $s_{\mathrm{R}}$ of auditory tones presented, respectively, left and right [49, 50]. The rodent's choice probability is modeled as $P($ Left $)=\lambda / 2+(1-\lambda) /\left(1+e^{-\lambda}\right)$ where $\lambda$ is a lapse probability and

$$
A=w_{0}+w_{\mathrm{c}} b_{\mathrm{c}}^{(-1)}+w_{\bar{x}} \bar{s}+\sum_{t=0}^{q}\left(w_{\mathrm{L}}^{(-t)} s_{\mathrm{L}}^{(-t)}+w_{\mathrm{R}}^{(-t)} s_{\mathrm{R}}^{(-t)}\right)
$$

where $w_{\mathrm{L}}^{(-t)}$ and $w_{\mathrm{R}}^{(-t)}$ are coefficients of the $s_{\mathrm{L}}$ and $s_{\mathrm{R}}$ regressors, respectively, from $t$ trials back; $b_{\mathrm{c}}^{(-1)}$ is the correct side on the previous trial $(\mathrm{L}=+1, \mathrm{R}=-1$ ), used to capture the win-stay/loseswitch strategy; $\bar{s}$ is a long-term history regressor (an exponentially-weighted running mean of past stimuli with time constant $\tau$ ); and $w_{0}$ is the bias. This choice of regressors best described rodents' behavior in the task [49]. We fix $\lambda=0.02$ and $\tau=20$ trials, thus leaving $D=9$ free parameters $\boldsymbol{\theta}=\left(w_{0}, w_{\mathrm{c}}, w_{\bar{x}}, \boldsymbol{w}_{\mathrm{L}}^{(0,-1,-2)}, \boldsymbol{w}_{\mathrm{R}}^{(0,-1,-2)}\right)$. We fit $10^{4}$ trials from a representative subject dataset [50] using IBS with $N_{\text {rep }}=500$, which produces $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\text {MAP }}\right) \approx 3.18$.

# 4.2 Results

To assess the model evidence approximation, Fig. 2 shows the absolute difference between true and estimated log marginal likelihood ('LML loss'), using the ELBO as a proxy for VBMC. Differences in LML of 10+ points are often considered 'decisive evidence' in a model comparison [51], while differences $\ll 1$ are negligible; so for practical usability of a method we aim for a LML loss $<1$.
As a measure of loss to judge the quality of the posterior approximation, Fig. 3 shows the mean marginal total variation distance (MMTV) between approximate posterior and ground truth. Given two pdfs $p$ and $q$, we define MMTV $(p, q)=\frac{1}{2 D} \sum_{i=1}^{D} \int\left|p_{i}\left(x_{i}\right)-q_{i}\left(x_{i}\right)\right| d x_{i}$, where $p_{i}$ and $q_{i}$ denote the marginal densities along the $i$-th dimension. Since the MMTV only looks at differences in the marginals, we also examined the "Gaussianized" symmetrized Kullback-Leibler divergence (gsKL), a metric sensitive to differences in mean and covariance [5]. We found that MMTV and gsKL follow qualitatively similar trends, so we show the latter in the Supplement.
First, our results confirm that, in the presence of noisy log-likelihoods, methods that use 'global' acquisition functions largely outperform methods that use pointwise estimates of uncertainty, as noted in [10]. In particular, 'uncertainty sampling' acquisition functions are unusable with VBMC in the presence of noise, exemplified here by the poor performance of VBMC-NPRO (see also Supplement for further tests). WSABI shows the worst performance here due to a GP representation (the square root transform) which interacts badly with noise on the log-likelihood. Previous state-of-the art method GP-fmiQR performs well with a simple synthetic problem (Ricker), but fails on complex scenarios such as Rodent 2AFC, Neuronal selectivity, or Bayesian timing, likely due to excessive exploration

---

#### Page 8

> **Image description.** This image contains a series of line graphs comparing the performance of different algorithms on various problems. Each graph shows the "LML loss" (Log Marginal Likelihood loss) on the y-axis versus "Function evaluations" on the x-axis. The y-axis uses a logarithmic scale, ranging from 0.1 to 10^3. The x-axis ranges from 0 to a maximum value between 200 and 400. Each graph is labeled with a problem name (e.g., "Ricker", "aDDM (S1)", "Timing", "Multisensory (S1)", "Neuronal (V1)", "Rodent") and a "D =" value indicating dimensionality.
>
> The graphs display the performance of five algorithms, each represented by a different colored line:
>
> - "wsabi" (red line with shaded area)
> - "vbmc-npro" (green line)
> - "vbmc-eig" (light blue dashed line)
> - "vbmc-imiqr" (dark blue dotted line)
> - "vbmc-viqr" (black solid line)
>
> A horizontal dashed line is present at y = 1 on each graph, presumably indicating a desirable error threshold. The shaded areas around the "wsabi" lines represent confidence intervals.
>
> The graphs are arranged in two rows. The first row contains "Ricker", "aDDM (S1)", "aDDM (S2)", and "Timing". The second row contains "Multisensory (S1)", "Multisensory (S2)", "Neuronal (V1)", "Neuronal (V2)", and "Rodent".

Figure 2: Model evidence loss. Median absolute error of the log marginal likelihood (LML) estimate with respect to ground truth, as a function of number of likelihood evaluations, on different problems. A desirable error is below 1 (dashed line). Shaded areas are $95 \%$ CI of the median across 100 runs.

> **Image description.** This image contains a series of line graphs comparing the performance of different algorithms on various problems. Each graph shows the Median Mean Total Variation (MMTV) on the y-axis versus the number of function evaluations on the x-axis. The graphs are arranged in a grid of two rows and five columns.
>
> Here's a breakdown of the common elements and individual graphs:
>
> - **Common Elements:**
>
>   - **Axes:** Each graph has a y-axis labeled "MMTV" ranging from 0 to 1, and an x-axis labeled "Function evaluations". The x-axis ranges vary between graphs, from 0-200/300 to 0-400.
>   - **Horizontal Dashed Line:** A horizontal dashed line is present at MMTV = 0.2 on each graph.
>   - **Lines:** Each graph contains multiple lines representing different algorithms. The algorithms are: gp-imiqr (green), vbmc-npro (light green), vbmc-eig (light blue), vbmc-imiqr (dotted dark blue), and vbmc-viqr (black). Shaded areas around the gp-imiqr and vbmc-npro lines represent confidence intervals.
>   - **Titles:** Each graph has a title indicating the problem being addressed, such as "Ricker", "aDDM (S1)", "aDDM (S2)", "Timing", "Multisensory (S1)", "Multisensory (S2)", "Neuronal (V1)", "Neuronal (V2)", and "Rodent".
>   - **D Value:** Each graph has a "D = [number]" annotation indicating the dimensionality of the problem.
>
> - **Individual Graphs (from left to right, top to bottom):**
>
>   1.  **Ricker:** D = 3.
>   2.  **aDDM (S1):** D = 4.
>   3.  **aDDM (S2):** D = 4.
>   4.  **Timing:** D = 5.
>   5.  **Multisensory (S1):** D = 6.
>   6.  **Multisensory (S2):** D = 6.
>   7.  **Neuronal (V1):** D = 7.
>   8.  **Neuronal (V2):** D = 7.
>   9.  **Rodent:** D = 9.
>
> The lines in each graph generally show a decreasing trend, indicating that the MMTV decreases (and thus the algorithm's performance improves) as the number of function evaluations increases. The algorithms' relative performance varies across the different problems.

Figure 3: Posterior estimation loss (MMTV). Median mean marginal total variation distance (MMTV) between the algorithm's posterior and ground truth, as a function of number of likelihood evaluations. A desirable target (dashed line) is less than 0.2 , corresponding to more than $80 \%$ overlap between true and approximate posterior marginals (on average across model parameters).

(see Supplement). VBMC-EIG performs reasonably well on most problems, but also struggles on Rodent 2AFC and Bayesian timing. Overall, VBMC-IMIQR and VBMC-VIQR systematically show the best and most robust performance, with VBMC-VIQR marginally better on most problems, except Rodent 2AFC. Both achieve good approximations of the model evidence and of the true posteriors within the limited budget (see Supplement for comparisons with ground-truth posteriors).

Table 1 compares the average algorithmic overhead of methods based on $a_{\text {IMIQR }}$ and $a_{\text {VIQR }}$, showing the computational advantage of the variational approach of VBMC-VIQR.

Then, we looked at how robust different methods are to different degrees of log-likelihood noise. We considered three benchmark problems for which we could easily compute the log-likelihood exactly. For each problem, we emulated different levels of noise by adding Gaussian observation noise to

---

#### Page 9

Table 1: Average algorithmic overhead per likelihood evaluation (in seconds) over a full run, assessed on a single-core reference machine (mean $\pm 1$ SD across 100 runs).

| Algorithm  |           Model            |                            |                            |                            |                            |                            |
| :--------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: |
|            |           Ricker           |            aDDM            |           Timing           |        Multisensory        |          Neuronal          |           Rodent           |
| VBMC-VIQR  | $\mathbf{1 . 5 \pm 0 . 1}$ | $\mathbf{1 . 5 \pm 0 . 1}$ | $\mathbf{1 . 8 \pm 0 . 2}$ | $\mathbf{2 . 0 \pm 0 . 2}$ | $\mathbf{2 . 8 \pm 0 . 8}$ | $\mathbf{2 . 6 \pm 0 . 2}$ |
| VBMC-IMIQR |       $5.5 \pm 0.5$        |       $5.1 \pm 0.3$        |       $5.8 \pm 0.6$        |       $5.6 \pm 0.3$        |       $6.5 \pm 1.3$        |       $5.6 \pm 0.4$        |
|  GP-IMIQR  |       $15.6 \pm 0.9$       |       $16.0 \pm 1.7$       |       $17.1 \pm 1.2$       |       $26.3 \pm 1.8$       |       $29.6 \pm 2.8$       |       $40.1 \pm 2.1$       |

exact log-likelihood evaluations, with $\sigma_{\text {obs }} \in[0,7]$ (see Fig. 4). Most algorithms only perform well with no or very little noise, whereas the performance of VBMC-VIQR (and, similarly, VBMC-IMIQR) degrades gradually with increasing noise. For these two algorithms, acceptable results can be reached for $\sigma_{\text {obs }}$ as high as $\approx 7$, although for best results even with hard problems we would recommend $\sigma_{\text {obs }} \lesssim 3$. We see that the Neuronal problem is particularly hard, with both WSABI and GP-IMIQR failing to converge altogether even in the absence of noise.

> **Image description.** This image contains two panels, A and B, each displaying three line graphs. Each graph plots the performance of different algorithms against varying levels of log-likelihood noise.
>
> **Panel A:**
>
> - **Type:** Three line graphs showing "LML loss" (Log Marginal Likelihood loss) on the y-axis versus "Log-likelihood noise $\sigma_{obs}$" on the x-axis.
> - **Arrangement:** Three graphs are arranged horizontally, labeled "aDDM (S1)", "Timing", and "Neuronal (V1)" from left to right.
> - **Axes:** The y-axis (LML loss) is on a logarithmic scale, ranging from 0.01 to 10^3. The x-axis (Log-likelihood noise) ranges from 0 to 7, with tick marks at 0, 1, 2, 3.5, 5, and 7.
> - **Data:** Each graph contains multiple lines, each representing a different algorithm:
>   - wsabi (red, dashed)
>   - gp-imiqr (green, solid)
>   - vbmc-npro (light green, solid)
>   - vbmc-eig (light blue, dashed)
>   - vbmc-imiqr (dark blue, dotted)
>   - vbmc-viqr (black, solid)
>     Each line is surrounded by a shaded area representing the confidence interval. A horizontal dashed line is present at y=1.
>
> **Panel B:**
>
> - **Type:** Three line graphs showing "MMTV" (Mean Marginal Total Variation) on the y-axis versus "Log-likelihood noise $\sigma_{obs}$" on the x-axis.
> - **Arrangement:** Similar to Panel A, three graphs are arranged horizontally, labeled "aDDM (S1)", "Timing", and "Neuronal (V1)" from left to right.
> - **Axes:** The y-axis (MMTV) ranges from 0 to 1. The x-axis (Log-likelihood noise) ranges from 0 to 7, with tick marks at 0, 1, 2, 3.5, 5, and 7.
> - **Data:** Each graph contains multiple lines, each representing a different algorithm, using the same color scheme as Panel A. Each line is surrounded by a shaded area representing the confidence interval. A horizontal dashed line is present at y=0.2.

Figure 4: Noise sensitivity. Final performance metrics of all algorithms with respect to ground truth, as a function of log-likelihood observation noise $\sigma_{\text {obs }}$, for different problems. For all metrics, we plot the median after $50 \times(D+2)$ log-likelihood evaluations, and shaded areas are $95 \%$ CI of the median across 100 runs. A. Absolute error of the log marginal likelihood (LML) estimate. B. Mean marginal total variation distance (MMTV).

Lastly, we tested how robust VBMC-VIQR is to imprecise estimates of the observation noise, $\widehat{\sigma}_{\text {obs }}(\boldsymbol{\theta})$. We reran VBMC-VIQR on the three problems of Fig. 4 while drawing $\widehat{\sigma}_{\text {obs }} \sim$ Lognormal $\left(\ln \sigma_{\text {obs }}, \sigma_{\sigma}^{2}\right)$ for increasing values of noise-of-estimating-noise, $\sigma_{\sigma} \geq 0$. We found that at worst the performance of VBMC degrades only by $\sim 25 \%$ with $\sigma_{\sigma}$ up to 0.4 (i.e., $\widehat{\sigma}_{\text {obs }}$ roughly between $0.5-2.2$ times the true value); showing that VBMC is robust to imprecise noise estimates (see Supplement for details).

# 5 Conclusions

In this paper, we addressed the problem of approximate Bayesian inference with only a limited budget of noisy log-likelihood evaluations. For this purpose, we extended the VBMC framework to work in the presence of noise by testing several new acquisition functions and by introducing variational whitening for a more accurate posterior approximation. We showed that with these new features VBMC achieves state-of-the-art inference performance on a novel challenging benchmark that uses a variety of models and real data sets from computational and cognitive neuroscience, covering areas such as neuronal modeling, human and rodent psychophysics, and value-based decision-making.
Our benchmark also revealed that common synthetic test problems, such as the Ricker and g-and-k models (see Supplement for the latter), may be too simple for surrogate-based methods, as good performance on these problems (e.g., GP-IMIQR) may not generalize to real models and datasets.
In conclusion, our extensive analyses show that VBMC with the $a_{\text {VIQR }}$ acquisition function is very effective for approximate Bayesian inference with noisy log-likelihoods, with up to $\sigma_{\text {obs }} \approx 3$, and models up to $D \lesssim 10$ and whose evaluation take about a few seconds or more. Future work should focus on improving the flexibility of the GP representation, scaling the method to higher dimensions, and investigating theoretical guarantees for the VBMC algorithm.

---

# Variational Bayesian Monte Carlo with Noisy Likelihoods - Backmatter

---

#### Page 10

# Broader Impact 

We believe this work has the potential to lead to net-positive improvements in the research community and more broadly in society at large. First, this paper makes Bayesian inference accessible to noncheap models with noisy log-likelihoods, allowing more researchers to express uncertainty about their models and model parameters of interest in a principled way; with all the advantages of proper uncertainty quantification [2]. Second, with the energy consumption of computing facilities growing incessantly every hour, it is our duty towards the environment to look for ways to reduce the carbon footprint of our algorithms [52]. In particular, traditional methods for approximate Bayesian inference can be extremely sample-inefficient. The 'smart' sample-efficiency of VBMC can save a considerable amount of resources when model evaluations are computationally expensive.
Failures of VBMC can return largely incorrect posteriors and values of the model evidence, which if taken at face value could lead to wrong conclusions. This failure mode is not unique to VBMC, but a common problem of all approximate inference techniques (e.g., MCMC or variational inference [2,53]). VBMC returns uncertainty on its estimate and comes with a set of diagnostic functions which can help identify issues. Still, we recommend the user to follow standard good practices for validation of results, such as posterior predictive checks, or comparing results from different runs.
Finally, in terms of ethical aspects, our method - like any general, black-box inference technique - will reflect (or amplify) the explicit and implicit biases present in the models and in the data, especially with insufficient data [54]. Thus, we encourage researchers in potentially sensitive domains to explicitly think about ethical issues and consequences of the models and data they are using.

## Acknowledgments and Disclosure of Funding

We thank Ian Krajbich for sharing data for the aDDM model; Robbe Goris for sharing data and code for the neuronal model; Marko Järvenpää and Alexandra Gessner for useful discussions about their respective work; Nisheet Patel for helpful comments on an earlier version of this manuscript; and the anonymous reviewers for constructive remarks. This work has utilized the NYU IT High Performance Computing resources and services. This work was partially supported by the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence (FCAI).

## References

[1] MacKay, D. J. (2003) Information theory, inference and learning algorithms. (Cambridge University Press).
[2] Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., \& Rubin, D. B. (2013) Bayesian Data Analysis (3rd edition). (CRC Press).
[3] Wood, S. N. (2010) Statistical inference for noisy nonlinear ecological dynamic systems. Nature 466, $1102-1104$.
[4] Price, L. F., Drovandi, C. C., Lee, A., \& Nott, D. J. (2018) Bayesian synthetic likelihood. Journal of Computational and Graphical Statistics 27, 1-11.
[5] Acerbi, L. (2018) Variational Bayesian Monte Carlo. Advances in Neural Information Processing Systems 31, 8222-8232.
[6] Acerbi, L. (2019) An exploration of acquisition and mean functions in Variational Bayesian Monte Carlo. Proceedings of The 1st Symposium on Advances in Approximate Bayesian Inference (PMLR) 96, 1-10.
[7] Rasmussen, C. \& Williams, C. K. I. (2006) Gaussian Processes for Machine Learning. (MIT Press).
[8] O’Hagan, A. (1991) Bayes-Hermite quadrature. Journal of Statistical Planning and Inference 29, 245-260.
[9] Ghahramani, Z. \& Rasmussen, C. E. (2002) Bayesian Monte Carlo. Advances in Neural Information Processing Systems 15, 505-512.
[10] Järvenpää, M., Gutmann, M. U., Vehtari, A., Marttinen, P., et al. (2020) Parallel Gaussian process surrogate Bayesian inference with noisy likelihood evaluations. Bayesian Analysis.
[11] Gessner, A., Gonzalez, J., \& Mahsereci, M. (2019) Active multi-information source Bayesian quadrature. Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial Intelligence (UAI 2019) p. 245.

---

#### Page 11

[12] Järvenpää, M., Gutmann, M. U., Vehtari, A., Marttinen, P., et al. (2018) Gaussian process modelling in approximate Bayesian computation to estimate horizontal gene transfer in bacteria. The Annals of Applied Statistics 12, 2228-2251.
[13] Järvenpää, M., Gutmann, M. U., Pleska, A., Vehtari, A., Marttinen, P., et al. (2019) Efficient acquisition rules for model-based approximate Bayesian computation. Bayesian Analysis 14, 595-622.
[14] Rasmussen, C. E. (2003) Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. Bayesian Statistics 7, 651-659.
[15] Kandasamy, K., Schneider, J., \& Póczos, B. (2015) Bayesian active learning for posterior estimation. Twenty-Fourth International Joint Conference on Artificial Intelligence.
[16] Wang, H. \& Li, J. (2018) Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions. Neural Computation pp. 1-23.
[17] Osborne, M., Duvenaud, D. K., Garnett, R., Rasmussen, C. E., Roberts, S. J., \& Ghahramani, Z. (2012) Active learning of model evidence using Bayesian quadrature. Advances in Neural Information Processing Systems 25, 46-54.
[18] Gunter, T., Osborne, M. A., Garnett, R., Hennig, P., \& Roberts, S. J. (2014) Sampling for inference in probabilistic models with fast Bayesian quadrature. Advances in Neural Information Processing Systems 27, 2789-2797.
[19] Briol, F.-X., Oates, C., Girolami, M., \& Osborne, M. A. (2015) Frank-Wolfe Bayesian quadrature: Probabilistic integration with theoretical guarantees. Advances in Neural Information Processing Systems 28, 1162-1170.
[20] Chai, H., Ton, J.-F., Garnett, R., \& Osborne, M. A. (2019) Automated model selection with Bayesian quadrature. Proceedings of the 36th International Conference on Machine Learning 97, 931-940.
[21] Jones, D. R., Schonlau, M., \& Welch, W. J. (1998) Efficient global optimization of expensive black-box functions. Journal of Global Optimization 13, 455-492.
[22] Brochu, E., Cora, V. M., \& De Freitas, N. (2010) A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. arXiv preprint arXiv:1012.2599.
[23] Snoek, J., Larochelle, H., \& Adams, R. P. (2012) Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems 25, 2951-2959.
[24] Picheny, V., Ginsbourger, D., Richet, Y., \& Caplin, G. (2013) Quantile-based optimization of noisy computer experiments with tunable precision. Technometrics 55, 2-13.
[25] Gutmann, M. U. \& Corander, J. (2016) Bayesian optimization for likelihood-free inference of simulatorbased statistical models. The Journal of Machine Learning Research 17, 4256-4302.
[26] Acerbi, L. \& Ma, W. J. (2017) Practical Bayesian optimization for model fitting with Bayesian adaptive direct search. Advances in Neural Information Processing Systems 30, 1834-1844.
[27] Letham, B., Karrer, B., Ottoni, G., Bakshy, E., et al. (2019) Constrained Bayesian optimization with noisy experiments. Bayesian Analysis 14, 495-519.
[28] Papamakarios, G. \& Murray, I. (2016) Fast $\varepsilon$-free inference of simulation models with Bayesian conditional density estimation. Advances in Neural Information Processing Systems 29, 1028-1036.
[29] Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Öcal, K., Nonnenmacher, M., \& Macke, J. H. (2017) Flexible statistical inference for mechanistic models of neural dynamics. Advances in Neural Information Processing Systems 30, 1289-1299.
[30] Greenberg, D. S., Nonnenmacher, M., \& Macke, J. H. (2019) Automatic posterior transformation for likelihood-free inference. International Conference on Machine Learning pp. 2404-2414.
[31] Gonçalves, P. J., Lueckmann, J.-M., Deistler, M., Nonnenmacher, M., Öcal, K., Bassetto, G., Chintaluri, C., Podlaski, W. F., Haddad, S. A., Vogels, T. P., et al. (2019) Training deep neural density estimators to identify mechanistic models of neural dynamics. bioRxiv p. 838383.
[32] Gramacy, R. B. \& Lee, H. K. (2012) Cases for the nugget in modeling computer experiments. Statistics and Computing 22, 713-722.
[33] Neal, R. M. (2003) Slice sampling. Annals of Statistics 31, 705-741.
[34] Kingma, D. P. \& Welling, M. (2013) Auto-encoding variational Bayes. Proceedings of the 2nd International Conference on Learning Representations.
[35] Miller, A. C., Foti, N., \& Adams, R. P. (2017) Variational boosting: Iteratively refining posterior approximations. Proceedings of the 34th International Conference on Machine Learning 70, 2420-2429.
[36] Kingma, D. P. \& Ba, J. (2014) Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations.

---

#### Page 12

[37] Kanagawa, M. \& Hennig, P. (2019) Convergence guarantees for adaptive Bayesian quadrature methods. Advances in Neural Information Processing Systems 32, 6234-6245.
[38] Carpenter, B., Gelman, A., Hoffman, M., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M. A., Guo, J., Li, P., \& Riddell, A. (2016) Stan: A probabilistic programming language. Journal of Statistical Software 20.
[39] Ankenman, B., Nelson, B. L., \& Staum, J. (2010) Stochastic kriging for simulation metamodeling. Operations Research 58, 371-382.
[40] Haldane, J. (1945) On a method of estimating frequencies. Biometrika 33, 222-225.
[41] van Opheusden, B., Acerbi, L., \& Ma, W. J. (2020) Unbiased and efficient log-likelihood estimation with inverse binomial sampling. arXiv preprint arXiv:2001.03985.
[42] Lyu, X., Binois, M., \& Ludkovski, M. (2018) Evaluating Gaussian process metamodels and sequential designs for noisy level set estimation. arXiv preprint arXiv:1807.06712.
[43] Krajbich, I., Armel, C., \& Rangel, A. (2010) Visual fixations and the computation and comparison of value in simple choice. Nature Neuroscience 13, 1292.
[44] Jazayeri, M. \& Shadlen, M. N. (2010) Temporal context calibrates interval timing. Nature Neuroscience 13, 1020-1026.
[45] Acerbi, L., Wolpert, D. M., \& Vijayakumar, S. (2012) Internal representations of temporal statistics and feedback calibrate motor-sensory interval timing. PLoS Computational Biology 8, e1002771.
[46] Körding, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., \& Shams, L. (2007) Causal inference in multisensory perception. PLoS One 2, e943.
[47] Acerbi, L., Dokka, K., Angelaki, D. E., \& Ma, W. J. (2018) Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS Computational Biology 14, e1006110.
[48] Goris, R. L., Simoncelli, E. P., \& Movshon, J. A. (2015) Origin and function of tuning diversity in macaque visual cortex. Neuron 88, 819-831.
[49] Akrami, A., Kopec, C. D., Diamond, M. E., \& Brody, C. D. (2018) Posterior parietal cortex represents sensory history and mediates its effects on behaviour. Nature 554, 368-372.
[50] Roy, N. A., Bak, J. H., Akrami, A., Brody, C., \& Pillow, J. W. (2018) Efficient inference for time-varying behavior during learning. Advances in Neural Information Processing Systems 31, 5695-5705.
[51] Kass, R. E. \& Raftery, A. E. (1995) Bayes factors. Journal of the American Statistical Association 90, $773-795$.
[52] Strubell, E., Ganesh, A., \& McCallum, A. (2019) Energy and policy considerations for deep learning in NLP. Annual Meeting of the Association for Computational Linguistics.
[53] Yao, Y., Vehtari, A., Simpson, D., \& Gelman, A. (2018) Yes, but did it work?: Evaluating variational inference. Proceedings of the 35th International Conference on Machine Learning 80, 5581-5590.
[54] Chen, I., Johansson, F. D., \& Sontag, D. (2018) Why is my classifier discriminatory? Advances in Neural Information Processing Systems 31, 3539-3550.
[55] Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., \& Saul, L. K. (1999) An introduction to variational methods for graphical models. Machine Learning 37, 183-233.
[56] Bishop, C. M. (2006) Pattern Recognition and Machine Learning. (Springer).
[57] Knuth, D. E. (1992) Two notes on notation. The American Mathematical Monthly 99, 403-422.
[58] Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., \& Riddell, A. (2017) Stan: A probabilistic programming language. Journal of Statistical Software 76.
[59] Haario, H., Laine, M., Mira, A., \& Saksman, E. (2006) Dram: Efficient adaptive MCMC. Statistics and Computing 16, 339-354.
[60] Blei, D. M., Kucukelbir, A., \& McAuliffe, J. D. (2017) Variational inference: A review for statisticians. Journal of the American Statistical Association 112, 859-877.
[61] Robert, C. P., Cornuet, J.-M., Marin, J.-M., \& Pillai, N. S. (2011) Lack of confidence in approximate Bayesian computation model choice. Proceedings of the National Academy of Sciences 108, 15112-15117.

---

# Variational Bayesian Monte Carlo with Noisy Likelihoods - Appendix

---

#### Page 13

# Supplementary Material

In this Supplement we include a number of derivations, implementation details, and additional results omitted from the main text.

Code used to generate the results and figures in the paper is available at https://github.com/lacerbi/infbench. The VBMC algorithm with added support for noisy models is available at https://github.com/acerbilab/vbmc.

## Contents

A Background information ..... 13
A. 1 Variational inference ..... 14
A. 2 Gaussian processes ..... 14
A. 3 Adaptive Bayesian quadrature ..... 15
B Algorithmic details ..... 15
B. 1 Modified VBMC features ..... 15
B. 2 Variational whitening ..... 17
C Acquisition functions ..... 17
C. 1 Observation noise ..... 17
C. 2 Expected information gain (EIG) ..... 17
C. 3 Integrated median / variational interquantile range (IMIQR/ VIQR) ..... 18
D Benchmark details ..... 19
D. 1 Problem specification ..... 19
D. 2 Algorithm specification ..... 19
D. 3 Computing infrastructure ..... 21
E Additional results ..... 21
E. 1 Gaussianized symmetrized KL divergence (gsKL) metric ..... 21
E. 2 Worse-case analysis ( $90 \%$ quantile) ..... 22
E. 3 Ablation study ..... 22
E. 4 Comparison of true and approximate posteriors ..... 24
E. 5 Sensitivity to imprecise noise estimates ..... 24
E. 6 g-and-k model ..... 25

## A Background information

For ease of reference, in this Section we recap the three key theoretical ingredients used to build the Variational Bayesian Monte Carlo (VBMC) framework, that is variational inference, Gaussian processes and adaptive Bayesian quadrature. The material presented here is largely based and expands on the "theoretical background" section of [5].

---

#### Page 14

# A. 1 Variational inference

Let $\boldsymbol{\theta} \in \mathcal{X} \subseteq \mathbb{R}^{D}$ be a parameter vector of a model of interest, and $\mathcal{D}$ a dataset. Variational inference is an approximate inference framework in which an intractable posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$ is approximated by a simpler distribution $q(\boldsymbol{\theta}) \equiv q_{\phi}(\boldsymbol{\theta})$ that belongs to a parametric family indexed by parameter vector $\phi$, such as a multivariate normal or a mixture of Gaussians $[55,56]$. Thus, the goal of variational inference is to find $\phi$ for which the variational posterior $q_{\phi}$ is "closest" in approximation to the true posterior, according to some measure of discrepancy.
In variational Bayes, the discrepancy between approximate and true posterior is quantified by the Kullback-Leibler (KL) divergence,

$$
D_{\mathrm{KL}}\left[q_{\phi}(\boldsymbol{\theta}) \| p(\boldsymbol{\theta} \mid \mathcal{D})\right]=\mathbb{E}_{\phi}\left[\log \frac{q_{\phi}(\boldsymbol{\theta})}{p(\boldsymbol{\theta} \mid \mathcal{D})}\right]
$$

where we adopted the compact notation $\mathbb{E}_{\phi} \equiv \mathbb{E}_{q_{\phi}}$. Crucially, $D_{\mathrm{KL}}(q \| p) \geq 0$ and the equality is achieved if and only if $q \equiv p . D_{\mathrm{KL}}$ is not symmetric, and the specific choice of using $D_{\mathrm{KL}}[q \| p]$ (reverse $D_{\mathrm{KL}}$ ) as opposed to $D_{\mathrm{KL}}[p \| q]$ (forward $D_{\mathrm{KL}}$ ) is a key feature of the variational framework.
The variational approach casts Bayesian inference as an optimization problem, which consists of finding the variational parameter vector $\phi$ that minimizes Eq. S1. We can rewrite Eq. S1 as

$$
\log p(\mathcal{D})=D_{\mathrm{KL}}\left[q_{\phi}(\boldsymbol{\theta}) \| p(\boldsymbol{\theta} \mid \mathcal{D})\right]+\mathcal{F}\left[q_{\phi}\right]
$$

where on the left-hand side we have the model evidence, and on the right-hand side the KL divergence plus the negative free energy, defined as

$$
\mathcal{F}\left[q_{\phi}\right]=\mathbb{E}_{\phi}\left[\log \frac{p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})}{q_{\phi}(\boldsymbol{\theta})}\right]=\mathbb{E}_{\phi}[f(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]
$$

with $f(\boldsymbol{\theta}) \equiv \log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})=\log p(\mathcal{D}, \boldsymbol{\theta})$ the log joint probability, and $\mathcal{H}[q]$ the entropy of $q$. Now, since as mentioned above the KL divergence is a non-negative quantity, from Eq. S2 we have $\mathcal{F}[q] \leq \log p(\mathcal{D})$, with equality holding if $q(\boldsymbol{\theta}) \equiv p(\boldsymbol{\theta} \mid \mathcal{D})$. For this reason, Eq. S3 is known as the evidence lower bound (ELBO), so called because it is a lower bound to the log marginal likelihood or model evidence. Importantly, maximization of the variational objective, Eq. S3, is equivalent to minimization of the KL divergence, and produces both an approximation of the posterior $q_{\phi}$ and the ELBO, which can be used as a metric for model selection.
Classically, $q$ is chosen to belong to a family (e.g., a factorized posterior, or mean field) such that both the expected $\log$ joint in Eq. S3 and the entropy afford analytical solutions, which are then used to yield closed-form equations for a coordinate ascent algorithm. In the VBMC framework, instead, $f(\boldsymbol{\theta})$ is assumed to be a potentially expensive black-box function, which prevents a direct computation of Eq. S3 analytically or via simple numerical integration.

## A. 2 Gaussian processes

Gaussian processes (GPs) are a flexible class of statistical models for specifying prior distributions over unknown functions $f: \mathcal{X} \subseteq \mathbb{R}^{D} \rightarrow \mathbb{R}$ [7]. GPs are defined by a mean function $m: \mathcal{X} \rightarrow \mathbb{R}$ and a positive definite covariance, or kernel function $\kappa: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$. VBMC uses the common squared exponential (rescaled Gaussian) kernel,

$$
\kappa\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\prime}\right)=\sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\theta}^{\prime}, \boldsymbol{\Sigma}_{\ell}\right) \quad \text { with } \boldsymbol{\Sigma}_{\ell}=\operatorname{diag}\left[\ell^{(1)^{2}}, \ldots, \ell^{(D)^{2}}\right]
$$

where $\sigma_{f}$ is the output length scale, $\ell$ is the vector of input length scales, and $\Lambda \equiv(2 \pi)^{\frac{D}{2}} \prod_{i=1}^{D} \ell^{(i)}$ is equal to the normalization factor of the Gaussian (this notation makes it easy to apply Gaussian identities used in Bayesian quadrature). As a mean function, VBMC uses a negative quadratic function to ensure well-posedness of the variational formulation, and defined as $[5,6]$

$$
m(\boldsymbol{\theta}) \equiv m_{0}-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(\theta^{(i)}-\theta_{\mathrm{m}}^{(i)}\right)^{2}}{\omega^{(i)^{2}}}
$$

where $m_{0}$ denotes the maximum, $\boldsymbol{\theta}_{\mathrm{m}}$ is the location, and $\boldsymbol{\omega}$ is a vector of length scales. Finally, GPs are also characterized by a likelihood or observation noise model, which is assumed here to be

---

#### Page 15

Gaussian with known variance $\sigma_{\text {obs }}^{2}(\boldsymbol{\theta})$ for each point in the training set (in the original formulation of VBMC, observation noise is assumed to be a small positive constant).
Conditioned on training inputs $\boldsymbol{\Theta}=\left\{\boldsymbol{\theta}_{1}, \ldots, \boldsymbol{\theta}_{N}\right\}$, observed function values $\boldsymbol{y}=f(\boldsymbol{\Theta})$ and observation noise $\sigma_{\text {obs }}^{2}(\boldsymbol{\Theta})$, the posterior GP mean and covariance are available in closed form [7],

$$
\begin{aligned}
\overline{f}_{\boldsymbol{\Xi}}(\boldsymbol{\theta}) \equiv \mathbb{E}[f(\boldsymbol{\theta}) \mid \boldsymbol{\Xi}, \boldsymbol{\psi}] & =\kappa(\boldsymbol{\theta}, \boldsymbol{\Theta})\left[\kappa(\boldsymbol{\Theta}, \boldsymbol{\Theta})+\boldsymbol{\Sigma}_{\mathrm{obs}}(\boldsymbol{\Theta})\right]^{-1}(\boldsymbol{y}-m(\boldsymbol{\Theta}))+m(\boldsymbol{\theta}) \\
C_{\boldsymbol{\Xi}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\prime}\right) \equiv \operatorname{Cov}\left[f(\boldsymbol{\theta}), f\left(\boldsymbol{\theta}^{\prime}\right) \mid \boldsymbol{\Xi}, \boldsymbol{\psi}\right] & =\kappa\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\prime}\right)-\kappa(\boldsymbol{\theta}, \boldsymbol{\Theta})\left[\kappa(\boldsymbol{\Theta}, \boldsymbol{\Theta})+\boldsymbol{\Sigma}_{\mathrm{obs}}(\boldsymbol{\Theta})\right]^{-1} \kappa\left(\boldsymbol{\Theta}, \boldsymbol{\theta}^{\prime}\right)
\end{aligned}
$$

where $\boldsymbol{\Xi}=\left\{\boldsymbol{\Theta}, \boldsymbol{y}, \boldsymbol{\sigma}_{\text {obs }}\right\}$ is the set of training function data for the GP; $\boldsymbol{\psi}$ is a hyperparameter vector for the GP mean, covariance, and likelihood; and $\boldsymbol{\Sigma}_{\text {obs }}(\boldsymbol{\Theta}) \equiv \operatorname{diag}\left[\sigma_{\text {obs }}^{2}\left(\boldsymbol{\theta}_{1}\right), \ldots, \sigma_{\text {obs }}^{2}\left(\boldsymbol{\theta}_{N}\right)\right]$ is the observation noise (diagonal) matrix.

# A. 3 Adaptive Bayesian quadrature

Bayesian quadrature, also known as cubature when dealing with multi-dimensional integrals, is a technique to obtain Bayesian estimates of intractable integrals of the form $[8,9]$

$$
Z=\int_{\mathcal{X}} f(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d \boldsymbol{\theta}
$$

where $f$ is a function of interest and $\pi$ a known probability distribution. For the purpose of VBMC, we consider the domain of integration $\mathcal{X}=\mathbb{R}^{D}$. When a GP prior is specified for $f$, since integration is a linear operator, the integral $Z$ is also a Gaussian random variable whose posterior mean and variance are [9]

$$
\mathbb{E}_{f \mid \Xi}[Z]=\int \overline{f}_{\boldsymbol{\Xi}}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d \boldsymbol{\theta}, \quad \mathbb{V}_{f \mid \Xi}[Z]=\iint C_{\boldsymbol{\Xi}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\prime}\right) \pi(\boldsymbol{\theta}) \pi\left(\boldsymbol{\theta}^{\prime}\right) d \boldsymbol{\theta} d \boldsymbol{\theta}^{\prime}
$$

Importantly, if $f$ has a Gaussian kernel and $\pi$ is a Gaussian or mixture of Gaussians (among other functional forms), the integrals in Eq. S8 have closed-form solutions.

Active sampling The point $\boldsymbol{\theta}_{*} \in \mathcal{X}$ to evaluate next to improve our estimate of the integral (Eq. S7) is chosen via a proxy optimization of a given acquisition function $a: \mathcal{X} \rightarrow \mathbb{R}$, that is $\boldsymbol{\theta}_{*}=\operatorname{argmax}_{\boldsymbol{\theta}} a(\boldsymbol{\theta})$. Previously introduced acquisition functions for Bayesian quadrature include the expected entropy, which minimizes the expected entropy of the integral after adding $\boldsymbol{\theta}_{*}$ to the training set [17], and a family of strategies under the name of uncertainty sampling, whose goal is generally to find the point with maximal (pointwise) variance of the integrand at $\boldsymbol{\theta}_{*}$ [18]. The standard acquisition function for VBMC is prospective uncertainty sampling (see main text and $[5,6]$ ). Recent work proved convergence guarantees for active-sampling Bayesian quadrature under a broad class of acquisition functions which includes various forms of uncertainty sampling [37].

## B Algorithmic details

We report here implementation details of new or improved features of the VBMC algorithm omitted from the main text.

## B. 1 Modified VBMC features

In this section, we describe minor changes to the basic VBMC framework. For implementation details of the algorithm which have remained unchanged, we refer the reader to the main text and Supplement of the original VBMC paper [5].

Reliability index In VBMC, the reliability index $r(t)$ is a metric computed at the end of each iteration $t$ and determines, among other things, the termination condition [5]. We recall that $r(t)$ is computed as the arithmetic mean of three reliability features:

1. The absolute change in mean ELBO from the previous iteration: $r_{1}(t)=$ $|\mathbb{E}[\operatorname{ELBO}(t)]-\mathbb{E}[\operatorname{ELBO}(t-1)]| / \Delta_{\mathrm{SD}}$.

---

#### Page 16

2. The uncertainty of the current ELBO: $r_{2}(t)=\sqrt{\mathrm{V}[\operatorname{ELBO}(t)]} / \Delta_{\mathrm{SD}}$.
3. The change in 'Gaussianized' symmetrized KL divergence (see Eq. S21) between the current and previous-iteration variational posterior $q_{t} \equiv q_{\phi_{t}}(\boldsymbol{\theta}): r(t)=\operatorname{gsKL}\left(q_{t} \| q_{t-1}\right) / \Delta_{\mathrm{KL}}$.

The parameters $\Delta_{\mathrm{SD}}$ and $\Delta_{\mathrm{KL}}$ are tolerance hyperparameters, chosen such that $r_{j} \lesssim 1$, with $j=$ $1,2,3$, for features that are deemed indicative of a good solution. We set $\Delta_{\mathrm{KL}}=0.01 \cdot \sqrt{D}$ as in the original VBMC paper. To account for noisy observations, we set $\Delta_{\mathrm{SD}}$ in the current iteration equal to the geometric mean between the baseline $\Delta_{\mathrm{SD}}^{\text {base }}=0.1$ (from the original VBMC paper) and the GP noise in the high-posterior density region, $\sigma_{\text {obs }}^{\text {hpd }}$, and constrain it to be in the $[0.1,1]$ range. That is,

$$
\Delta_{\mathrm{SD}}=\min \left[1, \max \left[0.1, \sqrt{\Delta_{\mathrm{SD}}^{\text {base }} \cdot \sigma_{\text {obs }}^{\text {hpd }}}\right]\right]
$$

where $\sigma_{\text {obs }}^{\text {hpd }}$ is computed as the median observation noise at the top $20 \%$ points in terms of log-posterior value in the GP training set.

Regularization of acquisition functions In VBMC, active sampling is performed by maximizing a chosen acquisition function $a: \mathcal{X} \subseteq \mathbb{R}^{D} \rightarrow[0, \infty)$, where $\mathcal{X}$ is the support of the target density (see Section C). In practice, in VBMC we maximize a regularized acquisition function

$$
a^{\mathrm{reg}}(\boldsymbol{\theta} ; a) \equiv a(\boldsymbol{\theta}) b_{\mathrm{var}}(\boldsymbol{\theta}) b_{\mathrm{bnd}}(x)
$$

where $b_{\text {var }}(\boldsymbol{\theta})$ is a GP variance regularization term introduced in [5],

$$
b_{\mathrm{var}}(\boldsymbol{\theta})=\exp \left\{-\left(\frac{V^{\mathrm{reg}}}{V_{\mathbb{R}}(\boldsymbol{\theta})}-1\right)\left\|\overline{V_{\mathbb{R}}(\boldsymbol{\theta})}<V^{\mathrm{reg}}\right\|\right\}
$$

where $V_{\mathbb{R}}(\boldsymbol{\theta})$ is the posterior latent variance of the GP, $V^{\text {reg }}$ a regularization parameter (we use $V^{\text {reg }}=10^{-4}$ ), and we denote with $[|\cdot|]$ Iverson's bracket [57], which takes value 1 if the expression inside the bracket is true, 0 otherwise. Eq. S11 penalizes the selection of points too close to an existing input, which might produce numerical issues.
The $b_{\text {bnd }}$ term is a new term that we added in this work to discard points too close to the parameter bounds, which would map to very large positive or negative values in the unbounded inference space,

$$
b_{\text {bnd }}(\boldsymbol{\theta})= \begin{cases}1 & \text { if } \tilde{\theta}^{(i)} \geq \mathrm{LB}_{\varepsilon}^{(i)} \wedge \tilde{\theta}^{(i)} \leq \mathrm{UB}_{\varepsilon}^{(i)}, \text { for all } 1 \leq i \leq D \\ 0 & \text { otherwise }\end{cases}
$$

where $\tilde{\boldsymbol{\theta}}(\boldsymbol{\theta})$ is the parameter vector remapped to the original space, and $\mathrm{LB}_{\varepsilon}^{(i)} \equiv \mathrm{LB}^{(i)}+\varepsilon\left(\mathrm{UB}^{(i)}-\right.$ $\left.\mathrm{LB}^{(i)}\right), \mathrm{UB}_{\varepsilon}^{(i)} \equiv \mathrm{UB}^{(i)}-\varepsilon\left(\mathrm{UB}^{(i)}-\mathrm{LB}^{(i)}\right)$, with $\varepsilon=10^{-5}$.

GP hyperparameters and priors The GP model in VBMC has $3 D+3$ hyperparameters, $\boldsymbol{\psi}=$ $\left(\boldsymbol{\ell}, \sigma_{f}, \bar{\sigma}_{\text {obs }}, m_{0}, \boldsymbol{\theta}_{\mathrm{m}}, \boldsymbol{\omega}\right)$. All scale hyperparameters, that is $\left\{\boldsymbol{\ell}, \sigma_{f}, \bar{\sigma}_{\text {obs }}, \boldsymbol{\omega}\right\}$, are defined in log space. Each hyperparameter has an independent prior, either bounded uniform or a truncated Student's $t$ distribution with mean $\mu$, scale $\sigma$, and $\nu=3$ degrees of freedom. GP hyperparameters and their priors are reported in Table S1.

|           Hyperparameter           | Description            |                Prior mean $\mu$                | Prior scale $\sigma$ |
| :--------------------------------: | :--------------------- | :--------------------------------------------: | :------------------: |
|         $\log \ell^{(i)}$          | Input length scale     | $\log \left[\sqrt{\frac{D}{6} L^{(i)}}\right]$ | $\log \sqrt{10^{3}}$ |
|         $\log \sigma_{f}$          | Output scale           |                    Uniform                     |          -           |
| $\log \bar{\sigma}_{\text {obs }}$ | Base observation noise |             $\log \sqrt{10^{-5}}$              |         0.5          |
|              $m_{0}$               | Mean function maximum  |                    Uniform                     |          -           |
|       $x_{\mathrm{m}}^{(i)}$       | Mean function location |                    Uniform                     |          -           |
|        $\log \omega^{(i)}$         | Mean function scale    |                    Uniform                     |          -           |

Table S1: GP hyperparameters and their priors. See text for more information.

In Table S1, $\boldsymbol{L}$ denotes the vector of plausible ranges along each coordinate dimension, with $L^{(i)} \equiv$ $\operatorname{PUB}^{(i)}-\operatorname{PLB}^{(i)}$. The base observation noise $\bar{\sigma}_{\text {obs }}^{2}$ is a constant added to the input-dependent observation

---

#### Page 17

noise $\sigma_{\text {obs }}^{2}(\boldsymbol{\theta})$. Note that we have modified the GP hyperparameter priors with respect to the original VBMC paper, and these are now the default settings for both noisy and noiseless inference. In particular, we removed dependence of the priors from the GP training set (the 'empirical Bayes' approach previously used), as it was found to occasionally generate unstable behavior.

Frequent retrain In the original VBMC algorithm, the GP model and variational posterior are retrained only at the end of each iteration, corresponding to $n_{\text {active }}=5$ likelihood evaluations. However, in the presence of observation noise, approximation of both the GP and the variational posterior may benefit from a more frequent update. Thus, for noisy likelihoods we introduced a frequent retrain, that is fast re-training of both the GP and of the variational posterior within the active sampling loop, after each new function evaluation. This frequent update sets VBMC on par with other algorithms, such as GP-IMIQR and WSABI, which similarly retrain the GP representation after each likelihood evaluation. In VBMC, frequent retrain is active throughout the warm-up stage. After warm-up, we activate frequent retrain only when the previous iteration's reliability index $r(t-1)>3$, indicating that the solution has not stabilized yet.

# B. 2 Variational whitening

We start performing variational whitening $\tau_{\mathrm{vw}}$ iterations after the end of warm-up, and then subsequently at increasing intervals of $k \tau_{\mathrm{vw}}$ iterations, where $k$ is the count of previously performed whitenings ( $\tau_{\mathrm{vw}}=5$ in this work). Moreover, variational whitening is postponed until the reliability index $r(t)$ of the current iteration is below 3, indicating a degree of stability of the current variational posterior (see Section B.1). Variational whitening consists of a linear transformation $\mathbf{W}$ of the inference space (a rotation and rescaling) such that the variational posterior $q_{\phi}$ obtains unit diagonal covariance matrix. We compute the covariance matrix $\mathbf{C}_{\phi}$ of $q_{\phi}$ analytically, and we set the entries whose correlation is less than 0.05 in absolute value to zero, yielding a corrected covariance matrix $\overline{\mathbf{C}}_{\phi}$. We then calculate the whitening transform $\mathbf{W}$ by performing a singular value decomposition (SVD) of $\overline{\mathbf{C}}_{\phi}$.

## C Acquisition functions

In this Section, we report derivations and additional implementation details for the acquisition functions introduced in the main text.

## C. 1 Observation noise

All acquisition functions in the main text require knowledge of the log-likelihood observation noise $\sigma_{\text {obs }}(\boldsymbol{\theta})$ at an arbitrary point $\boldsymbol{\theta} \in \mathcal{X}$. However, we only assumed availability of an estimate $\left(\widehat{\sigma}_{\text {obs }}\right)_{n}$ of $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{n}\right)$ for all parameter values evaluated so far, $1 \leq n \leq N$. We estimate values of $\sigma_{\text {obs }}(\boldsymbol{\theta})$ outside the training set via a simple nearest-neighbor approximation, that is

$$
\sigma_{\mathrm{obs}}\left(\boldsymbol{\theta}_{\star}\right)=\sigma_{\mathrm{obs}}\left(\boldsymbol{\theta}_{n}\right) \quad \text { for } n=\arg \min _{1 \leq n \leq N} d_{\ell}\left(\boldsymbol{\theta}_{\star}, \boldsymbol{\theta}_{n}\right)
$$

where $d_{\ell}$ is the rescaled Euclidean distance between two points in inference space, where each coordinate dimension $i$ has been rescaled by the GP input length $\ell_{i}$, with $1 \leq i \leq D$. When multiple GP hyperparameter samples are available, we use the geometric mean of each input length across samples. Eq. S13 may seem like a coarse approximation, but we found it effective in practice.

## C. 2 Expected information gain (EIG)

The expected information gain (EIG) acquisition function $a_{\text {EIG }}$ is based on a mutual information maximizing acquisition function for Bayesian quadrature introduced in [11].
First, note that the information gain is defined as the KL-divergence between posterior and prior; in our case, between the posterior of the log joint $\mathcal{G}$ after observing value $y_{\star}$ at $\boldsymbol{\theta}_{\star}$, and the current posterior over $\mathcal{G}$ given the observed points in the training set, $\boldsymbol{\Xi}=\left\{\boldsymbol{\Theta}, \boldsymbol{y}, \boldsymbol{\sigma}_{\text {obs }}\right\}$. Since $y_{\star}$ is yet to be observed, we consider then the expected information gain of performing a measurement at $\boldsymbol{\theta}_{\star}$, that is

$$
\operatorname{EIG}\left(\boldsymbol{\theta}_{\star} ; \boldsymbol{\Xi}_{t}\right)=\mathbb{E}_{y_{\star} \mid \boldsymbol{\theta}_{\star}}\left[D_{\mathrm{KL}}\left(p(\mathcal{G} \mid \boldsymbol{\Xi} \cup\left\{\left(\boldsymbol{\theta}_{\star}, y_{\star}, \sigma_{\mathrm{obs} \star}\right)\right\}\right) \| p(\mathcal{G} \mid \boldsymbol{\Xi}) \mid\right]
$$

---

#### Page 18

It can be shown that Eq. S14 is identical to the mutual information between $\mathcal{G}$ and $\boldsymbol{y}_{*}$ [1]

$$
I\left[\mathcal{G} ; y_{*}\right]=H[\mathcal{G}]+H\left[y_{*}\right]-H\left[\mathcal{G}, y_{*}\right]
$$

where $H(\cdot)$ denotes the (marginal) differential entropy and $H(\cdot, \cdot)$ the joint entropy. By the definition of GP, $y_{*}$ is normally distributed, and so is each component $\mathcal{G}_{k}$ of the log-joint, due to Bayesian quadrature (see Section A). As a weighted sum of normally distributed random variables, $\mathcal{G}$ is also normally distributed, and so is the joint distribution of $y_{*}$ and $\mathcal{G}$. We recall that the differential entropy of a bivariate normal distribution with covariance matrix $\boldsymbol{\Lambda} \in \mathbb{R}^{2 \times 2}$ is $H=\log (2 \pi e)+\frac{1}{2} \log |\boldsymbol{\Lambda}|$. Thus we have (see Eq. 7 in the main text)

$$
a_{\mathrm{ElG}}\left(\boldsymbol{\theta}_{*}\right) \equiv I\left[\mathcal{G} ; y_{*}\right]=-\frac{1}{2} \log \left(1-\rho^{2}\left(\boldsymbol{\theta}_{*}\right)\right), \quad \text { with } \rho\left(\boldsymbol{\theta}_{*}\right) \equiv \frac{\mathbb{E}_{\boldsymbol{\phi}}\left[C_{\Xi}\left(f(\cdot), f\left(\boldsymbol{\theta}_{*}\right)\right)\right]}{\sqrt{v_{\Xi}\left(\boldsymbol{\theta}_{*}\right) \mathbb{V}_{f(\Xi)}[\mathcal{G}]}}
$$

where we used the scalar correlation $\rho(\cdot)$ [11]; and $C_{\Xi}(\cdot, \cdot)$ is the GP posterior covariance, $v_{\Xi}(\cdot)$ the GP posterior predictive variance (including observation noise), and $\mathbb{V}_{f(\Xi)}[\mathcal{G}]$ the posterior variance of the expected log joint - all given the current training set $\Xi$.
The expected value at the numerator of $\rho\left(\boldsymbol{\theta}_{*}\right)$ is

$$
\begin{aligned}
\mathbb{E}_{\boldsymbol{\phi}}\left[C_{\Xi}\left(f(\cdot), f\left(\boldsymbol{\theta}_{*}\right)\right)\right] & =\int q(\boldsymbol{\theta}) C_{\Xi}\left(f(\boldsymbol{\theta}), f\left(\boldsymbol{\theta}_{*}\right)\right) d \boldsymbol{\theta} \\
& =\sum_{k=1}^{K} w_{k} \int \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) C_{\Xi}\left(f(\boldsymbol{\theta}), f\left(\boldsymbol{\theta}_{*}\right)\right) d \boldsymbol{\theta} \\
& =\sum_{k=1}^{K} w_{k} \mathcal{K}_{k}\left(\boldsymbol{\theta}_{*}\right)
\end{aligned}
$$

where we recall that $w_{k}, \boldsymbol{\mu}_{k}$, and $\sigma_{k}$ are, respectively, the mixture weight, mean, and scale of the $k$-th component of the variational posterior $q$, for $1 \leq k \leq K ; \boldsymbol{\Sigma}$ is a common diagonal covariance matrix $\boldsymbol{\Sigma} \equiv \operatorname{diag}\left[\lambda^{(1)^{2}}, \ldots, \lambda^{(D)^{2}}\right] ;$ and $C_{\Xi}$ is the GP posterior covariance as per Eq. S6. Finally, each term in Eq. S17 can be written as

$$
\begin{aligned}
\mathcal{K}_{k}\left(\boldsymbol{\theta}_{*}\right)= & \int \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)\left[\sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\theta}_{*}, \boldsymbol{\Sigma}_{\ell}\right) \ldots\right. \\
& \left.\ldots-\sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\Theta}, \boldsymbol{\Sigma}_{\ell}\right)\left[\kappa(\boldsymbol{\Theta}, \boldsymbol{\Theta})+\boldsymbol{\Sigma}_{\text {obs }}(\boldsymbol{\Theta})\right]^{-1} \sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{\Theta} ; \boldsymbol{\theta}_{*}, \boldsymbol{\Sigma}_{\ell}\right)\right] d \boldsymbol{\theta} \\
= & \sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{\theta}_{*} ; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{\ell}+\sigma_{k}^{2} \boldsymbol{\Sigma}\right)-\sigma_{f}^{2} \Lambda \boldsymbol{z}_{k}^{\top}\left[\kappa(\boldsymbol{\Theta}, \boldsymbol{\Theta})+\boldsymbol{\Sigma}_{\text {obs }}(\boldsymbol{\Theta})\right]^{-1} \mathcal{N}\left(\boldsymbol{\Theta} ; \boldsymbol{\theta}_{*}, \boldsymbol{\Sigma}_{\ell}\right)
\end{aligned}
$$

where $\boldsymbol{z}_{k}$ is a $N$-dimensional vector with entries $z_{k}^{(n)}=\sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{\mu}_{k} ; \boldsymbol{\theta}_{n}, \sigma_{k}^{2} \boldsymbol{\Sigma}+\boldsymbol{\Sigma}_{\ell}\right)$ for $1 \leq n \leq N$.

# C. 3 Integrated median / variational interquantile range (IMIQR/ VIQR)

The integrated median interquantile range (IMIQR) acquisition function has been recently proposed in [10] as a robust, principled metric for posterior estimation with noisy evaluations (see also Eq. 8 in the main text),

$$
a_{\mathrm{IMQR}}\left(\boldsymbol{\theta}_{*}\right)=-2 \int_{\mathcal{X}} \exp \left(\bar{f}_{\Xi}(\boldsymbol{\theta})\right) \sinh \left(u s_{\Xi ; ; \boldsymbol{\theta}_{*}}(\boldsymbol{\theta})\right) d \boldsymbol{\theta}
$$

It combines two ideas: (a) using the interquantile range (IQR) as a robust measure of uncertainty, as opposed to the variance; and (b) approximating the median integrated IQR loss, which follows from decision-theoretic principles but is intractable, with the integrated median IQR, which can be computed somewhat more easily [10]. Note that Eq. S19 differs slightly from Eq. 30 in [10] in that in our definition the prior term is subsumed into the joint distribution, with no loss of generality.
A major issue with Eq. S19 is that the integral is still intractable. By noting that $\exp \left(\bar{f}_{\Xi}(\boldsymbol{\theta})\right)$ is the joint distribution, in VBMC we can replace it with the variational posterior, obtaining thus the variational (integrated median) interquantile range acquisition function $o_{\text {VIQR }}$ (see main text).

---

#### Page 19

# D Benchmark details

We report here details about the benchmark setup, in particular parameter bounds and dataset information for all problems in the benchmark (Section D.1); how we adapted the wsABI and GP-IMIQR algorithms for the purpose of our noisy benchmark (Section D.2); and the computing infrastructure (Section D.3).

## D. 1 Problem specification

Parameter bounds We report in Table S2 the parameter bounds used in the problems of the noisyinference benchmark. We denote with LB and UB the hard lower and upper bounds, respectively; whereas with PLB and PUB we denote the 'plausible' lower and upper bounds, respectively [5, 26]. Plausible ranges should identify a region of high posterior probability mass in parameter space given our knowledge of the model and of the data; lacking other information, these are recommended to be set to e.g. the $\sim 68 \%$ high-density interval according to the marginal prior probability in each dimension [5]. Plausible values are used to initialize and set hyperparameters of some of the algorithms. For example, the initial design for VBMC and GP-IMIQR is drawn from a uniform distribution over the plausible box in inference space.

## Dataset information

- Ricker: We generated a synthetic dataset of $T=50$ observations using the "true" parameter vector $\boldsymbol{\theta}_{\text {true }}=(3.8,10,0.3)$ with $T=50$, as in [10].
- Attentional drift-diffusion model (aDDM): We used fixation and choice data from two participants (subject \#13 and subject \#16 from [43]) who completed all $N=100$ trials in the experiment without technical issues (reported as 'missing trials' in the data).
- Bayesian timing: We analyzed reproduced time intervals of one representative subject from Experiment 3 (uniform distribution; subject \#2) in [45], with $N=1512$ trials.
- Multisensory causal inference: We examined datasets of subject \#1 and \#2 from the explicit causal inference task ('unity judgment') in [47]; with respectively $N=1069$ and $N=857$ trials, across three different visual coherence conditions.
- Neuronal selectivity: We analyzed two neurons (one from area V1, one from area V2 of primate visual cortex) from [48], both with $N=1760$ trials. The same datasets have been used in previous optimization and inference benchmarks [5, 6, 26].
- Rodent 2AFC: We took a representative rat subject from [49], already used for demonstration purposes by [50], limiting our analysis of choice behavior to the last $N=10^{4}$ trials in the data set.

## D. 2 Algorithm specification

WSABI Warped sequential active Bayesian integration (WSABI) is a technique to compute the log marginal likelihood via GP surrogate models and Bayesian quadrature [18]. In this work, we use wsABI as an example of a surrogate-based method for model evidence approximation different from VBMC. The key idea of WSABI is to model directly the square root of the likelihood function $\mathcal{L}$ (as opposed to the log-likelihood) via a GP,

$$
\hat{\mathcal{L}}(\boldsymbol{\theta}) \equiv \sqrt{2(\mathcal{L}(\boldsymbol{\theta})-\alpha)} \quad \Longrightarrow \quad \mathcal{L}(\boldsymbol{\theta})=\alpha+\frac{1}{2} \hat{\mathcal{L}}(\boldsymbol{\theta})^{2}
$$

where $\alpha$ is a small positive scalar. If $\hat{\mathcal{L}}$ is modeled as a GP, $\mathcal{L}$ is not itself a GP (right-hand side of Eq. S20), but it can be approximated as a GP via a linearization procedure (WSABI-L in [18]), which is the approach we follow throughout our work.
The wsABI algorithm requires an unlimited inference space $\mathcal{X} \equiv \mathbb{R}^{D}$ and a multivariate normal prior [18]. In our benchmark, all parameters have bound constraints, so we first map the original space to an unbounded inference space via a rescaled logit transform, with an appropriate log-Jacobian correction to the log posterior (see e.g., [5,58]). Also, in our benchmark all priors are assumed to be uniform. Thus, we pass to WSABI a 'pseudo-prior' consisting of a multivariate normal centered on the middle of the plausible box, and with standard deviations equal to half the plausible range in each

---

#### Page 20

Table S2: Parameters and bounds of all models (before remapping to inference space).

|        Model         |                       Parameter                       |             Description              |   LB   |  UB  |  PLB   |  PUB  |
| :------------------: | :---------------------------------------------------: | :----------------------------------: | :----: | :--: | :----: | :---: |
|        Ricker        |                      $\log (r)$                       |         Growth factor (log)          |   3    |  5   |  3.2   |  4.8  |
|                      |                        $\phi$                         |          Observed fraction           |   4    |  20  |  5.6   | 18.4  |
|                      |                $\sigma_{\varepsilon}$                 |             Growth noise             |   0    | 0.8  |  0.08  | 0.72  |
|         aDDM         |                          $d$                          |              Drift rate              |   0    |  5   |  0.1   |   2   |
|                      |                        $\beta$                        |       Attentional bias factor        |   0    |  1   |  0.1   |  0.9  |
|                      |                $\sigma_{\varepsilon}$                 |           Diffusion noise            |  0.1   |  2   |  0.2   |   1   |
|                      |                       $\lambda$                       |              Lapse rate              |  0.01  | 0.2  |  0.03  |  0.1  |
| Bayesian <br> timing |                   $w_{\mathrm{s}}$                    |   Sensory noise (Weber's fraction)   |  0.01  | 0.5  |  0.05  | 0.25  |
|                      |                   $w_{\text {in }}$                   |    Motor noise (Weber's fraction)    |  0.01  | 0.5  |  0.05  | 0.25  |
|                      |                  $\mu_{\mathrm{p}}$                   |         Prior mean (seconds)         |  0.3   | 1.95 |  0.6   | 0.975 |
|                      |                 $\sigma_{\mathrm{p}}$                 |  Prior standard deviation (seconds)  | 0.0375 | 0.75 | 0.075  | 0.375 |
|                      |                       $\lambda$                       |              Lapse rate              |  0.01  | 0.2  |  0.02  | 0.05  |
|     Multisensory     |               $\sigma_{\text {vist }}$                |        Vestibular noise (deg)        |  0.5   |  80  |   1    |  40   |
|        causal        | $\sigma_{\text {vis }}\left(c_{\text {low }}\right)$  |  Visual noise, low coherence (deg)   |  0.5   |  80  |   1    |  40   |
|    inference (CI)    | $\sigma_{\text {vis }}\left(c_{\text {med }}\right)$  | Visual noise, medium coherence (deg) |  0.5   |  80  |   1    |  40   |
|                      | $\sigma_{\text {vis }}\left(c_{\text {high }}\right)$ |  Visual noise, high coherence (deg)  |  0.5   |  80  |   1    |  40   |
|                      |                       $\kappa$                        |      'Sameness' threshold (deg)      |  0.25  | 180  |   1    |  45   |
|                      |                       $\lambda$                       |              Lapse rate              | 0.005  | 0.5  |  0.01  |  0.2  |
|       Neuronal       |                     $\theta_{1}$                      | Preferred direction of motion (deg)  |   0    | 360  |   90   |  270  |
|     selectivity      |                     $\theta_{2}$                      | Preferred spatial freq. (cycles/deg) |  0.05  |  15  |  0.5   |  10   |
|                      |                     $\theta_{3}$                      |     Aspect ratio of 2-D Gaussian     |  0.1   | 3.5  |  0.3   |  3.2  |
|                      |                     $\theta_{4}$                      |      Derivative order in space       |  0.1   | 3.5  |  0.3   |  3.2  |
|                      |                     $\theta_{5}$                      |       Gain inhibitory channel        |   -1   |  1   | $-0.3$ |  0.3  |
|                      |                     $\theta_{6}$                      |          Response exponent           |   1    | 6.5  |  1.01  |   5   |
|                      |                     $\theta_{7}$                      |      Variance of response gain       | 0.001  |  10  | 0.015  |   1   |
|        Rodent        |                        $w_{0}$                        |                 Bias                 |  $-3$  |  3   |  $-1$  |   1   |
|         2AFC         |                   $w_{\mathrm{c}}$                    |  Weight on 'previous correct side'   |  $-3$  |  3   |  $-1$  |   1   |
|                      |                        $w_{2}$                        |     Weight on long-term history      |  $-3$  |  3   |  $-1$  |   1   |
|                      |                $w_{\mathrm{L}}^{(0)}$                 |   Weight on left stimulus $(t=0)$    |  $-3$  |  3   |  $-1$  |   1   |
|                      |                $w_{\mathrm{L}}^{(-1)}$                |   Weight on left stimulus $(t=-1)$   |  $-3$  |  3   |  $-1$  |   1   |
|                      |                $w_{\mathrm{L}}^{(-2)}$                |   Weight on left stimulus $(t=-2)$   |  $-3$  |  3   |  $-1$  |   1   |
|                      |                $w_{\mathrm{R}}^{(0)}$                 |   Weight on right stimulus $(t=0)$   |  $-3$  |  3   |  $-1$  |   1   |
|                      |                $w_{\mathrm{R}}^{(-1)}$                |  Weight on right stimulus $(t=-1)$   |  $-3$  |  3   |  $-1$  |   1   |
|                      |                $w_{\mathrm{R}}^{(-2)}$                |  Weight on right stimulus $(t=-2)$   |  $-3$  |  3   |  $-1$  |   1   |

coordinate direction in inference space (see Section D.1). We then correct for this added pseudo-prior by subtracting the log-pseudo-prior value from each log-joint evaluation.
wsABI with noisy likelihoods The original wsabi algorithm does not explicitly support observation noise in the (log-)likelihood. Thus, we modified wsabi to include noisy likelihood evaluations, by mapping noise on the log-likelihood to noise in the square-root likelihood via an unscented transform, and by modifying wsabi's uncertainty-sampling acquisition function to account for observation noise (similarly to what we did for $a_{\text {npne }}$, see main text). However, we found the noise-adjusted wsABI to perform abysmally, even worse than the original wsABI on our noisy benchmark. This failure is likely due to the particular representation used by wsabi (Eq. S20). Crucially, even moderate noise on the log-likelihood translates to extremely large noise on the (square-root) likelihood. Due to this large observation noise, the latent GP will revert to the GP mean function, which corresponds to the constant $\alpha$ (Eq. S20). In the presence of modeled log-likelihood noise, thus, the GP representation of wsabi becomes near-constant and practically useless. For this reason, here and in the main text we report the results of wsabi without explicitly added support for observation noise. More work is

---

#### Page 21

needed to find an alternative representation of WSABI which would not suffer from observation noise, but it is beyond the scope of our paper.

GP-IMIQR For the GP-IMIQR algorithm described in [10], we used the latest implementation (v3) publicly available at: https://github.com/mjarvenpaa/parallel-GP-SL. We considered the IMIQR acquisition function with sequential sampling strategy; the best-performing acquisition function in the empirical analyses in [10]. We used the code essentially 'as is', with minimal changes to interface the algorithm to our noisy benchmark. We ran the algorithm with the recommended default hyperparameters. Given the particularly poor performance of GP-IMIQR on some problems (e.g., Timing, Neuronal), which we potentially attributed to convergence failures of the default MCMC sampling algorithm (DRAM; [59]), we also reran the method with an alternative and robust sampling method (parallel slice sampling; [33,47]). However, performance of GP-IMIQR with slice sampling was virtually identical, and similarly poor, to its performance with DRAM (data not shown). We note that the same grave issues with the Neuronal model emerged even when we forced initialization of the algorithm in close vicinity of the mode of the posterior (data not shown). We attribute the inability of GP-IMIQR to make significant progress on some problems to excessive exploration, which may lead to GP instabilities; although further investigation is needed to identify the exact causes, beyond the scope of this work.

# D. 3 Computing infrastructure

All benchmark runs were performed on MATLAB 2017a (Mathworks, Inc.) using a High Performance Computing cluster whose details can be found at the following link: https://wikis.nyu.edu/ display/NYUHPC/Clusters+--+Prince. Since different runs may have been assigned to compute nodes with vastly different loads or hardware, we regularly assessed execution speed by performing a set of basic speed benchmark operations (bench in MATLAB; considering only numerical tasks). Running times were then converted to the estimated running time on a reference machine, a laptop computer with 16.0 GB RAM and Intel(R) Core(TM) i7-6700HQ CPU @ 2.60 GHz , forced to run single-core during the speed test.

## E Additional results

We include here a series of additional experimental results and plots omitted from the main text for reasons of space. First, we report the results of the posterior inference benchmark with a different metric (Section E.1). Then, we present results of a robustness analysis of solutions across runs (Section E.2) and of an ablation study (Section E.3). In Section E.4, we show a comparison of true and approximate posteriors for all problems in the benchmark. Then, we study sensitivity of VBMC-VIQR to imprecision in the log-likelihood noise estimates (Section E.5). Finally, we report results for an additional synthetic problem, the g-and-k model (Section E.6).

## E. 1 Gaussianized symmetrized KL divergence (gsKL) metric

In the main text, we measured the quality of the posterior approximation via the mean marginal total variation distance (MMTV) between true and approximate posteriors, which quantifies the distance between posterior marginals. Here we consider an alternative loss metric, the "Gaussianized" symmetrized Kullback-Leibler divergence (gsKL), which is sensitive instead to differences in means and covariances [5]. Specifically, the gsKL between two pdfs $p$ and $q$ is defined as

$$
\operatorname{gsKL}(p, q)=\frac{1}{2}\left[D_{\mathrm{KL}}(\mathcal{N}[p]||\mathcal{N}[q])+D_{\mathrm{KL}}\left(\mathcal{N}[q]||\mathcal{N}[p])\right]\right.
$$

where $\mathcal{N}[p]$ is a multivariate normal distribution with mean equal to the mean of $p$ and covariance matrix equal to the covariance of $p$ (and same for $q$ ). Eq. S21 can be expressed in closed form in terms of the means and covariance matrices of $p$ and $q$.
Fig. S1 shows the gsKL between approximate posterior and ground truth, for all algorithms and inference problems considered in the main text. For reference, two Gaussians with unit variance and whose means differ by $\sqrt{2}$ (resp., $\frac{1}{2}$ ) have a gsKL of 1 (resp., $\frac{1}{8}$ ). For this reason, we consider a desirable target to be (much) less than 1 . Results are qualitatively similar to what we observed for the MMTV metric (Fig. 3 in the main text), in that the ranking and convergence properties of different

---

#### Page 22

methods is the same for MMTV and gsKL. In particular, previous state-of-the art method GP-IMIQR fails to converge in several challenging problems (Timing, Neuronal and Rodent); among the variants of VBMC, VBMC-VIQR and VBMC-IMIQR are the only ones that perform consistently well.

> **Image description.** This image is a figure containing a series of line graphs comparing the performance of different algorithms. Each graph shows the Gaussianized symmetrized KL divergence (gsKL) on the y-axis against the number of function evaluations on the x-axis.
>
> - **Overall Structure:** The figure is arranged as a grid of 5x2 plots. Each plot corresponds to a different problem or dataset, as indicated by the titles above each graph.
>
> - **Axes:**
>
>   - The y-axis is labeled "gsKL" and uses a logarithmic scale ranging from 10^-2 to 10^4 or 10^6 depending on the plot.
>   - The x-axis is labeled "Function evaluations" and shows a linear scale ranging from 0 to 200, 300 or 400 depending on the plot.
>
> - **Titles and Labels:**
>
>   - The titles of the graphs are: "Ricker", "aDDM (S1)", "aDDM (S2)", "Timing", "Multisensory (S1)", "Multisensory (S2)", "Neuronal (V1)", "Neuronal (V2)", and "Rodent".
>   - Each title is followed by "D = [number]", where the number varies from 3 to 9.
>   - A legend on the top right of the figure identifies the different algorithms: "gp-imiqr", "vbmc-npro", "vbmc-eig", "vbmc-imiqr", and "vbmc-viqr". Each algorithm is represented by a different colored line (green, light green, light blue, dotted dark blue, and black, respectively).
>
> - **Data Representation:**
>
>   - Each algorithm's performance is represented by a line on each graph.
>   - Shaded areas around some of the lines (particularly for "gp-imiqr" and "vbmc-npro") likely represent confidence intervals or standard deviations.
>   - A dashed horizontal line at y=1 is present in each graph, presumably indicating a desirable target value for the gsKL.
>
> - **Visual Patterns:**
>   - In most graphs, the "vbmc-viqr" (black line) and "vbmc-imiqr" (dotted dark blue line) algorithms converge to a lower gsKL value more quickly than the other algorithms.
>   - The "gp-imiqr" (green line) algorithm often performs worse than the other algorithms, especially for problems like "Timing" and "Neuronal (V1)".
>   - The y-axis scales vary between plots, especially for "Neuronal (V1)" and "Neuronal (V2)", which go up to 10^6, indicating potentially higher gsKL values for these problems.

Figure S1: Posterior estimation loss (gsKL). Median Gaussianized symmetrized KL divergence (gsKL) between the algorithm's posterior and ground truth, as a function of number of likelihood evaluations. A desirable target (dashed line) is less than 1. Shaded areas are $95 \%$ CI of the median across 100 runs.

# E. 2 Worse-case analysis ( $90 \%$ quantile)

In the main text and other parts of this Supplement, we showed for each performance metric the median performance across multiple runs, to convey the 'average-case' performance of an algorithm; in that we expect performance to be at least as good as the median for about half of the runs. To assess the robustness of an algorithm, we are also interested in a 'worse-case' analysis that looks at higher quantiles of the distribution of performance, which are informative of how bad performance can reasonably get (e.g., we expect only about one run out of ten to be worse than the $90 \%$ quantile).

We show in Figure S2 the $90 \%$ quantile of the MMTV metric, to be compared with Fig. 3 in the main text (results for other metrics are analogous). These results show that the best-performing algorithms, VBMC-VIQR and VBMC-IMIQR, are also the most robust, as both methods manage to achieve good solutions most of the time (with one method working slightly better than the other on some problems, and vice versa). By contrast, other methods such as GP-IMIQR show more variability, in that on some problems (e.g., aDDM) they may have reasonably good median performance, but much higher error when looking at the $90 \%$ quantile.

## E. 3 Ablation study

We show here the performance of the VBMC algorithm after removing some of the features considered in the main paper. As a baseline algorithm we take VBMC-VIQR. First, we show VBMC-NOWV, obtained by removing from the baseline the 'variational whitening' feature (see main text and Section B.2). Second, we consider a variant of VBMC-VIQR in which we do not sample GP hyperparameters from the hyperparameter posterior, but simply obtain a point estimate through maximum-a-posteriori estimation (VBMC-MAP). Optimizing GP hyperperameters, as opposed to a Bayesian treatment of hyperparameters, is a common choice for many surrogate-based methods (e.g., WSABI, GP-IMIQR, although the latter integrates analytically over the GP mean function), so we investigate whether it is needed for VBMC. Finally, we plot the performance of VBMC in its original implementation (VBMC-OLD), as per the VBMC paper [5]. For reference, we also plot both the VBMC-VIQR and GP-IMIQR algorithms, as per Fig. 3 in the main text.

---

#### Page 23

> **Image description.** The image is a figure containing nine line graphs arranged in a 3x3 grid. Each graph displays the performance of different algorithms in terms of posterior estimation loss as a function of function evaluations.
>
> Each graph has the following structure:
>
> - **Title:** Each graph has a title indicating the model being evaluated: "Ricker", "aDDM (S1)", "aDDM (S2)", "Timing", "Multisensory (S1)", "Multisensory (S2)", "Neuronal (V1)", "Neuronal (V2)", and "Rodent".
> - **D = value:** Each graph also includes a "D =" value, indicating the dimensionality of the model. The values are D=3, D=4, D=4, D=5, D=6, D=6, D=7, D=7, and D=9, respectively.
> - **Axes:** The y-axis is labeled "MMTV (90% quantile)" and ranges from 0 to 1. The x-axis is labeled "Function evaluations" and ranges from 0 to 200, 300, or 400 depending on the graph.
> - **Lines:** Each graph contains multiple lines, each representing a different algorithm. The algorithms are "gp-imiqr" (green), "vbmc-npro" (light green), "vbmc-eig" (light blue), "vbmc-imiqr" (dotted blue), and "vbmc-viqr" (black). Shaded areas around each line represent the 95% confidence interval of the median across 100 runs.
> - **Dashed Line:** A horizontal dashed line is present at y = 0.2 on each graph.
>
> A legend on the top right graph identifies the different algorithms and their corresponding line colors and styles.

Figure S2: Worse-case posterior estimation loss (MMTV). 90\% quantile of the mean marginal total variation distance (MMTV) between the algorithm's posterior and ground truth, as a function of number of likelihood evaluations. A desirable target (dashed line) is less than 0.2 , corresponding to more than $80 \%$ overlap between true and approximate posterior marginals (on average across model parameters). Shaded areas are $95 \%$ CI of the $90 \%$ quantile across 100 runs.

We show in Fig. S3 the results for the MMTV metric, although results are similar for other inference metrics. We can see that all 'lesioned' versions of VBMC perform generally worse than VBMC-VIQR, to different degree, and more visibly in more difficult inference problems. However, for example, VBMC-MAP still performs substantially better than GP-IMIQR, suggesting that the difference in performance between VBMC and GP-IMIQR is not simply because VBMC marginalizes over GP hyperparameters. It is also evident that the previous version of VBMC (VBMC-OLD) is extremely ineffective in the presence of noisy log-likelihoods.

> **Image description.** This image contains a set of line graphs comparing the performance of different algorithms. There are nine subplots arranged in a 3x3 grid. Each subplot displays the mean marginal total variation distance (MMTV) on the y-axis as a function of function evaluations on the x-axis.
>
> Each subplot has the following structure:
>
> - **Title:** Each subplot has a title indicating the model being evaluated (e.g., "Ricker", "aDDM (S1)", "Timing", "Multisensory (S1)", "Neuronal (V1)", "Rodent").
> - **D Value:** Each subplot also displays a "D = [number]" value, likely representing the dimensionality of the problem.
> - **Axes:** The x-axis is labeled "Function evaluations" and ranges from 0 to either 200/300 or 400. The y-axis is labeled "MMTV" and ranges from 0 to 1.
> - **Lines:** Each subplot contains multiple lines, each representing a different algorithm:
>   - gp-imiqr (green)
>   - vbmc-old (black dotted)
>   - vbmc-nowv (blue dashed)
>   - vbmc-map (pink dotted)
>   - vbmc-viqr (black solid)
> - **Shaded Areas:** Shaded areas around each line represent confidence intervals.
> - **Horizontal Dashed Line:** A horizontal dashed line is present at y=0.2 in each subplot.
>
> The legend on the top right identifies each algorithm with its corresponding line style and color. The general trend across all subplots is that the MMTV decreases as the number of function evaluations increases, indicating improved performance of the algorithms over time. The relative performance of the algorithms varies across different models.

Figure S3: Lesion study; posterior estimation loss (MMTV). Median mean marginal total variation distance (MMTV) between the algorithm's posterior and ground truth, as a function of number of likelihood evaluations. Shaded areas are $95 \%$ CI of the median across 100 runs.

---

#### Page 24

> **Image description.** The image is a figure displaying multiple plots arranged in a grid, comparing true and approximate marginal posterior distributions for different parameters across several problems. Each row represents a different problem, and each column represents a different parameter within that problem.
>
> - **General Layout:** The plots are organized in a matrix format. Each cell contains a plot showing a probability distribution. The problems are labeled on the right side of the figure, and the parameters are labeled above each column.
>
> - **Plot Details:** Each plot shows two distributions:
>
>   - A red line represents the "true" marginal posterior distribution, obtained through MCMC sampling.
>   - Multiple black lines represent marginal distributions of approximate posteriors, obtained using VBMC-VIQR. There are five black lines in each plot, representing five randomly chosen approximate posteriors.
>
> - **Problems and Parameters:** The problems listed on the right side are:
>
>   - Ricker
>   - aDDM (S1)
>   - aDDM (S2)
>   - Timing
>   - Multisensory (S1)
>   - Multisensory (S2)
>   - Neuronal (V1)
>   - Neuronal (V2)
>   - Rodent
>
>   The parameters vary depending on the problem. Examples of parameters include:
>
>   - `log(r)`
>   - `phi`
>   - `sigma_epsilon`
>   - `d`
>   - `beta`
>   - `lambda`
>   - `w_s`
>   - `w_m`
>   - `mu_p`
>   - `sigma_p`
>   - `sigma_vis(C_low)`
>   - `sigma_vis(C_med)`
>   - `sigma_vis(C_high)`
>   - `sigma_vest`
>   - `kappa`
>   - `theta_1` through `theta_7`
>   - `w_L^(0)`
>   - `w_R^(0)`
>   - `w_0`
>   - `w_L^(-1)`
>   - `w_R^(-1)`
>   - `w_L^(-2)`
>   - `w_R^(-2)`
>   - `w_c`
>   - `w_s`
>
> - **Visual Patterns:** The plots show how well the approximate posteriors (black lines) match the true posterior (red line). In some cases, the black lines closely follow the red line, indicating a good approximation. In other cases, there is more variability among the black lines, suggesting a less accurate approximation.

Figure S4: True and approximate marginal posteriors. Each panel shows the ground-truth marginal posterior distribution (red line) for each parameter of problems in the noisy benchmark (rows). For each problem, black lines are marginal distributions of five randomly-chosen approximate posteriors returned by VBMC-VIQR.

# E. 4 Comparison of true and approximate posteriors

We plot in Fig. S4 a comparison between the 'true' marginal posteriors, obtained for all problems via extensive MCMC sampling, and example approximate posteriors recovered by VBMC-VIQR after $50 \times(D+2)$ likelihood evaluations, the budget allocated for the benchmark. As already quantified by the MMTV metric, we note that VBMC is generally able to obtain good approximations of the true posterior marginals. The effect of noise becomes more prominent when the posteriors are nearly flat, in which case we see greater variability in the VBMC solutions for some parameters (e.g., in the challenging Rodent problem). Note that this is also a consequence of choosing non-informative uniform priors over bounded parameter ranges in our benchmark, which is not necessarily best practice on real problems; (weakly) informative priors should be preferred in most cases [2].
To illustrate the ability of VBMC-VIQR to recover complex interactions in the posterior distribution (and not only univariate marginals) in the presence of noise, we plot in Fig. S5 the full pairwise posterior for one of the problems in the benchmark (Timing model). We can see that the approximate posterior matches the true posterior quite well, with some underestimation of the distribution tails. Underestimation of posterior variance is a common problem for variational approximations [60] and magnified here by the presence of noisy log-likelihood evaluations, and it represents a potential direction of improvement for future work.

## E. 5 Sensitivity to imprecise noise estimates

In this paragraph, we look at how robust VBMC-VIQR is to different degrees of imprecision in the estimation of log-likelihood noise. We consider the same setup with three example problems as in the noise sensitivity analysis reported in main text (Fig. 4 in the main text). For this analysis, we fixed the emulated noise to $\sigma_{\text {obs }}(\boldsymbol{\theta})=2$ for all problems. We then assumed that the estimated noise $\widehat{\sigma}_{\text {obs }}(\boldsymbol{\theta})$, instead of being known (nearly) exactly, is drawn randomly as $\widehat{\sigma}_{\text {obs }} \sim$ Lognormal $\left(\ln \sigma_{\text {obs }}, \sigma_{\sigma}^{2}\right)$, where $\sigma_{\sigma} \geq 0$ represents the jitter of the noise estimates on a logarithmic scale.

---

#### Page 25

> **Image description.** The image shows two triangle plots, labeled A and B, comparing the 'true' posterior distribution of a Timing model with an approximate posterior obtained using VBMC-VIQR.
>
> **Panel A: True**
>
> - The panel is labeled "A" in the top left corner and "True" centered above the plot.
> - It displays a triangle plot consisting of histograms on the diagonal and contour plots below the diagonal.
> - The parameters being analyzed are: w_s, w_m, μ_p, σ_p, and λ. These parameters are labeled along the x and y axes.
> - The histograms on the diagonal show the marginal posterior distribution for each parameter. They are all unimodal, but have different shapes.
> - The contour plots below the diagonal show the 2D marginal posterior distributions for each pair of parameters. The contours are colored in shades of blue and yellow, indicating the density of the distribution. The contours show the relationships between the parameters.
>
> **Panel B: VBMC-VIQR**
>
> - The panel is labeled "B" in the top left corner and "VBMC-VIQR" centered above the plot.
> - It also displays a triangle plot, structured identically to panel A.
> - The parameters are the same as in panel A: w_s, w_m, μ_p, σ_p, and λ.
> - The histograms on the diagonal show the marginal posterior distribution for each parameter.
> - The contour plots below the diagonal show the 2D marginal posterior distributions for each pair of parameters. The shapes of the contour plots are similar to those in panel A, but there are some differences, indicating that the VBMC-VIQR approximation is not perfect.

Figure S5: True and approximate posterior of Timing model. A. Triangle plot of the 'true' posterior (obtained via MCMC) for the Timing model. Each panel below the diagonal is the contour plot of the 2-D marginal posterior distribution for a given parameter pair. Panels on the diagonal are histograms of the 1-D marginal posterior distribution for each parameter (as per Fig. S4). B. Triangle plot of a typical variational solution returned by VBMC-VIQR.

We tested the performance of VBMC-VIQR for different values of noise-of-estimating-noise, $\sigma_{\sigma} \geq 0$ (see Fig. S6). We found that up to $\sigma_{\sigma} \approx 0.4$ (that is, $\widehat{\sigma}_{\text {obs }}$ varying roughly between $0.5-2.2$ times the true value) the quality of the inference degrades only slightly. For example, at worst the MMTV metric rises from 0.13 to 0.16 on the Timing problem (less than $\sim 25 \%$ increase), and in the other problems it is barely affected. These results show that VBMC-VIQR is quite robust to imprecise noise estimates. Combined with the fact that we expect estimates of the noise obtained from methods such as IBS to be very precise [41], imprecision in the noise estimates should not be an issue in practice.

> **Image description.** This image contains two panels, labeled A and B, each containing three line graphs. Each line graph shows the performance of the VBMC-VIQR method with respect to ground truth, as a function of noise-of-estimating-noise (sigma).
>
> Panel A:
>
> - The y-axis is labeled "LML loss" and has a logarithmic scale ranging from 0.01 to 10^3 (1000).
> - The x-axis is labeled "Noise-estimation noise σ" and ranges from 0 to 0.7 in increments of 0.1.
> - The three graphs are labeled "aDDM (S1)", "Timing", and "Neuronal (V1)".
> - Each graph displays a black line representing the median value, surrounded by a gray shaded area indicating the 95% confidence interval of the median across 50 runs.
> - A dashed horizontal line is present at y=1.
>
> Panel B:
>
> - The y-axis is labeled "MMTV" and ranges from 0 to 1.
> - The x-axis is labeled "Noise-estimation noise σ" and ranges from 0 to 0.7 in increments of 0.1.
> - The three graphs are labeled "aDDM (S1)", "Timing", and "Neuronal (V1)".
> - Each graph displays a black line representing the median value, surrounded by a gray shaded area indicating the 95% confidence interval of the median across 50 runs.
> - A dashed horizontal line is present at y=0.2.
>
> A legend in the top right of panel A indicates that the black line represents "vbmc-viqr".

Figure S6: Sensitivity to imprecise noise estimates. Performance metrics of VBMC-VIQR with respect to ground truth, as a function of noise-of-estimating-noise $\sigma_{\sigma}$. For all metrics, we plot the median and shaded areas are $95 \%$ CI of the median across 50 runs. A. Absolute error of the log marginal likelihood (LML) estimate. B. Mean marginal total variation distance (MMTV).

## E. 6 g-and-k model

We report here results of another synthetic test model omitted from the main text. The g-and-k model is a common benchmark simulation model represented by a flexible probability distribution defined via its quantile function,

$$
Q\left(\Phi^{-1}(p) ; \boldsymbol{\theta}\right)=a+b\left(1+c \frac{1-\exp \left(-g \Phi^{-1}(p)\right)}{1+\exp \left(-g \Phi^{-1}(p)\right)}\right)\left[1+\left(\Phi^{-1}(p)\right)^{2}\right]^{k} \Phi^{-1}(p)
$$

where $a, b, c, g$ and $k$ are parameters and $p \in[0,1]$ is a quantile. As in previous studies, we fix $c=0.8$ and infer the parameters $\boldsymbol{\theta}=(a, b, g, k)$ using the synthetic likelihood (SL) approach [3,4,10]. We use the same dataset as $[4,10]$, generated with "true" parameter vector $\boldsymbol{\theta}_{\text {true }}=(3,1,2,0.5)$, and for the log-SL estimation the same four summary statistics obtained by fitting a skew $t$-distribution to a set

---

#### Page 26

of samples generated from Eq. S22. We use $N_{\text {sim }}=100$, which produces fairly precise observations, with $\sigma_{\text {obs }}\left(\boldsymbol{\theta}_{\text {MAP }}\right) \approx 0.14$. In terms of parameter bounds, we set $\mathrm{LB}=(2.5,0.5,1.5,0.3)$ and $\mathrm{UB}=(3.5,1.5,2.5,0.7)$ as in [10]; and $\mathrm{PLB}=(2.6,0.6,1.6,0.34)$ and $\mathrm{PUB}=(3.4,1.4,2.4,0.66)$.

> **Image description.** This image contains three line graphs, labeled A, B, and C, comparing the performance of various algorithms. Each graph plots the performance metric on the y-axis against the number of function evaluations on the x-axis.
>
> - **Graph A:**
>
>   - Title: "g-and-k" and "D = 4"
>   - Y-axis: "LML loss" with a logarithmic scale from 0.01 to 1000.
>   - X-axis: "Function evaluations" ranging from 0 to 300.
>   - A dashed horizontal line is present at y=1.
>   - Several colored lines represent different algorithms: green (gp-imiqr), red (wsabi), light green (vbmc-npro), light blue (vbmc-eig), dotted blue (vbmc-imiqr), and black (vbmc-viqr). Shaded areas around the lines represent the 95% confidence interval.
>
> - **Graph B:**
>
>   - Y-axis: "MMTV" with a linear scale from 0 to 1.
>   - X-axis: "Function evaluations" ranging from 0 to 300.
>   - Dashed horizontal lines are present at y=0.2 and y=0.5.
>   - The same algorithms are plotted as in Graph A, using the same color scheme and confidence intervals.
>
> - **Graph C:**
>   - Y-axis: "gsKL" with a logarithmic scale from 0.01 to 10000.
>   - X-axis: "Function evaluations" ranging from 0 to 300.
>   - A dashed horizontal line is present at y=1.
>   - The same algorithms are plotted as in Graph A and B, using the same color scheme and confidence intervals.
>
> A legend on the right side of the image identifies each algorithm with its corresponding color: gp-imiqr (green), wsabi (red), vbmc-npro (light green), vbmc-eig (light blue), vbmc-imiqr (dotted blue), and vbmc-viqr (black).

Figure S7: Performance on g-and-k model. Performance metrics of various algorithms with respect to ground truth, as a function of number of likelihood evaluations, on the g-and-k model problem. For all metrics, we plot the median and shaded areas are $95 \%$ CI of the median across 100 runs. A. Absolute error of the log marginal likelihood (LML) estimate. B. Mean marginal total variation distance (MMTV). C. "Gaussianized" symmetrized Kullback-Leibler divergence (gsKL).

We show in Fig. S7 the performance of all methods introduced in the main text for three different inference metric: the log marginal likelihood (LML) loss, and both the mean marginal total variation distance (MMTV) and the "Gaussianized" symmetrized Kullback-Leibler divergence (gsKL) between approximate posterior and ground-truth posterior. For algorithms other than VBMC, we only report metrics they were designed for (posterior estimation for GP-IMIQR, model evidence for WSABI). The plots show that almost all algorithms (except WSABI) eventually converge to a very good performance across metrics, with only some differences in the speed of convergence. These results suggest that the g-and-k problem as used, e.g., in [10] might be a relatively easy test case for surrogate-based Bayesian inference; as opposed to the challenging real scenarios of our main benchmark, in which we find striking differences in performance between algorithms. Since we already present a simple synthetic scenario in the main text (the Ricker model), we did not include the g-and-k model as part of our main noisy-benchmark.
Finally, we note that when performing simulation-based inference based on summary statistics (such as here with the g-and-k model, and the Ricker model discussed in the main text), computing the marginal likelihood may not be a reliable approach for model comparison [61]. However, this is not a concern when performing simulation-based inference with methods that compute the log-likelihood with the entire data, such as IBS [41], as per all the other example problems in the main text.