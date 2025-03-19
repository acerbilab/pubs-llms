```
@article{acerbi2018variational,
  title={Variational Bayesian Monte Carlo},
  author={Luigi Acerbi},
  year={2018},
  journal={The Thirty-second Annual Conference on Neural Information Processing Systems (NeurIPS 2018)}
}
```

---

#### Page 1

# Variational Bayesian Monte Carlo

Luigi Acerbi\*<br>Department of Basic Neuroscience<br>University of Geneva<br>luigi.acerbi@unige.ch

#### Abstract

Many probabilistic models of interest in scientific computing and machine learning have expensive, black-box likelihoods that prevent the application of standard techniques for Bayesian inference, such as MCMC, which would require access to the gradient or a large number of likelihood evaluations. We introduce here a novel sample-efficient inference framework, Variational Bayesian Monte Carlo (VBMC). VBMC combines variational inference with Gaussian-process based, active-sampling Bayesian quadrature, using the latter to efficiently approximate the intractable integral in the variational objective. Our method produces both a nonparametric approximation of the posterior distribution and an approximate lower bound of the model evidence, useful for model selection. We demonstrate VBMC both on several synthetic likelihoods and on a neuronal model with data from real neurons. Across all tested problems and dimensions (up to $D=10$ ), VBMC performs consistently well in reconstructing the posterior and the model evidence with a limited budget of likelihood evaluations, unlike other methods that work only in very low dimensions. Our framework shows great promise as a novel tool for posterior and model inference with expensive, black-box likelihoods.

## 1 Introduction

In many scientific, engineering, and machine learning domains, such as in computational neuroscience and big data, complex black-box computational models are routinely used to estimate model parameters and compare hypotheses instantiated by different models. Bayesian inference allows us to do so in a principled way that accounts for parameter and model uncertainty by computing the posterior distribution over parameters and the model evidence, also known as marginal likelihood or Bayes factor. However, Bayesian inference is generally analytically intractable, and the statistical tools of approximate inference, such as Markov Chain Monte Carlo (MCMC) or variational inference, generally require knowledge about the model (e.g., access to the gradients) and/or a large number of model evaluations. Both of these requirements cannot be met by black-box probabilistic models with computationally expensive likelihoods, precluding the application of standard Bayesian techniques of parameter and model uncertainty quantification to domains that would most need them.
Given a dataset $\mathcal{D}$ and model parameters $\boldsymbol{x} \in \mathbb{R}^{D}$, here we consider the problem of computing both the posterior $p(\boldsymbol{x} \mid \mathcal{D})$ and the marginal likelihood (or model evidence) $p(\mathcal{D})$, defined as, respectively,

$$
p(\boldsymbol{x} \mid \mathcal{D})=\frac{p(\mathcal{D} \mid \boldsymbol{x}) p(\boldsymbol{x})}{p(\mathcal{D})} \quad \text { and } \quad p(\mathcal{D})=\int p(\mathcal{D} \mid \boldsymbol{x}) p(\boldsymbol{x}) d \boldsymbol{x}
$$

where $p(\mathcal{D} \mid \boldsymbol{x})$ is the likelihood of the model of interest and $p(\boldsymbol{x})$ is the prior over parameters. Crucially, we consider the case in which $p(\mathcal{D} \mid \boldsymbol{x})$ is a black-box, expensive function for which we have a limited budget of function evaluations (of the order of few hundreds).
A promising approach to deal with such computational constraints consists of building a probabilistic model-based approximation of the function of interest, for example via Gaussian processes (GP)

[^0]
[^0]: \*Website: luigiacerbi.com. Alternative e-mail: luigi.acerbi@gmail.com.

---

#### Page 2

[1]. This statistical surrogate can be used in lieu of the original, expensive function, allowing faster computations. Moreover, uncertainty in the surrogate can be used to actively guide sampling of the original function to obtain a better approximation in regions of interest for the application at hand. This approach has been extremely successful in Bayesian optimization [2, 3, 4, 5, 6] and in Bayesian quadrature for the computation of intractable integrals [7, 8].
In particular, recent works have applied GP-based Bayesian quadrature to the estimation of the marginal likelihood [8, 9, 10, 11], and GP surrogates to build approximations of the posterior [12, 13]. However, none of the existing approaches deals simultaneously with posterior and model inference. Moreover, it is unclear how these approximate methods would deal with likelihoods with realistic properties, such as medium dimensionality (up to $D \sim 10$ ), mild multi-modality, heavy tails, and parameters that exhibit strong correlations-all common issues of real-world applications.
In this work, we introduce Variational Bayesian Monte Carlo (VBMC), a novel approximate inference framework that combines variational inference and active-sampling Bayesian quadrature via GP surrogates. ${ }^{1}$ Our method affords simultaneous approximation of the posterior and of the model evidence in a sample-efficient manner. We demonstrate the robustness of our approach by testing VBMC and other inference algorithms on a variety of synthetic likelihoods with realistic, challenging properties. We also apply our method to a real problem in computational neuroscience, by fitting a model of neuronal selectivity in visual cortex [14]. Among the tested methods, VBMC is the only one with consistently good performance across problems, showing promise as a novel tool for posterior and model inference with expensive likelihoods in scientific computing and machine learning.

# 2 Theoretical background

### 2.1 Variational inference

Variational Bayes is an approximate inference method whereby the posterior $p(\boldsymbol{x} \mid \mathcal{D})$ is approximated by a simpler distribution $q(\boldsymbol{x}) \equiv q_{\phi}(\boldsymbol{x})$ that usually belongs to a parametric family [15, 16]. The goal of variational inference is to find the variational parameters $\phi$ for which the variational posterior $q_{\phi}$ "best" approximates the true posterior. In variational methods, the mismatch between the two distributions is quantified by the Kullback-Leibler (KL) divergence,

$$
\mathrm{KL}\left[q_{\phi}(\boldsymbol{x}) \| p(\boldsymbol{x} \mid \mathcal{D})\right]=\mathbb{E}_{\phi}\left[\log \frac{q_{\phi}(\boldsymbol{x})}{p(\boldsymbol{x} \mid \mathcal{D})}\right]
$$

where we adopted the compact notation $\mathbb{E}_{\phi} \equiv \mathbb{E}_{q_{\phi}}$. Inference is then reduced to an optimization problem, that is finding the variational parameter vector $\phi$ that minimizes Eq. 2. We rewrite Eq. 2 as

$$
\log p(\mathcal{D})=\mathcal{F}\left[q_{\phi}\right]+\mathrm{KL}\left[q_{\phi}(\boldsymbol{x}) \| p(\boldsymbol{x} \mid \mathcal{D})\right]
$$

where

$$
\mathcal{F}\left[q_{\phi}\right]=\mathbb{E}_{\phi}\left[\log \frac{p(\mathcal{D} \mid \boldsymbol{x}) p(\boldsymbol{x})}{q_{\phi}(\boldsymbol{x})}\right]=\mathbb{E}_{\phi}[f(\boldsymbol{x})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{x})\right]
$$

is the negative free energy, or evidence lower bound (ELBO). Here $f(\boldsymbol{x}) \equiv \log p(\mathcal{D} \mid \boldsymbol{x}) p(\boldsymbol{x})=$ $\log p(\mathcal{D}, \boldsymbol{x})$ is the log joint probability and $\mathcal{H}[q]$ is the entropy of $q$. Note that since the KL divergence is always non-negative, from Eq. 3 we have $\mathcal{F}[q] \leq \log p(\mathcal{D})$, with equality holding if $q(\boldsymbol{x}) \equiv p(\boldsymbol{x} \mid \mathcal{D})$. Thus, maximization of the variational objective, Eq. 4, is equivalent to minimization of the KL divergence, and produces both an approximation of the posterior $q_{\phi}$ and a lower bound on the marginal likelihood, which can be used as a metric for model selection.
Normally, $q$ is chosen to belong to a family (e.g., a factorized posterior, or mean field) such that the expected log joint in Eq. 4 and the entropy can be computed analytically, possibly providing closed-form equations for a coordinate ascent algorithm. Here, we assume that $f(\boldsymbol{x})$, like many computational models of interest, is an expensive black-box function, which prevents a direct computation of Eq. 4 analytically or via simple numerical integration.

### 2.2 Bayesian quadrature

Bayesian quadrature, also known as Bayesian Monte Carlo, is a means to obtain Bayesian estimates of the mean and variance of non-analytical integrals of the form $\langle f\rangle=\int f(\boldsymbol{x}) \pi(\boldsymbol{x}) d \boldsymbol{x}$, defined on

[^0]
[^0]: ${ }^{1}$ Code available at https://github.com/acerbilab/vbmc.

---

#### Page 3

a domain $\mathcal{X}=\mathbb{R}^{D}[7,8]$. Here, $f$ is a function of interest and $\pi$ a known probability distribution. Typically, a Gaussian Process (GP) prior is specified for $f(\boldsymbol{x})$.
Gaussian processes GPs are a flexible class of models for specifying prior distributions over unknown functions $f: \mathcal{X} \subseteq \mathbb{R}^{D} \rightarrow \mathbb{R}$ [1]. GPs are defined by a mean function $m: \mathcal{X} \rightarrow \mathbb{R}$ and a positive definite covariance, or kernel function $\kappa: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$. In Bayesian quadrature, a common choice is the Gaussian kernel $\kappa\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{x}^{\prime}, \boldsymbol{\Sigma}_{\ell}\right)$, with $\boldsymbol{\Sigma}_{\ell}=\operatorname{diag}\left[\ell^{(1)^{2}}, \ldots, \ell^{(D)^{2}}\right]$, where $\sigma_{f}$ is the output length scale and $\ell$ is the vector of input length scales. Conditioned on training inputs $\mathbf{X}=\left\{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n}\right\}$ and associated function values $\boldsymbol{y}=f(\mathbf{X})$, the GP posterior will have latent posterior conditional mean $\bar{f}_{\boldsymbol{\Xi}}(\boldsymbol{x}) \equiv \bar{f}(\boldsymbol{x} ; \boldsymbol{\Xi}, \boldsymbol{\psi})$ and covariance $C_{\boldsymbol{\Xi}}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) \equiv C\left(\boldsymbol{x}, \boldsymbol{x}^{\prime} ; \boldsymbol{\Xi}, \boldsymbol{\psi}\right)$ in closed form (see [1]), where $\overline{\boldsymbol{\Xi}}=\{\mathbf{X}, \boldsymbol{y}\}$ is the set of training function data for the GP and $\boldsymbol{\psi}$ is a hyperparameter vector for the GP mean, covariance, and likelihood.

Bayesian integration Since integration is a linear operator, if $f$ is a GP, the posterior mean and variance of the integral $\int f(\boldsymbol{x}) \pi(\boldsymbol{x}) d \boldsymbol{x}$ are [8]

$$
\mathbb{E}_{f \mid \mathbb{E}}[\langle f\rangle]=\int \bar{f}_{\boldsymbol{\Xi}}(\boldsymbol{x}) \pi(\boldsymbol{x}) d \boldsymbol{x}, \quad \mathbb{V}_{f \mid \mathbb{E}}[\langle f\rangle]=\int \int C_{\boldsymbol{\Xi}}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) \pi(\boldsymbol{x}) d \boldsymbol{x} \pi\left(\boldsymbol{x}^{\prime}\right) d \boldsymbol{x}^{\prime}
$$

Crucially, if $f$ has a Gaussian kernel and $\pi$ is a Gaussian or mixture of Gaussians (among other functional forms), the integrals in Eq. 5 can be computed analytically.
Active sampling For a given budget of samples $n_{\text {max }}$, a smart choice of the input samples $\mathbf{X}$ would aim to minimize the posterior variance of the final integral (Eq. 5) [11]. Interestingly, for a standard GP and fixed GP hyperparameters $\boldsymbol{\psi}$, the optimal variance-minimizing design does not depend on the function values at $\mathbf{X}$, thereby allowing precomputation of the optimal design. However, if the GP hyperparameters are updated online, or the GP is warped (e.g., via a log transform [9] or a square-root transform [10]), the variance of the posterior will depend on the function values obtained so far, and an active sampling strategy is desirable. The acquisition function $a: \mathcal{X} \rightarrow \mathbb{R}$ determines which point in $\mathcal{X}$ should be evaluated next via a proxy optimization $\boldsymbol{x}_{\text {next }}=\operatorname{argmax}_{\boldsymbol{x}} a(\boldsymbol{x})$. Examples of acquisition functions for Bayesian quadrature include the expected entropy, which minimizes the expected entropy of the integral after adding $\boldsymbol{x}$ to the training set [9], and the much faster to compute uncertainty sampling strategy, which maximizes the variance of the integrand at $\boldsymbol{x}[10]$.

# 3 Variational Bayesian Monte Carlo (VBMC)

We introduce here Variational Bayesian Monte Carlo (VBMC), a sample-efficient inference method that combines variational Bayes and Bayesian quadrature, particularly useful for models with (moderately) expensive likelihoods. The main steps of VBMC are described in Algorithm 1, and an example run of VBMC on a nontrivial 2-D target density is shown in Fig. 1.

VBMC in a nutshell In each iteration $t$, the algorithm: (1) sequentially samples a batch of 'promising' new points that maximize a given acquisition function, and evaluates the (expensive) log joint $f$ at each of them; (2) trains a GP model of the log joint $f$, given the training set $\boldsymbol{\Xi}_{t}=\left\{\mathbf{X}_{t}, \boldsymbol{y}_{t}\right\}$ of points evaluated so far; (3) updates the variational posterior approximation, indexed by $\boldsymbol{\phi}_{t}$, by optimizing the ELBO. This loop repeats until the budget of function evaluations is exhausted, or some other termination criterion is met (e.g., based on the stability of the found solution). VBMC includes an initial warm-up stage to avoid spending computations in regions of low posterior probability mass (see Section 3.5). In the following sections, we describe various features of VBMC.

### 3.1 Variational posterior

We choose for the variational posterior $q(\boldsymbol{x})$ a flexible "nonparametric" family, a mixture of $K$ Gaussians with shared covariances, modulo a scaling factor,

$$
q(\boldsymbol{x}) \equiv q_{\boldsymbol{\phi}}(\boldsymbol{x})=\sum_{k=1}^{K} w_{k} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)
$$

where $w_{k}, \boldsymbol{\mu}_{k}$, and $\sigma_{k}$ are, respectively, the mixture weight, mean, and scale of the $k$-th component, and $\boldsymbol{\Sigma}$ is a covariance matrix common to all elements of the mixture. In the following, we assume

---

#### Page 4

```
Algorithm 1 Variational Bayesian Monte Carlo
Input: target log joint \(f\), starting point \(\boldsymbol{x}_{0}\), plausible bounds PLB, PUB, additional options
    Initialization: \(t \leftarrow 0\), initialize variational posterior \(\phi_{0}\), StOPSAMPLING \(\leftarrow\) false
    repeat
        \(t \leftarrow t+1\)
        if \(t \triangleq 1\) then \(\triangleright\) Initial design, Section 3.5
            Evaluate \(y_{0} \leftarrow f\left(\boldsymbol{x}_{0}\right)\) and add \(\left(\boldsymbol{x}_{0}, y_{0}\right)\) to the training set \(\boldsymbol{\Xi}\)
            for \(2 \ldots n_{\text {init }}\) do
                Sample a new point \(\boldsymbol{x}_{\text {new }} \leftarrow\) Uniform[PLB, PUB]
                Evaluate \(y_{\text {new }} \leftarrow f\left(\boldsymbol{x}_{\text {new }}\right)\) and add \(\left(\boldsymbol{x}_{\text {new }}, y_{\text {new }}\right)\) to the training set \(\boldsymbol{\Xi}\)
        else
            for \(1 \ldots n_{\text {active }}\) do \(\triangleright\) Active sampling, Section 3.3
                Actively sample a new point \(\boldsymbol{x}_{\text {new }} \leftarrow \operatorname{argmax}_{\boldsymbol{x}} a(\boldsymbol{x})\)
                Evaluate \(y_{\text {new }} \leftarrow f\left(\boldsymbol{x}_{\text {new }}\right)\) and add \(\left(\boldsymbol{x}_{\text {new }}, y_{\text {new }}\right)\) to the training set \(\boldsymbol{\Xi}\)
                for each \(\psi_{1}, \ldots, \psi_{n_{\text {sp }}}$, perform rank-1 update of the GP posterior
            if not StOPSAMPLING then \(\triangleright\) GP hyperparameter training, Section 3.4
                \(\left\{\psi_{1}, \ldots, \psi_{n_{\text {sp }}}\right\} \leftarrow\) Sample GP hyperparameters
            else
                \(\psi_{1} \leftarrow\) Optimize GP hyperparameters
            \(K_{t} \leftarrow\) Update number of variational components \(\triangleright\) Section 3.6
            \(\phi_{t} \leftarrow\) Optimize ELBO via stochastic gradient descent \(\triangleright\) Section 3.2
            Evaluate whether to StOPSAMPLING and other TERMINATIONCRITERIA
    until fevals \(>\) MaxFunEvals or TERMINATIONCRITERIA \(\triangleright\) Stopping criteria, Section 3.7
    return variational posterior \(\phi_{t}, \mathbb{E}[\) ELBO \(], \sqrt{\mathbb{V}[\) ELBO \(]}\)
```

a diagonal matrix $\boldsymbol{\Sigma} \equiv \operatorname{diag}\left[\lambda^{(1)^{2}}, \ldots, \lambda^{(D)^{2}}\right]$. The variational posterior for a given number of mixture components $K$ is parameterized by $\phi \equiv\left(w_{1}, \ldots, w_{K}, \boldsymbol{\mu}_{1}, \ldots, \boldsymbol{\mu}_{K}, \sigma_{1}, \ldots, \sigma_{K}, \boldsymbol{\lambda}\right)$, which has $K(D+2)+D$ parameters. The number of components $K$ is set adaptively (see Section 3.6).

# 3.2 The evidence lower bound

We approximate the ELBO (Eq. 4) in two ways. First, we approximate the log joint probability $f$ with a GP with a squared exponential (rescaled Gaussian) kernel, a Gaussian likelihood with observation noise $\sigma_{o b n}>0$ (for numerical stability [17]), and a negative quadratic mean function, defined as

$$
m(\boldsymbol{x})=m_{0}-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(x^{(i)}-x_{\mathrm{m}}^{(i)}\right)^{2}}{\omega^{(i)^{2}}}
$$

where $m_{0}$ is the maximum value of the mean, $\boldsymbol{x}_{\mathrm{m}}$ is the location of the maximum, and $\boldsymbol{\omega}$ is a vector of length scales. This mean function, unlike for example a constant mean, ensures that the posterior GP predictive mean $\bar{f}$ is a proper log probability distribution (that is, it is integrable when exponentiated). Crucially, our choice of variational family (Eq. 6) and kernel, likelihood and mean function of the GP affords an analytical computation of the posterior mean and variance of the expected log joint $\mathbb{E}_{\phi}[f]$ (using Eq. 5), and of their gradients (see Supplementary Material for details). Second, we approximate the entropy of the variational posterior, $\mathcal{H}\left[q_{\phi}\right]$, via simple Monte Carlo sampling, and we propagate its gradient through the samples via the reparametrization trick [18, 19]. ${ }^{2}$ Armed with expressions for the mean expected log joint, the entropy, and their gradients, we can efficiently optimize the (negative) mean ELBO via stochastic gradient descent [21].
Evidence lower confidence bound We define the evidence lower confidence bound (ELCBO) as

$$
\operatorname{ELCBO}(\phi, f)=\mathbb{E}_{f \mid \mathbb{E}}\left[\mathbb{E}_{\phi}[f]\right]+\mathcal{H}\left[q_{\phi}\right]-\beta_{\mathrm{LCB}} \sqrt{\mathbb{V}_{f \mid \mathbb{E}}\left[\mathbb{E}_{\phi}[f]\right]}
$$

where the first two terms are the ELBO (Eq. 4) estimated via Bayesian quadrature, and the last term is the uncertainty in the computation of the expected log joint multiplied by a risk-sensitivity parameter

[^0]
[^0]: ${ }^{2}$ We also tried a deterministic approximation of the entropy proposed in [20], with mixed results.

---

#### Page 5

> **Image description.** The image is a figure composed of multiple panels showing the progression of a Variational Bayesian Monte Carlo (VBMC) algorithm.
>
> Panel A consists of six subplots arranged in two rows, each representing a different iteration of the algorithm (Iteration 1, Iteration 5 (end of warm-up), Iteration 8, Iteration 11, Iteration 14, and Iteration 17). Each subplot displays a 2D space with axes labeled x1 and x2. Within each subplot, there are:
> _ Contour plots: These are multi-colored, concentric lines indicating the probability density of the variational posterior. The color gradient ranges from blue (outermost) to yellow/white (innermost), suggesting increasing density.
> _ Red crosses: These indicate the centers of the variational mixture components. \* Black dots: These represent the training samples.
> The progression across the iterations shows the contour plots becoming more complex and fitting the distribution of the training samples more closely.
>
> Panel B is a line graph titled "Model evidence". The x-axis is labeled "Iterations" and ranges from 1 to 17. The y-axis ranges from -4 to 3. There are two lines plotted:
> _ ELBO (red): This line represents the Evidence Lower Bound. It starts high, drops sharply, and then gradually increases and plateaus. A shaded red area around the line indicates the 95% confidence interval of the ELBO.
> _ LML (black): This is a horizontal line representing the true log marginal likelihood, labeled with a value of approximately -2.27.
>
> Panel C is a contour plot titled "True posterior". It shows a 2D space with axes labeled x1 and x2. The plot contains blue contour lines representing the true target probability density function (pdf). The shape resembles a U-shaped valley.

Figure 1: Example run of VBMC on a 2-D pdf. A. Contour plots of the variational posterior at different iterations of the algorithm. Red crosses indicate the centers of the variational mixture components, black dots are the training samples. B. ELBO as a function of iteration. Shaded area is $95 \%$ CI of the ELBO in the current iteration as per the Bayesian quadrature approximation (not the error wrt ground truth). The black line is the true log marginal likelihood (LML). C. True target pdf.

$\beta_{\text {LCB }}$ (we set $\beta_{\text {LCB }}=3$ unless specified otherwise). Eq. 8 establishes a probabilistic lower bound on the ELBO, used to assess the improvement of the variational solution (see following sections).

# 3.3 Active sampling

In VBMC, we are performing active sampling to compute a sequence of integrals $\mathbb{E}_{\boldsymbol{\phi}_{1}}[f], \mathbb{E}_{\boldsymbol{\phi}_{2}}[f], \ldots, \mathbb{E}_{\boldsymbol{\phi}_{T}}[f]$, across iterations $1, \ldots, T$ such that (1) the sequence of variational parameters $\phi_{t}$ converges to the variational posterior that minimizes the KL divergence with the true posterior, and (2) we have minimum variance on our final estimate of the ELBO. Note how this differs from active sampling in simple Bayesian quadrature, for which we only care about minimizing the variance of a single fixed integral. The ideal acquisition function for VBMC will correctly balance exploration of uncertain regions and exploitation of regions with high probability mass to ensure a fast convergence of the variational posterior as closely as possible to the ground truth.
We describe here two acquisition functions for VBMC based on uncertainty sampling. Let $V_{\Xi}(\boldsymbol{x}) \equiv$ $C_{\Xi}(\boldsymbol{x}, \boldsymbol{x})$ be the posterior GP variance at $\boldsymbol{x}$ given the current training set $\boldsymbol{\Xi}$. 'Vanilla' uncertainty sampling for $\mathbb{E}_{\boldsymbol{\phi}}[f]$ is $a_{\mathrm{uv}}(\boldsymbol{x})=V_{\Xi}(\boldsymbol{x}) q_{\boldsymbol{\phi}}(\boldsymbol{x})^{2}$, where $q_{\phi}$ is the current variational posterior. Since $a_{\mathrm{uv}}$ only maximizes the variance of the integrand under the current variational parameters, we expect it to be lacking in exploration. To promote exploration, we introduce prospective uncertainty sampling,

$$
a_{\mathrm{pro}}(\boldsymbol{x})=V_{\Xi}(\boldsymbol{x}) q_{\boldsymbol{\phi}}(\boldsymbol{x}) \exp \left(\bar{f}_{\Xi}(\boldsymbol{x})\right)
$$

where $\bar{f}_{\Xi}$ is the GP posterior predictive mean. $a_{\text {pro }}$ aims at reducing uncertainty of the variational objective both for the current posterior and at prospective locations where the variational posterior might move to in the future, if not already there (high GP posterior mean). The variational posterior in $a_{\text {pro }}$ acts as a regularizer, preventing active sampling from following too eagerly fluctuations of the GP mean. For numerical stability of the GP, we include in all acquisition functions a regularization factor to prevent selection of points too close to existing training points (see Supplementary Material).
At the beginning of each iteration after the first, VBMC actively samples $n_{\text {active }}$ points ( $n_{\text {active }}=5$ by default in this work). We select each point sequentially, by optimizing the chosen acquisition function via CMA-ES [22], and apply fast rank-one updates of the GP posterior after each acquisition.

### 3.4 Adaptive treatment of GP hyperparameters

The GP model in VBMC has $3 D+3$ hyperparameters, $\boldsymbol{\psi}=\left(\boldsymbol{\ell}, \sigma_{f}, \sigma_{c b n}, m_{0}, \boldsymbol{x}_{\mathrm{m}}, \boldsymbol{\omega}\right)$. We impose an empirical Bayes prior on the GP hyperparameters based on the current training set (see Supplementary

---

#### Page 6

Material), and we sample from the posterior over hyperparameters via slice sampling [23]. In each iteration, we collect $n_{\text {gp }}=$ round $(80 / \sqrt{n})$ samples, where $n$ is the size of the current GP training set, with the rationale that we require less samples as the posterior over hyperparameters becomes narrower due to more observations. Given samples $\{\boldsymbol{\psi}\} \equiv\left\{\boldsymbol{\psi}_{1}, \ldots, \boldsymbol{\psi}_{n_{\mathrm{gp}}}\right\}$, and a random variable $\chi$ that depends on $\boldsymbol{\psi}$, we compute the expected mean and variance of $\chi$ as

$$
\mathbb{E}[\chi \mid\{\boldsymbol{\psi}\}]=\frac{1}{n_{\mathrm{gp}}} \sum_{j=1}^{n_{\mathrm{gp}}} \mathbb{E}\left[\chi \mid \boldsymbol{\psi}_{j}\right], \quad \mathbb{V}[\chi \mid\{\boldsymbol{\psi}\}]=\frac{1}{n_{\mathrm{gp}}} \sum_{j=1}^{n_{\mathrm{gp}}} \mathbb{V}\left[\chi \mid \boldsymbol{\psi}_{j}\right]+\operatorname{Var}\left[\left\{\mathbb{E}\left[\chi \mid \boldsymbol{\psi}_{j}\right]\right\}_{j=1}^{n_{\mathrm{gp}}}\right]
$$

where $\operatorname{Var}[\cdot]$ is the sample variance. We use Eq. 10 to compute the GP posterior predictive mean and variances for the acquisition function, and to marginalize the expected log joint over hyperparameters.
The algorithm adaptively switches to a faster maximum-a-posteriori (MAP) estimation of the hyperparameters (via gradient-based optimization) when the additional variability of the expected log joint brought by multiple samples falls below a threshold for several iterations, a signal that sampling is bringing little advantage to the precision of the computation.

# 3.5 Initialization and warm-up

The algorithm is initialized by providing a starting point $\boldsymbol{x}_{0}$ (ideally, in a region of high posterior probability mass) and vectors of plausible lower/upper bounds PLB, PUB, that identify a region of high posterior probability mass in parameter space. In the absence of other information, we obtained good results with plausible bounds containing the peak of prior mass in each coordinate dimension, such as the top $\sim 0.68$ probability region (that is, mean $\pm 1 \mathrm{SD}$ for a Gaussian prior). The initial design consists of the provided starting point(s) $\boldsymbol{x}_{0}$ and additional points generated uniformly at random inside the plausible box, for a total of $n_{\text {init }}=10$ points. The plausible box also sets the reference scale for each variable, and in future work might inform other aspects of the algorithm [6]. The VBMC algorithm works in an unconstrained space $\left(\boldsymbol{x} \in \mathbb{R}^{D}\right.$ ), but bound constraints to the variables can be easily handled via a nonlinear remapping of the input space, with an appropriate Jacobian correction of the log probability density [24] (see Section 4.2 and Supplementary Material). ${ }^{3}$
Warm-up We initialize the variational posterior with $K=2$ components in the vicinity of $\boldsymbol{x}_{0}$, and with small values of $\sigma_{1}, \sigma_{2}$, and $\boldsymbol{\lambda}$ (relative to the width of the plausible box). The algorithm starts in warm-up mode, during which VBMC tries to quickly improve the ELBO by moving to regions with higher posterior probability. During warm-up, $K$ is clamped to only two components with $w_{1} \equiv w_{2}=1 / 2$, and we collect a maximum of $n_{\mathrm{gp}}=8$ hyperparameter samples. Warm-up ends when the ELCBO (Eq. 8) shows an improvement of less than 1 for three consecutive iterations, suggesting that the variational solution has started to stabilize. At the end of warm-up, we trim the training set by removing points whose value of the log joint probability $y$ is more than $10 \cdot D$ points lower than the maximum value $y_{\max }$ observed so far. While not necessary in theory, we found that trimming generally increases the stability of the GP approximation, especially when VBMC is initialized in a region of very low probability under the true posterior. To allow the variational posterior to adapt, we do not actively sample new points in the first iteration after the end of warm-up.

### 3.6 Adaptive number of variational mixture components

After warm-up, we add and remove variational components following a simple set of rules.
Adding components We define the current variational solution as improving if the ELCBO of the last iteration is higher than the ELCBO in the past few iterations $\left(n_{\text {recent }}=4\right)$. In each iteration, we increment the number of components $K$ by 1 if the solution is improving and no mixture component was pruned in the last iteration (see below). To speed up adaptation of the variational solution to a complex true posterior when the algorithm has nearly converged, we further add two extra components if the solution is stable (see below) and no component was recently pruned. Each new component is created by splitting and jittering a randomly chosen existing component. We set a maximum number of components $K_{\max }=n^{2 / 3}$, where $n$ is the size of the current training set $\mathbb{E}$.

Removing components At the end of each variational optimization, we consider as a candidate for pruning a random mixture component $k$ with mixture weight $w_{k}<w_{\min }$. We recompute the ELCBO

[^0]
[^0]: ${ }^{3}$ The available code for VBMC currently supports both unbounded variables and bound constraints.

---

#### Page 7

without the selected component (normalizing the remaining weights). If the 'pruned' ELCBO differs from the original ELCBO less than $\varepsilon$, we remove the selected component. We iterate the process through all components with weights below threshold. For VBMC we set $w_{\min }=0.01$ and $\varepsilon=0.01$.

# 3.7 Termination criteria

At the end of each iteration, we assign a reliability index $\rho(t) \geq 0$ to the current variational solution based on the following features: change in ELBO between the current and the previous iteration; estimated variance of the ELBO; KL divergence between the current and previous variational posterior (see Supplementary Material for details). By construction, a $\rho(t) \lesssim 1$ is suggestive of a stable solution. The algorithm terminates when obtaining a stable solution for $n_{\text {stable }}=8$ iterations (with at most one non-stable iteration in-between), or when reaching a maximum number $n_{\max }$ of function evaluations. The algorithm returns the estimate of the mean and standard deviation of the ELBO (a lower bound on the marginal likelihood), and the variational posterior, from which we can cheaply draw samples for estimating distribution moments, marginals, and other properties of the posterior. If the algorithm terminates before achieving long-term stability, it warns the user and returns a recent solution with the best ELCBO, using a conservative $\beta_{\mathrm{LCB}}=5$.

## 4 Experiments

We tested VBMC and other common inference algorithms on several artificial and real problems consisting of a target likelihood and an associated prior. The goal of inference consists of approximating the posterior distribution and the log marginal likelihood (LML) with a fixed budget of likelihood evaluations, assumed to be (moderately) expensive.

Algorithms We tested VBMC with the 'vanilla' uncertainty sampling acquisition function $a_{\text {un }}$ (VBMC-U) and with prospective uncertainty sampling, $a_{\text {pos }}$ (VBMC-P). We also tested simple Monte Carlo (SMC), annealed importance sampling (AIS), the original Bayesian Monte Carlo (BMC), doubly-Bayesian quadrature (BBQ [9]) ${ }^{4}$, and warped sequential active Bayesian integration (WSABI, both in its linearized and moment-matching variants, WSABI-L and WSABI-M [10]). For the basic setup of these methods, we follow [10]. Most of these algorithms only compute an approximation of the marginal likelihood based on a set of sampled points, but do not directly compute a posterior distribution. We obtain a posterior by training a GP model (equal to the one used by VBMC) on the log joint evaluated at the sampled points, and then drawing $2 \cdot 10^{4} \mathrm{MCMC}$ samples from the GP posterior predictive mean via parallel slice sampling [23, 25]. We also tested two methods for posterior estimation via GP surrogates, BAPE [12] and AGP [13]. Since these methods only compute an approximate posterior, we obtain a crude estimate of the log normalization constant (the LML) as the average difference between the log of the approximate posterior and the evaluated log joint at the top $20 \%$ points in terms of posterior density. For all algorithms, we use default settings, allowing only changes based on knowledge of the mean and (diagonal) covariance of the provided prior.

Procedure For each problem, we allow a fixed budget of $50 \times(D+2)$ likelihood evaluations, where $D$ is the number of variables. Given the limited number of samples, we judge the quality of the posterior approximation in terms of its first two moments, by computing the "Gaussianized" symmetrized KL divergence (gsKL) between posterior approximation and ground truth. The gsKL is defined as the symmetrized KL between two multivariate normal distributions with mean and covariances equal, respectively, to the moments of the approximate posterior and the moments of the true posterior. We measure the quality of the approximation of the LML in terms of absolute error from ground truth, the rationale being that differences of LML are used for model comparison. Ideally, we want the LML error to be of order 1 of less, since much larger errors could severely affect the results of a comparison (e.g., differences of LML of 10 points or more are often presented as decisive evidence in favor of one model [26]). On the other hand, errors $\lesssim 0.1$ can be considered negligible; higher precision is unnecessary. For each algorithm, we ran at least 20 separate runs per test problem with different random seeds, and report the median gsKL and LML error and the $95 \%$ CI of the median calculated by bootstrap. For each run, we draw the starting point $\boldsymbol{x}_{0}$ (if requested by the algorithm) uniformly from a box within 1 prior standard deviation (SD) from the prior mean. We use the same box to define the plausible bounds for VBMC.

[^0]
[^0]: ${ }^{4}$ We also tested $\mathrm{BBQ}^{*}$ (approximate GP hyperparameter marginalization), which perfomed similarly to BBQ.

---

#### Page 8

4.1 Synthetic likelihoods

Problem set We built a benchmark set of synthetic likelihoods belonging to three families that represent typical features of target densities (see Supplementary Material for details). Likelihoods in the lumpy family are built out of a mixture of 12 multivariate normals with component means drawn randomly in the unit $D$-hypercube, distinct diagonal covariances with SDs in the $[0.2,0.6]$ range, and mixture weights drawn from a Dirichlet distribution with unit concentration parameter. The lumpy distributions are mildly multimodal, in that modes are nearby and connected by regions with non-neglibile probability mass. In the Student family, the likelihood is a multivariate Student's $t$-distribution with diagonal covariance and degrees of freedom equally spaced in the $[2.5,2+D / 2]$ range across different coordinate dimensions. These distributions have heavy tails which might be problematic for some methods. Finally, in the cigar family the likelihood is a multivariate normal in which one axis is 100 times longer than the others, and the covariance matrix is non-diagonal after a random rotation. The cigar family tests the ability of an algorithm to explore non axis-aligned directions. For each family, we generated test functions for $D \in\{2,4,6,8,10\}$, for a total of 15 synthetic problems. For each problem, we pick as a broad prior a multivariate normal with mean centered at the expected mean of the family of distributions, and diagonal covariance matrix with SD equal to 3-4 times the SD in each dimension. For all problems, we compute ground truth values for the LML and the posterior mean and covariance analytically or via multiple 1-D numerical integrals.

Results We show the results for $D \in\{2,6,10\}$ in Fig. 2 (see Supplementary Material for full results, in higher resolution). Almost all algorithms perform reasonably well in very low dimension ( $D=2$ ), and in fact several algorithms converge faster than VBMC to the ground truth (e.g., WSABIL). However, as we increase in dimension, we see that all algorithms start failing, with only VBMC peforming consistently well across problems. In particular, besides the simple $D=2$ case, only VBMC obtains acceptable results for the LML with non-axis aligned distributions (cigar). Some algorithms (such as AGP and BAPE) exhibited large numerical instabilities on the cigar family, despite our best attempts at regularization, such that many runs were unable to complete.

> **Image description.** The image consists of two panels, A and B, each containing three rows of line graphs. Each row corresponds to a different likelihood function: "Lumpy", "Student", and "Cigar". Within each row, there are three graphs representing different dimensions: "2D", "6D", and "10D".
>
> **Panel A:**
>
> - **Title:** "A"
> - **Y-axis Label:** "Median LML error" (log scale from 10^-4 to 10^4)
> - **X-axis Label:** "Function evaluations" (linear scale from 0 to 200, 400, or 600 depending on the graph)
> - **Graphs:** Each graph displays multiple lines, each representing a different algorithm. A legend to the right of the graphs lists the algorithms with corresponding line styles and colors:
>   - smc (dotted gray)
>   - ais (orange)
>   - bmc (red)
>   - wsabi-L (purple)
>   - wsabi-M (blue)
>   - bbq (cyan)
>   - agp (green)
>   - bape (olive green, dashed)
>   - vbmc-U (dark red)
>   - vbmc-P (black, thick)
> - A horizontal dashed line is present at y=1.
>
> **Panel B:**
>
> - **Title:** "B"
> - **Y-axis Label:** "Median gsKL" (log scale from 10^-4 to 10^6)
> - **X-axis Label:** "Function evaluations" (linear scale from 0 to 200, 400, or 600 depending on the graph)
> - **Graphs:** Similar to Panel A, each graph displays multiple lines representing different algorithms, using the same color scheme as in Panel A.
> - A horizontal dashed line is present at y=1.
>
> In both panels, the graphs show the performance of different algorithms as a function of function evaluations for different likelihood functions and dimensions. The y-axis represents the error metric (LML error in Panel A, gsKL in Panel B).

Figure 2: Synthetic likelihoods. A. Median absolute error of the LML estimate with respect to ground truth, as a function of number of likelihood evaluations, on the lumpy (top), Student (middle), and cigar (bottom) problems, for $D \in\{2,6,10\}$ (columns). B. Median "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth. For both metrics, shaded areas are $95 \%$ CI of the median, and we consider a desirable threshold to be below one (dashed line).

# 4.2 Real likelihoods of neuronal model

Problem set For a test with real models and data, we consider a computational model of neuronal orientation selectivity in visual cortex [14]. We fit the neural recordings of one V1 and one V2 cell with the authors' neuronal model that combines effects of filtering, suppression, and response nonlinearity [14]. The model is analytical but still computationally expensive due to large datasets and a cascade of several nonlinear operations. For the purpose of our benchmark, we fix some parameters of the original model to their MAP values, yielding an inference problem with $D=7$ free

---

#### Page 9

parameters of experimental interest. We transform bounded parameters to uncontrained space via a logit transform [24], and we place a broad Gaussian prior on each of the transformed variables, based on estimates from other neurons in the same study [14] (see Supplementary Material for more details on the setup). For both datasets, we computed the ground truth with $4 \cdot 10^{5}$ samples from the posterior, obtained via parallel slice sampling after a long burn-in. We calculated the ground truth LML from posterior MCMC samples via Geyer's reverse logistic regression [27], and we independently validated it with a Laplace approximation, obtained via numerical calculation of the Hessian at the MAP (for both datasets, Geyer's and Laplace's estimates of the LML are within $\sim 1$ point).

> **Image description.** This image contains two sets of line graphs, labeled A and B, each with two panels labeled V1 and V2.
>
> **Panel A:**
>
> - **Type:** Line graphs.
> - **Title:** Neuronal model.
> - **Y-axis:** "Median LML error", with a logarithmic scale ranging from 10<sup>-2</sup> to 10<sup>4</sup>.
> - **X-axis:** "Function evaluations", ranging from 0 to 400.
> - **Data:** Multiple colored lines representing different algorithms:
>   - smc (dashed gray)
>   - ais (dashed tan)
>   - bmc (pink)
>   - wsabi-L (red)
>   - wsabi-M (blue)
>   - bbq (purple)
>   - agp (teal)
>   - bape (dashed green)
>   - vbmc-U (dark pink)
>   - vbmc-P (black)
> - A horizontal dashed line is present at y=1.
> - The V1 and V2 panels show similar data, but with different performance characteristics for the algorithms.
>
> **Panel B:**
>
> - **Type:** Line graphs.
> - **Y-axis:** "Median gsKL", with a logarithmic scale ranging from 10<sup>-2</sup> to 10<sup>6</sup>.
> - **X-axis:** "Function evaluations", ranging from 0 to 400.
> - **Data:** Uses the same color scheme and algorithms as Panel A. The lines are more erratic, especially for the algorithms that perform poorly. Shaded regions around some lines indicate variability.
> - A horizontal dashed line is present at y=1.
> - The V1 and V2 panels show similar data, but with different performance characteristics for the algorithms.
>
> **Overall:**
>
> The graphs compare the performance of different algorithms (smc, ais, bmc, wsabi-L, wsabi-M, bbq, agp, bape, vbmc-U, and vbmc-P) in terms of "Median LML error" and "Median gsKL" as a function of "Function evaluations" for two datasets V1 and V2. The logarithmic scale on the y-axes allows for visualizing a wide range of values. The vbmc-U and vbmc-P algorithms appear to perform the best, showing the most rapid decrease in error and gsKL.

Figure 3: Neuronal model likelihoods. A. Median absolute error of the LML estimate, as a function of number of likelihood evaluations, for two distinct neurons $(D=7)$. B. Median "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth. See also Fig. 2.

Results For both datasets, VBMC is able to find a reasonable approximation of the LML and of the posterior, whereas no other algorithm produces a usable solution (Fig. 3). Importantly, the behavior of VBMC is fairly consistent across runs (see Supplementary Material). We argue that the superior results of VBMC stem from a better exploration of the posterior landscape, and from a better approximation of the log joint (used in the ELBO), related but distinct features. To show this, we first trained GPs (as we did for the other methods) on the samples collected by VBMC (see Supplementary Material). The posteriors obtained by sampling from the GPs trained on the VBMC samples scored a better gsKL than the other methods (and occasionally better than VBMC itself). Second, we estimated the marginal likelihood with WSABI-L using the samples collected by VBMC. The LML error in this hybrid approach is much lower than the error of WSABI-L alone, but still higher than the LML error of VBMC. These results combined suggest that VBMC builds better (and more stable) surrogate models and obtains higher-quality samples than the compared methods.
The performance of VBMC-U and VBMC-P is similar on synthetic functions, but the 'prospective' acquisition function converges faster on the real problem set, so we recommend $a_{\text {pro }}$ as the default. Besides scoring well on quantitative metrics, VBMC is able to capture nontrivial features of the true posteriors (see Supplementary Material for examples). Moreover, VBMC achieves these results with a relatively small computational cost (see Supplementary Material for discussion).

# 5 Conclusions

In this paper, we have introduced VBMC, a novel Bayesian inference framework that combines variational inference with active-sampling Bayesian quadrature for models with expensive black-box likelihoods. Our method affords both posterior estimation and model inference by providing an approximate posterior and a lower bound to the model evidence. We have shown on both synthetic and real model-fitting problems that, given a contained budget of likelihood evaluations, VBMC is able to reliably compute valid, usable approximations in realistic scenarios, unlike previous methods whose applicability seems to be limited to very low dimension or simple likelihoods. Our method, thus, represents a novel useful tool for approximate inference in science and engineering.
We believe this is only the starting point to harness the combined power of variational inference and Bayesian quadrature. Not unlike the related field of Bayesian optimization, VBMC paves the way to a plenitude of both theoretical (e.g., analysis of convergence, development of principled acquisition functions) and applied work (e.g., application to case studies of interest, extension to noisy likelihood evaluations, algorithmic improvements), which we plan to pursue as future directions.

---

# Variational Bayesian Monte Carlo - Backmatter

---

#### Page 10

# Acknowledgments 

We thank Robbe Goris for sharing data and code for the neuronal model; Michael Schartner and Rex Liu for comments on an earlier version of the paper; and three anonymous reviewers for useful feedback.

## References

[1] Rasmussen, C. \& Williams, C. K. I. (2006) Gaussian Processes for Machine Learning. (MIT Press).
[2] Jones, D. R., Schonlau, M., \& Welch, W. J. (1998) Efficient global optimization of expensive black-box functions. Journal of Global Optimization 13, 455-492.
[3] Brochu, E., Cora, V. M., \& De Freitas, N. (2010) A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. arXiv preprint arXiv:1012.2599.
[4] Snoek, J., Larochelle, H., \& Adams, R. P. (2012) Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems 25, 2951-2959.
[5] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., \& de Freitas, N. (2016) Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE 104, 148-175.
[6] Acerbi, L. \& Ma, W. J. (2017) Practical Bayesian optimization for model fitting with Bayesian adaptive direct search. Advances in Neural Information Processing Systems 30, 1834-1844.
[7] O’Hagan, A. (1991) Bayes-Hermite quadrature. Journal of Statistical Planning and Inference 29, 245-260.
[8] Ghahramani, Z. \& Rasmussen, C. E. (2002) Bayesian Monte Carlo. Advances in Neural Information Processing Systems 15, 505-512.
[9] Osborne, M., Duvenaud, D. K., Garnett, R., Rasmussen, C. E., Roberts, S. J., \& Ghahramani, Z. (2012) Active learning of model evidence using Bayesian quadrature. Advances in Neural Information Processing Systems 25, 46-54.
[10] Gunter, T., Osborne, M. A., Garnett, R., Hennig, P., \& Roberts, S. J. (2014) Sampling for inference in probabilistic models with fast Bayesian quadrature. Advances in Neural Information Processing Systems 27, 2789-2797.
[11] Briol, F.-X., Oates, C., Girolami, M., \& Osborne, M. A. (2015) Frank-Wolfe Bayesian quadrature: Probabilistic integration with theoretical guarantees. Advances in Neural Information Processing Systems 28, 1162-1170.
[12] Kandasamy, K., Schneider, J., \& Póczos, B. (2015) Bayesian active learning for posterior estimation. Twenty-Fourth International Joint Conference on Artificial Intelligence.
[13] Wang, H. \& Li, J. (2018) Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions. Neural Computation pp. 1-23.
[14] Goris, R. L., Simoncelli, E. P., \& Movshon, J. A. (2015) Origin and function of tuning diversity in macaque visual cortex. Neuron 88, 819-831.
[15] Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., \& Saul, L. K. (1999) An introduction to variational methods for graphical models. Machine Learning 37, 183-233.
[16] Bishop, C. M. (2006) Pattern Recognition and Machine Learning. (Springer).
[17] Gramacy, R. B. \& Lee, H. K. (2012) Cases for the nugget in modeling computer experiments. Statistics and Computing 22, 713-722.
[18] Kingma, D. P. \& Welling, M. (2013) Auto-encoding variational Bayes. Proceedings of the 2nd International Conference on Learning Representations.
[19] Miller, A. C., Foti, N., \& Adams, R. P. (2017) Variational boosting: Iteratively refining posterior approximations. Proceedings of the 34th International Conference on Machine Learning 70, 2420-2429.
[20] Gershman, S., Hoffman, M., \& Blei, D. (2012) Nonparametric variational inference. Proceedings of the 29th International Coference on Machine Learning.
[21] Kingma, D. P. \& Ba, J. (2014) Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations.
[22] Hansen, N., Müller, S. D., \& Koumoutsakos, P. (2003) Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES). Evolutionary Computation 11, 1-18.
[23] Neal, R. M. (2003) Slice sampling. Annals of Statistics 31, 705-741.

---

#### Page 11

[24] Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., \& Riddell, A. (2017) Stan: A probabilistic programming language. Journal of Statistical Software 76.
[25] Gilks, W. R., Roberts, G. O., \& George, E. I. (1994) Adaptive direction sampling. The Statistician 43, $179-189$.
[26] Kass, R. E. \& Raftery, A. E. (1995) Bayes factors. Journal of the American Statistical Association 90, $773-795$.
[27] Geyer, C. J. (1994) Estimating normalizing constants and reweighting mixtures. (Technical report).
[28] Knuth, D. E. (1992) Two notes on notation. The American Mathematical Monthly 99, 403-422.
[29] Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., \& Rubin, D. B. (2013) Bayesian Data Analysis (3rd edition). (CRC Press).
[30] Yao, Y., Vehtari, A., Simpson, D., \& Gelman, A. (2018) Yes, but did it work?: Evaluating variational inference. arXiv preprint arXiv:1802.02538.

---

# Variational Bayesian Monte Carlo - Appendix

---

#### Page 12

# Supplementary Material

In this Supplement we include a number of derivations, implementation details, and additional results omitted from the main text.

Code used to generate the results in the paper is available at https://github.com/lacerbi/infbench. The VBMC algorithm is available at https://github.com/acerbilab/vbmc.

## Contents

A Computing and optimizing the ELBO ..... 13
A. 1 Stochastic approximation of the entropy ..... 13
A.1.1 Gradient of the entropy ..... 13
A. 2 Expected log joint ..... 14
A.2.1 Posterior mean of the integral and its gradient ..... 15
A.2.2 Posterior variance of the integral ..... 15
A.2.3 Negative quadratic mean function ..... 16
A. 3 Optimization of the approximate ELBO ..... 16
A.3.1 Reparameterization ..... 16
A.3.2 Choice of starting points ..... 16
A.3.3 Stochastic gradient descent ..... 16
B Algorithmic details ..... 17
B. 1 Regularization of acquisition functions ..... 17
B. 2 GP hyperparameters and priors ..... 17
B. 3 Transformation of variables ..... 17
B. 4 Termination criteria ..... 18
B.4.1 Reliability index ..... 18
B.4.2 Long-term stability termination condition ..... 18
B.4.3 Validation of VBMC solutions ..... 19
C Experimental details and additional results ..... 19
C. 1 Synthetic likelihoods ..... 19
C. 2 Neuronal model ..... 20
C.2.1 Model parameters ..... 20
C.2.2 True and approximate posteriors ..... 20
D Analysis of VBMC ..... 23
D. 1 Variability between VBMC runs ..... 23
D. 2 Computational cost ..... 23
D. 3 Analysis of the samples produced by VBMC ..... 24

---

#### Page 13

# A Computing and optimizing the ELBO

For ease of reference, we recall the expression for the ELBO, for $\boldsymbol{x} \in \mathbb{R}^{D}$,

$$
\mathcal{F}\left[q_{\boldsymbol{\phi}}\right]=\mathbb{E}_{\boldsymbol{\phi}}\left[\log \frac{p(\mathcal{D} \mid \boldsymbol{x}) p(\boldsymbol{x})}{q_{\boldsymbol{\phi}}(\boldsymbol{x})}\right]=\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{x})]+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{x})\right]
$$

with $\mathbb{E}_{\boldsymbol{\phi}} \equiv \mathbb{E}_{q_{\boldsymbol{\phi}}}$, and of the variational posterior,

$$
q(\boldsymbol{x}) \equiv q_{\boldsymbol{\phi}}(\boldsymbol{x})=\sum_{k=1}^{K} w_{k} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)
$$

where $w_{k}, \boldsymbol{\mu}_{k}$, and $\sigma_{k}$ are, respectively, the mixture weight, mean, and scale of the $k$-th component, and $\boldsymbol{\Sigma} \equiv \operatorname{diag}\left[\lambda^{(1)^{2}}, \ldots, \lambda^{(D)^{2}}\right]$ is a diagonal covariance matrix common to all elements of the mixture. The variational posterior for a given number of mixture components $K$ is parameterized by $\boldsymbol{\phi} \equiv\left(w_{1}, \ldots, w_{K}, \boldsymbol{\mu}_{1}, \ldots, \boldsymbol{\mu}_{K}, \sigma_{1}, \ldots, \sigma_{K}, \boldsymbol{\lambda}\right)$.
In the following paragraphs we derive expressions for the ELBO and for its gradient. Then, we explain how we optimize it with respect to the variational parameters.

## A. 1 Stochastic approximation of the entropy

We approximate the entropy of the variational distribution via simple Monte Carlo sampling as follows. Let $\mathbf{R}=\operatorname{diag}[\boldsymbol{\lambda}]$ and $N_{\mathrm{s}}$ be the number of samples per mixture component. We have

$$
\begin{aligned}
\mathcal{H}[q(\boldsymbol{x})] & =-\int q(\boldsymbol{x}) \log q(\boldsymbol{x}) d \boldsymbol{x} \\
& \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k} \log q\left(\sigma_{k} \mathbf{R} \varepsilon_{s, k}+\boldsymbol{\mu}_{k}\right) \quad \text { with } \quad \varepsilon_{s, k} \sim \mathcal{N}\left(\mathbf{0}, \mathbb{I}_{D}\right) \\
& =-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k} \log q\left(\boldsymbol{\xi}_{s, k}\right) \quad \text { with } \quad \boldsymbol{\xi}_{s, k} \equiv \sigma_{k} \mathbf{R} \varepsilon_{s, k}+\boldsymbol{\mu}_{k}
\end{aligned}
$$

where we used the reparameterization trick separately for each component [18, 19]. For VBMC, we set $N_{\mathrm{s}}=100$ during the variational optimization, and $N_{\mathrm{s}}=2^{15}$ for evaluating the ELBO with high precision at the end of each iteration.

## A.1.1 Gradient of the entropy

The derivative of the entropy with respect to a variational parameter $\phi \in\{\mu, \sigma, \lambda\}$ (that is, not a mixture weight) is

$$
\begin{aligned}
\frac{d}{d \phi} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k} \frac{d}{d \phi} \log q\left(\boldsymbol{\xi}_{s, k}\right) \\
& =-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k}\left(\frac{\partial}{\partial \phi}+\sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \phi} \frac{\partial}{\partial \xi_{s, k}^{(i)}}\right) \log q\left(\boldsymbol{\xi}_{s, k}\right) \\
& =-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \phi} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \phi} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, k}^{(i)}-\mu_{l}^{(i)}}{\left(\sigma_{k} \lambda^{(i)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where from the second to the third row we used the fact that the expected value of the score is zero, $\mathbb{E}_{q(\boldsymbol{\xi})}\left[\frac{\partial}{\partial \phi} \log q(\boldsymbol{\xi})\right]=0$.

---

#### Page 14

In particular, for $\phi=\mu_{j}^{(m)}$, with $1 \leq m \leq D$ and $1 \leq j \leq K$,

$$
\begin{aligned}
\frac{d}{d \mu_{j}^{(m)}} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \mu_{j}^{(m)}} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{w_{j}}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \frac{1}{q\left(\boldsymbol{\xi}_{s, j}\right)} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, j}^{(m)}-\mu_{l}^{(m)}}{\left(\sigma_{l} \lambda^{(m)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, j} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where we used that fact that $\frac{d \xi_{s, k}^{(i)}}{d \mu_{j}^{(m)}}=\delta_{i m} \delta_{j k}$.
For $\phi=\sigma_{j}$, with $1 \leq j \leq K$,

$$
\begin{aligned}
\frac{d}{d \sigma_{j}} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \sigma_{j}} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{w_{j}}{K^{2} N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \frac{1}{q\left(\boldsymbol{\xi}_{s, j}\right)} \sum_{i=1}^{D} \lambda^{(i)} \varepsilon_{s, j}^{(i)} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, j}^{(i)}-\mu_{l}^{(i)}}{\left(\sigma_{l} \lambda^{(i)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, j} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where we used that fact that $\frac{d \xi_{s, k}^{(i)}}{d \sigma_{j}}=\lambda^{(i)} \varepsilon_{s, j}^{(i)} \delta_{j k}$.
For $\phi=\lambda^{(m)}$, with $1 \leq m \leq D$,

$$
\begin{aligned}
\frac{d}{d \lambda^{(m)}} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \lambda^{(m)}} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k} \sigma_{k} \varepsilon_{s, k}^{(m)}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, k}^{(m)}-\mu_{l}^{(m)}}{\left(\sigma_{l} \lambda^{(m)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where we used that fact that $\frac{d \xi_{s, k}^{(i)}}{d \lambda^{(m)}}=\sigma_{k} \varepsilon_{s, k}^{(i)} \delta_{i m}$.
Finally, the derivative with respect to variational mixture weight $w_{j}$, for $1 \leq j \leq K$, is

$$
\frac{\partial}{\partial w_{j}} \mathcal{H}[q(\boldsymbol{x})] \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}}\left[\log q\left(\boldsymbol{\xi}_{s, j}\right)+\sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} q_{j}\left(\boldsymbol{\xi}_{s, k}\right)\right]
$$

# A. 2 Expected log joint

For the expected log joint we have

$$
\begin{aligned}
\mathcal{G}[q(\boldsymbol{x})]=\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{x})] & =\sum_{k=1}^{K} w_{k} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) f(\boldsymbol{x}) d \boldsymbol{x} \\
& =\sum_{k=1}^{K} w_{k} \mathcal{I}_{k}
\end{aligned}
$$

To solve the integrals in Eq. S9 we approximate $f(\boldsymbol{x})$ with a Gaussian process (GP) with a squared exponential (that is, rescaled Gaussian) covariance function,

$$
\mathbf{K}_{p q}=\kappa\left(\boldsymbol{x}_{p}, \boldsymbol{x}_{q}\right)=\sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{x}_{p} ; \boldsymbol{x}_{q}, \boldsymbol{\Sigma}_{\ell}\right) \quad \text { with } \boldsymbol{\Sigma}_{\ell}=\operatorname{diag}\left[\ell^{(1)^{2}}, \ldots, \ell^{(D)^{2}}\right]
$$

where $\Lambda \equiv(2 \pi)^{\frac{D}{2}} \prod_{i=1}^{D} \ell^{(i)}$ is equal to the normalization factor of the Gaussian. ${ }^{1}$ For the GP we also assume a Gaussian likelihood with observation noise variance $\sigma_{\text {obs }}^{2}$ and, for the sake of exposition, a constant mean function $m \in \mathbb{R}$. We will later consider the case of a negative quadratic mean function, as per the main text.

[^0]
[^0]: ${ }^{1}$ This choice of notation makes it easy to apply Gaussian identities used in Bayesian quadrature.

---

#### Page 15

# A.2.1 Posterior mean of the integral and its gradient

The posterior predictive mean of the GP, given training data $\boldsymbol{\Xi}=\{\mathbf{X}, \boldsymbol{y}\}$, where $\mathbf{X}$ are $n$ training inputs with associated observed values $\boldsymbol{y}$, is

$$
\bar{f}(\boldsymbol{x})=\kappa(\boldsymbol{x}, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1}(\boldsymbol{y}-m)+m
$$

Thus, for each integral in Eq. S9 we have in expectation over the GP posterior

$$
\begin{aligned}
\mathbb{E}_{f \mid \mathbb{E}}\left[\mathcal{I}_{k}\right] & =\int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) \bar{f}(\boldsymbol{x}) d \boldsymbol{x} \\
& =\left[\sigma_{f}^{2} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) \mathcal{N}\left(\boldsymbol{x} ; \mathbf{X}, \boldsymbol{\Sigma}_{\ell}\right) d \boldsymbol{x}\right]\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m)+m \\
& =\boldsymbol{z}_{k}^{\top}\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m)+m
\end{aligned}
$$

where $\boldsymbol{z}_{k}$ is a $n$-dimensional vector with entries $z_{k}^{(p)}=\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{\mu}_{k} ; \boldsymbol{x}_{p}, \sigma_{k}^{2} \boldsymbol{\Sigma}+\boldsymbol{\Sigma}_{\ell}\right)$ for $1 \leq p \leq n$. In particular, defining $\tau_{k}^{(i)} \equiv \sqrt{\sigma_{k}^{2} \lambda^{(i)^{2}}+\ell^{(i)^{2}}}$ for $1 \leq i \leq D$,

$$
z_{k}^{(p)}=\frac{\sigma_{f}^{2}}{(2 \pi)^{\frac{D}{2}} \prod_{i=1}^{D} \tau_{k}^{(i)}} \exp \left\{-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(\mu_{k}^{(i)}-\boldsymbol{x}_{p}^{(i)}\right)^{2}}{\tau_{k}^{(i)^{2}}}\right\}
$$

We can compute derivatives with respect to the variational parameters $\phi \in(\mu, \sigma, \lambda)$ as

$$
\begin{aligned}
\frac{\partial}{\partial \mu_{j}^{(l)}} z_{k}^{(p)} & =\delta_{j k} \frac{\boldsymbol{x}_{p}^{(l)}-\mu_{k}^{(l)}}{\tau_{k}^{(l)^{2}}} z_{k}^{(p)} \\
\frac{\partial}{\partial \sigma_{j}} z_{k}^{(p)} & =\delta_{j k} \sum_{i=1}^{D} \frac{\lambda^{(i)^{2}}}{\tau_{k}^{(i)^{2}}}\left[\frac{\left(\mu_{k}^{(i)}-\boldsymbol{x}_{p}^{(i)}\right)^{2}}{\tau_{k}^{(i)^{2}}}-1\right] \sigma_{k} z_{k}^{(p)} \\
\frac{\partial}{\partial \lambda^{(l)}} z_{k}^{(p)} & =\frac{\sigma_{k}^{2}}{\tau_{k}^{(l)^{2}}}\left[\frac{\left(\mu_{k}^{(l)}-\boldsymbol{x}_{p}^{(l)}\right)^{2}}{\tau_{k}^{(l)^{2}}}-1\right] \lambda^{(l)} z_{k}^{(p)}
\end{aligned}
$$

The derivative of Eq. S9 with respect to mixture weight $w_{k}$ is simply $\mathcal{I}_{k}$.

## A.2.2 Posterior variance of the integral

We compute the variance of Eq. S9 under the GP approximation as [8]

$$
\begin{aligned}
\operatorname{Var}_{f \mid X}[\mathcal{G}] & =\int \int q(\boldsymbol{x}) q\left(\boldsymbol{x}^{\prime}\right) C_{\mathbb{E}}\left(f(\boldsymbol{x}), f\left(\boldsymbol{x}^{\prime}\right)\right) d \boldsymbol{x} d \boldsymbol{x}^{\prime} \\
& =\sum_{j=1}^{K} \sum_{k=1}^{K} w_{j} w_{k} \int \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{j}, \sigma_{j}^{2} \boldsymbol{\Sigma}\right) \mathcal{N}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) C_{\mathbb{E}}\left(f(\boldsymbol{x}), f\left(\boldsymbol{x}^{\prime}\right)\right) d \boldsymbol{x} d \boldsymbol{x}^{\prime} \\
& =\sum_{j=1}^{K} \sum_{k=1}^{K} w_{j} w_{k} \mathcal{J}_{j k}
\end{aligned}
$$

where $C_{\Xi}$ is the GP posterior predictive covariance,

$$
C_{\Xi}\left(f(\boldsymbol{x}), f\left(\boldsymbol{x}^{\prime}\right)\right)=\kappa\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)-\kappa(\boldsymbol{x}, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1} \kappa\left(\mathbf{X}, \boldsymbol{x}^{\prime}\right)
$$

Thus, each term in Eq. S15 can be written as

$$
\begin{aligned}
\mathcal{J}_{j k}= & \int \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{j}, \sigma_{j}^{2} \boldsymbol{\Sigma}\right)\left[\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{x}^{\prime}, \boldsymbol{\Sigma}_{\ell}\right)-\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \mathbf{X}, \boldsymbol{\Sigma}_{\ell}\right)\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1} \sigma_{f}^{2} \mathcal{N}\left(\mathbf{X} ; \boldsymbol{x}^{\prime}, \boldsymbol{\Sigma}_{\ell}\right)\right] \times \\
& \times \mathcal{N}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) d \boldsymbol{x} d \boldsymbol{x}^{\prime} \\
= & \sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{\mu}_{j} ; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{\ell}+\left(\sigma_{j}^{2}+\sigma_{k}^{2}\right) \boldsymbol{\Sigma}\right)-\boldsymbol{z}_{j}^{\top}\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1} \boldsymbol{z}_{k}
\end{aligned}
$$

---

#### Page 16

# A.2.3 Negative quadratic mean function

We consider now a GP with a negative quadratic mean function,

$$
m(\boldsymbol{x}) \equiv m_{\mathrm{NQ}}(\boldsymbol{x})=m_{0}-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(x^{(i)}-x_{\mathrm{m}}^{(i)}\right)^{2}}{\omega^{(i)^{2}}}
$$

With this mean function, for each integral in Eq. S9 we have in expectation over the GP posterior,

$$
\begin{aligned}
\mathbb{E}_{f \mid \mathbb{E}}\left[\mathcal{I}_{k}\right] & =\int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)\left[\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \mathbf{X}, \boldsymbol{\Sigma}_{t}\right)\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m(\mathbf{X}))+m(\boldsymbol{x})\right] d \boldsymbol{x} \\
& =\boldsymbol{z}_{k}^{\top}\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m(\mathbf{X}))+m_{0}+\nu_{k}
\end{aligned}
$$

where we defined

$$
\nu_{k}=-\frac{1}{2} \sum_{i=1}^{D} \frac{1}{\omega^{(i)^{2}}}\left(\mu_{k}^{(i)^{2}}+\sigma_{k}^{2} \lambda^{(i)^{2}}-2 \mu_{k}^{(i)} x_{\mathrm{m}}^{(i)}+x_{\mathrm{m}}^{(i)^{2}}\right)
$$

## A. 3 Optimization of the approximate ELBO

In the following paragraphs we describe how we optimize the ELBO in each iteration of VBMC, so as to find the variational posterior that best approximates the current GP model of the posterior.

## A.3.1 Reparameterization

For the purpose of the optimization, we reparameterize the variational parameters such that they are defined in a potentially unbounded space. The mixture means, $\boldsymbol{\mu}_{k}$, remain the same. We switch from mixture scale parameters $\sigma_{k}$ to their logarithms, $\log \sigma_{k}$, and similarly from coordinate length scales, $\lambda^{(i)}$, to $\log \lambda^{(i)}$. Finally, we parameterize mixture weights as unbounded variables, $\eta_{k} \in \mathbb{R}$, such that $w_{k} \equiv e^{\eta_{k}} / \sum_{l} e^{\eta_{l}}$ (softmax function). We compute the appropriate Jacobian for the change of variables and apply it to the gradients calculated in Sections A. 1 and A.2.

## A.3.2 Choice of starting points

In each iteration, we first perform a quick exploration of the ELBO landscape in the vicinity of the current variational posterior by generating $n_{\text {fast }} \cdot K$ candidate starting points, obtained by randomly jittering, rescaling, and reweighting components of the current variational posterior. In this phase we also add new mixture components, if so requested by the algorithm, by randomly splitting and jittering existing components. We evaluate the ELBO at each candidate starting point, and pick the point with the best ELBO as starting point for the subsequent optimization.

For most iterations we use $n_{\text {fast }}=5$, except for the first iteration and the first iteration after the end of warm-up, for which we set $n_{\text {fast }}=50$.

## A.3.3 Stochastic gradient descent

We optimize the (negative) ELBO via stochastic gradient descent, using a customized version of Adam [21]. Our modified version of Adam includes a time-decaying learning rate, defined as

$$
\alpha_{t}=\alpha_{\min }+\left(\alpha_{\max }-\alpha_{\min }\right) \exp \left[-\frac{t}{\tau}\right]
$$

where $t$ is the current iteration of the optimizer, $\alpha_{\min }$ and $\alpha_{\max }$ are, respectively, the minimum and maximum learning rate, and $\tau$ is the decay constant. We stop the optimization when the estimated change in function value or in the parameter vector across the past $n_{\text {batch }}$ iterations of the optimization goes below a given threshold.
We set as hyperparameters of the optimizer $\beta_{1}=0.9, \beta_{2}=0.99, \epsilon \approx 1.49 \cdot 10^{-8}$ (square root of double precision), $\alpha_{\min }=0.001, \tau=200, n_{\text {batch }}=20$. We set $\alpha_{\max }=0.1$ during warm-up, and $\alpha_{\max }=0.01$ thereafter.

---

#### Page 17

# B Algorithmic details

We report here several implementation details of the VBMC algorithm omitted from the main text.

## B. 1 Regularization of acquisition functions

Active sampling in VBMC is performed by maximizing an acquisition function $a: \mathcal{X} \subseteq \mathbb{R}^{D} \rightarrow$ $[0, \infty)$, where $\mathcal{X}$ is the support of the target density. In the main text we describe two such functions, uncertainty sampling ( $a_{\mathrm{us}}$ ) and prospective uncertainty sampling ( $a_{\mathrm{pro}}$ ).
A well-known problem with GPs, in particular when using smooth kernels such as the squared exponential, is that they become numerically unstable when the training set contains points which are too close to each other, producing a ill-conditioned Gram matrix. Here we reduce the chance of this happening by introducing a correction factor as follows. For any acquisition function $a$, its regularized version $a^{\text {reg }}$ is defined as

$$
a^{\mathrm{reg}}(\boldsymbol{x})=a(\boldsymbol{x}) \exp \left\{-\left(\frac{V^{\mathrm{reg}}}{\overline{V_{\mathbb{R}}(\boldsymbol{x})}}-1\right)\left|\left|V_{\mathbb{R}}(\boldsymbol{x})<V^{\mathrm{reg}}\right|\right|\right\}
$$

where $V_{\mathbb{R}}(\boldsymbol{x})$ is the total posterior predictive variance of the GP at $\boldsymbol{x}$ for the given training set $\mathbb{E}, V^{\text {reg }}$ a regularization parameter, and we denote with $|[\cdot]|$ Iverson's bracket [28], which takes value 1 if the expression inside the bracket is true, 0 otherwise. Eq. S22 enforces that the regularized acquisition function does not pick points too close to points in $\mathbb{E}$. For VBMC, we set $V^{\text {reg }}=10^{-4}$.

## B. 2 GP hyperparameters and priors

The GP model in VBMC has $3 D+3$ hyperparameters, $\boldsymbol{\psi}=\left(\boldsymbol{\ell}, \sigma_{f}, \sigma_{\mathrm{obs}}, m_{0}, \boldsymbol{x}_{\mathrm{m}}, \boldsymbol{\omega}\right)$. We define all scale hyperparameters, that is $\left\{\boldsymbol{\ell}, \sigma_{f}, \sigma_{\mathrm{obs}}, \boldsymbol{\omega}\right\}$, in log space.
We assume independent priors on each hyperparameter. For some hyperparameters, we impose as prior a broad Student's $t$ distribution with a given mean $\mu$, scale $\sigma$, and $\nu=3$ degrees of freedom. Following an empirical Bayes approach, mean and scale of the prior might depend on the current training set. For all other hyperparameters we assume a uniform flat prior. GP hyperparameters and their priors are reported in Table S1.

|        Hyperparameter        | Description                            |                           Prior mean $\mu$                           |                                                                        Prior scale $\sigma$                                                                         |
| :--------------------------: | :------------------------------------- | :------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      $\log \ell^{(i)}$       | Input length scale (i-th dimension)    | $\log \operatorname{SD}\left[\mathbf{X}_{\text {hpd }}^{(i)}\right]$ | $\max \left\{2, \log \frac{\operatorname{diam}\left[\mathbf{X}_{\text {hpd }}^{(i)}\right]}{\operatorname{SD}\left[\mathbf{X}_{\text {hpd }}^{(i)}\right]}\right\}$ |
|      $\log \sigma_{f}$       | Output scale                           |                               Uniform                                |                                                                                  -                                                                                  |
| $\log \sigma_{\text {obs }}$ | Observation noise                      |                             $\log 0.001$                             |                                                                                 0.5                                                                                 |
|           $m_{0}$            | Mean function maximum                  |                 $\max \boldsymbol{y}_{\text {hpd }}$                 |                                                   $\operatorname{diam}\left[\boldsymbol{y}_{\text {hpd }}\right]$                                                   |
|    $x_{\mathrm{m}}^{(i)}$    | Mean function location (i-th dim.)     |                               Uniform                                |                                                                                  -                                                                                  |
|     $\log \omega^{(i)}$      | Mean function length scale (i-th dim.) |                               Uniform                                |                                                                                  -                                                                                  |

Table S1: GP hyperparameters and their priors. See text for more information.

In Table S1, $\operatorname{SD}[\cdot]$ denotes the sample standard deviation and $\operatorname{diam}[\cdot]$ the diameter of a set, that is the maximum element minus the minimum. We define the high posterior density training set, $\mathbb{E}_{\text {hpd }}=\left\{\mathbf{X}_{\text {hpd }}, \boldsymbol{y}_{\text {hpd }}\right\}$, constructed by keeping a fraction $f_{\text {hpd }}$ of the training points with highest target density values. For VBMC, we use $f_{\text {hpd }}=0.8$ (that is, we only ignore a small fraction of the points in the training set).

## B. 3 Transformation of variables

In VBMC, the problem coordinates are defined in an unbounded internal working space, $\boldsymbol{x} \in \mathbb{R}^{D}$. All original problem coordinates $x_{\text {orig }}^{(i)}$ for $1 \leq i \leq D$ are independently transformed by a mapping $g_{i}: \mathcal{X}_{\text {orig }}^{(i)} \rightarrow \mathbb{R}$ defined as follows.

---

#### Page 18

Unbounded coordinates are 'standardized' with respect to the plausible box, $g_{\text {unb }}\left(x_{\text {orig }}\right)=$ $\frac{x_{\text {orig }}-(\text { PLB }+ \text { PUB }) / 2}{\text { PUB-PLB }}$, where PLB and PUB are here, respectively, the plausible lower bound and plausible upper bound of the coordinate under consideration.
Bounded coordinates are first mapped to an unbounded space via a logit transform, $g_{\text {bnd }}\left(x_{\text {orig }}\right)=$ $\log \left(\frac{z}{1-z}\right)$ with $z=\frac{x_{\text {orig }}-L B}{\mathrm{UB}-\mathrm{LB}}$, where LB and UB are here, respectively, the lower and upper bound of the coordinate under consideration. The remapped variables are then 'standardized' as above, using the remapped PLB and PUB values after the logit transform.

Note that probability densities are transformed under a change of coordinates by a multiplicative factor equal to the inverse of the determinant of the Jacobian of the transformation. Thus, the value of the observed log joint $y$ used by VBMC relates to the value $y_{\text {orig }}$ of the log joint density, observed in the original (untransformed) coordinates, as follows,

$$
y(\boldsymbol{x})=y^{\text {orig }}\left(\boldsymbol{x}_{\text {orig }}\right)-\sum_{i=1}^{D} \log g_{i}^{\prime}\left(\boldsymbol{x}_{\text {orig }}\right)
$$

where $g_{i}^{\prime}$ is the derivative of the transformation for the $i$-th coordinate, and $\boldsymbol{x}=g\left(\boldsymbol{x}_{\text {orig }}\right)$. See for example [24] for more information on transformations of variables.

# B. 4 Termination criteria

The VBMC algorithm terminates when reaching a maximum number of target density evaluations, or when achieving long-term stability of the variational solution, as described below.

## B.4.1 Reliability index

At the end of each iteration $t$ of the VBMC algorithm, we compute a set of reliability features of the current variational solution.

1. The absolute change in mean ELBO from the previous iteration:

$$
\rho_{1}(t)=\frac{|\mathbb{E}[\operatorname{ELBO}(t)]-\mathbb{E}[\operatorname{ELBO}(t-1)]|}{\Delta_{\mathrm{SD}}}
$$

where $\Delta_{\mathrm{SD}}>0$ is a tolerance parameter on the error of the ELBO. 2. The uncertainty of the current ELBO:

$$
\rho_{2}(t)=\frac{\sqrt{\mathbb{V}[\operatorname{ELBO}(t)]}}{\Delta_{\mathrm{SD}}}
$$

3. The change in symmetrized KL divergence between the current variational posterior $q_{t} \equiv$ $q_{\phi_{t}}(\boldsymbol{x})$ and the one from the previous iteration:

$$
\rho_{3}(t)=\frac{\mathrm{KL}\left(q_{t} \| q_{t-1}\right)+\mathrm{KL}\left(q_{t-1} \| q_{t}\right)}{2 \Delta_{\mathrm{KL}}}
$$

where for Eq. S26 we use the Gaussianized KL divergence (that is, we compare solutions only based on their mean and covariance), and $\Delta_{\mathrm{KL}}>0$ is a tolerance parameter for differences in variational posterior.

The parameters $\Delta_{\mathrm{SD}}$ and $\Delta_{\mathrm{KL}}$ are chosen such that $\rho_{j} \lesssim 1$, with $j=1,2,3$, for features that are deemed indicative of a good solution. For VBMC, we set $\Delta_{\mathrm{SD}}=0.1$ and $\Delta_{\mathrm{KL}}=0.01 \cdot \sqrt{D}$.
The reliability index $\rho(t)$ at iteration $t$ is obtained by averaging the individual reliability features $\rho_{j}(t)$.

## B.4.2 Long-term stability termination condition

The long-term stability termination condition is reached at iteration $t$ when:

1. all reliability features $\rho_{j}(t)$ are below 1 ;

---

#### Page 19

2. the reliability index $\rho$ has remained below 1 for the past $n_{\text {stable }}$ iterations (with the exception of at most one iteration, excluding the current one);
3. the slope of the ELCBO computed across the past $n_{\text {stable }}$ iterations is below a given threshold $\Delta_{\text {IMPRO }}>0$, suggesting that the ELCBO is stationary.

For VBMC, we set by default $n_{\text {stable }}=8$ and $\Delta_{\text {IMPRO }}=0.01$. For computing the ELCBO we use $\beta_{\text {LCB }}=3$ (see Eq. 8 in the main text).

# B.4.3 Validation of VBMC solutions

Long-term stability of the variational solution is suggestive of convergence of the algorithm to a (local) optimum, but it should not be taken as a conclusive result without further validation. In fact, without additional information, there is no way to know whether the algorithm has converged to a good solution, let alone to the global optimum. For this reason, we recommend to run the algorithm multiple times and compare the solutions, and to perform posterior predictive checks [29]. See also [30] for a discussion of methods to validate the results of variational inference.

## C Experimental details and additional results

## C. 1 Synthetic likelihoods

We plot in Fig. S1 synthetic target densities belonging to the test families described in the main text (lumpy, Student, cigar), for the $D=2$ case. We also plot examples of solutions returned by VBMC after reaching long-term stability, and indicate the number of iterations.

> **Image description.** The image consists of six contour plots arranged in a 2x3 grid. Each column represents a different synthetic target density: "Lumpy", "Student", and "Cigar". The top row shows the "True" target densities, while the bottom row shows the corresponding variational posteriors returned by VBMC (Variational Bayesian Monte Carlo).
>
> Each plot has x1 and x2 axes. The plots are contained within square frames.
>
> - **Column 1 (Lumpy):**
>
>   - Top: A contour plot of the "True" lumpy density, showing an irregular shape with multiple peaks. The contours are nested and colored in shades of blue and yellow.
>   - Bottom: A contour plot of the VBMC solution for the lumpy density. The shape is similar to the "True" density but slightly smoother. The text "Iteration 11" is below the plot.
>
> - **Column 2 (Student):**
>
>   - Top: A contour plot of the "True" Student density, showing a roughly square shape with rounded corners. The contours are nested and colored in shades of blue.
>   - Bottom: A contour plot of the VBMC solution for the Student density. The shape is nearly circular. The text "Iteration 9" is below the plot.
>
> - **Column 3 (Cigar):**
>   - Top: A contour plot of the "True" cigar density, showing a highly elongated shape along a diagonal. The contours are nested and colored in shades of blue and green.
>   - Bottom: A scatter plot representing the VBMC solution for the cigar density. It shows a cluster of blue circles aligned along a diagonal, similar to the "True" density. The text "Iteration 22" is below the plot.
>
> The text labels "Lumpy", "Student", and "Cigar" are above their respective columns. The text "True" is above each plot in the top row, and "VBMC" is above each plot in the bottom row. The axes are labeled "x1" and "x2".

Figure S1: Synthetic target densities and example solutions. Top: Contour plots of twodimensional synthetic target densities. Bottom: Contour plots of example variational posteriors returned by VBMC, and iterations until convergence.

Note that VBMC, despite being overall the best-performing algorithm on the cigar family in higher dimensions, still underestimates the variance along the major axis of the distribution. This is because the variational mixture components have axis-aligned (diagonal) covariances, and thus many mixture components are needed to approximate non-axis aligned densities. Future work should investigate alternative representations of the variational posterior to increase the expressive power of VBMC, while keeping its computational efficiency and stability.
We plot in Fig. S2 the performance of selected algorithms on the synthetic test functions, for $D \in\{2,4,6,8,10\}$. These results are the same as those reported in Fig. 2 in the main text, but with higher resolution. To avoid clutter, we exclude algorithms with particularly poor performance

---

#### Page 20

or whose plots are redundant with others. In particular, the performance of VBMC-U is virtually identical to VBMC-P here, so we only report the latter. Analogously, with a few minor exceptions, WSABI-M performs similarly or worse than WSABI-L across all problems. AIS suffers from the lack of problem-specific tuning, performing no better than SMC here, and the AGP algorithm diverges on most problems. Finally, we did not manage to get BAPE to run on the cigar family, for $D \leq 6$, without systematically incurring in numerical issues with the GP approximation (with and without regularization of the BAPE acquisition function, as per Section B.1), so these plots are missing.

# C. 2 Neuronal model

As a real model-fitting problem, we considered in the main text a neuronal model that combines effects of filtering, suppression, and response nonlinearity, applied to two real data sets (one V1 and one V2 neurons) [14]. The purpose of the original study was to explore the origins of diversity of neuronal orientation selectivity in visual cortex via a combination of novel stimuli (orientation mixtures) and modeling [14]. This model was also previously considered as a case study for a benchmark of Bayesian optimization and other black-box optimization algorithms [6].

## C.2.1 Model parameters

In total, the original model has 12 free parameters: 5 parameters specifying properties of a linear filtering mechanism, 2 parameters specifying nonlinear transformation of the filter output, and 5 parameters controlling response range and amplitude. For the analysis in the main text, we considered a subset of $D=7$ parameters deemed 'most interesting' by the authors of the original study [14], while fixing the others to their MAP values found by our previous optimization benchmark [6].
The seven model parameters of interest from the original model, their ranges, and the chosen plausible bounds are reported in Table S2.

| Parameter | Description                                  |    LB |  UB |  PLB | PUB |
| :-------: | :------------------------------------------- | ----: | --: | ---: | --: |
|  $x_{1}$  | Preferred direction of motion (deg)          |     0 | 360 |   90 | 270 |
|  $x_{2}$  | Preferred spatial frequency (cycles per deg) |  0.05 |  15 |  0.5 |  10 |
|  $x_{3}$  | Aspect ratio of 2-D Gaussian                 |   0.1 | 3.5 |  0.3 | 3.2 |
|  $x_{4}$  | Derivative order in space                    |   0.1 | 3.5 |  0.3 | 3.2 |
|  $x_{5}$  | Gain inhibitory channel                      |    -1 |   1 | -0.3 | 0.3 |
|  $x_{6}$  | Response exponent                            |     1 | 6.5 |    2 |   5 |
|  $x_{7}$  | Variance of response gain                    | 0.001 |  10 | 0.01 |   1 |

Table S2: Parameters and bounds of the neuronal model (before remapping).

Since all original parameters are bounded, for the purpose of our analysis we remapped them to an unbounded space via a shifted and rescaled logit transform, correcting the value of the log posterior with the log Jacobian (see Section B.3). For each parameter, we set independent Gaussian priors in the transformed space with mean equal to the average of the transformed values of PLB and PUB (see Table S2), and with standard deviation equal to half the plausible range in the transformed space.

## C.2.2 True and approximate posteriors

We plot in Fig. S3 the 'true' posterior obtained via extensive MCMC sampling for one of the two datasets (V2 neuron), and we compare it with an example variational solution returned by VBMC after reaching long-term stability (here in 52 iterations, which correspond to 260 target density evaluations).
We note that VBMC obtains a good approximation of the true posterior, which captures several features of potential interest, such as the correlation between the inhibition gain $\left(x_{5}\right)$ and response exponent $\left(x_{6}\right)$, and the skew in the preferred spatial frequency $\left(x_{2}\right)$. The variational posterior, however, misses some details, such as the long tail of the aspect ratio $\left(x_{3}\right)$, which is considerably thinner in the approximation than in the true posterior.

---

#### Page 21

> **Image description.** This image contains two panels, A and B, each displaying a series of line graphs. Each panel contains five subplots arranged in a row, and each row represents a different problem ("Lumpy", "Student", "Cigar").
>
> Panel A:
>
> - Each subplot in panel A displays "Median LML error" on the y-axis (log scale) versus "Function evaluations" on the x-axis (linear scale).
> - The x-axis ranges vary between subplots, with maximum values of 200, 300, 400, 400, and 600.
> - The y-axis ranges from 10^-4 to 10 for the "Lumpy" and "Student" problems, and from 0.1 to 10^4 for the "Cigar" problem.
> - Each subplot contains multiple lines, each representing a different algorithm: "smc" (dotted gray), "bmc" (solid gray), "wsabi-L" (solid pink), "bbq" (dashed green), "bape" (solid green), and "vbmc-P" (solid black). Shaded regions around the lines represent confidence intervals.
> - A horizontal dashed line is present at y=1 in each subplot.
> - The columns are labeled "2D", "4D", "6D", "8D", and "10D".
>
> Panel B:
>
> - Each subplot in panel B displays "Median gsKL" on the y-axis (log scale) versus "Function evaluations" on the x-axis (linear scale).
> - The x-axis ranges are the same as in Panel A.
> - The y-axis ranges from 10^-4 to 10 for the "Lumpy" and "Student" problems, and from 10^-2 to 10^6 for the "Cigar" problem.
> - The same algorithms are represented with the same line styles and colors as in Panel A. Shaded regions around the lines represent confidence intervals.
> - A horizontal dashed line is present at y=1 in each subplot.
> - The columns are labeled "2D", "4D", "6D", "8D", and "10D".
>
> Overall:
> The image presents a comparison of different algorithms on synthetic likelihood problems, evaluating their performance based on "Median LML error" and "Median gsKL" metrics. The performance is shown as a function of the number of function evaluations for different problem dimensionalities (2D, 4D, 6D, 8D, 10D).

Figure S2: Full results on synthetic likelihoods. A. Median absolute error of the LML estimate with respect to ground truth, as a function of number of likelihood evaluations, on the lumpy (top), Student (middle), and cigar (bottom) problems, for $D \in\{2,4,6,8,10\}$ (columns). B. Median "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth. For both metrics, shaded areas are $95 \%$ CI of the median, and we consider a desirable threshold to be below one (dashed line). This figure reproduces Fig. 2 in the main text with more details. Note that panels here may have different vertical axes.

---

#### Page 22

> **Image description.** This image contains two triangle plots, one labeled "True" at the top and the other labeled "VBMC (iteration 52)" at the bottom. Each triangle plot displays a matrix of plots representing the posterior distribution of parameters in a model.
>
> Each row and column corresponds to a parameter, labeled x1 through x7. The diagonal elements of each matrix are histograms, representing the 1-D marginal distribution of the posterior for each parameter. The elements below the diagonal are contour plots, representing the 2-D marginal distribution for each pair of parameters.
>
> The "True" plot shows smoother, more defined contours and histograms, while the "VBMC (iteration 52)" plot shows similar shapes but with some differences in the contours and histograms. The axes are labeled with numerical values corresponding to the range of each parameter.

Figure S3: True and approximate posterior of neuronal model (V2 neuron). Top: Triangle plot of the 'true' posterior (obtained via MCMC) for the neuronal model applied to the V2 neuron dataset. Each panel below the diagonal is the contour plot of the 2-D marginal distribution for a given parameter pair. Panels on the diagonal are histograms of the 1-D marginal distribution of the posterior for each parameter. Bottom: Triangle plot of a typical variational solution returned by VBMC.

---

#### Page 23

# D Analysis of VBMC

In this section we report additional analyses of the VBMC algorithm.

## D. 1 Variability between VBMC runs

In the main text we have shown the median performance of VBMC, but a crucial question for a practical application of the algorithm is the amount of variability between runs, due to stochasticity in the algorithm and choice of starting point (in this work, drawn uniformly randomly inside the plausible box). We plot in Fig. S4 the performance of one hundred runs of VBMC on the neuronal model datasets, together with the 50th (the median), 75th, and 90th percentiles. The performance of VBMC on this real problem is fairly robust, in that some runs take longer but the majority of them converges to quantitatively similar solutions.

> **Image description.** This image contains two panels, A and B, each displaying two line graphs. All four graphs share a similar structure, plotting the performance of an algorithm across multiple runs.
>
> **Panel A:**
>
> - **Title:** A is in the top left corner.
> - **Y-axis:** Labeled "Median LML error" on a logarithmic scale from 10^-2 to 10^4. The label "Neuronal model" is placed vertically to the left of the y-axis label.
> - **X-axis:** Labeled "Function evaluations" ranging from 0 to 400.
> - **Graphs:** Two graphs, labeled "V1" and "V2" at the top, showing the error as a function of function evaluations. Each graph contains multiple thin grey lines representing individual runs of the algorithm. Thicker lines represent the 50th (solid), 75th (dashed), and 90th (dotted) percentiles across runs, as indicated by a legend. A horizontal dashed line is present at y=1.
>
> **Panel B:**
>
> - **Title:** B is in the top left corner.
> - **Y-axis:** Labeled "Median gsKL" on a logarithmic scale from 10^-2 to 10^6.
> - **X-axis:** Labeled "Function evaluations" ranging from 0 to 400.
> - **Graphs:** Two graphs, labeled "V1" and "V2" at the top, showing the error as a function of function evaluations. Each graph contains multiple thin grey lines representing individual runs of the algorithm. Thicker lines represent the 50th (solid), 75th (dashed), and 90th (dotted) percentiles across runs, as indicated by a legend in Panel A. A horizontal dashed line is present at y=1.
>
> In both panels, the graphs show a general trend of decreasing error/divergence as the number of function evaluations increases, indicating convergence of the algorithm. The percentile lines provide insight into the variability of performance across different runs.

Figure S4: Variability of VBMC performance. A. Absolute error of the LML estimate, as a function of number of likelihood evaluations, for the two neuronal datasets. Each grey line is one of 100 distinct runs of VBMC. Thicker lines correspond to the 50th (median), 75th, and 90th percentile across runs (the median is the same as in Fig. 3 in the main text). B. "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth, for 100 distinct runs of VBMC. See also Fig. 3 in the main text.

## D. 2 Computational cost

The computational cost of VBMC stems in each iteration of the algorithm primarily from three sources: active sampling, GP training, and variational optimization. Active sampling requires repeated computation of the acquisition function (for its optimization), whose cost is dominated by calculation of the posterior predictive variance of the GP, which scales as $O\left(n^{2}\right)$, where $n$ is the number of training points. GP training scales as $O\left(n^{3}\right)$, due to inversion of the Gram matrix. Finally, variational optimization scales as $O(K n)$, where $K$ is the number of mixture components. In practice, we found in many cases that in early iterations the costs are equally divided between the three phases, but later on both GP training and variational optimization dominate the algorithmic cost. In particular, the number of components $K$ has a large impact on the effective cost.
As an example, we plot in Fig. S5 the algorithmic cost per function evaluation of different inference algorithms that have been run on the V1 neuronal dataset (algorithmic costs are similar for the V2 dataset). We consider only methods which use active sampling with a reasonable performance on at least some of the problems. We define as algorithmic cost the time spent inside the algorithm, ignoring the time used to evaluate the log likelihood function. For comparison, evaluation of the log likelihood of this problem takes about 1 s on the reference laptop computer we used. Note that for the WSABI and BBQ algoritms, the algorithmic cost reported here does not include the additional computational cost of sampling an approximate distrbution from the GP posterior (WSABI and BBQ, per se, only compute an approximation of the marginal likelihood).
VBMC on this problem exhibits a moderate cost of 2-3 s per function evaluation, when averaged across the entire run. Moreover, many runs would converge within 250-300 function evaluations, as shown in Figure S4, further lowering the effective cost per function evaluation. For the considered budget of function evaluations, WSABI (in particular, WSABI-L) is up to one order of magnitude faster than VBMC. This speed is remarkable, although it does not offset the limited performance of

---

#### Page 24

> **Image description.** This is a line graph comparing the algorithmic cost per function evaluation for different algorithms performing inference on a V1 neuronal dataset.
>
> The graph has the following characteristics:
>
> - **Title:** "Neuronal model (V1)" is at the top of the graph.
>
> - **Axes:**
>
>   - The x-axis is labeled "Function evaluations" and ranges from approximately 0 to 400.
>   - The y-axis is labeled "Median algorithmic cost per function evaluation (s)" and uses a logarithmic scale, ranging from 0.01 to 100.
>
> - **Data Series:** The graph displays several data series, each representing a different algorithm:
>
>   - **wsabi-L:** A red line that starts low and gradually increases.
>   - **wsabi-M:** A blue line that starts low and gradually increases, staying below the vbmc-P line.
>   - **bbq:** A purple line that is initially high and very jagged, with sharp peaks, but gradually smooths out.
>   - **bape:** A dashed green line that fluctuates around a value of approximately 2.
>   - **vbmc-P:** A solid black line that starts high, dips, and then gradually increases, staying around a value of 1.
>
> - **Confidence Intervals:** Shaded areas around each line represent the 95% confidence interval of the median.
>
> - **Horizontal Line:** A dashed horizontal line is present at y = 1.
>
> - **Legend:** A legend on the right side of the graph identifies each data series with its corresponding algorithm name and color.

Figure S5: Algorithmic cost per function evaluation. Median algorithmic cost per function evaluation, as a function of number of likelihood function evaluations, for different algorithms performing inference over the V1 neuronal dataset. Shaded areas are $95 \%$ CI of the median.

the algorithm on more complex problems. WSABI-M is generally more expensive than WSABI-L (even though still quite fast), with a similar or slightly worse performance. Here our implementation of BAPE results to be slightly more expensive than VBMC. Perhaps it is possible to obtain faster implementations of BAPE, but, even so, the quality of solutions would still not match that of VBMC (also, note the general instability of the algorithm). Finally, we see that BBQ incurs in a massive algorithmic cost due to the complex GP approximation and expensive acquisition function used. Notably, the solutions obtained by BBQ in our problem sets are relatively good compared to the other algorithms, but still substantially worse than VBMC on all but the easiest problems, despite the much larger computational overhead.
The dip in cost that we observe in VBMC at around 275 function evaluations is due to the switch from GP hyperparameter sampling to optimization. The cost of BAPE oscillates because of the cost of retraining the GP model and MCMC sampling from the approximate posterior every 10 function evaluations. Similarly, by default BBQ retrains the GP model ten times, logarithmically spaced across its run, which appears here as logarithmically-spaced spikes in the cost.

# D. 3 Analysis of the samples produced by VBMC

We report the results of two control experiments to better understand the performance of VBMC.
For the first control experiment, shown in Fig. S6A, we estimate the log marginal likelihood (LML) using the WSABI-L approximation trained on samples obtained by VBMC (with the $a_{\text {pro }}$ acquisition function). The LML error of WSABI-L trained on VBMC samples is lower than WSABIL alone, showing that VBMC produces higher-quality samples and, given the same samples, a better approximation of the marginal likelihood. The fact that the LML error is still substantially higher in the control than with VBMC alone demonstrates that the error induced by the WSABI-L approximation can be quite large.
For the second control experiment, shown in Fig. S6B, we produce $2 \cdot 10^{4}$ posterior samples from a GP directly trained on the log joint distribution at the samples produced by VBMC. The quality of this posterior approximation is better than the posterior obtained by other methods, although generally not as good as the variational approximation (in particular, it is much more variable). While it is possible that the posterior approximation via direct GP fit could be improved, for example by using ad-hoc methods to increase the stability of the GP training procedure, this experiment shows that VBMC is able to reliably produce a high-quality variational posterior.

---

#### Page 25

> **Image description.** The image contains two panels, labeled A and B, each containing two line graphs. All four graphs share a similar structure.
>
> **Panel A:**
>
> - **Title:** "Neuronal model" is written vertically on the left side of the panel.
> - **Graphs:** Two line graphs, labeled "V1" and "V2" above each graph.
>   - X-axis: "Function evaluations", ranging from 0 to 400.
>   - Y-axis: "Median LML error", with a logarithmic scale ranging from 10^-2 to 10^4.
>   - Data: Each graph displays three lines representing different algorithms:
>     - "vbmc-P" (black line)
>     - "vbmc-control" (olive green line with shaded area around the line)
>     - "wsabi-L" (light red line with shaded area around the line)
>   - A dashed horizontal line is present at y=1.
>
> **Panel B:**
>
> - **Graphs:** Two line graphs, labeled "V1" and "V2" above each graph.
>   - X-axis: "Function evaluations", ranging from 0 to 400.
>   - Y-axis: "Median gsKL", with a logarithmic scale ranging from 10^-2 to 10^6.
>   - Data: Each graph displays three lines representing different algorithms:
>     - "vbmc-P" (black line)
>     - "vbmc-control" (olive green line with shaded area around the line)
>     - "wsabi-L" (light red line with shaded area around the line)
>   - A dashed horizontal line is present at y=1.
>
> **Legend:**
>
> - A legend is present between the two panels, mapping line colors to algorithm names:
>   - Black line: "vbmc-P"
>   - Olive green line: "vbmc-control"
>   - Light red line: "wsabi-L"
>
> In summary, the image presents four line graphs comparing the performance of three different algorithms ("vbmc-P", "vbmc-control", and "wsabi-L") across two different metrics ("Median LML error" and "Median gsKL") for two distinct neurons (V1 and V2) as a function of function evaluations. The shaded areas around the lines represent the 95% confidence interval of the median.

Figure S6: Control experiments on neuronal model likelihoods. A. Median absolute error of the LML estimate, as a function of number of likelihood evaluations, for two distinct neurons $(D=7)$. For the control experiment, here we computed the LML with WSABI-L trained on VBMC samples. B. Median "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth. For this control experiment, we produced posterior samples from a GP directly trained on the log joint at the samples produced by VBMC. For both metrics, shaded areas are $95 \%$ CI of the median, and we consider a desirable threshold to be below one (dashed line). See text for more details, and see also Fig. 3 in the main text.