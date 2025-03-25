```
@article{chang2025inference,
  title={Inference-Time Prior Adaptation in Simulation-Based Inference via Guided Diffusion Models},
  author={Paul Edmund Chang and Severi Rissanen and Nasrulloh Ratu Bagus Satrio Loka and Daolang Huang and Luigi Acerbi},
  year={2025},
  journal={7th Symposium on Advances in Approximate Bayesian Inference (AABI) - Workshop track}
}
```

---

#### Page 1

# Inference-Time Prior Adaptation in Simulation-Based Inference via Guided Diffusion Models

Paul Edmund Chang, Severi Rissanen, Nasrulloh RBS Loka, Daolang Huang, Luigi Acerbi

#### Abstract

Amortized simulator-based inference has emerged as a powerful framework for tackling inverse problems and Bayesian inference in many computational sciences by learning the reverse mapping from observed data to parameters. Once trained on many simulated parameter-data pairs, these methods afford parameter inference for any particular dataset, yielding high-quality posterior samples with only one or a few forward passes of a neural network. While amortized methods offer significant advantages in terms of efficiency and reusability across datasets, they are typically constrained by their training conditions - particularly the prior distribution of parameters used during training. In this paper, we introduce PriorGuide, a technique that enables on-the-fly adaptation to arbitrary priors at inference time for diffusionbased amortized inference methods. Our method allows users to incorporate new information or expert knowledge at runtime without costly retraining.

## 1 INTRODUCTION

Simulation-based inference has become a fundamental tool across computational sciences, enabling parameter estimation in complex systems where the forward model (simulator) is available but its likelihood is intractable (Cranmer et al., 2020). In a Bayesian framework, we express prior beliefs about parameters as distributions and update them given observations (Robert, 2007). While traditional inference methods such as Markov Chain Monte Carlo (MCMC) are the gold standard with tractable likelihoods (Gelman et al., 2014), recent neural network approaches can directly learn the inverse mapping from observations to posterior distributions over model parameters (Greenberg et al., 2019; Radev et al., 2020). These methods are typically amortized, enabling efficient inference after training and facilitating meta-learning across related problems (Brown et al., 2020). In this context, 'inference' takes on a unified meaning: the neural network's forward pass directly produces a posterior estimate.

Modern generative modeling techniques such as transformers (Vaswani et al., 2017), flow-matching (Lipman et al., 2023), and diffusion models (Ho et al., 2020; Song et al., 2021) have proven particularly effective for this inverse modeling task, with recent work demonstrating state-of-the-art performance in simulation-based inference (Wildberger et al., 2024; Gloeckler et al., 2024; Chang et al., 2024). These methods learn the inverse mapping by generating training data - (model parameters, data) pairs - through simulation, typically using a uniform training distribution over parameters, equivalent to the prior, to ensure broad coverage of the parameter space.

However, this approach faces key limitations in practice. First, practitioners often possess domainspecific knowledge that could improve inference if incorporated as prior beliefs. Second, researchers may need to conduct prior sensitivity analysis to understand how their modeling assumptions affect conclusions (Elsemüller et al., 2024). Current methods either require retraining with new priors or offer only limited solutions. As the field moves toward larger foundation models for amortized inference (Hollmann et al., 2025), retraining becomes increasingly impractical.

While recent work has proposed techniques for prior specification at inference time (Elsemüller et al., 2024; Chang et al., 2024; Whittle et al., 2025), these amortized approaches are restricted to specific family of priors considered during training - from factorized histograms to Gaussian mixture models. While some of these families are very flexible in principle, training over the space of all meaningful runtime priors becomes rapidly infeasible. Diffusion interval guidance offers runtime

---

#### Page 2

> **Image description.** The image consists of five heatmaps arranged horizontally, each representing a different stage or method in a Bayesian inference process. Each heatmap is a square plot with the x-axis labeled "μ" (mu) and the y-axis labeled "σ" (sigma). The heatmaps use a color gradient, likely from dark purple to bright yellow, to indicate the density or probability of different values of μ and σ.
>
> - **Panel (a) Prior:** The heatmap shows a diagonal elongated region of higher density (yellow/green) stretching from the upper left to the lower right, indicating a prior belief about the relationship between μ and σ.
> - **Panel (b) Likelihood:** The heatmap shows a small, concentrated region of high density (yellow) in the lower right corner, representing the likelihood derived from some observations.
> - **Panel (c) Posterior:** The heatmap shows a distribution that is a combination of the prior and likelihood. The region of high density (yellow/green) is more concentrated than the prior but shifted from the likelihood, representing the Bayesian posterior.
> - **Panel (d) No PriorGuide:** The heatmap shows a concentrated region of high density (yellow/green) in the lower right corner, similar to the likelihood, but with a more pixelated or discrete appearance. This represents the result of a diffusion model without a prior.
> - **Panel (e) PriorGuide:** The heatmap shows a distribution that is similar to the Bayesian posterior. The region of high density (yellow/green) is more spread out than the likelihood, representing the result of a diffusion model with a prior. The distribution has a pixelated or discrete appearance.
>
> Below each heatmap is a text label:
> (a) Prior
> (b) Likelihood
> (c) Posterior
> (d) No PriorGuide
> (e) PriorGuide

Figure 1: Posterior inference with and without PriorGuide. The plots show the mean $\mu$ and standard deviation $\sigma$ parameters of a Gaussian toy model. Prior (a) and likelihood (b) from some observations $\mathbf{x}$ (not shown) yield Bayesian posterior (c). A standard diffusion model trained on a uniform distribution over $\mu, \sigma$ (no prior) matches the likelihood (d). PriorGuide can implement the specified prior for $\mu, \sigma$ at runtime, matching the Bayesian posterior (e).

prior specification, but limited to simple range constraints (Gloeckler et al., 2024). A general solution for incorporating arbitrary priors at runtime remains an open challenge.

Contributions. We introduce PriorGuide, a method that enables flexible incorporation of arbitrary prior beliefs at inference time for diffusion-based amortized inference models. Our approach requires no modifications to the base diffusion model's training procedure and supports more complex priors than previously explored methods. Our method works with existing diffusion-based inference models by implementing the prior as a guidance term. We demonstrate PriorGuide's effectiveness on synthetic examples and a challenging inverse problem. See Fig. 1 for an illustration of our method.

# 2 BACKGROUND

Diffusion models are a powerful framework for generative modeling that transforms samples from arbitrary to simple distributions and vice versa through a gradual noising and denoising process (SohlDickstein et al., 2015). In the forward process, starting from a distribution $p\left(\boldsymbol{\theta}_{0}\right)$, Gaussian noise is progressively added to the samples until, at the end of the process $(t=1)$, the distribution converges to a simple terminal distribution (typically Gaussian). The forward process can be described as:

$$
p\left(\boldsymbol{\theta}_{t}\right)=\int \mathcal{N}\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}, \sigma(t)^{2} \mathbf{I}\right) p\left(\boldsymbol{\theta}_{0}\right) \mathrm{d} \boldsymbol{\theta}_{0}
$$

where $\sigma(t)$ defines the noise variance schedule as a function of time (typically increasing with $t$ ), and $\boldsymbol{\theta}_{t}$ represents the noisy samples at time $t$. The corresponding reverse process reconstructs the original sample distribution from noise, and can be formulated as either a stochastic differential equation (SDE) or an ordinary differential equation (ODE). For the Variance Exploding (VE) SDE (Song et al., 2021; Karras et al., 2022), the reverse process takes the form:

$$
\text { Reverse SDE: } \quad \mathrm{d} \boldsymbol{\theta}_{t}=-2 \hat{\sigma}(t) \sigma(t) \nabla_{\boldsymbol{\theta}} \log p\left(\boldsymbol{\theta}_{t}\right) \mathrm{d} t+\sqrt{2 \hat{\sigma}(t) \sigma(t)} \mathrm{d} \omega_{t}
$$

where $\nabla_{\boldsymbol{\theta}} \log p\left(\boldsymbol{\theta}_{t}\right)$ is the score function (gradient of the log-density), $\mathrm{d} \omega_{t}$ is a Wiener process representing Brownian motion (noise), and $\hat{\sigma}(t)$ is the time derivative of the variance schedule.

Learning the Score Function. The score function $\nabla_{\boldsymbol{\theta}} \log p\left(\boldsymbol{\theta}_{t}\right)$ can be approximated using a neural network $s\left(\boldsymbol{\theta}_{t}, t\right)$, trained to minimize the denoising score matching loss (Hyvärinen \& Dayan, 2005; Vincent, 2011; Song et al., 2021):

$$
\mathcal{L}_{\mathrm{DSM}}=\mathbb{E}_{t \sim \mathcal{U}(0,1)} \mathbb{E}_{\boldsymbol{\theta}_{0} \sim p\left(\boldsymbol{\theta}_{0}\right)} \mathbb{E}_{\boldsymbol{\theta}_{t} \sim \mathcal{N}\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}, \sigma(t)^{2} \mathbf{I}\right)}\left\|s\left(\boldsymbol{\theta}_{t}, t\right)-\nabla_{\boldsymbol{\theta}_{t}} \log p\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}\right)\right\|_{2}^{2}
$$

Once trained, the network $s\left(\boldsymbol{\theta}_{t}, t\right)$ approximates the gradient of the log-probability density of noised distributions and affords sampling through the reverse SDE (Eq. (2)). Starting from a sample $\boldsymbol{\theta}_{t} \sim \mathcal{N}\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}, \sigma_{\max }^{2} \mathbf{I}\right)$ for $t=1$ with sufficiently large $\sigma_{\max }$, integrating the reverse process backward in time approximately reconstructs the original distribution $p\left(\boldsymbol{\theta}_{0}\right)$.

Tweedie's Formula. Tweedie's formula provides a key connection between the posterior mean of $\boldsymbol{\theta}_{0}$ given $\boldsymbol{\theta}_{t}$ and the score function:

$$
\mathbb{E}\left[\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}\right]=\mu_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)=\boldsymbol{\theta}_{t}+\sigma(t)^{2} \nabla_{\boldsymbol{\theta}_{t}} \log p\left(\boldsymbol{\theta}_{t}\right)
$$

---

#### Page 3

This relationship enables direct estimation of the posterior mean at any noise level and establishes an equivalence between $\mu_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)$ and $s\left(\boldsymbol{\theta}_{t}, t\right)$.

The diffusion framework's flexibility stems largely from its ability to incorporate guidance mechanisms, which afford steering the sampling process toward desired outcomes by including additional information or constraints. Notable examples include classifier guidance (Dhariwal \& Nichol, 2021) and classifier-free guidance (Ho \& Salimans, 2022), which afford controlled generation without retraining the model. For inverse problems, this guidance framework has been extended to incorporate likelihood information, particularly for Gaussian likelihoods (Chung et al., 2023; Song et al., 2023a).

For the inverse problems in this work, we learn a score function to approximate the conditional mapping $\nabla_{\boldsymbol{\theta}_{t}} \log p\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right)$ using the direct conditional training approach of Gloeckler et al. (2024). In this framework, the observation $\mathbf{x}$ is provided directly to the score network $s\left(\boldsymbol{\theta}_{t}, t, \mathbf{x}\right)$, similar to the context in conditional neural processes (Garnelo et al., 2018). While our experiments in this paper use this direct approach, we note PriorGuide can also be applied to models using joint training with in-painting guidance (Lugmayr et al., 2022). In either case, PriorGuide adapts the guidance framework to transform the trained prior into an arbitrary prior at inference time.

## 3 PRiORGUIDE

Consider an inverse problem where we observe data $\mathbf{x}$ and aim to infer parameters $\boldsymbol{\theta}$. Standard diffusion models for inverse problems are trained to approximate $\nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta} \mid \mathbf{x})$ via a learned score function $s\left(\boldsymbol{\theta}_{t}, t, \mathbf{x}\right)$, and sampling from the model produces posterior samples $p(\boldsymbol{\theta} \mid \mathbf{x})$ that are anchored to the training distribution (prior) $p(\boldsymbol{\theta})$. This constraint limits flexibility when new prior information becomes available, as incorporating it would traditionally require retraining the score model.

Given a diffusion model trained to sample from posterior $p(\boldsymbol{\theta} \mid \mathbf{x})$ with prior $p(\boldsymbol{\theta})$, our goal is to sample from a modified posterior $q(\boldsymbol{\theta} \mid \mathbf{x})$ that incorporates a new prior $q(\boldsymbol{\theta})$ without retraining. PriorGuide affords prior modification at sampling time by leveraging a basic statistical relationship:
Proposition 1. Let the posterior under the original prior be given as $p(\boldsymbol{\theta} \mid \mathbf{x}) \propto p(\boldsymbol{\theta}) p(\mathbf{x} \mid \boldsymbol{\theta})$, and let the posterior under the new prior be $q(\boldsymbol{\theta} \mid \mathbf{x}) \propto q(\boldsymbol{\theta}) p(\mathbf{x} \mid \boldsymbol{\theta})$. Then, sampling from $q(\boldsymbol{\theta} \mid \mathbf{x})$ is equivalent to sampling from $\rho(\boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathbf{x})$ with $\rho(\boldsymbol{\theta}) \equiv \frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta})}$ the new-over-old prior ratio.

Proof. We can rewrite the new posterior $q(\boldsymbol{\theta} \mid \mathbf{x})$ as

$$
q(\boldsymbol{\theta} \mid \mathbf{x}) \propto q(\boldsymbol{\theta}) p(\mathbf{x} \mid \boldsymbol{\theta})=\frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta})} p(\boldsymbol{\theta}) p(\mathbf{x} \mid \boldsymbol{\theta}) \propto \frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta})} p(\boldsymbol{\theta} \mid \mathbf{x})=\rho(\boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathbf{x})
$$

where the prior ratio $\rho(\boldsymbol{\theta}) \equiv \frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta})}$ takes the role of an importance weighing function.
Modified Posterior Score. Prop. 1, combined with the properties of diffusion models, allows us to express the score of the modified posterior at any time $t$ as:

$$
\begin{aligned}
q\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right) & \propto \int \rho\left(\boldsymbol{\theta}_{0}\right) p\left(\boldsymbol{\theta}_{0} \mid \mathbf{x}\right) p\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}\right) \mathrm{d} \boldsymbol{\theta}_{0} \\
\nabla_{\boldsymbol{\theta}_{t}} \log q\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right) & =\nabla_{\boldsymbol{\theta}_{t}} \log \int \rho\left(\boldsymbol{\theta}_{0}\right) p\left(\boldsymbol{\theta}_{0} \mid \mathbf{x}\right) p\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}, \mathbf{x}\right) \mathrm{d} \boldsymbol{\theta}_{0} \\
& =\nabla_{\boldsymbol{\theta}_{t}} \log \int \rho\left(\boldsymbol{\theta}_{0}\right) p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}, \mathbf{x}\right) p\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right) \mathrm{d} \boldsymbol{\theta}_{0} \\
& =\nabla_{\boldsymbol{\theta}_{t}} \log \int \rho\left(\boldsymbol{\theta}_{0}\right) p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}, \mathbf{x}\right) \mathrm{d} \boldsymbol{\theta}_{0}+\nabla_{\boldsymbol{\theta}_{t}} \log p\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right)
\end{aligned}
$$

where in Eq. (5) we write the modified posterior as an integral over $\boldsymbol{\theta}_{0}$ by noting that $q\left(\boldsymbol{\theta}_{0} \mid \mathbf{x}\right) \propto$ $\rho\left(\boldsymbol{\theta}_{0}\right) p\left(\boldsymbol{\theta}_{0} \mid \mathbf{x}\right)$ and then propagate this information to time $t$ via the transition kernel $p\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}\right)$. In Eq. (6) we write the score, and then re-express the joint probability $p\left(\boldsymbol{\theta}_{0} \mid \mathbf{x}\right) p\left(\boldsymbol{\theta}_{t} \mid \boldsymbol{\theta}_{0}\right)=p\left(\boldsymbol{\theta}_{0}, \boldsymbol{\theta}_{t} \mid \mathbf{x}\right)$ as $p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}, \mathbf{x}\right) p\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right)$, which allows us to separate the contribution of the new prior guidance from the original score model $s\left(\boldsymbol{\theta}_{t}, t, \mathbf{x}\right)$. In multiple steps we exploit the fact that multiplicative constants inside the integral disappear under the score.

---

#### Page 4

We can draw samples from $q\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right)$ via the reverse diffusion process using the modified score:

$$
\nabla_{\boldsymbol{\theta}_{t}} \log q\left(\boldsymbol{\theta}_{t} \mid \mathbf{x}\right) \approx \nabla_{\boldsymbol{\theta}_{t}} \log \mathbb{E}_{p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}, \mathbf{x}\right)}\left[\rho\left(\boldsymbol{\theta}_{0}\right)\right]+s\left(\boldsymbol{\theta}_{t}, t, \mathbf{x}\right)
$$

where first term estimates how the new prior's influence propagates to time $t$ (guidance term) and the second term is our trained score model. This is a common way to implement a guidance function (Chung et al., 2023; Song et al., 2023a;; Rissanen et al., 2024), where now the guidance function is the prior ratio. In the rest of this section, we apply several approximation techniques to estimate the guidance term.

# 3.1 Approximating the Guidance Function

To approximate the guidance term in Eq. (9) efficiently while maintaining flexible inference-time priors, we introduce two approximations. Following recent work (Song et al., 2023a; Peng et al., 2024; Rissanen et al., 2024), we first model the reverse transition kernel as a Gaussian distribution. We then introduce a novel approach that represents $\rho(\boldsymbol{\theta})$ as a Gaussian mixture model. This representation enables both an analytical solution and preserves flexibility in the model. While previous research on inverse problems has explored guidance with linear-Gaussian observation models (Song et al., 2023a), these can be viewed as special cases of our method when using a single mixture component.

Reverse Transition Kernel Approximation. We first approximate the reverse transition kernel $p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}\right)$ as a Gaussian distribution centered at $\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)$, obtained from the score function via Tweedie's formula, Eq. (4). This approximation is common in the guidance literature (Chung et al., 2023; Song et al., 2023a; Peng et al., 2024; Rissanen et al., 2024; Finzi et al., 2023; Bao et al., 2022). For the covariance matrix $\boldsymbol{\Sigma}_{0 \mid t}$, we adopt a simple yet effective approximation inspired by Song et al. (2023a); Ho et al. (2022):

$$
\boldsymbol{\Sigma}_{0 \mid t}=\frac{\sigma(t)^{2}}{1+\sigma(t)^{2}} \mathbf{I}
$$

This approximation acts as a time-dependent scaling factor that naturally aligns with the diffusion process - starting at the identity matrix when $t=1$ and approaching zero as $t \rightarrow 0$, effectively increasing the precision of our prior guidance at smaller timesteps.

Prior Ratio Approximation. We then approximate the prior ratio function $\rho(\boldsymbol{\theta})=\frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta})}$ as a generalized mixture of Gaussians:

$$
\rho(\boldsymbol{\theta}) \approx \sum_{i=1}^{K} w_{i} \mathcal{N}\left(\boldsymbol{\theta} \mid \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right), \quad \rho(\boldsymbol{\theta}) \geq 0
$$

where $\left\{w_{i}, \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right\}_{i=1}^{K}$ represent the weights, means and covariance matrices of the mixture. Since this represents a ratio rather than a distribution, the mixture weights need not be positive nor sum to one, as long as the ratio remains non-negative, potentially enabling more expressive approximations such as subtractive mixtures (Loconte et al., 2024). Notably, when $p(\boldsymbol{\theta})$ is uniform (as in our experiments), $\rho(\boldsymbol{\theta})$ reduce to $q(\boldsymbol{\theta})$, and we directly specify it as a Gaussian mixture. For non-uniform training distributions, the ratio function can be fit with a generalized Gaussian mixture approximation, which can theoretically approximate any continuous function (Sorenson \& Alspach, 1971).

Guidance Term. With these Gaussian approximations, the guidance term becomes:

$$
\nabla_{\boldsymbol{\theta}_{t}} \log \mathbb{E}_{p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}, \mathbf{x}\right)}\left[\rho\left(\boldsymbol{\theta}_{0}\right)\right] \approx \nabla_{\boldsymbol{\theta}_{t}} \log \int \sum_{i=1}^{K} w_{i} \mathcal{N}\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \mathcal{N}\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \boldsymbol{\Sigma}_{0 \mid t}\right) \mathrm{d} \boldsymbol{\theta}_{0}
$$

This integral can be solved analytically (full derivation in Appendix A.1), yielding:

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}_{t}} \log \mathbb{E}_{p\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\theta}_{t}, \mathbf{x}\right)}\left[\rho\left(\boldsymbol{\theta}_{0}\right)\right] & \approx \frac{\sum_{i=1}^{K} w_{i} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widehat{\boldsymbol{\Sigma}}_{i}\right)\left(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)\right)^{\mathbf{T}} \widehat{\boldsymbol{\Sigma}}_{i}^{-1} \nabla_{\boldsymbol{\theta}_{t}} \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)}{\sum_{i=1}^{K} w_{i} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widehat{\boldsymbol{\Sigma}}_{i}\right)} \\
& =\sum_{i}^{K} \hat{w}_{i}\left(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)\right)^{\mathbf{T}} \widehat{\boldsymbol{\Sigma}}_{i}^{-1} \nabla_{\boldsymbol{\theta}_{t}} \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)
\end{aligned}
$$

---

#### Page 5

where $\widehat{\boldsymbol{\Sigma}}_{i}=\boldsymbol{\Sigma}_{i}+\boldsymbol{\Sigma}_{0 \mid t}$ and $\hat{w}_{i}=w_{i} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widehat{\boldsymbol{\Sigma}}_{i}\right) / \sum_{j=1}^{K} w_{j} \mathcal{N}\left(\boldsymbol{\mu}_{j} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widehat{\boldsymbol{\Sigma}}_{j}\right)$. For typical inverse problems where parameter dimensionality is below 100, these calculations remain computationally tractable. However, higher-dimensional problems would require additional approximations, particularly for the log determinant and matrix inversion.
Finally, the PriorGuide update to the mean of the reverse kernel can be expressed concisely using Tweedie's formula, Eq. (4), and our derived guidance term, Eq. (14):

$$
\mu_{0 \mid t}^{\text {new }}\left(\boldsymbol{\theta}_{t}\right)=\mu_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)+\sigma(t)^{2} \sum_{i}^{K} \hat{w}_{i}\left(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)\right)^{\mathrm{T}} \widehat{\boldsymbol{\Sigma}}_{i}^{-1} \nabla_{\boldsymbol{\theta}_{t}} \boldsymbol{\mu}_{0 \mid t}
$$

This update intuitively combines the original prediction $\mu_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)$ with a weighted sum of correction terms from our new prior. The correction magnitude is controlled by both the noise schedule $\sigma(t)^{2}$ and the distance between the mixture components and current prediction.

## 4 EXPERIMENTS

We evaluate PriorGuide using the base model from Simformer (Gloeckler et al., 2024), trained with the variance exploding SDE (Song et al., 2021). Notably, our method requires no modifications to the original diffusion model's training procedure and works by adjusting the guidance term at inference time as described in Section 3.

Toy Gaussian Example. We first consider a simple but illustrative case: inferring $\boldsymbol{\theta}=(\mu, \sigma)$, the mean $\mu$ and standard deviation $\sigma$ of a Gaussian distribution from 10 data points. The prior is a correlated multivariate Gaussian (Fig. 1a), and the likelihood for a specific set of observations $\mathbf{x}$ is shown in Fig. 1b. We numerically compute the true posterior for comparison (Fig. 1c). The base model is trained with a uniform prior over $(\mu, \sigma)$ to learn the inverse mapping. When sampling without prior guidance (Fig. 1d), the model focuses on the region around the data, reflecting its uniform training prior. However, when using PriorGuide (Fig. 1e), the samples closely match the true posterior (Fig. 1c), despite the substantial separation between prior and likelihood regions. This demonstrates PriorGuide's ability to successfully incorporate new priors at inference time to recover $q(\boldsymbol{\theta} \mid \mathbf{x})$. Complete experimental details are provided in Appendix A.2.

Two Moons with Correlated Prior. The two moons example is a common benchmark for simulation-based inference. Here, we add a strong correlated prior $q(\boldsymbol{\theta})$ to test how our method handles a multi-modal scenario (Fig. A.1a). PriorGuide correctly captures the multimodality of the problem through its posterior distribution (Fig. A.1b). For validation, we compare PriorGuide's results with a ground truth baseline obtained by retraining the base model with $q(\boldsymbol{\theta})$; the comparison of samples is shown in Fig. A.1c. For quantitative validation, we compared samples from PriorGuide and the retrained model across 10 different observations $\mathbf{x}$ using the Classifier 2-Sample Tests (C2ST) score (Lopez-Paz \& Oquab, 2017). The C2ST score measures how well a classifier can distinguish between two sets of samples, with 0.5 indicating indistinguishable samples. Between the retrained model and PriorGuide samples, we obtain a score of $0.623 \pm 0.044$. For context, the score between the base diffusion model and standard MCMC samples is $0.523 \pm 0.016$, demonstrating that PriorGuide generates comparable samples without requiring retraining. See Appendix A. 2 for model details.

Benchmark SBI Tasks. Finally, we evaluate PriorGuide on two simulation-based inference tasks of increasing complexity: the Ornstein-Uhlenbeck Process (OUP), a time-series model with two latent variables (Uhlenbeck \& Ornstein, 1930), and the Turin model (Turin et al., 1972), a radio propagation simulator with four parameters that generates 101-dimensional signal data.
For both tasks, we set the sampling distribution of $\boldsymbol{\theta}$ in two ways: (i) as a uniform distribution and (ii) as a correlated Gaussian mixture distribution. We can then test the ability of a model of incorporating prior information by passing useful information about the sampled $\boldsymbol{\theta}$. In the uniform case, we provide information by sampling the prior location from a Gaussian around the true $\boldsymbol{\theta}$, and giving that Gaussian prior to models that support runtime priors, following Chang et al., 2024. In the correlated Gaussian mixture case, we pass a prior that exactly matches the true inference-time sampling distribution. Further experimental details are provided in Appendix A.2.

---

#### Page 6

|       |                     | Uniform $\boldsymbol{\theta}$ sampling |              |              | Mixture $\boldsymbol{\theta}$ sampling |              |                          |
| :---: | :-----------------: | :------------------------------------: | :----------: | :----------: | :------------------------------------: | :----------: | :----------------------: |
|       |                     |               Simformer                |     ACE      |     ACEP     |               PriorGuide               |  Simformer   |        PriorGuide        |
|  OUP  | RMSE $(\downarrow)$ |              $0.61(0.03)$              | $0.59(0.00)$ | $0.21(0.02)$ |        $\mathbf{0 . 1 7}(0.01)$        | $0.51(0.04)$ | $\mathbf{0 . 4 0}(0.02)$ |
|       | MMD $(\downarrow)$  |              $0.19(0.01)$              | $0.15(0.00)$ | $0.04(0.00)$ |        $\mathbf{0 . 0 3}(0.00)$        | $0.16(0.02)$ | $\mathbf{0 . 0 9}(0.01)$ |
| Turin | RMSE $(\downarrow)$ |              $0.25(0.00)$              | $0.25(0.00)$ | $0.10(0.01)$ |        $\mathbf{0 . 0 7}(0.00)$        | $0.26(0.00)$ | $\mathbf{0 . 1 8}(0.00)$ |
|       | MMD $(\downarrow)$  |              $0.11(0.00)$              | $0.11(0.00)$ | $0.02(0.00)$ |        $\mathbf{0 . 0 1}(0.00)$        | $0.08(0.00)$ | $\mathbf{0 . 0 4}(0.00)$ |

Table 1: Comparison of SBI task metrics for $\boldsymbol{\theta}$ prediction; mean (standard deviation) over 5 runs. Best results are bolded. Left: Uniform sampling distribution for $\boldsymbol{\theta}$, with an informative Gaussian prior given to ACEP and PriorGuide. Right: Correlated mixture sampling distribution, with the same distribution given as prior to PriorGuide.

As a baseline, we compare our method, PriorGuide, with the same base SimFormer model without prior guidance (Gloeckler et al., 2024). We also consider another amortized inference method, the Amortized Conditioning Engine (ACE; Chang et al., 2024), whose ACEP variant affords runtime incorporation of factorized priors seen during training. Table 1 presents the benchmark results. In the uniform $\boldsymbol{\theta}$ case, we compare PriorGuide with an informative Gaussian prior against Simformer and ACE (both without priors), and ACE with the same simple prior (ACEP). In the mixture sampling case, we compare base SimFormer with PriorGuide guided by the sampling distribution as prior. ${ }^{1}$ PriorGuide outperforms all baselines in both settings, demonstrating its capabilities of incorporating prior information at test time without retraining. Example visualizations of results on the SBI experiments are presented in Appendix A.3.

## 5 RELATED WORK

PriorGuide builds on advances in three key areas: diffusion models for inverse problems, simulationbased inference (SBI), and guidance techniques for controllable generation. Recent work has adapted diffusion models to scientific applications with intractable forward models, treating inverse problems as conditional generation (Chung et al., 2023). Methods like those in Gloeckler et al. (2024) train diffusion models to directly approximate the posterior. However, these approaches fix the prior during training, limiting their flexibility. Recent work in Elsemüller et al. (2024); Chang et al. (2024); Whittle et al. (2025) showed the effectiveness of inference time priors, but the approach is limited. In inverse problems, reconstruction guidance (Chung et al., 2023) incorporates likelihood gradients during sampling. Related approaches from Rissanen et al. (2024); Finzi et al. (2023); Bao et al. (2022); Peng et al. (2024) use Tweedie's formula to guide sampling, but focus on refining the likelihood term rather than modifying the prior. PriorGuide uniquely repurposes guidance mechanisms to inject new prior information, combining the flexibility of score-based methods with the expressiveness of Gaussian mixture priors.

## 6 DISCUSSION

In this work, we introduced PriorGuide, a technique that enables the use of flexible, user-defined priors at inference time for diffusion-based amortized inference methods. Our experiments demonstrate that PriorGuide can effectively recover posterior distributions under new priors. This capability is particularly valuable in scientific applications where prior knowledge is often refined post-training, for prior sensitivity analysis or with large inference models, where retraining is undesirable.

Limitations. While PriorGuide offers significant flexibility, it has several important limitations: First, the computational cost scales with parameter dimensionality due to the weighted averaging over Gaussian components. Very high-dimensional problems may require additional approximations to maintain efficiency. Furthermore, PriorGuide assumes the new prior ratio can be well-approximated by a Gaussian mixture. While highly expressive, this may not capture all possible prior distributions, particularly those with heavy tails or discrete components. Future work could develop automatic conversion of arbitrary priors into approximate Gaussian mixtures. Additionally, integrating PriorGuide with in-painting style guidance techniques could enhance its applicability to a wider range of inverse problems by removing the need to specify conditioning variables upfront, offering further flexibility.

[^0]
[^0]: ${ }^{1}$ ACEP does not afford complex correlated priors, so it is not included.
