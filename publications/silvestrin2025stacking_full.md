```
@article{silvestrin2025stacking,
  title={Stacking Variational Bayesian Monte Carlo},
  author={Francesco Silvestrin and Chengkun Li and Luigi Acerbi},
  year={2025},
  journal={TMLR},
  doi={10.48550/arXiv.2504.05004},
  url={https://www.semanticscholar.org/paper/e236094d9b9f6a20a5e103968138345cde97e845}
}
```

---

#### Page 1

# Stacking Variational Bayesian Monte Carlo

Francesco Silvestrin<br>Department of Computer Science<br>University of Helsinki<br>Chengkun Li<br>chengkun.li@helsinki.fi<br>Department of Computer Science<br>University of Helsinki

Luigi Acerbi
luigi.acerbi@helsinki.fi
Department of Computer Science
University of Helsinki

## Abstract

Approximate Bayesian inference for models with computationally expensive, black-box likelihoods poses a significant challenge, especially when the posterior distribution is complex. Many inference methods struggle to explore the parameter space efficiently under a limited budget of likelihood evaluations. Variational Bayesian Monte Carlo (VBMC) is a sampleefficient method that addresses this by building a local surrogate model of the log-posterior. However, its conservative exploration strategy, while promoting stability, can cause it to miss important regions of the posterior, such as distinct modes or long tails. In this work, we introduce Stacking Variational Bayesian Monte Carlo (S-VBMC), a method that overcomes this limitation by constructing a robust, global posterior approximation from multiple independent VBMC runs. Our approach merges these local approximations through a principled and inexpensive post-processing step that leverages VBMC's mixture posterior representation and per-component evidence estimates. Crucially, S-VBMC requires no additional likelihood evaluations and is naturally parallelisable, fitting seamlessly into existing inference workflows. We demonstrate its effectiveness on two synthetic problems designed to challenge VBMC's exploration and two real-world applications from computational neuroscience, showing substantial improvements in posterior approximation quality across all cases. Our code is available as a Python package at https://github.com/acerbilab/svbmc.

## 1 Introduction

Bayesian inference provides a powerful framework for parameter estimation and uncertainty quantification, but it is usually intractable, requiring approximate inference techniques (Brooks et al., 2011; Blei et al., 2017). Many scientific and engineering problems involve black-box models (Sacks et al., 1989; Kennedy \& O'Hagan, 2001), where likelihood evaluation is time-consuming and gradients cannot be easily obtained, making traditional approximate inference approaches computationally prohibitive.

A promising approach to tackle expensive likelihoods is to construct a statistical surrogate model that approximates the target distribution, similar in spirit to surrogate approaches to global optimisation using Gaussian processes, generally known as Bayesian optimisation (Rasmussen \& Williams, 2006; Garnett, 2023). However, unlike Bayesian optimisation, where the goal is to find a single point estimate (the global optimum), here the aim is to reconstruct the shape of the entire posterior distribution. In this setting, attempting to build a single global surrogate model may lead to numerical instabilities and poor approximations when the target distribution is complex or multi-modal, without ad hoc solutions (Wang \& Li, 2018; Järvenpää et al., 2021; Li et al., 2025). Local or constrained surrogate models, while more limited in scope, tend to be more stable and reliable in practice (El Gammal et al., 2023; Järvenpää \& Corander, 2024).

---

#### Page 2

Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018) exemplifies this local approach, using active sampling to train a Gaussian process surrogate for the unnormalised log-posterior on which it performs variational inference. VBMC adopts a conservative exploration strategy that yields stable, local approximations (Acerbi, 2019). Compared to other surrogate-based approaches, the method offers a versatile set of features: it returns the approximate posterior as a tractable distribution (a mixture of Gaussians); it provides a lower bound for the model evidence (ELBO) via Bayesian quadrature (Ghahramani & Rasmussen, 2002), useful for model selection; and it can handle noisy log-likelihood evaluations (Acerbi, 2020), which arise in simulation-based models through estimation techniques such as inverse binomial sampling (van Opheusden et al., 2020) and synthetic likelihood (Wood, 2010; Price et al., 2018). However, VBMC's limited likelihood evaluation budget combined with its local exploration strategy can leave it vulnerable to potentially missing regions of the target posterior – particularly for distributions with distinct modes or long tails.

In this work, we propose a practical, yet effective approach to constructing global surrogate models while overcoming the limitations of standard VBMC by combining multiple local approximations. We introduce Stacking Variational Bayesian Monte Carlo (S-VBMC), a method for merging independent VBMC inference runs into a coherent global posterior approximation. S-VBMC inherits its parent algorithm's operating regime and is intended for low- to moderate-dimensional problems (D ≤ 10). Our approach leverages VBMC's unique properties – its mixture posterior representation and per-component Bayesian quadrature estimates of the ELBO – to combine and reweigh each component through a simple post-processing step. Figure 1 shows an example of two separate posteriors and the combined ("stacked") result obtained with S-VBMC.

> **Image description.** The image displays two side-by-side contour plots, illustrating the concept of combining separate posterior distributions into a single, "stacked" posterior. Both plots share identical axis ranges and labels.
>
> The left panel, titled "Separate VBMC posteriors", shows two distinct sets of contour lines. The x-axis is labeled "$\theta_2$" with numerical ticks at 1e-5, 2e-5, and 3e-5. The y-axis is labeled "$\theta_4$" with numerical ticks at -0.01, 0.00, and 0.01. One set of contours is colored blue, forming an elongated, roughly elliptical shape concentrated in the upper-left portion of the plot, peaking around (1e-5, 0.01). The second set of contours is colored orange, forming a similar elongated shape concentrated in the lower-right portion, peaking around (2e-5 to 3e-5, 0.00 to -0.01). These two distributions overlap significantly in the central region of the plot, creating a criss-cross pattern. The inner contours of each set represent higher probability density.
>
> The right panel, titled "Stacked posterior", presents a single set of magenta-colored contour lines. The x-axis and y-axis are identical to the left panel, labeled "$\theta_2$" and "$\theta_4$" respectively, with the same tick marks. This single distribution appears as an elongated, banana-shaped region that broadly encompasses the areas covered by both the blue and orange contours from the left panel. It stretches from the upper-left corner of the plot to the lower-right corner, indicating a strong negative correlation between $\theta_2$ and $\theta_4$. The contours are densest in the central part of this elongated shape, gradually spreading out towards the edges, signifying varying levels of probability density within the combined distribution.
>
> In essence, the image visually demonstrates how two distinct, overlapping posterior distributions (blue and orange) can be merged to form a single, broader, and more comprehensive "stacked" posterior distribution (magenta).

Figure 1: Two separate VBMC posteriors (left, shown as blue and orange contours) and the resulting stacked posterior via S-VBMC (right, red contour) for a neuronal model with real data (see Section 4.5); showing the marginal distribution of two out of the five model parameters.

Crucially, our method requires no additional evaluations of either the original model or the surrogate. This approach is easily parallelisable and naturally fits existing VBMC pipelines that already employ multiple independent runs (Huggins et al., 2023). While our method could theoretically extend to other variational approaches based on mixture posteriors, VBMC is uniquely suitable for it as re-estimation of the ELBO would otherwise become impractical with expensive likelihoods (see Section 3).

**Related work.** Our work addresses the challenge of building global posterior approximations by combining local solutions from the VBMC framework (Acerbi, 2018; 2019; 2020). While the idea of combining posterior distributions has been explored before, previous approaches differ substantially in their goals and methodology.

"Stacking" was first introduced in the context of supervised learning as a method for model averaging (Wolpert, 1992). Given a set of predictive models, the idea was to use a weighted average of their outputs, with the weights optimised to minimise the leave-one-out squared error. This approach has then been

---

#### Page 3

adapted to average Bayesian predictive distributions (Yao et al., 2018; 2022), which the authors named Bayesian stacking. This still relies on a leave-one-out strategy to optimise predictive performance, which requires access to the likelihood per data point, while S-VBMC optimises the ELBO on the full dataset, allowing treatment of the log-joint as a black box.

Another relevant approach is variational boosting (Guo et al., 2016; Miller et al., 2017; Campbell \& Li, 2019). This method builds on black-box variational inference (Ranganath et al., 2014) and entails iteratively running a series of variational optimisations on the whole dataset, with each iteration increasing the complexity (number of components) of a mixture posterior distribution. This allows practitioners to obtain arbitrarily complex (and accurate) posteriors, trading compute time for inference accuracy. However, the process is inherently sequential, whilst S-VBMC can be implemented as a simple post-processing step, allowing individual (VBMC) inference runs to happen in parallel, offering significant computational advantages, further substantiated by VBMC's surrogate-based approach and sample efficiency, which make it particularly suitable for problems with expensive likelihoods.

Parallel computations have, on the other hand, been leveraged in a number of "divide-and-conquer" or embarrassingly parallel approximate inference techniques, starting from embarrassingly parallel Markov Chain Monte Carlo (MCMC) (Neiswanger et al., 2014). All these methods are based on dividing the data into subsets which are processed separately to obtain a set of "sub-posteriors", which are then merged in various ways to recreate the full (approximate) posterior (Wang \& Dunson, 2013; Wang et al., 2015; Nemeth \& Sherlock, 2018; Srivastava et al., 2018; Scott et al., 2022; De Souza et al., 2022; Chan et al., 2023). These methods are mostly motivated by the need to process very large datasets (Scott et al., 2022), or by privacy concerns requiring federated learning (Liang et al., 2025). In contrast, S-VBMC is motivated by a need to capture complex posteriors that elude single inference runs. Therefore, individual runs all use the complete dataset. This allows our method not to rely on the quality and representativeness of the sub-datasets, and to remain robust to individual run failures.

Outline. We first introduce variational inference and VBMC (Section 2), then present our algorithm for stacking VBMC posteriors (Section 3). We demonstrate the effectiveness of our approach through experiments on two synthetic problems and two real-world applications that are challenging for VBMC (Section 4). We then discuss an observed bias buildup in the ELBO estimation and propose a practical heuristic to counteract it (Section 5). We finally discuss our results (Section 6) and conclude with closing remarks (Section 7). Appendix A contains supplementary materials.

# 2 Background

### 2.1 Bayesian Inference

Given a dataset $\mathcal{D}$, and a model parametrised by the vector $\boldsymbol{\theta} \in \mathbb{R}^{D}$ describing how $\mathcal{D}$ was generated, Bayesian inference represents a principled framework to infer the probability distributions over $\boldsymbol{\theta}$. This is achieved through Bayes' rule:

$$
p(\boldsymbol{\theta} \mid \mathcal{D})=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{p(\mathcal{D})}
$$

where $p(\boldsymbol{\theta})$ represents the prior over model parameters, $p(\mathcal{D} \mid \boldsymbol{\theta})$ the likelihood and $p(\mathcal{D})=\int p(\boldsymbol{\theta}) p(\mathcal{D} \mid$ $\boldsymbol{\theta}) d \boldsymbol{\theta}$ the normalising constant, also called marginal likelihood or model evidence. This latter quantity is particularly useful for Bayesian model selection (MacKay, 2003), but the integral is often intractable, requiring some approximation technique (Brooks et al., 2011; Blei et al., 2017). In the following section, we discuss one of such techniques.

### 2.2 Variational Inference

Considering the setup outlined in Eq. 1, variational inference (Blei et al., 2017) is a technique that approximates the true posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$ with a parametric distribution $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$, where $\boldsymbol{\phi}$ denotes the optimisable

---

#### Page 4

variational parameters. This is achieved by maximising the evidence lower bound (ELBO):

$$
\operatorname{ELBO}(\boldsymbol{\phi})=\mathbb{E}_{q_{\phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]
$$

where the first term is the expected log-joint distribution (the joint being likelihood times prior) and the second term is the entropy of the variational posterior. Maximising Eq. 2 is equivalent to minimising the Kullback-Leibler divergence between $q_{\phi}(\boldsymbol{\theta})$ and the true posterior, as

$$
\begin{aligned}
D_{\mathrm{KL}}\left[q_{\phi}(\boldsymbol{\theta}) \| p(\boldsymbol{\theta} \mid \mathcal{D})\right] & =\mathbb{E}_{q_{\phi}}\left[\log \frac{q_{\phi}(\boldsymbol{\theta})}{p(\boldsymbol{\theta} \mid \mathcal{D})}\right] \\
& =-\mathbb{E}_{q_{\phi}}[\log p(\boldsymbol{\theta}, \mathcal{D})]+\mathbb{E}_{q_{\phi}}[\log p(\mathcal{D})]+\mathbb{E}_{q_{\phi}}\left[\log q_{\phi}(\boldsymbol{\theta})\right]
\end{aligned}
$$

and, since the model evidence $p(\mathcal{D})$ is already a constant,

$$
\begin{aligned}
\log p(\mathcal{D})-D_{\mathrm{KL}}\left[q_{\phi}(\boldsymbol{\theta}) \| p(\boldsymbol{\theta} \mid \mathcal{D})\right] & =\mathbb{E}_{q_{\phi}}[\log p(\boldsymbol{\theta}, \mathcal{D})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right] \\
& =\operatorname{ELBO}(\boldsymbol{\phi})
\end{aligned}
$$

Crucially, since $D_{\mathrm{KL}}[q \| p] \geq 0$, the ELBO provides a lower bound on the log model evidence $\log p(\mathcal{D})$ (hence the name), with equality when the approximation matches the true posterior (i.e., when $D_{\mathrm{KL}}[q \| p]=0$ ), thus constituting a useful metric for model selection.

# 2.3 Variational Bayesian Monte Carlo (VBMC)

VBMC (Acerbi, 2018; 2020) is a sample-efficient technique to obtain a variational approximation of a target density with only a small number of likelihood evaluations, often of the order of a few hundred. VBMC uses a Gaussian process (GP) as a surrogate of the log-joint, Bayesian quadrature to calculate the expected log-joint, and active sampling to decide which parameters to evaluate next. As an in-depth knowledge of the inner workings of VBMC is not necessary to understand our core contribution, here we only describe its relevant aspects. An interested reader should refer to the original papers (Acerbi, 2018; 2020) or Appendix A.1, where we provide a more detailed description of the algorithm.

As mentioned above, VBMC uses a GP as a surrogate of the target, which is often an unnormalised logposterior (i.e., the log-joint). Crucially, VBMC performs variational inference on the surrogate

$$
f(\boldsymbol{\theta}) \approx \log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})
$$

instead of the true, expensive model. The efficacy of VBMC hinges on the quality of such approximation.
The variational posterior in VBMC is defined as a flexible mixture of $K$ components,

$$
q_{\phi}(\boldsymbol{\theta})=\sum_{k=1}^{K} w_{k} q_{k, \phi}(\boldsymbol{\theta})
$$

where $q_{k}$ is the $k$-th component (a multivariate normal) and $w_{k}$ its mixture weight, with $\sum_{k=1}^{K} w_{k}=1$ and $w_{k} \geq 0$. Plugging in the mixture posterior, the ELBO (Eq. 2) becomes:

$$
\operatorname{ELBO}(\boldsymbol{\phi})=\sum_{k=1}^{K} w_{k} \mathbb{E}_{q_{k, \phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]=\sum_{k=1}^{K} w_{k} I_{k}+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]
$$

where we defined the $k$-th component of the expected log-joint as:

$$
I_{k}=\mathbb{E}_{q_{k, \phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})] \approx \mathbb{E}_{q_{k, \phi}}[f(\boldsymbol{\theta})]=\hat{I}_{k}
$$

The efficacy of VBMC stems from the fact that Eq. 8 has a closed-form Gaussian expression via Bayesian quadrature (O'Hagan, 1991; Ghahramani \& Rasmussen, 2002), which yields posterior mean $\hat{I}_{k}$ and covariance matrix $J_{k k^{\prime}}$ (Acerbi, 2018). The entropy of a mixture of Gaussians does not have an analytical solution, but

---

#### Page 5

gradients can be estimated via Monte Carlo. Thus, using the posterior mean of Eq. 8 as a plug-in estimator for the expected log-joint of each component, Eq. 7 can be efficiently optimised via stochastic gradient ascent (Kingma \& Ba, 2014).

VBMC differs from other approaches that directly try to apply Bayesian quadrature to solve Bayes' rule (e.g., Ghahramani \& Rasmussen, 2002; Osborne et al., 2012; Gunter et al., 2014; Adachi et al., 2022), in that instead it leverages Bayesian quadrature to estimate the ELBO. This is a simpler problem, since instead of attempting to solve the global integral of Eq. 1, namely the integral of prior times likelihood, where the prior is often diffuse, it mainly deals with the local integrals in Eq. 7, namely the expected log-joint, i.e., the integral of the approximate posterior times log-joint, where the approximate posterior is under our control and often more localised, and the entropy, which can often be estimated or approximated for known distributions.

# 3 Stacking VBMC

In this work, we introduce Stacking VBMC (S-VBMC), a novel approach to merge different variational posteriors obtained from different runs on the same model and dataset. Crucially, these runs can happen in parallel, with no information exchange required. The core idea is that a single global surrogate of the log-joint $f(\boldsymbol{\theta})$ might be inaccurate in some regions of the posterior. Therefore, it would be beneficial to leverage the combination of $M$ local surrogates $\left\{f_{m}(\boldsymbol{\theta})\right\}_{m=1}^{M}$ instead, each yielding a good approximation of the target in a different parameter region.

### 3.1 Optimisation objective

Given $M$ independent VBMC runs, one obtains $M$ variational posteriors

$$
q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})=\sum_{k=1}^{K_{m}} w_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

each with $K_{m}$ Gaussian components, as well as $M$ different $\tilde{\mathbf{I}}_{m}$ vectors, as per Eq. 8, with $\tilde{\mathbf{I}}_{m}=$ $\left(\tilde{I}_{m, 1}, \ldots, \tilde{I}_{m, K_{m}}\right)$. Our approach consists of "stacking" the Gaussian components of all posteriors $q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})$, leaving all individual components' parameters (means and covariances) unchanged, and reoptimising all the weights. The full set of new mixture weights is denoted by the vector $\hat{\mathbf{w}}=\left\{\tilde{w}_{m, k}\right\}_{m=1, k=1}^{M, K_{m}}$. The complete set of parameters for the final stacked posterior, $q_{\tilde{\boldsymbol{\phi}}}$, is denoted by $\tilde{\boldsymbol{\phi}}$. This set consists of the frozen parameters (all means and covariances) from the original components $\left\{q_{k, \boldsymbol{\phi}_{m}}\right\}$ and the newly optimised weights $\hat{\mathbf{w}}$.

Thus, given the stacked posterior

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

we optimise the global evidence lower bound with respect to the weights $\hat{\mathbf{w}}$,

$$
\operatorname{ELBO}_{\text {stacked }}(\hat{\mathbf{w}})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \tilde{I}_{m, k}+\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

It should be noted that each VBMC run applies its own parameter transformation $g_{m}(\cdot)$ during inference (Acerbi, 2020), so variational posteriors are returned in transformed coordinates $q_{\boldsymbol{\phi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right)$. For clarity, in this section, we present all quantities in a common parameter space of $\boldsymbol{\theta}$ and VBMC and S-VBMC posteriors expressed accordingly as $q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})$, and $q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})$, respectively. In practice, when applying our stacking approach, we account for these per-run transformations via change-of-variables (Jacobian) corrections; see Appendix A. 2 for details.

---

#### Page 6

# 3.2 Algorithm

As the entropy term in Eq. 11 does not have a closed-form solution, it needs to be estimated via Monte Carlo sampling. To do this, we take $S$ samples $\left\{\mathbf{x}_{m, k}^{(s)} \sim q_{k, \boldsymbol{\phi}_{m}}\right\}_{s=1}^{S}$ from each component of the stacked posterior, and estimate the entropy as

$$
\mathcal{H}\left[q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] \approx-\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k}\left(\frac{1}{S} \sum_{s=1}^{S} \log q_{\hat{\boldsymbol{\phi}}}\left(\mathbf{x}_{m, k}^{(s)}\right)\right)
$$

This works because

$$
\begin{aligned}
\mathcal{H}\left[q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] & =-\mathbb{E}_{q_{\hat{\boldsymbol{\phi}}}}\left[\log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] \\
& =-\int q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta}) \log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta}) d \boldsymbol{\theta} \\
& =-\int \sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta}) \log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta}) d \boldsymbol{\theta} \\
& =-\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \int q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta}) \log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta}) d \boldsymbol{\theta} \\
& =-\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \mathbb{E}_{q_{k, \boldsymbol{\phi}_{m}}}\left[\log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
\end{aligned}
$$

where we approximate the expectation via Monte Carlo,

$$
\mathbb{E}_{q_{k, \boldsymbol{\phi}_{m}}}\left[\log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] \approx \frac{1}{S} \sum_{s=1}^{S} \log q_{\hat{\boldsymbol{\phi}}}\left(\mathbf{x}_{m, k}^{(s)}\right)
$$

Importantly, with this formulation, the entropy is differentiable with respect to the weights $\tilde{\mathbf{w}}$.
Since weights must be non-negative and sum to one (i.e., they live on the probability simplex), for the purpose of optimisation we follow standard practices in computational statistics (e.g., Carpenter et al., 2017): we reparameterise the weights with unconstrained logits a and compute the objective with softmax weights,

$$
\tilde{w}_{m, k}=\frac{\exp \left(a_{m, k}\right)}{\sum_{m^{\prime}=1}^{M} \sum_{k^{\prime}=1}^{K_{m^{\prime}}} \exp \left(a_{m^{\prime}, k^{\prime}}\right)}
$$

We initialise the logits as

$$
a_{m, k}=\log \left(w_{m, k}\right)+\operatorname{ELBO}\left(\boldsymbol{\phi}_{m}\right)-\max _{\substack{1 \leq m^{\prime} \leq M \\ 1 \leq k^{\prime} \leq K_{m^{\prime}}}} \left(\log \left(w_{m^{\prime}, k^{\prime}}\right)+\operatorname{ELBO}\left(\boldsymbol{\phi}_{m^{\prime}}\right)\right)
$$

where the last term ensures that $\max (\mathbf{a})=0$, preventing underflow in the exponentials. This is equivalent to initialising the weights as

$$
\tilde{w}_{m, k} \propto w_{m, k} \exp \left(\operatorname{ELBO}\left(\boldsymbol{\phi}_{m}\right)\right)
$$

effectively weighting each individual posterior by its approximate evidence (i.e., the exponential of the corresponding ELBO). This assigns a higher initial weight to the components coming from the best runs, providing a good starting point for the optimisation process.
We can then use the Adam optimiser (Kingma \& Ba, 2014) to optimise the stacked ELBO. An overview of the procedure is outlined in Algorithm 1.

Notably, the optimisation can be performed as a pure post-processing step, requiring neither evaluations of the original likelihood $p(\mathcal{D} \mid \boldsymbol{\theta})$ nor of the surrogate models $f_{m}$, only that the estimates $\hat{I}_{m, k}$ are stored, as in current implementations (Huggins et al., 2023).

---

#### Page 7

# Algorithm 1 S-VBMC

Input: outputs of $M$ VBMC runs $\left\{\boldsymbol{\phi}_{m}, \tilde{\mathbf{I}}_{m}, \operatorname{ELBO}\left(\boldsymbol{\phi}_{m}\right)\right\}_{m=1}^{M}$

1. Stack the posteriors to obtain a mixture of $\sum_{m=1}^{M} K_{m}$ Gaussians
2. Reparametrise the weights with unconstrained logits a
3. Initialise the logits as in Eq. 16
4. repeat
   (a) Compute the normalised weights $\hat{\mathbf{w}}$ as in Eq. 15
   (b) Estimate $\mathrm{ELBO}_{\text {stacked }}(\hat{\mathbf{w}})$ as in Eq. 11 with the entropy approximated as in Eq. 12
   (c) Compute $\frac{d \mathrm{ELBO}_{\text {stacked }}(\mathbf{a})}{d \mathbf{a}}$
   (d) Update a with a step of Adam (Kingma \& Ba, 2014)
   until $\mathrm{ELBO}_{\text {stacked }}(\hat{\mathbf{w}})$ converges
   return $\hat{\boldsymbol{\phi}}, \mathrm{ELBO}_{\text {stacked }}$

Our stacking method hinges on the key feature of VBMC of providing accurate estimates $\hat{I}_{m, k}$. While in principle Eq. 11 could apply to any collection of variational posterior mixtures, without an efficient way of calculating each $I_{k}$ (Eq. 8), optimisation of the stacked ELBO would require many likelihood evaluations, which would be prohibitive for problems with expensive, black-box likelihoods.

In the following, we demonstrate the efficacy of this approach.

## 4 Experiments

In this section, we describe our experimental procedure (Section 4.1), baselines (Section 4.2), evaluation metrics (Section 4.3), and results for synthetic (Section 4.4) and real-world problems (Section 4.5). Finally, we briefly discuss computational costs in Section 4.6. Additional results are reported in the appendix, including additional experiments (Appendix A.3), more extensive tables of results (Appendix A.4), and example visualisations of posterior approximations (Appendix A.5).

### 4.1 Procedure

We first tested our method on two synthetic problems, designed to be particularly challenging for VBMC, and then on two real-world datasets and models (see below for full descriptions). We considered both noiseless problems (exact estimation) and noisy problems where Gaussian noise with $\sigma=3$ is applied to each log-likelihood measurement, emulating what practitioners might find when estimating the likelihood via simulation (van Opheusden et al., 2020). For each benchmark, we considered 100 VBMC runs obtained with the pyvbmc Python package (Huggins et al., 2023) that satisfied the following conditions:

1. the algorithm had converged (as assessed by the pyvbmc software);
2. $\max _{1 \leq k \leq K}\left(J_{k, k}\right)<5$, so all the estimates $\hat{I}_{k}$ had an associated variance lower than 5 .

The reason for the latter is that S-VBMC's efficacy is strongly dependent on accurate VBMC estimates of the real $I_{k}$ from Bayesian quadrature (see Sections 2.3 and 3.2), and we sought to filter out "poorly converged" runs where this might not be the case. The results of this filtering procedure can be found in Appendix A.4.1. We performed all VBMC runs using the default settings (which always resulted in posteriors with 50 components, $K_{m}=50 \forall m \in\{1, \ldots, M\}$ ) and random uniform initialisation within plausible parameter bounds (Acerbi, 2018). To investigate the effect of combining a different number of posteriors, we adopted a bootstrapping approach: from these 100 runs, we randomly sampled and stacked

---

#### Page 8

with S-VBMC a varying number of runs (between 2 and 40) twenty times each, and computed the median with corresponding $95 \%$ confidence interval (computed from 10000 bootstrap resamples) for all metrics, described below. For all benchmark problems, the entropy is approximated as in Eq. 12 with $S=20$ during optimisation, and a final estimation (reported in all tables and figures) is performed with $S=100$ after convergence. The ELBO is optimised using Adam (Kingma \& Ba, 2014) with learning rate set to 0.1 .
All the experiments presented in this work were run on an AMD EPYC 7452 Processor with 16GB of RAM.

# 4.2 Baseline methods

Black-box variational inference. We used black-box variational inference (BBVI; Ranganath et al., 2014) as a baseline for all our benchmark problems.

Our implementation follows Li et al. (2025). For gradient-free black-box models, we cannot use the reparameterisation trick (Kingma \& Welling, 2013) to estimate ELBO gradients. Instead, we employ the score function estimator (REINFORCE; Ranganath et al., 2014) with control variates to reduce gradient variance.
The variational posterior is parameterised as a mixture of Gaussians (MoG) with either $K=50$ or $K=500$ components, matching the form used in VBMC. We initialise the component means near the origin by adding Gaussian noise $(\sigma=0.1)$ and set all component variances to 0.01 . We optimise the ELBO using Adam (Kingma \& Ba, 2014) with stochastic gradients, performing a grid search over Monte Carlo sample sizes $\{1,10,100\}$ and learning rates $\{0.01,0.001\}$. We select the best hyperparameters based on the estimated ELBO.

For a fair comparison with S-VBMC, we set the target evaluation budget for a BBVI run to $2000(D+2)$ and $3000(D+2)$ evaluations for noiseless and noisy problems, respectively, matching the maximum evaluations used by 40 VBMC runs in total.

Naive Stacking. As a further baseline, we implemented a "naive" version of our stacking approach, consisting of a simple averaging of the individual VBMC posteriors. For this, we rewrite the stacked posterior as

$$
q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \tilde{\omega}_{m} q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

and simply set

$$
\tilde{\omega}_{1}=\tilde{\omega}_{2}=\ldots=\tilde{\omega}_{m}=1 / M
$$

We call this approach "Naive Stacking" (NS).
As for S-VBMC, we ran NS by randomly sampling a varying number of runs (between 2 and 40) twenty times each, and then computed the median with corresponding $95 \%$ confidence intervals for all metrics.

### 4.3 Metrics

Following Acerbi (2020); Li et al. (2025), we evaluate our method using three metrics:

1. The absolute difference between true and estimated log marginal likelihood ( $\Delta \mathrm{LML}$ ), where values $<1$ are considered negligible for model selection (Burnham \& Anderson, 2003).
2. The mean marginal total variation distance (MMTV), which measures the average (lack of) overlap between true and approximate posterior marginals across dimensions:

$$
\operatorname{MMTV}(p, q)=\frac{1}{2 D} \sum_{d=1}^{D} \int_{-\infty}^{\infty}\left|p_{d}\left(x_{d}\right)-q_{d}\left(x_{d}\right)\right| d x_{d}
$$

where $p_{d}$ and $q_{d}$ denote the marginal distributions along the $d$-th dimension.

---

#### Page 9

3. The "Gaussianised" symmetrised KL divergence (GsKL), which evaluates differences in means and covariances between the approximate and true posterior:

$$
\operatorname{GsKL}(p, q)=\frac{1}{2 D}\left[D_{\mathrm{KL}}(\mathcal{N}[p] \| \mathcal{N}[q])+D_{\mathrm{KL}}(\mathcal{N}[q] \| \mathcal{N}[p])\right]
$$

where $\mathcal{N}[p]$ denotes a Gaussian with the same mean and covariance as $p$.
We consider MMTV $<0.2$ and $\mathrm{GsKL}<\frac{1}{8}$ as target thresholds for reasonable posterior approximation ( Li et al., 2025). Ground-truth estimates of the log marginal likelihood and posterior distributions are obtained through numerical integration, extensive MCMC sampling, or analytical methods as appropriate for each problem.

# 4.4 Synthetic problems

GMM target. Our synthetic GMM target consists of a mixture of 20 bivariate Gaussian components arranged in four distinct clusters. The cluster centroids were positioned at $(-8,-8),(-7,7),(6,-6)$ and $(5,5)$. Around each centroid, we placed five Gaussian components with means drawn from $\mathcal{N}\left(\boldsymbol{\mu}_{c}, \mathbf{I}\right)$, where $\boldsymbol{\mu}_{c}$ is the respective cluster centroid and $\mathbf{I}$ is the $2 \times 2$ identity matrix. Each component was assigned unit marginal variances and a correlation coefficient of $\pm 0.5$ (randomly selected with equal probability). This configuration produces an irregular mixture structure that requires a substantial number of components to approximate accurately. All components were assigned equal mixing weights. The resulting distribution is illustrated in Figure 2 (top panels).

> **Image description.** This image displays a grid of six 2D scatter plots, arranged in two rows and three columns. Each plot shares a dark purple background and common axis labels and ranges. The x-axis is labeled "$\theta_1$" and the y-axis is labeled "$\theta_2$", both ranging from -10 to 10 with major ticks at -10, 0, and 10. The plots illustrate the overlap between ground truth densities (depicted by color gradients and contours) and samples from posterior approximations (depicted by red points) for two different synthetic benchmarks.
>
> The top row of panels pertains to a "GMM target" distribution:
>
> - **Top-left panel (VBMC)**: The background shows four distinct, irregularly shaped density regions, colored with gradients from blue to green to yellow, with white contour lines. These regions are located roughly in the four quadrants of the plot: top-left, top-right, bottom-left, and bottom-right. The red points, representing the posterior approximation, are densely clustered in a single, irregular shape primarily within the bottom-right quadrant, showing limited overlap with the ground truth density.
> - **Top-middle panel (S-VBMC (5 posteriors))**: The background density regions are similar to the top-left panel. The red points are now distributed across all four quadrants, forming four distinct clusters. Each cluster of red points largely overlaps with one of the background density regions, indicating improved approximation compared to the VBMC method.
> - **Top-right panel (S-VBMC (20 posteriors))**: The background density regions remain consistent. The red points are again distributed across all four quadrants, forming four distinct clusters. The overlap between the red points and the background density regions appears even more complete and precise than in the S-VBMC (5 posteriors) panel, suggesting further improved approximation with more posteriors.
>
> The bottom row of panels pertains to a "ring target" distribution:
>
> - **Bottom-left panel (VBMC)**: The background features a single, thin, light grey-blue circular contour centered at (0,0), with a radius of approximately 7-8 units. The red points, representing the posterior approximation, form a dense arc along the bottom-left portion of this circle, covering roughly one-quarter to one-third of the circumference, indicating a partial approximation of the ring.
> - **Bottom-middle panel (S-VBMC (5 posteriors))**: The background shows the same light grey-blue circular contour. The red points now form a more extensive arc along the circle, covering more than half of its circumference, but still leaving a noticeable gap in the approximation.
> - **Bottom-right panel (S-VBMC (20 posteriors))**: The background again shows the same light grey-blue circular contour. The red points now form a nearly complete circle, almost entirely overlapping the background contour, with only minimal or no visible gaps, demonstrating a highly accurate approximation of the ring target.

Figure 2: Examples of overlap between the ground truth and the posterior when combining different numbers of VBMC runs on the GMM (top panels) and ring (bottom panels) synthetic benchmarks. The red points indicate samples from the posterior approximation, with the target density depicted with colour gradients in the background.

In our experiments, we used both a noiseless version of this target (i.e., exact target evaluation) and a noisy one, where we applied i.i.d. Gaussian noise $(\sigma=3)$ to each log-likelihood evaluation.

Ring target. Our second synthetic target is a ring-shaped distribution defined by the probability density function

$$
p_{\text {ring }}\left(\theta_{1}, \theta_{2}\right) \propto \exp \left(-\frac{(r-R)^{2}}{2 \sigma^{2}}\right)
$$

---

#### Page 10

where $r=\sqrt{\left(\theta_{1}-c_{1}\right)^{2}+\left(\theta_{2}-c_{2}\right)^{2}}$ represents the radial distance from centre $\left(c_{1}, c_{2}\right), R$ is the ring radius, and $\sigma$ controls the width of the annulus. We set $R=8, \sigma=0.1$, and centred the ring at $\left(c_{1}, c_{2}\right)=(1,-2)$. The small value of $\sigma$ produces a narrow annular distribution that challenges VBMC's exploration capabilities. The resulting distribution is shown in Figure 2 (bottom panels).
As with the GMM target, we used both a noiseless version of this benchmark and a noisy one $(\sigma=3)$ in our experiments.

> **Image description.** This image is a multi-panel figure composed of 16 line graphs arranged in a 4x4 grid. Each row represents a different synthetic problem, and each column displays a different performance metric. The graphs plot metrics as a function of the "N. of runs" (number of VBMC runs stacked), with median values and 95% confidence intervals indicated by error bars.
>
> The four rows are labeled:
>
> - **(a) GMM (D = 2, noiseless)**
> - **(b) GMM (D = 2, σ = 3)**
> - **(c) Ring (D = 2, noiseless)**
> - **(d) Ring (D = 2, σ = 3)**
>
> The four columns represent the following metrics:
>
> - **ELBO** (Estimated Lower Bound)
> - **Δ LML** (Difference in Log Marginal Likelihood)
> - **MMTV** (Maximum Mean Total Variation)
> - **GsKL** (Generalized symmetric Kullback-Leibler divergence)
>
> All x-axes are labeled "N. of runs" and show tick marks at 1, 4, 8, 16, 24, 32, and 40. The y-axes for Δ LML and GsKL are on a logarithmic scale.
>
> Four different methods are plotted:
>
> - **S-VBMC** (blue line with circular markers)
> - **VBMC** (red line with upward-pointing triangular markers)
> - **NS** (grey line with downward-pointing triangular markers)
> - **BBVI** (a single green data point with a square marker, consistently plotted at N. of runs = 40)
>
> In the ELBO panels, a solid black horizontal line represents the groundtruth LML. In the Δ LML, MMTV, and GsKL panels, a dashed black horizontal line indicates a desirable threshold, with good performance being below this line.
>
> **Detailed observations for each row:**
>
> - **Row (a) GMM (D = 2, noiseless):**
>
>   - **ELBO:** S-VBMC (blue) and NS (grey) rapidly increase and converge towards the solid black groundtruth LML line, with S-VBMC slightly outperforming NS at higher run counts. VBMC (red) starts lower and shows slower convergence. BBVI (green) is very close to the groundtruth.
>   - **Δ LML:** All methods start with high values. S-VBMC (blue) shows a steep decrease, consistently dropping below the dashed threshold line for N. of runs greater than 8. NS (grey) also decreases but remains above S-VBMC. VBMC (red) stays high. BBVI (green) is very low, well below the threshold.
>   - **MMTV:** Similar to Δ LML, S-VBMC (blue) decreases sharply and goes below the dashed threshold line. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is very low.
>   - **GsKL:** Similar to Δ LML, S-VBMC (blue) decreases significantly, crossing the dashed threshold line. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is very low.
>
> - **Row (b) GMM (D = 2, σ = 3) (noisy):**
>
>   - **ELBO:** S-VBMC (blue) increases and plateaus slightly above the solid black groundtruth LML line. NS (grey) increases and converges closer to the groundtruth. VBMC (red) shows slower convergence. BBVI (green) is close to the groundtruth.
>   - **Δ LML:** S-VBMC (blue) decreases significantly, crossing the dashed threshold, but its values are generally higher than in the noiseless case. NS (grey) decreases but remains above S-VBMC. VBMC (red) stays high. BBVI (green) is low.
>   - **MMTV:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is low.
>   - **GsKL:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is low.
>
> - **Row (c) Ring (D = 2, noiseless):**
>
>   - **ELBO:** S-VBMC (blue) and NS (grey) rapidly increase and converge towards the solid black groundtruth LML line, with S-VBMC slightly outperforming NS at higher run counts. VBMC (red) starts lower and shows slower convergence. BBVI (green) is significantly lower than the groundtruth.
>   - **Δ LML:** S-VBMC (blue) decreases significantly, going below the dashed threshold. NS (grey) also decreases but stays above S-VBMC. VBMC (red) stays high. BBVI (green) is very high, well above the threshold.
>   - **MMTV:** S-VBMC (blue) decreases sharply and goes below the dashed threshold. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is high.
>   - **GsKL:** S-VBMC (blue) decreases significantly, crossing the dashed threshold. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is high.
>
> - **Row (d) Ring (D = 2, σ = 3) (noisy):**
>   - **ELBO:** S-VBMC (blue) increases and plateaus slightly above the solid black groundtruth LML line. NS (grey) increases and converges closer to the groundtruth. VBMC (red) shows slower convergence. BBVI (green) is significantly lower than the groundtruth.
>   - **Δ LML:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but remains above S-VBMC. VBMC (red) stays high. BBVI (green) is very high.
>   - **MMTV:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is high.
>   - **GsKL:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is high.
>
> In summary, S-VBMC (blue) generally shows the best performance among the iterative methods, consistently reaching or surpassing the desirable thresholds for Δ LML, MMTV, and GsKL, and converging to the groundtruth LML for ELBO. BBVI (green) performs very well for GMM problems but struggles with Ring problems. VBMC (red) consistently shows the highest values for Δ LML, MMTV, and GsKL, indicating poorer performance compared to S-VBMC and NS. NS (grey) generally performs better than VBMC but not as well as S-VBMC. Error bars are visible for all data points, indicating the variability of the metrics.

Figure 3: Synthetic problems. Metrics plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval). Metrics are plotted for S-VBMC (blue), VBMC (red), and NS (grey). The best BBVI results are shown in green. The black horizontal line in the ELBO panels represents the groundtruth LML, while the dashed lines on $\Delta \mathrm{LML}$, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3)

Results. Results in Figure 3 and Table A. 3 show that merging more posteriors leads to a steady improvement in the GsKL and MMTV metrics, which measure the quality of the posterior approximation. Remarkably, S-VBMC proves to be robust to noisy targets, with minor differences between noiseless and noisy settings. In all our synthetic benchmarks (both in noiseless and noisy settings), we observe that the values of the GsKL and MMTV metrics, which directly compare to the ground-truth target densities, reach good values - and start to plateau, or at least to improve at a much slower pace - by the time 10 VBMC

---

#### Page 11

posteriors are stacked. This suggests that a value of $M=10$ (or even $M \approx 6$ for noiseless problems, where metrics tend to converge faster) would be sufficient to obtain accurate stacked posteriors.

S-VBMC outperforms the BBVI baseline on the ring-shaped synthetic target. The BBVI baseline performs well and is only marginally worse compared to S-VBMC only on the GMM problem, where it effectively managed to capture the four clusters (see Figure 2 for a visualisation). While NS exhibits a similar improvement pattern with increasing values of $M$, S-VBMC consistently outperforms it, with differences being larger in noiseless settings.

As expected by design, individual VBMC runs tended to explore the two synthetic target distributions only partially, leading to poor performance. Still, the random initialisations allowed different runs to discover different portions of the posterior, allowing the merging process to cover the whole target (see Figure 2).

Finally, we observe that, in noisy settings, while the ELBO keeps increasing, the $\Delta$ LML error (difference between ELBO and true log marginal likelihood) initially decreases but then increases again as further components are added, a point which we will discuss later.

# 4.5 Real-world problems.

Neuronal model. Our first real-world problem involved fitting five biophysical parameters of a detailed compartmental model of a hippocampal CA1 pyramidal neuron. The model was constructed based on experimental data comprising a three-dimensional morphological reconstruction and electrophysiological recordings of neuronal responses to current injections. The deterministic neuronal responses were simulated using the NEURON simulation environment (Hines \& Carnevale, 1997; Hines et al., 2009), applying current step inputs that matched the experimental protocol. The model's parameters characterise key biophysical properties: intracellular axial resistivity $\left(\theta_{1}\right)$, leak current reversal potential $\left(\theta_{2}\right)$, somatic leak conductance $\left(\theta_{3}\right)$, dendritic conductance gradient $\left(\theta_{4}\right.$, per $\mu \mathrm{m}$ ), and a dendritic surface scaling factor $\left(\theta_{5}\right)$. Based on independent measurements of membrane potential fluctuations, observation noise was modelled as a stationary Gaussian process with zero mean and a covariance function estimated from the data. The covariance structure was captured by the product of a cosine and an exponentially decaying function. For a similar approach applied to cerebellar Golgi cells, see Szoboszlay et al. (2016).

This model allowed exact log-likelihood evaluations, and no noise was added.
Multisensory causal inference model. Perceptual causal inference involves determining whether multiple sensory stimuli originate from a common source, a problem of particular interest in computational cognitive neuroscience (Körding et al., 2007). Our second real-world problem involved fitting a visuo-vestibular causal inference model to empirical data from a representative participant (S1 from Acerbi et al., 2018). In each trial of the modelled experiment, participants seated in a moving chair reported whether they perceived their movement direction $\left(s_{\text {vest }}\right)$ as congruent with an experimentally-manipulated looming visual field $\left(s_{\text {vis }}\right)$. The model assumes participants receive noisy sensory measurements, with vestibular information $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right)$ and visual information $z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}^{2}$ and $\sigma_{\text {vis }}^{2}$ represent sensory noise variances. The visual coherence level $c$ was experimentally manipulated across three levels $\left(c_{\text {low }}, c_{\text {med }}, c_{\text {high }}\right)$. The model assumes participants judge the stimuli as having a common cause when the absolute difference between sensory measurements falls below a threshold $\kappa$, with a lapse rate $\lambda$ accounting for random responses. The model parameters $\boldsymbol{\theta}$ comprise the visual noise parameters $\sigma_{\text {vis }}\left(c_{\text {low }}\right), \sigma_{\text {vis }}\left(c_{\text {med }}\right)$, $\sigma_{\text {vis }}\left(c_{\text {high }}\right)$, vestibular noise $\sigma_{\text {vest }}$, lapse rate $\lambda$, and decision threshold $\kappa$ (Acerbi et al., 2018).

We fitted this model assuming log-likelihood measurement noise ( $\sigma=3$, which we applied to each loglikelihood evaluation).

Results. The results in Figure 4 and Table A. 4 confirm our earlier findings of improvements across the posterior metrics. We also find that S-VBMC is robust to noisy targets for real data, with performance that improves with the increasing number of stacked runs in the multisensory model problem.

Similar to what we observed for the synthetic problems, we find that stacking $\approx 10$ VBMC posteriors seems to be sufficient to vastly improve posterior quality (as indexed by the GsKL and MMTV metrics), compared to

---

#### Page 12

a single VBMC run $(M=1)$, with relatively small additional improvements for $M>10$. The only exception to this is the GsKL metric in the neuronal model, where the lower confidence intervals start to go below the desirable threshold (see Section 4.3) only if at least 18 posteriors are merged, and the full confidence interval never fully stabilises below such a threshold. However, this was our most challenging benchmark problem, as evidenced by the very poor performance obtained by both single-run VBMC and BBVI. Thus, while S-VBMC here does not achieve a posterior approximation fully on par with gold-standard MCMC (used to obtain the ground-truth posterior in this problem), the stacking process still vastly improves posterior quality at a much lower computational cost.

Finally, similarly to the synthetic problems, we see that S-VBMC performs consistently better than standard VBMC and BBVI across all metrics in both scenarios. Despite similar improvement patterns with increasing values of $M$, as observed above, S-VBMC consistently outperforms NS, with starker differences in the noisy setting (i.e., the multisensory model).

> **Image description.** A multi-panel figure composed of eight line graphs arranged in two rows and four columns, displaying various metrics as a function of the number of runs. A common legend is positioned at the bottom center of the figure.
>
> The figure is divided into two main sections, labeled (a) and (b), each representing a different model:
>
> **Panel (a): Neuronal model ($D=5$, noiseless)**
> This top row contains four sub-plots, all sharing the same x-axis labeled "N. of runs" with tick values 1, 4, 8, 16, 24, 32, and 40.
>
> - **ELBO (leftmost plot):**
>
>   - Y-axis: "ELBO" ranging from -7500 to -7450.
>   - A solid black horizontal line is present at approximately -7450, representing the groundtruth LML.
>   - S-VBMC (blue line with circular markers and error bars) starts around -7455 at N=1, rapidly increases, and stabilizes just below the black line, indicating convergence towards the groundtruth.
>   - VBMC (single red circular marker with error bar) is shown at N=1, around -7455.
>   - NS (grey line with square markers and error bars) generally follows the S-VBMC trend but slightly below it, also stabilizing near the groundtruth.
>   - BBVI (single green square marker with a large error bar) is plotted at N=40, significantly lower around -7495, with its error bar truncated at the bottom of the plot.
>
> - **Δ LML (second from left):**
>
>   - Y-axis: "Δ LML" on a logarithmic scale from 0.1 to 100.
>   - A dashed black horizontal line is present at 1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 3 at N=1, decreases sharply, and stabilizes below the dashed threshold, reaching values around 0.5 for higher N.
>   - VBMC (red) is at N=1, around 3.
>   - NS (grey) starts around 2.5 at N=1, decreases, and stabilizes just above or around the threshold of 1.
>   - BBVI (green) is at N=40, with a high value around 50, well above the threshold.
>
> - **MMTV (third from left):**
>
>   - Y-axis: "MMTV" on a linear scale from 0.0 to 0.6.
>   - A dashed black horizontal line is present at 0.2, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.25 at N=1, decreases rapidly, and stabilizes significantly below the threshold, around 0.08.
>   - VBMC (red) is at N=1, around 0.3.
>   - NS (grey) starts around 0.2 at N=1, decreases, and stabilizes around 0.15, still below the threshold but higher than S-VBMC.
>   - BBVI (green) is at N=40, with a high value around 0.55, well above the threshold.
>
> - **GsKL (rightmost plot):**
>   - Y-axis: "GsKL" on a logarithmic scale from 0.1 to 100.
>   - A dashed black horizontal line is present at 0.1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 20 at N=1, decreases sharply, and stabilizes at or slightly below the threshold, around 0.1.
>   - VBMC (red) is at N=1, around 100.
>   - NS (grey) starts around 10 at N=1, decreases, and stabilizes around 0.5, above the threshold.
>   - BBVI (green) is at N=40, with a high value around 20, well above the threshold.
>
> **Panel (b): Multisensory ($D=6$, $\sigma=3$)**
> This bottom row also contains four sub-plots, sharing the same x-axis labeled "N. of runs" with identical tick values as panel (a).
>
> - **ELBO (leftmost plot):**
>
>   - Y-axis: "ELBO" ranging from -450 to -444.
>   - A solid black horizontal line is present at approximately -444, representing the groundtruth LML.
>   - S-VBMC (blue) starts around -445 at N=1, increases, and stabilizes just below the black line, indicating convergence.
>   - VBMC (red) is at N=1, around -445.
>   - NS (grey) generally follows the S-VBMC trend but stabilizes slightly lower, around -444.5.
>   - BBVI (green) is at N=40, significantly lower around -448.5, with a large error bar.
>
> - **Δ LML (second from left):**
>
>   - Y-axis: "Δ LML" on a logarithmic scale from 0.1 to 100.
>   - A dashed black horizontal line is present at 1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.5 at N=1, increases, and stabilizes above the threshold, around 1.5.
>   - VBMC (red) is at N=1, around 0.5.
>   - NS (grey) starts around 0.7 at N=1, increases, and stabilizes below the threshold, around 0.8.
>   - BBVI (green) is at N=40, with a value around 2, above the threshold.
>
> - **MMTV (third from left):**
>
>   - Y-axis: "MMTV" on a linear scale from 0.00 to 0.20.
>   - A dashed black horizontal line is present at 0.2, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.15 at N=1, decreases rapidly, and stabilizes significantly below the threshold, around 0.08.
>   - VBMC (red) is at N=1, around 0.18.
>   - NS (grey) starts around 0.12 at N=1, decreases, and stabilizes around 0.1, below the threshold but higher than S-VBMC.
>   - BBVI (green) is at N=40, with a value around 0.11, below the threshold.
>
> - **GsKL (rightmost plot):**
>   - Y-axis: "GsKL" on a logarithmic scale from 0.01 to 1.
>   - A dashed black horizontal line is present at 0.1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.15 at N=1, decreases sharply, and stabilizes significantly below the threshold, around 0.03.
>   - VBMC (red) is at N=1, around 0.2.
>   - NS (grey) starts around 0.1 at N=1, decreases, and stabilizes around 0.06, below the threshold but higher than S-VBMC.
>   - BBVI (green) is at N=40, with a value around 0.15, above the threshold.
>
> **Legend (bottom center):**
> The legend defines the visual representations for four different methods, each with an associated colored square and error bar icon:
>
> - BBVI: Green square with error bar
> - NS: Grey square with error bar
> - VBMC: Red square with error bar
> - S-VBMC: Blue square with error bar
>
> All data points for S-VBMC and NS are connected by lines, indicating their performance across varying numbers of runs. VBMC and BBVI are shown as single points with error bars, representing their performance at specific run counts (N=1 for VBMC, N=40 for BBVI).

Figure 4: Real-world problems. Metrics plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval). Metrics are plotted for S-VBMC (blue), VBMC (red), and NS (grey). The best BBVI results are shown in green. The black horizontal line in the ELBO panels represents the groundtruth LML, while the dashed lines on $\Delta$ LML, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3). The BBVI error bar in the plot displaying the ELBO in the neuronal model (top left) is truncated for clarity.

# 4.6 Computational overhead

Here we present details about the additional computational cost (quantified as compute time) introduced by S-VBMC on top of VBMC. Additional runtime analyses can be found in Appendix A.3.2.

Figure 5 illustrates how S-VBMC introduces a relatively small computational overhead, even when comparing the post-process cost of S-VBMC with the average cost of one VBMC run, under the idealised condition where the $M$ VBMC runs happen all in parallel. ${ }^{1}$ In particular, running our algorithm with $M \approx 10$ - which vastly improves the resulting posterior, as shown above and in Appendices A.4.2 and A. 5 - adds a small amount of post-processing time to VBMC for all our benchmark problems ( $\approx 5$-15\% overhead).

Put together, our results confirm that S-VBMC yields high returns in terms of inference performance at a very marginal cost in terms of compute time.

[^0]
[^0]: ${ }^{1}$ In practice, completing $M$ VBMC runs will be more expensive due to additional parallelisation costs, making S-VBMC's relative overhead even smaller than what we report here.

---

#### Page 13

> **Image description.** A multi-panel plot, arranged in two rows and three columns, displays six line graphs showing "Compute time (s)" on the y-axis against "N. of runs" on the x-axis. Each subplot represents a different benchmark problem, indicated by its title. All subplots share common axis labels and a consistent visual style, including a light gray grid.
>
> **Common Elements Across Subplots:**
>
> - **X-axis**: Labeled "N. of runs", ranging from 0 to 40. Major tick marks are present at 1, 4, 8, 16, 24, 32, and 40.
> - **Y-axis**: Labeled "Compute time (s)". The range and major tick marks vary for each subplot.
> - **Data Representation**: Each subplot contains two sets of data points:
>   - A single red circular data point with a vertical error bar, representing "VBMC". This point is consistently plotted at N. of runs = 1.
>   - A series of blue circular data points, connected by implied lines, each with a vertical error bar, representing "S-VBMC (post-process)". These points show an increasing trend as "N. of runs" increases.
>
> **Legend:**
> A legend is positioned below the bottom-center subplot:
>
> - <span style="color:red">**&#x25CF;**</span> <span style="color:red">**I**</span> VBMC
> - <span style="color:blue">**&#x25CF;**</span> <span style="color:blue">**I**</span> S-VBMC (post-process)
>
> **Individual Subplot Descriptions:**
>
> 1.  **Top-Left Panel:**
>
>     - **Title**: GMM ($D=2$, noiseless)
>     - **Y-axis**: Ranges from 0 to 150, with major ticks at 0, 50, 100, 150.
>     - **VBMC (red)**: Approximately (1, 85) with an error bar spanning roughly 70 to 125.
>     - **S-VBMC (blue)**: Starts near (1, 5) and increases in a curvilinear fashion to approximately (40, 150). Error bars generally widen with increasing N. of runs.
>
> 2.  **Top-Middle Panel:**
>
>     - **Title**: Ring ($D=2$, noiseless)
>     - **Y-axis**: Ranges from 0 to 200, with major ticks at 0, 100, 200.
>     - **VBMC (red)**: Approximately (1, 190) with an error bar spanning roughly 140 to 240.
>     - **S-VBMC (blue)**: Starts near (1, 5) and increases in a curvilinear fashion to approximately (40, 200).
>
> 3.  **Top-Right Panel:**
>
>     - **Title**: Neuronal model ($D=5$, noiseless)
>     - **Y-axis**: Ranges from 0 to 750, with major ticks at 0, 250, 500, 750.
>     - **VBMC (red)**: Approximately (1, 650) with an error bar spanning roughly 500 to 800.
>     - **S-VBMC (blue)**: Starts near (1, 10) and increases in a curvilinear fashion to approximately (40, 800).
>
> 4.  **Bottom-Left Panel:**
>
>     - **Title**: GMM ($D=2$, $\sigma=3$)
>     - **Y-axis**: Ranges from 0 to 400, with major ticks at 0, 100, 200, 300, 400.
>     - **VBMC (red)**: Approximately (1, 240) with an error bar spanning roughly 170 to 330.
>     - **S-VBMC (blue)**: Starts near (1, 10) and increases in a curvilinear fashion to approximately (40, 450).
>
> 5.  **Bottom-Middle Panel:**
>
>     - **Title**: Ring ($D=2$, $\sigma=3$)
>     - **Y-axis**: Ranges from 0 to 800, with major ticks at 0, 200, 400, 600, 800.
>     - **VBMC (red)**: Approximately (1, 540) with an error bar spanning roughly 400 to 780.
>     - **S-VBMC (blue)**: Starts near (1, 20) and increases in a curvilinear fashion to approximately (40, 500).
>
> 6.  **Bottom-Right Panel:**
>     - **Title**: Multisensory ($D=6$, $\sigma=3$)
>     - **Y-axis**: Ranges from 0 to 2000, with major ticks at 0, 1000, 2000.
>     - **VBMC (red)**: Approximately (1, 850) with an error bar spanning roughly 650 to 1100.
>     - **S-VBMC (blue)**: Starts near (1, 20) and increases in a curvilinear fashion to approximately (40, 2300).
>
> The plots visually compare the compute time of VBMC (a single run) against the post-processing computational overhead of S-VBMC as more runs are stacked, across different models and conditions.

Figure 5: Compute time of a single VBMC run (red) and post-processing time only (i.e., computational overhead) of S-VBMC (blue) plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples). Each subplot represents a different benchmark problem. The values plotted here correspond to the actual computation times of the experiments described in Sections 4.4 and 4.5.

# 5 ELBO estimation bias

As briefly mentioned in Section 4.4, in our results we observe that, in noisy settings, while the ELBO keeps increasing as more VBMC runs are stacked, the $\Delta \mathrm{LML}$ error also increases, after an initial decrease (see the second column of Figures 3 and 4). This apparently odd result can be explained by the presence of a positive bias build-up in the estimated ELBO. This bias is visible in Figure 3 (b) and (d), first column, and Figure 4 (b), first column, in that the estimated ELBO from S-VBMC on these problems slightly "overshoots" the ground truth LML (the S-VBMC estimates, blue dots, end above the black horizontal line). As discussed in Section 2.2, the ELBO is always lower than the LML, with equality when the approximate posterior perfectly corresponds to the true posterior. However, in these results, the estimated ELBO grows larger than the ground-truth LML. As this cannot be true, there must be a positive bias in the ELBO estimate. As one can see in the aforementioned figures (first column), this bias builds up as more and more VBMC runs are stacked. What characterises these problems is the fact that they use a stochastic estimator for the likelihood (as opposed to exact likelihood evaluations used in the other problems, where the estimate does not overshoot).

While this bias surprisingly does not affect other posterior quality metrics, which keep improving (or plateau) with increasing $M$, it might constitute an issue when using $\mathrm{ELBO}_{\text {stacked }}$ for model comparison. In this section, we provide a simplified model for how the bias might statistically arise in terms of the "winner's curse" and then provide an effective heuristic to counteract the bias.

### 5.1 Origin of bias

Here we analyse the ELBO overestimation observed in our results through a simplified example that illustrates one potential mechanism for this bias. In short, we suggest this occurs because all $I_{m, k}$ are noisy estimates of the true expected log-joint contributions, causing S-VBMC to overweigh the most overestimated

---

#### Page 14

mixture components - an effect that increases with the number of components $M$. While other factors may contribute, this analysis provides insight into why the bias tends to increase with the number of merged VBMC runs.

For the sake of argument, consider $M$ VBMC runs that return identical posteriors $q_{\boldsymbol{\phi}_{1}}(\boldsymbol{\theta})=\ldots=q_{\boldsymbol{\phi}_{M}}(\boldsymbol{\theta})$, each with a single component. The stacked posterior takes the form:

$$
q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \hat{w}_{m} q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

For each single-component posterior, the expected log-joint is approximated as

$$
I_{m}=\mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})\right] \approx \mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[f_{m}(\boldsymbol{\theta})\right]
$$

where $f_{m}(\boldsymbol{\theta})$ is the surrogate log-joint from the $m$-th VBMC run. Since all posteriors share identical parameters, their entropies are equal:

$$
\mathcal{H}\left[q_{\boldsymbol{\phi}_{1}}(\boldsymbol{\theta})\right]=\mathcal{H}\left[q_{\boldsymbol{\phi}_{2}}(\boldsymbol{\theta})\right]=\ldots=\mathcal{H}\left[q_{\boldsymbol{\phi}_{M}}(\boldsymbol{\theta})\right]
$$

The stacked posterior is thus a mixture of identical components with different associated values $I_{m}$. The optimal mixture weights $\hat{\mathbf{w}}$ depend solely on the noisy estimates of $I_{m}$ :

$$
\hat{I}_{m}=\mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[f_{m}(\boldsymbol{\theta})\right]=\mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})\right]+\epsilon_{m}
$$

where $\epsilon_{m} \sim \mathcal{N}\left(0, J_{m}\right)$ represents estimation noise with variance $J_{m}$. Since all posteriors are identical and derived from the same data and model, differences in expected log-joint estimates arise purely from noise deriving from the Gaussian process surrogates $f_{m}$.

Given that entropy remains constant under merging, in this scenario optimising $\mathrm{ELBO}_{\text {stacked }}$ reduces to selecting the posterior with the highest expected log-joint estimate. If we denote $\hat{I}_{\max }=\max _{m} \hat{I}_{m}$, the optimal ELBO becomes

$$
\mathrm{ELBO}_{\text {stacked }}^{*}=\hat{I}_{\max }+\mathcal{H}\left[q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

Since the true expected log-joint is identical across posteriors, the optimisation selects the most overestimated value. The magnitude of this overestimation increases with both $M$ and the observation noise for $f_{m}$, introducing a positive bias in $\mathrm{ELBO}_{\text {stacked }}^{*}$ that grows with the number of stacked runs and is more substantial for surrogates obtained from noisy log-likelihood observations.

While this simplified scenario does not capture the complexity of practical applications - where posteriors have multiple, non-overlapping components - it illustrates a fundamental issue: if we model each $\hat{I}_{m, k}$ as the sum of the true $I_{m, k}$ and noise, the merging process will favour overestimated components, biasing the final $\mathrm{ELBO}_{\text {stacked }}$ estimate upward.

This hypothesis is substantiated by our results, as we only observe a noticeable bias in problems with noisy targets, where levels of noise in the VBMC estimation of $I_{m, k}$ are non-negligible (note that VBMC outputs an estimate of such noise, see Section 2).

# 5.2 Debiasing heuristic

We propose here a heuristic approach to counteract the bias in the ELBO estimate.
First, as a baseline we can estimate the per-component ground-truth expected log-joint $I_{m, k}$, and the full expected log-joint summed over components, using Monte Carlo sampling on the true log-joint instead of the VBMC estimates (see Eqs. 8 and 11). This estimate is unbiased and does not increase with $M$, as opposed to the one estimated by S-VBMC, as shown for all noisy benchmarks in Figure 6. However, this requires numerous additional evaluations of the model's log-likelihood, which is assumed to be expensive, so it is not in general a viable solution.

---

#### Page 15

One potentially effective heuristic to counteract the bias - or at least prevent it from increasing with $M$ would be to cap its value post-hoc to some reasonable quantity. This heuristic has the advantage of being straightforward and inexpensive to implement, and, crucially, of not involving any tweaks to the ELBO optimisation process, thus not requiring any additional hyperparameters.

> **Image description.** A multi-panel line graph titled "Expected log-joint (S-VBMC)" displays three individual plots arranged horizontally, each comparing two data series with error bars across varying numbers of runs.
>
> Each of the three panels shares a common x-axis labeled "N. of runs", with tick marks at 2, 4, 8, 16, 24, 32, and 40. The y-axis for all panels is labeled "Expected log-joint", but the numerical range differs for each plot. Two data series are plotted in each panel: one represented by blue circular markers with blue error bars, and the other by black circular markers with black error bars.
>
> The panels are as follows:
>
> 1.  **Left Panel: "GMM, σ = 3"**
>
>     - The y-axis ranges from -1.6 to -0.8, with major ticks at -1.6, -1.4, -1.2, -1.0, and -0.8.
>     - The blue data series starts around -1.5 and shows a clear increasing trend, rising to approximately -0.85 by 40 runs. The error bars are visibly larger at lower run counts.
>     - The black data series remains relatively flat, hovering around -1.6, with smaller error bars.
>
> 2.  **Middle Panel: "Ring, σ = 3"**
>
>     - The y-axis ranges from -0.8 to 0.0, with major ticks at -0.8, -0.6, -0.4, -0.2, and 0.0.
>     - The blue data series starts around -0.5 and exhibits a consistent increasing trend, reaching approximately -0.05 by 40 runs. Error bars are present and appear to shrink slightly as the number of runs increases.
>     - The black data series maintains a relatively stable value around -0.8, with consistently small error bars.
>
> 3.  **Right Panel: "Multisensory, σ = 3"**
>     - The y-axis ranges from -452.5 to -451.0, with major ticks at -452.5, -452.0, -451.5, and -451.0.
>     - The blue data series begins around -452.0 and shows an upward trend, stabilizing near -451.05 at 40 runs. Error bars are visible.
>     - The black data series stays relatively constant around -452.5, displaying small error bars.
>
> Below the three panels, a legend clarifies the data series:
>
> - A blue circular marker with a horizontal error bar represents "E_stacked".
> - A black circular marker with a horizontal error bar represents "E_MC".
>
> Across all three plots, the blue data series consistently shows an increasing trend as the "N. of runs" increases, while the black data series remains relatively stable, suggesting a baseline or ground truth value. The blue series generally has larger error bars, particularly at lower run counts, compared to the black series.

Figure 6: Expected log-joint as estimated by S-VBMC ( $E_{\text {stacked }}$, blue) compared to that estimated via Monte Carlo sampling with numerous additional evaluations of the true log-joint ( $E_{\mathrm{MC}}$, black, which we use as the ground truth) for all our experiments with noisy targets. Dots represent the median value (of the 20 S-VBMC runs) and error bars $95 \%$ confidence intervals (computed from 10000 bootstrap resamples).

If we define the expected log-joint as estimated by S-VBMC as

$$
E_{\text {stacked }}=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \hat{w}_{m, k} \hat{I}_{m, k}
$$

the 'capped' ELBO will be

$$
\operatorname{ELBO}_{\text {stacked }}^{(\text {capped })}=\widehat{E}_{\text {stacked }}+\mathcal{H}\left[q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

where

$$
\widehat{E}_{\text {stacked }}=\min \left(E_{\text {stacked }}, E_{\text {cap }}\right)
$$

Note that, as mentioned in Section 3.1, each VBMC run performs its own parameter transformation $g_{m}(\cdot)$ during inference (Acerbi, 2020), which prevents meaningful comparisons of the expected log-joint and its components across runs. Throughout this section, we refer to and use the corrected VBMC estimates $\hat{I}_{m, k}$ as expressed in the common parameter space of $\boldsymbol{\theta}$. More details on this correction can be found in Appendix A.2. In what follows, we discuss two candidates for $E_{\text {cap }}$.

Capping with median expected log-joint (run-wise). As a possible candidate for $E_{\text {cap }}$, we considered the median of the VBMC estimates of the expected log-joint from individual runs. So, if

$$
E_{m}=\sum_{k=1}^{K_{m}} w_{m, k} \hat{I}_{m, k}
$$

is the VBMC estimate of the expected log-joint from the $m$-th individual run, then

$$
E_{\text {median }}=\operatorname{median}_{1 \leq m \leq M}\left(E_{m}\right)
$$

---

#### Page 16

Capping with median expected log-joint (component-wise). As the value of $E_{\text {median }}$ might be unstable and heavily dependent on individual VBMC runs (especially for low values of $M$ ), we also consider the median of the expected log-joints with respect to all individual components,

$$
I_{\text {median }}=\underset{1 \leq m \leq M}{\text { median }}\left(\tilde{I}_{m, k}\right)
$$

As each VBMC run typically has $K_{m}=50$, even with $M=2$ we would have a stacked posterior with 100 components, which should ensure more stability in the estimate.

Debiasing results. We applied both these corrections to all our experiments with noisy targets (described in Sections 4.4 and 4.5), with results shown in Figure 7. We observe that both solutions are effective in preventing the ELBO bias buildup for all three benchmark problems. The bias itself is still present for the ring and multisensory targets, but it remains roughly constant with increasing values of $M$, and in both cases is very limited. We also observe that using $I_{\text {median }}$ as $E_{\text {cap }}$ yields the most stable solution, with debiased ELBO values fluctuating less, and confidence intervals being considerably less wide. While this is true for all three scenarios, it is particularly evident in the multisensory benchmark, where the ELBO capped with $E_{\text {median }}$ tends to fluctuate wildly, sometimes leading to ELBO overestimation, and sometimes to ELBO underestimation. In contrast, capping with $I_{\text {median }}$ ensures a reliably contained bias $(<0.5)$. These results suggest that the latter method is more reliable across benchmarks and should therefore be preferred by practitioners.

> **Image description.** A multi-panel line graph titled "ELBO debiasing (S-VBMC)" displays the effects of different debiasing heuristics on noisy targets across varying numbers of runs. The figure consists of three subplots arranged horizontally, each representing a different benchmark problem: "GMM, $\sigma = 3$", "Ring, $\sigma = 3$", and "Multisensory, $\sigma = 3$". All three subplots share a common x-axis labeled "N. of runs" and a common y-axis labeled "ELBO". A legend at the bottom of the figure clarifies the different data series.
>
> Each subplot shows four distinct data series, represented by different colored markers and associated vertical error bars, plotted against the "N. of runs" (ranging from 2 to 40, with tick marks at 2, 4, 8, 16, 24, 32, 40). A horizontal black dotted line is present in each panel, representing a reference ELBO value, identified as ELBO$_{MC}$ from the context. The error bars represent 95% confidence intervals.
>
> **Panel 1: GMM, $\sigma = 3$**
>
> - The y-axis ranges from 2.5 to 3.5.
> - The "No capping" series (blue circles) starts around 2.4 at 2 runs, increases steeply, and then plateaus around 3.7, significantly above the reference dotted line (at ELBO = 3.0). Its error bars are relatively small.
> - The "$E_{\text{cap}} = E_{\text{median}}$" series (orange circles) starts around 2.5, shows some fluctuation, and then stabilizes around 2.9-3.0, close to the reference line. This series exhibits larger error bars, particularly at lower run counts.
> - The "$E_{\text{cap}} = I_{\text{median}}$" series (purple triangles) starts around 2.7, increases, and then stabilizes consistently around 2.9, also close to the reference line. Its error bars are moderate, generally smaller than the orange series.
> - The "ELBO$_{MC}$" series (black circles) starts around 2.3, increases, and then stabilizes around 2.9, closely tracking the reference dotted line. Its error bars are the smallest among all series.
>
> **Panel 2: Ring, $\sigma = 3$**
>
> - The y-axis ranges from 1.5 to 3.0.
> - The "No capping" series (blue circles) starts around 1.6 at 2 runs, increases sharply, and then plateaus around 3.1, well above the reference dotted line (at ELBO = 2.25). Its error bars are small.
> - The "$E_{\text{cap}} = E_{\text{median}}$" series (orange circles) starts around 1.8, shows considerable fluctuation, and then stabilizes around 2.4-2.5, slightly above the reference line. This series has very large error bars, especially at lower run counts.
> - The "$E_{\text{cap}} = I_{\text{median}}$" series (purple triangles) starts around 1.5, increases, and then stabilizes around 2.4, also slightly above the reference line. Its error bars are moderate, larger than the black and blue series but smaller than the orange.
> - The "ELBO$_{MC}$" series (black circles) starts around 1.3, increases, and then stabilizes around 2.2, closely tracking the reference dotted line. Its error bars are small.
>
> **Panel 3: Multisensory, $\sigma = 3$**
>
> - The y-axis ranges from -445.0 to -443.5.
> - The "No capping" series (blue circles) starts around -445.0 at 2 runs, increases rapidly, and then plateaus around -443.7, significantly above the reference dotted line (at ELBO = -444.7). Its error bars are small.
> - The "$E_{\text{cap}} = E_{\text{median}}$" series (orange circles) starts around -444.8, exhibits extreme fluctuations, and then stabilizes around -444.4, noticeably above the reference line. This series displays exceptionally large error bars across all run counts.
> - The "$E_{\text{cap}} = I_{\text{median}}$" series (purple triangles) starts around -444.7, increases, and then stabilizes around -444.4, also above the reference line. Its error bars are moderate, significantly smaller than the orange series but larger than the blue and black series.
> - The "ELBO$_{MC}$" series (black circles) starts around -445.2, increases, and then stabilizes around -444.7, closely tracking the reference dotted line. Its error bars are small.
>
> **Legend:**
> The legend, positioned below the three panels, uses horizontal error bar symbols to represent the data series:
>
> - A blue circle with a horizontal error bar is labeled "No capping".
> - An orange circle with a horizontal error bar is labeled "$E_{\text{cap}} = E_{\text{median}}$".
> - A purple triangle with a horizontal error bar is labeled "$E_{\text{cap}} = I_{\text{median}}$".
> - A black circle with a horizontal error bar is labeled "ELBO$_{\text{MC}}$".

Figure 7: Effects of debiasing heuristics on noisy targets, with dots representing the median value and error bars $95 \%$ confidence intervals (computed from 10000 bootstrap resamples) across 20 S -VBMC runs. Each panel displays the uncorrected ELBO as output by S-VBMC (blue), the capped ELBO using $E_{\text {median }}$ (orange), the capped ELBO using $I_{\text {median }}$ (purple), and the ground-truth ELBO obtained via Monte Carlo $\mathrm{ELBO}_{\mathrm{MC}}$ (black), for distinct benchmark problems. In each panel, the black dotted line is the ground-truth log marginal likelihood.

# 6 Discussion

The core problem we tackled in this work is that certain target posteriors might have properties (e.g., multimodality, long tails, long and narrow shapes) that pose significant challenges to global surrogate-based approaches to Bayesian inference like VBMC. VBMC in particular was developed to tackle problems with expensive likelihood functions, and thus relies on a limited target evaluation budget and an active sampling strategy (Acerbi, 2019), which, while effective in many cases (Acerbi, 2018; 2020), makes it vulnerable to leaving high-density probability regions unexplored (e.g., missing a mode). Therefore, attempting to build

---

#### Page 17

a single, global surrogate approximation of complex targets might be counter-productive. Our proposed solution relies on the idea that, as an alternative, we can rely on a collection of local surrogates to build a better, global posterior.

In this work, we introduced S-VBMC, a simple approach for merging independent VBMC runs in a principled way to yield a global posterior approximation. We tested its effectiveness on synthetic problems that we expected VBMC to have trouble with, such as targets with multiple modes and long, narrow shapes, as well as on two real-world problems from computational neuroscience. We also probed S-VBMC's robustness to noise, by adding Gaussian noise to log-likelihood evaluations for some of our benchmark targets. This approach mimics realistic scenarios where the log-likelihood itself is not available, but can be estimated via simulation (Wood, 2010; van Opheusden et al., 2020; Järvenpää et al., 2021; Acerbi, 2020). As expected, our results show that individual VBMC runs fail to yield good posterior approximations in our challenging problems. Conversely, S-VBMC is remarkably effective across benchmarks, with all metrics steadily improving as more VBMC runs are merged, with minor differences between noisy and noiseless settings. Importantly, we built S-VBMC to be robust to variability in the quality of individual VBMC posteriors. We initialise the weights $\hat{\mathbf{w}}$ using each run's ELBO (Eq. 16), which immediately downweights low-ELBO runs, and the subsequent optimisation further reduces their influence. To verify this robustness, in all benchmarks we deliberately retained low-ELBO (but converged) runs and only excluded those flagged as non-converged by pyvbmc (Huggins et al., 2023) and poorly converged ones, with large uncertainty associated with the estimates $\hat{I}_{k}$ (see Section 4.1). Despite this, the stacked posterior still improved as more runs were combined, suggesting that low-ELBO VBMC posteriors are naturally assigned negligible weight.

S-VBMC inherits the limitations of VBMC and other surrogate-based inference methods, such as applicability to relatively low-dimensional target posteriors (up to $\approx 10$ dimensions; see Acerbi, 2018; 2020; Järvenpää et al., 2021). While this fundamental constraint of the Gaussian process surrogate remains, it is possible that, in moderately high dimensions $(D \approx 10-15)$, diverse initialisations might allow different runs to capture complementary regions of the posterior, yielding incremental gains even when each surrogate is only locally accurate. However, we do not expect S-VBMC to overcome the core scaling issues of the surrogate itself, and a careful empirical study of higher-dimensional cases is left for future work. With that being said, S-VBMC effectively addresses some of VBMC's limitations, such as dealing with more complex posterior shapes and multimodality.

Our results show that S-VBMC represents a practical and effective approach for performing approximate Bayesian inference when the target log-likelihood is expensive or the posterior distribution exhibits specific features. In scenarios where the likelihood is fast to evaluate and noiseless, and the posterior largely unimodal, traditional inference methods such as Markov Chain Monte Carlo (MCMC) are likely to outperform S-VBMC and should remain the default choice. While methods like MCMC represent a gold standard in most cases, it is worth noting that standard MCMC algorithms can struggle to efficiently explore complex posteriors with multiple, well-separated modes, for which there is no off-the-shelf, easy solution. In contrast, S-VBMC's strategy of combining multiple independent local approximations, when paired with a diverse set of starting points, makes it better suited to explore challenging global structures. This suggests S-VBMC may be viable even in scenarios where the likelihood is not prohibitively expensive, but the posterior geometry is difficult for sequential samplers to navigate due to isolated modes.

Moreover, S-VBMC integrates seamlessly into the established best practices for its parent algorithm. For robustness and convergence diagnostics, performing several independent VBMC runs from different starting points is already recommended (Huggins et al., 2023). S-VBMC leverages this existing diagnostic workflow, providing a principled method to not only assess the inference landscape but to combine these runs into an improved, global posterior approximation at a small added computational cost. We intentionally designed S-VBMC as a simple post-processing step that requires no modification to the core VBMC algorithm. This simplicity is a key strength, as it preserves the embarrassingly parallel nature of running multiple independent inferences. While one could devise more sophisticated methods involving, for example, communication between runs to promote diversity and exploration, such approaches would sacrifice the ease of implementation and seamless integration that make S-VBMC a practical tool for practitioners. As our results show, this approach is most impactful for problems with features like multiple modes (GMM target), long and nar-

---

#### Page 18

row shapes (ring target), or heavy tails (neuronal model, see Appendix A.5), where individual runs explore different facets of the posterior.

To investigate the practical convenience of our approach, we measured and reported the computational overhead of S-VBMC. Our results, combined with our previous analyses, suggest that our approach yields considerable improvements in terms of posterior quality with a small added cost in terms of compute time, assuming VBMC runs can be executed in parallel. In light of this, we recommend using $M=10$ as a default choice, as it offers a strong accuracy-cost trade-off: most of the gains materialise by $M=10$, which in our experiments added a modest $\approx 5-15 \%$ overhead cost, assuming the VBMC runs are executed in parallel. As mentioned earlier, our method is, in principle, applicable for stacking mixture posteriors produced by any inference scheme, not strictly limited to VBMC. However, without a closed-form solution for the expected log-joints of the individual components of each run $\left(\left\{\mathbf{I}_{m}\right\}_{m=1}^{M}\right.$, estimates of which are available in VBMC), Monte Carlo estimates are required, greatly inflating the number of necessary likelihood evaluations, and thus the computational cost. This wouldn't be the case for inexpensive likelihoods, but, as discussed above, VBMC would not necessarily be the primary recommended approach.

Finally, we discussed the main noticeable downside of S-VBMC, namely an observed bias in the ELBO estimation building up with increasing numbers of merged VBMC runs in problems with noisy targets, and proposed a simple heuristic (applicable in an inexpensive post-processing step) that mitigates this. Even though this bias buildup didn't seem to affect posterior quality in our experiments, it would constitute a problem if one were to use the (inflated) ELBO estimates for model comparison. Our debiasing approach reduces this bias to a smaller magnitude and prevents it from building up, so we encourage practitioners to adopt it (or some other debiasing technique) when using S-VBMC estimates of the ELBO for model comparison in problems with noisy likelihoods.

We should note that, although we discussed a plausible candidate source for the ELBO bias build-up in Section 5.1, this work does not contain a thorough investigation of this phenomenon and its causes. Some of the bias could come from the internals of the VBMC algorithm itself, which could explain the presence of a residual bias (see Section 5.2). Furthermore, our debiasing method simply consists of a post-processing heuristic, which is attractive for its simplicity and speed, but preserves the bias build-up mechanism during optimisation and inference. Precise identification of bias sources could allow their neutralisation at inference time, possibly leading the algorithm to shift its focus from overestimating the expected log-joint to optimising an unbiased stacked ELBO. Nonetheless, an interesting finding is that in our benchmark problems the bias phenomenon did not affect posterior approximation quality as measured by our metrics, which kept improving steadily and approaching ground-truth posteriors while stacking more VBMC runs.

# 7 Conclusion

In this work, we introduced S-VBMC, a simple, novel approach for stacking variational posteriors generated by separate, possibly parallel, VBMC runs, and tested it on a set of challenging targets. We further suggested an effective and inexpensive method to address one main drawback of our approach (i.e., the ELBO bias build-up). Our results, both in terms of performance and compute time, show its practical convenience for VBMC users, especially when tackling particularly challenging inference problems.

## Acknowledgments

This work was supported by Research Council of Finland (grants 358980 and 356498). The authors wish to thank the Finnish Computing Competence Infrastructure (FCCI) for supporting this project with computational and data storage resources. The authors also acknowledge the research environment provided by ELLIS Institute Finland.

The model referred to as the "neuronal model" in the main text and appendices was developed by Dániel Terbe, Balázs Szabó and Szabolcs Káli (HUN-REN Institute of Experimental Medicine, Budapest, Hungary), using data collected by Miklós Szoboszlay in Zoltán Nusser's laboratory (HUN-REN Institute of Experimental Medicine, Budapest, Hungary). The authors thank these researchers for making their data and model available for the work described in this paper.

---

#### Page 19

# References

Luigi Acerbi. Variational Bayesian Monte Carlo. Advances in Neural Information Processing Systems, 31: $8222-8232,2018$.

Luigi Acerbi. An exploration of acquisition and mean functions in Variational Bayesian Monte Carlo. In Symposium on Advances in Approximate Bayesian Inference, pp. 1-10. PMLR, 2019.

Luigi Acerbi. Variational Bayesian Monte Carlo with noisy likelihoods. Advances in Neural Information Processing Systems, 33:8211-8222, 2020.

Luigi Acerbi, Kalpana Dokka, Dora E Angelaki, and Wei Ji Ma. Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS Computational Biology, 14 (7):e1006110, 2018.

Masaki Adachi, Satoshi Hayakawa, Martin Jørgensen, Harald Oberhauser, and Michael A Osborne. Fast Bayesian inference with batch Bayesian quadrature via kernel recombination. Advances in Neural Information Processing Systems, 35:16533-16547, 2022.

David M Blei, Alp Kucukelbir, and Jon D McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518):859-877, 2017.

Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng. Handbook of Markov Chain Monte Carlo. CRC press, 2011.

Kenneth P Burnham and David R Anderson. Model selection and multimodel inference: a practical information-theoretic approach. Springer Science \& Business Media, 2003.

Trevor Campbell and Xinglong Li. Universal boosting variational inference. Advances in Neural Information Processing Systems, 32:3484-3495, 2019.

Bob Carpenter, Andrew Gelman, Matthew D Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. Stan: A probabilistic programming language. Journal of Statistical Software, 76:1-32, 2017.

Ryan SY Chan, Murray Pollock, Adam M Johansen, and Gareth O Roberts. Divide-and-conquer fusion. Journal of Machine Learning Research, 24(193):1-82, 2023.

Daniel A De Souza, Diego Mesquita, Samuel Kaski, and Luigi Acerbi. Parallel MCMC without embarrassing failures. International Conference on Artificial Intelligence and Statistics, pp. 1786-1804, 2022.

Jonas El Gammal, Nils Schöneberg, Jesús Torrado, and Christian Fidler. Fast and robust Bayesian inference using Gaussian processes with GPry. Journal of Cosmology and Astroparticle Physics, 2023(10):021, 2023.

Daniel Foreman-Mackey. Corner.py: Scatterplot matrices in Python. Journal of Open Source Software, 1 $(2): 24,2016$.

Roman Garnett. Bayesian optimization. Cambridge University Press, 2023.
Zoubin Ghahramani and Carl Rasmussen. Bayesian Monte Carlo. Advances in Neural Information Processing Systems, 15:505-512, 2002.

Tom Gunter, Michael A Osborne, Roman Garnett, Philipp Hennig, and Stephen J Roberts. Sampling for inference in probabilistic models with fast Bayesian quadrature. Advances in Neural Information Processing Systems, 27:2789-2797, 2014.

Fangjian Guo, Xiangyu Wang, Kai Fan, Tamara Broderick, and David B Dunson. Boosting variational inference. arXiv preprint arXiv:1611.05559, 2016.

Michael Hines, Andrew P Davison, and Eilif Muller. NEURON and Python. Frontiers in Neuroinformatics, $3: 391,2009$.

---

#### Page 20

Michael L Hines and Nicholas T Carnevale. The NEURON simulation environment. Neural Computation, 9 (6):1179-1209, 1997.

Bobby Huggins, Chengkun Li, Marlon Tobaben, Mikko J. Aarnos, and Luigi Acerbi. PyVBMC: Efficient Bayesian inference in Python. Journal of Open Source Software, 8(86):5428, 2023.

Marko Järvenpää and Jukka Corander. Approximate Bayesian inference from noisy likelihoods with Gaussian process emulated MCMC. Journal of Machine Learning Research, 25(366):1-55, 2024.

Marko Järvenpää, Michael U Gutmann, Aki Vehtari, Pekka Marttinen, et al. Parallel Gaussian process surrogate Bayesian inference with noisy likelihood evaluations. Bayesian Analysis, 16(1):147-178, 2021.

Marc C Kennedy and Anthony O'Hagan. Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63(3):425-464, 2001.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations, 2014.

Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. Proceedings of the 2nd International Conference on Learning Representations, 2013.

Konrad P Körding, Ulrik Beierholm, Wei Ji Ma, Steven Quartz, Joshua B Tenenbaum, and Ladan Shams. Causal inference in multisensory perception. PLoS One, 2(9):e943, 2007.

Chengkun Li, Grégoire Clarté, Martin Jørgensen, and Luigi Acerbi. Fast post-process Bayesian inference with variational sparse Bayesian quadrature. Statistics and Computing, 35(6):167, 2025.

Jiajun Liang, Qian Zhang, Wei Deng, Qifan Song, and Guang Lin. Bayesian federated learning with Hamiltonian Monte Carlo: Algorithm and theory. Journal of Computational and Graphical Statistics, 34(2): $509-518,2025$.

David JC MacKay. Information theory, inference and learning algorithms. Cambridge University Press, 2003 .

Andrew C Miller, Nicholas J Foti, and Ryan P Adams. Variational boosting: Iteratively refining posterior approximations. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 2420-2429. PMLR, 2017.

Willie Neiswanger, Chong Wang, and Eric Xing. Asymptotically exact, embarrassingly parallel MCMC. Uncertainty in Artificial Intelligence - Proceedings of the 30th Conference, UAI 2014, 2014.

Christopher Nemeth and Chris Sherlock. Merging MCMC subposteriors through Gaussian-process approximations. Bayesian Analysis, 13(2):507-530, 2018.

Anthony O'Hagan. Bayes-Hermite quadrature. Journal of Statistical Planning and Inference, 29(3):245-260, 1991 .

Michael Osborne, Roman Garnett, Zoubin Ghahramani, David K Duvenaud, Stephen J Roberts, and Carl Rasmussen. Active learning of model evidence using Bayesian quadrature. Advances in Neural Information Processing Systems, 25:46-54, 2012.

Leah F Price, Christopher C Drovandi, Anthony Lee, and David J Nott. Bayesian synthetic likelihood. Journal of Computational and Graphical Statistics, 27(1):1-11, 2018.

Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational inference. In Artificial Intelligence and Statistics, pp. 814-822. PMLR, 2014.
C. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

Jerome Sacks, Susannah B Schiller, and William J Welch. Designs for computer experiments. Technometrics, $31(1): 41-47,1989$.

---

#### Page 21

Steven L Scott, Alexander W Blocker, Fernando V Bonassi, Hugh A Chipman, Edward I George, and Robert E McCulloch. Bayes and big data: The consensus Monte Carlo algorithm. In Big Data and Information Theory, pp. 8-18. Routledge, 2022.

Sanvesh Srivastava, Cheng Li, and David B Dunson. Scalable Bayes via barycenter in Wasserstein space. Journal of Machine Learning Research, 19(8):1-35, 2018.

Miklos Szoboszlay, Andrea Lőrincz, Frederic Lanore, Koen Vervaeke, R Angus Silver, and Zoltan Nusser. Functional properties of dendritic gap junctions in cerebellar Golgi cells. Neuron, 90(5):1043-1056, 2016.

Bas van Opheusden, Luigi Acerbi, and Wei Ji Ma. Unbiased and efficient log-likelihood estimation with inverse binomial sampling. PLoS Computational Biology, 16(12):e1008483, 2020.

Hongqiao Wang and Jinglai Li. Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions. Neural Computation, 30(11):3072-3094, 2018.

Xiangyu Wang and David B Dunson. Parallelizing MCMC via Weierstrass sampler. arXiv preprint arXiv:1312.4605, 2013.

Xiangyu Wang, Fangjian Guo, Katherine A Heller, and David B Dunson. Parallelizing MCMC with random partition trees. Advances in Neural Information Processing Systems, 28:451-459, 2015.

David H Wolpert. Stacked generalization. Neural Networks, 5(2):241-259, 1992.
Simon N Wood. Statistical inference for noisy nonlinear ecological dynamic systems. Nature, 466(7310): $1102-1104,2010$.

Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman. Using stacking to average Bayesian predictive distributions (with discussion). Bayesian Analysis, 13(3):917-1007, 2018.

Yuling Yao, Aki Vehtari, and Andrew Gelman. Stacking for non-mixing Bayesian computations: The curse and blessing of multimodal posteriors. Journal of Machine Learning Research, 23(79):1-45, 2022.

---

#### Page 22

# A Appendix

This appendix provides additional details and analyses to complement the main text, included in the following sections:

- An overview of Variational Bayesian Monte Carlo, A. 1
- A description of how S-VBMC handles VBMC's parameter transformations, A. 2
- Additional experiments, A. 3
- Full experimental results, A. 4
- Example posterior visualisations, A. 5

## A. 1 An overview of Variational Bayesian Monte Carlo

In this appendix we briefly describe Variational Bayesian Monte Carlo (VBMC). This is a simple overview of the various components of the algorithm, and a full in-depth description of these is beyond the scope of this appendix. For further details, see Acerbi (2018; 2020).

As mentioned in Section 2.3, VBMC addresses the problem of expensive likelihoods with black-box properties by using a surrogate for the log-joint (see Eq. 5). Like many other surrogate-based approaches (Garnett, 2023), VBMC uses a Gaussian process (GP) to approximate its expensive target (Rasmussen \& Williams, 2006). GPs are stochastic processes such that any finite collection $f\left(\mathbf{x}_{1}\right), \ldots, f\left(\mathbf{x}_{n}\right)$ follows a multivariate normal distribution. A GP is fully specified by a mean function

$$
m(\mathbf{x})=\mathbb{E}[f(\mathbf{x})]
$$

a covariance function (also called a kernel)

$$
\kappa\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\operatorname{cov}\left[f(\mathbf{x}), f\left(\mathbf{x}^{\prime}\right)\right]
$$

and an observation noise model or likelihood. In VBMC, as in most surrogate modelling approaches, the likelihood is assumed to be Gaussian, which affords closed-form GP posterior computations. Specifically, given a training set $(\mathbf{X}, \mathbf{y}, \mathbf{S})$ with $\mathbf{X}$ being the observed input locations, $\mathbf{y}$ the corresponding observed function values, and $\mathbf{S}$ a diagonal covariance matrix representing observation noise, the posterior mean and covariance functions for a test input location $\overline{\mathbf{x}}$ are, respectively,

$$
\mu_{p}(\overline{\mathbf{x}})=\kappa(\overline{\mathbf{x}}, \mathbf{X})(\kappa(\mathbf{X}, \mathbf{X})+\mathbf{S})^{-1}(\mathbf{y}-m(\mathbf{X}))+m(\overline{\mathbf{x}})
$$

and

$$
\kappa_{p}(\overline{\mathbf{x}}, \overline{\mathbf{x}})=\kappa(\overline{\mathbf{x}}, \overline{\mathbf{x}})-\kappa(\overline{\mathbf{x}}, \mathbf{X})(\kappa(\mathbf{X}, \mathbf{X})+\mathbf{S})^{-1} \kappa(\mathbf{X}, \overline{\mathbf{x}})
$$

For further details on GPs and their use in machine learning, see Rasmussen \& Williams (2006).
Crucially, a GP surrogate does not yield a usable posterior approximation, as the integral of Eq. 1 (Bayes' rule) remains intractable. Even if the integral can be solved, it does not yield a usable approximation of the posterior, such as the ability to draw samples from it. To address this point, VBMC makes use of Bayesian quadrature, a method for obtaining Bayesian estimates of intractable integrals (O'Hagan, 1991; Ghahramani \& Rasmussen, 2002). Given an integral

$$
\mathcal{J}=\int f(\mathbf{x}) \pi(\mathbf{x}) d \mathbf{x}
$$

where $f$ is the target function and $\pi$ is a known probability distribution, if a GP prior is specified for $f$, the integral $\mathcal{J}$ is a Gaussian random variable with posterior mean

$$
\mathbb{E}_{f}[\mathcal{J}]=\int \mu_{p}(\mathbf{x}) \pi(\mathbf{x}) d \mathbf{x}
$$

---

#### Page 23

and variance

$$
\mathbb{V}_{f}[\mathcal{J}]=\int \int \kappa_{p}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) \pi(\mathbf{x}) \pi\left(\mathbf{x}^{\prime}\right) d \mathbf{x} d \mathbf{x}^{\prime}
$$

If $f$ has a Gaussian kernel and $\pi(\mathbf{x})$ is a mixture of Gaussians - which is the case for the VBMC approximate posterior -, both integrals have closed-form solutions.
As mentioned in Section 2, VBMC leverages Bayesian quadrature to estimate the ELBO by building a surrogate model of the log-joint $f(\boldsymbol{\theta}) \approx \log p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})$. In particular, the GP model $f$ uses an exponentiated quadratic (more commonly known, if incorrectly, as squared exponential) kernel and a negative quadratic mean function (Acerbi, 2018; 2019). The latter ensures integrability of the function and is equivalent to an inductive bias towards Gaussian posteriors, but note that it does not limit the modelled target to be a Gaussian, as the GP can model arbitrary deviations from the mean function.
Therefore, putting everything together, the posterior mean of the surrogate ELBO can be calculated as

$$
\mathbb{E}_{f}[\operatorname{ELBO}(\boldsymbol{\phi})]=\mathbb{E}_{f}\left[\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{\theta})]\right]+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{\theta})\right]
$$

where $\mathbb{E}_{f}\left[\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{\theta})]\right]$ is the posterior mean of the GP surrogate of the log-joint (i.e., the expected value of the expected log-joint). Here the expected log-joint takes the form

$$
\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{\theta})]=\int f(\boldsymbol{\theta}) q_{\boldsymbol{\phi}}(\boldsymbol{\theta}) d \boldsymbol{\theta}
$$

which, since $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$ is a mixture of Gaussians, affords closed-form solutions for its posterior mean and variance, as well as its gradients.
To build a good surrogate approximation of the log-joint, a number of likelihood evaluations are needed to train the GP. Assuming the likelihood is expensive to compute, it is desirable to keep this number relatively contained. VBMC tackles this through an iterative process. In each iteration, VBMC selects the next points to evaluate by optimising an acquisition function that trades off exploration of new posterior regions and exploitation of high-density regions (see Acerbi, 2019; 2020). It then uses the resulting log-joint values to refine the GP surrogate, which is in turn used to refine the variational posterior via Bayesian quadrature. The VBMC algorithm begins with only two mixture components in a warm-up stage used to construct an initial surrogate model (see Acerbi, 2018). After the initial warm-up iterations are concluded, VBMC starts adding new components to the mixture to refine the posterior approximation. This process of acquiring new points, using them to improve the surrogate model, and then refining the variational approximation (possibly increasing the number of components), is repeated until a convergence criterion is met or until the likelihood evaluation budget is exceeded.

---

#### Page 24

# A. 2 Change-of-variables corrections in S-VBMC

Problem setting. Due to the variational whitening feature introduced in Acerbi (2020), each VBMC run operates in its own transformed parameter space, whereas in Acerbi (2018) all VBMC runs shared the same transformed parameter space (a fixed transform for bounded variables). An interested reader should refer to Acerbi (2020) for more information about variational whitening, and an in-depth discussion of this is beyond the scope of this appendix.
In the context of the $m$-th VBMC run, we call $g_{m}(\cdot)$ the function determining the parameter transformation, $g_{m}(\boldsymbol{\theta})$ the transformed parameters, and $\boldsymbol{\psi}_{m}$ the parameters of the variational posterior expressed in the transformed space, $q_{\boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right)$.
Crucially, VBMC returns approximate posteriors in the transformed space, and, since each run is executed independently, and has its own transformation $g_{m}(\cdot)$, the densities obtained with the different approximate posterior parameters $q_{\boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right)$ are not directly comparable. To calculate the stacked ELBO, we need to operate in the common (original) parameter space, in which the parameters $\boldsymbol{\theta}$ are expressed.
Concretely, this means applying appropriate corrections to the densities used to evaluate the stacked ELBO. In the following, we provide a brief introduction to such corrections and discuss how they can be applied to compute the entropy and the expected log-joint (i.e., the two terms of the stacked ELBO, see Eq. 11) in the common parameter space. Importantly, these corrections do not depend on the mixture weights of the stacked posterior, preserving the differentiability of our objective function with respect to $\hat{\mathbf{w}}$.

The Jacobian correction. Parameter transformations can be handled via the Jacobian correction. Given a random variable $\boldsymbol{x}$, its probability $p_{\boldsymbol{x}}(\boldsymbol{x})$ and a transformation $\boldsymbol{y}=T(\boldsymbol{x})$, we have

$$
p_{\boldsymbol{x}}(\boldsymbol{x})=p_{\boldsymbol{y}}(T(\boldsymbol{x}))\left|\operatorname{det} \frac{\partial T(\boldsymbol{x})}{\partial \boldsymbol{x}}\right|
$$

where the correction is applied with the absolute value of the determinant of the Jacobian of $T(\boldsymbol{x})$. Conveniently, the pyvbmc software (Huggins et al., 2023) provides a function to evaluate the (log) absolute value of the determinant of the Jacobian of the inverse transformation (evaluated at $g_{m}(\boldsymbol{\theta})$ ), which, in this toy example, would mean

$$
p_{\boldsymbol{x}}(\boldsymbol{x})=\frac{p_{\boldsymbol{y}}(T(\boldsymbol{x}))}{\left|\operatorname{det} \frac{\partial T^{-1}(T(\boldsymbol{x}))}{\partial T(\boldsymbol{x})}\right|}
$$

To tidy our notation, we will call our corrections term for the $m$-th transformation

$$
J_{m}\left(g_{m}(\boldsymbol{\theta})\right) \equiv\left|\operatorname{det} \frac{\partial g_{m}^{-1}\left(g_{m}(\boldsymbol{\theta})\right)}{\partial g_{m}(\boldsymbol{\theta})}\right|
$$

Bringing this back to our case, the variational posterior of the $m$-th VBMC run can be written as

$$
q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})=\sum_{k=1}^{K_{m}} w_{m, k} q_{k, \boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right) J_{m}^{-1}\left(g_{m}(\boldsymbol{\theta})\right)
$$

and the stacked posterior as

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} q_{k, \boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right) J_{m}^{-1}\left(g_{m}(\boldsymbol{\theta})\right)
$$

In line with the notation used in the main text, $\boldsymbol{\phi}_{m}$ and $\tilde{\boldsymbol{\phi}}$ are the parameters of the $m$-th VBMC posterior and of the stacked posterior, respectively, expressed in the common parameter space. With the exception of $\mathbf{w}_{m}$ and $\hat{\mathbf{w}}$ (which are not affected by the transformation), these parameters are unknown, but, as we will show in the following paragraphs, they are not needed to estimate the stacked ELBO.

---

#### Page 25

The corrected entropy. One term of the stacked ELBO is the entropy

$$
\mathcal{H}\left[q_{\hat{\boldsymbol{\phi}}}\right]=-\mathbb{E}_{q_{\hat{\boldsymbol{\phi}}}}\left[\log q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

for which no closed-form solution is available. We estimate it via Monte Carlo as in Eq. 12 of the main text, but crucially evaluate all component densities in the original space using the correction shown in Eqs. A. 13 and A.14. Concretely, for each component $q_{k, \boldsymbol{\psi}_{m}}$ we draw $S$ samples in the $m$-th transformed space, $\left\{\mathbf{z}_{m, k}^{(s)} \sim q_{k, \boldsymbol{\psi}_{m}}\right\}_{s=1}^{S}$, and map them to the original space, $\mathbf{x}_{m, k}^{(s)}=g_{m}^{-1}\left(\mathbf{z}_{m, k}^{(s)}\right)$. Then, for every sample $\mathbf{x}_{m, k}^{(s)}$ and for every component $q_{k^{\prime}, \boldsymbol{\phi}_{m^{\prime}}}$, we compute the per-component log-density in the original space via

$$
\log q_{k^{\prime}, \boldsymbol{\phi}_{m^{\prime}}}\left(\mathbf{x}_{m, k}^{(s)}\right)=\log q_{k^{\prime}, \boldsymbol{\psi}_{m^{\prime}}}\left(g_{m^{\prime}}\left(\mathbf{x}_{m, k}^{(s)}\right)\right)-\log J_{m^{\prime}}\left(g_{m^{\prime}}\left(\mathbf{x}_{m, k}^{(s)}\right)\right)
$$

and then aggregate

$$
\log q_{\hat{\boldsymbol{\phi}}}\left(\mathbf{x}_{m, k}^{(s)}\right)=\log \sum_{m^{\prime}=1}^{M} \sum_{k^{\prime}=1}^{K_{m^{\prime}}} \hat{w}_{m^{\prime}, k^{\prime}} q_{k^{\prime}, \boldsymbol{\phi}_{m^{\prime}}}\left(\mathbf{x}_{m, k}^{(s)}\right)
$$

via log-sum-exp. Then these values can be plugged into Eq. 12 of the main text to estimate the entropy. Importantly, the transformations do not depend on the mixture weights, so the entropy remains differentiable with respect to $\hat{\mathbf{w}}$.

The corrected expected log-joint. Let $\hat{L}_{m, k}$ be the VBMC estimate (computed in the run's transformed coordinates) of the component-wise expected log-joint,

$$
\hat{L}_{m, k} \approx \mathbb{E}_{q_{k, \boldsymbol{\psi}_{m}}}\left[\log p_{m}\left(\mathcal{D}, g_{m}(\boldsymbol{\theta})\right)\right]
$$

where $p_{m}\left(\mathcal{D}, g_{m}(\boldsymbol{\theta})\right)$ is the log-joint reparametrised with the $m$-th transform. To express this expectation in the common parameter space we apply the correction determined by $g_{m}(\boldsymbol{\theta})$

$$
\hat{I}_{m, k}=\hat{L}_{m, k}-\mathbb{E}_{q_{k, \boldsymbol{\psi}_{m}}}\left[\log J_{m}\left(g_{m}(\boldsymbol{\theta})\right)\right] \approx \mathbb{E}_{q_{k, \boldsymbol{\phi}_{m}}}\left[\log p(\mathcal{D}, \boldsymbol{\theta})\right]
$$

In practice we estimate the (per-component) Jacobian term in Eq. A. 19 using the same samples $\left\{\mathbf{z}_{m, k}^{(s)} \sim q_{k, \boldsymbol{\psi}_{m}}\right\}_{s=1}^{S}$ employed for the entropy and setting

$$
\mathbb{E}_{q_{k, \boldsymbol{\psi}_{m}}}\left[\log J_{m}\left(g_{m}(\boldsymbol{\theta})\right)\right] \approx \frac{1}{S} \sum_{s=1}^{S} \log J_{m}\left(\mathbf{z}_{m, k}^{(s)}\right)
$$

After applying these corrections, and those to the entropy described above, we can calculate the corrected stacked ELBO as in Eq. 11 of the main text.

Importantly, the $\hat{I}_{m, k}$ described here (i.e., the corrected ones) are the values we use in Section 5.2 for debiasing the stacked ELBO in noisy settings.

---

#### Page 26

# A. 3 Additional experiments

## A.3.1 S-VBMC variant

To probe the benefits of optimising the ELBO with respect to the weights of the individual components, we performed additional experiments comparing the version of S-VBMC presented in the main text ("allweights") with a S-VBMC variant where we only reweigh the weights of each individual VBMC posterior ("posterior-only"). Specifically, we considered a stacked posterior written as:

$$
q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \hat{\omega}_{m} q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

For this "posterior-only" variant, we optimised the global ELBO with respect to the weights $\hat{\omega}_{m}$ assigned to each posterior. This is similar to the naive stacking approach seen in the main paper (Eq. 18), with the difference that the posterior weights are now optimised. We ran this method for all the benchmark problems described in Sections 4.4 and 4.5, using the same bootstrapping procedure described in Section 4.1.

The results of this comparison are reported in Figures A. 1 and A.2, and in further detail in Appendix A.4.2. We observe that optimising with respect to $\hat{\boldsymbol{\omega}}$ ("posterior-only") performs well, with both MMTV and GsKL metrics steadily improving with increased numbers of stacked posteriors. In fact, for most problems, "posterior-only" S-VBMC performs comparably to the "all-weights" variant presented in the main paper, which optimises all components weights $\hat{\mathbf{w}}$. Still, the "all-weights" variant performs slightly better in the GsKL metric and in some challenging scenarios (e.g., the multisensory model), so it remains our base recommendation, paired with the debiasing approach described in Section 5.

## A.3.2 Additional runtime analyses

Here we report the total runtime cost (in seconds) of BBVI for all our benchmark problems compared to that of running VBMC 40 times and stacking the resulting posteriors with S-VBMC. We consider this particular number of runs because the BBVI target evaluation budget was set to match that of S-VBMC with $M=40$. For both, we report the median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples. For S-VBMC, each resample consisted of 40 VBMC runs and one S-VBMC run, then the S-VBMC runtime was added to that of the VBMC run with the highest runtime. This follows from the assumption that VBMC is run 40 times in parallel, and S-VBMC can be launched the moment the last VBMC run has converged.

It is important to note that, as is common in the surrogate-based literature (Acerbi, 2018; Wang \& Li, 2018; Acerbi, 2020; Järvenpää et al., 2021; El Gammal et al., 2023; Järvenpää \& Corander, 2024), in this work, we demonstrated the efficacy of our method on several problems where function evaluations are not computationally expensive, as a full benchmark with multiple expensive models is highly impractical. Therefore, wall-clock time needs to be interpreted carefully as a metric when comparing methods with different likelihood evaluation costs.

We can directly compare VBMC and S-VBMC, as we did in Section 4.6, because by construction they use the same backbone method and have the same evaluation costs (S-VBMC adds a small post-processing cost, which, crucially, does not depend on the cost of likelihood evaluation). Conversely, comparisons to non-VBMC methods become highly problem-dependent. The typical solution would consist of matching the number of function evaluations (as we did, see Section 4.2), for which non-surrogate-based baselines would be at a significant disadvantage, as demonstrated in previous work (Acerbi, 2018; 2020).

These considerations are crucial to interpret these results, displayed in Table A.1. As expected, BBVI is much faster than S-VBMC on problems with fast likelihood evaluation, but as soon as the likelihood becomes more expensive ( $\approx 0.7$ seconds per evaluation for the neuronal model) the cost of non-VBMC methods increases dramatically, illustrating the kind of scenarios VBMC was developed to solve in the first place.

Finally, it is worth noting that, as shown in Figures 3 and 4 and Tables A. 3 and A.4, BBVI performs substantially worse than S-VBMC across our examples, particularly in our real-world problems. Therefore, even where there may be runtime advantages (with the caveats discussed above), these come at a considerable cost in terms of posterior quality.

---

#### Page 27

> **Image description.** A complex figure composed of a 4x4 grid of line graphs, totaling 16 individual plots, designed to compare the performance of two versions of S-VBMC.
>
> The figure is organized into four rows, each representing a different problem type, and four columns, each representing a different performance metric.
>
> **Rows (Problem Types):**
>
> - **Row (a):** Titled "GMM (D = 2, noiseless)".
> - **Row (b):** Titled "GMM (D = 2, σ = 3)".
> - **Row (c):** Titled "Ring (D = 2, noiseless)".
> - **Row (d):** Titled "Ring (D = 2, σ = 3)".
>
> **Columns (Performance Metrics):**
>
> - **Column 1:** Labeled "ELBO" on the y-axis.
> - **Column 2:** Labeled "Δ LML" on the y-axis.
> - **Column 3:** Labeled "MMTV" on the y-axis.
> - **Column 4:** Labeled "GsKL" on the y-axis.
>
> **Common Elements Across All Graphs:**
>
> - **X-axis:** All graphs share the same x-axis, labeled "N. of runs", with tick marks at 4, 8, 16, 24, 32, and 40.
> - **Data Representation:** Two distinct datasets are plotted in each graph:
>   - A blue line with circular markers, accompanied by vertical error bars.
>   - A yellow line with upward-pointing triangular markers, also accompanied by vertical error bars.
> - **Legend:** Located at the bottom of the entire figure:
>   - A yellow triangle and line segment denotes ""posterior-only" S-VBMC".
>   - A blue circle and line segment denotes ""all-weights" S-VBMC".
>
> **Specific Details for Each Column:**
>
> - **ELBO (Column 1):**
>
>   - The y-axis has a linear scale, with ranges varying per row (e.g., 2.4 to 3.0 in row (a), 1.5 to 3.0 in row (d)).
>   - A solid horizontal black line is present in each ELBO plot, representing the ground-truth LML.
>   - In general, both blue and yellow lines show an increasing trend with "N. of runs". In rows (a) and (c) (noiseless problems), both lines typically converge towards the black ground-truth LML line. In rows (b) and (d) (σ = 3 problems), the blue line consistently achieves higher ELBO values than the yellow line, though neither fully reaches the ground-truth LML within the plotted range.
>
> - **Δ LML (Column 2), MMTV (Column 3), and GsKL (Column 4):**
>   - **Y-axis Scales:**
>     - Δ LML and GsKL plots utilize a logarithmic y-axis scale (e.g., 0.001 to 1 for Δ LML, 0.0001 to 1 for GsKL).
>     - MMTV plots use a linear y-axis scale, with ranges varying per row (e.g., 0.0 to 0.2 in row (a), 0.0 to 0.6 in row (c)).
>   - **Threshold Line:** A dashed horizontal black line is present in all plots in these three columns, indicating a desirable performance threshold.
>   - **Visual Patterns:** For all three metrics, both blue and yellow lines generally show a decreasing trend with "N. of runs", indicating improving performance. The blue line ("all-weights" S-VBMC) consistently shows lower values (better performance) and often falls below the dashed threshold more reliably or quickly than the yellow line ("posterior-only" S-VBMC), particularly in rows (b) and (d). The error bars indicate the variability of the measurements.
>
> In summary, the figure visually compares the convergence and performance metrics of two S-VBMC weighting schemes across different problem complexities, with "all-weights" S-VBMC (blue) generally demonstrating superior or faster convergence, especially in the more challenging noisy scenarios.

Figure A.1: Performance comparison between the two versions of S-VBMC ("all-weights" and "posterioronly") on synthetic problems. Metrics are plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples) for S-VBMC when the ELBO is optimised with respect to "all-weights" (blue) and "posterior-only" weights (yellow). The black horizontal line in the ELBO panels represents the ground-truth LML, while the dashed lines on $\Delta$ LML, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3)

---

#### Page 28

> **Image description.** A multi-panel figure presenting eight line graphs arranged in two rows and four columns, comparing the performance of two versions of S-VBMC on two different models. Each graph plots a specific metric against the number of VBMC runs.
>
> The figure is divided into two main sections, labeled (a) and (b), each occupying a row.
>
> **Section (a): Neuronal model ($D=5$, noiseless)**
> This top row consists of four line graphs, all sharing the same x-axis representing "N. of runs" from 4 to 40, with major ticks at 4, 8, 16, 24, 32, and 40. Each graph displays two data series: one in blue with circular markers and one in yellow with downward-pointing triangular markers. Both series include vertical error bars representing confidence intervals.
>
> - **Panel 1 (ELBO):** The y-axis is labeled "ELBO" and ranges from approximately -7456 to -7450. Both blue and yellow data series show an increasing trend, starting around -7454 and gradually rising to approximately -7452. The blue series generally shows slightly higher ELBO values than the yellow series. A thick black horizontal line is present at approximately -7451.5, representing the ground-truth LML.
> - **Panel 2 ($\Delta$ LML):** The y-axis is labeled "$\Delta$ LML" and ranges from 0 to 1. Both series show a decreasing trend, starting above 1 and dropping towards 0. The blue series generally decreases faster and reaches lower values than the yellow series, especially as the number of runs increases. A thin black dashed horizontal line is present at y=1, indicating a desirable threshold.
> - **Panel 3 (MMTV):** The y-axis is labeled "MMTV" and ranges from 0.0 to 0.4. Both series show a decreasing trend, starting around 0.25 and dropping towards 0.1 or below. The blue series generally achieves slightly lower MMTV values than the yellow series after about 16 runs. A thin black dashed horizontal line is present at y=0.2, indicating a desirable threshold.
> - **Panel 4 (GsKL):** The y-axis is labeled "GsKL" and uses a logarithmic scale, ranging from 0.1 to 10. Both series show a decreasing trend, starting around 10 and dropping towards 0.1 or below. The blue series generally shows lower GsKL values and tighter error bars than the yellow series, particularly at higher numbers of runs. A thin black dashed horizontal line is present at y=0.1, indicating a desirable threshold.
>
> **Section (b): Multisensory model ($D=6$, $\sigma=3$)**
> This bottom row also consists of four line graphs, mirroring the structure of the top row. The x-axis for the leftmost panel is labeled "N. of runs" (4 to 40), and this applies to all panels in this row. The two data series (blue circles and yellow downward triangles) with error bars are consistent.
>
> - **Panel 1 (ELBO):** The y-axis is labeled "ELBO" and ranges from approximately -444.5 to -443.5. Both series show an increasing trend, starting around -444.5 and rising. The blue series consistently shows higher ELBO values than the yellow series across all runs. A thick black horizontal line is present at approximately -444.6, representing the ground-truth LML.
> - **Panel 2 ($\Delta$ LML):** The y-axis is labeled "$\Delta$ LML" and ranges from 0 to 1. Both series show a decreasing trend, starting above 1 and dropping. The blue series generally decreases faster and reaches lower values than the yellow series, with both converging towards 0. A thin black dashed horizontal line is present at y=1, indicating a desirable threshold.
> - **Panel 3 (MMTV):** The y-axis is labeled "MMTV" and ranges from 0.00 to 0.20. Both series show a decreasing trend, starting around 0.15 and dropping towards 0.1 or below. The blue series generally achieves slightly lower MMTV values than the yellow series, especially after about 16 runs. A thin black dashed horizontal line is present at y=0.20, indicating a desirable threshold.
> - **Panel 4 (GsKL):** The y-axis is labeled "GsKL" and uses a logarithmic scale, ranging from 0.1 to 10. Both series show a decreasing trend, starting around 10 and dropping towards 0.1 or below. The blue series generally shows lower GsKL values and tighter error bars than the yellow series. A thin black dashed horizontal line is present at y=0.1, indicating a desirable threshold.
>
> A common legend is located below the bottom row of graphs. It identifies the two data series:
>
> - An icon showing a yellow downward-pointing triangle with vertical error bars is labeled: "posterior-only" S-VBMC
> - An icon showing a blue circle with vertical error bars is labeled: "all-weights" S-VBMC

Figure A.2: Performance comparison between the two versions of S-VBMC ("all-weights" and "posterioronly") on real-world problems. Metrics are plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples) for S-VBMC when the ELBO is optimised with respect to "all-weights" (blue) and "posterior-only" weights (yellow). The black horizontal line in the ELBO panels represents the ground-truth LML, while the dashed lines on $\Delta$ LML, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3)

Table A.1: BBVI runtime (in seconds) compared to that of 40 (parallel) VBMC runs and their subsequent stacking with S-VBMC. Values show median with $95 \%$ confidence interval in brackets. Bold entries indicate the best median performance (i.e., lowest compute time).

|                                 |           Algorithm           |                                       |
| :------------------------------ | :---------------------------: | :-----------------------------------: |
| Benchmark                       |       BBVI runtime (s)        |       VBMC + S-VBMC runtime (s)       |
| GMM (noiseless)                 |  $\mathbf{9 . 9}[9.4,10.3]$   |         $458.3[428.3,510.8]$          |
| GMM $(\sigma=3)$                | $\mathbf{1 2 . 8}[12.2,13.7]$ |         $857.8[759.0,954.8]$          |
| Ring (noiseless)                | $\mathbf{1 2 . 2}[11.7,12.6]$ |         $559.3[516.9,794.2]$          |
| Ring $(\sigma=3)$               | $\mathbf{1 4 . 0}[13.4,15.0]$ |        $1269.9[1206.0,1557.4]$        |
| Neuronal model (noiseless)      |    $8497.0[8411.2,8617.8]$    | $\mathbf{1 5 6 1 . 8}[1445.1,1900.1]$ |
| Multisensory model $(\sigma=3)$ | $\mathbf{2 4 . 5}[21.2,27.3]$ |        $3149.5[2616.3,3525.1]$        |

---

#### Page 29

# A. 4 Full experimental results

## A.4.1 Filtering procedure

Here we briefly present the results of our filtering procedure, described in Section 4.1. As shown in Table A.2, VBMC had considerable trouble when run on the neuronal model, with over half the runs failing to converge (as assessed by the pyvbmc software, Huggins et al., 2023), suggesting a complex, non-trivial posterior structure which is reflected in the poor performance of other inference methods (see Figure 4 and Table A.4). Convergence issues were also found with the noisy Ring target, although to a lesser extent. Once nonconverged runs were discarded, our second filtering criterion (i.e., excluding poorly converged runs with excessive uncertainty associated with the $\hat{I}_{k}$ estimates) led to considerably fewer exclusions overall, with only the Ring target being somewhat affected ( $8 \%$ and $13 \%$ of runs discarded in noiseless and noisy settings, respectively).

All our VBMC runs were indexed, and, for our experiments, we used the 100 filtered runs with the lowest indices.

Table A.2: Result of our filtering procedures. This table shows the total number of VBMC runs we performed ("Total"), those that did not converge ("Non-converged") and converged poorly ("Poorly converged") out of the total, and the ones that passed both filtering criteria ("Remaining").

|                            | VBMC runs |               |                  |           |
| :------------------------- | :-------: | :-----------: | :--------------: | :-------: |
| Benchmark                  |   Total   | Non-converged | Poorly converged | Remaining |
| GMM (noiseless)            |    120    |       2       |        3         |    115    |
| GMM $(\sigma=3)$           |    150    |       0       |        5         |    145    |
| Ring (noiseless)           |    120    |       3       |        9         |    108    |
| Ring $(\sigma=3)$          |    149    |      34       |        15        |    100    |
| Neuronal model (noiseless) |    300    |      159      |        1         |    140    |
| Multisensory $(\sigma=3)$  |    150    |       0       |        1         |    149    |

## A.4.2 Posterior metrics

We present a comprehensive comparison of S-VBMC against VBMC, NS and BBVI in Tables A. 3 and A.4, complementing the visualisations in Figures 3, 4, A. 1 and A.2. We consider both the version of S-VBMC described in the main text (where the ELBO is optimised with respect to the component weights $\hat{\mathbf{w}}$, "allweights"), and the one described in Appendix A.3.1 (where the ELBO is optimised with respect to the posterior weights $\hat{\boldsymbol{\omega}}$, "posterior-only").

For both synthetic problems (Table A.3) and real-world problems (Table A.4), S-VBMC generally demonstrates consistently improved posterior approximation metrics compared to the baselines. However, we observe an increase in $\Delta$ LML error with larger numbers of stacked runs in problems with noisy targets. This increase likely stems from the accumulation of ELBO estimation bias, a phenomenon analysed in detail in Section 5.

---

#### Page 30

Table A.3: Comparison of S-VBMC, VBMC, and BBVI performance on synthetic benchmark problems. Values show median with $95 \%$ confidence intervals (computed from 10000 bootstrap resamples) in brackets. Bold entries indicate best median performance; multiple entries are bolded when confidence intervals overlap with the best median. For compactness, we label the S-VBMC version described in the main text "w.r.t. $\hat{\mathbf{w}}$ ", (indicating that the ELBO is optimised with respect to "all-weights" $\hat{\mathbf{w}}$ ), and the version described in Appendix A.3.1 "w.r.t. $\hat{\boldsymbol{\omega}}$ ", (indicating that the ELBO is optimised with respect to $\hat{\boldsymbol{\omega}}$, or "posterior-only").

|                      Algorithm                       |       Benchmarks        |                      |                          |                         |                   |                          |
| :--------------------------------------------------: | :---------------------: | :------------------: | :----------------------: | :---------------------: | :---------------: | :----------------------: |
|                                                      |           GMM           |                      |                          |          Ring           |                   |                          |
|                                                      | $\Delta \mathbf{L M L}$ |         MMTV         |           GsKL           | $\Delta \mathbf{L M L}$ |       MMTV        |           GsKL           |
|                                                      |        Noiseless        |                      |                          |                         |                   |                          |
|                  BBVI, MoG $(K=50)$                  |  $0.059[0.028,0.075]$   | $0.059[0.035,0.08]$  |  $0.0083[0.0011,0.010]$  |      $8[0.8,9.6]$       | $0.51[0.48,0.53]$ |     $0.72[0.66,1.2]$     |
|                 BBVI, MoG $(K=500)$                  |   $0.053[0.029,0.11]$   | $0.052[0.043,0.07]$  |  $0.0087[0.0025,0.013]$  |      $8.3[6.9,12]$      | $0.47[0.45,0.49]$ |    $0.67[0.55,0.81]$     |
|                         VBMC                         |     $1.4[0.7,1.4]$      |  $0.54[0.39,0.55]$   |       $13[7.6,14]$       |     $1.2[1.2,1.3]$      | $0.53[0.51,0.56]$ |      $9.4[7.2,14]$       |
|                     NS (10 runs)                     |   $0.091[0.07,0.12]$    |  $0.15[0.12,0.17]$   |   $0.054[0.034,0.075]$   |    $0.16[0.09,0.24]$    | $0.19[0.18,0.22]$ |   $0.04[0.021,0.091]$    |
|                     NS (20 runs)                     |  $0.047[0.037,0.062]$   |  $0.11[0.089,0.12]$  |   $0.027[0.018,0.032]$   |   $0.11[0.073,0.13]$    | $0.18[0.16,0.18]$ |   $0.028[0.018,0.049]$   |
| S-VBMC (w.r.t. $\hat{\boldsymbol{\omega}}, 10$ runs) | $0.0042[0.0037,0.0087]$ | $0.032[0.031,0.04]$  | $0.0017[0.00081,0.003]$  |   $0.08[0.057,0.22]$    | $0.16[0.15,0.2]$  |  $0.0065[0.0029,0.043]$  |
| S-VBMC (w.r.t. $\hat{\boldsymbol{\omega}}, 20$ runs) | $0.0059[0.0035,0.0074]$ | $0.031[0.024,0.035]$ | $0.0011[0.00064,0.0016]$ |   $0.04[0.037,0.048]$   | $0.15[0.14,0.15]$ |  $0.002[0.0013,0.0028]$  |
|     S-VBMC (w.r.t. $\hat{\mathbf{w}}, 10$ runs)      | $0.0089[0.0043,0.015]$  | $0.036[0.028,0.05]$  |  $0.0015[0.0011,0.004]$  |  $0.034[0.027,0.047]$   | $0.14[0.14,0.14]$ | $0.0013[0.00081,0.0023]$ |
|     S-VBMC (w.r.t. $\hat{\mathbf{w}}, 20$ runs)      | $0.0046[0.0028,0.0072]$ | $0.031[0.026,0.036]$ | $0.0013[0.00047,0.0019]$ |  $0.022[0.019,0.026]$   | $0.14[0.13,0.14]$ | $0.0011[0.00096,0.0014]$ |
|                  Noisy $(\sigma=3)$                  |                         |                      |                          |                         |                   |                          |
|                  BBVI, MoG $(K=50)$                  |    $0.23[0.11,0.43]$    |  $0.13[0.092,0.18]$  |    $0.03[0.01,0.12]$     |     $4.3[3.3,4.7]$      | $0.51[0.47,0.54]$ |     $1.1[0.65,1.7]$      |
|                 BBVI, MoG $(K=500)$                  |   $0.27[0.076,0.45]$    |  $0.1[0.094,0.13]$   |   $0.019[0.011,0.034]$   |      $4.7[4,5.5]$       | $0.93[0.91,0.94]$ |       $48[28,49]$        |
|                         VBMC                         |    $0.98[0.78,1.1]$     |  $0.44[0.43,0.47]$   |      $9.7[8.5,11]$       |     $1.3[1.1,1.5]$      | $0.62[0.57,0.65]$ |       $38[24,95]$        |
|                     NS (10 runs)                     |   $0.11[0.066,0.19]$    |  $0.17[0.14,0.16]$   |   $0.066[0.029,0.12]$    |   $0.082[0.066,0.12]$   | $0.23[0.21,0.26]$ |   $0.056[0.033,0.091]$   |
|                     NS (20 runs)                     |  $0.056[0.046,0.082]$   |   $0.1[0.09,0.12]$   |   $0.017[0.011,0.026]$   |    $0.19[0.16,0.22]$    | $0.18[0.16,0.2]$  |   $0.023[0.017,0.03]$    |
| S-VBMC (w.r.t. $\hat{\boldsymbol{\omega}}, 10$ runs) |    $0.19[0.16,0.24]$    |  $0.13[0.11,0.14]$   |  $0.012[0.0069,0.031]$   |    $0.23[0.12,0.28]$    | $0.23[0.21,0.24]$ |    $0.02[0.01,0.026]$    |
| S-VBMC (w.r.t. $\hat{\boldsymbol{\omega}}, 20$ runs) |    $0.32[0.39,0.34]$    | $0.089[0.078,0.098]$ |  $0.0082[0.004,0.013]$   |    $0.37[0.34,0.41]$    | $0.18[0.17,0.19]$ |  $0.0054[0.004,0.011]$   |
|     S-VBMC (w.r.t. $\hat{\mathbf{w}}, 10$ runs)      |    $0.32[0.27,0.45]$    |  $0.11[0.092,0.13]$  |  $0.016[0.0058,0.034]$   |    $0.39[0.34,0.45]$    | $0.2[0.19,0.31]$  |  $0.053[0.0089,0.025]$   |
|     S-VBMC (w.r.t. $\hat{\mathbf{w}}, 20$ runs)      |    $0.53[0.51,0.61]$    | $0.09[0.084,0.097]$  | $0.0049[0.0036,0.0072]$  |    $0.68[0.63,0.71]$    | $0.17[0.17,0.18]$ | $0.0045[0.0025,0.0071]$  |

Table A.4: Comparison of S-VBMC, VBMC, and BBVI performance on neuronal and multisensory causal inference models. Bold entries indicate best median performance; multiple entries are bolded when confidence intervals overlap with the best median. See the caption of Table A. 3 for further details.

|                      Algorithm                       |           Benchmarks            |                      |                      |                         |                    |                    |
| :--------------------------------------------------: | :-----------------------------: | :------------------: | :------------------: | :---------------------: | :----------------: | :----------------: |
|                                                      | Multisensory model $(\sigma=3)$ |                      |                      |     Neuronal model      |                    |                    |
|                                                      |     $\Delta \mathbf{L M L}$     |         MMTV         |         GsKL         | $\Delta \mathbf{L M L}$ |        MMTV        |        GsKL        |
|                  BBVI, MoG $(K=50)$                  |         $1.7[1.5,4.9]$          |  $0.11[0.097,0.13]$  |   $0.17[0.16,0.2]$   |      $44[33,120]$       |  $0.6[0.56,0.64]$  |    $20[17,23]$     |
|                 BBVI, MoG $(K=500)$                  |         $1.8[1.6,2.5]$          |  $0.31[0.28,0.33]$   |  $0.53[0.48,0.55]$   |     $170[140,260]$      |  $0.67[0.64,0.7]$  |    $21[18,26]$     |
|                         VBMC                         |        $0.32[0.23,0.37]$        |  $0.18[0.17,0.19]$   |  $0.21[0.17,0.23]$   |       $3[3,3.1]$        | $0.32[0.31,0.32]$  |   $140[97,190]$    |
|                     NS (10 runs)                     |        $0.46[0.41,0.52]$        |  $0.12[0.11,0.12]$   | $0.072[0.056,0.078]$ |     $1.8[1.8,1.9]$      | $0.17[0.16,0.18]$  |  $1.2[0.27,1.4]$   |
|                     NS (20 runs)                     |        $0.55[0.5,0.59]$         |  $0.1[0.096,0.11]$   | $0.06[0.052,0.068]$  |     $1.8[1.7,1.9]$      | $0.17[0.15,0.18]$  |   $1[0.35,1.6]$    |
| S-VBMC (w.r.t. $\hat{\boldsymbol{\omega}}, 10$ runs) |        $0.73[0.63,0.86]$        |  $0.11[0.097,0.12]$  | $0.062[0.052,0.074]$ |     $1.8[1.7,1.8]$      | $0.14[0.12,0.15]$  |  $0.67[0.36,1.1]$  |
| S-VBMC (w.r.t. $\hat{\boldsymbol{\omega}}, 20$ runs) |        $0.88[0.86,0.95]$        |  $0.1[0.095,0.11]$   | $0.047[0.044,0.057]$ |     $1.5[1.5,1.6]$      | $0.11[0.087,0.13]$ | $0.3[0.037,0.57]$  |
|     S-VBMC (w.r.t. $\hat{\mathbf{w}}, 10$ runs)      |         $0.93[0.89,1]$          | $0.091[0.086,0.094]$ | $0.042[0.038,0.05]$  |     $1.7[1.6,1.7]$      | $0.14[0.11,0.15]$  | $0.47[0.17,0.79]$  |
|     S-VBMC (w.r.t. $\hat{\mathbf{w}}, 20$ runs)      |         $1.2[1.2,1.3]$          | $0.079[0.076,0.091]$ | $0.039[0.03,0.044]$  |     $1.5[1.5,1.6]$      | $0.12[0.092,0.13]$ | $0.48[0.059,0.54]$ |

---

#### Page 31

# A. 5 Example posterior visualisations

We use corner plots (Foreman-Mackey, 2016) to visualise exemplar posterior approximations from different algorithms, including S-VBMC, VBMC and BBVI. These plots depict one-dimensional marginal distributions and all pairwise two-dimensional marginals of the posterior samples. Example results (chosen at random among the runs reported in Section 4 and Appendix A.4) are shown in Figures A.3, A.4, A.5, and A.6. SVBMC consistently improves the posterior approximations over standard VBMC and generally outperforms BBVI, showing a closer alignment with the target posterior.

> **Image description.** A multi-panel figure displays four "corner plots," each visualizing posterior distributions for two variables, $\theta_1$ and $\theta_2$, under different conditions. The panels are arranged in a 2x2 grid, labeled (a) through (d).
>
> Each corner plot consists of three sub-plots:
>
> - A central square plot showing the two-dimensional joint density of $\theta_1$ (x-axis) and $\theta_2$ (y-axis) using contour lines and shading.
> - An upper rectangular plot showing the one-dimensional marginal distribution of $\theta_1$ as a histogram.
> - A right rectangular plot showing the one-dimensional marginal distribution of $\theta_2$ as a histogram.
>
> Across all panels, the x-axis for $\theta_1$ and the y-axis for $\theta_2$ range from approximately -10 to 10. The histograms' y-axes represent density or frequency. In each sub-plot, two distributions are shown: a "target" distribution outlined in black (and gray contours/shading for 2D plots) and an "approximation" distribution outlined in orange (and orange contours/shading for 2D plots). The black target distribution consistently shows a multimodal structure with four distinct clusters in the 2D joint plot (at approximately (-8, 8), (8, 8), (-8, -8), and (8, -8)), and two peaks in each 1D marginal histogram (around -8 and 8).
>
> The specific content of each panel is as follows:
>
> - **Panel (a) "VBMC"**:
>
>   - **2D Plot**: The gray contours show four distinct clusters. The orange contours and shading show only one cluster, located in the bottom-right quadrant (around (8, -8)).
>   - **$\theta_1$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the rightmost peak of the black distribution (around 8).
>   - **$\theta_2$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the leftmost peak of the black distribution (around -8).
>     This indicates that the VBMC approximation only captures one of the four modes of the target posterior.
>
> - **Panel (b) "S-VBMC (20 runs)"**:
>
>   - **2D Plot**: Both the gray and orange contours and shading are very similar, clearly showing all four distinct clusters. The orange contours are slightly more prominent.
>   - **$\theta_1$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>   - **$\theta_2$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>     This indicates that the S-VBMC approximation closely matches the target posterior, capturing all four modes.
>
> - **Panel (c) "VBMC (noisy)"**:
>
>   - **2D Plot**: The gray contours show four distinct clusters. The orange contours and shading show only one cluster, located in the top-right quadrant (around (8, 8)).
>   - **$\theta_1$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the rightmost peak of the black distribution (around 8).
>   - **$\theta_2$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the rightmost peak of the black distribution (around 8).
>     Similar to panel (a), the VBMC approximation under noisy conditions also only captures one of the four modes of the target posterior, but a different one.
>
> - **Panel (d) "S-VBMC (20 runs, noisy)"**:
>   - **2D Plot**: Both the gray and orange contours and shading are very similar, clearly showing all four distinct clusters. The orange contours are slightly more prominent.
>   - **$\theta_1$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>   - **$\theta_2$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>     Similar to panel (b), the S-VBMC approximation under noisy conditions also closely matches the target posterior, capturing all four modes.
>
> In summary, panels (a) and (c) show that VBMC struggles to approximate the multimodal target posterior, capturing only one mode, while panels (b) and (d) demonstrate that S-VBMC successfully approximates all modes of the target posterior, even under noisy conditions. The black distributions representing the target posterior appear consistent across all four panels.

---

#### Page 32

> **Image description.** A 2x2 grid of four multi-panel plots, labeled (e), (f), (g), and (h), each visualizing posterior and ground truth distributions for two variables, $\theta_1$ and $\theta_2$. Each of these four larger panels is a "corner plot" displaying both 1D marginal histograms and a 2D joint contour plot.
>
> **Overall Layout and Common Elements:**
>
> - The image is structured as a grid of four distinct sub-figures.
> - Each sub-figure consists of three individual plots:
>   - A central square plot showing the 2D joint distribution of $\theta_1$ (horizontal axis) and $\theta_2$ (vertical axis).
>   - A rectangular plot positioned directly above the central plot, illustrating the 1D marginal distribution of $\theta_1$.
>   - A rectangular plot positioned directly to the right of the central plot, illustrating the 1D marginal distribution of $\theta_2$.
> - **Axis Labels and Ranges**:
>   - The horizontal axis for the central and top plots is consistently labeled "$\theta_1$", with values ranging from -10 to 10. Major tick marks are present at -8, 0, and 8, with minor ticks at -10 and 10.
>   - The vertical axis for the central and right plots is consistently labeled "$\theta_2$", with values ranging from -10 to 10. Major tick marks are present at -10, 0, and 10.
> - **Data Representation**:
>   - In the 1D marginal plots (histograms), distributions are depicted as stepped outlines. Black outlines represent ground truth samples, while orange outlines represent posterior samples. Both distributions are visibly bimodal, with prominent peaks centered around -8 and 8.
>   - In the 2D joint plots (contour plots), distributions are represented by concentric contours. Black contours represent ground truth samples, and orange contours represent posterior samples. These plots consistently show four distinct modes (clusters) arranged roughly in a square pattern, located approximately at (-8, -8), (-8, 8), (8, -8), and (8, 8).
>
> **Specific Sub-figure Details:**
>
> - **(e) BBVI, MoG (K = 50)**:
>
>   - The black (ground truth) and orange (posterior) distributions show a very close visual alignment across all three plots. The stepped histogram shapes and the positions and densities of the concentric contours are nearly identical, indicating a strong agreement between the posterior and ground truth.
>
> - **(f) BBVI, MoG (K = 500)**:
>
>   - Similar to panel (e), the black and orange distributions are in very close agreement. The orange contours in the 2D plot appear slightly smoother and perhaps marginally more defined than in panel (e), suggesting a potentially better or higher-resolution fit.
>
> - **(g) BBVI, MoG (K = 50), noisy**:
>
>   - In this panel, the orange posterior distributions show noticeable deviations from the black ground truth.
>   - In the top $\theta_1$ marginal histogram, the orange distribution's peaks are slightly lower, and the distribution is slightly higher in the middle region compared to the black ground truth.
>   - In the right $\theta_2$ marginal histogram, the orange distribution exhibits a more pronounced peak around 0, and its peaks at -8 and 8 are slightly lower and broader than the black ground truth.
>   - In the 2D contour plot, the orange contours are visibly less sharp and appear more diffuse or spread out compared to the tightly clustered black contours, especially around the four modes.
>
> - **(h) BBVI, MoG (K = 500), noisy**:
>   - The orange posterior distributions in this panel show improved alignment with the black ground truth compared to panel (g), although some differences persist.
>   - The 1D histograms (top for $\theta_1$ and right for $\theta_2$) show the orange distributions more closely matching the black ones than in panel (g), with less pronounced discrepancies in peak heights and widths.
>   - In the 2D contour plot, the orange contours are more defined and closer to the black ground truth contours than in panel (g), indicating a better approximation of the four modes, though they still appear slightly less sharp than the black contours.

Figure A.3: GMM $(D=2)$ example posterior visualisation. Orange contours and points represent posterior samples obtained from different algorithms, while the black contours and points represent ground truth samples.

---

#### Page 33

> **Image description.** This image presents a 2x2 grid of four multi-panel plots, labeled (a), (b), (c), and (d), each illustrating distributions of parameters $\theta_1$ and $\theta_2$ under different conditions. Each of the four main panels is a corner plot, containing three sub-plots arranged in an L-shape: a 1D marginal distribution for $\theta_1$ at the top, a 2D joint distribution for $\theta_1$ and $\theta_2$ at the bottom-left, and a 1D marginal distribution for $\theta_2$ at the bottom-right. The top-right position in each corner plot is empty.
>
> Common elements across all panels:
>
> - The horizontal axis for the top-middle and bottom-left plots is labeled "$\theta_1$", with tick marks at -6, 0, and 6.
> - The vertical axis for the bottom-left and bottom-right plots is labeled "$\theta_2$", with tick marks at -6 and 0.
> - In each sub-plot, two distributions are shown: one with a dark grey/black outline and another with an orange outline. The dark grey/black lines generally appear smoother, while the orange lines are typically step-like, resembling histograms.
> - The 2D joint distribution plots consistently feature a large white circular region in the center, surrounded by a ring-like distribution.
>
> Detailed description of each panel:
>
> **Panel (a) VBMC:**
>
> - **Top-middle plot ($\theta_1$):** Shows a U-shaped distribution. The dark grey/black outline forms a smooth U-shape, high at the ends (around $\theta_1 = -6$ and $6$) and low in the middle (around $\theta_1 = 0$). The orange distribution closely follows this U-shape but appears as a step-like histogram, slightly lower in magnitude than the dark grey/black distribution.
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. A prominent dark grey/black ring encircles the central white area, representing the joint distribution. Inside this dark grey/black ring, a fainter, step-like orange contour forms a similar ring, indicating the estimated distribution. The orange contours are more granular.
> - **Bottom-right plot ($\theta_2$):** Similar to the $\theta_1$ plot, it displays a U-shaped distribution. The dark grey/black outline is smooth and U-shaped, while the orange distribution is a step-like histogram, closely mirroring the dark grey/black shape but slightly lower.
> - **Label:** (a) VBMC
>
> **Panel (b) S-VBMC (20 runs):**
>
> - **Top-middle plot ($\theta_1$):** Shows a U-shaped distribution. The dark grey/black outline is a smooth U-shape. The orange distribution is a step-like histogram that also forms a U-shape, but its peaks at the ends appear slightly higher and more pronounced compared to panel (a), suggesting a tighter distribution at the extremes.
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. The dark grey/black ring is present, similar to panel (a). However, the orange contours within this ring are more distinct and appear spiky or "starburst-like", suggesting multiple modes or a more complex, less smooth estimation of the circular distribution compared to panel (a).
> - **Bottom-right plot ($\theta_2$):** Displays a U-shaped distribution. The dark grey/black outline is smooth. The orange distribution is a step-like histogram, with peaks at the ends that are slightly higher and more pronounced than in panel (a).
> - **Label:** (b) S-VBMC (20 runs)
>
> **Panel (c) VBMC (noisy):**
>
> - **Top-middle plot ($\theta_1$):** The dark grey/black outline maintains its U-shape. In contrast to panels (a) and (b), the orange distribution here shows a distinct M-shape, with two prominent peaks around $\theta_1 = -3$ and $\theta_1 = 3$, and a clear dip in the middle (around $\theta_1 = 0$). This indicates a bimodal distribution for $\theta_1$.
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. The dark grey/black ring is visible. The orange contours form a ring, but they appear less uniform and more "patchy" or "noisy" compared to panel (a), with some areas of higher density.
> - **Bottom-right plot ($\theta_2$):** The dark grey/black outline is U-shaped. The orange distribution is a step-like histogram, forming a U-shape similar to panel (a), but possibly with slightly more jaggedness.
> - **Label:** (c) VBMC (noisy)
>
> **Panel (d) S-VBMC (20 runs, noisy):**
>
> - **Top-middle plot ($\theta_1$):** The dark grey/black outline is U-shaped. The orange distribution is a step-like histogram forming a U-shape, similar to panel (b), but potentially with more jaggedness or slightly higher peaks, reflecting the "noisy" context. It does not show the M-shape seen in panel (c).
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. The dark grey/black ring is present. The orange contours within this ring are very distinct and exhibit a pronounced "starburst" or "spiky" pattern, similar to panel (b) but perhaps even more accentuated, suggesting a highly variable or multimodal estimation of the circular distribution under noisy conditions.
> - **Bottom-right plot ($\theta_2$):** The dark grey/black outline is U-shaped. The orange distribution is a step-like histogram, forming a U-shape similar to panel (b), but potentially with more jaggedness or slightly higher peaks.
> - **Label:** (d) S-VBMC (20 runs, noisy)

0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0

---

#### Page 34

> **Image description.** The image displays a 2x2 grid of four multi-panel plots, labeled (e), (f), (g), and (h), each illustrating a "Ring (D=2) example posterior visualisation." Each of the four main panels contains a 2x2 sub-grid of plots, where the top-right sub-plot is intentionally left blank. The remaining three sub-plots within each panel are: a 2D contour plot in the bottom-left, a 1D marginal distribution plot (step plot) for $\theta_1$ in the top-left, and a 1D marginal distribution plot (step plot) for $\theta_2$ in the bottom-right.
>
> Common elements across all panels:
>
> - The x-axis for the 2D contour plot and the 1D $\theta_1$ plot is labeled $\theta_1$, with tick marks at -6, 0, and 6.
> - The y-axis for the 2D contour plot and the 1D $\theta_2$ plot is labeled $\theta_2$, with tick marks at -6, 0, and 6.
> - In the 2D contour plots, a broad ring-shaped distribution is indicated by grey contour lines. Overlaid on this, a second distribution is shown with orange/brown contour lines.
> - In the 1D marginal distribution plots, a black step plot represents one distribution, and an orange step plot represents another. The black step plots consistently show a bimodal distribution with peaks around -6 and 6 for both $\theta_1$ and $\theta_2$.
>
> Detailed description of each panel:
>
> **Panel (e): BBVI, MoG ($K=50$)**
>
> - **2D Contour Plot (bottom-left):** The grey contours form a distinct ring centered at (0,0). The orange/brown contours are concentrated within the left half of this ring, showing a somewhat diffuse distribution with higher density around $\theta_1 \approx -3$ and $\theta_2 \approx 0$.
> - **1D $\theta_1$ Plot (top-left):** The black step plot shows two peaks at approximately -6 and 6. The orange step plot shows a single, broader peak centered around -3, gradually decreasing towards 6.
> - **1D $\theta_2$ Plot (bottom-right):** The black step plot shows two peaks at approximately -6 and 6. The orange step plot shows a single, relatively broad peak centered around 0.
>
> **Panel (f): BBVI, MoG ($K=500$)**
>
> - **2D Contour Plot (bottom-left):** The grey contours again form a clear ring. The orange/brown contours are much more tightly concentrated than in (e), forming a sharp, distinct peak within the top-left quadrant of the ring, specifically around $\theta_1 \approx -3$ and $\theta_2 \approx 3$.
> - **1D $\theta_1$ Plot (top-left):** The black step plot has peaks at -6 and 6. The orange step plot shows a very sharp and tall peak centered precisely at -3, with very low values elsewhere.
> - **1D $\theta_2$ Plot (bottom-right):** The black step plot has peaks at -6 and 6. The orange step plot shows a very sharp and tall peak centered precisely at 3, with very low values elsewhere.
>
> **Panel (g): BBVI, MoG ($K=50$), noisy**
>
> - **2D Contour Plot (bottom-left):** The grey contours form the characteristic ring. The orange/brown contours are more spread out and irregular compared to (e), still largely concentrated on the left side of the ring, but with a less defined central peak and more scattered smaller concentrations.
> - **1D $\theta_1$ Plot (top-left):** The black step plot has peaks at -6 and 6. The orange step plot is irregular, showing a primary peak around -3 but also significant values and smaller peaks across the range, including around 3.
> - **1D $\theta_2$ Plot (bottom-right):** The black step plot has peaks at -6 and 6. The orange step plot is also irregular, showing a primary peak around 0 but with noticeable activity and smaller peaks around -3 and 3.
>
> **Panel (h): BBVI, MoG ($K=500$), noisy**
>
> - **2D Contour Plot (bottom-left):** The grey contours form the ring. The orange/brown contours are extremely concentrated, forming a very sharp and tall peak near the top center of the ring, specifically around $\theta_1 \approx 0$ and $\theta_2 \approx 6$.
> - **1D $\theta_1$ Plot (top-left):** The black step plot has peaks at -6 and 6. The orange step plot shows an exceptionally sharp and tall peak precisely at 0, with minimal values elsewhere.
> - **1D $\theta_2$ Plot (bottom-right):** The black step plot has peaks at -6 and 6. The orange step plot shows an exceptionally sharp and tall peak precisely at 6, with minimal values elsewhere.

Figure A.4: Ring $(D=2)$ example posterior visualisation. See the caption of Figure A. 3 for further details.

---

#### Page 35

> **Image description.** A two-panel figure displaying two corner plots, each illustrating the marginal and joint posterior distributions for five parameters, labeled theta_1 through theta_5. Both panels are structured identically, featuring a lower triangular matrix of subplots with 1D histograms on the diagonal and 2D density plots on the off-diagonal.
>
> **Panel (a): VBMC**
> This panel is labeled "(a) VBMC" at the bottom center.
>
> - **Diagonal Plots (1D Histograms):** These plots show the marginal distributions for each parameter. Each histogram displays two distributions: one outlined in black and another filled in orange. The orange and black distributions are nearly identical, showing strong overlap.
>   - The histogram for theta_1 (top-left) is unimodal and bell-shaped, centered approximately between 35 and 40.
>   - The histogram for theta_2 is highly skewed to the right, peaking at a very small value (around 1e-5 to 2e-5) and extending towards 8e-5.
>   - The histogram for theta_3 is unimodal and bell-shaped, centered around 2.8 to 3.0.
>   - The histogram for theta_4 is highly skewed to the right, peaking near 0.00 and extending towards 0.08.
>   - The histogram for theta_5 (bottom-right) is unimodal and bell-shaped, centered around -64.5.
> - **Off-Diagonal Plots (2D Density Plots):** These plots show the joint distributions between pairs of parameters. Each plot contains a scatter of faint gray points, overlaid with black contour lines and orange-filled contours, indicating regions of higher density.
>   - For example, the plot of theta_2 vs theta_1 shows an elongated, slightly curved distribution, with the highest density around theta_1 values of 35-40 and theta_2 values around 2e-5.
>   - The plot of theta_3 vs theta_1 shows an elliptical distribution, centered around (theta_1 ~ 35-40, theta_3 ~ 2.8-3.0).
>   - The plot of theta_4 vs theta_2 shows a distribution heavily concentrated near the origin (0, 0), with a tail extending along the theta_2 axis.
>   - The plot of theta_5 vs theta_1 shows a circular distribution, centered around (theta_1 ~ 35-40, theta_5 ~ -64.5).
> - **Axes Labels:** The y-axis labels for the rows are theta_2, theta_3, theta_4, and theta_5, from top to bottom. The x-axis labels for the columns are theta_1, theta_2, theta_3, and theta_4, from left to right. The x-axis for the bottom-right diagonal plot is labeled theta_5.
>
> **Panel (b): S-VBMC (20 runs)**
> This panel is labeled "(b) S-VBMC (20 runs)" at the bottom center.
>
> - The visual content of this panel is almost identical to Panel (a). The shapes, peak locations, and spread of all 1D histograms and 2D density plots appear to be consistent with those in Panel (a). The color scheme (black outlines, orange fills, gray scatter points) and contour lines are also the same.
> - The distributions for each parameter and parameter pair show the same patterns as described for Panel (a), indicating a high degree of similarity in the inferred distributions between "VBMC" and "S-VBMC (20 runs)".
> - **Axes Labels:** The axes are labeled identically to Panel (a).
>   (b) S-VBMC (20 runs)

---

#### Page 36

> **Image description.** A two-panel figure, labeled (c) and (d), each displaying a 5x5 lower-triangular matrix of plots, commonly known as a corner plot or matrix plot. These plots visualize the marginal and joint posterior distributions of five parameters, denoted as θ1, θ2, θ3, θ4, and θ5. Each panel compares two distributions: one outlined in black (with gray shading for 2D plots) and another outlined in orange (with orange shading and dashed contours for 2D plots).
>
> **Panel (c):**
> This panel is labeled "(c) BBVI, MoG (K = 50)" below the plot matrix.
> The plot matrix is arranged with parameters θ1, θ2, θ3, θ4, and θ5 labeling the horizontal axes of the columns (from left to right) and the vertical axes of the rows (from bottom to top).
>
> - **Diagonal Plots (1D Marginal Distributions):** The plots along the diagonal display one-dimensional histograms for each parameter. Each histogram shows two distributions: one with a solid black outline and another with a solid orange outline.
>   - For θ1 (top-left), the black distribution is a tall, narrow, unimodal histogram centered around 35-40. The orange distribution is wider and flatter, shifted slightly to the right.
>   - For θ2, the black distribution is a very narrow, tall peak near the left edge, while the orange is a broader, shorter peak further to the right.
>   - For θ3, both black and orange distributions are unimodal, with the black being taller and narrower, centered around 3.0, and the orange being wider and slightly shifted.
>   - For θ4, the black distribution is a very narrow, tall peak at the left edge (near 0.00), and the orange is a broader, shorter peak slightly to the right.
>   - For θ5 (bottom-right), the black distribution is a tall, narrow, unimodal histogram centered around -64.5, and the orange is a wider, flatter distribution shifted to the right.
> - **Off-Diagonal Plots (2D Joint Distributions):** The plots in the lower triangle of the matrix display two-dimensional joint distributions for pairs of parameters.
>   - One distribution is represented by solid black contour lines and gray shading, indicating regions of higher density.
>   - The second distribution is represented by dashed orange contour lines and orange shading.
>   - Faint gray dots are visible in the background of these 2D plots, likely representing underlying samples.
>   - The shapes of these 2D contours vary, showing different correlations and dependencies between parameter pairs. For example, the plot for (θ2, θ1) shows a distinct, elongated black contour with a separate, smaller orange cluster. The plot for (θ3, θ1) shows concentric elliptical black contours with an overlapping, slightly shifted orange distribution.
> - **Axis Labels and Ticks:**
>   - θ1: horizontal axis ticks at approximately 30, 45.
>   - θ2: horizontal axis ticks at approximately 1e-5, 4, 8.
>   - θ3: horizontal axis ticks at approximately 2.5, 3.0, 3.5.
>   - θ4: horizontal axis ticks at approximately 0.00, 0.04, 0.08.
>   - θ5: horizontal axis ticks at approximately -64.8, -64.5, -64.2.
>   - Vertical axis ticks for each row (e.g., θ2, θ3, θ4, θ5) show corresponding numerical values, generally increasing upwards.
>
> **Panel (d):**
> This panel is labeled "(d) BBVI, MoG (K = 500)" below the plot matrix.
> The visual structure and content of this panel are strikingly similar to panel (c). It also consists of a 5x5 lower-triangular matrix of 1D histograms on the diagonal and 2D contour plots off-diagonal, comparing the same two distributions (black/gray and orange/dashed orange).
>
> - **Comparison with Panel (c):** While the overall patterns are consistent, subtle differences can be observed. The orange distributions in panel (d) (K=500) generally appear to align more closely with the black distributions compared to panel (c) (K=50). For instance, the orange 1D histogram for θ1 in panel (d) is slightly taller and better centered, more closely resembling the black distribution than its counterpart in panel (c). Similar subtle improvements in alignment or concentration of the orange distributions relative to the black ones can be observed across several plots, suggesting a potentially better approximation with K=500.
> - **Axis Labels and Ticks:** The axis labels and tick values are identical to those in panel (c).
>
> In both panels, the black distributions generally appear more concentrated and unimodal, while the orange distributions are often broader, sometimes shifted, or show slightly different modes, particularly in panel (c).

Figure A.5: Neuronal model $(D=5)$ example posterior visualisation. See the caption of Figure A. 3 for further details.

---

#### Page 37

> **Image description.** This image displays two multi-panel corner plots, labeled (a) and (b), arranged vertically. Each corner plot is a square matrix showing the marginal and joint posterior distributions for six parameters, $\theta_1$ through $\theta_6$. The plots are designed to compare two different distributions, represented by black and orange outlines/fills.
>
> **Panel (a): VBMC**
> This panel presents a 6x6 matrix of plots, where the diagonal elements show one-dimensional marginal distributions (histograms), and the off-diagonal elements in the lower triangle show two-dimensional joint distributions (scatter plots with contour lines). The upper triangle of the matrix is empty.
>
> - **Diagonal Plots (1D Marginal Distributions)**: These are histograms for each parameter ($\theta_1$ to $\theta_6$). Each histogram displays two distributions: one outlined in black and one outlined in orange. The orange distribution often appears slightly filled or more prominent.
>   - For $\theta_1$, the distribution is bimodal, with the orange distribution slightly shifted to the right compared to the black one.
>   - For $\theta_2$, $\theta_3$, $\theta_5$, and $\theta_6$, the distributions are unimodal and bell-shaped, with the black and orange histograms largely overlapping, indicating similar distributions.
>   - For $\theta_4$, the distribution is skewed right, and the orange distribution is slightly shifted to the right of the black one.
>   - The x-axis for these plots corresponds to the parameter value, and the y-axis represents density.
> - **Off-Diagonal Plots (2D Joint Distributions)**: These plots show the relationship between pairs of parameters. Each plot contains a dense cloud of light gray data points. Overlaid on these points are contour lines: solid black lines, solid orange-filled contours, and a dashed orange line. The orange-filled contours typically represent the central tendency or higher density regions of one distribution, while the black lines represent another.
>   - The contours generally show elliptical or elongated shapes, indicating various degrees of correlation between the parameters. For instance, the plot for $\theta_2$ vs $\theta_1$ shows a positive correlation, while $\theta_3$ vs $\theta_1$ shows a negative correlation.
>   - The x-axis for each column of off-diagonal plots is labeled at the bottom of the matrix (from $\theta_1$ to $\theta_5$), and the y-axis for each row is labeled on the left side of the matrix (from $\theta_2$ to $\theta_6$).
> - **Axis Labels and Ticks**: Numerical tick marks and labels are present on the outer edges of the matrix, indicating the ranges for each parameter (e.g., $\theta_1$ from approximately 5 to 10, $\theta_2$ from 12 to 18, $\theta_3$ from 16 to 32, $\theta_4$ from 5 to 10, $\theta_5$ from 0.1 to 0.2, $\theta_6$ from 18 to 24).
> - **Panel Label**: Below this matrix, the text "(a) VBMC" is displayed.
>
> **Panel (b): S-VBMC (20 runs)**
> This panel has an identical structure and layout to panel (a), also presenting a 6x6 corner plot for the same six parameters, $\theta_1$ through $\theta_6$.
>
> - **Diagonal Plots (1D Marginal Distributions)**: Similar to panel (a), these histograms compare black and orange distributions. The visual characteristics of the distributions (e.g., bimodality for $\theta_1$, unimodality for others, skewness for $\theta_4$) and the relative positions of the black and orange distributions are very similar to those observed in panel (a).
> - **Off-Diagonal Plots (2D Joint Distributions)**: These plots also show light gray scatter points with overlaid black contour lines, orange-filled contours, and dashed orange lines. The shapes, orientations, and densities of the contours, as well as the implied correlations between parameters, closely resemble those in panel (a).
> - **Axis Labels and Ticks**: The axis labels ($\theta_1$ to $\theta_6$) and numerical tick marks are consistent with panel (a).
> - **Panel Label**: Below this matrix, the text "(b) S-VBMC (20 runs)" is displayed.
>
> **Overall Comparison**:
> Both panels visually represent very similar sets of distributions. The black distributions (both 1D histograms and 2D contours) generally align well with the orange distributions, suggesting that the methods represented by "VBMC" and "S-VBMC (20 runs)" produce comparable results in estimating the underlying distributions. The orange dashed lines in the 2D plots consistently follow the shape of the orange-filled contours.
> (b) S-VBMC (20 runs)

---

#### Page 38

> **Image description.** This image displays two separate corner plots, labeled (c) and (d), each presenting a matrix of one-dimensional histograms and two-dimensional kernel density estimates for six parameters, $\theta_1$ through $\theta_6$. Both plots share a similar structure and color scheme, comparing two different distributions.
>
> **Panel (c): BBVI, MoG (K = 50)**
> This panel is a 6x6 grid of subplots, with the upper right triangle empty.
>
> - **Diagonal Subplots (1D Histograms):** These plots show the marginal distributions for each parameter. Each histogram displays two distributions: one represented by a solid orange stepped line and another by a solid black stepped line. The orange distribution generally appears narrower and taller, indicating a more concentrated probability mass, while the black distribution is broader. For example, the histogram for $\theta_1$ (top-left) shows both distributions peaking around 4-5, with the orange distribution being notably higher and narrower. The x-axis labels for these diagonal plots are $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6$ from left to right.
> - **Off-Diagonal Subplots (2D Density Plots):** These plots show the joint distributions between pairs of parameters. Each plot features two sets of contours and shading:
>   - One distribution is represented by solid black contour lines and light grey shading, indicating broader, less concentrated areas.
>   - The second distribution is represented by dashed orange contour lines and solid orange shading, typically appearing more concentrated and often nested within the black contours. The orange shading is generally darker and more centrally located, suggesting higher density.
>   - The y-axes for the rows are labeled $\theta_2, \theta_3, \theta_4, \theta_5, \theta_6$ from top to bottom. The x-axes for the columns are labeled $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5$ from left to right.
>   - The shapes of these 2D densities vary, ranging from somewhat circular to elongated ellipses, indicating different correlations between the parameters. For instance, the plot of $\theta_2$ vs $\theta_1$ shows an elongated, positively correlated distribution.
>
> **Panel (d): BBVI, MoG (K = 500)**
> This panel has the identical structure and parameter labels as panel (c).
>
> - **Diagonal Subplots (1D Histograms):** Similar to panel (c), these plots show two distributions in orange and black stepped lines. However, in panel (d), the orange distributions generally appear even narrower and taller, and they show a closer alignment or overlap with the black distributions compared to panel (c). This suggests a more precise or converged estimate for the orange distribution.
> - **Off-Diagonal Subplots (2D Density Plots):** These plots also show two distributions with black solid contours/grey shading and orange dashed contours/orange shading. A key visual difference from panel (c) is that the orange contours and shading in panel (d) are noticeably more concentrated and tightly nested within the black contours. The orange regions appear smaller and more intense, indicating a tighter joint distribution that aligns more closely with the center of the broader black/grey distribution. The shapes and orientations of the densities are similar to panel (c) but with the orange distributions showing less spread.
>
> In summary, both panels are corner plots comparing two distributions across six parameters. Panel (d) visually suggests a more concentrated and potentially better-aligned approximation (represented in orange) compared to panel (c), where the orange distributions are somewhat broader and less perfectly aligned with the black/grey distributions.

Figure A.6: Multisensory model $(D=6, \sigma=3)$ example posterior visualisation. See the caption of Figure A. 3 for further details.
