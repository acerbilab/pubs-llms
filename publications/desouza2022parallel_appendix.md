# Parallel MCMC Without Embarrassing Failures - Appendix

---

#### Page 12

# Supplementary Material: Parallel MCMC Without Embarrassing Failures 

In this Supplement, we include extended explanations, implementation details, and additional results omitted from the main text.

Code for our algorithm and to generate the results in the paper is available at: https://github.com/ spectraldani/pai.

## A Failure modes of embarrassingly parallel MCMC explained

In Section 2.1 of the main text we presented three major failure modes of embarrassingly parallel MCMC (Markov Chain Monte Carlo) methods. These failure modes are illustrated in Fig 1 in the main text. In this section, we further explain these failure types going through Fig 1 in detail.

## A. 1 Mode collapse (Fig 1A)

Fig 1A illustrates mode collapse. In this example, the true subposteriors $p_{1}$ and $p_{2}$ both have two modes (see Fig 1A, top two panels). However, while in $p_{1}$ the two modes are relatively close to each other, in $p_{2}$ they are farther apart. Thus, when we run an MCMC chain on $p_{2}$, it gets stuck into a single high-density region and is unable to jump to the other mode ('unexplored mode', shaded area). This poor exploration of $p_{2}$ is fatal to any standard combination strategy used in parallel MCMC, erasing entire regions of the posterior (Fig 1A, bottom panel). While in this example we use PART (Wang et al., 2015), Section 4.1 shows this common pathology in several other methods as well. Our proposed solution to this failure type is sample sharing (see Section C.2).

## A. 2 Model mismatch (Fig 1B)

Fig 1B draws attention to model mismatch in subposterior surrogates. In this example, MCMC runs smoothly on both subposteriors. However, when we fit regression-based surrogates - here, Gaussian processes (GPs) to the MCMC samples, the behavior of these models away from subposterior samples can be unpredictable. In our example, the surrogate $q_{2}$ hallucinates a mode that does not exist in the true subposterior $p_{2}$ ('model hallucination', shaded area in Fig 1B). Our proposed solution to correct this potential issue is to explore uncertain areas of the surrogate using active subposterior sampling (see Section C.3).

In this example, we used a simple GP surrogate approach for illustration purposes. The more sophisticated GP-based Distributed Importance Sampler (GP-DIS; Nemeth and Sherlock (2018)) might seem to provide an alternative solution to model hallucination, in that the hallucinated regions with low true density would be down-weighted by importance sampling. However, the DIS step only works if there are 'good' samples that cover regions with high true density that can be up-weighted. If no such samples are present, importance sampling will not work. As an example of this failure, Fig 3 in the main text shows that GP-DIS (Nemeth and Sherlock, 2018) does not recover from the model hallucination (if anything, the importance sampling step concentrates the hallucinated posterior even more).

## A. 3 Underrepresented tails (Fig 1C)

Fig 1C shows how neglecting low-density regions (underrepresented tails) can affect the performance of parallel MCMC. In the example, the true subposterior $p_{2}$ has long tails that are not thoroughly represented by MCMC samples. This under-representation is due both to actual difficulty of the MCMC sampler in exploring the tails, and to mere Monte Carlo noise as there is little (although non-zero) mass in the tails, so the number of samples is low. Consequently, the surrogate $q_{2}$ is likely to underestimate the density in this unexplored region, in which the other subposterior $p_{1}$ has considerable mass. In this case, even though $q_{1}$ is a perfect fit to $p_{1}$, the product $q_{1}(\theta) q_{2}(\theta)$ mistakenly attributes near-zero density to the combined posterior in said region (Fig 1C, bottom panel). In our method, we address this issue via multiple solutions, in that both sample sharing and active sampling would help uncover underrepresentation of the tails in relevant regions.

---

#### Page 13

For further illustration of the effectiveness of our proposed solutions to these failure modes, see Section D.1.

# B Gaussian processes 

Gaussian processes (GPs) are a flexible class of statistical models for specifying prior distributions over unknown functions $f: \mathcal{X} \subseteq \mathbb{R}^{D} \rightarrow \mathbb{R}$ (Rasmussen and Williams, 2006). In this section, we describe the GP model used in the paper and details of the training procedure.

## B. 1 Gaussian process model

In the paper, we use GPs as surrogate models for log-posteriors (and log-subposteriors), for which we use the following model. We recall that GPs are defined by a positive definite covariance, or kernel function $\kappa: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$; a mean function $m: \mathcal{X} \rightarrow \mathbb{R}$; and a likelihood or observation model.

Kernel function. For simplicity, we use the common and equivalently-named squared exponential, Gaussian, or exponentiated quadratic kernel,

$$
\kappa\left(x, x^{\prime} ; \sigma_{f}^{2}, \ell_{1}, \ldots, \ell_{D}\right)=\sigma_{f}^{2} \exp \left[-\frac{1}{2}\left(x-x^{\prime}\right) \boldsymbol{\Sigma}_{\ell}^{-1}\left(x-x^{\prime}\right)^{\top}\right] \quad \text { with } \boldsymbol{\Sigma}_{\ell}=\operatorname{diag}\left[\ell_{1}^{2}, \ldots, \ell_{D}^{2}\right]
$$

where $\sigma_{f}$ is the output length scale and $\left(\ell_{1}, \ldots, \ell_{D}\right)$ is the vector of input length scales. Our algorithm does not hinge on choosing this specific kernel and more appropriate kernels might be used depending on the application (e.g., the spectral mixture kernel might provide more flexibility; see Wilson and Adams, 2013).

Mean function. When using GPs as surrogate models for log-posterior distributions, it is common to assume a negative quadratic mean function such that the surrogate posterior (i.e., the exponentiated surrogate log-posterior) is integrable (Nemeth and Sherlock, 2018; Acerbi, 2018, 2019, 2020). A negative quadratic can be interpreted as a prior assumption that the target posterior is a multivariate normal; but note that the GP can model deviations from this assumption and represent multimodal distributions as well (see for example Fig 3 in the main text). In this paper, we use

$$
m\left(x ; m_{0}, \mu_{1}, \ldots, \mu_{D}, \omega_{1}, \ldots, \omega_{D}\right) \equiv m_{0}-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(x_{i}-\mu_{i}\right)^{2}}{\omega_{i}^{2}}
$$

where $m_{0}$ denotes the maximum, $\left(\mu_{1}, \ldots, \mu_{D}\right)$ is the location vector, and $\left(\omega_{1}, \ldots, \omega_{D}\right)$ is a vector of length scales.
Observation model. Finally, GPs are also characterized by a likelihood or observation noise model. Throughout the paper we assume exact observations of the target log-posterior (or log-subposterior) so we use a Gaussian likelihood with a small variance $\sigma^{2}=10^{-3}$ for numerical stability.

## B. 2 Gaussian process inference and training

Inference. Conditioned on training inputs $\mathbf{X}=\left\{x_{1}, \ldots, x_{N}\right\}$, observed function values $\mathbf{y}=f(\mathbf{X})$ and GP hyperparameters $\psi$, the posterior GP mean and covariance are available in closed form (Rasmussen and Williams, 2006),

$$
\begin{aligned}
\bar{f}_{\mathbf{X}, \mathbf{y}}(x) \equiv \mathbb{E}[f(x) \mid \mathbf{X}, \mathbf{y}, \psi] & =\kappa(x, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma^{2} \mathbb{I}_{D}\right]^{-1}(\mathbf{y}-m(\mathbf{X}))+m(x) \\
C_{\mathbf{X}, \mathbf{y}}\left(x, x^{\prime}\right) \equiv \operatorname{Cov}\left[f(x), f\left(x^{\prime}\right) \mid \mathbf{X}, \mathbf{y}, \psi\right] & =\kappa\left(x, x^{\prime}\right)-\kappa(x, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})++\sigma^{2} \mathbb{I}_{D}\right]^{-1} \kappa\left(\mathbf{X}, x^{\prime}\right)
\end{aligned}
$$

where $\psi$ is a hyperparameter vector for the GP mean, covariance, and likelihood (see Section B. 1 above); and $\mathbb{I}_{D}$ is the identity matrix in $D$ dimensions.

Training. Training a GP means finding the hyperparameter vector(s) that best represent a given dataset of input points and function observations $(\mathbf{X}, \mathbf{y})$. In this paper, we train all GP models by maximizing the log marginal likelihood of the GP plus a log-prior term that acts as regularizer, a procedure known as maximum-a-posteriori estimation. Thus, the training objective to maximize is

$$
\begin{aligned}
\log p(\psi \mid \mathbf{X}, \mathbf{y})= & -\frac{1}{2}(\mathbf{y}-m(\mathbf{X} ; \psi))^{\top}\left[\kappa(\mathbf{X}, \mathbf{X} ; \psi)+\sigma^{2} \mathbb{I}_{D}\right]^{-1}(\mathbf{y}-m(\mathbf{X} ; \psi)) \\
& +\frac{1}{2} \log \operatorname{det}\left(\kappa(\mathbf{X}, \mathbf{X} ; \psi)+\sigma^{2} \mathbb{I}_{D}\right)+\log p(\psi)+\text { const }
\end{aligned}
$$

where $p(\psi)$ is the prior over GP hyperparameters, described below.

---

#### Page 14

| Hyperparameter | Description | Prior distribution |
| :-- | :-- | :-- |
| $\log \sigma_{f}^{2}$ | Output scale | - |
| $\log \ell^{(i)}$ | Input length scale | $\log \mathcal{N}\left(\log \sqrt{\frac{D}{6}} L^{(i)}, \log \sqrt{10^{3}}\right)$ |
| $m_{0}$ | Mean function maximum | SmoothBox $\left(y_{\min }, y_{\max }, 1.0\right)$ |
| $x_{m}^{(i)}$ | Mean function location | SmoothBox $\left(B_{\min }^{(i)}, B_{\max }^{(i)}, 0.01\right)$ |
| $\log \omega^{(i)}$ | Mean function scale | $\log \mathcal{N}\left(\log \sqrt{\frac{D}{6}} L^{(i)}, \log \sqrt{10^{3}}\right)$ |

Table S1: Priors over GP hyperparameters. See text for more details

Priors. We report the prior $p(\psi)$ over GP hyperparameters in Table S1, assuming independent priors over each hyperparameter and dimension $1 \leq i \leq D$. We set the priors based on broad characteristics of the training set, an approach known as empirical Bayes which can be seen as an approximation to a hierarchical Bayesian model. In the table, $\mathbf{B}$ is the 'bounding box' defined as the $D$-dimensional box including all the samples observed by the GP so far plus a $10 \%$ margin; $\mathbf{L}=\mathbf{B}_{\max }-\mathbf{B}_{\min }$ is the vector of lengths of the bounding box; and $y_{\max }$ and $y_{\min }$ are, respectively, the largest and smallest observed function values of the GP training set. $\log \mathcal{N}(\mu, \sigma)$ denotes the log-normal distribution and SmoothBox $(a, b, \sigma)$ is defined as a uniform distribution on the interval $[a, b]$ with a Gaussian tail with standard deviation $\sigma$ outside the interval. If a distribution is not specified, we assumed a flat prior.

# B. 3 Implementation details 

All GP models in the paper are implemented using GPyTorch ${ }^{1}$ (Gardner et al., 2018), a modern package for GP modeling based on the PyTorch machine learning framework (Paszke et al., 2019). For maximal accuracy, we performed GP computations enforcing exact inference via Cholesky decomposition, as opposed to the asymptotically faster conjugate gradient implementation which however might not reliably converge to the exact solution (Maddox et al., 2021). For parts related to active sampling, we used the BoTorch ${ }^{2}$ package for active learning with GPs (Balandat et al., 2020), implementing the acquisition functions used in the paper as needed.

We used the same GP model and training procedure described in this section for all GP-based methods in the paper, which include our proposed approach and others (e.g., Nemeth and Sherlock, 2018).

## C Algorithm details

```
Algorithm S1 Parallel Active Inference (PAI)
Input: Data partitions \(\mathcal{D}_{1}, \ldots, \mathcal{D}_{K}\); prior \(p(\theta)\); likelihood function \(p(\mathcal{D} \mid \theta)\).
    \mathrm{parfor} 1 \ldotsK\) do
    \(\triangleright\) Parallel steps
        \(\mathcal{S}_{k} \leftarrow \mathrm{MCMC}\) samples from \(p_{k}(\theta) \propto p(\theta)^{1 / K} p\left(\mathcal{D}_{k} \mid \theta\right)\)
        \(\mathcal{S}_{k}^{\prime} \leftarrow \operatorname{ActiveSubSAMPLE}\left(\mathcal{S}_{k}\right)\)
        send \(\mathcal{S}_{k}^{\prime}\) to all other nodes, receive \(\mathcal{S}_{\backslash k}^{\prime}=\bigcup_{j \neq k} \mathcal{S}_{j}^{\prime}\)
        \(\mathcal{S}_{k}^{\prime \prime} \leftarrow \mathcal{S}_{k}^{\prime} \cup\) SelectSharedSamples \(\left(\mathcal{S}_{k}^{\prime}, \mathcal{S}_{\backslash k}^{\prime}\right)\)
        \(\mathcal{S}_{k}^{\prime \prime \prime} \leftarrow \mathcal{S}_{k}^{\prime \prime} \cup\) ActiveRefinement \(\left(\mathcal{S}_{k}^{\prime \prime}\right)\)
        train GP model \(\mathcal{L}_{k}\) of the \(\log\) subposterior on \(\left(\mathcal{S}_{k}^{\prime \prime \prime}, \log p_{k}\left(\mathcal{S}_{k}^{\prime \prime \prime}\right)\right)\)
    end parfor
    combine subposteriors: \(\log q(\theta)=\sum_{k=1}^{K} \mathcal{L}_{k}(\theta)\)
    \(\triangleright\) Centralized step, see Section 3.4
    Optional: refine \(\log q(\theta)\) with Distributed Importance Sampling (DIS)
    \(\triangleright\) See Section 2.2
```

In this section, we describe additional implementation details for the steps of the Parallel Active Inference (PAI) algorithm introduced in the main text. Algorithm S1 illustrates the various steps. Each function called by the algorithm may involve several sub-steps (e.g., fitting interim surrogate GP models) which are detailed in the following sections.

[^0]
[^0]:    ${ }^{1}$ https://gpytorch.ai/
    ${ }^{2}$ https://botorch.org/

---

#### Page 15

# C. 1 Subposterior modeling via GP regression 

Here we expand on Section 3.1 in the main text. We recall that at this step each node $k \in\{1, \ldots, K\}$ has run MCMC on the local subposterior $p_{k}$, obtaining a set of samples $\mathcal{S}_{k}$ and their log-subposterior values $\log p_{k}(\mathcal{S})$.
The goal now is to 'thin' the samples to a subset which is still very informative of the shape of the subposterior, so that it can be used as a training set for the GP surrogate. The rationale is that using all the samples for GP training is expensive in terms of both computational and communication costs, and it can lead to numerical instabilities. In previous work, Nemeth and Sherlock (2018) have used random thinning which is not guaranteed to keep relevant parts of the posterior. ${ }^{3}$

The main details that we cover here are: (1) how we pick the initial subset of samples $\mathcal{S}_{k}^{(0)} \subseteq \mathcal{S}_{k}$ used to train the initial GP surrogate; (2) how we subsequently perform active subsampling to expand the initial subset to include relevant points from $\mathcal{S}_{k}$.

Initial subset: To bootstrap the GP surrogate model, we use a distance clustering method to select the initial subset of samples that we use to fit a first GP model. As this initial subset will be further refined, the main characteristic for selecting the samples is for them to be spread out. For our experiments, we choose a simple $k$-medoids method (Park and Jun, 2009) with $n_{\text {med }}=20 \cdot(D+2)$ medoids implemented in the scikit-learn-extras library ${ }^{4}$. The output $n_{\text {med }}$ medoids represent $\mathcal{S}_{k}^{(0)}$.

Active subsampling: After having selected the initial subset $\mathcal{S}_{k}^{(0)}$, we perform $T$ iterations of active subsampling. In each iteration $t+1$, we greedily select a batch of $n_{\text {batch }}$ points from the set $\mathcal{S}_{k} \backslash \mathcal{S}_{k}^{(t)}$ obtained by maximizing a batch version of the maximum interquantile range (MAXIQR) acquisition function, as described by Järvenpää et al. (2021); see also main text. The MAXIQR acquisition function has a parameter $u$ which controls the tradeoff between exploitation of regions of high posterior density and exploration of regions with high posterior uncertainty in the GP surrogate. To strongly promote exploration, after a preliminary analysis on a few toy problems, we set $u=20$ throughout the experiments presented in the paper. In the paper, we set $n_{\text {batch }}=D$ and the number of iterations $T=25$, based on a rule of thumb for the total budget of samples required by similar algorithms to achieve good performance at a given dimension (e.g., Acerbi 2018; Järvenpää et al. 2021). The GP model is retrained after the acquisition of each batch. The procedure locally returns a subset of samples $\mathcal{S}_{k}^{\prime}=\mathcal{S}_{k}^{(T)}$.

## C. 2 Sample sharing

In this section, we expand on Section 3.2 of the main text. We recall that at this step node $k$ receives samples $\mathcal{S}_{\backslash k}^{\prime}=\bigcup_{j \neq k} \mathcal{S}_{j}^{\prime}$ from the other subposteriors. To avoid incorporating too many data points into the local surrogate model (for the reasons explained previously), we consider adding a data point to the current surrogate only if: (a) the local model cannot properly predict this additional point; and (b) predicting the exact value would make a difference. If the number of points that are eligible under these criteria is greater than $n_{\text {share }}$, the set is further thinned using $k$-medoids.
Concretely, let $\theta^{\star} \in \mathcal{S}_{\backslash k}^{\prime}$ be the data point under consideration. We evaluate the true subposterior log density at the point, $y^{\star}=\log p_{k}\left(\theta^{\star}\right)$, and the surrogate GP posterior latent mean and variance at the point, which are, respectively, $\mu_{\star}=\bar{f}\left(\theta_{\star}\right)$ and $\sigma_{\star}^{2}=C\left(\theta^{\star}, \theta^{\star}\right)$. We then consider two criteria:
a. First, we compute the density of the true value under the surrogate prediction and check if it is above a certain threshold: $\mathcal{N}\left(\log y^{\star} ; \mu_{\star}, \sigma_{\star}^{2}\right)>R$, where $R=0.01$ in this paper. This criterion is roughly equivalent to including a point only if $\left|\mu_{\star}-y^{\star}\right| \gtrsim R^{\prime} \sigma^{\star}$, for an appropriate choice of $R^{\prime}$ (ignoring a sublinear term in $\sigma^{\star}$ ), implying that a point is considered for addition if the GP prediction differs from the actual value more than a certain number of standard deviations.
b. Second, if a point meets the first criterion, we check if the value of the point is actually relevant for the surrogate model. Let $y_{\max }$ be the maximum subposterior log-density observed at the current node (i.e., approximately, the log-density at the mode). We exclude a point at this stage if both the GP prediction and the true value $y^{\star}$ are below the threshold $y_{\max }-20 D$, meaning that the point has very low density and the GP correctly predicts that it is very low density (although might not predict the exact value).

[^0]
[^0]:    ${ }^{3}$ Note that instead of thinning we could use sparse GP approximations (e.g., Titsias, 2009). However, the interaction between sparse GP approximations and active learning is not well-understood. Moreover, we prefer not to introduce an additional layer of approximation that could reduce the accuracy of our subposterior surrogate models.
    ${ }^{4}$ https://github.com/scikit-learn-contrib/scikit-learn-extra

---

#### Page 16

Each of the above criteria is checked for all points in $\mathcal{S}_{\backslash k}^{\prime}$ in parallel (the second criterion only for all the points that pass the first one). Note that the second criterion is optional, but in our experiments we found that it increases numerical stability of the GP model, removing points with very low (log) density that are difficult for the GP to handle (for example, Acerbi (2018); Järvenpää et al. (2021) adopt a similar strategy of discarding very low-density points). More robust GP kernels (see Section B.1) might not need this additional step.

If the number of points that pass both criteria for inclusion is larger than $n_{\text {share }}$, then $k$-medoids is run on the set of points under consideration with $n_{\text {share }}$ medoids, where $n_{\text {share }}=25 D$. The procedure run at node $k$ locally returns a subset of samples $\mathcal{S}_{k}^{\prime \prime}$.

# C. 3 Active subposterior refinement 

Here we expand on Section 3.3 of the main text. We recall that up to this point, the local GP model of the log-subposterior at node $k$ was trained only using selected subset of samples from the original MCMC runs (local and from other nodes), denoted by $\mathcal{S}_{k}^{\prime \prime}$.

In this step, each node $k$ locally acquires new points by iteratively optimizing the MAXIQR acquisition function (Eq. 2 in the main text) over a domain $\mathcal{X} \subseteq \mathbb{R}^{D}$. For the first iteration, the space $\mathcal{X}$ is defined as the bounding box of $\bigcup_{k} \mathcal{S}_{k}^{\prime}$ plus a $10 \%$ margin. In other words, $\mathcal{X}$ is initially the hypercube that contains all samples from all subposteriors obtained in the sample sharing step (Section C.2) extended by a $10 \%$ margin. The limits of $\mathcal{X}$ at the first iteration are computed during the previous stage, without any additional communication cost. In each subsequent iteration, $\mathcal{X}$ is iteratively extended to include the newly sampled points plus a $10 \%$ margin, thus expanding the bounding box if a recently acquired point falls near the boundary.

New points are selected greedily in batches of size $n_{\text {batch }}=D$, using the same batch formulation of the MAXIQR acquisition function used in Section C.1. The local GP surrogate is retrained at the end of each iteration, to ensure that the next batch of points targets the regions of the log-subposterior which are most important to further refine the surrogate. For the purpose of this work, we repeat the process for $T_{\text {active }}=25$ iterations, selected based on similar active learning algorithms (Acerbi et al., 2018; Järvenpää et al., 2021). Future work should investigate an appropriate termination rule to dynamically vary the number of iterations.

The outcome of this step is a final local set of samples $\mathcal{S}_{k}^{\prime \prime \prime}$ and the log-subposterior GP surrogate model $\mathcal{L}_{k}$ trained on these samples. Both of these are sent back to the central node for the final combination step.

## D Experiment details and additional results

In this section, we report additional results and experimental details omitted from the main text for reasons of space.

## D. 1 Ablation study

As an ablation study, Fig S1 breaks down the effect of each step of PAI in the multi-modal posterior experiment from Section 4.1 of the main text. The first panel shows the full approximate posterior if we were combining it right after active subsampling (Section C.1), using neither sample sharing nor active refinement. Note that this result suffers from Failure mode I (mode collapse; see Section A.1), as active subsampling only on the local MCMC samples is not sufficient to recover the missing modes. The second panel incorporates sample sharing, which covers the missing regions but now suffers from Failure mode II (model mismatch; see Section A.2) with an hallucinated mode in a region where the true posterior has low density. Finally, the third panel shows full-fledged PAI, which further applies active sampling to explore the hallucinated mode and corrects the density around it. The final result of PAI perfectly matches the ground truth (as displayed in the fourth panel).

## D. 2 Performance evaluation

In this section, we describe in detail the metrics used to assess the performance of the methods in the main text, how we compute these metrics, and the related statistical analyses for reporting our results.

Metrics. In the main text, we measured the quality of the posterior approximations via the mean marginal total variation distance (MMTV), 2-Wasserstein (W2) distance, and Gaussianized symmetrized Kullback-Leibler divergence (GsKL) between true and appoximate posteriors. For all metrics, lower is better. We describe the three metrics and their features below:

---

#### Page 17

> **Image description.** The image consists of four heatmaps arranged horizontally, each representing an approximate combined posterior density. The heatmaps are square and share the same axes, labeled θ1 on the x-axis and θ2 on the y-axis, both ranging from -1 to 1.
> 
> *   **Panel 1:** Titled "Active subsampling". It shows a heatmap with a pale background and four small, elongated reddish regions located near the corners of the square.
> *   **Panel 2:** Titled "Sample sharing". This heatmap has the same four reddish regions as the first panel, but it also includes a circular reddish region in the center of the square. The background color is similar to the first panel.
> *   **Panel 3:** Titled "PAI". This heatmap displays four reddish regions in the corners, similar to the first two panels. The background color is pale.
> *   **Panel 4:** Titled "Ground Truth". This heatmap shows four elongated bluish regions in the corners on a pale bluish background.
> 
> To the right of the "Ground Truth" heatmap is a vertical colorbar labeled "Log posterior density". The colorbar ranges from approximately -93 (at the bottom, corresponding to blue) to 7 (at the top, corresponding to red). The colorbar indicates the mapping between color and log posterior density values.

Figure S1: Ablation study for PAI on the multi-modal posterior. From left to right: The first panel shows the approximate combined posterior density for our method without sample sharing and active learning; the second panel uses an additional step to share samples (but no active refinement); the third shows results for full-fledged PAI. The rightmost plot is the ground truth posterior. Note that both sample sharing and active refinement are important steps for PAI: sample sharing helps account for missing posterior regions while active sampling corrects model hallucinations (see text for details).

- The MMTV quantifies the (lack of) overlap between true and approximate posterior marginals, defined as

$$
\operatorname{MMTV}(p, q)=\frac{1}{2 D} \sum_{d=1}^{D} \int_{-\infty}^{\infty}\left[p_{d}^{\mathrm{M}}\left(x_{d}\right)-q_{d}^{\mathrm{M}}\left(x_{d}\right)\right] d x_{d}
$$

where $p_{d}^{\mathrm{M}}$ and $q_{d}^{\mathrm{M}}$ denote the marginal densities of $p$ and $q$ along the $d$-th dimension. Eq. S5 has a direct interpretation in that, for example, a MMTV metric of 0.5 implies that the posterior marginals overlap by $50 \%$ (on average across dimensions). As a rule of thumb, we consider a threshold for a reasonable posterior approximation to be MMTV $<0.2$, that is more than $80 \%$ overlap.

- Wasserstein distances measure the cost of moving amounts of probability mass from one distribution to the other so that they perfectly match - a commonly-used distance metric across distributions. The W2 metric, also known as earth mover's distance, is a special case of Wasserstein distance that uses the Euclidean distance as its cost function. The W2 distance between two density functions $p$ and $q$, with respective supports $\mathcal{X}$ and $\mathcal{Y}$ is given by

$$
\mathrm{W} 2(p, q)=\left[\inf _{T \in \mathcal{T}} \int_{x \in \mathcal{X}} \int_{y \in \mathcal{Y}}\|x-y\|_{2} T(x, y) d x d y\right]^{\frac{1}{2}}
$$

where $\mathcal{T}$ denotes the set of all joint density functions over $\mathcal{X} \times \mathcal{Y}$ with marginals exactly $p$ and $q$. In practice, we use empirical approximations of $p$ and $q$ to compute the W2, which simplifies Eq. S6 to a linear program.

- The GsKL metric is sensitive to differences in means and covariances, being defined as

$$
\operatorname{GsKL}(p, q)=\frac{1}{2}\left[D_{\mathrm{KL}}(\mathcal{N}[p] \|\mathcal{N}[q])+D_{\mathrm{KL}}(\mathcal{N}[q] \|\mathcal{N}[p])\right]
$$

where $D_{\mathrm{KL}}(p \| q)$ is the Kullback-Leibler divergence between distributions $p$ and $q$ and $\mathcal{N}[p]$ is a multivariate normal distribution with mean equal to the mean of $p$ and covariance matrix equal to the covariance of $p$ (and same for $q$ ). Eq. S7 can be expressed in closed form in terms of the means and covariance matrices of $p$ and $q$. For reference, two Gaussians with unit variance and whose means differ by $\sqrt{2}$ (resp., $\frac{1}{2}$ ) have a GsKL of 1 (resp., $\frac{1}{8}$ ). As a rule of thumb, we consider a desirable target to be (much) less than 1.
Computing the metrics. For each method, we computed the metrics based on samples from the combined approximate posteriors. For methods whose approximate posterior is a surrogate GP model (i.e., GP, GP-DIS, PAI, PAI-DIS in the paper), we drew samples from the surrogate model using importance sampling/resampling (Robert and Casella, 2013). As proposal distribution for importance sampling we used a mixture of a uniform distribution over a large hyper-rectangle and a distribution centered on the region of high posterior density (the latter to increase the precision of our estimates). We verified that our results did not depend on the specific choice of proposal distribution.

---

#### Page 18

Statistical analyses. For each problem and each method, we reran the entire parallel inference procedure ten times with ten different random seeds. The same ten seeds were used for all methods - implying among other things that, for each problem, all methods were tested on the same ten random partitions of the data and using the same MCMC samples on those partitions. The outcome of each run is a triplet of metrics (MMTV, W2, GsKL) with respect to ground truth. We computed mean and standard deviation of the metrics across the ten runs, which are reported in tables in the main text. For each problem and metric, we highlighted in bold all methods whose mean performance does not differ in a statistically significant way from the best-performing method. Since the metrics are not normally distributed, we tested statistical significance via bootstrap ( $n_{\text {bootstrap }}=10^{6}$ bootstrapped datasets) with a threshold for statistical significance of $\alpha=0.05$.

# D. 3 Model details and further plots 

We report here additional details for some of the models used in the experiments in the main paper, and plots for the experiment from computational neuroscience.

Multi-modal posterior. In Section 4.1 of the main text we constructed a synthetic multi-modal posterior with four modes. We recall that the generative model is

$$
\begin{aligned}
\theta \sim p(\theta) & =\mathcal{N}\left(0, \sigma_{p}^{2}\right)_{2} \\
y_{1}, \ldots, y_{N} & \sim p\left(y_{n} \mid \theta\right)=\sum_{i=1}^{2} \frac{1}{2} \mathcal{N}\left(y_{n} ; P_{i}\left(\theta_{i}\right), \sigma_{l}^{2}\right)
\end{aligned}
$$

where $\theta \in \mathbb{R}^{2}, \sigma_{p}=\sigma_{l}=1 / 4$ and $P_{i}$ 's are second-degree polynomial functions. To induce a posterior with four modes, we chose $P_{1}\left(\theta_{1}\right)$ and $P_{2}\left(\theta_{2}\right)$ to be polynomials with exactly two roots, such that, when the observations are drawn from the full generative model in Eq. D.3, each root will induce a local maximum of the posterior in the vicinity of the root (after considering the shrinkage effect of the prior). The polynomials are defined as $P_{1}(x)=P_{2}(x)=(0.6-x)(-0.6-x)$, so the posterior modes will be in the vicinity of $\theta^{\star} \in\{(0.6,0.6),(-0.6,0.6),(0.6,-0.6),(-0.6,-0.6)\}$.

Multisensory causal inference. In Section 4.4 of the main text, we modeled a benchmark visuo-vestibular causal inference experiment (Acerbi et al., 2018; Acerbi, 2020) which is representative of many similar models and tasks in the fields of computational and cognitive neuroscience. In the modeled experiment, human subjects, sitting in a moving chair, were asked in each trial whether the direction of movement $s_{\text {vest }}$ matched the direction $s_{\text {vis }}$ of a looming visual field. We assume subjects only have access to noisy sensory measurements $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right)$, $z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}$ is the vestibular noise and $\sigma_{\text {vis }}(c)$ is the visual noise, with $c \in\left\{c_{\text {low }}, c_{\text {med }}, c_{\text {high }}\right\}$ distinct levels of visual coherence adopted in the experiment. We model subjects' responses with a heuristic 'Fixed' rule that judges the source to be the same if $\left|z_{\text {vis }}-z_{\text {vest }}\right|<\kappa$, plus a probability $\lambda$ of giving a random response (Acerbi et al., 2018). Model parameters are $\theta=\left(\sigma_{\text {vest }}, \sigma_{\text {vis }}\left(c_{\text {low }}\right), \sigma_{\text {vis }}\left(c_{\text {med }}\right), \sigma_{\text {vis }}\left(c_{\text {high }}\right), \kappa, \lambda\right)$, nonlinearly mapped to $\mathbb{R}^{6}$. In the paper, we fit real data from subject S1 of (Acerbi et al., 2018). Example approximate posteriors for the PAI-DIS and GP-DIS methods, the best-performing algorithms in this example, are shown in Fig S2.

## D. 4 Scalability of PAI to large datasets

In the main paper, as per common practice in the field, we used moderate dataset sizes ( $\sim 10 \mathrm{k}$ data points) to easily calculate ground truth. This choice bears no loss of generality because increasing dataset size only makes subposteriors sharper, which does not increase the difficulty of parallel inference (although more data would not necessarily resolve multimodality, e.g. due to symmetries of the model). On the other hand, small datasets make the reporting of run-times not meaningful, as they are dominated by overheads.

To assess the performance of PAI on large datasets, we ran PAI on the model of Section 4.1, but now with 1 million data points ( $K=10$ partitions). Average metrics for PAI were excellent and similar to what we had before: $\mathrm{ MTV}=0.009, \mathrm{~W} 2=0.005, \mathrm{GKL}=2 \mathrm{e}-05$, while all other methods still failed. Moreover, run-times for this experiment illustrate the advantages of using PAI in practice.

We ran experiments using computers equipped with two 8 -core Xeon E5 processors and 16GB or RAM each. Here, the total time for parallel inference was 57 minutes - 50 for subposterior MCMC sampling +7 for all PAI steps. By contrast, directly running MCMC on the whole dataset took roughly 6 hours.

---

#### Page 19

> **Image description.** This image shows two sets of scatter plot matrices, one labeled "PAI-DIS (ours)" and the other "GP-DIS". Each matrix displays pairwise relationships between parameters labeled $\theta_1$ through $\theta_6$.
> 
> Each scatter plot matrix is arranged in a lower triangular format. The x and y axes of each individual scatter plot represent two different parameters. The diagonal elements are omitted. The plots are populated with a dense scatter of points, colored in a gradient from blue to red, with red indicating a higher density of points. A legend in the top-left plot of each matrix indicates that the red points are labeled as "Samples". The axes are labeled with numerical values.

Figure S2: PAI-DIS and GP-DIS on the multisensory causal inference task. Each panel shows twodimensional posterior marginals as samples from the combined approximate posterior (red) against the ground truth (blue). While PAI-DIS (top figure) and GP-DIS (bottom figure) perform similarly in terms of metrics, PAI-DIS captures some features of the posterior shape more accurately, such as the 'boomerang' shape of the $\theta_{3}$ marginals (middle row).