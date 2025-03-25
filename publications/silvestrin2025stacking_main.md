```
@article{silvestrin2025stacking,
  title={Stacking Variational Bayesian Monte Carlo},
  author={Silvestrin, Francesco and Li, Chengkun and Acerbi, Luigi},
  year={2025},
  journal={7th Symposium on Advances in Approximate Bayesian Inference (AABI) - Workshop track}
}
```

---

#### Page 1

# Stacking Variational Bayesian Monte Carlo

Francesco Silvestrin, Chengkun Li, Luigi Acerbi

#### Abstract

Variational Bayesian Monte Carlo (VBMC) is a sample-efficient method for approximate Bayesian inference with computationally expensive likelihoods. While VBMC's local surrogate approach provides stable approximations, its conservative exploration strategy and limited evaluation budget can cause it to miss regions of complex posteriors. In this work, we introduce Stacking Variational Bayesian Monte Carlo (S-VBMC), a method that constructs global posterior approximations by merging independent VBMC runs through a principled and inexpensive post-processing step. Our approach leverages VBMC's mixture posterior representation and per-component evidence estimates, requiring no additional likelihood evaluations while being naturally parallelisable. We demonstrate S-VBMC's effectiveness on two synthetic problems designed to challenge VBMC's exploration capabilities and two real-world applications from computational neuroscience, showing substantial improvements in posterior approximation quality across all cases.

## 1. Introduction

Bayesian inference provides a powerful framework for parameter estimation and uncertainty quantification, but it is usually intractable requiring approximate inference techniques (Brooks et al., 2011; Blei et al., 2017). Many scientific and engineering problems involve black-box models (Sacks et al., 1989; Kennedy and O’Hagan, 2001), where likelihood evaluation is time-consuming and gradients cannot be easily obtained, making traditional approximate inference approaches computationally prohibitive.

A promising approach to tackle expensive likelihoods is to construct a statistical surrogate model that approximates the target distribution, similar in spirit to surrogate approaches to global optimisation using Gaussian processes (Williams and Rasmussen, 2006; Garnett, 2023). However, attempting to build a single global surrogate model may lead to numerical instabilities and poor approximations when the target distribution is complex or multi-modal, without ad hoc solutions (Wang and Li, 2018; Järvenpää et al., 2020; Li et al., 2024). Local or constrained surrogate models, while more limited in scope, tend to be more stable and reliable in practice (El Gammal et al., 2023; Järvenpää and Corander, 2024).

Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018) exemplifies this local approach, using active sampling to train a Gaussian process surrogate for the unnormalised log-posterior on which it performs variational inference. VBMC adopts a conservative exploration strategy that yields stable, local approximations (Acerbi, 2019). Compared to other surrogate-based approaches, the method offers a versatile set of features: it returns the approximate posterior as a tractable distribution (a mixture of Gaussians); it provides a lower bound for the model evidence (ELBO) via Bayesian quadrature (Ghahramani and Rasmussen, 2002), useful for model selection; and it can handle noisy log-likelihood evaluations (Acerbi, 2020), which arise in simulator-based models through estimation techniques

---

#### Page 2

such as inverse binomial sampling (van Opheusden et al., 2020) and synthetic likelihood (Wood, 2010; Price et al., 2018). However, VBMC's limited sampling budget combined with its local exploration strategy can leave it vulnerable to potentially missing regions of the target posterior - particularly for distributions with distinct modes or long tails.

In this work, we propose a practical, yet effective approach to constructing global surrogate models while overcoming the limitations of standard VBMC by combining multiple local approximations. We introduce Stacking Variational Bayesian Monte Carlo (S-VBMC), a method for merging independent VBMC inference runs into a coherent global posterior approximation. Our approach leverages VBMC's unique properties - its mixture posterior representation and per-component Bayesian quadrature estimates of the ELBO - to combine and reweigh each component through a simple post-processing step.

Crucially, our method requires no additional evaluations of either the original model or the surrogate. This approach is easily parallelisable and naturally fits existing VBMC pipelines that already employ multiple independent runs (Huggins et al., 2023). While our method could theoretically extend to other variational approaches based on mixture posteriors, VBMC is uniquely suitable for it as re-estimation of the ELBO would otherwise become impractical with expensive likelihoods (see Section 3).

We first introduce variational inference and VBMC (Section 2), then present our algorithm for stacking VBMC posteriors (Section 3). We demonstrate the effectiveness of our approach through experiments on two synthetic problems and two real-world applications that are challenging for VBMC (Section 4). We conclude with closing remarks (Section 5). Appendix A contains supplementary materials, including a discussion of related work (A.1).

# 2. Background

Variational Inference. Consider a model with prior $p(\boldsymbol{\theta})$ and likelihood $p(\mathcal{D} \mid \boldsymbol{\theta})$, where $\boldsymbol{\theta} \in \mathbb{R}^{D}$ is a vector of model parameters and $\mathcal{D}$ a specific dataset. Variational inference (Blei et al., 2017) approximates the true posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$ with a parametric distribution $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$ by maximising the evidence lower bound (ELBO):

$$
\operatorname{ELBO}(\boldsymbol{\phi})=\mathbb{E}_{q_{\boldsymbol{\phi}}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{\theta})\right]
$$

where the first term is the expected log joint distribution (the joint being likelihood times prior) and the second term the entropy of the variational posterior. Maximising Eq. 1 is equivalent to minimising the Kullback-Leibler divergence between $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$ and the true posterior. The ELBO provides a lower bound on the log model evidence $\log p(\mathcal{D})$, with equality when the approximation matches the true posterior.

Variational Bayesian Monte Carlo (VBMC). VBMC is a sample-efficient technique to obtain a variational approximation with only a small number of likelihood evaluations, often of the order of a few hundreds. VBMC uses a Gaussian process (GP) as a surrogate of the log-joint, Bayesian quadrature to calculate the expected log-joint, and active sampling to decide which parameters to evaluate next (see Acerbi, 2018, 2020 for details). Crucially, VBMC performs variational inference on the surrogate, instead of the true, expensive model.

---

#### Page 3

In VBMC, the variational posterior is defined as

$$
q_{\boldsymbol{\phi}}(\boldsymbol{\theta})=\sum_{k=1}^{K} w_{k} q_{k, \boldsymbol{\phi}}(\boldsymbol{\theta})
$$

where $q_{k}$ is the $k$-th component (a multivariate normal) and $w_{k}$ its mixture weight, with $\sum_{k=1}^{K} w_{k}=1$ and $w_{k} \geq 0$. Plugging in the mixture posterior, the ELBO (Eq. 1) becomes:

$$
\operatorname{ELBO}(\boldsymbol{\phi})=\sum_{k=1}^{K} w_{k} \mathbb{E}_{q_{k, \phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{\theta})\right]=\sum_{k=1}^{K} w_{k} I_{k}+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{\theta})\right]
$$

where we defined the $k$-th component of the expected log-joint as:

$$
I_{k}=\mathbb{E}_{q_{k, \phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})] \approx \mathbb{E}_{q_{k, \phi}}[f(\boldsymbol{\theta})]
$$

with $f(\boldsymbol{\theta}) \approx \log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})$ the GP surrogate of the log-joint. Eq. 4 has a closed-form Gaussian expression via Bayesian quadrature, which yields posterior mean $I_{k}$ and covariance matrix $J_{k k^{\prime}}$ (Acerbi, 2018). The entropy of a mixture of Gaussians does not have an analytical solution, but gradients can be estimated via Monte Carlo. Thus, using the posterior mean of Eq. 4 as a plug-in estimator for the expected log-joint of each component, Eq. 3 can be efficiently optimised via stochastic gradient ascent (Kingma and Ba, 2014).

# 3. Stacking VBMC

In this work, we introduce Stacking VBMC (S-VBMC), a novel approach to merge different variational posteriors obtained from different runs on the same model and dataset.

Given $M$ independent VBMC runs, one obtains $M$ variational posteriors $q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})$, each with $K_{m}$ Gaussian components, as defined in Eq. 2, as well as $M$ different $\mathbf{I}_{m}$ vectors, as per Eq. 4. Our approach consists of "stacking" the Gaussian components of all posteriors $q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})$ leaving all individual components parameters (means and covariances) unchanged, and reoptimising all the weights. Thus, given the stacked posterior

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \hat{w}_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

we optimise the global evidence lower bound with respect to the weights $\hat{\mathbf{w}}$,

$$
\operatorname{ELBO}_{\text {stacked }}(\hat{\mathbf{w}})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \hat{w}_{m, k} I_{m, k}+\mathcal{H}\left[q_{\tilde{\phi}}^{\cdot}(\boldsymbol{\theta})\right]
$$

Notably, this optimisation can be performed as a pure post-processing step, requiring neither evaluations of the original likelihood $p(\mathcal{D} \mid \boldsymbol{\theta})$ nor of the surrogate models $f_{m}$, only that the estimates $I_{m, k}$ are stored, as in current implementations (Huggins et al., 2023).

Our stacking method hinges on the key feature of VBMC of providing accurate estimates $I_{m, k}$. While in principle Eq. 6 could apply to any collection of variational posterior mixtures, without an efficient way of calculating each $I_{k}$ (Eq. 4), optimisation of the stacked ELBO would require many likelihood evaluations, which would be prohibitive for problems with expensive, black-box likelihoods. Figure 1 shows an example of two separate posteriors and the stacked result. In the following, we demonstrate the efficacy of this approach.

---

#### Page 4

> **Image description.** The image consists of two contour plots side-by-side, each visualizing a posterior distribution. The plot on the left is titled "Separate VBMC posteriors," while the plot on the right is titled "Stacked posterior." Both plots have the same axes: the x-axis is labeled "θ₂" and ranges from approximately 1e-5 to 3e-5, while the y-axis is labeled "θ₄" and ranges from -0.01 to 0.01.
>
> In the left plot, there are two distinct sets of contour lines. One set is colored blue and is concentrated in the upper-left portion of the plot. The other set is colored orange and is concentrated in the lower-right portion of the plot. These two sets of contours appear to represent two separate posterior distributions.
>
> In the right plot, there is a single set of contour lines colored in magenta. These contours form a single, elongated shape that appears to be a combination or stacking of the two separate distributions shown in the left plot. The contours are denser in the center of the shape, indicating a higher probability density.

Figure 1: Two separate VBMC posteriors (left) and stacked posterior after running SVBMC (right) for a neuronal model with real data (see Section 4); showing the marginal distribution of two out of the 5 model parameters.

# 4. Experiments

Procedure. We tested our method on two synthetic problems, designed to be particularly challenging for VBMC, as well as on two real-world datasets and models (see Appendix A. 2 for full descriptions). We considered both noiseless problems (exact estimation) and noisy problems where Gaussian noise with $\sigma=3$ is applied to each log-likelihood measurement, emulating what practitioners might find when estimating the likelihood via simulation (van Opheusden et al., 2020). For each benchmark, we performed 100 VBMC converging runs with default settings and random uniform initialisation within plausible parameter bounds (Acerbi, 2018). To investigate the effect of combining a different number of posteriors, we then randomly sampled and stacked with S-VBMC a varying number of runs (between 2 and 40) ten times each, and computed the median and interquartile range for all metrics.

Following Acerbi (2020); Li et al. (2024), we use three main metrics for evaluating the posterior approximation of our algorithm: the absolute difference between the true log marginal likelihood (LML) and its variational approximation (the ELBO); the mean marginal total variation distance (MMTV) between the approximate posterior and ground truth; and the "Gaussianised" symmetrised KL divergence (GsKL) between variational posterior and ground truth (see Appendix A. 3 for a detailed description).

We used black-box variational inference (BBVI; Ranganath et al., 2014) as a baseline for all our benchmark problems. The target density evaluation budget for BBVI is $2000(D+$ 2) for noiseless problems and $3000(D+2)$ for noisy problems, which correspond to the maximum number of evaluations used in total by 40 VBMC runs (see Appendix A.4).

Our results are described below and reported in full in Appendix A.5, with example visualisations of posterior approximations in Appendix A.6.

Synthetic problems. The first synthetic target consists of a $2 D$ Gaussian mixture model (GMM) with 20 components clustered around four distant centroids. We expected VBMC to discover only one of the clusters in each run. The second synthetic target (ring) consists of a very narrow ring-shaped distribution in two dimensions. We expected VBMC to only cover part of it in each run due to the limited budget (50) of Gaussian components.

---

#### Page 5

Results in Figure 2 and Table A. 1 show that merging more posteriors leads to a steady improvement in the GsKL and MMTV metrics, which measure the quality of the posterior approximation. Remarkably, S-VBMC proves to be robust to noisy targets, with minimal differences between noiseless and noisy settings. S-VBMC outperforms the BBVI baseline and regular VBMC on the ring-shaped synthetic target. The BBVI baseline performs well and similarly to S-VBMC only on the GMM problem, where it effectively managed to capture the four clusters (see Figure A. 1 for a visualisation). As expected by design, individual VBMC runs tended to explore the two synthetic target distributions only partially. Still, the random initialisations allowed different runs to discover different portions of the posterior, allowing the merging process to cover the whole target (see Figure A.1).

Finally, we observe that while the ELBO keeps increasing, the $\Delta$ LML error (difference between ELBO and true log marginal likelihood) initially decreases but then increases again as further components are added, a point which we will discuss later.

> **Image description.** This image contains two rows of plots, each row consisting of four separate plots. The top row is labeled "(a) GMM (D = 2)" and the bottom row is labeled "(b) Ring (D = 2)". Each plot within a row displays different metrics as a function of the number of VBMC runs.
>
> Here's a breakdown of the visual elements:
>
> - **General Layout:** The image is divided into two panels, (a) and (b). Each panel contains four plots arranged horizontally.
>
> - **Plots:** Each plot is a scatter plot with error bars. The x-axis represents the number of VBMC runs, with values 1, 4, 8, 16, 24, 32, and 40. Data is plotted in blue, orange, green, and purple. The blue and orange data points are connected with lines and have error bars. The green and purple data points appear only at the x-value of 40 and also have error bars.
>
> - **Plot Titles and Y-axes:**
>
>   - The first plot in each row is labeled "ELBO" on the y-axis.
>   - The second plot in each row is labeled "Δ LML" on the y-axis.
>   - The third plot in each row is labeled "MMTV" on the y-axis.
>   - The fourth plot in each row is labeled "GsKL" on the y-axis.
>
> - **Horizontal Lines:** Each plot contains a horizontal line. In the ELBO plots, the line is solid black. In the other plots, the line is dashed black.
>
> - **Colors and Data Representation:**
>
>   - Blue data points represent likelihood evaluations that are noiseless.
>   - Orange data points represent likelihood evaluations that are noisy with σ=3 log-likelihood noise.
>   - Green and purple data points represent the best BBVI results for noiseless and noisy likelihood, respectively.
>
> - **Text:** The x-axis label for all plots is "VBMC runs". The y-axis labels are "ELBO", "Δ LML", "MMTV", and "GsKL". The panel labels are "(a) GMM (D = 2)" and "(b) Ring (D = 2)".

Figure 2: Synthetic problems. Metrics plotted as a function of the number of VBMC runs stacked (median and interquartile range). Likelihood evaluations are noiseless (blue) or noisy with $\sigma=3$ log-likelihood noise (orange). The best BBVI results for noiseless and noisy likelihood are shown in green and purple. The black horizontal line in the ELBO panels represents the ground-truth LML while the dashed lines on $\Delta$ LML, MMTV and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Appendix A.3).

Real-world problems. Finally, we tested VBMC on two real-world models and datasets. First, we fitted the 5 biophysical parameters of a morphologically detailed neuronal model of hippocampal pyramidal cells (similar to Szoboszlay et al., 2016 for cerebellar Golgi cells) to experimental data consisting of a detailed three-dimensional reconstruction and electrophysiological recordings (Golding et al., 2005) of one of such cells. Then, we fitted a 6-parameter model of multisensory causal inference (Körding et al., 2007) to human behavioural data from a visuo-vestibular task (subject S1 from Acerbi et al., 2018), assuming log-likelihood measurement noise $(\sigma=3)$. This model describes how participants judge whether visual

---

#### Page 6

> **Image description.** The image contains two rows of plots, each row consisting of four subplots. Each subplot displays a metric as a function of the number of VBMC runs stacked. The first row is labeled "(a) Neuronal model (D = 5)", and the second row is labeled "(b) Multisensory (D = 6, σ = 3)".
>
> **Row 1: Neuronal Model (D=5)**
>
> - **Subplot 1:** A scatter plot showing "ELBO" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis ranges from -7500 to -7440. Blue data points with error bars are plotted, showing a relatively stable ELBO value as the number of VBMC runs increases, with a green data point at x=40. A horizontal black line is present.
> - **Subplot 2:** A scatter plot showing "Δ LML" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis ranges from 0 to 100. Blue data points with error bars are plotted, showing a relatively stable Δ LML value as the number of VBMC runs increases, with a green data point at x=40.
> - **Subplot 3:** A scatter plot showing "MMTV" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis ranges from 0.0 to 0.6. Blue data points with error bars are plotted, showing a decreasing MMTV value as the number of VBMC runs increases, with a green data point at x=40. A horizontal dashed black line is present.
> - **Subplot 4:** A scatter plot showing "GsKL" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis is on a logarithmic scale, ranging from 0.1 to 100. Blue data points with error bars are plotted, showing a decreasing GsKL value as the number of VBMC runs increases, with a green data point at x=40. A horizontal dashed black line is present.
>
> **Row 2: Multisensory (D=6, σ=3)**
>
> - **Subplot 1:** A scatter plot showing "ELBO" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis ranges from -447.5 to -442.5. Orange data points with error bars are plotted, showing an increasing ELBO value as the number of VBMC runs increases, with a purple data point at x=40. A horizontal black line is present.
> - **Subplot 2:** A scatter plot showing "Δ LML" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis ranges from 0 to 4. Orange data points with error bars are plotted, showing an increasing Δ LML value as the number of VBMC runs increases, with a purple data point at x=40. A horizontal dashed black line is present.
> - **Subplot 3:** A scatter plot showing "MMTV" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis ranges from 0.0 to 0.2. Orange data points with error bars are plotted, showing a decreasing MMTV value as the number of VBMC runs increases, with a purple data point at x=40. A horizontal dashed black line is present.
> - **Subplot 4:** A scatter plot showing "GsKL" on the y-axis and "VBMC runs" on the x-axis. The x-axis ranges from 1 to 40. The y-axis is on a logarithmic scale, ranging from 0.1 to 1. Orange data points with error bars are plotted, showing a decreasing GsKL value as the number of VBMC runs increases, with a purple data point at x=40. A horizontal dashed black line is present.

Figure 3: Real-world problems. Metrics plotted as a function of the number of VBMC runs stacked (median and interquartile range). Metrics plotted as a function of the number of VBMC runs stacked (median and interquartile range). Likelihood evaluations are noiseless (blue) or noisy with $\sigma=3$ log-likelihood noise (orange). The best BBVI results for noiseless and noisy likelihood are shown in green and purple. See Figure 2 caption for additional details.

and vestibular motion cues share a common cause, incorporating sensory noise parameters and decision rules to account for participant responses in different experimental conditions.

The results in Figure 3 and Table A. 2 confirm our earlier findings of improvements across the posterior metrics. We also find that S-VBMC is robust to noisy targets for real data, with performance that improves with increasing number of stacked runs in the multisensory model problem, and consistently better than standard VBMC and the BBVI baseline.

ELBO estimation bias. Our results show that merging more VBMC runs leads to a positive bias build-up in the estimated ELBO, particularly with noisy log-likelihood problems. This likely occurs because all $I_{m, k}$ are noisy estimates of the true expected log-joint contributions, causing S-VBMC to overweigh the most overestimated mixture components - an effect that increases with the number of components $M$ (see Appendix A.7). While this bias surprisingly does not affect other posterior quality metrics, which keep improving (or plateau) with increasing $M$, it should be considered when using $\mathrm{ELBO}_{\text {stacked }}$ for model comparison. Future work should investigate bias sources and potential debiasing techniques.

# 5. Conclusions

In this work, we introduced S-VBMC, an approach for merging independent VBMC runs in a principled way to yield a global posterior approximation. We showed its effectiveness on challenging synthetic and real-world problems, as well as its robustness to noise. We briefly discussed the positive bias in the ELBO estimation introduced (or amplified) by the stacking process, leaving further investigation for future work.
