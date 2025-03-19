# A Framework for Testing Identifiability of Bayesian Models of Perception - Appendix

---

#### Page 1

# A Framework for Testing Identifiability of Bayesian Models of Perception - Supplementary Material

Luigi Acerbi ${ }^{1,2}$ Wei Ji Ma ${ }^{2}$ Sethu Vijayakumar ${ }^{1}$<br>${ }^{1}$ School of Informatics, University of Edinburgh, UK<br>${ }^{2}$ Center for Neural Science \& Department of Psychology, New York University, USA<br>\{luigi.acerbi, weijima\}@nyu.edu sethu.vijayakumar@ed.ac.uk

## Contents

1 Introduction 1
2 Bayesian observer model 1
2.1 Mapping to internal measurement space ..... 2
2.2 Moment-based parametrization of a unimodal mixture of two Gaussians ..... 2
2.3 Computation of the expected loss ..... 3
2.4 Mapping densities from task space to internal measurement space and vice versa ..... 4
3 Model identifiability 5
3.1 Derivation of Eq. 12 in the paper ..... 5
4 Supplementary methods and results 5
4.1 Sampling from the approximate expected posterior density ..... 6
4.2 Temporal context and interval timing ..... 6
4.3 Slow-speed prior in speed perception ..... 6
5 Extensions of the observer model ..... 7
5.1 Expected loss in arbitrary spaces ..... 8

## 1 Introduction

Here we report supplementary information to the main paper, such as extended mathematical derivations and implementation details. For ease of reference, this document follows the same division in sections of the paper, and supplementary methods are reported in the same order as they are originally referenced in the main text.

## 2 Bayesian observer model

We describe here several technical details regarding the construction of the Bayesian observer model which are omitted in the paper.

---

#### Page 2

# 2.1 Mapping to internal measurement space

The mapping to internal measurement space is a mathematical trick to deal with observer models whose sensory noise magnitude is stimulus-dependent in task space. For this derivation, let us assume that the measurement probability density, $p_{\text {meas }}(x \mid s)$, can be expressed as a Gaussian with stimulus-dependent noise:

$$
p_{\text {meas }}(x \mid s)=\mathcal{N}\left(x \mid s, \sigma_{\text {meas }}^{2}(s)\right)
$$

In the case of Weber's law, we would have $\sigma_{\text {meas }}(s)=b s$, with $b>0$ standing for Weber's constant (this feature of noise is called the scalar property in time perception [1, 2]).
The problem with Eq. S1 is that the measurement distribution is Gaussian but the likelihood (function of $s$ ) is not - which is unwieldy for the computation of the posterior. A solution consists in finding a transformed space in which the likelihood is (approximately) Gaussian. It is easy to show that a mapping of the form:

$$
f(s)=\int_{-\infty}^{s} \frac{1}{\sigma_{\text {meas }}\left(s^{\prime}\right)} d s^{\prime}+\text { const }
$$

achieves this goal. In fact, we can write an informal proof as follows:

$$
\begin{aligned}
f(x) & =f\left(s+\sigma_{\text {meas }}(s) \cdot \eta\right) \\
& \approx f(s)+f^{\prime}(s) \cdot \sigma_{\text {meas }}(s) \cdot \eta \\
& =t+\eta
\end{aligned}
$$

where $\eta$ is a normally distributed random variable with zero mean and unit variance. The second passage of Eq. S3 uses a first-order Taylor expansion, under the assumption that the noise magnitude is low compared to the magnitude of the stimulus. The last passage shows that the measurement variable is approximately Gaussian in internal measurement space with mean $t \equiv f(s)$ and unit variance.

For Weber's law, the solution of Eq. S2 has a logarithmic form $f(s) \propto \log s$, which is commonly used in the psychophysics literature. We want the mapping to cover both constant noise and scalar noise (and intermediate cases), so we consider a generalized transform, $f(s) \propto \log \left(1+\frac{s}{s_{0}}\right)$, where the base magnitude parameter $s_{0}$ controls whether the mapping is purely logarithmic (for $s_{0} \rightarrow 0$ ), linear (for $s_{0} \rightarrow \infty$ ) or in-between [3]. For the paper, we further generalize the mapping, Eq. 2 in the main text, by adding a power exponent $d$ that allows to reproduce Steven's power law of sensation [4]. Note that the exponent $d$ has no effect if the mapping is (close to) purely logarithmic.

### 2.2 Moment-based parametrization of a unimodal mixture of two Gaussians

Let us consider a mixture of two Gaussian distributions:

$$
p(s)=w \mathcal{N}\left(x \mid \mu_{1}, \sigma_{1}^{2}\right)+(1-w) \mathcal{N}\left(x \mid \mu_{2}, \sigma_{2}^{2}\right)
$$

We want to express its parameters ( $w, \mu_{i}, \sigma_{i}$, for $i=1,2$ ) as a function of the standardized moments of the distribution: mean $\mu$, variance $\sigma^{2}$, skewness $\gamma$ and (excess) kurtosis $\kappa$, with the additional constraint of unimodality. The first two standardized moments are $\mu=0$ and $\sigma^{2}=1$ (this is without loss of generality, as we may later rescale and shift the resulting distribution to match arbitrary values of $\mu$ and $\sigma^{2}$ ). Since there are five parameters and only four constraints, we will find a solution (or multiple solutions) as a function of the remaining parameter $w$.

- For the special case $\gamma=0$ and $\kappa \leq 0$ we have:

$$
\mu_{1}=\sqrt[4]{-\frac{\kappa}{2}}, \quad \mu_{2}=-\mu_{1}, \quad \sigma_{1}^{2}=\sigma_{2}^{2}=\sqrt{1-\sqrt{-\frac{\kappa}{2}}}
$$

- For the special case $\gamma=0$ with $\kappa>0$ the solutions are:

$$
\mu_{1}=\mu_{2}=0, \quad \sigma_{1}^{2}=1 \mp \frac{\sqrt{(1-w) \kappa}}{\sqrt{3 w}}, \quad \sigma_{2}^{2}=1 \pm \frac{\sqrt{w \kappa}}{\sqrt{3}(1-w)}
$$

---

#### Page 3

- Finally, for the general case $\gamma \neq 0$ :

$$
\begin{aligned}
\mu_{1}= & -\frac{1-w}{w} \cdot \mu_{2} \\
\mu_{2}= & \text { Roots }\left[\left(2-6 w+8 w^{2}-6 w^{3}+2 w^{4}\right) y^{6}+\left(4 w^{2} \gamma-12 w^{3} \gamma+8 w^{4} \gamma\right) y^{3}\right. \\
& \left.+\left(3 w^{3} \kappa-3 w^{4} \kappa\right) y^{2}-w^{4} \gamma^{2}\right] \\
\sigma_{1}^{2}= & 1+\frac{(w-1) \mu_{2}}{3 w^{4} \gamma}\left[3 w^{3} \kappa+(5-7 w) w^{2} \gamma \mu_{2}-2(w-1)\left(1-w+w^{2}\right) \mu_{2}^{4}\right] \\
\sigma_{2}^{2}= & 1+\frac{\kappa}{\gamma \mu_{2}}+\left(\frac{2}{3 w}-\frac{7}{3}\right) \mu_{2}^{2}-2(w-1)\left(1+w^{2}-w\right) \frac{\mu_{2}^{5}}{3 w^{3} \gamma}
\end{aligned}
$$

where Roots specifies the real roots of the polynomial in square brackets.
The final degree of freedom is chosen by picking the value of $w$ that locally maximizes the differential entropy of the distribution while respecting the requirement of unimodality and within the range $0.025 \leq w \leq 0.975$. The latter constraint is added to prevent highly degenerate solutions such as, e.g., $w \rightarrow 0$ with $\sigma_{1}^{2} \rightarrow \infty$. At the implementation level, we do not perform these computations at every step but we precomputed a table that maps a pair of values $(\gamma, \kappa)$ to a parameter vector $w, \mu_{i}, \sigma_{j}$ for $j=1,2$ that uniquely identifies a mixture of two Gaussians. ${ }^{1}$ This table also encodes the boundaries of $\gamma, \kappa$ since not all values are allowed (see Figure S1).

> **Image description.** The image is a plot showing the relationship between skewness and excess kurtosis, with a shaded region indicating valid values of a parameter 'w'.
>
> - **Axes:** The horizontal axis is labeled "Skewness" and ranges from -5 to 5. The vertical axis is labeled "Excess kurtosis" and ranges from 0 to 60.
> - **Black Line:** A solid black line forms a curve that opens upwards, representing a lower bound on the relationship between skewness and excess kurtosis. The curve appears to be a parabola.
> - **Shaded Region:** A crescent-shaped region is shaded in varying shades of gray. The shading represents the values of 'w', as indicated by the colorbar on the right side of the image. The colorbar ranges from 0 (black) to 1 (white), with intermediate values represented by shades of gray. This shaded region lies above the black line.
> - **Colorbar:** A vertical colorbar is located on the right side of the plot. It is labeled with 'w' and ranges from 0 to 1, with corresponding shades of gray.

Figure S1: Tabulated values of $w$ as a function of skewness and kurtosis. The values of the mixing weight $w$ that respect the constraints of unimodality and $0.025 \leq w \leq 0.975$ cover a crescentshaped region in the domain of skewness $\gamma$ and excess kurtosis $\kappa$ (shaded area). The black line represents the hard bound between skewness and kurtosis that applies to all univariate distributions, that is $\kappa \geq \gamma^{2}-2$.

# 2.3 Computation of the expected loss

The observer's prior is written as:

$$
q_{\text {prior }}(t)=\sum_{m=1}^{M} w_{m} \mathcal{N}\left(t \mid \mu_{m}, a^{2}\right)
$$

[^0]
[^0]: ${ }^{1}$ Thanks to symmetries we need to precompute the table only for $\gamma \geq 0$ and we flip the sign of $\mu_{1}$ and $\mu_{2}$ for $\gamma<0$.

---

#### Page 4

where $a$ is the lattice spacing and we have defined $\mu_{m} \equiv \mu_{\min }+(m-1) a$ (see Eq. 4 in the paper). The internal measurement likelihood takes the form:

$$
q_{\text {meas }}(x \mid t)=\sum_{j=1}^{2} \tilde{\pi}_{j} \mathcal{N}\left(x \mid t+\tilde{\mu}_{j}, \tilde{\sigma}_{j}^{2}\right)
$$

with $\tilde{\pi}_{1} \equiv \tilde{\pi}, \tilde{\pi}_{2} \equiv 1-\tilde{\pi}$ (see the corresponding Eq. 3 in the paper). The posterior distribution is computed by multiplying Eq. S8 and S9:

$$
\begin{aligned}
q_{\text {post }}(t \mid x) & =\sum_{m=1}^{M} \sum_{j=1}^{2} w_{m} \tilde{\pi}_{j} \mathcal{N}\left(t \mid \mu_{m}, a^{2}\right) \mathcal{N}\left(t \mid x-\tilde{\mu}_{j}, \tilde{\sigma}_{j}^{2}\right) \\
& =\sum_{m=1}^{M} \sum_{j=1}^{2} \gamma_{m j} \mathcal{N}\left(t \mid \nu_{m j}, \tau_{m j}^{2}\right)
\end{aligned}
$$

obtained after some algebraic manipulations and where we have defined:

$$
\begin{aligned}
& \gamma_{m j} \equiv w_{m} \tilde{\pi}_{j} \mathcal{N}\left(\mu_{m} \mid x-\tilde{\mu}_{j}, a^{2}+\tilde{\sigma}_{j}^{2}\right) \\
& \nu_{m j} \equiv \frac{\mu_{m} \tilde{\sigma}_{j}^{2}+\left(x-\tilde{\mu}_{j}\right) a^{2}}{a^{2}+\tilde{\sigma}_{j}^{2}} \\
& \tau_{m j}^{2} \equiv \frac{a^{2} \tilde{\sigma}_{j}^{2}}{a^{2}+\tilde{\sigma}_{j}^{2}}
\end{aligned}
$$

The loss function depends on the signed error in internal measurement space and is defined as:

$$
\mathcal{L}(\hat{t}-t)=-\sum_{k=1}^{2} \pi_{k}^{\ell} \mathcal{N}\left(\hat{t}-t \mid \mu_{k}^{\ell}, \sigma_{k}^{\ell^{2}}\right)
$$

with $\pi_{1}^{\ell} \equiv \pi^{\ell}$ and $\pi_{2}^{\ell} \equiv 1-\pi^{\ell}$ (see Eq. 6 in the paper). The expected loss for estimate $\hat{t}$, given measurement $x$, therefore takes the closed analytical form:

$$
\begin{aligned}
\mathbb{E}[\mathcal{L} ; \hat{t}, x]_{q_{\text {post }}} & =\int q_{\text {post }}(t \mid x) \mathcal{L}(\hat{t}-t) d t \\
& =-\sum_{m=1}^{M} \sum_{j=1}^{2} \gamma_{m j} \sum_{k=1}^{2} \pi_{k}^{\ell} \int \mathcal{N}\left(t \mid \nu_{m j}, \tau_{m j}^{2}\right) \mathcal{N}\left(t \mid \hat{t}-\mu_{k}^{\ell}, \sigma_{k}^{\ell^{2}}\right) d t \\
& =-\sum_{m=1}^{M} \sum_{j=1}^{2} \gamma_{m j} \sum_{k=1}^{2} \pi_{k}^{\ell} \mathcal{N}\left(\hat{t} \mid \nu_{m j}+\mu_{k}^{\ell}, \tau_{m j}^{2}+\sigma_{k}^{\ell^{2}}\right)
\end{aligned}
$$

Eq. S13 generalizes a previous result [5, Eq. 4] to the case of likelihoods and loss functions that are mixtures of Gaussians. Thanks to the expression of Eq. S13 as a mixture of Gaussians, the global minimum of the expected loss can be found very efficiently through an adaptation of Newton's method [5, 6]. Note that computational efficiency is not merely a desirable feature, but rather a key requirement for tractability of our analysis of the complex parameter space. In Section 5.1 we discuss how the framework can be generalized to a loss function whose error is computed in arbitrary spaces.

# 2.4 Mapping densities from task space to internal measurement space and vice versa

Variables are mapped from task space to internal measurement space (and vice versa) through the mappings described in Eq. 2 in the main text. Mapping of densities needs to take into account the Jacobian of the transformation.

A distribution $p(s)$ in task space is converted into internal measurment space as:

$$
q(t)=\left|f^{-1^{\prime}}(t)\right| p\left(f^{-1}(t)\right)=\left[\frac{s_{0}}{A d}\left(e^{\frac{t-B}{A}}-1\right)^{\frac{1}{A}-1} e^{\frac{t-B}{A}}\right] p\left(f^{-1}(t)\right)
$$

---

#### Page 5

Conversely, the inverse transform of a density $q(t)$ from internal measurement space to task space is:

$$
p(s)=\left|f^{\prime}(s)\right| q(f(s))=\left[\frac{A d\left(s / s_{0}\right)^{d-1}}{1+\left(s / s_{0}\right)^{d}}\right] q(f(s))
$$

# 3 Model identifiability

A pivotal role in our a priori identifiability analysis is taken by the equation that links the expected $\log$ likelihood to the KL-divergence between the response distributions. Here we show the derivation.

### 3.1 Derivation of Eq. 12 in the paper

We want to find a closed-form solution for Eq. 11 in the paper:

$$
\langle\log \operatorname{Pr}(\mathcal{D} \mid \boldsymbol{\theta})\rangle=\int_{|\mathcal{D}|=N_{\mathrm{e}}} \log \operatorname{Pr}(\mathcal{D} \mid \boldsymbol{\theta}) \operatorname{Pr}\left(\mathcal{D} \mid \boldsymbol{\theta}^{*}\right) d \mathcal{D}
$$

First, we divide the dataset $\mathcal{D}$ as follows. Recall that the experiment presents a discrete set of stimuli $s_{i}$ with relative frequency $P_{i}$, for $1 \leq i \leq N_{\text {exp }}$. We assume that the number of trials for each stimulus is allocated a priori to match relative frequencies (a common practice in psychophysical experiments). Therefore, dataset $\mathcal{D}$ can be divided in $N_{\text {exp }}$ sub-datasets $\mathcal{D}_{i}$ with respectively $P_{i} N_{\text {tr }}$ trials each. Assuming independence between trials and thanks to linearity of the expectation operator, we can write:

$$
\langle\log \operatorname{Pr}(\mathcal{D} \mid \boldsymbol{\theta})\rangle=\sum_{i=1}^{N_{\text {exp }}}\left\langle\log \operatorname{Pr}\left(\mathcal{D}_{i} \mid \boldsymbol{\theta}\right)\right\rangle
$$

where each sub-dataset $\mathcal{D}_{i}$ only contains trials that show a specific stimulus $s_{i}$. In the following we compute the expectation of the log likelihood for a sub-dataset with a single stimulus.
Let us consider a sub-dataset $\mathcal{D}_{i}$ with $N \equiv P_{i} N_{\text {tr }}$ trials and stimulus $s_{i}$. The true distribution of responses in each trial is assumed to be stationary with distribution $p(r) \equiv p_{\text {resp }}\left(r \mid s_{i}, \boldsymbol{\theta}^{*}\right)$, whereas the distribution of responses according to model $\boldsymbol{\theta}$ is represented by $q(r) \equiv p_{\text {resp }}\left(r \mid s_{i}, \boldsymbol{\theta}\right)$. The expected $\log$ likelihood of the sub-dataset for model $\boldsymbol{\theta}$ under true model $\boldsymbol{\theta}^{*}$ is:

$$
\begin{aligned}
\left\langle\log \operatorname{Pr}\left(\mathcal{D}_{i} \mid \boldsymbol{\theta}\right)\right\rangle_{\operatorname{Pr}\left(\mathcal{D}_{i} \mid \boldsymbol{\theta}^{*}\right)} & =\int \operatorname{Pr}\left(r_{1}, \ldots, r_{N} \mid \boldsymbol{\theta}^{*}\right) \log \operatorname{Pr}\left(r_{1}, \ldots, r_{N} \mid \boldsymbol{\theta}\right) d r_{1} \times \ldots \times d r_{N} \\
& =\int \operatorname{Pr}\left(r_{1}, \ldots, r_{N} \mid \boldsymbol{\theta}^{*}\right)\left[\log \prod_{j=1}^{N} q\left(r_{j}\right)\right] d r_{1} \times \ldots \times d r_{N} \\
& =\sum_{j=1}^{N} \int \operatorname{Pr}\left(r_{1}, \ldots, r_{N} \mid \boldsymbol{\theta}^{*}\right) \log q\left(r_{j}\right) d r_{1} \times \ldots \times d r_{N} \\
& =N \int p(r) \log q(r) d r \\
& =-P_{i} N_{\mathrm{tr}} \cdot\left[D_{\mathrm{KL}}(p \mid q)+H(p)\right]
\end{aligned}
$$

where $D_{\mathrm{KL}}(p \mid q)$ is the Kullback-Leibler (KL) divergence, a non-symmetric measure of the difference between two probability distributions widely used in information theory, and $H(p)$ is the (differential) entropy of $p$. The last passage follows from the definition of cross-entropy [7]. Note that the entropy of $p$ does not depend on $\boldsymbol{\theta}$, so the entropy term is constant for our purposes. Combining Eqs. S17 and S18 we obtain Eq. 12 in the paper.

## 4 Supplementary methods and results

We report here additional details and results omitted for clarity from the main text.

---

#### Page 6

# 4.1 Sampling from the approximate expected posterior density

The observer models we consider in the paper have 26-41 parameters, which correspond to a fairly high-dimensional parameter space. We assumed indepedent, non-informative priors on each model parameter, uniform on a reasonably inclusive range. Some parameters that naturally cover several orders of magnitude (e.g., the mixing weights $w_{m}$, for $1 \leq m \leq M$ ) were represented in log scale. ${ }^{2}$ Also, the kurtosis parameters of likelihoods and loss function ( $\kappa, \tilde{\kappa}, \kappa_{\ell}$ ) were represented in a transformed kurtosis space with $\kappa^{\prime} \equiv \sqrt{\kappa+2}$ (in this space, skewness and kurtosis are on a similar scale; the hard bound $\kappa \geq \gamma^{2}-2$ becomes $\left.\kappa^{\prime} \geq|\gamma|\right)$.
We explored a priori identifiability in the large parameter space by sampling observers from the approximate expected posterior density, Eqs. 12 and 13 in the paper, via an adaptive MCMC algorithm [9]. Note that we computed the KL-divergence between the (rescaled) response distributions of a candidate model $\boldsymbol{\theta}$ and of the reference model $\boldsymbol{\theta}^{*}$, Eq. 12, only inside the range of experimental stimuli (this is equivalent to the experimental practice of discarding responses outside a certain range, to avoid edge effects). For each specific experimental design, we ran 6-10 parallel chains with different starting points near $\theta^{*}\left(5 \cdot 10^{3}\right.$ burn-in steps, $5 \cdot 10^{4}$ to $2 \cdot 10^{5}$ samples per chain, depending on model complexity). To check for convergence, we computed Gelman and Rubin's potential scale reduction statistic $R$ for all parameters [10]. Large values of $R$ denote convergence problems whereas values close to 1 suggest convergence. For all experimental designs and parameters, $R$ was generally $\lesssim 1.1$. Paired with a visual check of the marginal pdfs of the sampled chains, this result suggests a resonable degree of convergence. Finally, chains were thinned to reduce autocorrelations, storing about $N_{\text {smpl }}=10^{4}$ sampled observers per experimental design.
As an additional consistency check, we performed a 'posterior predictive check' (see e.g. [11]) with the sampled observers, that is we verified that the predicted behavior of the sampled observers matches the behavior of the reference observer across some relevant statistics (if not, it means that the sampling algorithm is not working correctly). We chose as relevant summary statistics the means and standard deviations of the observers' responses, as a function of stimulus $s_{i}$ and experimental condition (computed for each sampled observer via Eq. 9 in the paper). We found that the predicted response means were generally in excellent agreement with the 'true' response means of the reference observer. Distributions of predicted response variances showed some minor bias, but were still in good statistical agreement with the true response variance.

### 4.2 Temporal context and interval timing

The set of stimuli $s_{i}$ used in the experiment is comprised of $N_{\text {exp }}=11$ equiprobable, regularly spaced intervals over the relevant range (Short 494-847 ms, Long 847-1200 ms) [2].
To reconstruct the observer's average prior (Figure 2 a in the paper), for each sampled observer, we computed the prior in internal space (Eq. 4 in the paper) and transformed it back to task space via Eq. S15; the mean prior is obtained by averaging all sampled priors. We also computed the first four central moments of each sampled prior in task space, whose distributions are shown in Figure 2 a in the main text. The reconstruction error for each sampled prior was assessed through the symmetric KL-divergence with the prior of the reference observer (the standard KL-divergence produces similar results).
Note that observer models BSL, MAP and MTR were tested on both the Short and Long ranges (models SRT and LNG were simulated only on either the Short or the Long range). Figure 2 in the paper reports only data for the Short range; we show here the priors recovered in the Long range condition for the same models (Figure S2). Results are qualitatively similar to what we observed for the Short range, with similar deviations from the reference prior and the same ranking between experimental designs.

### 4.3 Slow-speed prior in speed perception

The set of stimuli $s_{i}$ is comprised of $N_{\text {exp }}=6$ equiprobable motion speeds: $s \in\{0.5,1,2,4,8,12\}$ deg/s [3].

[^0]
[^0]: ${ }^{2}$ Note that a uniform prior in log space implies a prior of the form $\sim 1 / x$ in standard space [8].

---

#### Page 7

> **Image description.** This image contains two panels, labeled "a" and "b", presenting data related to internal representations in interval timing.
>
> Panel a: This panel consists of a 4x5 grid of plots. The rows are labeled "BSL", "LNG", "MAP", and "MTR" on the left. The columns are labeled "Prior", "Mean", "SD", "Skewness", and "Kurtosis" at the top. Each plot in the "Prior" column shows a distribution, with a black outline, a red line, and a gray shaded area. The x-axis ranges from 847 to 1200 ms. The y-axis ranges from 0 to 5. The plots in the "Mean", "SD", "Skewness", and "Kurtosis" columns display distributions with black outlines, filled with a combination of light blue and purple. A vertical black line and a dashed green line are present in each of these plots. The x-axes for "Mean", "SD", "Skewness", and "Kurtosis" range from 900 to 1100 ms, 50 to 100 ms, -2 to 2, and 0 to 4, respectively. The y-axes range from 0 to 20, 0 to 40, 0 to 1, and 0 to 1, respectively.
>
> Panel b: This panel shows a box plot. The y-axis is labeled "KL" and ranges from 0.01 to 10, with tick marks at 0.1 and 1. The x-axis is labeled with "BSL", "LNG", "MAP", and "MTR". Above each label is a number: "0.07", "0.26", "0.06", and "0.61", respectively, with a "P\*" label above these numbers. Four gray box plots are displayed, corresponding to the labels on the x-axis.

Figure S2: Internal representations in interval timing (Long condition). Accuracy of the reconstructed priors in the Long range; each row corresponds to a different experimental design. Figure 2 in the main text reports data for the Short range in the same format. See caption of Figure 2 in the main text for a detailed legend. a: The first column shows the reference prior and the recovered mean prior. The other columns display the recovered central moments of the prior. b: Box plots of the symmetric KL-divergence between the reconstructed priors and the prior of the reference observer.

We reconstructed the observer's average $\log$ prior in task space (Figure 3 a in the paper) for each sampled observer. To capture the shape of the sampled priors, we fit each of them with a parametric formula: $\log q(s)=-k_{\text {prior }} \log \left(s^{2}+s_{\text {prior }}^{2}\right)+c_{\text {prior }}$, via least-squares estimation. The distribution of fitted parameters $k_{\text {prior }}$ and $s_{\text {prior }}$ is shown in Figure 3 a in the main text.
We show here the results for an additional observer model (FLT) which incorporates a uniformly flat prior (Figure S3). The model inference correctly recovers a flat prior with exponent $k_{\text {prior }} \approx 0$ (compare it with Figure 3 in the paper).

> **Image description.** The image presents a series of plots and box plots, divided into three sections labeled 'a', 'b', and 'c', visually representing data related to internal representations in speed perception. The data is labeled as "FLT".
>
> Section 'a' consists of three plots.
> _ The first plot, labeled "Log prior", shows a heatmap-like representation. The y-axis ranges from -10 to 0, and the x-axis ranges from approximately 0.5 to 8, labeled as "deg/s". A red horizontal band is visible near the top of the plot.
> _ The second plot, labeled "$k_{\text{prior}}$", displays a distribution curve, with the y-axis ranging from 0 to 1. The x-axis ranges from -1 to 1. The curve is a combination of multiple overlapping distributions, colored in blue, green, and light gray. A vertical black line is present near the center of the distribution. \* The third plot, labeled "$s_{\text{prior}}$", is similar to the second plot, displaying a distribution curve with the y-axis ranging from 0 to 1 and the x-axis ranging from -1 to 1, labeled as "deg/s". The curve is a combination of multiple overlapping distributions, colored in green and light gray. A vertical black line is present near the center of the distribution.
>
> Section 'b' contains a single box plot. \* The box plot is labeled "FLT" on the x-axis. The y-axis, labeled "KL", is on a logarithmic scale, ranging from 0.01 to 10. The box plot itself is gray.
>
> Section 'c' consists of six plots.
> _ The first plot is labeled "$s_0$". The y-axis ranges from 0 to 10, and the x-axis ranges from 0.01 to 1, labeled as "deg/s". A distribution curve is shown, filled with a gradient from light to dark blue. A solid vertical black line and a dashed vertical green line are present.
> _ The second plot is labeled "$\sigma_{\text{High}}$". The y-axis ranges from 0 to 10, and the x-axis ranges from 0 to 0.4. A distribution curve is shown, filled with a gradient from light to dark blue. A solid vertical black line and a dashed vertical green line are present.
> _ The third plot is labeled "$\sigma_{\text{Low}}$". The y-axis ranges from 0 to 10, and the x-axis ranges from 0.2 to 0.6. A distribution curve is shown, filled with a gradient from light to dark blue. A solid vertical black line and a dashed vertical green line are present.
> _ The fourth plot is labeled "$\tilde{\sigma}_{\text{High}}$". The y-axis ranges from 0 to 10, and the x-axis ranges from 0 to 0.4. A distribution curve is shown, with multiple overlapping distributions, colored in blue, green, and light gray. \* The fifth plot is labeled "$\tilde{\sigma}_{\text{Low}}$". The y-axis ranges from 0 to 10, and the x-axis ranges from 0.2 to 0.6. A distribution curve is shown, with multiple overlapping distributions, colored in blue, green, and light gray.

Figure S3: Internal representations in speed perception (flat prior). Accuracy of the reconstructed internal representations (priors and likelihoods) for an observer with a uniformly flat prior. Figure 3 in the main text reports data for other two observer models in the same format. See caption of Figure 3 in the main text for a detailed legend. a: The first panel shows the reference log prior and the recovered mean log prior. The other two panels display the approximate posteriors of $k_{\text {prior }}$ and $s_{\text {prior }}$. b: Box plot of the symmetric KL-divergence between the reconstructed and reference prior. c: Approximate posterior distributions for sensory mapping and sensory noise parameters.

# 5 Extensions of the observer model

We discuss here an extension of the framework presented in the main text.

---

#### Page 8

# 5.1 Expected loss in arbitrary spaces

In the paper we have used a loss function that depends on the error, i.e. on the difference between the estimate and the true value of the stimulus, in internal measurement space (Eq. S12). However, we might want to compute the error in task space, or more in general in an arbitrary loss space defined by a mapping $g(s): s \rightarrow l$ parametrized by $s_{0}^{\ell}$ and $d^{\ell}$ (see Eq. 2 in the paper). Ideally, we still want to find a closed-form expression for the expected loss. We can write the loss function in the new loss space as:

$$
\mathcal{L}(g(\hat{s})-g(s))=\mathcal{L}\left(g\left(f^{-1}(\hat{t})\right)-g\left(f^{-1}(t)\right)\right)=\mathcal{L}(h(\hat{t})-h(t))
$$

where we have defined the composite function $h \equiv g \circ f^{-1}$. Clearly the original expression of the loss in internal measurement space is recovered if $g \equiv f$. We can rewrite the expected loss as follows:

$$
\begin{aligned}
\mathbb{E}\left[\mathcal{L}\right]_{\mathrm{q}_{\text {post }}}(\hat{t})= & -\sum_{m=1}^{M} \sum_{j=1}^{2} \gamma_{m j} \sum_{k=1}^{2} \pi_{k}^{t} \mathcal{N}\left(h(\hat{t}) \mid \nu_{m j}+\mu_{k}^{t}, \tau_{m j}^{2}+\sigma_{k}^{\ell^{2}}\right) \\
& \times \int \mathcal{N}\left(h(t) \mid \bar{\nu}_{m j k}(\hat{t}), \bar{\tau}_{m j k}^{2}\right) d t
\end{aligned}
$$

where we have defined:

$$
\bar{\nu}_{m j k}(\hat{t}) \equiv \frac{\nu_{m j} \sigma_{k}^{\ell^{2}}+\left(h(\hat{t})-\mu_{k}^{\ell}\right) \tau_{m j}^{2}}{\tau_{m j}^{2}+\sigma_{k}^{\ell^{2}}}, \quad \bar{\tau}_{m j k} \equiv \frac{\tau_{m j}^{2} \sigma_{k}^{\ell^{2}}}{\tau_{m j}^{2}+\sigma_{k}^{\ell^{2}}}
$$

In order to perform the integration in Eq. S20, we Taylor-expand $h(t)$ up to the first order around the mean of each integrated Gaussian, $\bar{\nu}_{m j k}(\hat{t})$. We can perform this linearization without major loss of accuracy since the Gaussians in the integral are narrow, their variance bounded from above by $a^{2}\left(\bar{\tau}_{m j k}^{2}<\tau_{m j}^{2}<a^{2}\right.$, see Eqs. S11 and S21). The integration yields:

$$
\mathbb{E}\left[\mathcal{L}\right]_{\mathrm{q}_{\text {post }}}(\hat{t}) \approx-\sum_{m=1}^{M} \sum_{j=1}^{2} \gamma_{m j} \sum_{k=1}^{2} \pi_{k}^{t} \mathcal{N}\left(h(\hat{t}) \mid \nu_{m j}+\mu_{k}^{\ell}, \tau_{m j}^{2}+\sigma_{k}^{\ell^{2}}\right) \frac{1}{h^{\prime}\left(\bar{\nu}_{m j k}(\hat{t})\right)}
$$

Eq. S22 is not a regular mixture of Gaussians, but we can write its first and second derivative analytically, which in principle allows to apply Newton's method for numerical minimization. This derivation shows that the techniques developed in the paper can be extended to the general case of a loss function based in an arbitrary space (including, in particular, task space).

## Acknowledgments

We thank Jonathan Pillow, Paolo Puggioni, Peggy Seriès, and three anonymous reviewers for useful comments on earlier versions of the work.

## References

[1] Rakitin, B. C., Gibbon, J., Penney, T. B., Malapani, C., Hinton, S. C., \& Meck, W. H. (1998) Scalar expectancy theory and peak-interval timing in humans. J Exp Psychol Anim Behav Process 24, 15-33.
[2] Jazayeri, M. \& Shadlen, M. N. (2010) Temporal context calibrates interval timing. Nat Neurosci 13, $1020-1026$.
[3] Stocker, A. A. \& Simoncelli, E. P. (2006) Noise characteristics and prior expectations in human visual speed perception. Nat Neurosci 9, 578-585.
[4] Stevens, S. S. (1957) On the psychophysical law. Psychol Rev 64, 153-181.
[5] Acerbi, L., Vijayakumar, S., \& Wolpert, D. M. (2014) On the origins of suboptimality in human probabilistic inference. PLoS Comput Biol 10, e1003661.
[6] Carreira-Perpiñán, M. A. (2000) Mode-finding for mixtures of gaussian distributions. IEEE T Pattern Anal 22, 1318-1323.
[7] Cover, T. M. \& Thomas, J. A. (2012) Elements of information theory. (John Wiley \& Sons).
[8] Jaynes, E. T. (2003) Probability theory: the logic of science. (Cambridge University Press).

---

#### Page 9

[9] Haario, H., Laine, M., Mira, A., \& Saksman, E. (2006) DRAM: efficient adaptive MCMC. Stat Comput 16, 339-354.
[10] Gelman, A. \& Rubin, D. B. (1992) Inference from iterative simulation using multiple sequences. Stat Sci 7, 457-472.
[11] Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., \& Rubin, D. B. (2013) Bayesian data analysis (3rd edition). (CRC Press).
