```
@inproceedings{mikkola2024preferential,
  title={Preferential Normalizing Flows},
  author={Mikkola, Petrus and Acerbi, Luigi and Klami, Arto},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024},
}
```

---

#### Page 1

# Preferential Normalizing Flows

Petrus Mikkola, Luigi Acerbi, Arto Klami\*<br>Department of Computer Science, University of Helsinki<br>first.last@helsinki.fi

#### Abstract

Eliciting a high-dimensional probability distribution from an expert via noisy judgments is notoriously challenging, yet useful for many applications, such as prior elicitation and reward modeling. We introduce a method for eliciting the expert's belief density as a normalizing flow based solely on preferential questions such as comparing or ranking alternatives. This allows eliciting in principle arbitrarily flexible densities, but flow estimation is susceptible to the challenge of collapsing or diverging probability mass that makes it difficult in practice. We tackle this problem by introducing a novel functional prior for the flow, motivated by a decision-theoretic argument, and show empirically that the belief density can be inferred as the function-space maximum a posteriori estimate. We demonstrate our method by eliciting multivariate belief densities of simulated experts, including the prior belief of a general-purpose large language model over a real-world dataset.

## 1 Introduction

Representing beliefs as probability distributions can be useful, particularly as prior probability distributions in Bayesian inference - especially in high-dimensional, non-asymptotic settings where the prior strongly influences the posterior [Gelman et al., 2017] - or as probabilistic alternatives to reward models [Leike et al., 2018, Ouyang et al., 2022]. Our goal is to elicit a complex multivariate probability density from an expert, as a representation of their beliefs. By expert, we mean an information source with a belief over a problem of interest, termed belief density, which does not permit direct evaluation or sampling. The problem is an instance of expert knowledge elicitation, where the belief is elicited by asking elicitation queries such as quantiles of the distribution [O'Hagan, 2019]. The current elicitation literature (see Mikkola et al. 2023 for a recent overview) focuses almost exclusively on extremely simple distributions, mostly products of univariate distributions of known parametric form. Some isolated works have considered more flexible distributions, for instance quantile-parameterized distributions [Perepolkin et al., 2024] for univariate cases, or Gaussian processes [Oakley and O'Hagan, 2007] and copulas for modelling low-dimensional dependencies [Clemen et al., 2000], but we want to move considerably beyond that and elicit flexible beliefs using modern neural network representations [Bishop and Bishop, 2023]. The main challenges are identifying elicitation queries that are sufficiently informative to infer the belief density while being feasible for the expert to answer reliably, and selecting a model class for the belief density that can represent flexible beliefs without simplifying assumptions but that can still be efficiently estimated.

Normalizing flows are a natural family for representing flexible distributions [Papamakarios et al., 2021]. When using flows for modelling a density $p(\mathbf{x})$, learning is usually based on either a set of samples $\mathbf{x} \sim p(\mathbf{x})$ drawn from the distribution (density estimation; Dinh et al., 2014) or on the log density $\log p(\mathbf{x})$ evaluated at flow samples, $\mathbf{x} \sim q(\mathbf{x})$, in the variational inference formulation [Rezende and Mohamed, 2015]. Neither strategy applies to our setup, since we do not have the luxury of sampling from the belief density and obviously cannot evaluate it either. In addition to

[^0]
[^0]: \*Equal contribution

---

#### Page 2

> **Image description.** The image contains four subplots arranged in a 1x4 grid, each depicting a two-dimensional density estimation result. Each subplot shows a heatmap overlaid with contour lines, along with scattered red and blue points.
>
> - **General Layout:** The subplots are labeled (a), (b), (c), and (d) in the top left corner of each panel.
>
> - **Heatmaps:** The heatmaps use a color gradient, where darker colors (primarily dark purple/blue) indicate lower density and lighter colors (yellow/green) indicate higher density. The heatmaps appear to represent estimated probability densities.
>
> - **Contour Lines:** Each subplot also features contour lines, which are white or light purple, representing the true density. These lines trace the boundaries of areas with equal probability density.
>
> - **Red and Blue Points:** Scattered throughout each subplot are red and blue points. The red points are labeled as "preferred points" and the blue points as "non-preferred points" in the figure caption.
>
> - **Subplot (a):** The heatmap shows a small, concentrated area of high density in the lower-left corner and some scattered patches of higher density elsewhere. The contour lines form an elongated, curved shape.
>
> - **Subplot (b):** The heatmap shows a more diffuse distribution of density, with a broader area of higher density compared to (a). The contour lines are less well-aligned with the heatmap.
>
> - **Subplot (c):** The heatmap shows a more focused area of high density, closely aligned with the curved shape defined by the contour lines. The density estimation appears more accurate than in (a) or (b).
>
> - **Subplot (d):** Similar to (c), the heatmap shows a focused area of high density that closely matches the shape of the contour lines. The density estimation appears even more accurate than in (c).
>
> In summary, the image visually compares the performance of density estimation using normalizing flows under different conditions, with heatmaps representing estimated densities, contour lines representing true densities, and red/blue points representing preference data. Subplots (c) and (d) show improved density estimation compared to (a) and (b).

(a) $n=10$, w/o prior (b) $n=10$, w/o prior (c) $n=10$, w/ prior (d) $n=100$, w/ prior

Figure 1: Illustration of belief densities elicited from preferential ranking data by a normalizing flow (contour: true density; heatmap: estimated flow; red: preferred points; blue: non-preferred points). (a)-(b): Typical failure modes of collapsing and diverging mass, when training a flow with just $n=10$ rankings. (c)-(d): The proposed functional prior resolves the issues, and already with 10 rankings we can learn the correct belief density, matching the result of the flow trained on larger data.

the well-known challenges of training normalizing flows, the setup introduces new difficulties; in particular, a flexible flow easily collapses or finds a way of allocating probability mass in undesirable ways. Significant literature on resolving these issues exists [Behrmann et al., 2021, Salmona et al., 2022, Cornish et al., 2020], but conclusive solutions that guarantee stable learning are still missing. Our solution offers new tools for controlling the flow in low-density areas, and hence we contribute for the general flow literature despite focusing on the specific new task.
We build on established literature on knowledge elicitation for the interaction with the expert. Distributions are primarily characterized by their location and covariance structure, yet humans are notoriously bad at assessing covariances between variables [Jennings et al., 1982, Wilson, 1994]. However, human preferences, with potentially strong interconnections between variables, can be recovered by asking individuals to compare or rank alternatives, a topic studied under discrete choice theory [Train, 2009]. The most studied random utility models (RUMs) interpret human choice as utility maximization with an additive noise component [Marschak, 1959]. To infer the correlation structure in human beliefs indirectly from elicitation data, we study a setup where the expert compares or ranks alternatives (events) based on their probability so that their decisions can be modeled by a RUM. In practice, this means that the data for learning the flow will take the form of choice sets $\mathcal{C}_{k}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{k}\right\}$ of candidates presented to the expert, combined with their choices indicating the preference over the alternatives based on their probability. We stress that candidates $\mathbf{x}$ are here not samples from the belief density but are instead provided by some other unknown process, such as an active learning method [Houlsby et al., 2011]. The only information about the belief density comes from the choice.

We are not aware of any previous works that learn flows from preferential comparisons. We first discuss some additional challenges caused by preferential data, and then show how we can leverage preferential structure to improve learning. Specifically, our learning objective corresponds to a function-space maximum a posteriori (FS-MAP), where Bayesian inference is conducted on the function (flow) itself, not its parameters [Wolpert, 1993, Qiu et al., 2024]. The learning objective is exact, in contrast to flow-based algorithms that model phenomena involving discontinuities [Nielsen et al., 2020, Hoogeboom et al., 2021], such as the argmax operator in the RUM model. By construction, the choice sets explicitly include candidates $\mathbf{x}$ that were not preferred by the expert, carrying information about relative densities of preferred vs. not preferred points. This allows us to introduce a functional prior that encourages allocating more mass to regions with high probability under a RUM with exponential noise, solving the collapsing and diverging probability mass problem that poses a challenge for flow inference in small data scenarios.
In summary, we introduce the novel problem of inferring probability density from preferential data using normalizing flows and provide a practical solution. We model the expert's choice as a RUM with exponentially distributed noise, and query the expert for comparison or ranking of $k$ alternatives. We derive the likelihoods for $k$-wise comparisons and rankings and study the distribution of the most preferred point among $k$ alternatives, which we term the $k$-wise winner. Based on the interpretation of the $k$-wise winner distribution as a tempered and tilted belief density, we introduce an empirical

---

#### Page 3

function prior and the FS-MAP objective for learning the flow. Finally, we validate our method using both synthetic and real data sets.

# 2 Why learning the density from preferential data is challenging?

Learning flows from small samples is challenging, especially in higher dimensions even when learning from direct data, such as samples from the density. Figure 1 illustrates two common challenges of collapsing and diverging probability mass; the illustration is based on our setup to showcase the proposed solution, but the same problems occur in the classical setup. The "collapsing mass" scenario is a form of overfitting, similar to mode collapse in mixture models [Li et al., 2007], but more extreme for flexible models.

In the "diverging mass" problem, the model places probability mass in the regions of low probability. The problem has connections to difficulties in training [Behrmann et al., 2021, Dhaka et al., 2021, Vaitl et al., 2022, Liang et al., 2022] and issues with coupling flows with increasing depth, which tend to produce exponentially large sample values [Behrmann et al., 2021, Andrade, 2024]. One intuitive explanation is that we simply have no information on how the flow should behave far from the training samples, and an arbitrarily flexible model will at least in some cases behave unexpectedly.
If already learning a flow from samples drawn from the density itself is difficult, is it even possible to infer the belief density from preferential data? For instance, for the most popular RUM model (Plackett-Luce; Luce, 1959, Plackett, 1975) we cannot in the noiseless case differentiate between the true density and any normalised positive monotonic transformation of it:
Proposition 2.1 (Unidentifiability of a noiseless RUM). Let $p_{\star}$ be the expert's belief density. For $k \geq$ 2, let $\mathcal{D}_{\text {rank }}:=\left\{\mathbf{x}_{1} \succ \mathbf{x}_{2} \succ \ldots \succ \mathbf{x}_{k}\right\}$ be a $k$-wise ranking (see Definition 3.3). If $W \sim \operatorname{Gumbel}(0, \beta)$, then for any positive monotonic transformation $g$ holds $\lim _{\beta \rightarrow 0} p\left(\mathcal{D}_{\text {rank }} \mid g \circ p_{\star}, \beta\right)=1$. Proof in $B$.

In other words, the noiseless solution is not even unique and resolving this requires a way of quantifying the relative utility. Noisy RUM induces such a metric due to the noise magnitude providing a natural scale but even then the belief is identifiable only up to a noise scale; see A for a concrete example for the Thurstone-Mosteller model [Thurstone, 1927, Mosteller, 1951].
Another new challenge is that the candidates $\mathbf{x}$ presented to the expert are given by some external process. In the simplest case, they are drawn independently from some unknown distribution $\lambda(\mathbf{x})$, which does not need to relate to the belief density $p_{\star}$. We need a formulation that affords estimating $p_{\star}$ directly, ideally under minimal assumptions on the distribution besides $\lambda(\mathbf{x})>0$ for $p_{\star}(\mathbf{x})>0$.
Despite these challenges, we can indeed learn flows as estimates of belief densities as will be explained next, in part by leveraging standard machinery in discrete choice theory to model the expert's choices and in part by introducing a new functional prior for the normalizing flow. The choice process separates the candidate samples $\mathbf{x}$ into preferred and non-preferred ones, and we can use this split to construct a prior that helps learning the flow. That is, the preferential setup also opens new opportunities to address problems in learning flows.

## 3 Random utility model with exponentially distributed noises

The random utility model represents the decision maker's stochastic utility $U$ as the sum of a deterministic utility and a stochastic perturbation [Train, 2009],

$$
U(\mathbf{x})=f(\mathbf{x})+W(\mathbf{x})
$$

where $f: \mathcal{X} \rightarrow \mathbb{R}$ is a deterministic function called representative utility, and $W$ is a stochastic noise process, often independent across $\mathbf{x}$. The relationship between these concepts and the task will be made specific in Assumptions 1 to 3 . We assume that the domain $\mathcal{X}$ is a compact subset of $\mathbb{R}^{d}$. Given a set $\mathcal{C} \subset \mathcal{X}$ of possible alternatives, the expert selects a specific opinion $\mathbf{x} \in \mathcal{C}$ through the noisy utility maximization,

$$
\mathbf{x} \sim \underset{\mathbf{x}^{\prime} \in \mathcal{C}}{\arg \max } U\left(\mathbf{x}^{\prime}\right)
$$

Definition 3.1 (choice set). Let $k \geq 2$. The choice set is a set of $k$ alternatives, denoted by $\mathcal{C}_{k}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{k}\right\}$. We assume that $\mathcal{C}_{k}$ is a set of i.i.d. samples from a probability density $\lambda$ over $\mathcal{X}$, but note that the formulation can be generalized to other processes.

---

#### Page 4

For example, if we ask for a pairwise comparison $\mathcal{C}_{2}=\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$, the expert's answer would be $\mathbf{x} \succ \mathbf{x}^{\prime}$ if $f(\mathbf{x})+w(\mathbf{x})>f\left(\mathbf{x}^{\prime}\right)+w\left(\mathbf{x}^{\prime}\right)$ for given a realization $w$ of $W$. We denote the random utility model with a representative utility $f$, a stochastic noise process $W$, and a choice set $\mathcal{C}_{k}$, by $\operatorname{RUM}\left(\mathcal{C}_{k}, f, W\right)$.

We make the common assumption of representing the probabilistic beliefs of a (human) expert in logarithmic form [Dehaene, 2003].
Assumption 1. $f(\mathbf{x})=\log p_{\star}(\mathbf{x})$; noise is additive for log-density.
Assumption 2. $f$ is bounded and continuous.
Inspired by Malmberg and Hössjer [2012, 2014], we assume that the noise is exponentially distributed and thus belongs to the exponential family [Azari et al., 2012].
Assumption 3. $W(\mathbf{x}) \sim \operatorname{Exp}(s)$ independently for any $\mathbf{x} \in \mathcal{X}$
With Assumption 1, this corresponds to a model where in the limit of infinitely many alternatives, the expert chooses a point by sampling their belief density (Corollary A.2). The parameter $s$ is here a precision parameter, the reciprocal of the standard deviation of $\operatorname{Exp}(s)$. There are two popular types of preferential queries [Fürnkranz and Hüllermeier, 2011].
Definition 3.2 ( $k$-wise comparison). A preferential query that asks the expert to choose the most preferred alternative from $\mathcal{C}_{k}$ is called a $k$-wise comparison. The choice is denoted by $\mathbf{x} \succ \mathcal{C}_{k}$.
Definition 3.3 ( $k$-wise ranking). A preferential query that asks the expert to rank the alternatives in $\mathcal{C}_{k}$ from the most preferred to the least preferred is a called $k$-wise ranking. The expert's feedback is the ordering $\mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{1}} \succ \ldots \succ \mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{k}}$ for some permutation $\pi$.

Note that the top-ranked sample of k -wise ranking is the same as the k -wise comparison choice, and the k -wise ranking can be formed as a sequence of k -wise comparisons by repeatedly removing the selected candidate from the choice set, as assumed in the Plackett-Luce model [Plackett, 1975]. Hence, common theoretical tools cover both cases.

# 3.1 The $k$-wise winner

The chosen point of a $k$-wise comparison is central to us for two reasons. First, its distribution provides the likelihood for inference when data come in the format of $k$-wise rankings or comparisons. Second, its distribution in the limit as $k \rightarrow \infty$ offers insights for designing a prior that mitigates the challenge of collapsing and diverging probability mass.
Definition 3.4 ( $k$-wise winner). A random vector $X_{k}^{*}$ given by the following generative process is called as $k$-wise winner.

1. Sample $k$-samples from $\lambda(\mathbf{x})$, and denote $\mathcal{C}_{k}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{k}\right\}$.
2. Sample $\mathbf{x}$ from a Categorical distribution with support $\mathcal{C}_{k}$ and with probabilities given by $\operatorname{RUM}\left(\mathcal{C}_{k} ; \log p_{\star}(\mathbf{x}), \operatorname{Exp}(s)\right)$.
   The density of the $k$-wise winner is proportional to the $k$-wise comparison likelihood $p\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right)$, namely to $p\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right) \lambda\left(\mathcal{C}_{k}\right)$. The likelihood of the $k$-wise comparisons takes the following form.
   Proposition 3.5. Let $\mathcal{C}_{k}$ be a choice set of $k \geq 2$ alternatives. Denote $C=\mathcal{C}_{k} \backslash\{\mathbf{x}\}$ and $f_{C}^{\star}=$ $\max _{\mathbf{x}_{j} \in C} f\left(\mathbf{x}_{j}\right)$. The winning probability of a point $\mathbf{x} \in \mathcal{C}_{k}$ equals to
   $P\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right)=\sum_{l=0}^{k-1} \frac{\exp (-s(l+1) \max \left\{f_{C}^{\star}-f(\mathbf{x}), 0\right\})}{l+1} \sum_{\operatorname{sym} \mathbf{x}_{j} \in C}^{l}-\exp \left(-s\left(f(\mathbf{x})-f\left(\mathbf{x}_{j}\right)\right)\right)$,
   where $\sum_{\operatorname{sym} \mathbf{x}_{j} \in C}^{l}$ denotes the $l^{\text {th }}$ elementary symmetric sum of the set $C$.
   Proof. See B.
   The $k$-wise ranking likelihood is a product of the $k$-wise comparison likelihoods where the winners are sequentially removed from the choice set and provided in Appendix A as Equation (A.4).
   In the limit of infinitely many comparisons, the $k$-wise distribution reduces to a tempered belief density tilted by the sampling distribution $\lambda$ [Malmberg and Hössjer, 2012, 2014].

---

#### Page 5

> **Image description.** The image consists of two line graphs, labeled (a) and (b), each displaying several curves. Both graphs have the same x-axis, labeled "x", ranging from approximately -5 to 5, and the same y-axis, ranging from 0.0 to 0.4.
>
> In graph (a), there are seven curves. One is a dashed dark blue line labeled "p.". The other six are solid lines in varying shades of gray, labeled "k=10", "k=8", "k=6", "k=4", "k=3", and "k=2". All curves are bell-shaped, centered around x=0. As the value of 'k' decreases, the curves become flatter and wider. The dashed blue line "p." is the narrowest and tallest curve.
>
> In graph (b), there are six curves. One is a dashed dark blue line labeled "k=2". The other five are solid lines in varying shades of green, labeled "p^0.01", "p^0.05", "p^0.2", "p^0.5", "p^0.9", and "p.". These curves are also bell-shaped and centered around x=0. The curve labeled "p." is the narrowest and tallest, while "p^0.01" is the flattest.
>
> The graphs appear to be comparing different distributions or approximations, with the parameter 'k' and the exponent of 'p' influencing the shape of the curves.

Figure 2: (a) The $k$-wise winner distribution converges to the belief density as $k \rightarrow \infty$. (b) The $k$-wise winner distribution can be approximated by a tempered belief density. For example, the tempered belief density with an exponent $1 / 5$ approximates well the pairwise winner distribution.

Theorem 3.6. If $f$ is bounded and continuous, then the limit distribution of $X_{k}^{\star}$ as $k \rightarrow \infty$ exists and its density is given by,

$$
p(\mathbf{x})=\frac{\exp (s f(\mathbf{x})) \lambda(\mathbf{x})}{\int \exp \left(s f\left(\mathbf{x}^{\prime}\right)\right) \lambda\left(\mathbf{x}^{\prime}\right) d \mathbf{x}^{\prime}}
$$

Proof. Apply Theorem 18.4 in [Malmberg and Hössjer, 2014] to our setting and note that the first sentence in the proof of Theorem 18.4 is incorrect. For a random variable $Y=X / s$ with $X \sim \operatorname{Exp}(s)$ it holds that $Y \sim \operatorname{Exp}\left(s^{2}\right)$. However, for a random variable $Y=s X$ with $X \sim \operatorname{Exp}(s)$ it holds that $Y \sim \operatorname{Exp}(1)$. Thus, the correct standardization is $Y \leftarrow s Y$.

# 3.2 The $k$-wise winner distribution as a tilted and tempered belief density

Building on the definitions and theorems above, we now introduce the central idea of how to model the belief density based on the $k$-wise winner distribution. The RUM precision parameter $s$ acts as a temperature parameter for the belief density, as $p(\mathbf{x}) \propto \lambda(\mathbf{x}) p_{\star}(\mathbf{x})^{s}$, by Eq. (4). In general, there is no connection between $\lambda(\mathbf{x})$ and $p_{\star}(\mathbf{x})$, but intuition can be gained by considering some extreme cases. For $\lambda=p_{\star}$ we have $p(\mathbf{x}) \propto p_{\star}(\mathbf{x})^{s+1}$, whereas for uniform $\lambda(\mathbf{x})$ and $s=1$ the limit distribution is the belief. This is also apparent from Corollary A.2. However, our interest is in cases where $k$ is finite.

For $k<\infty$, forming the $k$-wise winner distribution requires marginalising over the choice set (Proposition 3.5). The formulas can be found in the Appendix (Corollary A.3), and do not have elegant analytic simplifications. However, they empirically resemble tempered versions of the actual belief as illustrated in Figure 2. In other words, finite $k$ plays a similar role as the RUM noise precision $s$. When resorting to $k<\infty$, the choice distribution (Eq. (A.2)) does not match the belief density for the true noise precision $s$, but we can improve the fit by selecting some alternative noise precision such that the choice distribution better approximates the belief. We will later use this connection to build a prior over the flow, and note that for this purpose we do not need an exact theoretical characterization: It is sufficient to know that for some choice of $k$ and $s$ the choice distribution can resemble the target density, at least to the degree that it can be used as a basis for prior information. Given that $k$ is typically fixed, $s$ can be varied in the prior, implying that the further $s$ is from the 'optimal' value, the greater the prior misspecification.
The idea is empirically illustrated in Figure 2. The $k$-wise winner distribution for varying $k$ is shown in Figure 2(a), where the belief density is a truncated standard normal on the interval $[-5,5]$, comparisons are sampled uniformly over the interval, and $s=1$. As $k$ increases, the $k$-wise winner distribution approaches the belief density (here $k=10$ is already very close), but we can equivalently approach the same density by changing the noise level (Figure 2(b)).

---

#### Page 6

# 4 Belief density as normalizing flow

We model the belief density $p_{\star}$ with a normalizing flow [Rezende and Mohamed, 2015, Papamakarios et al., 2021]. We introduce a new learning principle and objective for the flow which is compatible with any standard flow architecture, as long as it affords easy computation of the flow density. A normalizing flow is an invertible mapping $T$ from a latent space $\mathcal{Z} \subset \mathbb{R}^{d}$ into a target space $\mathcal{X} \subset \mathbb{R}^{d}$. $T$ consists of a sequence of invertible transformations, so that the forward (generative) direction $\mathbf{z} \mapsto T(\mathbf{z})$ is fast to compute and the backward (normalizing) direction $\mathbf{x} \mapsto T^{-1}(\mathbf{x})$ is either known in closed form or can be approximated efficiently.
The base distribution on $\mathcal{Z}$ is a simple distribution such as a multivariate normal, whose density is denoted by $p_{z}$. If we denote the parametrized $T$ by $T_{\phi}$ given the flow network parameters $\phi$, the parameterized model of the log belief density, denoted by $f_{\phi}$, can be written as,

$$
f_{\phi}(\mathbf{x})=\log p_{z}\left(T_{\phi}^{-1}(\mathbf{x})\right)+\log \left|\operatorname{det} J_{T_{\phi}^{-1}}(\mathbf{x})\right|
$$

where $J_{T_{\phi}^{-1}}$ is the Jacobian of $T_{\phi}^{-1}$. What complicates the learning of $f_{\phi}$ in our case is the absence of a direct method to sample from $p_{\star}(\mathbf{x})$, ruling out the conventional algorithms [e.g., Papamakarios et al., 2021]. Instead, we devise a new learning objective explained next.

### 4.1 Function-space Bayesian inference

Our starting point is to perform Bayesian inference for the flow network parameters given preferential dataset $\mathcal{D}=\left\{\left(\mathbf{x}^{(i)}, \mathcal{C}_{k}^{(i)}\right) \mid \mathbf{x}^{(i)} \succ \mathcal{C}_{k}^{(i)}\right\}_{i=1}^{n}$ ( $k$-wise comparisons) or $\mathcal{D}=\left\{\left(\sigma^{(i)}, \mathcal{C}_{k}^{(i)}\right) \mid\right.$ $\sigma^{(i)}$ is a permutation on $\mathcal{C}_{k}^{(i)}\}_{i=1}^{n}$ ( $k$-wise rankings),

$$
p(\phi \mid \mathcal{D}) \propto p(\mathcal{D} \mid \phi) p(\phi)
$$

where $p(\mathcal{D} \mid \phi)$ is the likelihood and $p(\phi)$ is the prior for the flow network parameters. It is difficult to devise a good prior $p(\phi)$, and we instead perform inference over the function [Wolpert, 1993],

$$
p\left(f_{\phi} \mid \mathcal{D}\right) \propto p\left(\mathcal{D} \mid f_{\phi}\right) p\left(f_{\phi}\right)
$$

where $p\left(\mathcal{D} \mid f_{\phi}\right)$ is the preferential likelihood for comparisons, Eq. (3), or rankings, Eq. (A.4). The function-space prior is easier to specify as we can focus on the characteristics of the log belief density itself, not its parametrization. The function-space prior is often evaluated at a finite set of representer points $\bar{X}=\left(\hat{\mathbf{x}}_{1}, \ldots, \hat{\mathbf{x}}_{m}\right)$, where $m$ should be chosen to be large to capture the behavior of the function at high resolution $p\left(f_{\phi}(\bar{X})\right)$ [Wolpert, 1993, Qiu et al., 2024]. For example, when $f$ is a Gaussian process, the prior representer points in the posterior corresponds to the datapoints (e.g., Equation 3.12 in Rasmussen and Williams, 2006). Following the considerations above, we construct our prior knowledge of $f_{\phi}$ on a subset of datapoints.

### 4.2 Empirical functional prior

To address the issue of collapsing or diverging probability mass, we introduce an empirical functional prior whose finite marginals at winner points $\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right\}$ are independently distributed as

$$
p(\mathbf{f}) \propto p_{\text {unif }}(\mathbf{f}) \prod_{i} \exp \left(\mathbf{f}_{i}\right)
$$

where $\mathbf{f}:=\left(f_{\phi}\left(\mathbf{x}_{1}\right), \ldots, f_{\phi}\left(\mathbf{x}_{n}\right)\right)$ and $p_{\text {unif }}$ is an uninformative bounded (hyper) prior that guarantees that the functional prior is proper.
The functional prior Eq. (6) is a special case of a class of priors, $p(\mathbf{f}) \propto \prod_{i} \boldsymbol{\lambda}_{i} \exp \left(s \mathbf{f}_{i}\right)$, derived from the following decision-theoretic argument under the exponential RUM model. Let us decompose the preference dataset into winners and losers $\mathcal{D}_{k}=\mathcal{D}_{k}^{+} \cup \mathcal{D}_{k}^{\neq}$by defining $\mathcal{D}_{k}^{+} :=\left\{\mathbf{x} \mid \exists \mathcal{C}_{k}\right.$ s.t. $\left.\mathbf{x} \succ \mathcal{C}_{k}\right\}$ and $\mathcal{D}_{k}^{\neq}:=\mathcal{D}_{k} \backslash \mathcal{D}_{k}^{+}$. The functional prior is the probability of observing only the $k$-wise winners,

$$
p(\mathbf{f}) \propto p\left(\mathcal{D}_{k}^{+} \mid \mathbf{f}, s, \lambda\right) p_{\text {unif }}(\mathbf{f})
$$

where $p_{\text {unif }}(\mathbf{f}) \propto 1$ (when $f$ is bounded, Assumption 2). The idea is to consider higher $k$ or $s$ (less noise) than the true ones, as both choices make the density more peaked around the modes of $p_{\star}$ (see

---

#### Page 7

Figures 2(a) and 2(b)). This choice encourages the flow to place more mass on the winner points in a way that is consistent with the underlying decision model. We consider $k=\infty$ and $s \in(0, \infty)$, where $s$ should be an increasing function of the true $k$. While setting $k=\infty$ reduces the functional prior to a closed form Eq. (4) by Theorem 3.6, the normalizing constant remains difficult. However, for the special case of $\lambda \propto 1$ and $s=1$, the normalizing constant equals one. We make this choice to retain computational tractability, reminding that the construct is only used as a prior intended for regularizing the solution and does not need to match the true density as such. This comes at the cost of increased prior misspecification, which can, in turn, degrade the quality of the fit, especially when the true value of $k$ is small (compare Figure 4(a) ( $\mathrm{k}=2$ ) versus Figure 1(d) ( $\mathrm{k}=5$ )).

# 4.2.1 Function-space maximum a posteriori

We train the flow $T_{\phi}$ by maximizing the unnormalized function-space posterior density of $f_{\phi}$ conditioned on the preferential data $\mathcal{D}=\mathcal{D}^{\succ} \cup \mathcal{D}^{\neq}$, using stochastic gradient ascent [Kingma and Ba, 2014]. Denoting all points in $\mathcal{D}$ by $\mathbf{X}$ and all winner points in $\mathcal{D}^{\succ}$ by $\mathbf{X}_{s}$, the training objective is

$$
\sum \log \mathcal{L}\left(\mathcal{D} \mid f_{\phi}(\mathbf{X}), s\right)+\sum f_{\phi}\left(\mathbf{X}_{s}\right)
$$

where $\mathcal{L}$ is the $k$-wise comparison or ranking likelihood as per Eqs. (3) and (A.4). In the case of ranking data, the winner point $\mathbf{x} \in \mathbf{X}_{s}$, is the first-ranked alternative in each individual $k$-wise ranking, meaning that $\mathbf{x}$ is a $k$-wise winner. A global optimum of Eq. (7) is the function-space maximum a posteriori estimate of the belief density. Pseudo-codes for the overall algorithm (Algorithms 1) and the forward pass for the unnormalized log-posterior (Algorithms 2) are provided in the Appendix. The computational cost of training is similar to standard flow learning from equally many samples.

## 5 Experiments

We first evaluate our method on synthetic data with choices made by simulating the RUM model, to validate the algorithm in cases where the ground truth is known while covering both cases where the responses follow the assumed RUM model and where they do not. We then demonstrate how the method could be used in a realistic elicitation scenario, using a large language model (LLM) as a proxy for a human expert [Choi et al., 2022]. As with a real human, an LLM is unlikely to follow the exact RUM model, but compared to a real user, the LLM expert can tirelessly answer unlimited questions and possible ethical issues and risks relating to human subjects are avoided. LLMs carry their own biases and risks [Tjuatja et al., 2024], but the focus here is on evaluating our algorithm. Code for reproducing all experiments is available at https://github.com/petrus-mikkola/prefflow.

Setup In the main experiments we use $k$-wise ranking with $k=5$, using relatively few queries to remain relevant for the intended use-cases where the expert's capacity in providing the information is clearly limited. Since learning a preference of higher dimensions is more difficult, we scale the number of queries $n$ linearly with $d$ but still stay substantially below the large-sample scenarios typically considered in flow learning. The details, together with the choice of the flow and the candidate distribution $\lambda$, are provided below for each experiment. As a flow model, we use RealNVP [Dinh et al., 2017] when $d=2$ and Neural Spline Flow [Durkan et al., 2019] when $d>2$, implemented on top of [Stimper et al., 2023]. For more details, see Appendix C.4.

Evaluation We assess performance qualiatively via visual comparison of $2 d$ and $1 d$ marginal distributions between the target belief density and the flow estimate of the belief density, and quantitatively by numerically computing three metrics: the log-likelihood of the preferential data, the Wasserstein distance, and the mean marginal total variation distance (MMTV; Acerbi, 2020) between the target and the estimate. The numerical results are reported as the means and standard deviations of the metrics over replicate runs. As a baseline, we report the results of a method that uses the same preferential comparisons and optimizes the same training objective, but instead of using a flow to represent $\exp (f)$ we directly assume the density is a factorized normal distribution parameterized by means and (log-transformed) standard deviations of all dimensions. This exact method has not been presented in the previous literature, but was designed to validate the merit of the flow representation.

---

#### Page 8

Table 1: Accuracy of the density represented as a flow (flow) compared to a factorized normal distribution (normal), both learned from preferential data, in three metrics: log-likelihood, Wasserstein distance, and the mean marginal total variation (MMTV). Averages over 20 repetitions (but excluding a few crashed runs), with standard deviations.

|                 | log-likelihood $(\uparrow)$ |                      |    wasserstein    |                  | MMTV $(\downarrow)$ |                  |
| :-------------- | :-------------------------: | :------------------: | :---------------: | :--------------: | :-----------------: | :--------------: |
|                 |           normal            |         flow         |      normal       |       flow       |       normal        |       flow       |
| Onemoon2D       |    -1.98 ( $\pm 0.12$ )     | -1.09 ( $\pm 0.12$ ) | $0.45(\pm 0.04)$  | $0.25(\pm 0.04)$ |  $0.30(\pm 0.02)$   | $0.21(\pm 0.02)$ |
| Gaussian6D      |    -1.40 ( $\pm 0.07$ )     | -0.12 ( $\pm 0.02$ ) | $1.74(\pm 0.06)$  | $1.29(\pm 0.05)$ |  $0.20(\pm 0.01)$   | $0.09(\pm 0.01)$ |
| Twogaussians10D |    -3.99 ( $\pm 0.06$ )     | -0.09 ( $\pm 0.01$ ) | $7.31(\pm 0.12)$  | $2.60(\pm 0.06)$ |  $0.47(\pm 0.01)$   | $0.08(\pm 0.00)$ |
| Twogaussians20D |    -6.35 ( $\pm 0.12$ )     | -0.08 ( $\pm 0.01$ ) | $11.07(\pm 0.15)$ | $4.55(\pm 0.07)$ |  $0.47(\pm 0.00)$   | $0.08(\pm 0.00)$ |
| Funnel10D       |    -2.21 ( $\pm 0.06$ )     | -0.09 ( $\pm 0.01$ ) | $5.13(\pm 0.04)$  | $3.92(\pm 0.04)$ |  $0.27(\pm 0.00)$   | $0.18(\pm 0.01)$ |
| Abalone7D       |    -5.53 ( $\pm 0.03$ )     | -2.16 ( $\pm 0.12$ ) | $0.53(\pm 0.00)$  | $0.34(\pm 0.01)$ |  $0.26(\pm 0.00)$   | $0.29(\pm 0.01)$ |

> **Image description.** The image presents a series of plots arranged in a matrix format, displaying relationships between different variables. It appears to be a visualization technique used to explore the joint distribution of multiple variables. The image is divided into two main sections, each containing a similar matrix of plots.
>
> **Left Section:**
>
> - This section consists of a 3x3 matrix of plots, examining the relationships between "Length", "Diameter", and "Height".
> - The diagonal plots display the marginal distributions of each variable. These plots show probability density curves, with both a black and a purple line plotted.
> - The off-diagonal plots show scatter plots representing the joint distributions of pairs of variables. These scatter plots use a color gradient, with darker colors indicating higher densities of points. The scatter plots show the relationship between "Length" and "Diameter", "Length" and "Height", and "Diameter" and "Height".
>
> **Right Section:**
>
> - This section consists of a 4x4 matrix of plots, examining the relationships between "HouseAge", "AveRooms", "AveBedrms", and "Population".
> - The diagonal plots display the marginal distributions of each variable. The "HouseAge" plot is a step plot or histogram. The other diagonal plots are histograms.
> - The off-diagonal plots show scatter plots representing the joint distributions of pairs of variables. These scatter plots use a grayscale gradient, with darker shades indicating higher densities of points.
>
> The arrangement of the plots in a matrix format allows for a concise visualization of the relationships between all pairs of variables. The diagonal plots provide information about the individual distributions of each variable, while the off-diagonal plots show how the variables are related to each other.

Figure 3: Cross-plot of selected variables of the estimated flow in the Abalone (left) and LLM knowledge elicitation experiment (middle), and the marginal density of the same variables for the ground truth density in the LLM experiment (right). See Figures C. 6 and C. 7 for other variables.

# 5.1 Synthetic tasks

First, we study the method on synthetic scenarios. For the first set of experiments, we assume a known density $p_{\star}$ and simulate the preferential responses from the assumed $\operatorname{RUM}\left(\mathcal{C}_{k}, \log p_{\star}, \operatorname{Exp}(1)\right)$. We consider five different target distributions: Onemoon2D, Gaussian6D, Twogaussians10D, Twogaussians20D, and Funnel10D. The densities of the target distributions can be found in Appendix C.1. For all cases we used $100 d$ queries and $\lambda$ as a mixture of uniform and Gaussian distribution centered on the mean of the target, with the mixture probability $1 / 3$ for the Gaussian; this technical simplification ensures a sufficient ratio of the samples to align with the target density even when $d$ is high. Table 1 shows that for all scenarios we can learn a useful flow; all metrics are substantially improved compared to the method based on the normal model and visual inspection (Appendix C.5) confirms the solutions match the true densities well.
Abalone regression data. Having validated the method when the queries follow the assumed RUM model with a synthetic belief density, we consider a more realistic target density. We first fit a flow model to the continuous covariates of the regression data abalone [Nash et al., 1995], and then use the fitted flow as a ground-truth belief density in the elicitation experiment. The elicitation queries correspond to all $k$-combinations of the dataset size $n=4177$. The numerical results are again provided in Table 1. Figure 3 shows that the learned flow captures the correlations between variables almost perfectly, which can be hard to see as the flow (heatmap) overlaps the true density (contour). There is some mismatch in the marginals, which is also indicated by the MMTV metric. In terms of the Wasserstein distance and visual comparison (Figure C.8), the flow based method clearly outperforms the baseline.

---

#### Page 9

> **Image description.** The image shows two panels, (a) and (b), displaying visualizations of belief densities.
>
> Panel (a), labeled "(a) Onemoon2D, k = 2, n = 100", shows a 2D density plot. The background color ranges from dark purple to yellow, indicating density, with yellow representing the highest density. There is a prominent crescent-shaped region of high density in the lower-left quadrant. Contour lines are drawn around this high-density area. Red plus signs and blue horizontal lines are scattered across the plot, likely representing samples or data points.
>
> Panel (b), labeled "(b) Gaussian6D, k = 2, n = 1000", displays a matrix of plots. The matrix is arranged in a triangular format. The diagonal elements are line plots showing density estimates. The off-diagonal elements are 2D density plots similar to the one in panel (a), with the same color scheme indicating density. The axes of these plots are labeled with numerical values, presumably representing the range of the variables. The plots in panel (b) appear to represent pairwise relationships between variables in a 6-dimensional space.

Figure 4: Illustration of belief densities elicited from pairwise comparisons by a normalizing flow.

# 5.2 Expert elicitation with LLM as the expert

In this experiment, we prompt a LLM to provide its belief on how the features of the California housing dataset [Pace and Barry, 1997] are distributed. This resembles a hypothetical expert elicitation scenario, but the human expert is replaced with a LLM (Claude 3 Haiku by Anthropic in March 2024, see Appendix C. 2 for the prompt and detailed setup) for easier experimentation. From the perspective of the flow learning algorithm the setup is similar to the intended use-cases.

We query in total $220 k$-wise rankings through prompting, where the alternatives $\mathcal{C}_{k}$ are uniformly sampled over the domain specified by 1 st and 99 th percentiles of each variable in the California housing dataset. The range was chosen to ensure $\lambda(x)$ covers approximately the support of the density, but avoiding outliers. While we lack access to the ground-truth belief density, we can compare the learned LLM's belief density to the empirical data distribution of the California housing dataset, not known for the LLM. Figure 3 shows that there is a remarkable similarity between the distributions such as the marginals of the variables AveRooms, AveBedrms, Population, and AveOccup are all correctly distributed on the lower ends of their ranges (which are very broad due to the uniform $\lambda(x)$ ). Figure D. 1 shows that the flow trained without the functional prior of (6) is considerably worse, confirming the FS-MAP estimate is superior to maximum likelihood. While there might be multiple mechanisms for how the LLM forms its knowledge about this specific dataset [Brown et al., 2020], many of the features have clear intuitive meaning. For instance, houses are all but guaranteed to have only a few bedrooms, instead of tens.

### 5.3 Ablation study

We validate the sensitivity of the results with respect to the cardinality of the choice set $k$, the number of comparisons/rankings $n$, the noise level $1 / s$, and the choice of distribution $\lambda$ from which the candidates are sampled. In this section, we report a subset of the analysis for varying $k$, while the rest can be found in Appendix D. Table 2 presents the results of experiments on synthetic scenarios (Section 5.1) by varying $k \in\{2,3,5,10\}$ while keeping $n$ fixed. We observe that the accuracy naturally improves as a function of $k$. The common special case in which the expert is queried through pairwise comparisons $(k=2)$ is shown in Figure 4 for the Onemoon2D experiment with $n=100$ and the Gaussian6D experiment with $n=1000$. The results indicate that we can already roughly learn the target with $k=2$ that is most convenient for a user, but naturally with somewhat lower accuracy. For further analysis and more details, see Appendix D. The main takeaway is that low values of $s$ or $k$, especially when $n$ is large, can cause the flow estimate to become overly dispersed due to higher prior misspecification.

## 6 Discussion

Theoretical and empirical analysis validate our main claim: It is possible to learn flexible distributions from preferential data, and the proposed algorithm solves the problem for some non-trivial but

---

#### Page 10

Table 2: Wasserstein distances for varying $k$ across different experiments

|                           |       $k=2$       |       $k=3$       |       $k=5$       |      $k=10$       |
| :------------------------ | :---------------: | :---------------: | :---------------: | :---------------: |
| Onemoon2D $(n=100)$       | $0.70( \pm 0.09)$ | $0.39( \pm 0.05)$ | $0.17( \pm 0.03)$ | $0.11( \pm 0.03)$ |
| Gaussian6D $(n=100)$      | $2.69( \pm 0.30)$ | $2.01( \pm 0.25)$ | $1.46( \pm 0.11)$ | $1.04( \pm 0.04)$ |
| Funnel10D $(n=500)$       | $4.82( \pm 0.12)$ | $4.36( \pm 0.12)$ | $3.96( \pm 0.05)$ | $3.83( \pm 0.04)$ |
| Twogaussians10D $(n=500)$ | $5.47( \pm 0.24)$ | $3.81( \pm 0.26)$ | $2.57( \pm 0.08)$ | $2.20( \pm 0.02)$ |

synthetic scenarios, with otherwise arbitrary but largely unimodal true beliefs. However, open questions worthy of further investigation remain on the path towards practical elicitation tools.
The method is efficient only for exponential noise with $s=1$ that gives an analytic prior. Other choices would require explicit normalization or energy-based modeling techniques [Chao et al., 2024]. For a given RUM precision we can, in principle, solve for $k$ such that $s=1$ becomes approximately correct due to the tempering interpretation (Figure 2), but there are no guarantees that $s=1$ is good enough for any $k$ sufficiently small for practical use, and this requires an explicit estimate of the noise precision. The ablation studies show that for fixed $k$, increasing $n$ generally improves the accuracy and already fairly small $n$ is sufficient for learning a good estimate (Table D.2). For very large $n$, the accuracy can slightly deteriorate. We believe that this is due to prior misspecification that encourages overestimation of the variation due to the fact that $k$ is finite but in the prior it is assumed to be infinite. Figure D. 4 confirms that for a large $n$ the shape of the estimate is extremely close to the target density and the slightly worse Wasserstein distance is due to overestimating the width.
We primarily experimented with k -wise ranging with $k=5$ and relatively few comparisons. However, we demonstrated that we can learn the beliefs with somewhat limited accuracy already from the most convenient case of pairwise comparisons $(k=2)$, which is important for practical applications. Finally, we focused on the special case of sampling the candidates independently from $\lambda(\mathbf{x})$. In many elicitation scenarios they could be result of some active choice instead, for example an acquisition function maximizing information gain [MacKay, 1992]. The basic learning principle generalizes for this setup, but additional work would be needed for theoretical and empirical analysis.

# 7 Conclusions

The current tools for representing and eliciting multivariate human beliefs are fundamentally limited. This limits the value of knowledge elicitation in general and introduces biases that are difficult to analyze and communicate when the true beliefs do not match the simplified assumed families. Modern flexible distribution models offer a natural solution for representing also complex human beliefs, but until now we have lacked the means of inferring them from ecologically valid human judgements. We provided the first such method by showing how normalizing flows can be estimated from preferential data using a functional prior induced by the setup. Our focus was in specifying the problem setup and validating the computational algorithm, paving way for future applications involving real human judgements. Despite focusing on the scenario where the elicitation judgements are made by a human expert, the algorithm can be used for learning flows from all kinds of comparison and ranking data.

---

# Preferential Normalizing Flows - Backmatter

---

## Broader Impact

Our goal is to eventually provide methods for accurately characterising human expert beliefs, complementing the existing toolbox with techniques that make less restrictive assumptions and hence support adaptation of better computational tools in a broad range of applications. Knowledge elicitation tools are frequently used e.g. in decision-making and policy recommendations as assistive tools. For such applications, it is critically important to ensure that the mathematical tools are reliable and transparent, and further validation of the methodology is needed before the specific method proposed here could be used in critical applications.

## Acknowledgments and Disclosure of Funding

The work was supported by the Research Council of Finland Flagship programme: Finnish Center for Artificial Intelligence FCAI, and by the grants 345811, 358980, 356498, and 363317. The authors acknowledge support from CSC - IT Center for Science, Finland, for computational resources.

---

#### Page 11

# References

Luigi Acerbi. Variational Bayesian Monte Carlo with noisy likelihoods. Advances in Neural Information Processing Systems, 33:8211-8222, 2020.

Daniel Andrade. Stable training of normalizing flows for high-dimensional variational inference. arXiv preprint arXiv:2402.16408, 2024.

Hossein Azari, David Parks, and Lirong Xia. Random utility theory for social choice. In Advances in Neural Information Processing Systems, volume 25, 2012.

Jens Behrmann, Paul Vicol, Kuan-Chieh Wang, Roger Grosse, and Joern-Henrik Jacobsen. Understanding and mitigating exploding inverses in invertible neural networks. In International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 1792-1800. PMLR, 2021.

Christopher M Bishop and Hugh Bishop. Deep learning: Foundations and concepts. Springer Nature, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877-1901, 2020.

Chen-Hao Chao, Wei-Fang Sun, Yen-Chang Hsu, Zsolt Kira, and Chun-Yi Lee. Training energy-based normalizing flow with score-matching objectives. Advances in Neural Information Processing Systems, 36, 2024.

Kristy Choi, Chris Cundy, Sanjari Srivastava, and Stefano Ermon. LMPriors: Pre-trained language models as task-specific priors. arXiv preprint arXiv:2210.12530, 2022.

Robert T. Clemen, Gregory W. Fischer, and Robert L. Winkler. Assessing dependence: Some experimental results. Management Science, 46(8):1100-1115, 2000. ISSN 00251909, 15265501.

Rob Cornish, Anthony Caterini, George Deligiannidis, and Arnaud Doucet. Relaxing bijectivity constraints with continuously indexed normalising flows. International Conference on Machine Learning, pages 2133-2143, 2020.

Stanislas Dehaene. The neural basis of the Weber-Fechner law: a logarithmic mental number line. Trends in cognitive sciences, 7(4):145-147, 2003.

Akash Kumar Dhaka, Alejandro Catalina, Manushi Welandawe, Michael R Andersen, Jonathan Huggins, and Aki Vehtari. Challenges and opportunities in high dimensional variational inference. Advances in Neural Information Processing Systems, 34:7787-7798, 2021.

Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In International Conference on Learning Representations, 2017.

Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In Advances in Neural Information Processing Systems, volume 32, 2019.

Johannes Fürnkranz and Eyke Hüllermeier. Preference Learning. Springer-Verlag New York, Inc., 2011.

Andrew Gelman, Daniel Simpson, and Michael Betancourt. The prior can often only be understood in the context of the likelihood. Entropy, 19(10):555, 2017.

---

#### Page 12

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454-12465, 2021.

Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.
D. Jennings, T. M. Amabile, and L. Ross. Informal covariation assessment: Data-based vs. theorybased judgments. In Daniel Kahneman, Paul Slovic, and Amos Tversky, editors, Judgment Under Uncertainty: Heuristics and Biases, pages 211-230. Cambridge University Press, 1982.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018.

Jia Li, Surajit Ray, and Bruce G Lindsay. A nonparametric statistical approach to clustering via mode identification. Journal of Machine Learning Research, 8(8), 2007.

Feynman Liang, Michael Mahoney, and Liam Hodgkinson. Fat-tailed variational inference with anisotropic tail adaptive flows. In International Conference on Machine Learning, pages 1325713270. PMLR, 2022.

R Duncan Luce. Individual choice behavior, volume 4. Wiley New York, 1959.
David JC MacKay. Information-based objective functions for active data selection. Neural computation, 4(4):590-604, 1992.

Hannes Malmberg and Ola Hössjer. Argmax over continuous indices of random variables-an approach using random fields. Technical report, Technical report, Division of Mathematical Statistics, Department of Mathematics, Stockholm University, 2012.

Hannes Malmberg and Ola Hössjer. Probabilistic choice with an infinite set of options: an approach based on random sup measures. Modern Problems in Insurance Mathematics, pages 291-312, 2014.

Jacob Marschak. Binary choice constraints on random utility indicators. Technical report, Cowles Foundation for Research in Economics, Yale University, 1959.

Petrus Mikkola, Osvaldo A. Martin, Suyog Chandramouli, Marcelo Hartmann, Oriol Abril Pla, Owen Thomas, Henri Pesonen, Jukka Corander, Aki Vehtari, Samuel Kaski, Paul-Christian Bürkner, and Arto Klami. Prior Knowledge Elicitation: The Past, Present, and Future. Bayesian Analysis, pages $1-33,2023$.

Frederick Mosteller. Remarks on the method of paired comparisons: The least squares solution assuming equal standard deviations and equal correlations. Psychometrika, 16(1):3-9, 1951.

Warwick Nash, Tracy Sellers, Simon Talbot, Andrew Cawthorn, and Wes Ford. Abalone. UCI Machine Learning Repository, 1995.

Didrik Nielsen, Priyank Jaini, Emiel Hoogeboom, Ole Winther, and Max Welling. SurVAE flows: Surjections to bridge the gap between VAEs and flows. Advances in Neural Information Processing Systems, 33:12685-12696, 2020.

Jeremy E. Oakley and Anthony O'Hagan. Uncertainty in prior elicitations: A nonparametric approach. Biometrika, 94, 2007.

Anthony O’Hagan. Expert knowledge elicitation: Subjective but scientific. The American Statistician, 73(sup1):69-81, 2019.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730-27744, 2022.

---

#### Page 13

R Kelley Pace and Ronald Barry. Sparse spatial autoregressions. Statistics \& Probability Letters, 33 (3):291-297, 1997.

George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. The Journal of Machine Learning Research, 22(1):2617-2680, 2021.

Dmytro Perepolkin, Benjamin Goodrich, and Ullrika Sahlin. Hybrid elicitation and quantileparametrized likelihood. Statistics and Computing, 34(1):11, 2024.

Robin L Plackett. The analysis of permutations. Journal of the Royal Statistical Society Series C: Applied Statistics, 24(2):193-202, 1975.

Shikai Qiu, Tim GJ Rudner, Sanyam Kapoor, and Andrew G Wilson. Should we learn most likely functions or parameters? Advances in Neural Information Processing Systems, 36, 2024.

Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian processes for machine learning, volume 2. MIT press, 2006.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pages 1530-1538. PMLR, 2015.

Antoine Salmona, Valentin De Bortoli, Julie Delon, and Agnes Desolneux. Can push-forward generative models fit multimodal distributions? Advances in Neural Information Processing Systems, 35:10766-10779, 2022.

Vincent Stimper, David Liu, Andrew Campbell, Vincent Berenz, Lukas Ryll, Bernhard Schölkopf, and José Miguel Hernández-Lobato. normflows: A PyTorch package for normalizing flows. Journal of Open Source Software, 8(86):5361, 2023.
L. L. Thurstone. A law of comparative judgment. Psychological Review, 34(4):273-286, 1927.

Lindia Tjuatja, Valerie Chen, Tongshuang Wu, Ameet Talwalkwar, and Graham Neubig. Do LLMs exhibit human-like response biases? A case study in survey design. Transactions of the Association for Computational Linguistics, 12:1011-1026, 2024.

Kenneth E Train. Discrete choice methods with simulation. Cambridge university press, 2009.
Lorenz Vaitl, Kim A Nicoli, Shinichi Nakajima, and Pan Kessel. Gradients should stay on path: better estimators of the reverse-and forward KL divergence for normalizing flows. Machine Learning: Science and Technology, 3(4):045006, 2022.

Alyson G Wilson. Cognitive factors affecting subjective probability assessment, volume 4. Citeseer, 1994.

David H Wolpert. Bayesian backpropagation over io functions rather than weights. Advances in Neural Information Processing Systems, 6, 1993.

John I Yellott Jr. The relationship between Luce's choice axiom, Thurstone's theory of comparative judgment, and the double exponential distribution. Journal of Mathematical Psychology, 15(2): $109-144,1977$.

---

# Preferential Normalizing Flows - Appendix

---

#### Page 14

# A Theoretical results

In Section 2 we mentioned that noisy RUM models can be identified for known noise levels. This can be illustrated by the following example:
Example A.1. Consider a probit model (Thurstone-Mosteller) model [Thurstone, 1927, Mosteller, 1951]. Denote the probability mass function by $p$, its values at two points $p(\mathbf{x})$ and $p\left(\mathbf{x}^{\prime}\right)$, and their difference by $\Delta p=p(\mathbf{x})-p\left(\mathbf{x}^{\prime}\right)$. For a sufficiently large data set of pairwise comparisons, we can estimate the winning probability of $\mathbf{x}: \mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime}\right)=q$. Since noise follows $N\left(0, \sigma^{2}\right)$, we can deduce that $\mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime}\right)=\Phi_{\sigma^{2}}(\Delta p)$, where $\Phi_{\sigma^{2}}$ is the cumulative distribution function of $N\left(0, \sigma^{2}\right)$. So, $\mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime}\right)=\Phi_{\sigma^{2}}(\Delta p)$ and $\Delta p=\Phi_{\sigma^{2}}^{-1}(q)$. Since $p$ is the probability mass function, from a pair of equations we obtain $p(\mathbf{x})=\left(1-\Phi_{\sigma^{2}}^{-1}(q)\right) / 2$ and $p\left(\mathbf{x}^{\prime}\right)=\left(1+\Phi_{\sigma^{2}}^{-1}(q)\right) / 2$. For the known noise level $\sigma^{2}, p$ is identified.

Section 3 refers to the following corollary that relates the RUM model with sampling.
Corollary A.2. Consider that presenting an $\infty$-wise comparison with the choice set $\mathcal{C}=\mathcal{X}$ to the expert is equivalent to presenting a $k$-wise comparison with large $k$ and points sampled uniformly over $\mathcal{X}$. If the expert choice model follows $\operatorname{RUM}\left(\mathcal{X} ; \log p_{\star}(\mathbf{x})\right.$, $\left.\operatorname{Exp}(1)\right)$, then asking the expert to pick the most likely alternative out of all possible alternatives is equivalent to sampling from their belief density.

Proof. Since $\lambda(\mathbf{x})=1 / \operatorname{vol}(\mathcal{X})$, the terms $\lambda(\mathbf{x})$ and $\lambda\left(\mathbf{x}^{\prime}\right)$ in Eq. (4) cancel out. The denominator equals $\int p_{\star}(\mathbf{x}) d \mathbf{x}=1$, because $f(\mathbf{x})=\log p_{\star}(\mathbf{x})$ and $s=1$. Thus, $p(\mathbf{x} \succ \mathcal{X})=\exp (s f(\mathbf{x}))=$ $p_{\star}(\mathbf{x})$.
Corollary A.3. For $k=2$, the probability density of $X_{k}^{\star}$ equals to

$$
p_{X_{k}^{\star}}(\mathbf{x})=2 \lambda(\mathbf{x}) \int_{\mathcal{X}} F_{\text {Laplace }(0, l / \delta)}\left(p_{\star}(\mathbf{x})-p_{\star}\left(\mathbf{x}^{\prime}\right)\right) \lambda\left(\mathbf{x}^{\prime}\right) d \mathbf{x}^{\prime}
$$

For $2<k<\infty$, the probability density of $X_{k}^{\star}$ is proportional to

$$
p_{X_{k}^{\star}}(\mathbf{x}) \propto \lambda(\mathbf{x}) \int_{\mathcal{X}^{k-1}} P\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right) d \lambda\left(\mathcal{C}_{k} \backslash\{\mathbf{x}\}\right)
$$

where $P\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right)$ is given by Proposition 3.5.
For $k=\infty$, the probability density of $X_{k}^{\star}$ equals to

$$
p_{X_{k}^{\star}}(\mathbf{x})=C \lambda(\mathbf{x}) p_{\star}^{\star}(\mathbf{x})
$$

where $C>0$. If $\lambda(\mathbf{x})=1 / \operatorname{vol}(\mathcal{X})$ and $s=1$, then $C \lambda(\mathbf{x})=1$.
Proof. Case $k=2$.

$$
p_{X_{k}^{\star}}(\mathbf{x}) \propto \int_{\mathcal{X}} \mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime} \mid p_{\star}, s\right) \lambda\left(\mathbf{x}^{\prime}\right) \lambda(\mathbf{x}) d \mathbf{x}^{\prime}
$$

The normalizing constant can be computed by using Fubini's theorem. Since $\mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime} \mid\right.$ $\left.p_{\star}\right) \lambda\left(\mathbf{x}^{\prime}\right) \lambda(\mathbf{x})$ is $\mathcal{X} \times \mathcal{X}$ integrable, it holds that

$$
\int_{\mathcal{X}} \int_{\mathcal{X}} \mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime} \mid p_{\star}, s\right) \lambda\left(\mathbf{x}^{\prime}\right) \lambda(\mathbf{x}) d \mathbf{x}^{\prime} d \mathbf{x}=\int_{\mathcal{X} \times \mathcal{X}} \mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime} \mid p_{\star}, s\right) \lambda\left(\mathbf{x}^{\prime}\right) \lambda(\mathbf{x}) d\left(\mathbf{x}^{\prime}, \mathbf{x}\right)=0.5
$$

by the symmetry and the fact that $\mathrm{P}\left(\mathbf{x} \succ \mathbf{x}^{\prime} \mid p_{\star}, s\right)+\mathrm{P}\left(\mathbf{x}^{\prime} \succ \mathbf{x} \mid p_{\star}, s\right)=1$. So, the normalizing constant is 2 .

Case $k=\infty$. It follows from Theorem 3.6.

## $k$-wise ranking likelihood

The $k$-wise ranking likelihood $\mathrm{P}\left(\mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{1}} \succ \ldots \succ \mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{k}} \mid \mathcal{C}_{k}\right)$ can be computed as a product of $k$-wise comparison likelihoods,

$$
\prod_{j=1}^{k-1} \mathrm{P}\left(\mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{j}} \succ \mathcal{C}^{(j)} \mid \mathcal{C}^{(j)}\right)
$$

where $\mathcal{C}^{(1)}=\mathcal{C}_{k}, \ldots, \mathcal{C}^{(k-1)}=\mathcal{C}_{k} \backslash\left\{\mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{1}}, \ldots, \mathbf{x}_{\pi\left(\mathcal{C}_{k}\right)_{k-2}}\right\}$.

---

#### Page 15

# B Proofs

This section provides the proofs for the propositions made in the main paper.
Proposition. Let $p^{*}$ be the expert's belief density. Denote $N=k-1$, so that $\mathcal{D}_{N}=\left\{\mathbf{x}_{1} \succ \mathbf{x}_{2} \succ\right.$ $\left.\ldots \succ \mathbf{x}_{N} \succ \mathbf{x}_{N+1}\right\}$. If $W \sim \operatorname{Gumbel}(0, \beta)$, then for any positive monotonic transformation $g$, and for $f \equiv g \circ p^{*}$ it holds,

$$
p\left(\mathcal{D}_{N} \mid f\right) \xrightarrow{\beta \rightarrow 0} 1
$$

Proof. Let $f(\mathbf{x})=g\left(p^{*}(\mathbf{x})\right)$. Then, by Yellott Jr [1977],

$$
p\left(\mathcal{D}_{N} \mid f, \beta\right)=\prod_{n=1}^{N+1} \frac{e^{\frac{1}{\beta} f\left(\mathbf{x}_{n}\right)}}{\sum_{i=n}^{N+1} e^{\frac{1}{\beta} f\left(\mathbf{x}_{i}\right)}}=\prod_{n=1}^{N+1} \frac{e^{\frac{1}{\beta} g\left(p^{*}\left(\mathbf{x}_{n}\right)\right)}}{\sum_{i=n}^{N+1} e^{\frac{1}{\beta} g\left(p^{*}\left(\mathbf{x}_{i}\right)\right)}}
$$

By the product law for limits,

$$
\begin{aligned}
\lim _{\beta \rightarrow 0+} p\left(\mathcal{D}_{N} \mid f, \beta\right) & =\prod_{n=1}^{N+1} \lim _{\beta \rightarrow 0+} \frac{e^{\frac{1}{\beta} g\left(p^{*}\left(\mathbf{x}_{n}\right)\right)}}{\sum_{i=n}^{N+1} e^{\frac{1}{\beta} g\left(p^{*}\left(\mathbf{x}_{i}\right)\right)}} \\
& =\prod_{n=1}^{N+1} \mathbb{1}\left(g\left(p^{*}\left(\mathbf{x}_{n}\right)\right)=\max _{n \leq i \leq N+1} g\left(p^{*}\left(\mathbf{x}_{i}\right)\right)\right) \\
& =\prod_{n=1}^{N+1} \mathbb{1}\left(p^{*}\left(\mathbf{x}_{n}\right)=\max _{n \leq i \leq N+1} p^{*}\left(\mathbf{x}_{i}\right)\right) \\
& =\prod_{n=1}^{N+1} \mathbb{1}\left(p^{*}\left(\mathbf{x}_{n}\right)=p^{*}\left(\mathbf{x}_{n}\right)\right) \\
& =1
\end{aligned}
$$

The second equation holds because the softmax converges pointwise to the argmax in the limit of the temperature approaches zero. The third equation holds because $g$ preserves the order. The fourth equation holds because $p^{*}\left(\mathbf{x}_{1}\right)>p^{*}\left(\mathbf{x}_{2}\right)>\ldots>p^{*}\left(\mathbf{x}_{N+1}\right)$.

Proposition. Let $\mathcal{C}_{k}$ be a choice set of $k \geq 2$ alternatives. Denote $C=\mathcal{C}_{k} \backslash\{\mathbf{x}\}$ and $f_{C}^{\star}=$ $\max _{\mathbf{x}_{j} \in C} f\left(\mathbf{x}_{j}\right)$. The winning probability of a point $\mathbf{x} \in \mathcal{C}_{k}$ equals to
$P\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right)=\sum_{l=0}^{k-1} \frac{\exp (-s(l+1) \max \left\{f_{C}^{\star}-f(\mathbf{x}), 0\right\})}{l+1} \sum_{\text {sym: } \mathbf{x}_{j} \in C}^{l}-\exp \left(-s\left(f(\mathbf{x})-f\left(\mathbf{x}_{j}\right)\right)\right)$,
where $\sum_{\text {sym: } \mathbf{x}_{j} \in \mathcal{C}_{k} \backslash\{\mathbf{x}\}}$ denotes the $l^{\text {th }}$ elementary symmetric sum of the set $C$.

---

#### Page 16

Proof. Fix $\mathbf{x} \in \mathcal{C}_{k}$, and for any $w \geq 0$ denote $\mathbf{1}_{w}=\mathbb{I}_{\left\{f(\mathbf{x})+w \geq f_{C}^{\star}\right\}}$.

$$
\begin{aligned}
\mathrm{P}\left(\mathbf{x} \succ \mathcal{C}_{k} \mid \mathcal{C}_{k}\right) & =\mathrm{P}\left(\bigcap_{x_{j} \in C}\left\{f(\mathbf{x})+W(\mathbf{x})>f\left(\mathbf{x}_{j}\right)+W\left(\mathbf{x}_{j}\right)\right\}\right) \\
& =\int \mathrm{P}\left(\bigcap_{x_{j} \in C}\left\{f(\mathbf{x})+W(\mathbf{x})>f\left(\mathbf{x}_{j}\right)+W\left(\mathbf{x}_{j}\right)\right\} \mid W(\mathbf{x})\right) \mathrm{P}(d W(\mathbf{x})) \\
& =\int_{0}^{\infty} \mathrm{P}\left(\bigcap_{x_{j} \in C}\left\{W\left(\mathbf{x}_{j}\right)<f(\mathbf{x})+w-f\left(\mathbf{x}_{j}\right)\right\} \mid w\right) s e^{-s w} d w \\
& =\int_{0}^{\infty} s e^{-s w} \prod_{x_{j} \in C} \mathrm{P}\left(\left\{W\left(\mathbf{x}_{j}\right)<f(\mathbf{x})+w-f\left(\mathbf{x}_{j}\right)\right\} \mid w\right) d w \\
& =\int_{0}^{\infty} s e^{-s w} \prod_{x_{j} \in C}\left(1-e^{-s\left(f(\mathbf{x})+w-f\left(\mathbf{x}_{j}\right)\right)}\right) \mathbb{I}_{\left\{f(\mathbf{x})+w \geq f\left(\mathbf{x}_{j}\right)\right\}} d w \\
& =\int_{0}^{\infty} s e^{-s w} \prod_{x_{j} \in C} \frac{1}{e^{s w}}\left(e^{s w}-e^{-s\left(f(\mathbf{x})-f\left(\mathbf{x}_{j}\right)\right)}\right) \mathbf{1}_{w} d w \\
& =\int_{0}^{\infty} s e^{-k s w} \mathbf{1}_{w} \prod_{x_{j} \in C}\left(e^{s w}-e^{-s\left(f(\mathbf{x})-f\left(\mathbf{x}_{j}\right)\right)}\right) d w
\end{aligned}
$$

Denote $c_{j}:=-\exp \left(-s\left(f(\mathbf{x})-f\left(\mathbf{x}_{j}\right)\right)\right)$. Let $b_{l}$ be the $l^{t h}$ elementary symmetric sum of the $c_{j}$ over $j$ s. The $l^{t h}$ elementary symmetric sum is the sum of all products of $l$ distinct $c_{j}$ over $j$ s. We can write,

$$
\begin{aligned}
& \int_{0}^{\infty} s e^{-k s w} \mathbf{1}_{w} \prod_{x_{j} \in C}\left(e^{s w}-e^{-s\left(f(\mathbf{x})-f\left(\mathbf{x}_{j}\right)\right)}\right) d w \\
& =\int_{0}^{\infty} s e^{-k s w} \mathbf{1}_{w} \sum_{l=0}^{k-1} b_{l} e^{(k-1-l) s w} d w \\
& =s \int_{0}^{\infty} \sum_{l=0}^{k-1} b_{l} e^{(k-1-l) s w-k s w} \mathbf{1}_{w} d w \\
& =s \sum_{l=0}^{k-1} \frac{b_{l}}{s(l+1)} \int_{0}^{\infty} s(l+1) e^{-s(l+1) w} \mathbf{1}_{w} d w \\
& =\sum_{l=0}^{k-1} \frac{b_{l}}{l+1} \int_{\max \left\{f_{C}^{\star}-f(\mathbf{x}), 0\right\}}^{\infty} s(l+1) e^{-s(l+1) w} d w \\
& =\sum_{l=0}^{k-1} \frac{b_{l}\left(1-G_{s(l+1)}\left(\max \left\{f_{C}^{\star}-f(\mathbf{x}), 0\right\}\right)\right)}{l+1} \\
& =\sum_{l=0}^{k-1} \frac{b_{l} \exp (-s(l+1) \max \left\{f_{C}^{\star}-f(\mathbf{x}), 0\right\})}{l+1}
\end{aligned}
$$

with convention that $b_{0}=1$ and $G_{\eta}$ denotes the cumulative distribution function of $\operatorname{Exp}(\eta)$.

---

#### Page 17

Algorithm 1 Full algorithm
require: preferential data $\mathcal{D}_{\text {full }}$
while not converged do
sample mini-batch $\mathcal{D} \sim \mathcal{D}_{\text {full }}$
$\Delta \phi \propto \nabla_{\phi} \operatorname{FS}-\operatorname{Posterior}(\phi \mid \mathcal{D})$
end while

Algorithm 2 FS-Posterior $(\phi \mid \mathcal{D})$
require: precision $s$
input: flow parameters $\phi$, mini-batch $\mathcal{D}$
$\mathbf{X}=$ design matrix of $\mathcal{D}$
$\mathbf{X}_{>}=$ winner points of $\mathbf{X}$
loglik $=\sum \log \mathcal{L}\left(\mathcal{D} \mid f_{\phi}(\mathbf{X}), s\right)$
logprior $=\sum f_{\phi}\left(\mathbf{X}_{>}\right)$
return: loglik + logprior

# C Experimental details

## C. 1 Target distributions

The logarithmic unnormalized densities of the target distributions used in the synthetic experiments are listed below.

Onemoon2D : $\quad-\frac{1}{2}\left(\frac{\|\mathbf{x}\|-2}{0.2}\right)^{2}-\frac{1}{2}\left(\frac{x_{0}+2}{0.3}\right)^{2}$
Gaussian6D : $\quad-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{T} \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}), \boldsymbol{\mu}=2\left(\begin{array}{cccc}(-1)^{1} \\ (-1)^{2} \\ \vdots \\ (-1)^{6}\end{array}\right), \Sigma=\left(\begin{array}{cccc}\frac{6}{10} & 0.4 & \cdots & 0.4 \\ 0.4 & \frac{6}{10} & \cdots & 0.4 \\ \vdots & \vdots & \ddots & \vdots \\ 0.4 & 0.4 & \cdots & \frac{6}{10}\end{array}\right)$
Twogaussians : $\quad \log \left(\frac{1}{2} \exp \left(\log \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma_{1}\right)\right)+\frac{1}{2} \exp \left(\log \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma_{2}\right)\right)\right)$,
$\sigma^{2}=1, \rho=0.9, d \in\{10,20\}, \boldsymbol{\mu}=3 \mathbf{1}_{d}, \Sigma_{1}=\left[\begin{array}{cccc}\sigma^{2} & \rho \sigma^{2} & \rho \sigma^{2} & \cdots & \rho \sigma^{2} \\ \rho \sigma^{2} & \sigma^{2} & \rho \sigma^{2} & \cdots & \rho \sigma^{2} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \rho \sigma^{2} & \rho \sigma^{2} & \rho \sigma^{2} & \cdots & \sigma^{2}\end{array}\right]$,
$\Sigma_{2}=\left[\begin{array}{cccccc}\sigma^{2} & -\rho \sigma^{2} & \rho \sigma^{2} & \cdots & (-1)^{d-1} \rho \sigma^{2} \\ -\rho \sigma^{2} & \sigma^{2} & -\rho \sigma^{2} & \cdots & (-1)^{d-2} \rho \sigma^{2} \\ \rho \sigma^{2} & -\rho \sigma^{2} & \sigma^{2} & \cdots & (-1)^{d-3} \rho \sigma^{2} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ (-1)^{d-1} \rho \sigma^{2} & (-1)^{d-2} \rho \sigma^{2} & (-1)^{d-3} \rho \sigma^{2} & \cdots & \sigma^{2}\end{array}\right]$
Funnel10D : $\quad-\left(\frac{\left(x_{0}-1\right)^{2}}{a^{2}}\right)-\sum_{i=1}^{10-1}\left(\log \left(2 \pi \exp \left(2 b x_{0}\right)\right)+\frac{\left(x_{i}-1\right)^{2}}{\exp \left(2 b x_{0}\right)}\right), a=3, b=0.25$

## C. 2 LLM expert elicitation experiment

The version of the Claude 3 model used in the LLM expert elicitation experiment was claude-3-haiku-20240307. In the experiment, we used the following prompt template to query the LLM. The configurations specified within the XML tags <configuration> were sampled from $\lambda$, a uniform distribution. The prompt template is as follows:

Data definition:
<data>
California Housing

---

#### Page 18

We collected information on the variables using all the block groups in California from the 1990 Census. In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. Naturally, the geographical area included varies inversely with the population density. We computed distances among the centroids of each block group as measured in latitude and longitude. This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). A household is a group of people residing within a home.

Number of Variables: 8 continuous
Variable Information:

- MedInc median income (expressed in hundreds of thousands of dollars) in block group
- HouseAge median house age in block group
- AveRooms average number of rooms per household
- AveBedrms average number of bedrooms per household
- Population block group population
- AveOccup average number of household members
- Latitude block group latitude
- Longitude block group longitude
  $</$ data $>$
  The variables are:
  <variables>
  {MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude}
  $</$ variables>
  always reported in this order.
  Given these combinations of variables below, please list them from most likely to least likely in your opinion. Consider what each variable represents and its realistic value in light of the properties of the dataset.
  <configurations>
  $\mathrm{A}=0.79,0.81,0.40,0.60,0.74,0.49,0.59,0.75,0.00,0.04$
  $\mathrm{B}=0.09,0.10,0.22,0.92,0.16,0.95,0.02,0.91,0.25,0.02$
  $\mathrm{C}=0.72,0.50,0.17,0.70,0.37,0.78,0.15,0.14,0.05,0.05$
  $\mathrm{D}=0.39,0.69,0.27,0.63,0.25,0.13,0.81,0.89,0.31,0.02$
  $\mathrm{E}=0.34,0.52,0.01,0.34,0.90,0.42,0.49,0.02,0.26,0.04$
  $</$ configurations>
  <task>

1. First, think your answer step by step, considering the model and data definition in detail.
2. Then discuss each combination separately in light of your thoughts about data. Do not assign an ordering yet.
3. Finally, summarize all your considerations.
4. At the end, write your final ordering as a comma-separated list of letters within an XML tag <order></order>.
   $</$ task>

---

#### Page 19

Table C.1: Accuracy of the density represented as a flow (flow) compared to a factorized normal distribution (normal), both learned from preferential data, in three metrics: log-likelihood, Wasserstein distance, and the mean marginal total variation (MMTV). Averages over 20 repetitions (but excluding a few crashed runs), with standard deviations.

|               | log-likelihood $(\uparrow)$ |                    |    wasserstein    |                   | MMTV $(\downarrow)$ |                   |
| :------------ | :-------------------------: | :----------------: | :---------------: | :---------------: | :-----------------: | :---------------: |
|               |           normal            |        flow        |      normal       |       flow        |       normal        |       flow        |
| Abalone7D     |     $-5.53( \pm 0.03)$      | $-2.16( \pm 0.12)$ | $0.53( \pm 0.00)$ | $0.34( \pm 0.01)$ |  $0.26( \pm 0.00)$  | $0.29( \pm 0.01)$ |
| mod-Abalone7D |     $-5.25( \pm 0.07)$      | $-3.52( \pm 0.09)$ | $1.05( \pm 0.00)$ | $0.65( \pm 0.01)$ |          -          |         -         |

# C. 3 Modified Abalone7D experiment

By modifying Abalone7D experiment, we can consider a synthetic technical validation constructed so that the data distribution is more realistic. We do this by mis-using a regression data set so that the response variable is interpreted as indication of preference and the queries are formed by presenting the expert a choice between different samples. If we denote by $g\left(\mathbf{x}_{i}\right)$ the regression function for the $i$ th covariate set, then $\mathbf{x}_{i}$ is chosen over $\mathbf{x}_{j}$ if $g\left(\mathbf{x}_{i}\right)>g\left(\mathbf{x}_{j}\right)$. We remark that the task itself is not particularly interesting as the response variable does not correspond to any real belief (instead, we learn a distribution over the covariates for which the response variable is high), but it is still useful for validating the algorithm as we now need to cope with choice sets that do not match any simple distribution $\lambda(\mathbf{x})$. Instead, the choice sets are now formed by uniform sampling over the sample indices, which means they are drawn from the marginal distribution of the covariates. Note that this is different from the target density, which is the density of covariates for samples with high response variables.

In the experiment, we do not assume any noise on the expert response. Hence, the expert follows a noiseless RUM with the representative utility $g\left(\mathbf{x}_{i}\right)$ equals to the response variable of $i$ th covariate $\mathbf{x}_{i}$. This means that the choice distribution resembles Dirac delta function at the points with the highest response variables. For this reason, we cannot compute the MMTV metric as it involves integrating over the absolute differences of the marginals, which leads to numerical issues. The numerical results are provided in Table C. 1 and the visual results in Figure C.1.

## C. 4 Other experimental details

Hyperparameters. In all the experiments, we use the value $s=1$ in the preferential likelihood regardless of how misspecified it is with respect to the ground-truth model. Neural Spline Models have 2 hidden layers and 128 hidden units. The number of flows is 6,8 , or 10 depending on the problem complexity. RealNVP models have 4 hidden layers and 2 hidden units. The number of flows is 36 when the number of rankings is more than 50 , and 8 otherwise. Other architecture-specific details correspond to the default values implemented in the normflows package, a PyTorch-based implementation of normalizing flows [Stimper et al., 2023].
Optimization details. Models are trained for a varying number of iterations from $10^{5}$ to $5 \times 10^{5}$ with the Adamax optimizer [Kingma and Ba, 2014] and varying batch size from 2 to 8 . The learning rate varies from $10^{-5}$ to $5 \times 10^{-3}$ depending on the problem dimensionality, with higher learning rates for higher-dimensional problems. A small weight decay of $10^{-6}$ was applied.
Computational environment. Models are trained and evaluated on a server with nodes of two Intel Xeon processors, code name Cascade Lake, with 20 cores each running at 2.1 GHz . Double precision numbers were used to improve the stability of the training process. We did not explicitly record the training times or memory consumption, but note that the considered data sets and flow architectures are all relatively small.
Experiment replications. Every experiment was replicated with 20 different seeds, ranging from 1 to 20 , but a few replications failed due to not well-known reasons, sometimes due to memory issues and sometimes due to numerical instabilities that led the replication to crash. The results are reported over the completed runs. In the main experiment table (Table 1), there was one failed replication in the Twogaussians20D experiment and two in the Onemoon2D experiment.
Dataset licence: Abalones: (CC BY 4.0) license, original source [Nash et al., 1995]

---

#### Page 20

> **Image description.** The image is a matrix of plots, specifically a cross-plot matrix, showing the relationships between different variables related to abalone characteristics. The variables are Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, and Shell weight.
>
> The matrix is arranged such that the diagonal elements are probability density plots of each individual variable. The off-diagonal elements are 2D density plots (heatmaps) showing the joint distribution of pairs of variables.
>
> - **Diagonal Plots:** These are probability density plots. Each plot shows a single peak. A vertical magenta line is present in each of these plots. The x-axis represents the variable name, and the y-axis represents the probability density.
> - **Off-Diagonal Plots:** These are 2D density plots, essentially heatmaps. The color intensity represents the density of data points. The plots show a positive correlation between most of the variables, as indicated by the elongated shape of the density clouds. The x and y axes of these plots correspond to the variables being compared. The color scheme used for the density plots appears to be a gradient from dark purple to yellow, with yellow indicating higher density.
>
> The axes are labeled with numerical values. The ranges vary depending on the variable, but they are generally between 0 and 1 for Length, Diameter, and Height; between 0 and 2 for Whole weight and Shucked weight; and between 0 and 1 for Viscera weight and Shell weight.
>
> The arrangement of the plots is as follows:
>
> - **Row 1:** Length vs. Length (density plot), Length vs. Diameter (heatmap), Length vs. Height (heatmap), Length vs. Whole weight (heatmap), Length vs. Shucked weight (heatmap), Length vs. Viscera weight (heatmap), Length vs. Shell weight (heatmap).
> - **Row 2:** Diameter vs. Length (heatmap), Diameter vs. Diameter (density plot), Diameter vs. Height (heatmap), Diameter vs. Whole weight (heatmap), Diameter vs. Shucked weight (heatmap), Diameter vs. Viscera weight (heatmap), Diameter vs. Shell weight (heatmap).
> - **Row 3:** Height vs. Length (heatmap), Height vs. Diameter (heatmap), Height vs. Height (density plot), Height vs. Whole weight (heatmap), Height vs. Shucked weight (heatmap), Height vs. Viscera weight (heatmap), Height vs. Shell weight (heatmap).
> - **Row 4:** Whole weight vs. Length (heatmap), Whole weight vs. Diameter (heatmap), Whole weight vs. Height (heatmap), Whole weight vs. Whole weight (density plot), Whole weight vs. Shucked weight (heatmap), Whole weight vs. Viscera weight (heatmap), Whole weight vs. Shell weight (heatmap).
> - **Row 5:** Shucked weight vs. Length (heatmap), Shucked weight vs. Diameter (heatmap), Shucked weight vs. Height (heatmap), Shucked weight vs. Whole weight (heatmap), Shucked weight vs. Shucked weight (density plot), Shucked weight vs. Viscera weight (heatmap), Shucked weight vs. Shell weight (heatmap).
> - **Row 6:** Viscera weight vs. Length (heatmap), Viscera weight vs. Diameter (heatmap), Viscera weight vs. Height (heatmap), Viscera weight vs. Whole weight (heatmap), Viscera weight vs. Shucked weight (heatmap), Viscera weight vs. Viscera weight (density plot), Viscera weight vs. Shell weight (heatmap).
> - **Row 7:** Shell weight vs. Length (heatmap), Shell weight vs. Diameter (heatmap), Shell weight vs. Height (heatmap), Shell weight vs. Whole weight (heatmap), Shell weight vs. Shucked weight (heatmap), Shell weight vs. Viscera weight (heatmap), Shell weight vs. Shell weight (density plot).

Figure C.1: Full result plot for the modified Abalone7D experiment, where the target (unnormalized) belief density corresponds to the abalone age.

# C. 5 Plots of learned belief densities

Figure 3 presented a subset of the cross-plots for the multivariate densities for the Abalone regression data and the LLM experiment. Here, we provide the complete visual illustrations over the full density for both, as well as corresponding visualisations for all of the other experiments in Figures C. 2 to C.8.

---

#### Page 21

> **Image description.** This image is a matrix of plots, displaying pairwise relationships between six variables, labeled x1 through x6. The matrix is arranged such that the diagonal elements show the marginal distribution of each variable, while the off-diagonal elements show the joint distribution between pairs of variables.
>
> - **Diagonal Plots:** Each plot on the diagonal (from top-left to bottom-right) shows a 1D distribution. These are line plots. Each plot contains two curves: a black, jagged curve that represents the learned flow, and a smoother pink curve that represents the target distribution. The x-axis of each plot is not explicitly labeled with numerical values, but the range appears to be approximately from -4 to 4. The y-axis is also not explicitly labeled. The plots are labeled x1, x2, x3, x4, x5, and x6 respectively.
>
> - **Off-Diagonal Plots:** The plots off the diagonal show 2D joint distributions. These are displayed as heatmaps. The color scheme ranges from dark purple (low density) to light blue/cyan (high density). The x and y axes of these plots represent the values of the two variables being compared. The axes range from approximately -4 to 4. Each plot represents the joint distribution of the variables corresponding to its row and column. For example, the plot in the second row and first column shows the joint distribution of x2 (y-axis) and x1 (x-axis). The plots show a clustering of data points in the lower-left quadrant.

Figure C.2: Gaussian6D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned flow is depicted by dark blue contour sample points and its marginal by a black curve.

> **Image description.** The image is a 2D density plot with contour lines. The background is a gradient fill that transitions from dark purple in the lower right to lighter blue/green in the upper left.
>
> - A cluster of concentric, light blue contour lines is located in the left center of the image.
> - The area within the innermost contour line is filled with a gradient, transitioning from yellow in the center to green/blue towards the outer contours.
> - The x and y axes are labeled with values ranging from -3 to 3.

Figure C.3: Estimated belief density for the Onemoon2D data. See Figure 1 for other visualisations on the same density.

---

#### Page 22

> **Image description.** This image shows a matrix of plots, arranged in a triangular format, displaying relationships between variables labeled x1 through x10.
>
> The matrix consists of two types of plots:
>
> - **2D Density Plots:** These plots appear in the lower-left portion of each cell. They are heatmaps, with a color gradient from dark purple to bright yellow/white, indicating the density of data points. The x and y axes of these plots range from -5.0 to 5.0, with labels at -5.0, 0.0, and 5.0.
> - **1D Marginal Distribution Plots:** These plots appear on the diagonal of the matrix. They show the distribution of individual variables. Each plot contains two curves: a black curve and a pink curve. The x-axis ranges from -5.0 to 5.0 with labels at -5.0, 0.0, and 5.0. The y-axis ranges from 0.0 to 5.0 with labels at 0.0, 2.5, and 5.0.
>
> The matrix is structured such that the cell at row _i_ and column _j_ (where _i_ > _j_) displays the 2D density plot of variables _xi_ and _xj_. The diagonal cells (where _i_ = _j_) display the 1D marginal distribution plots of variable _xi_.
>
> The variables are labeled x1 to x10, with the labels appearing above the diagonal plots and below the bottom row of plots.

Figure C.4: Twogaussians10D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned flow is depicted by dark blue contour sample points and its marginal by a black curve.

---

#### Page 23

> **Image description.** This image presents a matrix of plots, resembling a correlation plot, displaying relationships between different variables labeled x1 through x10.
>
> The matrix is arranged such that the diagonal elements are histograms, and the off-diagonal elements are two-dimensional density plots.
>
> - **Histograms (Diagonal):** Each histogram shows the distribution of a single variable (x1 to x10). The x-axis of each histogram represents the variable's value, and the y-axis represents the frequency or density. The histograms are black lines on a white background.
>
> - **Density Plots (Off-Diagonal):** The off-diagonal plots are two-dimensional density plots, where the color intensity represents the density of data points. The plots use a color gradient, transitioning from dark purple (low density) to yellow/green (high density). The x and y axes of these plots represent the values of the two variables being compared. The axes range from approximately -5 to 5.
>
> - **Variable Labels:** Each row and column is labeled with a variable name (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10). The labels are placed above the histograms and below the bottom row of density plots.
>
> - **Arrangement:** The plots are arranged in a lower triangular matrix, meaning that only the plots below and including the diagonal are present. The upper triangular part is left blank.
>
> The overall arrangement suggests an analysis of the relationships between these ten variables, likely representing data from a statistical model or dataset. The density plots provide insights into the joint distributions of pairs of variables, while the histograms show the marginal distributions of each variable.

Figure C.5: Estimated belief density for the Funnel10D data. The narrow funnel dimension $(x 1)$ is extremely difficult to capture accurately, but the flow still extends more in that dimension, seen as clear skew in all marginal histograms.

---

#### Page 24

> **Image description.** This image contains two triangular grid plots side-by-side, visualizing distributions and relationships between variables.
>
> The plot on the left is titled "Learned LLM prior", and the plot on the right is titled "California housing dataset distribution". Each plot consists of a grid of subplots arranged in a triangular fashion. The diagonal subplots display histograms of individual variables. The off-diagonal subplots display 2D density plots (contour plots) showing the relationships between pairs of variables.
>
> The variables plotted are:
>
> - MedInc
> - HouseAge
> - AveRooms
> - AveBedrms
> - Population
> - AveOccup
> - Latitude
> - Longitude
>
> The x and y axes of the subplots are labeled with these variable names. The histograms along the diagonal show the distribution of each individual variable. The contour plots show the joint distribution of pairs of variables, with darker areas indicating higher density.

Figure C.6: Full result plot for the LLM expert elicitation experiment, complementing the partial plot presented in Figure 3.

> **Image description.** This image is a matrix of plots, displaying pairwise relationships between different variables related to abalone characteristics. The matrix is arranged such that the diagonal elements are probability density plots (line graphs), while the off-diagonal elements are density plots (heatmaps).
>
> - **Overall Structure:** The image is organized as a matrix, where each row and column corresponds to a different variable. The variables are "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", and "Shell weight".
>
> - **Diagonal Plots (Probability Density Plots):** Each plot on the diagonal shows the probability density of a single variable. There are two curves in each of these plots: a black curve and a pink curve. The x-axis represents the variable's value, and the y-axis represents the probability density.
>
> - **Off-Diagonal Plots (Density Plots/Heatmaps):** These plots show the joint density of two variables. The x and y axes correspond to the two variables being compared. The color intensity represents the density of data points in that region, with brighter colors indicating higher density. The plots appear to use a color gradient, transitioning from dark purple to brighter shades, potentially representing increasing density.
>
> - **Axes and Labels:**
>
>   - The x-axis labels are present for each plot in the bottom row, showing the variable name (Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight). The top row has no labels.
>   - The y-axis labels are present for each plot in the leftmost column, showing the variable name (Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight). The rightmost column has no labels.
>   - The x and y axes are scaled differently depending on the variable.
>
> - **Specific Observations:**
>   - The relationship between Length and Diameter appears to be strongly linear, as indicated by the concentrated density along a diagonal line in the corresponding heatmap.
>   - The distributions of "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", and "Shell weight" are all unimodal, as seen in the diagonal plots.
>   - The black and pink curves in the diagonal plots appear to represent different distributions or models for each variable.

Figure C.7: Full result plot for the Abalone7D experiment, complementing the partial plot presented in Figure 3. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned flow is depicted by dark blue contour sample points and its marginal by a black curve.

---

#### Page 25

> **Image description.** The image is a correlation plot, displaying pairwise relationships between several variables related to abalone characteristics. It's structured as a matrix of subplots.
>
> - **Overall Structure:** The plot is organized as an n x n grid, where n is the number of variables. The variables are: Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, and Shell weight. The diagonal subplots display the distribution of each individual variable, while the off-diagonal subplots show the joint distribution of two variables.
>
> - **Diagonal Subplots:** These subplots show the distribution of a single variable. Each subplot contains a black line representing the estimated probability density function (PDF) of the variable. A pink line, closely following the black line, is also present. The x-axis represents the variable's value, and the y-axis represents the density.
>
> - **Off-Diagonal Subplots:** These subplots display the joint distribution of two variables. They are scatter plots, with the x-axis representing one variable and the y-axis representing the other. The data points are represented by a density map, where the color indicates the density of points in that region. The color scheme ranges from dark purple (low density) to yellow/green (high density).
>
> - **Axes Labels:** The x-axis labels are placed along the bottom row of subplots and are "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", and "Shell weight". The y-axis labels are placed along the leftmost column of subplots and are "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", and "Shell weight". Each axis has numerical values ranging from 0.0 to 1.0, with the exception of "Whole weight" and "Shucked weight", which range up to 2.0.
>
> - **Visual Patterns:** The scatter plots reveal the relationships between the variables. For example, there appears to be a strong positive correlation between "Length" and "Diameter", as well as between "Whole weight" and "Shucked weight". The distributions on the diagonal show the range and shape of each variable's values. The height variable appears to have a distribution skewed towards lower values.

Figure C.8: Full result plot for the Abalone7D experiment for the baseline method.

---

#### Page 26

> **Image description.** This image contains two triangular grid plots, each displaying the pairwise relationships between several variables. The left plot is titled "Learned LLM prior (MLE)" and the right plot is titled "Learned LLM prior (FS-MAP)". Each plot consists of a matrix of subplots, where the diagonal subplots show the distribution of a single variable, and the off-diagonal subplots show the joint distribution of two variables.
>
> Here's a breakdown of the visual elements:
>
> - **Arrangement:** Each plot is a matrix where the variable names are listed along the bottom row and the left-most column. The matrix is triangular, meaning that only the lower triangle of the matrix is populated with plots. The upper triangle is left blank.
> - **Variable Names:** The variables are MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, and Longitude.
> - **Diagonal Subplots (Histograms):** These subplots show the distribution of each individual variable. They appear as histograms or kernel density estimates. The y-axis is not explicitly labeled, but the x-axis is labeled with the variable's name.
> - **Off-Diagonal Subplots (Scatter Plots/Contour Plots):** These subplots show the relationship between two variables. They appear as density plots or contour plots, where darker regions indicate higher density of data points.
> - **Axes Labels:** The axes of the off-diagonal subplots are not explicitly labeled with numerical values, but the variable names are used as axis labels.
> - **Visual Differences:** The key difference between the two plots is the shape and spread of the distributions. The "Learned LLM prior (MLE)" plot shows more dispersed and potentially diverging distributions, especially for variables like Population and AveOccup. The "Learned LLM prior (FS-MAP)" plot shows more concentrated and less extreme distributions.
> - **Text:** The titles "Learned LLM prior (MLE)" and "Learned LLM prior (FS-MAP)" are located above the respective plots. The variable names are written along the bottom and left sides of each plot.

Figure D.1: The effect of the functional prior on the learned belief density in Experiment 5.2. The left plot corresponds to learning the LLM's belief density using maximum likelihood in the training, and the right plot to using function-space maximum a posteriori with the proposed functional prior. We hypothesize that the extreme marginals (e.g. median income) obtained from maximum likelihood estimation are due to problems with collapsing or diverging probability mass.

Table D.1: The means of the variables based on (first row) the distribution of California housing dataset and the sample from the preferential flow fitted to the LLM's feedback trained (second row) with the likelihood only and (third row) with the both likelihood and prior.

|                | MedInc HouseAge | AveRooms | AveBedrms | Population | AveOccup |  Lat   |      Long      |
| :------------- | :-------------: | :------: | :-------: | :--------: | :------: | :----: | :------------: |
| True data      |      3.87       |  28.64   |   5.43    |    1.1     | 1425.48  |  3.07  | $35.63-119.57$ |
| Flow w/ prior  |      9.83       |  43.01   |  125.18   |    8.07    | 22983.76 | 1290.0 | $28.81-117.94$ |
| Flow w/o prior |      5.91       |  27.19   |   6.28    |    1.58    | 2868.52  |  3.37  | $36.43-119.75$ |

## D Ablation studies

This section reports additional experimentation to complement the results presented in the main paper. Unless otherwise stated, the rest of the details in the experiments are as discussed in Sections 5 and C.4. The only exception is the number of flows, which are scaled by the number of rankings $n$ to increase flexibility in line with the available data. However, when $n$ is as in the main paper, the number of flows remains unchanged.

## D. 1 Effect of the functional prior

Figure D. 1 shows the effect of the functional prior for the LLM experiment, showcasing how the maximum likelihood estimate learning the flow without the functional prior exhibits the diverging mass property. Table D. 1 summarizes the densities in a quantitative manner by reporting the means for all variables. The table shows how the solution without the prior can be massively off already in terms of the mean estimate, for instance having the mean number of rooms at 125 .

## D. 2 Effect of the noise level $1 / s$

Figure D. 2 investigates the interplay of the true RUM noise and the assumed noise in the preferential likelihood on the Onemoon2D data.

---

#### Page 27

> **Image description.** The image is a 3x3 grid of heatmaps, visually representing data distributions. Each heatmap is a square with axes that are not explicitly labeled but are implied by the context. The color scheme ranges from dark purple/blue (low values) to yellow (high values).
>
> The columns of the grid are labeled at the top with the text "s_true = 0.01", "s_true = 1.0", and "s_true = 5.0". The rows are labeled on the left with the text "s_lik = 0.01", "s_lik = 1.0", and "s_lik = 5.0".
>
> Each heatmap displays a curved, elongated shape concentrated on the left side. The shape is most pronounced (brightest yellow) in the bottom left heatmap, where s_true = 0.01 and s_lik = 5.0. The other heatmaps show a similar curved shape, but with varying degrees of intensity and spread. The heatmaps in the rightmost column (s_true = 5.0) appear to have the most concentrated and least spread distributions. The heatmaps also contain some contour lines that trace the shape of the distribution.

Figure D.2: Preferential flow fitted via FS-MAP with varying precision levels in the data generation process (in RUM) $s_{\text {true }}$, and precision levels in the preferential likelihood $s_{l i k}$. The first column shows that a lower precision level in RUM leads to a more spread fitted flow, as expected. The middle plot is the only scenario where both the likelihood and the functional prior are correctly specified, resulting in the best result. Since the prior is misspecified in the bottom-right plot, the best results are not achieved, contrary to expectations. However, this misspecification does not lead to catastrophic performance deterioration but rather to a more spread-out fitted flow.

# D. 3 Effect of the candidate sampling distribution $\lambda$

We validate the sensitivity of the results in terms the choice of the distribution $\lambda$ that the candidates are sampled from. Figure D. 3 studies the effect of $\lambda$, the unknown distribution from which the candidates to be compared are sampled from, complementing the experiment reported in Section 5.1 and confirming the method is robust for the choice. In the original experiment the candidates were sampled from a mixture distribution of uniform and Gaussian distribution centered on the mean of the target, with the mixture probability $w=1 / 3$ for the Gaussian. Figure D. 3 reports the accuracy as a function of the choice of $w$ for one of the data sets (Onemoon2D), so that $\lambda$ goes from uniform to a Gaussian, and includes also an additional reference point where $\lambda$ equals the target. For all $w>0.5$ we reach effectively the same accuracy as when sampling from the target itself, and even for the hardest case of uniform $\lambda$ (with $w=0$ the distance is significantly smaller than the reference scale comparing the base distribution with the target.

## D. 4 Effect of the number of rankings $n$

We validate the sensitivity of the results in terms of the number of comparisons/rankings $n$. Table D.2, as well as Figures D. 4 and D.5, report the results of an experiment that studies the effect of $n$.

Increasing $n$ generally improves the accuracy and already fairly small $n$ is sufficient for learning a good estimate (Table D.2). For very large $n$, the accuracy can slightly deteriorate. We believe that this is due to prior misspecification that encourages overestimation of the variation due to the fact that $k$ is finite but in the prior it is assumed to be infinite. In the Onemoon2D experiment, Figure D. 4 confirms that for $n=1000$ the shape of the estimate is extremely close and the slightly worse Wasserstein distance is due to overestimating the width. The same holds for other experiments such as Twogaussians10D illustrated in Figure D.5.

---

#### Page 28

Figure D.3: The Onemoon2D experiment replicated for varying sampling distributions $\lambda$ from where the candidates are sampled. In original Onemoon2D, $\lambda$ is a mixture of uniform and Gaussian distribution centered on the mean of the target, with the mixture probability $w=1 / 3$ for the Gaussian. Here, $\lambda \in\{$ Uniform, Gaussian-Uniform Mixture, Target $\}$ with letting the mixture probability $w$ to vary in $\{0.1,1 / 4,1 / 3,1 / 2,2 / 3,3 / 4,1.0\}$. The rest of the details can be found in Section 5, specifically $n=200$ and $k=5$. The distance between the base density and the target density (1.85) provides a scale reference. The method is robust for the sampling distribution and for broad range of $w$ we reach essentially the same accuracy as when sampling from the target itself.

> **Image description.** The image is a line graph depicting the "Wasserstein distance" on the y-axis against "Mixture probability in Gaussian-uniform mixture" denoted as lambda on the x-axis.
>
> - **Axes:**
>
>   - The y-axis is labeled "Wasserstein distance" and ranges from 0.00 to 2.00 in increments of 0.25.
>   - The x-axis is labeled "Mixture probability in Gaussian-uniform mixture" and denoted as lambda. The x-axis values are: Uniform, 0.1, 0.250.33, 0.5, 0.66, 0.75, and 1.0 Target.
>
> - **Data:**
>
>   - A blue line connects data points representing the "Distance between flow and target density". Each data point is marked with a blue circle and has error bars. The line starts at approximately (Uniform, 0.7) and decreases as the mixture probability increases, leveling off around 0.1 for values greater than 0.5.
>   - A dashed blue curve is drawn above the first data point (Uniform), pointing to a label "Distance between base and target density" which is approximately at the y-axis value of 1.85.
>   - An arrow points to the data point at x-axis value "0.250.33" and has the label "Experiment in Section 5.1". The data point is circled with a dashed blue line.
>
> - **Text:**
>   - "Distance between flow and target density" is written next to the line graph data.
>   - "Distance between base and target density" is written next to the dashed blue curve.
>   - "Experiment in Section 5.1" is written next to the data point at x-axis value "0.250.33".
>   - The axis labels "Wasserstein distance" and "Mixture probability in Gaussian-uniform mixture" are clearly visible.

Table D.2: Wasserstein distances for varying $n$ (fixed $k=5$ ) across different experiments

| $n$             |                25 |                50 |               100 |              1000 |
| :-------------- | ----------------: | ----------------: | ----------------: | ----------------: |
| Onemoon2D       | $0.67( \pm 1.34)$ | $0.18( \pm 0.04)$ | $0.17( \pm 0.03)$ | $0.23( \pm 0.02)$ |
| Gaussian6D      | $1.70( \pm 0.22)$ | $1.50( \pm 0.19)$ | $1.46( \pm 0.11)$ | $1.26( \pm 0.04)$ |
| $n$             |                50 |               500 |              2000 |             10000 |
| Funnel10D       | $4.33( \pm 0.10)$ | $3.96( \pm 0.05)$ | $3.89( \pm 0.04)$ | $3.92( \pm 0.04)$ |
| Twogaussians10D | $2.69( \pm 0.31)$ | $2.57( \pm 0.08)$ | $2.61( \pm 0.05)$ | $2.66( \pm 0.04)$ |

> **Image description.** The image is a heatmap overlaid with contour lines, representing estimated belief densities in the Onemoon2D experiment.
>
> - The background is a dark purple color, with lighter shades of purple, blue, green, and yellow indicating higher density areas. The heatmap shows a crescent-shaped region of high density on the left side of the image, with a tail extending towards the lower right. The area of highest density is a yellow-green color.
> - Overlaid on the heatmap are several light blue contour lines, outlining the shape of the true density. These contours are concentrated in the same crescent-shaped region as the heatmap's high-density area. The contours are nested, indicating decreasing density as one moves outward from the center.
> - The x and y axes are visible, with tick marks but no labels. The x-axis ranges approximately from -15 to 10, and the y-axis ranges approximately from -10 to 10.

(a) $n=25, k=5$

> **Image description.** The image contains three panels, each depicting a heatmap overlaid with contour lines. Each panel represents an estimated belief density in the Onemoon2D experiment.
>
> - **Panel (a):** Labeled "(a) $n=25, k=5$". The heatmap shows a crescent-shaped region of higher density, transitioning from yellow/green in the center to blue/purple at the edges. The contour lines, in a light blue color, follow the shape of the crescent, indicating levels of equal density. The background is a dark purple.
>
> - **Panel (b):** Labeled "(b) $n=100, k=5$". Similar to panel (a), it displays a heatmap and contour lines. The crescent shape is more defined and concentrated compared to panel (a), with a more intense yellow/green core. The contour lines are also more closely spaced, suggesting a steeper density gradient.
>
> - **Panel (c):** Labeled "(c) $n=1000, k=5$". This panel also shows a heatmap and contour lines. The crescent shape is even more pronounced than in panel (b), with a brighter, more focused yellow/green core. The contour lines are tightly packed, indicating a high level of certainty. The overall spread of the density appears wider than in panel (b).

(b) $n=100, k=5$

> **Image description.** The image contains a heatmap overlaid with contour lines, likely representing a probability density function. The background of the plot is a dark purple color.
>
> - **Heatmap:** A crescent-shaped area in the left-center of the image displays a range of colors, transitioning from dark blue/purple on the outer edges to yellow/white at the center. This suggests a higher density or concentration of data points in the yellow/white region. The shape resembles a crescent moon.
> - **Contour Lines:** Light blue contour lines are superimposed on the heatmap, following the shape of the crescent. These lines indicate areas of equal density, with the innermost contours representing the highest density.
> - **Axes:** The image has x and y axes, though the labels are small and difficult to read. The axes appear to be scaled from approximately -6 to 6.

(c) $n=1000, k=5$

Figure D.4: The estimated belief densities in the Onemoon2D experiment of Table D. 2 (contour: true density; heatmap: estimated flow). While the coverage of the estimated density with $n=1000$ is better than with $n=100$, there is more spread with $n=1000$ than with $n=100$, which explains the slight performance deterioration in Table D.2.

---

#### Page 29

> **Image description.** The image shows a series of cross-plots arranged in a 3x3 grid, repeated three times with different parameter values. Each set of nine plots forms a triangular matrix. The plots are arranged so that the top row shows plots related to 'x1', the second row to 'x2', and the third row to 'x3'. The columns also correspond to 'x1', 'x2', and 'x3', respectively.
>
> Each plot in the top row is a 1D density plot, showing a black line representing an estimated density and a pink line representing the true density. The x-axis is labeled 'x1', 'x2', or 'x3' depending on the column.
>
> The plots below the diagonal are 2D heatmaps, with 'x2' on the y-axis and 'x1' on the x-axis in the second row, and 'x3' on the y-axis and 'x1' or 'x2' on the x-axis in the third row. The color gradient of the heatmap ranges from dark purple to bright cyan/green, presumably indicating density. The axes are labeled from -5.0 to 5.0 in increments of 2.5.
>
> The plots to the right of the diagonal are 1D density plots, similar to those in the top row.
>
> The entire set of plots is repeated three times, with the following labels underneath each set:
> (a) n = 50, k = 5
> (b) n = 500, k = 5
> (c) n = 10000, k = 5

Figure D.5: Cross-plots of selected variables of the estimated flow in the Twogaussians10D experiment of Table D. 2 (contour: true density; heatmap: estimated flow). While the coverage of the estimated density with $n=10000$ is better than with $n=500$, there is more spread with $n=10000$ than with $n=500$, which explains the slight performance deterioration in Table D.2.

# D. 5 Effect of the cardinality of the choice set $k$

Finally, to complement the ablation studies for $k$ on synthetic settings in Section 5.3, we rerun the LLM expert elicitation experiment with $k=2$. Figure D. 6 shows that the LLM expert also works with $k=2$. We replicated the original experiment conducted with $k=5$ and report the estimates side-by-side, visually confirming we learn the essential characteristics of the distribution in both cases. The results are not identical and the case of $k=5$ is likely more accurate (see e.g. the marginal distribution of the last feature), but there are no major qualitative differences between the two estimates.

> **Image description.** The image contains two sets of plots, arranged in a matrix format. Each set of plots is a visualization of a learned LLM prior, with the set on the left corresponding to k=2 and the set on the right corresponding to k=5, both with n=220.
>
> Each set of plots consists of a 4x4 grid, where each cell represents the relationship between two variables. The variables are: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, and Longitude.
>
> - The diagonal cells display histograms of individual variables. For example, the top-left cell shows the distribution of 'Population', the cell in the second row and second column shows the distribution of 'AveOccup', and so on. The histograms are represented by a series of vertical bars.
> - The off-diagonal cells show contour plots representing the joint distribution of pairs of variables. For instance, the cell in the third row and first column displays the joint distribution of 'Latitude' and 'MedInc'. The contour plots are represented by filled regions with varying shades of gray, indicating different density levels.
>
> The x-axis labels are: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, and Longitude.
> The y-axis labels are: Population, AveOccup, Latitude, and Longitude.
> The values on the axes are numerical and vary depending on the variable, such as Population (0-8000), AveOccup (250-1000), Latitude (34-40), and Longitude (-122.5 to -115.0).
>
> The text "Learned LLM prior (k=2,n=220)" is above the left set of plots, and "Learned LLM prior (k=5,n=220)" is above the right set of plots.

Figure D.6: The LLM expert elicitation experiment replicated for the setting of pairwise comparisons (left) and compared to the original setting of 5-wise rankings (right). The estimated flow remains qualitatively the same for the variables shown here (other variables omitted due to lack of space), and this holds true for the rest of the variables as well.