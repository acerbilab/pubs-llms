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
