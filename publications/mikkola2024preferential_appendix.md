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
