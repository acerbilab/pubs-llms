```
@article{souza2022parallel,
  title={Parallel MCMC Without Embarrassing Failures},
  author={Daniel Augusto R. M. A. {de Souza} and Diego Mesquita and Samuel Kaski and Luigi Acerbi},
  year={2022},
  journal={International Conference on Artificial Intelligence and Statistics (AISTATS 2022)}
}
```

---

#### Page 1

# Parallel MCMC Without Embarrassing Failures

Daniel Augusto de Souza ${ }^{1}$, Diego Mesquita ${ }^{2,3}$, Samuel Kaski ${ }^{2,4}$, Luigi Acerbi ${ }^{5}$<br>${ }^{1}$ University College London ${ }^{2}$ Aalto University ${ }^{3}$ Getulio Vargas Foundation<br>${ }^{4}$ University of Manchester ${ }^{5}$ University of Helsinki<br>daniel.souza.21@ucl.ac.uk, diego.mesquita@fgv.br, samuel.kaski@aalto.fi, luigi.acerbi@helsinki.fi

#### Abstract

Embarrassingly parallel Markov Chain Monte Carlo (MCMC) exploits parallel computing to scale Bayesian inference to large datasets by using a two-step approach. First, MCMC is run in parallel on (sub)posteriors defined on data partitions. Then, a server combines local results. While efficient, this framework is very sensitive to the quality of subposterior sampling. Common sampling problems such as missing modes or misrepresentation of low-density regions are amplified - instead of being corrected - in the combination phase, leading to catastrophic failures. In this work, we propose a novel combination strategy to mitigate this issue. Our strategy, Parallel Active Inference (PAI), leverages Gaussian Process (GP) surrogate modeling and active learning. After fitting GPs to subposteriors, PAI (i) shares information between GP surrogates to cover missing modes; and (ii) uses active sampling to individually refine subposterior approximations. We validate PAI in challenging benchmarks, including heavy-tailed and multi-modal posteriors and a real-world application to computational neuroscience. Empirical results show that PAI succeeds where previous methods catastrophically fail, with a small communication overhead.

## 1 INTRODUCTION

Markov Chain Monte Carlo (MCMC) methods have become a gold standard in Bayesian statistics (Gelman et al., 2013; Carpenter et al., 2017). However, scaling MCMC methods to large datasets is challenging due to their sequential nature and that they typically require many likelihood evaluations, implying repeated sweeps

[^0]through the data. Various approaches that leverage distributed computing have been proposed to mitigate these limitations (Angelino et al., 2016; Robert et al., 2018). In general, we can split these approaches between those that incur constant communication costs and those requiring frequent interaction between server and computing nodes (Vehtari et al., 2020).
Embarrassingly parallel MCMC (Neiswanger et al., 2014) is a popular class of methods which employs a divide-and-conquer strategy to sample from a target posterior, requiring only a single communication step. For dataset $\mathcal{D}$ and model parameters $\theta \in \mathbb{R}^{D}$, suppose we are interested in the Bayesian posterior $p(\theta \mid \mathcal{D}) \propto p(\theta) p(\mathcal{D} \mid \theta)$, where $p(\theta)$ is the prior and $p(\mathcal{D} \mid \theta)$ the likelihood. Embarrassingly parallel methods begin by splitting the data $\mathcal{D}$ into $K$ smaller partitions $\mathcal{D}_{1}, \ldots, \mathcal{D}_{K}$ so that we can rewrite the posterior as

$$
p(\theta \mid \mathcal{D}) \propto \prod_{k=1}^{K} p(\theta)^{1 / K} p\left(\mathcal{D}_{k} \mid \theta\right) \equiv \prod_{k=1}^{K} p_{k}(\theta)
$$

Next, an MCMC sampler is used to draw samples $\mathcal{S}_{k}$ from each subposterior $p_{k}(\theta)$, for $k=1 \ldots K$, in parallel. Then, the computing nodes send the local results to a central server, for a final aggregation step. These local results are either the samples themselves or approximations $q_{1}, \ldots, q_{K}$ built using them.
Works in embarrassingly parallel MCMC mostly focus on combination strategies. Scott et al. (2016) employ a weighted average of subposterior samples. Neiswanger et al. (2014) propose using multivariate-normal surrogates as well as non-parametric and semi-parametric forms. Wang et al. (2015) combine subposterior samples into a hyper-histogram with random partition trees. Nemeth and Sherlock (2018) leverage density values computed during MCMC to fit Gaussian process (GP) surrogates to log-subposteriors. Mesquita et al. (2019) use subposterior samples to fit normalizing flows and apply importance sampling to draw from their product.
Despite these advances, parallel MCMC suffers from an unaddressed limitation: its dependence on high-quality subposterior sampling. This requirement is especially

[^0]: Proceedings of the $25^{\text {th }}$ International Conference on Artificial Intelligence and Statistics (AISTATS) 2022, Valencia, Spain. PMLR: Volume 151. Copyright 2022 by the author(s).

---

#### Page 2

difficult to meet when subposteriors are multi-modal or heavy-tailed, in which cases MCMC chains often visit only a subset of modes and may underrepresent lowdensity regions. Furthermore, the surrogates $\left(q_{k}\right)_{k=1}^{K}$ built only on local MCMC samples might match poorly the true subposteriors if not carefully tuned.
Outline and contributions. We first discuss the failure modes of parallel MCMC (Section 2). Drawing insight from this discussion, Section 3 proposes a novel GP-based solution, Parallel Active Inference (PAI). After fitting the subposterior surrogates, PAI shares a subset of samples between computing nodes to prevent mode collapse. PAI also uses active learning to refine low-density regions and avoid catastrophic model mismatch. Section 4 validates our method on challenging benchmarks and a real-world neuroscience example. Finally, Section 5 reviews related work and Section 6 discusses strengths and limitations of PAI.

## 2 EMBARRASSINGLY PARALLEL MCMC: HOW CAN IT FAIL?

We recall the basic structure of a generic embarrassingly parallel MCMC algorithm in Algorithm 1. This schema has major failure modes that we list below, before discussing potential solutions. We also illustrate these pathologies in Fig 1. In this paper, for a function $f$ with scalar output and a set of points $\mathcal{S}=\left\{s_{1}, \ldots, s_{N}\right\}$, we denote with $f(\mathcal{S}) \equiv\left\{f\left(s_{1}\right), \ldots, f\left(s_{N}\right)\right\}$.

Algorithm 1 Generic embarrassingly parallel MCMC
Input: Data partitions $\mathcal{D}_{1}, \ldots, \mathcal{D}_{K}$; prior $p(\theta)$; likelihood function $p(\mathcal{D} \mid \theta)$.

1: parfor $1 \ldots K$ do
$\triangleright$ Parallel steps
2: $\quad \mathcal{S}_{k} \leftarrow$ MCMC samples from $p_{k}(\theta) \propto p(\theta)^{1 / K} p\left(\mathcal{D}_{k} \mid \theta\right)$
3: build subposterior model $q_{k}(\theta)$ from $\mathcal{S}_{k}$
4: end parfor
5: Combine: $q(\theta) \propto \prod_{k=1}^{K} q_{k}(\theta) \quad \triangleright$ Centralized step

### 2.1 Failure modes

I: Mode collapse. It is sufficient that one subposterior $q_{k}$ misses a mode for the combined posterior $q$ to lack that mode (see Fig 1A). While dealing with multiple modes is an open problem for MCMC, here the issue is exacerbated by the fact that a single failure propagates to the final solution. A back-of-the-envelope calculation shows that even if the chance of missing a mode is small, $\varepsilon>0$, the probability of mode collapse in the combined posterior is $\approx(1-\varepsilon)^{K}$ making it a likely occurrence for sufficiently large $K$.

Insight 1: For multimodal posteriors, mode collapse is almost inevitable unless the computing nodes can exchange information about the location of important posterior regions.

II: Catastrophic model mismatch. Since the $q_{k}$ are approximations of the true subposteriors $p_{k}$, small deviations between them are expected - this is not what we refer to here. Instead, an example of catastrophic model mismatch is when a simple parametric model such as a multivariate normal is used to model a multimodal posterior with separate modes (see Section 4). Even nonparametric methods can be victims of this failure. For example, GP surrogates are often used to model nonparametric deviations of the log posterior from a parametric 'mean function'. While these models can well represent multimodal posteriors, care is needed to avoid grossly mismatched solutions in which a $q_{k}$ 'hallucinates' posterior mass due to an improper placement of the GP mean function (Fig 1B).

Insight 2: We cannot take subposterior models at face value. Reliable algorithms should check and refine the $q_{k}$ 's to avoid catastrophic failures.

III: Underrepresented tails. This effect is more subtle than the failure modes listed above, but it contributes to accumulating errors in the estimate of the combined posterior. The main issue here is that, by construction, MCMC samples and subposterior models based on these samples focus on providing information about the high-posterior-mass region of the subposterior. However, different subposteriors may overlap only in their tail regions (Fig 1C), implying that the tails and the nearby 'deep' regions of each subposterior might actually be the most important in determining the exact shape of the combined posterior.

Insight 3: Subposterior models built only from MCMC samples (and their log density) can miss important information about the tails and nearby regions of the subposterior which would also contribute to the combined posterior.

### 2.2 Past solutions

Since there is no guarantee that $q$ approximates well the posterior $p$, Nemeth and Sherlock (2018) refine $q$ with an additional parallel step, called Distributed Importance Sampler (DIS). DIS uses $q$ as a proposal distribution and draws samples $\mathcal{S} \sim q$, that are then sent back for evaluation of the log density $\log p_{k}(\mathcal{S})$ at each parallel node. The true $\log$ density $\log p(\mathcal{S})=\sum_{k} \log p_{k}(\mathcal{S})$ is then used as a target for importance sampling/resampling (Robert and Casella, 2013). Technically, this step makes the algorithm not 'embarrassingly parallel' anymore, but the prospect of fixing catastrophic failures outweighs the additional communication cost. However, DIS does not fully solve the issues raised in Section 2.1. Notably, importance sampling will not

---

#### Page 3

> **Image description.** This image is a figure illustrating failure modes of embarrassingly parallel Markov Chain Monte Carlo (MCMC). It consists of three columns (A, B, and C), each representing a distinct failure type. Each column has three rows of plots. The top two rows in each column show two subposteriors, denoted as p1(theta) and p2(theta), while the bottom row shows the full posterior p(theta | D).
>
> Here's a breakdown of each column:
>
> - **Column A: Failure I: Mode collapse**
>
>   - Top Row: Shows subposterior p1(theta) as a dashed black line with blue circles representing MCMC samples. The distribution appears to have two modes. The y-axis ranges from 0.0 to 1.5.
>   - Middle Row: Shows subposterior p2(theta). It shows a single mode, with the other mode missing. This is indicated by a red shaded area labeled "Density mismatch" and a red arrow pointing to the missing mode labeled "Unexplored mode". The y-axis ranges from 0.0 to 1.5.
>   - Bottom Row: Shows the combined result. The "Ground truth" is represented by a dashed black line, showing two modes. The approximate posterior, combined using the "PART" method, is shown as an orange line. The orange line only captures one mode, indicating a failure. The y-axis ranges from 0.0 to 1.5, and the x-axis ranges from -3 to 3.
>
> - **Column B: Failure II: Model mismatch**
>
>   - Top Row: Shows subposterior p1(theta) as a dashed black line with blue circles representing MCMC samples. The distribution appears to be unimodal. The y-axis ranges from 0.0 to 0.4.
>   - Middle Row: Shows subposterior p2(theta). It shows a distribution with multiple modes, with an orange line representing the "hallucinated" region. A red arrow points to this region, labeled "Model hallucination", and the region is labeled "q2". The y-axis ranges from 0.0 to 0.4.
>   - Bottom Row: Shows the combined result. The "Ground truth" is represented by a dashed black line, showing two modes. The approximate posterior, combined using a "Gaussian process", is shown as an orange line. The orange line captures the general shape but has some discrepancies. The y-axis ranges from 0.0 to 0.4, and the x-axis ranges from -9 to 15.
>
> - **Column C: Failure III: Underrepresented tails**
>   - Top Row: Shows subposterior p1(theta) as a dashed black line with blue circles representing MCMC samples. The distribution appears to be unimodal. The y-axis ranges from 0.0 to 1.5.
>   - Middle Row: Shows subposterior p2(theta). It shows a distribution with a shorter tail. A red arrow points to the missing tail, labeled "Unexplored tail". The y-axis ranges from 0.0 to 1.5.
>   - Bottom Row: Shows the combined result. The "Ground truth" is represented by a dashed black line, showing two modes. The approximate posterior, combined using a "Non-parametric" method, is shown as an orange line. The orange line captures the general shape but has a narrower distribution. The y-axis ranges from 0.0 to 1.5, and the x-axis ranges from -3.0 to 3.0.
>
> Each plot has labels on the y-axis, p1(theta), p2(theta), or p(theta|D), and the bottom row of plots has the x-axis labeled as theta. The legend in each bottom plot indicates the method used to combine the subposteriors.

Figure 1: Failure modes of embarrassingly parallel MCMC. A-C. Each column illustrates a distinct failure type described in Section 2.1. For each column, the top rows show two subposteriors $p_{k}(\theta)$ (black dashed line: ground truth; blue circles: MCMC samples), and the bottom row shows the full posterior $p(\theta \mid \mathcal{D})$ with the approximate posterior combined using the method specified in the legend (orange line). These failure modes are general and not unique to the displayed algorithms (see Appendix A for details and further explanations).

help recover the missing regions of the posterior if $q$ does not cover them in the first place. DIS can help in some model mismatch cases, in that 'hallucinated' regions of the posterior will receive near-zero weights after the true density is retrieved.

### 2.3 Proposed solution

Drawing from the insights in Section 2.1, we propose two key ideas to address the blind spots of embarrassingly parallel MCMC. Here we provide an overview of our solution, which is described in detail in Section 3. The starting point is modeling subposteriors via Gaussian process surrogates (Fig 2A).

Sample sharing. We introduce an additional step in which each node shares a selected subset of MCMC samples with the others (Fig 2B). This step provides sufficient information for local nodes to address mode collapse and underrepresented tails. While this communication step makes our method not strictly 'embarrassingly' parallel, we argue it is necessary to avoid
posterior density collapse. Moreover, existing methods already consider an extra communication step (Nemeth and Sherlock, 2018), as mentioned in Section 2.2.

Active learning. We use active learning as a general principle whenever applicable. The general idea is to select points that are informative about the shape of the subposterior, minimizing the additional communication required. Active learning is used here in multiple steps: when selecting samples from MCMC to build the surrogate model (as opposed to thinning or random subsampling); as a way to choose which samples from other nodes to add to the current surrogate model of each subposterior $q_{k}$ (only informative samples are added); to actively sample new points to reduce uncertainty in the local surrogate $q_{k}$ (Fig 2C). Active learning contributes to addressing both catastrophic model mismatch and underrepresented tails.

Combined, these ideas solve the failure modes discussed previously (Fig 2D).

---

#### Page 4

> **Image description.** The image is a figure with four panels, labeled A, B, C, and D, illustrating the process of Parallel Active Inference (PAI). Each panel (except D) contains two plots, one above the other.
>
> Panel A, titled "Fit GPs Subposteriors," shows two plots. The top plot displays "log p1(θ)" on the y-axis (ranging from -20 to 8) and "θ" on the x-axis (ranging from -2.4 to 3.6). A black dashed line represents the "Ground truth," an orange dashed line represents "GP," and blue circles represent "Samples." The area around the orange dashed line is shaded in light orange, indicating a confidence interval. The bottom plot is similar, showing "log p2(θ)" on the y-axis and "θ" on the x-axis, with the same elements (Ground truth, GP, Samples).
>
> Panel B, titled "Sample sharing Subposteriors," also shows two plots similar to those in Panel A. The top plot shows "log p1(θ)" vs "θ", with the same elements as before. The bottom plot shows "log p2(θ)" vs "θ", but here, the distribution of blue circle "Samples" differs from Panel A. Red arrows connect the top and bottom plots, indicating the sharing of samples between the two subposteriors.
>
> Panel C, titled "Active refinement Subposteriors," again shows two plots similar to the previous panels. The top plot shows "log p1(θ)" vs "θ", and the bottom plot shows "log p2(θ)" vs "θ". In these plots, blue stars represent "Active samples," which are added to refine the GP surrogates.
>
> Panel D, titled "Combination," shows a single plot labeled "Combined result." The y-axis is "log p(θ|D)" (ranging from -20 to 8) and the x-axis is "θ" (ranging from -2.4 to 3.6). An orange line represents "PAI," and a black dashed line represents the "Ground truth." The two lines overlap almost perfectly, indicating a good match between the approximate posterior and the true posterior.
>
> All plots have gridlines.

Figure 2: Parallel active inference (PAI). A. Each log subposterior $\log p_{k}(\theta)$ (black dashed line) is modeled via Gaussian process surrogates (orange dashed line: mean GP; shaded area: $95 \%$ confidence interval) trained on MCMC samples $\mathcal{S}_{k}$ (blue circles) and their log-density $\log p_{k}\left(\mathcal{S}_{k}\right)$. Here, MCMC sampling on the second subposterior has missed a mode. B. Selected subsets of MCMC samples are shared across nodes, evaluated locally and added to the GP surrogates. Here, sample sharing helps finding the missing mode in the second subposterior, but the GP surrogate is now highly uncertain outside the samples. C. Subposteriors are refined by actively selecting new samples (stars) that resolve uncertainty in the surrogates. D. Subposteriors are combined into the full approximate log posterior (orange line); here a perfect match to the true log posterior (black dashed line).

## 3 PARALLEL ACTIVE INFERENCE

In this section, we present our framework, which we call Parallel Active Inference (PAI), designed to address the issues discussed in Section 2. The steps of our method are schematically explained in Fig 2 and the detailed algorithm is provided in Appendix C.

### 3.1 Subposterior modeling via GP regression

As per standard embarrassingly parallel algorithms, we assume each node computes a set of samples $\mathcal{S}_{k}$ and their $\log$ density, $\log p_{k}\left(\mathcal{S}_{k}\right)$, by running MCMC on the subposterior $p_{k}$. We model each log-subposterior $\mathcal{L}_{k}(\theta) \equiv \log q_{k}(\theta)$ using GP regression (Fig 2A; see Rasmussen and Williams (2006); Nemeth and Sherlock (2018); Görtler et al. (2019) and Appendix B for more information). We say that a GP surrogate model is trained on $\mathcal{S}_{k}$ as a shorthand for $\left(\mathcal{S}_{k}, \log p_{k}\left(\mathcal{S}_{k}\right)\right)$.

When building the GP model of the subposterior, it is not advisable to use all samples $\mathcal{S}_{k}$ because: (1) exact inference in GPs scales cubically in the number of training points (although see Wang et al. (2019)); (2) we want to limit communication costs when sharing samples between nodes; (3) there is likely high redundancy in $\mathcal{S}_{k}$ about the shape of the subposterior. Nemeth and Sherlock (2018) simply choose a subset of samples by 'thinning' a longer MCMC chain at regular intervals. Instead, we employ active subsampling as follows.

First, we pick an initial subset of $n_{0}$ samples $\mathcal{S}_{k}^{(0)} \subset \mathcal{S}_{k}$, that we use to train an initial GP (details in Appendix C.1). Then, we iteratively select points $\theta^{*}$ from $\mathcal{S}_{k}$ by maximizing the maximum interquantile range (MAXIQR) acquisition function (Järvenpää et al., 2021):

$$
\theta^{*}=\arg \max _{\theta}\left\{e^{m\left(\theta ; \mathcal{S}_{k}^{(t)}\right)} \sinh \left(u \cdot s\left(\theta ; \mathcal{S}_{k}^{(t)}\right)\right)\right\}
$$

where $m\left(\theta ; \mathcal{S}_{k}^{(t)}\right)$ and $s\left(\theta ; \mathcal{S}_{k}^{(t)}\right)$ are, respectively, the posterior latent mean and posterior latent standard deviation of the GP at the end of iteration $t \geq 0$; and $\sinh (z)=(\exp (z)-\exp (-z)) / 2$ for $z \in \mathbb{R}$ is the hyperbolic sine. Eq. 2 promotes selection of points with high posterior density for which the GP surrogate is also highly uncertain, with the trade-off controlled by $u>0$, where larger values of $u$ favor further exploration. In each iteration $t+1$, we greedily select a batch of $n_{\text {batch }}$ points at a time from $\mathcal{S}_{k} \backslash \mathcal{S}_{k}^{(t)}$ using a batch version of MAXIQR (Järvenpää et al., 2021). We add the selected points to the current training set, $\mathcal{S}_{k}^{(t+1)}$, and retrain the GP after each iteration (see Appendix C.1). After $T$ iterations, our procedure yields a subset of points $\mathcal{S}_{k}^{\prime} \equiv \mathcal{S}_{k}^{(T)} \subseteq \mathcal{S}_{k}$ that are highly informative about the shape of the subposterior.

### 3.2 Sample sharing

In this step, each node $k$ shares the selected samples $\mathcal{S}_{k}^{\prime}$ with all other nodes (Fig 2B). Thus, node $k$ gains access

---

#### Page 5

to the samples $\mathcal{S}_{\backslash k}^{\prime} \equiv \bigcup_{j \neq k} \mathcal{S}_{j}^{\prime}$. Importantly, $\mathcal{S}_{\backslash k}^{\prime}$ might contain samples from relevant subposterior regions that node $k$ has has not explored. As discussed in Section 3.1, for efficiency we avoid adding all points $\mathcal{S}_{\backslash k}^{\prime}$ to the current GP surrogate for subposterior $k$. Instead, we add a sample $\theta^{*} \in \mathcal{S}_{\backslash k}^{\prime}$ to the GP training set only if the prediction of the current GP surrogate deviates from the true subposterior $\log p_{k}\left(\theta^{*}\right)$ in a significant way (see Appendix C. 2 for details). After this step, we obtain an expanded set of points $\mathcal{S}_{k}^{\prime \prime}$ that includes information from all the nodes, minimizing the risk of mode collapse (see Section 2.1).

### 3.3 Active subposterior refinement

So far, the GP models have been trained using selected subsets of samples from the original MCMC runs. In this step, we refine the GP model of each subposterior by sampling new points (Fig 2C). Specifically, each node $k$ actively selects new points by optimizing the MAXIQR acquisition function (Eq. 2) over $\mathcal{X} \subseteq \mathbb{R}^{D}$ (see Appendix C.3). New points are selected greedily in batches of size $n_{\text {batch }}$, retraining the GP after each iteration. This procedure yields a refined set of points $\mathcal{S}_{k}^{\prime \prime \prime}$ which includes new points that better pinpoint the shape of the subposterior, reducing the risk of catastrophic model mismatch and underrepresented tails. The final log-subposterior surrogate model $\mathcal{L}_{k}$ is the GP trained on $\mathcal{S}_{k}^{\prime \prime \prime}$.

### 3.4 Combining the subposteriors

Finally, we approximate the full posterior $\log p(\theta \mid \mathcal{D})=$ $\sum_{k=1}^{K} \log p_{k}(\theta)$ by combining all subposteriors together (Fig 2D). Since each log-subposterior is approximated by a GP, the approximate full log-posterior is a sum of GPs and itself a GP, $\mathcal{L}(\theta)=\sum_{k=1}^{K} \mathcal{L}_{k}(\theta)$. Note that $\mathcal{L}(\theta)$, being a GP, is still a distribution over functions. We want then to obtain a point estimate for the (unnormalized) posterior density corresponding to $\exp \mathcal{L}(\theta)$. One choice is to take the posterior mean, which leads to the expectation of a log-normal density (Nemeth and Sherlock, 2018). We prefer a robust estimate and use the posterior median instead (Järvenpää et al., 2021). Thus, our estimate is

$$
q(\theta) \propto \exp \left\{\sum_{k=1}^{K} m_{k}\left(\theta ; \mathcal{S}_{k}^{\prime \prime \prime}\right)\right\}
$$

In low dimension $(D=1,2)$, Eq. 3 can be evaluated on a grid. In higher dimension, one could sample from $q(\theta)$ using MCMC methods such as NUTS (Hoffman and Gelman, 2014), as done by Nemeth and Sherlock (2018). However, $q(\theta)$ is potentially multimodal which does not lend itself to easy MCMC sampling. Alternatively, Acerbi (2018) runs variational inference on $q(\theta)$ using as variational distribution a mixture of Gaussians with a large number of components. Finally, for moderate $D$,
importance sampling/resampling with an appropriate (adaptive) proposal would also be feasible.

As a final optional step, after combining the subposteriors into the full approximate posterior $q(\theta)$, we can refine the solution using distributed importance sampling (DIS) as proposed by Nemeth and Sherlock (2018) and discussed in Section 2.2.

### 3.5 Complexity analysis

Similarly to conventional embarrassingly parallel MCMC, we can split the cost of running PAI into two main components. The first consists of local costs, which involve computations happening at individual computing nodes (i.e., model fitting and active refinement). The second are global (or aggregation) costs, which comprise communication and sampling from the combined approximate posterior.

### 3.5.1 Model fitting

After sampling their subposterior, each computing node $k$ has to fit the surrogate model on the subset of their samples, $\mathcal{S}_{k}^{\prime}$. These subsets are designed such that their size is $\mathcal{O}(D)$ (see Appendix C). Thus, the cost of fitting the surrogate GP models in each of the $K$ computing nodes is $\mathcal{O}\left(D^{3}\right)$. The same applies for $\mathcal{S}_{k}^{\prime \prime}$ and $\mathcal{S}_{k}^{\prime \prime \prime}$.

### 3.5.2 Communication costs

Traditional embarrassingly parallel MCMC methods only require two global communication steps: (1) the central server splits the $N$ observations among $K$ computing nodes; (2) each node sends $S$ subposterior samples of dimension $D$ back to the server, assuming nodes draw the same number of samples. Together, these steps amount to $\mathrm{O}(N+K S D)$ communication cost.

PAI imposes another communication step, in which nodes share their subposterior samples and incurring $\mathrm{O}(K S D)$ cost. Furthermore, supposing PAI acquires $A$ active samples to refine each subposterior, the cost of sending local results to servers is increased by $\mathrm{O}(K A D)$. PAI also incur a small additional cost for sending back the value of the GP hyperparameters $\mathrm{O}(K D)$. In sum, since usually $A \ll S$, the asymptotic communication cost of PAI is equivalent to traditional methods.

### 3.5.3 Active refinement costs

Active learning involves GP training and optimization of the acquisition function, but only a small number of likelihood evaluations. Thus, under the embarrassingly parallel MCMC assumption that likelihood evaluations are costly (e.g., due to large datasets), active learning is relatively inexpensive (Acerbi, 2018). More importantly, as shown in our ablation study in Appendix D.1, this step is crucial to avoid the pathologies of embarrassingly parallel MCMC.

---

#### Page 6

### 3.5.4 Sampling complexity

Sampling from the aggregate approximate posterior $q(\theta)$ only requires evaluating the GP predictive mean for each subposterior and does not require access to the data or all samples. The sampling cost is linear in the number of subposteriors $K$ and the size of each GP $\mathcal{O}(D)$. Even if $K$ is chosen to scale as the size of the actual data, each GP only requires a small training set, making them comparably inexpensive.

## 4 EXPERIMENTS

We evaluate PAI on a series of target posteriors with different challenging features. Subsection 4.1 shows results for a posterior with four distinct modes, which is prone to mode collapse (Fig 1A). Subsection 4.2 targets a posterior with heavy tails, which can lead to underrepresented tails (Fig 1C). Subsection 4.3 uses a rare event model to gauge how well our method performs when the true subposteriors are drastically different. Finally, Subsection 4.4 concludes with a real-world application to a model and data from computational neuroscience (Acerbi et al., 2018). We provide implementation details in Appendix B. 3 and source code is available at https://github.com/spectraldani/pai.

Algorithms. We compare basic PAI and PAI with the optional distributed importance sampling step (PAIDIS) against six popular and state-of-the-art (SOTA) embarrassingly parallel MCMC methods: the parametric, non-parametric and semi-parametric methods by Neiswanger et al. (2014); PART (Wang et al., 2015); and two other GP-surrogate methods (Nemeth and Sherlock, 2018), one using a simple combination of GPs (GP) and the other using the distributed importance sampler (GP-DIS; see Section 2.2).

Procedure. For each problem, we randomly split the data in equal-sized partitions and divide the target posterior into $K$ subposteriors (Eq. 1). We run MCMC separately on each subposterior using Stan with multiple chains (Carpenter et al., 2017). The same MCMC output is then processed by the different algorithms described above, yielding a combined approximate posterior for each method. To assess the quality of each posterior approximation, we compute the mean marginal total variation distance (MMTV), the 2-Wasserstein (W2) distance, and the Gaussianized symmetrized Kullback-Leibler (GsKL) divergence between the approximate and the true posterior, with each metric focusing on different features. For each problem, we computed ground-truth posteriors using numerical integration (for $D \leq 2$ ) or via extensive MCMC sampling in Stan (Carpenter et al., 2017). For all GP-based methods (PAI, PAI-DIS, GP, GP-DIS), we sampled from the potentially multimodal combined GP (Eq. 3) using importance sampling/resampling
with an appropriate proposal. We report results as mean $\pm$ standard deviation across ten runs in which the entire procedure was repeated with different random seeds. For all metrics, lower is better, and the best (statistically significant) results for each metric are reported in bold. See Appendix D. 2 for more details.

### 4.1 Multi-modal posterior

Setting. In this synthetic example, the data consist of $N=10^{3}$ samples $y_{1}, \ldots, y_{N}$ drawn from the following hierarchical model:

$$
\begin{aligned}
\theta \sim p(\theta) & =\mathcal{N}\left(0, \sigma_{p}^{2} \mathbb{1}_{2}\right) \\
y_{1}, \ldots, y_{N} & \sim p\left(y_{n} \mid \theta\right)=\sum_{i=1}^{2} \frac{1}{2} \mathcal{N}\left(P_{i}\left(\theta_{i}\right), \sigma_{l}^{2}\right)
\end{aligned}
$$

where $\theta \in \mathbb{R}^{2}, \sigma_{p}=\sigma_{l}=1 / 4$ and $P_{i}$ 's are seconddegree polynomial functions. By construction, the target posterior $p(\theta \mid y_{1}, \ldots, y_{N}) \propto p(\theta) \prod_{n=1}^{N} p\left(y_{n} \mid \theta\right)$ is multimodal with four modes. We run parallel inference on $K=10$ equal-sized partitions of the data. We provide more details regarding $P_{1}, P_{2}$ in Appendix D.3.
Results. Fig 3 shows the output of each parallel MCMC method for a typical run, displayed as samples from the approximate combined posterior overlaid on top of the ground-truth posterior. Due to MCMC occasionally missing modes in subposterior sampling, the combined posteriors from all methods except PAI lack at least one mode of the posterior (mode collapse, as seen in Fig 1A). Other methods also often inappropriately distribute mass in low-density regions (as seen in Fig 1B). In contrast, PAI accurately recovers all the high-density regions of the posterior achieving a nearperfect match. Table 1 shows that PAI consistently outperforms the other methods in terms of metrics.

Table 1: Multi-modal posterior.

| Model       |                    MMTV                     |                     W2                      |                                 GsKL                                  |
| :---------- | :-----------------------------------------: | :-----------------------------------------: | :-------------------------------------------------------------------: |
| Parametric  |               $0.89 \pm 0.12$               |               $1.08 \pm 0.33$               |                      $8.9 \pm 11 \times 10^{2}$                       |
| Semi-param. |               $0.81 \pm 0.09$               |               $1.08 \pm 0.12$               |                      $5.6 \pm 1.3 \times 10^{1}$                      |
| Non-param.  |               $0.81 \pm 0.09$               |               $1.12 \pm 0.09$               |                      $5.0 \pm 1.8 \times 10^{1}$                      |
| PART        |               $0.55 \pm 0.09$               |               $1.06 \pm 0.33$               |                      $7.3 \pm 14 \times 10^{2}$                       |
| GP          |               $0.93 \pm 0.16$               |               $1.01 \pm 0.36$               |                      $1.2 \pm 1.3 \times 10^{4}$                      |
| GP-DIS      |               $0.87 \pm 0.18$               |               $1.04 \pm 0.34$               |                      $4.8 \pm 14 \times 10^{16}$                      |
| PAI         | $\mathbf{0 . 0 3 7} \pm \mathbf{0 . 0 1 1}$ | $\mathbf{0 . 0 2 8} \pm \mathbf{0 . 0 1 1}$ | $\mathbf{1 . 6} \pm \mathbf{1 . 7} \times \mathbf{1 0}^{-\mathbf{4}}$ |
| PAI-DIS     | $\mathbf{0 . 0 3 4} \pm \mathbf{0 . 0 1 9}$ | $\mathbf{0 . 0 2 6} \pm \mathbf{0 . 0 0 8}$ | $\mathbf{3 . 9} \pm \mathbf{2 . 4} \times \mathbf{1 0}^{-\mathbf{5}}$ |

Large datasets. To illustrate the computational benefits of using PAI for larger datasets, we repeated the same experiment in this section but with $10^{5}$ data points in each of the $K=10$ partitions. Remarkably, even for this moderate-sized dataset, we notice a $6 \times$ speed-up - decreasing the total running time from 6 hours to 57 minutes, ( 50 for subposterior sampling +7 from PAI; see Appendix D.4). Overall, PAI's running

---

#### Page 7

> **Image description.** This image is a figure containing eight scatter plots arranged in a 2x4 grid, visualizing multi-modal posterior distributions. Each plot represents a different method for approximating the posterior.
>
> The plots share a common structure:
>
> - The x-axis is labeled "θ₁" and ranges from -1 to 1.
> - The y-axis is labeled "θ₂" and ranges from -1 to 1.
> - The background of each plot is a gradient of blue, representing the "True log posterior density", with darker blue indicating higher density. A colorbar to the right of the grid shows the gradient, ranging from -94 (light blue) to 6 (dark blue).
> - Red markers (either dots or crosses) represent "Samples" from the approximate posterior. A legend in the top-left plot identifies these red markers as "Samples".
>
> The plots are labeled as follows:
>
> - Top row: "PAI (ours)", "PAI-DIS (ours)", "GP", "GP-DIS"
> - Bottom row: "Parametric", "Semi-parametric", "Non-parametric", "PART"
>
> The visual patterns in the plots vary significantly:
>
> - "PAI (ours)" and "PAI-DIS (ours)" show four clusters of red dots, one in each quadrant, closely aligned with the darker blue regions.
> - "GP" shows a single cluster of red dots at the center of the plot.
> - "GP-DIS" shows a single red cross at the center of the plot.
> - "Parametric" shows four short, curved segments of dark blue, one in each quadrant, and a few red dots.
> - "Semi-parametric" shows the same four curved segments of dark blue, but with a large number of red crosses clustered around the center.
> - "Non-parametric" shows a large cluster of red crosses in the upper-right quadrant, with some scattered crosses elsewhere.
> - "PART" shows a horizontal band of red crosses along the bottom edge and some scattered crosses in the upper-right quadrant.

Figure 3: Multi-modal posterior. Each panel shows samples from the combined approximate posterior (red) against the ground truth (blue). With exception of PAI, all methods completely miss at least one high-density region. Moreover, PAI is the only method that does not assign mass to regions without modes.

> **Image description.** The image consists of four panels arranged in a 2x2 grid, displaying plots comparing two methods, PAI (ours) and GP, for approximating posterior distributions. The top row shows line graphs, while the bottom row shows heatmaps.
>
> - **Top Row: Line Graphs**
>
>   - The y-axis label is "Log marginal post. density," spanning from -50 to 2.
>   - The x-axis is unlabeled but ranges from approximately -8 to 1.
>   - Each graph plots two lines: a solid blue line labeled "Ground truth" and a dashed red line labeled "Prediction."
>   - The left panel is titled "PAI (ours)" and shows the "Prediction" line closely following the "Ground truth" line.
>   - The right panel is titled "GP" and shows the "Prediction" line deviating significantly from the "Ground truth" line, especially in the middle range of the x-axis.
>
> - **Bottom Row: Heatmaps**
>   - The x-axis is labeled "θ1," ranging from -8 to 1.
>   - The y-axis is labeled "θ2," ranging from -3 to 3.
>   - Each heatmap visualizes the log posterior density with a color gradient, where darker shades of red indicate higher density.
>   - Both heatmaps display a curved, crescent-shaped region of high density.
>   - A colorbar to the right of the right panel indicates the mapping between color and "Log posterior density," ranging from -50 to 2.
>   - The left panel corresponds to "PAI (ours)" and shows a more continuous and defined crescent shape.
>   - The right panel corresponds to "GP" and shows a similar crescent shape, but it appears slightly less defined and more diffuse.

Figure 4: Warped Student's t. Top: Log marginal posterior for $\theta_{1}$. Bottom: Log posterior density. Thanks to active sampling, PAI better captures details in the depths of the tails.

time is in the same order of magnitude as the previous SOTA (e.g. Wang et al., 2015; Nemeth and Sherlock, 2018). However, only PAI returns correct results while other methods fail.

### 4.2 Warped Student's t

Setting. We now turn to a synthetic example with heavy tails. Consider the following hierarchical model:

$$
\begin{aligned}
\theta \sim p(\theta) & =\mathcal{N}\left(0, \sigma_{p}^{2} \mathbb{I}_{2}\right) \\
y_{1}, \ldots, y_{N} & \sim p\left(y_{n} \mid \theta\right)=\operatorname{StudentT}\left(\nu, \theta_{1}+\theta_{2}^{2}, \sigma_{l}^{2}\right)
\end{aligned}
$$

where $\theta \in \mathbb{R}^{2}, \nu=5$ is the degrees of freedom of the Student's $t$-distribution, $\sigma_{p}=1$, and $\sigma_{l}=\sqrt{2}$. This model is a heavy-tailed variant of the Warped Gaussian model studied in earlier work, e.g., Nemeth and Sherlock (2018); Mesquita et al. (2019). As before, we generate $N=10^{3}$ samples and split the data into $K=10$ partitions for parallel inference.
Results. Fig 4 shows the full posterior and the marginal posterior for $\theta_{1}$ obtained using the two bestperforming methods without DIS refinement, PAI and GP (see Table 2). While PAI(-DIS) is very similiar to GP(-DIS) in terms of metrics, Fig 4 shows that, unlike GP(-DIS), PAI accurately captures the far tails of the posterior which could be useful for downstream tasks, avoiding failure mode III (Fig 1C).

Table 2: Warped Student's t.

| Model       |                    MMTV                     |                     W2                      |                             GsKL                             |
| :---------- | :-----------------------------------------: | :-----------------------------------------: | :----------------------------------------------------------: |
| Parametric  |               $0.51 \pm 0.01$               |               $0.71 \pm 0.07$               |                 $1.9 \pm 0.1 \times 10^{0}$                  |
| Semi-param. |               $0.50 \pm 0.03$               |               $0.57 \pm 0.05$               |                 $1.1 \pm 0.2 \times 10^{1}$                  |
| Non-param.  |               $0.51 \pm 0.02$               |               $0.59 \pm 0.03$               |                 $1.2 \pm 0.2 \times 10^{1}$                  |
| PART        |               $0.66 \pm 0.07$               |               $0.78 \pm 0.09$               |                 $1.2 \pm 0.7 \times 10^{2}$                  |
| GP          | $\mathbf{0 . 0 1 5} \pm \mathbf{0 . 0 0 3}$ | $\mathbf{0 . 0 0 3} \pm \mathbf{0 . 0 0 2}$ |                $4.5 \pm 10.5 \times 10^{-4}$                 |
| GP-DIS      |              $0.018 \pm 0.004$              | $\mathbf{0 . 0 0 2} \pm \mathbf{0 . 0 0 1}$ |                 $6.6 \pm 5.8 \times 10^{-3}$                 |
| PAI         | $\mathbf{0 . 0 1 5} \pm \mathbf{0 . 0 0 3}$ | $\mathbf{0 . 0 0 2} \pm \mathbf{0 . 0 0 1}$ | $\mathbf{1 . 2} \pm \mathbf{0 . 8} \times \mathbf{1 0}^{-6}$ |
| PAI-DIS     |              $0.018 \pm 0.003$              | $\mathbf{0 . 0 0 2} \pm \mathbf{0 . 0 0 1}$ |                 $3.8 \pm 3.4 \times 10^{-3}$                 |

### 4.3 Rare categorical events

Setting. To evaluate how our method copes with heterogeneous subposteriors, we run parallel inference for a synthetic example with Categorical likelihood

---

#### Page 8

and $N=10^{3}$ discrete observations split among three classes. To enforce heterogeneity, we make the first two classes rare (true probability $\theta_{1}=\theta_{2}=1 / N$ ) and the remaining class much more likely (true probability $\theta_{3}=(N-2) / N)$. Since we split the data into $K=10$ equal-sized parts, some of them will not contain even a single rare event. We perform inference over $\theta \in \Delta^{2}$ (probability 2 -simplex) with a symmetric Dirichlet prior with concentration parameter $\alpha=1 / 3$.
Results. Fig 5 shows the samples from the combined approximate posterior for each method. In this example, PAI-DIS matches the shape of the target posterior extremely well, followed closely by GP-DIS (see also Table 3). Notably, even standard PAI (without the DIS correction) produces a very good approximation of the posterior - a further display of the ability of PAI of capturing fine details of each subposterior, particularly important here in the combination step due to the heterogeneous subposteriors. By contrast, the other methods end up placing excessive mass in very-low-density regions (PART, Parametric, GP) or over-concentrating (Non-parametric, Semi-parametric).

Table 3: Rare categorical events.

| Model       |                    MMTV                     |                              W2                              |                             GsKL                             |
| :---------- | :-----------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Parametric  |               $0.26 \pm 0.14$               |                       $0.15 \pm 0.19$                        |                 $1.1 \pm 1.4 \times 10^{0}$                  |
| Semi-param. |               $0.49 \pm 0.21$               |                       $0.27 \pm 0.23$                        |                 $3.5 \pm 3.4 \times 10^{0}$                  |
| Non-param.  |               $0.43 \pm 0.17$               |                       $0.19 \pm 0.25$                        |                 $2.8 \pm 3.9 \times 10^{0}$                  |
| PART        |               $0.31 \pm 0.14$               |                       $0.08 \pm 0.13$                        |                 $8.6 \pm 10 \times 10^{-1}$                  |
| GP          |               $0.16 \pm 0.09$               |                       $0.04 \pm 0.07$                        |                 $3.5 \pm 4.8 \times 10^{-1}$                 |
| GP-DIS      |              $0.011 \pm 0.002$              |                 $6.3 \pm 0.9 \times 10^{-4}$                 |                 $1.1 \pm 1.5 \times 10^{-4}$                 |
| PAI         |              $0.028 \pm 0.027$              |                      $0.001 \pm 0.002$                       |                 $8.0 \pm 16 \times 10^{-3}$                  |
| PAI-DIS     | $\mathbf{0 . 0 0 9} \pm \mathbf{0 . 0 0 2}$ | $\mathbf{5 . 4} \pm \mathbf{0 . 8} \times \mathbf{1 0}^{-4}$ | $\mathbf{4 . 3} \pm \mathbf{2 . 1} \times \mathbf{1 0}^{-5}$ |

### 4.4 Multisensory causal inference

Setting. Causal inference (CI) in multisensory perception denotes the process whereby the brain decides whether distinct sensory cues come from the same source, a commonly studied problem in computational and cognitive neuroscience (Körding et al., 2007). Here we compute the posterior for a 6 -parameter CI model given the data of subject S1 from (Acerbi et al., 2018) (see Appendix D. 3 for model details). The fitted model is a proxy for a large class of similar models that would strongly benefit from parallelization due to likelihoods that do not admit analytical solutions, thus requiring costly numerical integration. For this experiment, we run parallel inference over $K=5$ partitions of the $N=1069$ observations in the dataset.
Results. Table 4 shows the outcome metrics of parallel inference. Similarly to the rare-events example, we find that PAI-DIS obtains an excellent approximation of the true posterior, with GP-DIS performing about equally well (slightly worse on the GsKL metric). Despite lacking the DIS refinement step, standard PAI
performs competitively, achieving a reasonably good approximation of the true posterior (see Appendix D.3). All the other methods perform considerably worse; in particular the GP method without the DIS step is among the worst-performing methods on this example.

Table 4: Multisensory causal inference.

| Model       |                  MMTV                   |                   W2                    |                             GsKL                             |
| :---------- | :-------------------------------------: | :-------------------------------------: | :----------------------------------------------------------: |
| Parametric  |             $0.40 \pm 0.05$             |              $4.8 \pm 0.6$              |                 $1.2 \pm 0.4 \times 10^{1}$                  |
| Semi-param. |             $0.68 \pm 0.07$             |              $9.7 \pm 9.6$              |                 $5.6 \pm 3.2 \times 10^{1}$                  |
| Non-param.  |             $0.26 \pm 0.02$             |             $0.52 \pm 0.14$             |                 $5.3 \pm 0.3 \times 10^{0}$                  |
| PART        |             $0.24 \pm 0.04$             |              $1.5 \pm 0.5$              |                 $8.0 \pm 5.4 \times 10^{0}$                  |
| GP          |             $0.49 \pm 0.25$             |               $17 \pm 23$               |                 $6.3 \pm 8.9 \times 10^{1}$                  |
| GP-DIS      | $\mathbf{0 . 0 7} \pm \mathbf{0 . 0 3}$ | $\mathbf{0 . 1 6} \pm \mathbf{0 . 0 7}$ |                 $8.7 \pm 14 \times 10^{-1}$                  |
| PAI         |             $0.16 \pm 0.04$             |             $0.56 \pm 0.21$             |                 $2.0 \pm 1.7 \times 10^{0}$                  |
| PAI-DIS     | $\mathbf{0 . 0 5} \pm \mathbf{0 . 0 4}$ | $\mathbf{0 . 1 4} \pm \mathbf{0 . 1 3}$ | $\mathbf{2 . 9} \pm \mathbf{3 . 6} \times \mathbf{1 0}^{-1}$ |

## 5 RELATED WORKS

While the main staple of embarrassingly parallel MCMC is being a divide-and-conquer algorithm, there are other methods that scale up MCMC using more intensive communication protocols. For instance, Ahn et al. (2014) propose a distributed version of stochastic gradient Langevin dynamics (SGLD, Welling and Teh, 2011) that constantly passes around the chain state to computing nodes, making updates only based on local data. However, distributed SGLD tends to diverge from the posterior when the communications are limited, an issue highlighted by recent work (El Mekkaoui et al., 2021; Vono et al., 2022). Outside the realm of MCMC, there are also works proposing expectation propagation as a framework for inference on partitioned data (Vehtari et al., 2020; Bui et al., 2018).

Our method, PAI, builds on top of related work on GP-based surrogate modeling and active learning for log-likelihoods and log-densities. Prior work used GP models and active sampling to learn the intractable marginal likelihood (Osborne et al., 2012; Gunter et al., 2014) or the posterior (Kandasamy et al., 2015a; Wang and Li, 2018; Järvenpää et al., 2021). Recently, the framework of Variational Bayesian Monte Carlo (VBMC) was introduced to simultaneously compute both the posterior and the marginal likelihood (Acerbi, 2018, 2019, 2020). PAI extends the above works by dealing with partitioned data in the embarrassingly parallel setting, similarly to Nemeth and Sherlock (2018), but with the key addition of active learning and other algorithmic improvements.

## 6 DISCUSSION

In this paper, we first exposed several potential major failure modes of existing embarrassingly parallel MCMC methods. We then proposed a solution with

---

#### Page 9

> **Image description.** The image presents a figure composed of six ternary plots arranged in a 2x3 grid, along with a color bar on the right. Each ternary plot visualizes samples from combined approximate posterior (red) on top of the true posterior (blue).
>
> Each ternary plot has the following structure:
>
> - A black equilateral triangle forms the main frame of the plot.
> - The vertices of the triangle are labeled with "θ1", "θ2", and "θ3".
> - Numerical values "1.0" and "0.0" are placed near the "θ3" vertex.
> - Numerical values "0.0" and "0.011" are placed near the "θ1" and "θ2" vertices.
> - A blue gradient, representing the "true log posterior density", is visible within the triangle. The intensity of the blue color varies, indicating different density levels.
> - Red "x" marks are scattered across the triangle, representing "samples".
>
> The six ternary plots are labeled as follows:
>
> 1.  Top left: "PAI (ours)"
> 2.  Top middle: "PAI-DIS (ours)"
> 3.  Top right: "GP"
> 4.  Bottom left: "PART"
> 5.  Bottom middle: "Non-parametric"
> 6.  Bottom right: "Parametric"
>
> Each plot also has the numerical label "0.989, 0.011" above the triangle.
>
> To the right of the ternary plots, a vertical color bar is present. It ranges from blue at the top to light blue at the bottom. The top of the color bar is labeled "13", while the bottom is labeled "0". To the left of the color bar, the text "True log posterior density" is written vertically.
>
> Below the ternary plots, a legend is present that consists of a red dot and the text "Samples".

Figure 5: Rare categorical events. Each ternary plot shows samples from the combined approximate posterior (red) on top of the true posterior (blue). Note that the panels are zoomed in on the relevant corner of the probability simplex. Of all methods, PAI is the one that best captures the shape of the posterior.

our new method, parallel active inference (PAI), which incorporates two key strategies: sample sharing and active learning. On a series of challenging benchmarks, we demonstrated that 'vanilla' PAI is competitive with current state-of-the-art parallel MCMC methods and deals successfully with scenarios (e.g., multi-modal posteriors) in which all other methods catastrophically fail. When paired with an optional refinement step (PAI-DIS), the proposed method consistently performs on par with or better than state-of-the-art. Our results show the promise of the proposed strategies to deal with the challenges arising in parallel MCMC. Still, the solution is no silver bullet and several aspects remain open for future research.

### 6.1 Limitations and future work

The major limitation of our method, a common problem to surrogate-based approaches, is scalability to higher dimensions. Most GP-based approaches for Bayesian posterior inference are limited to up to $\sim 10$ dimensions, see e.g. Acerbi, 2018, 2020; Järvenpää et al., 2021. Future work could investigate methods to scale GP surrogate modeling to higher dimensions, for example taking inspiration from high-dimensional approaches in Bayesian optimization (e.g., Kandasamy et al., 2015b).

More generally, the validity of any surrogate modeling approach hinges on the ability of the surrogate model to faithfully represent the subposteriors. Active learning helps, but model mismatch in our method is still a potential issue that hints at future work combining PAI with more flexible surrogates such as GPs with more flexible kernels (Wilson and Adams, 2013) or deep neural networks (Mesquita et al., 2019). For the latter, obtaining the uncertainty estimates necessary
for active learning would involve Bayesian deep learning techniques (e.g., Maddox et al., 2019).

As discussed before, our approach is not 'embarrassingly' parallel in that it requires a mandatory global communication step in the sample sharing part (see Section 3.2). The presence of additional communication steps seem inevitable to avoid catastrophic failures in parallel MCMC, and has been used before in the literature (e.g., the DIS step of Nemeth and Sherlock, 2018). Our method affords an optional final refinement step (PAI-DIS) which also requires a further global communication step. At the moment, there is no automated diagnostic to determine whether the optional DIS step is needed. Our results show that PAI already performs well without DIS in many cases. Still, future work could include an analysis of the GP surrogate uncertainty to recommend the DIS step when useful.
