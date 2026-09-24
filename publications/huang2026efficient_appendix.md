# Efficient Adaptive Data Acquisition via Pretrained Belief Representations - Appendix

---

## Appendix

The appendix is organized as follows:

- In Appendix A, we provide additional details of POLAR, including the backbone architectures, the policy head, and the training details.
- In Appendix B, we describe the experimental setups in detail, covering task specifications, baseline implementations, and evaluation protocols for the Bayesian experimental design, hyperparameter optimisation, and molecular docking benchmarks.
- In Appendix C, we extend POLAR to a loss-driven active learning task.
- In Appendix D, we report additional experimental results, including sNMC upper bounds for location finding, a backbone ablation with TabPFNv2.5, best-performance comparisons on CES, and per-search-space breakdowns on HPO-B.
- In Appendix E, we provide an overview of the computational resources and software dependencies used in this work.

## A Additional details about POLAR

### A.1 Backbone

The default backbone in our experiments is TabICLv2 [52], the regression variant of the TabICL family. TabICLv2 is a transformer-based in-context learner pretrained entirely on synthetic tabular data. Architecturally, TabICLv2 inherits the three-stage compress-then-ICL design of TabICLv1 [51]: a column-wise transformer that embeds each feature in isolation, a row-wise transformer that aggregates feature embeddings into a single fixed-dimensional representation per row, and a dataset-wise transformer that performs in-context learning across the entire context-query set. TabICLv2 extends this design with target-aware row embeddings, repeated feature grouping, and a query-aware scalable softmax that improves generalisation to longer contexts than those seen during pretraining; the model is pretrained with the Muon optimiser on a large corpus of synthetic datasets generated to maximise prior diversity. We use the official regression checkpoint released by the authors, which is trained to predict 999 conditional quantiles of $y^\star$ via the aggregated pinball loss. As a robustness check, we additionally evaluate POLAR with the TabPFNv2.5 backbone [23], using the public `tabpfn-v2.5-regressor-v2.5_default.ckpt` checkpoint; results are reported in Appendix D.1.

The auxiliary prediction loss aligns the backbone with the current task distribution by supervising its predictive head on target points sampled within each trajectory. The exact form depends on which quantity the backbone is trained to predict.

TabPFNv2.5 outputs a categorical distribution over discretised output bins, which we treat as an approximate predictive density. The auxiliary loss is the standard negative log-likelihood

$$
\ell_{\mathrm{pred}}^{\mathrm{NLL}} = -\log q_\psi(y_m \mid x_m, \mathcal{D}_{t-1}). \tag{A1}
$$

TabICLv2 outputs $K = 999$ conditional quantile estimates $\hat{q}_\psi^{(\tau_k)}(x_m, \mathcal{D}_{t-1})$ for evenly spaced quantile levels $\tau_k = k/(K+1)$, $k = 1, \ldots, K$. We follow the original training objective and adopt the aggregated pinball loss

$$
\ell_{\mathrm{pred}}^{\mathrm{pinball}}\bigl(q_\psi(\cdot \mid x_m, \mathcal{D}_{t-1}), \, y_m\bigr) = \frac{1}{K} \sum_{k=1}^{K} \rho_{\tau_k}\bigl(y_m - \hat{q}_\psi^{(\tau_k)}(x_m, \mathcal{D}_{t-1})\bigr), \tag{A2}
$$

where $\rho_\tau(u) = \max(\tau u, (\tau-1)u)$ is the standard pinball (tilted absolute) loss for quantile level $\tau$. In all experiments, the policy loss in Equation (6) and the auxiliary prediction loss are weighted equally, $\lambda_{\mathrm{pol}} = \lambda_{\mathrm{pred}} = 1$. For density-based rewards and NLPD evaluation in Appendix C, following Qu et al. [52], we convert the predicted quantiles into a continuous distribution by enforcing monotonicity, linearly interpolating the quantile function between adjacent levels, extrapolating exponential tails beyond the extreme predicted quantiles, and then evaluating $\log q_\psi(y \mid x, \mathcal{D}) = -\log \partial_\tau Q_\psi(\tau \mid x, \mathcal{D})\vert_{\tau = F_\psi(y \mid x, \mathcal{D})}$; the predictive mean is computed as $\mu_\psi(x, \mathcal{D}) = \int_0^1 Q_\psi(\tau \mid x, \mathcal{D}) \, d\tau$.

### A.2 Policy head

The default policy head used throughout the main experiments is a lightweight MLP with two hidden layers, hidden width 128, and GELU activations. It maps each candidate-conditioned representation to a scalar logit, and the policy distribution over candidates follows the softmax.

For the loss-driven active learning experiments in Appendix C, where the goal is to optimise predictive performance over a user-specified target set, we use a target-aware variant of the policy head. This variant uses a single transformer encoder layer (4 attention heads, hidden width 1024, GELU activation), allowing each candidate token to attend to the target-point representations produced by the backbone.

During training, we optimise the policy head and the backbone using AdamW. Gradients are clipped to unit global norm. We sample candidates stochastically from the policy distribution to ensure sufficient exploration during training. At test time, we instead select the argmax candidate.

## B Experimental details

### B.1 Benchmarking on Bayesian experimental design tasks

**Location finding** [61] is a widely adopted benchmark for sequential BED [15, 14, 38, 33]. The objective is to recover the positions of $K$ unknown sources in $\mathbb{R}^d$, denoted $\theta = \{\theta_k \in \mathbb{R}^d\}_{k=1}^K$, by adaptively choosing measurement locations $x \in \mathbb{R}^d$ at which noisy signal intensities are recorded. Each source radiates a signal whose magnitude attenuates with distance under an inverse-square law, so that a measurement taken at location $x$ aggregates contributions from every source:

$$
\mu(\theta, x) = b + \sum_{k=1}^K \frac{\alpha_k}{m + \lVert \theta_k - x \rVert^2}, \tag{A3}
$$

where the constants $\alpha_k$ specify the strength of each source, while $b > 0$ and $m > 0$ govern the background offset and the saturation behaviour of the signal at short range, respectively. The experimenter does not observe $\mu$ directly; instead, the log-intensity is observed under additive Gaussian noise:

$$
\log y \mid \theta, x \sim \mathcal{N}\bigl(\log \mu(\theta, x), \sigma^2\bigr). \tag{A4}
$$

Following standard practice, we instantiate the task with $K = 2$, $\alpha_k = 1$, $b = 0.1$, $m = 10^{-4}$, and $\sigma = 0.5$. Designs are constrained to the unit hypercube $x \in [0,1]^d$, and the prior over each source places independent uniform distributions on every coordinate, $\theta_{k,j} \sim \mathrm{Unif}[0,1]$ for $j = 1, \ldots, d$.

**Constant elasticity of substitution** (CES; 2) originates in economic theory and models how a consumer values bundles of goods. In the experimental design formulation, the experimenter aims to identify a participant’s latent preference structure by repeatedly presenting them with pairs of baskets and recording a (potentially noisy) judgement about their relative desirability. A basket $z \in [0,100]^K$ encodes the quantities held of each of $K$ goods. The participant’s preferences are captured by latent parameters $\theta = (\rho, \boldsymbol{\alpha}, u)$, where $\rho \in (0,1)$ determines the degree of substitutability between goods, $\boldsymbol{\alpha} \in \Delta^{K-1}$ assigns simplex-valued weights to the goods, and $u > 0$ modulates the overall responsiveness of the participant’s reports. A single experimental design takes the form of a basket pair $x = (z, z') \in [0,100]^{2K}$, and elicits a bounded preference rating $y \in [0,1]$.

The utility assigned to a basket follows the canonical CES form: $U(z) = \left(\sum_{i=1}^{K} z_i^{\rho} \alpha_i\right)^{\frac{1}{\rho}}$. We adopt the following prior specification over the latent parameters:

$$
\rho \sim \mathrm{Beta}(1,1), \qquad \boldsymbol{\alpha} \sim \mathrm{Dirichlet}(\mathbf{1}_K), \qquad \log u \sim \mathcal{N}(1, 3^2). \tag{A5}
$$

For a given query $x = (z, z')$ and parameter setting $\theta$, the underlying (unobserved) utility differential is generated from

$$
\eta \sim \mathcal{N}\Bigl(u \, (U(z) - U(z')), \; u^2 \, \tau^2 \, (1 + \lVert z - z' \rVert)^2\Bigr), \tag{A6}
$$

and then transformed through a sigmoid link and clipped to a bounded interval to yield the reported outcome: $y = \mathrm{clip}\bigl(\sigma(\eta), \epsilon, 1-\epsilon\bigr)$, with $\sigma(\cdot)$ the standard sigmoid. We use $K = 3$, $\tau = 0.005$, $\epsilon = 2^{-22}$, and query budget $T = 10$ in all experiments. Since TFMs are pretrained on standardised inputs and outputs, we standardise the features before feeding them to the backbone.

#### B.1.1 Baselines

**Deep Adaptive Design** [14] learns an amortised design policy by directly maximising the sPCE lower bound on the total EIG over $T$-step trajectories. The policy network consists of a two-layer MLP encoder with hidden width that maps each design-observation pair $(x_i, y_i)$ to a 16-dimensional embedding, a permutation-invariant pooling operation that aggregates these embeddings into a history representation, and a one-layer MLP emitter that maps the pooled representation to the next design. We train with Adam (learning rate $5 \times 10^{-5}$, $\beta = (0.8, 0.998)$), gradient clipping at 1.0, and an exponential learning-rate decay with factor 0.98 every 1000 epochs. For location finding, we use $L = 2 \times 10^4$ contrastive samples for sPCE estimation during training. For CES, we use $L = 10^5$.

**RL-BOED** [5] formulates sequential BED as a hidden-parameter Markov decision process and learns an amortised design policy using reinforcement learning. The reward is defined stepwise as the marginal contribution of the newly acquired design-observation pair to the sPCE. We use Randomized Ensembled Double Q-learning (REDQ) to train the continuous-design policy. The history encoder follows a DAD-style permutation-invariant architecture: each concatenated design-observation pair $(x_i, y_i)$ is passed through a two-layer MLP with 128 hidden units and ReLU activations, followed by a 64-dimensional linear output layer; the resulting embeddings are summed to form the history representation. The policy emitter outputs the mean and log-variance of independent Tanh-Gaussian distributions over the design dimensions. The critic networks use the same history encoder and concatenate the encoded history with the candidate design before passing it through a two-layer 128-unit ReLU MLP. We train with $L = 10^5$ contrastive samples for the sPCE reward. The remaining RL hyperparameters are listed in Table A1.

Table A1: Hyperparameters used for RL-BOED.

| Parameter | Location Finding | CES |
| --- | --- | --- |
| Critics $N$ | 2 | 2 |
| Random target subset $M$ | 2 | 2 |
| Discount factor $\gamma$ | 0.9 | 0.9 |
| Target update rate $\tau$ | $10^{-3}$ | $5 \times 10^{-3}$ |
| Policy learning rate | $10^{-4}$ | $3 \times 10^{-4}$ |
| Critic learning rate | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| Replay buffer size | $10^7$ | $10^6$ |
| Minimum buffer size | $10^5$ | $10^5$ |
| Entropy-temperature learning rate | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ |

**ALINE** [33] jointly amortises sequential design and posterior inference with a shared masked transformer architecture. Each design $x$ and observation $y$ is embedded by separate two-layer MLPs with hidden width 128 into 32-dimensional representations; context tokens are formed by summing the design and observation embeddings, candidate-query designs are represented by design embeddings alone, and one learnable target token is introduced for each latent parameter dimension. These tokens are processed by a 3-layer transformer encoder with model width 32, feedforward width 128, and 4 attention heads. A one-hidden-layer MLP acquisition head maps the query representations to a categorical distribution over the candidate design pool, while a separate one-hidden-layer MLP posterior head predicts a 10-component Gaussian mixture for each target parameter token. We train ALINE with a joint objective consisting of a negative log-likelihood term for posterior prediction and a REINFORCE-style design loss based on one-step improvements in target log-likelihood, weighted equally with $\alpha = 1$ and discount factor $\gamma = 1$. For the sample-efficiency experiments, we use AdamW with cosine annealing and layer-wise learning rates of $10^{-3}$ for the acquisition and posterior heads and $2 \times 10^{-4}$ for the shared embedder and transformer.

#### B.1.2 Evaluation details

Following Foster et al. [14], we evaluate all amortised policies using the sequential Prior Contrastive Estimation (sPCE) lower bound and the sequential Nested Monte Carlo (sNMC) upper bound on the total expected information gain. Given a sampled trajectory $(\theta_0, h_T) \sim p(\theta, h_T \mid \pi)$ together with $L$ contrastive samples $\theta_{1:L} \sim p(\theta)$ drawn independently from the prior, the sPCE and sNMC estimators are defined as

$$
\mathcal{L}_T(\pi, L) = \mathbb{E}\left[\log \frac{p(h_T \mid \theta_0, \pi)}{\frac{1}{L+1}\sum_{\ell=0}^{L} p(h_T \mid \theta_\ell, \pi)}\right], \quad \mathcal{U}_T(\pi, L) = \mathbb{E}\left[\log \frac{p(h_T \mid \theta_0, \pi)}{\frac{1}{L}\sum_{\ell=1}^{L} p(h_T \mid \theta_\ell, \pi)}\right], \tag{A7}
$$

where the outer expectation is taken over $\theta_0, h_T \sim p(\theta, h_T \mid \pi)$ and $\theta_{1:L} \sim p(\theta)$. Both bounds become tight as $L \to \infty$ at a rate $O(L^{-1})$ [14]. For training, POLAR uses $L = 10^5$ contrastive samples for both tasks. For the evaluations, we use $L = 10^6$ contrastive samples for location finding and $L = 10^7$ for CES.

We report performance against the total number of training samples consumed during policy learning, where a "sample" denotes a single design-observation pair $(x, y)$ produced by the simulator. This convention provides a uniform unit of comparison that is independent of architectural choices such as batch size, replay-buffer capacity, or auxiliary loss terms.

For DAD and ALINE, every gradient step consumes one fresh batch of trajectories of length $T$, so the cumulative number of training samples after $E$ epochs is $E \cdot B \cdot T$, where $B$ is the batch size and $T$ is the experiment horizon. For RL-BOED, which maintains a replay buffer, only the trajectories newly synthesised at each epoch contribute to the sample count; the batch size of the RL update governs how many transitions are drawn from the buffer per gradient step, but does not generate new simulator data, and is therefore excluded from the accounting. Concretely, if RL-BOED rolls out $B_{\mathrm{env}}$ new trajectories per epoch, the cumulative count after $E$ epochs is $E \cdot B_{\mathrm{env}} \cdot T$.

For POLAR, the auxiliary prediction loss requires evaluating the backbone on $M$ target points per trajectory in addition to the $T$ design-observation pairs visited by the policy. Each target point corresponds to a fresh sample drawn from the simulator and therefore contributes to the total sample budget. We thus account for both sources, giving a per-epoch consumption of $B \cdot (T + M)$ and a cumulative count of $E \cdot B \cdot (T + M)$ after $E$ epochs. Our sample-efficiency comparison counts only the simulator samples consumed during policy learning for the target task; the cost of pretraining the TFM is not included. We consider this accounting appropriate because the backbones are pretrained entirely on synthetic tabular data that is task-agnostic, cheap to generate, and shared across all downstream applications. Once pretrained, the same checkpoint is reused without modification across BED, BO, and AL tasks, so the pretraining cost is amortised across all downstream uses rather than attributable to any single task, analogous to how downstream evaluations of ImageNet-pretrained vision models or pretrained language models typically report finetuning cost rather than pretraining cost. POLAR inherits this off-the-shelf reusability from the broader tabular foundation model paradigm.

Both POLAR and ALINE operate over a finite candidate pool at each acquisition step, whereas DAD and RL-BOED produce continuous designs directly. To match the original formulations of all baselines while keeping the pool-based methods well-resourced, we use a candidate pool of $|\mathcal{C}| = 2000$ designs sampled uniformly from the design space at each step for location finding, and $|\mathcal{C}| = 20000$ for CES.

### B.2 Hyperparameter optimisation

#### B.2.1 Task description

HPO-B [1] is a large-scale meta-dataset for black-box hyperparameter optimisation (HPO) assembled from the OpenML repository. It contains 6.4 million hyperparameter evaluations. We use the curated HPO-B-v3 split, which retains the 16 search spaces and provides predefined meta-train, meta-validation, and meta-test partitions, together with five fixed initialisation seeds per test task to support reproducible comparisons. Each search space corresponds to the hyperparameter space of a particular machine-learning algorithm (e.g., SVM, XGBoost, glmnet), and each meta-dataset within a search space corresponds to evaluations on a specific tabular dataset. The hyperparameter ranges are normalised to $[0,1]^d$ and categorical hyperparameters are one-hot encoded. Following Maraval et al. [45], we evaluate on six search spaces that together cover a representative range of underlying model families and input dimensionalities: `glmnet` (search space 5860, $d = 2$), `rpart.preproc` (search space 4796, $d = 3$), `rpart` (search space 5859, $d = 6$), `ranger` (search space 5889, $d = 6$), `svm` (search space 5527, $d = 8$), and `xgboost` (search space 5906, $d = 16$). We train on the provided meta-training datasets and evaluate on the held-out meta-test tasks. We adopt the recommended evaluation protocol of Arango et al. [1]: each test task is run with the five fixed initialisation seeds for $T = 50$ acquisition steps, and we report the average normalised regret and average rank as defined in Arango et al. [1].

#### B.2.2 Baselines

**GP.** A standard Gaussian process surrogate fitted directly on the test task, with no use of meta-training data. We use an exact GP with a constant mean, a scaled RBF kernel, and a Gaussian likelihood. The kernel and noise hyperparameters are optimised by maximizing the exact marginal likelihood on the observed configurations after each BO round.

**Meta-GP.** Following Maraval et al. [45], a meta-trained Gaussian process pretrains the RBF kernel hyperparameters on the meta-training datasets of the corresponding search space and uses these pretrained values to initialise the GP at test time. Subsequent acquisitions then refit the kernel hyperparameters on the growing test-time observation set, starting from this meta-learned initialisation rather than from a generic prior.

**PFNs4BO [47].** PFNs4BO uses Prior-data Fitted Networks [48] as transformer-based surrogates for BO: a single transformer is meta-trained offline to approximate the posterior predictive distribution of a chosen prior over functions, after which inference for any new context set is performed in a single forward pass without further fitting. The strongest variant of the model is trained on the so-called HEBO+ prior, which augments the HEBO Gaussian process prior [12] with input and output warping and additional hyperparameter randomisation, yielding a surrogate that has been shown to outperform standard GP surrogates on the HPO-B benchmark. We use the publicly released `hebo-plus-model` checkpoint from the official PFNs4BO repository without any further finetuning on the meta-training tasks.

**NAP [45].** NAP is an end-to-end differentiable meta-BO framework that jointly meta-learns a transformer-based neural process [50] surrogate and an acquisition function via reinforcement learning, with an auxiliary supervised likelihood loss that stabilises training and provides an inductive bias toward valid probabilistic predictions. NAP is the closest amortised baseline to POLAR, in that both methods learn the acquisition rule rather than relying on a hand-designed one. Since the HPO-B search spaces, test tasks, and initialisation seeds used by Maraval et al. [45] are identical to ours, we directly use the per-step regret values reported in their official release for fair comparison.

**TabICL [52].** A surrogate-based ablation that uses the same finetuned TabICL backbone as POLAR, but discards the learned policy head and instead pairs the backbone’s predictive distribution with the EI acquisition function. This baseline isolates the contribution of learning the acquisition rule from the contribution of the meta-learned belief representation: the difference between TabICL and POLAR can be attributed entirely to replacing a hand-designed acquisition with one trained on top of the same belief representation.

#### B.2.3 Evaluation details

For each HPO-B test task, all methods start from the same official initialisation and are evaluated on the same candidate pool. We report the normalised regret

$$
r_t = \frac{f^\star - f_t^{\mathrm{best}}}{f^\star - f_{\min}}, \tag{A8}
$$

where $f_t^{\mathrm{best}}$ is the best objective value observed by the method up to step $t$, $f^\star$ is the best value in the full task pool, and $f_{\min}$ is the worst value in the same pool. We average normalised regret over test datasets and initialisation seeds.

We also report the normalised rank to compare methods across tasks with different regret scales. For each task, seed, and optimisation step, methods are ranked by their normalised regret, with lower regret receiving a better rank and ties assigned the average rank. These ranks are then averaged over datasets, seeds, and search spaces.

### B.3 Molecular docking optimisation

#### B.3.1 Task description

DOCKSTRING [17] is a benchmark for molecular optimisation built around AutoDock Vina [66] docking simulations against a curated panel of medically relevant protein targets. The full dataset comprises docking scores and binding poses for over 260000 drug-like molecules evaluated against 58 targets, where each docking score quantifies the predicted binding affinity between a ligand and a target. Lower scores indicate stronger predicted binding, so the optimisation objective is to identify molecules that minimise the docking score for a given target. We frame this as a pool-based BO problem. Each molecule is represented by its SMILES string and converted to a 512-bit Morgan fingerprint with radius 2, matching the featurisation used by García-Ortegón et al. [17] for their property-prediction baselines. We randomly subsample 10000 molecules from the full dataset and use this fixed subset throughout all experiments. We split the 14 protein targets used in our experiments into a meta-training set of 8 targets, JAK2, KIT, LCK, SRC, IGF1R, ABL1, MET, EGFR, and a held-out test set of 6 targets, PTK2, FGFR1, CSF1R, CDK2, MAPK14, KDR. All 14 are protein kinases, which share a conserved ATP-binding pocket and substantial structural homology across the family. This shared geometry means that the physicochemical features predictive of strong binding are correlated across targets, making the kinase family a natural setting for meta-learning.

#### B.3.2 Baselines

**GP with Tanimoto kernel.** Standard continuous kernels such as the RBF or Matérn kernels are poorly suited to high-dimensional binary fingerprint inputs, where Euclidean distance is a weak measure of chemical similarity. We instead equip the GP with the Tanimoto kernel, a standard choice for fingerprint-based molecular BO [22]. For two binary fingerprints $x, x' \in \{0,1\}^d$, the Tanimoto kernel is defined as

$$
k_{\mathrm{Tanimoto}}(x, x') = \sigma^2 \cdot \frac{\langle x, x' \rangle}{\|x\|^2 + \|x'\|^2 - \langle x, x' \rangle}, \tag{A9}
$$

where $\sigma^2$ is a learnable output scale and $\langle \cdot, \cdot \rangle$ denotes the inner product. The kernel returns 1 for identical fingerprints and 0 for fingerprints with no shared bits, providing a meaningful similarity signal for sparse binary representations. We refit the kernel hyperparameters by maximum marginal likelihood after each acquisition step.

**Random Forest.** Random forests are a standard surrogate in cheminformatics and are commonly paired with molecular fingerprints [64, 68]. We use an ensemble of 256 regression trees with bootstrapped training sets and the standard $\sqrt{d}$ feature-subsampling rule at each split. The surrogate is refit on the full set of observed molecules at each acquisition step.

**TabICL.** We use the same setup as we used in our HPO-B experiments, where the same finetuned TabICL backbone as POLAR is used but replaces the learned policy head with a hand-designed acquisition function.

#### B.3.3 Evaluation protocol

For each of the 6 held-out test targets, we evaluate all methods under the same protocol: 10 randomly sampled molecules form the initial context, after which the policy selects $T = 50$ acquisitions, with a candidate pool of size $|\mathcal{C}| = 2000$. We report the average regret and aggregate by averaging across the 6 test targets and 5 random seeds for each test target.

## C Loss-driven active learning

In this section, we evaluate POLAR on a loss-driven active learning task, where the goal is to select query points that improve predictive performance on a downstream task distribution $p^\star$ under a user-specified loss.

Active learning seeks to acquire labelled data that maximises the predictive performance of a downstream model. In practice, a clinician may care primarily about predicting high-risk patients accurately, a structural engineer about responses near a failure threshold, and a drug discovery pipeline about molecules with high binding affinity. Standard information-theoretic acquisition objectives such as BALD [29] and EPIG [62] treat all outcomes symmetrically and therefore cannot directly target such loss-weighted criteria. Huang et al. [34] formalise this gap and derive a principled extension via weighted Bregman divergences, yielding myopic acquisition rules such as weighted variance reduction (GP-VR$_w$) that explicitly target a user-specified weighted loss. We adopt the same loss-driven framing but instantiate it in our amortised setting.

Concretely, we consider pool-based active learning under a weighted loss $\ell_\omega(y, q) = \omega(y)\ell(y, q)$, where $\omega : \mathcal{Y} \to \mathbb{R}_+$ encodes which outcomes matter most for the downstream task. At round $t$, the learner observes a context $\mathcal{D}_{t-1} = \{(x_i, y_i)\}_{i=1}^{t-1}$, selects a query $x_t$ from a finite pool $\mathcal{C}_t$, and observes $y_t \sim p(y \mid x_t, z)$ where $z$ is the latent function. After $T$ rounds, performance is evaluated on a held-out target set drawn from the target input distribution $p_\star(x^\star)$.

POLAR instantiates this setting using the same two-component architecture as in our other experiments. TFM plays the role of an amortised surrogate model: given the current context $\mathcal{D}_{t-1}$ and a query input $x^\star$, a single forward pass yields the approximate posterior predictive $q_\psi(y^\star \mid x^\star, \mathcal{D}_{t-1})$, replacing the GP surrogate that classical AL methods refit at every round. The policy head then scores each candidate in $\mathcal{C}_t$ from the same forward pass and samples the next query $x_t$. For the utility, we use the weighted one-step improvement in predictive log-density on the target set,

$$
u_t = \mathbb{E}_{x^\star \sim p^\star}\big[\omega(y^\star)\big(\log q_\psi(y^\star \mid x^\star, \mathcal{D}_t) - \log q_\psi(y^\star \mid x^\star, \mathcal{D}_{t-1})\big)\big], \tag{A10}
$$

which measures the weighted improvement in the backbone’s own predictive log-density on the target set after observing the new query, and which generalises the dense sEPIG reward of Huang et al. [33] from the unweighted to the weighted setting. In our experiments, we adopt the exponential weighting $\omega(y) = \exp(\beta y)$ with $\beta = 10$, prioritising regions of high outcome value. At deployment, no GP fitting or acquisition optimisation is required: the next query is produced in a single forward pass per round.

### C.1 Training distribution

We train POLAR on synthetic functions drawn from a distribution over GPs with randomised kernels and hyperparameters, following a procedure similar to ALINE [33]. Each task is sampled by:

1. Drawing a kernel uniformly from {RBF, Matérn-3/2, Matérn-5/2};
2. Drawing a length-scale $\ell \sim \mathrm{LogUniform}(0.1, 2.0)$ and an output scale $\sigma_f \sim \mathrm{Uniform}(0.1, 1.0)$;
3. Sampling a function $f \sim \mathcal{GP}(0, k_{\ell,\sigma_f})$.

Before each forward pass, we normalise each function using task-level statistics computed independently of the observed context: for inputs, we use the known domain bounds, and for outputs, we estimate the normalisation mean and variance from a large set of reference points sampled from that function. The horizon is $T = 20$, with an initial context of 2 points and a candidate pool of size $K = 200$.

### C.2 Baselines

We compare against four non-amortised GP baselines.

**GP-RS** (random sampling) selects $x_t$ uniformly at random from the pool.

**GP-US** (uncertainty sampling) selects the candidate with the highest posterior predictive variance,

$$
\mathrm{US}(x) = \mathbb{V}[y \mid x, \mathcal{D}_{t-1}]. \tag{A11}
$$

**GP-VR** (variance reduction) selects the candidate that maximally reduces total predictive variance over the target set $\{x^\star_m\}_{m=1}^{M}$:

$$
\mathrm{VR}(x) = \sum_{m=1}^{M} \frac{\mathrm{Cov}_{t-1}(x^\star_m, x)^2}{\mathbb{V}[y \mid x, \mathcal{D}_{t-1}]}, \tag{A12}
$$

where $\mathrm{Cov}_{t-1}(x^\star, x)$ is the GP posterior covariance between the latent function values at $x^\star$ and $x$ given $\mathcal{D}_{t-1}$.

**GP-VR$_\omega$** (weighted variance reduction; 34) is the loss-matched counterpart of GP-VR, derived from the same weighted-Bregman framework as our reward. It targets the weighted predictive variance under the reweighted predictive $p_\omega(y \mid x^\star) \propto \omega(y) p(y \mid x^\star)$ rather than the unweighted variance.

### C.3 Evaluations

We evaluate on two distributions of test functions.

**GP synthetic.** We sample 100 functions from the same generative procedure used during training. This measures in-distribution performance.

**Benchmark functions.** We additionally evaluate on four standard regression benchmarks to measure out-of-distribution generalisation: Forrester, Gramacy-Lee, Higdon, and Sine-Gaussian Bumps. The functional forms are:

$$
\begin{aligned}
f_{\mathrm{Forr}}(x) &= (6x - 2)^2 \sin(12x - 4), \\
f_{\mathrm{GL}}(x) &= \frac{\sin(10\pi x)}{2x} + (x - 1)^4, \\
f_{\mathrm{Higdon}}(x) &= \sin(2\pi x / 10) + 0.2 \sin(2\pi x / 2.5), \\
f_{\mathrm{SGB}}(x) &= 2\sin(2x) + 8\phi_{2.5, 0.5}(x) + 10\phi_{7.5, 0.25}(x) - 6\phi_{-4.5, 0.5}(x).
\end{aligned}
$$

where $f_0(x) = \sin(2x)$ and $\phi_{\mu, \sigma}$ is a Gaussian density with mean $\mu$ and standard deviation $\sigma$. Before evaluation, we apply a fixed per-function normalisation: inputs are centred and scaled using the benchmark domain, and outputs are standardised using statistics estimated from Sobol samples over the same domain. All methods, including GP and POLAR, are evaluated on these same normalized benchmark versions.

For each test function (whether GP-sampled or benchmark), we run 100 independent seeds. Each seed draws an initial context of 2 points, a candidate pool of 200 points, and a target set of 100 points; all three sets are sampled uniformly. Performance is reported as weighted RMSE and weighted NLPD on the target set:

$$
\mathrm{RMSE}_\omega = \sqrt{\frac{\sum_{m=1}^{M} \omega(y^\star_m)(y^\star_m - \mu(x^\star_m))^2}{\sum_{m=1}^{M} \omega(y^\star_m)}}, \qquad \mathrm{NLPD}_\omega = -\frac{\sum_{m=1}^{M} \omega(y^\star_m)\log q(y^\star_m \mid x^\star_m)}{\sum_{m=1}^{M} \omega(y^\star_m)}, \tag{A13}
$$

where $\mu(x^\star)$ is the predictive mean under the surrogate at the end of the acquisition trajectory.

### C.4 Results

Table A2 reports final-step weighted RMSE and weighted NLPD on both evaluation distributions. First, the two loss-aware methods GP-VR$_\omega$ and POLAR substantially outperform the loss-agnostic baselines (GP-RS, GP-US, GP-VR) on both metrics and both evaluation distributions. This confirms the central message of Huang et al. [34]: when the downstream loss assigns non-uniform weight to outcomes, acquisition rules that ignore the weighting are systematically suboptimal. Second, POLAR consistently outperforms GP-VR$_\omega$ across both evaluation distributions, suggesting that learning the acquisition rule directly from data outperforms a hand-designed one. Figure A1 visualises the acquisition behaviour of GP-VR, GP-VR$_\omega$, and POLAR on the Sine-Gaussian Bumps function from Huang et al. [34].

## D Additional experimental results

### D.1 Location finding

Figure A2 and Figure A3 report complementary results on the location finding task. Figure A2(a) and Figure A2(b) show the sNMC upper bound on EIG against total training samples in the 2D and 5D settings, respectively. The relative ordering of methods mirrors the sPCE comparison in Figure 2.

Figure A3(a) examines whether POLAR’s gains depend critically on the choice of pretrained backbone by replacing TabICLv2 with TabPFNv2.5 on the 2D task. The TabPFN-backed policy converges more slowly during the early stages of training, but the two variants reach essentially the same EIG at convergence. This suggests that POLAR is not tied to a particular backbone: as long as the pretrained model produces representations from which the task-relevant belief can be approximately decoded, a lightweight policy head can turn them into an effective acquisition policy.

Table A2: Loss-driven active learning. Final-step weighted RMSE and weighted NLPD on GP synthetic functions (in-distribution) and benchmark functions (out-of-distribution).

| Methods | GP Synthetic: $\mathrm{RMSE}_\omega$ ($\downarrow$) | GP Synthetic: $\mathrm{NLPD}_\omega$ ($\downarrow$) | Benchmark Functions: $\mathrm{RMSE}_\omega$ ($\downarrow$) | Benchmark Functions: $\mathrm{NLPD}_\omega$ ($\downarrow$) |
| --- | --- | --- | --- | --- |
| GP-RS | $0.40 \pm 0.06$ | $-0.18 \pm 0.28$ | $0.74 \pm 0.06$ | $3.09 \pm 0.78$ |
| GP-US | $0.13 \pm 0.02$ | $-1.26 \pm 0.16$ | $0.45 \pm 0.05$ | $4.20 \pm 0.75$ |
| GP-VR | $0.17 \pm 0.03$ | $-1.18 \pm 0.16$ | $0.33 \pm 0.03$ | $3.30 \pm 0.78$ |
| $\text{GP-VR}_\omega$ | $0.11 \pm 0.02$ | $-1.65 \pm 0.15$ | $0.15 \pm 0.02$ | $-0.83 \pm 0.33$ |
| POLAR (ours) | $\mathbf{0.09} \pm 0.03$ | $\mathbf{-2.07} \pm 0.13$ | $\mathbf{0.05} \pm 0.01$ | $\mathbf{-1.68} \pm 0.10$ |

> **Image description.** A 3-row by 5-column grid of paired plots showing acquisition behaviour on the Sine-Gaussian Bumps function, one column per round (Round 10, 15, 20, 25, 30, left to right) and one row per method (labelled GP-VR, GP-VR$_w$, Ours, top to bottom). Each cell stacks two subplots sharing an x-axis (the 1D input domain, roughly $-8$ to $8$; tick labels $-5$, $0$, $5$ are printed only under the bottom row): a taller function plot above and a shorter acquisition-score plot below it. A shared legend beneath the grid marks: True (orange line, the underlying test function), Pred mean (blue line, the surrogate's predictive mean, with a light blue shaded uncertainty band), Next query (vertical red line), and Queried (black dots).
>
> The test function is visible in every function subplot: it oscillates with small amplitude across most of the domain (y-axis ticked at 0 and 20) and rises sharply to a peak near 20 close to the right edge ($x \approx 7$–$8$), with a smaller negative dip visible near $x \approx -4.5$.
>
> **Top row (GP-VR):** queried points (black dots) are spread fairly evenly across the whole domain in every round, and the predictive mean tracks the true function reasonably well except near the sharp right-edge peak, which it underestimates in Rounds 10 and 15 and matches from Round 20 onward, once points have been queried there. The acquisition-score subplots below show a broad, multi-modal curve spanning the entire x-range with several comparable local maxima rather than one dominant peak; the next-query marker (red vertical line) lands at different locations from round to round, tracking whichever bump is momentarily highest — consistent with the caption's description of GP-VR spreading queries across the domain.
>
> **Middle row (GP-VR$_w$):** the acquisition-score curves are more concentrated, typically showing one or two dominant narrow peaks instead of many small ones. The right-edge peak already has queried points at Round 10, the predictive mean follows it in every round shown, and more queried points accumulate there as rounds progress. The next-query marker sits at the highest peak of the acquisition curve, which moves across the domain (left side at Round 10, near the middle in Rounds 15–25) and reaches the right-edge peak at Round 30.
>
> **Bottom row (Ours):** the acquisition-score curve becomes narrower and taller across rounds (top y-axis tick 0.03, 0.05, 0.03, 0.25 and 0.50 for Rounds 10–30). From Round 20 onward its highest peak and the next-query marker sit on the right-edge peak, and in Rounds 25 and 30 it is a single sharp spike there, apart from a tiny bump on the left. Queried points increasingly stack up at and around this peak in the later rounds; the predictive mean underestimates the peak in Rounds 10 and 15 and matches it from Round 20 onward — illustrating the caption's point that POLAR concentrates queries near the high-value region.

Figure A1: Loss-driven active learning. Acquisition behaviour on the Sine-Gaussian Bumps function from Huang et al. [34]. The loss-agnostic GP-VR (top) spreads queries across the domain, while GP-VR$_\omega$ and POLAR (middle, bottom) concentrate queries near the high-value region.

Another natural question is which layer of the backbone supplies the most informative belief representation for policy learning. To test this, we re-train POLAR on the 2D task using representations taken from the last, second-to-last, fourth-to-last, and eighth-to-last transformer layers of the TabICLv2 backbone, keeping all other components fixed. Figure A3(b) shows the last and second-to-last layers perform comparably throughout training, with the second-to-last layer slightly ahead in the low-sample regime and the last layer marginally ahead at convergence. Earlier layers degrade performance monotonically. A natural concern is that the layer ordering in Figure A3(b) reflects the effect of finetuning rather than the structure of the pretrained backbone. Figure A3(c) repeats the comparison with a frozen backbone and shows the same ordering, confirming that later layers provide more informative belief representations regardless of whether the backbone is adapted. Therefore, we adopt the last layer as the default in all other experiments.

> **Image description.** Two side-by-side line plots of sNMC (an EIG upper bound) against training
> samples, for the 2D and 5D location-finding tasks (a, b), following the same style as Figure 2(a,b)
> but plotting sNMC instead of EIG and comparing four methods.
>
> **Panel (a)** ("Location Finding 2D"): x-axis "Training samples" on a log scale with ticks
> $3\times10^4$, $3\times10^5$, $3\times10^6$, $3\times10^7$, $3\times10^8$; y-axis "sNMC" from
> about 5 to 20. Four methods are shown as connected markers per the legend inside the panel: DAD
> (pink pentagons), RL-BOED (green circles), ALINE (light-blue triangles), and POLAR (orange stars),
> each with small horizontal error-bar caps (standard error across 1000 runs). DAD, RL-BOED and
> ALINE each have four points spanning $3\times10^5$ to $3\times10^8$: DAD rises from about 5.7 to
> 7.0 to 9.9 to 10.7; RL-BOED rises from about 3.2 to 8.8 to 10.6 to 12.3; ALINE rises from about 3.4
> to 5.3 to 9.8 and then jumps to about 18.5 at the last point, ending highest of the three. POLAR has
> only three points, at $3\times10^4$, $3\times10^5$ and $3\times10^6$, rising steeply from about 9.8
> to 15.4 to 20.0 and then stopping — already above every other curve's final value while using far
> fewer training samples.
>
> **Panel (b)** ("Location Finding 5D"): same layout, x-axis ticks $5\times10^4$, $5\times10^5$,
> $5\times10^6$, $5\times10^7$, $5\times10^8$; y-axis "sNMC" from about 2.5 to 15. No legend is
> shown in this panel (it appears in panel a). DAD, RL-BOED and ALINE again span four points from
> $5\times10^5$ to $5\times10^8$: DAD rises from about 1.6 to 4.8 to 9.3 to 10.5; RL-BOED rises from
> about 2.0 to 7.2 to 9.3 and then stays essentially flat at 9.3; ALINE rises from about 4.7 to 6.3 to
> 9.1 and then jumps to about 14.5, again ending highest. POLAR again has only three points, at
> $5\times10^4$, $5\times10^5$ and $5\times10^6$, rising from about 11.2 to 12.0 to 15.5 (steepest over the
> last step), above every other curve's final value while using far fewer training samples.

Figure A2: Location finding. (a) EIG upper bound (sNMC) against total training samples in the 2D setting. Error bars denote standard error across 1000 runs. (b) The same comparison in the 5D setting.

> **Image description.** Three grouped bar charts, all sharing the x-axis "Training samples" with
> three groups ($3\times10^4$, $3\times10^5$, $3\times10^6$) and the y-axis "EIG($\uparrow$)", each
> bar carrying a small error-bar cap at its top.
>
> **Panel (a)** ("Backbone Choice"): two bars per group — "POLAR (TabICL)" in orange and
> "POLAR (TabPFN)" in blue — y-axis from 0 to about 13.5. At $3\times10^4$, TabICL reaches about
> 9.7 versus TabPFN's about 7.0 (TabICL clearly higher). At $3\times10^5$ the two are close, about
> 12.6 (TabICL) versus 12.8 (TabPFN). At $3\times10^6$ they are again close, about 13.4 (TabICL)
> versus 13.3 (TabPFN). The gap between backbones shrinks as training samples increase.
>
> **Panel (b)** ("Representation Layer"): four bars per group, shaded from pale cream to dark
> brown — "Last layer", "2nd-to-last", "4th-to-last", "8th-to-last" — same y-axis scale (0 to about
> 13.5) as panel (a). At $3\times10^4$ the bars stand at roughly 9.5, 10.1, 8.8 and 8.1 (2nd-to-last
> slightly exceeds Last layer here). At $3\times10^5$: about 12.6, 12.8, 12.0, 10.6 (2nd-to-last again
> marginally ahead). At $3\times10^6$: about 13.5, 13.4, 13.2, 11.8, where the ordering is monotonic —
> Last layer $\geq$ 2nd-to-last $\geq$ 4th-to-last $\geq$ 8th-to-last. At every budget the 4th- and
> 8th-to-last layers give lower EIG than the last two, and the 8th-to-last is lowest.
>
> **Panel (c)** ("Representation Layer (Frozen)"): the same four bars and colors as panel (b), now
> labeled "Last layer (Frozen)", "2nd-to-last (Frozen)", "4th-to-last (Frozen)", "8th-to-last
> (Frozen)", but on a shorter y-axis (0 to 12). At $3\times10^4$: about 9.7, 9.8, 9.2, 8.0. At
> $3\times10^5$: about 10.9, 10.6, 9.9, 8.3. At $3\times10^6$: about 11.8, 11.4, 10.7, 9.6. Later layers
> again score higher (2nd-to-last marginally above Last layer at $3\times10^4$; strictly decreasing from
> Last layer to 8th-to-last at $3\times10^5$ and $3\times10^6$). At $3\times10^5$ and $3\times10^6$
> every frozen bar is lower than its finetuned counterpart in panel (b); at $3\times10^4$ the frozen and
> finetuned bars are similar.

Figure A3: Location finding. (a) Ablation study on the backbone choice. (b) Ablation study on the representation layer with backbone finetuning. (c) Ablation study on the representation layer with a frozen backbone.

### D.2 Constant elasticity of substitution

Table A3 reports each method’s best-attained performance. POLAR achieves the strongest sPCE among all methods, retaining its lead from the matched-budget comparison in Table 1. During training, we observed that both DAD and RL-BOED were prone to instability: their training losses diverged in the late stages, with sPCE peaking and then degrading. We therefore applied early stopping based on validation sPCE for both baselines and report the checkpoint at which validation sPCE was highest. A separate observation concerns the sPCE–sNMC gap. For RL-BOED, this gap is markedly larger than for the other methods. A wide gap at finite contrastive samples indicates that the EIG bounds are loose for this particular policy, rather than that the true EIG is correspondingly large.

Table A3: CES. Best performance comparison on the CES task. Results are reported as mean $\pm$ s.e. over 1,000 independent runs.

| Methods | sPCE | sNMC |
| --- | --- | --- |
| Random | $9.18 \pm 0.18$ | $11.86 \pm 0.34$ |
| DAD | $12.87 \pm 0.14$ | $16.44 \pm 0.39$ |
| RL-BOED | $14.19 \pm 0.09$ | $33.33 \pm 1.80$ |
| ALINE | $13.43 \pm 0.10$ | $18.13 \pm 0.28$ |
| POLAR | $14.32 \pm 0.09$ | $20.99 \pm 0.42$ |

> **Image description.** A 2×3 grid of step-function line plots, one per HPO-B search space
> (4796, 5859, 5860 on top; 5906, 5889, 5527 below), each showing Average Regret (log-scale y-axis)
> against Average step (x-axis, roughly 0 to 50, with tick marks at 0, 20 and 40). Seven methods are
> drawn in every panel with a shared color/marker code given in one legend below the grid: Random
> (gray triangles), GP (dark-blue circles), Meta-GP (light-blue crosses), PFNs4BO (magenta diamonds),
> NAP (green squares), TabICL (gold diamonds), and POLAR (orange stars). Each curve is a staircase
> with markers at roughly every 8 steps and a shaded band (one standard error across test tasks and
> seeds) around it; all curves in a panel start from the same value at step 0. The y-axis range differs
> from panel to panel (e.g. roughly $10^{-3}$–$10^{-1}$ for search space 4796 versus roughly
> $10^{-2}$–$10^{-1}$ for search space 5527), so absolute regret levels are not directly comparable
> across panels.
>
> **Search space 4796** (top left): POLAR separates from the pack earliest (by step $\approx$8) and,
> after a mid-trajectory plateau near $10^{-3}$, plunges sharply near the final steps to below the
> bottom of the axis, ending clearly lowest. NAP also reaches a low plateau (around $1.5\times10^{-3}$)
> by step $\approx$14, below POLAR until step $\approx$26. PFNs4BO sits in between, ending near
> $7\times10^{-3}$. Random, GP, Meta-GP and TabICL remain clustered much higher (roughly
> $3$–$5\times10^{-2}$ at the end) with little improvement.
>
> **Search space 5859** (top middle): POLAR drops fastest early and plateaus around $10^{-2}$ by
> step $\approx$16; NAP drops even further around step 32 to about $5\times10^{-3}$, ending lowest.
> Random stays highest throughout, declining only slowly. GP, Meta-GP, PFNs4BO and TabICL form an
> intermediate band that declines gradually.
>
> **Search space 5860** (top right): NAP stays close to its starting value (around $5\times10^{-2}$)
> throughout, the highest curve. POLAR declines fastest, reaches about $3\times10^{-4}$ by step
> $\approx$16–24 and falls off the bottom of the axis around step 28. PFNs4BO settles near
> $2\times10^{-4}$ from step $\approx$24 and falls off the bottom of the axis around step 44. GP holds a
> high plateau (around $3\times10^{-2}$) until a sharp drop near step 44 to about $2\times10^{-4}$.
> Random, Meta-GP and TabICL end on intermediate plateaus between about $7\times10^{-4}$ and
> $2\times10^{-3}$.
>
> **Search space 5906** (bottom left): POLAR again separates earliest and plateaus low (around
> $1.3\times10^{-2}$) from step $\approx$8 onward. Meta-GP tracks a middle path before plunging
> sharply around steps 36–40, ending below the visible axis range and undercutting POLAR. NAP holds
> a middle plateau (around $3\times10^{-2}$); Random, GP, PFNs4BO and TabICL stay clustered near
> their starting value with little decline.
>
> **Search space 5889** (bottom middle): curves decline gradually and stay closer together than in
> other panels, mostly between $10^{-1}$ and $10^{-2}$, with no early separation. TabICL runs lowest
> for most of the trajectory, drops to about $7\times10^{-3}$ around step 37 and plunges off the bottom
> of the axis at the final step, ending lowest. POLAR ends around $2\times10^{-2}$, between NAP and
> PFNs4BO, not clearly separated from them in this panel.
>
> **Search space 5527** (bottom right): the narrowest axis range of the six (roughly $10^{-2}$ to a
> bit above $10^{-1}$). POLAR separates early and holds the lowest regret until about step 30
> (plateau around $3\times10^{-2}$); NAP declines steadily, overtakes POLAR with a drop near step 34
> and ends lowest, at about $1.4\times10^{-2}$. GP remains highest and essentially flat around
> $1.2\times10^{-1}$ throughout. Random, TabICL,
> PFNs4BO and Meta-GP cluster in between, declining only slightly.

Figure A4: HPO-B. Per-search-space normalised regret over the acquisition trajectory. Shaded regions denote one standard error across test tasks and initialisation seeds.

### D.3 Hyperparameter optimisation

Figure A4 and Figure A5 break down the aggregate results of Section 6.2 by search space, reporting the normalised regret and the average rank, respectively, for each of the six HPO-B search spaces. POLAR is consistently among the top performers across search spaces, achieving the lowest regret on most of them and remaining competitive on the others. The advantage over the strongest baselines is most pronounced on the higher-dimensional search spaces (5527 and 5906), where the meta-learned belief representation appears to confer the largest benefit. On the lower-dimensional search spaces the comparison is closer, but POLAR retains either the best or second-best regret throughout the acquisition trajectory, and the average-rank curves in Figure A5 confirm that this consistency holds across tasks and seeds rather than being driven by a small number of favourable cases.

## E Computational resources and software

All experiments presented in this work, including model development, hyperparameter tuning, baseline evaluations, and preliminary analyses, were performed on a GPU cluster equipped with NVIDIA H200 GPUs. The total computational resources consumed for this research, across all development stages and experimental runs, are estimated to be approximately 2000 GPU hours. The wall-clock cost of training POLAR varies with the task: on the HPO-B benchmark, a strong policy is typically obtained within 30 minutes per search space; on the location finding, policy training takes around 5 hours; on the CES, it takes about 16 hours to get the final best performance; on the DOCKSTRING benchmark, the high input dimensionality increases the per-step cost of the backbone forward pass, and a full training run takes approximately one day. The core code base is implemented in PyTorch (https://pytorch.org, License: modified BSD license), which also underlies our policy head as well as the TabPFN and TabICL backbones. We use the publicly released TabPFNv2.5 backbone from the PriorLabs repository (https://github.com/PriorLabs/TabPFN, License: Apache 2.0), with the corresponding checkpoint hosted on Hugging Face (https://huggingface.co/Prior-Labs/tabpfn_2_5). For the TabICLv2 backbone, we use the official implementation released by the authors (https://github.com/soda-inria/tabicl, License: BSD 3-Clause), with the regression checkpoint hosted on Hugging Face (https://huggingface.co/jingang/TabICL-clf). BED baselines are adapted from the original authors’ publicly available code. For the HPO-B baselines, we use the publicly released hebo-plus-model checkpoint from the official PFNs4BO repository (https://github.com/automl/PFNs4BO, License: Apache 2.0), and we use the per-step regret values released as part of the official NAP codebase (https://github.com/huawei-noah/HEBO/tree/master/NAP) for the NAP comparison. The Gaussian process baselines are implemented using GPyTorch (https://github.com/cornellius-gp/gpytorch, License: MIT). The HPO-B benchmark is obtained from its official repository (https://github.com/releaunifreiburg/HPO-B, License: MIT), and the DOCKSTRING benchmark is obtained from its official repository (https://github.com/dockstring/dockstring, License: MIT).

> **Image description.** A 2×3 grid of line plots, one per HPO-B search space (4796, 5859, 5860 on
> top; 5906, 5889, 5527 below), companion to Figure A4 but plotting Average Rank (linear y-axis,
> 1 to 7, shared across all six panels) against Acquisition step (x-axis, roughly 0 to 50, ticks at
> 0, 20, 40). The same seven methods and color/marker code as Figure A4 are used, with one shared
> legend below the grid: Random (gray triangles), GP (dark-blue circles), Meta-GP (light-blue
> crosses), PFNs4BO (magenta diamonds), NAP (green squares), TabICL (gold diamonds), and POLAR
> (orange stars). Every curve starts at rank 4 at step 0 (the tied average of 7 methods) and
> fans out over the first few steps, then each is a step-like curve with a shaded standard-error band
> across test tasks and seeds.
>
> **Search space 4796** (top left): Random rises quickly to about 5.5 and GP climbs gradually to
> about 5.5; both end at the top of the ranking. POLAR drops to about rank 2.7 within the first few
> steps and stays among the lowest, ending lowest; PFNs4BO and NAP stay close to it (around rank 3,
> PFNs4BO briefly below it mid-trajectory), while TabICL and Meta-GP sit around rank 4.
>
> **Search space 5859** (top middle): Random rises highest of all six panels, to about 6.5, and stays
> there — the clearest worst performer. POLAR dips to about rank 2 by step $\approx$8, the lowest in
> this panel, before drifting up to about 3.3 by the end. TabICL climbs to about 4.5, second only to
> Random; GP and Meta-GP settle around 3.5–4 by the end; NAP and PFNs4BO cluster near rank 3.
>
> **Search space 5860** (top right): ranks stay more tightly clustered than in the other panels,
> mostly between 3 and 5, with no method separating sharply. POLAR and PFNs4BO trend toward the
> bottom of the band (around rank 3) by the end, joined by GP after it drops from about 4 to 3.3
> near step 43, while Random, Meta-GP, NAP and TabICL remain close together near 4–5.
>
> **Search space 5906** (bottom left): POLAR dips lowest early (around rank 1.7–2, roughly steps
> 8–16) then rises back to about 3.3 by the end. Meta-GP drops quickly to about 2.3–2.5, overtakes POLAR
> as lowest from around step 22, and falls further to about 1.8 after step 36. NAP holds a middle plateau (around
> rank 3); GP and TabICL trend highest, reaching about 5.5–6.
>
> **Search space 5889** (bottom middle): ranks are less separated than in most other panels, mostly
> between 3 and 5 throughout. TabICL drops to about rank 3 or below within the first few steps and
> stays lowest for most of the trajectory (PFNs4BO dips briefly to about 2.5 near step 8), ending
> near 2.7; PFNs4BO and POLAR join it near rank 3 after about step 20; GP and
> Meta-GP trend toward the top (around rank 5), with Random and NAP in between.
>
> **Search space 5527** (bottom right): GP rises to and holds the highest rank (around 5.5–5.7)
> throughout. POLAR drops to the lowest rank (about 2.6–3) within the first few steps and stays
> there; NAP declines gradually from about 3.5 to join it near rank 3 by the end. Random (about 4.7),
> TabICL (about 4.1), and PFNs4BO and Meta-GP (about 3.7) sit in between.

Figure A5: HPO-B. Per-search-space average rank over the acquisition trajectory. Shaded regions denote one standard error across test tasks and initialisation seeds.

---

*Transcribed from the PDF text layer and corrected with LLMs; text, equations, tables, and figure descriptions may contain mistakes.*
