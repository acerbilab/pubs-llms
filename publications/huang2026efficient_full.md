```
@inproceedings{huang2026efficient,
  title={Efficient Adaptive Data Acquisition via Pretrained Belief Representations},
  author={Daolang Huang and Zhuoyue Huang and Conor Hassan and Luigi Acerbi and Samuel Kaski and Tom Rainforth},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year={2026},
  url={https://arxiv.org/abs/2606.25197}
}
```

---

# Efficient Adaptive Data Acquisition via Pretrained Belief Representations

Daolang Huang<sup>1,2</sup>, Zhuoyue Huang<sup>5</sup>, Conor Hassan<sup>1,2</sup>, Luigi Acerbi<sup>3</sup>, Samuel Kaski<sup>1,2,4</sup>, Tom Rainforth<sup>5</sup>

<sup>1</sup> ELLIS Institute Finland

<sup>2</sup> Department of Computer Science, Aalto University, Finland

<sup>3</sup> Department of Computer Science, University of Helsinki, Finland

<sup>4</sup> Department of Computer Science, University of Manchester, UK

<sup>5</sup> Department of Statistics, University of Oxford, UK

## Abstract

Learning effective policies for adaptive data acquisition remains challenging: posterior-based methods rely on surrogate models and posterior approximations that can be misspecified or biased, while direct policy-learning methods map from historical observations and fail to exploit available model representations, making learning harder. We introduce policy learning with belief representations (POLAR), based on the insight that optimal data acquisition depends on the observation history only through a sufficient belief state. Specifically, POLAR decouples representation learning from policy learning by leveraging pretrained predictive foundation models as belief-state encoders, training a policy head on top of their representations. This yields a simple, unified amortised policy learning framework for Bayesian experimental design, Bayesian optimisation, and active learning, differing only in the task-specific utility used to train the policy. Empirically, we find that POLAR outperforms state-of-the-art amortised methods across diverse tasks while requiring far fewer training samples, demonstrating a significant step in the scalability and efficiency of amortised data acquisition.

## 1 Introduction

Many sequential learning problems can be posed as *adaptive data acquisition*: at each round, a learner chooses which query to make next to improve a downstream decision, based on the data observed so far, receives the resulting observation, and repeats until a budget is exhausted. This abstraction covers Bayesian experimental design (BED; 53), Bayesian optimisation (BO; 20), and active learning (AL; 59). These settings differ in their downstream objectives, but share the same central challenge: *mapping a growing observation history to acquisition decisions under a finite budget*.

Learning effective acquisition policies in this setting is difficult. Existing approaches fall into two families: those that work through task-specific posterior, and those that map observation histories directly to decisions. Both are problematic in different ways. Posterior-based methods fit a probabilistic model from the observed data and use the resulting posterior quantities to evaluate an acquisition function. However, both classical [59, 31, 20] and amortised [9, 36, 41] methods are bottlenecked by the quality of the posterior. In practice, model misspecification or approximation errors can distort the acquisition rule—which typically requires well-calibrated uncertainties—and these errors can compound over the course of a sequential acquisition process [39, 46]. The second family maps observation histories directly to the next decision [14, 38, 32, 45]. These methods avoid explicit posterior inference, but pay for this by having to learn a much more complex mapping from raw histories: they do not use architectures that exploit information in the model and must implicitly learn how the posterior changes with the data. In other words, they must simultaneously learn both how to *represent* the current dataset and how to *act* on that representation, which is often sample-inefficient and difficult to train.

> **Image description.** A schematic diagram in three linked boxes: a left box describing the POLAR
> architecture, and two boxes on the right showing the training stage (top) and deployment stage
> (bottom) as flowcharts with labeled arrows.
>
> **Left box (POLAR architecture).** At the top, a small mountain/compass icon and the label
> "POLAR" sit above a green rounded box "Policy head" (marked with a small network icon), joined by
> a "+" to a row labeled "Belief embeddings": a horizontal strip of small rounded rectangles shading
> from solid green on the left, through paler green, through a "..." gap, to pale green and finally
> an unfilled (white) box on the right. An upward arrow connects this strip to a box below labeled
> "Tabular Foundation Model", which contains three shaded sub-boxes side by side, drawn on a stack
> of cards suggesting repeated blocks: "Sample attn" (three vertical columns of three circles joined
> by curved arrows), "Feature attn" (three horizontal rows of three circles, with curved arrows from
> the outer circles of each row to its middle circle), and "MLP" (a small fully connected network
> icon). Below this, in italics, "Pretrained on diverse
> priors" labels three small icons: a line chart of wavy curves ("Random functions"), a small graph of
> connected nodes ("Random graphs"), and a small tree diagram ("Tree ensembles"), followed by "...".
>
> **Top-right box (Training stage).** A dashed-outline box labeled "POLAR" contains two stacked
> sub-boxes, "Policy head" (green) and "Backbone" (blue). A solid arrow leads from this box to
> "Simulated rollouts", then to "Task-specific utility" (containing three small pill labels "BED",
> "BO", "AL"), then down to "Policy loss". A solid arrow labeled "Update policy head" runs from
> "Policy loss" back to the "Policy head" sub-box. Separately, a "Prediction loss" box connects back
> to the "Backbone" sub-box via a dashed arrow labeled "Update backbone" with "optional" in italics
> beneath it. A darker gray panel encloses the policy loop ("Policy head", "Simulated rollouts",
> "Task-specific utility", "Policy loss"); the "Backbone" and "Prediction loss" row lies below it,
> outside the panel.
>
> **Bottom-right box (Deployment stage).** A dashed-outline box labeled "POLAR" (again containing
> "Policy head" and "Backbone" sub-boxes) has a solid arrow to "Next design" (a small molecule icon),
> then to "Experiment" (a flask icon), then down to "Outcome" (a clipboard-with-checklist icon), then
> left to "History" (a small table icon), which arrows back into the "POLAR" box, closing the loop.

Figure 1: Overview of POLAR. *Left*: POLAR uses a pretrained tabular foundation model as a belief encoder and trains a policy head on top of it. *Top right*: Policy learning is driven by task-specific utilities, while backbone adaptation is supervised by an optional prediction loss. *Bottom right*: At deployment, the policy maps the current history to the next design in a single forward pass.

The key insight that allows us to address these limitations is that, while the optimal sequential decisions depend only on the observation history through the corresponding posterior, a learned policy need not access this posterior explicitly: it may instead act on any representation that preserves the information needed to recover that task-relevant belief. This suggests a middle ground between raw observation histories and explicit posterior estimates. To exploit this insight, we therefore introduce an intermediate *belief representation* that captures underlying features in how the posterior varies with the data, without over-committing to a single posterior approximation.

Pretrained predictive foundation models provide a natural starting point for this belief representation. Namely, rather than learning a belief representation from observation histories from scratch, we can build on models whose pretraining has already shaped them to organise observed data into representations useful for prediction across a broad family of tasks. We instantiate this idea with Tabular Foundation Models (TFMs; 49), such as TabPFN [27, 23] and TabICL [51, 52], which produce candidate-conditioned representations from a context set in a single forward pass.

Specifically, we introduce **PO**licy **L**e**A**rning with Belief **R**epresentations (POLAR), an amortised data acquisition framework that uses the hidden representations from a TFM as our belief representation and trains a policy head to select the next query from the encoding this provides. In doing so, POLAR shifts amortised acquisition from jointly learning representations and decisions to learning how to act on pretrained belief representations. An overview of POLAR is shown in Figure 1.

**Contributions.** **(1)** We give a unified decision-theoretic view of adaptive data acquisition as learning acquisition policies on top of belief representations, rather than learning sequential decision-making directly from raw histories or acting through explicit posterior approximations. **(2)** We introduce a simple decoupled policy architecture that uses pretrained predictive foundation models as belief-state encoders and trains a policy head on top. **(3)** Across diverse adaptive data acquisition tasks, the learned policy substantially outperforms greedy acquisition on the same backbone and exceeds state-of-the-art amortised methods while requiring up to $100\times$ fewer task-specific training samples.

## 2 Preliminaries

**Adaptive data acquisition.** We consider a finite-horizon adaptive data acquisition problem in which a learner repeatedly selects designs based on the data observed so far. A latent world state $z \sim p(z)$ determines observations through a model $p(y \mid x, z)$, where $x \in \mathcal{X}$ is a design and $y \in \mathcal{Y}$ is the corresponding outcome. At round $t$, the learner has observed $\mathcal{D}_{t-1} = \{(x_i, y_i)\}_{i=1}^{t-1}$, chooses a feasible design $x_t \in \mathcal{X}_t(\mathcal{D}_{t-1}) \subseteq \mathcal{X}$, observes $y_t \sim p(\cdot \mid x_t, z)$, and appends $(x_t, y_t)$ to the history. An acquisition policy $\pi = (\pi_1, \ldots, \pi_T)$ is a sequence of decision rules, $\pi_t : \mathcal{H}_{t-1} \to \mathcal{X}$ satisfying $\pi_t(\mathcal{D}_{t-1}) \in \mathcal{X}_t(\mathcal{D}_{t-1})$. Adaptive data acquisition problems share this sequential structure, but differ in the downstream objective and which aspect of the world state matters for this. For example, in many BED problems, we seek information about unknown system parameters $z$ [8, 56, 53]; in active learning, we want to learn to make effective future predictions [62]; and in BO, we seek evaluations that rapidly identify a global optimum [20].

Existing approaches often reduce adaptive acquisition to repeated posterior-based scoring. Given history $\mathcal{D}_{t-1}$, they construct a posterior distribution, either by directly performing inference on the underlying model or using an amortised neural predictor [9, 47, 41], then select candidates with an acquisition function. Examples include Gaussian-process posteriors with Expected Improvement (EI) acquisition in BO [20] and parametric posteriors with Expected Information Gain (EIG) acquisition in BED [43, 44]. These methods rely on the quality of the posterior approximation exposed to the acquisition rule. Outside rare cases where exact inference is tractable, posterior approximation is hard [39, 46], and its errors can propagate into suboptimal acquisitions, especially across multiple rounds.

An alternative is to learn the acquisition policy. In this case, a parameterised policy is trained from simulated trajectories to maximise a task-specific sequential utility, without explicitly constructing a posterior or evaluating a hand-designed acquisition function. For example, amortised BED methods [14, 38, 5, 6, 32, 33, 7], map observation histories directly to experimental designs, and related end-to-end policies have appeared for BO [45, 71] and AL [33, 40]. This amortises test-time decision-making, but requires the same model to learn both a representation of the current dataset and an acquisition strategy from sequential supervision alone.

**Pretrained predictive models.** Recent work on Bayesian in-context learning has produced models that can approximate Bayesian prediction in a single forward pass, without iterative fitting or gradient updates. Transformer Neural Processes [50] and Prior-Fitted Networks [48] learn to map a context set of observed input-output pairs to predictive distributions for new inputs, after offline training on large families of synthetic tasks. More recent Tabular Foundation Models (TFMs), such as TabPFN [27, 23] and TabICL [51, 52], scale this idea to practical tabular prediction, achieving strong performance across a wide range of supervised learning problems. Given a dataset $\mathcal{D}$ and a candidate input $x$, these models produce a predictive distribution $q_\psi(\cdot \mid x, \mathcal{D})$ for the corresponding outcome, together with internal representations $r_\psi(x, \mathcal{D})$ that depend on both $x$ and $\mathcal{D}$. These representations are not designed for any particular sequential task, but can encode predictive uncertainty, local structure, and context-dependent function behaviour. Consequently, TFMs can be viewed as off-the-shelf amortised Bayesian predictors whose representations provide a natural substrate for acquisition policies.

## 3 A Unified Decision-Theoretic View of Adaptive Data Acquisition

To understand policy learning for adaptive data acquisition, we must first formalise what an ideal representation should capture. We do so by casting adaptive data acquisition under a unified Bayesian decision-theoretic framework.

**Bayes-optimal policy.** Consider a finite-horizon sequential data acquisition problem. Once all data has been acquired, a downstream action, $a \in \mathcal{A}$, is chosen, which incurs loss $\ell(a, w)$, where $w = \tau(z)$ represents the aspects of the overall world state that directly impact our loss. Depending on context, $a$ might be an estimate, a predictive distribution, or a physical decision. Following Bayesian decision theory [57, 44, 3, 34] and defining $p(w \mid \mathcal{D}_T)$ as the pushforward of our posterior onto $w$, the Bayes-optimal acquisition policy is now:

$$
\pi^\star = \operatorname*{arg\,min}_{\pi \in \Pi} \mathbb{E}_{p(\mathcal{D}_T;\pi)}\left[\min_{a \in \mathcal{A}} \mathbb{E}_{p(w \mid \mathcal{D}_T)}[\ell(a,w)]\right]. \tag{1}
$$

Here the expected loss of the Bayes action, $\min_{a \in \mathcal{A}} \mathbb{E}_{p(w \mid \mathcal{D}_T)}[\ell(a,w)]$, is a function only of our posterior beliefs for a given loss function. We can thus rewrite Equation (1) as an expected functional of the posterior: defining $s(q,w) := \ell(a^\star(q), w)$ where $a^\star(q) := \arg\min_{a \in \mathcal{A}} \mathbb{E}_{w \sim q}[\ell(a,w)]$, then

$$
\pi^\star = \operatorname*{arg\,min}_{\pi \in \Pi} \mathbb{E}_{p(\mathcal{D}_T,w;\pi)}\left[s\left(p_w(\cdot \mid \mathcal{D}_T), w\right)\right]. \tag{2}
$$

It turns out that this $s$ is always a proper scoring rule for distributions on $w$ [58, 21]. We can thus view Equation (2) as an *expected posterior uncertainty*, $\mathbb{E}_{p(\mathcal{D}_T;\pi)}[h_s[p_w(\cdot \mid \mathcal{D}_T)]]$, where our uncertainty measure is the *generalised entropy* $h_s[q] = \mathbb{E}_{w\sim q}[s(q,w)]$ [13, 4].

Under this unified foundational framework of sequential Bayesian decision-making, we can thus equivalently think about defining adaptive data gathering problems through defining either downstream losses, scoring rules, or uncertainty measures, all of which also (implicitly) define the target variables of interest for our data gathering, $w$. Most existing BED and active learning methods are explicitly based on defining uncertainty measures, which is typically taken to be entropy to yield the expected information gain [43, 53], though they can vary in what variables are targeted for learning, e.g., in AL, we can either take $w$ to be model parameters [29] or downstream predictions [62]. In BO, it is more common to work directly with a downstream loss, with $w$ taken to be the true function values at the queried inputs. For example, we can recover a strategy equivalent to targeting expected improvement by taking $\ell_{\mathrm{EI}}(a,f) = -f(a)$ and then $\mathcal{A}$ as the set of inputs we have queried (such that the minimisation over $a$ trivially yields the point with the highest posterior mean).

**From posterior beliefs to belief representations.** We can further rearrange Equation (2) to make clear the dependency of the optimal policy on historical data as follows (where $\mathcal{D}_{t:T} = \{(x_i,y_i)\}_{i=t}^{T}$)

$$
\pi^\star = \operatorname*{arg\,min}_{\pi \in \Pi} \mathbb{E}_{p(\mathcal{D}_{t-1};\pi)}\left[ \mathbb{E}_{p(w | \mathcal{D}_{t-1})p(\mathcal{D}_{t:T} | w;\pi)}\left[ s\left( \frac{p_w(\cdot | \mathcal{D}_{t-1})p(\mathcal{D}_{t:T} | w;\pi)}{\int p_w(w' | \mathcal{D}_{t-1})p(\mathcal{D}_{t:T} | w';\pi)dw'}, w \right) \right] \right].
$$

This yields a critical insight: The Bayes-optimal policy depends on the history only through the task-relevant posterior induced by that history (and more precisely the expected final posterior uncertainty that it induces). Equivalently, there exist maps $f_{\mathcal{D}}$, $f_{\mathrm{post}}$ such that

$$
\pi^\star(\cdot ; \mathcal{D}_{t-1}) = f_{\mathcal{D}}(\mathcal{D}_{t-1}) = f_{\mathrm{post}}\left(p_w(\cdot \mid \mathcal{D}_{t-1})\right).
\tag{3}
$$

Thus, any two datasets that induce the same task-relevant posterior should lead to the same optimal next action. This invariance suggests that the right input to policy learning is not necessarily the raw dataset itself, but some representation that preserves the information in $\mathcal{D}_{t-1}$ needed to recover the task-relevant belief. In particular, we want a representation that faithfully retains this information, while also providing the simplest possible mapping to the optimal policy decisions.

To this end, we break down $f_{\mathcal{D}} = g \circ \mathrm{enc}$ into a *belief encoder* $\mathrm{enc}$ and a *policy head* $g$. We call $e_t = \mathrm{enc}(\mathcal{D}_{t-1})$ a *belief representation* for $w$ if there exists a decoder $\mathrm{dec} : \mathcal{E} \to \mathcal{P}(\mathcal{W})$ such that

$$
p_w(\cdot \mid \mathcal{D}_{t-1}) = \mathrm{dec}(e_t).
\tag{4}
$$

Whenever the decoding condition above holds, there also exists a map $g$ for which

$$
\pi^\star(\cdot ; \mathcal{D}_{t-1}) = g(e_t),
\tag{5}
$$

because the task-relevant posterior is determined by $e_t$. Two extreme choices of $e_t$ recover the two dominant approaches in the literature:

- **Raw history: $e_t = \mathcal{D}_{t-1}$**. The raw history is trivially a belief representation, since the posterior can be recovered from it. This is the input used by direct policy-learning methods, but it leaves the policy head $g$ with the entire job of capturing the dependency of the posterior on the data, significantly complicating practical training.
- **Posterior: $e_t = p_w(\cdot \mid \mathcal{D}_{t-1})$**. This is the belief state which yields a trivial decoder (the identity function) and thus incorporates all desired posterior invariances (though there may be further invariances in the mapping from intermediary posteriors to expected future uncertainties), thus making the training of $g$ much easier. In practice, however, this belief state cannot usually be calculated exactly and is replaced by an approximation $\hat{p}_w(\cdot \mid \mathcal{D}_{t-1})$, breaking Equation (4) and exposing $g$ to the approximation error of the inference step.

Rather than going with either of these extremes, our key idea is to use an encoder that avoids needing an overly complex $g$ while still preserving the required information in $\mathcal{D}_{t-1}$ that determines optimal actions (and in particular avoiding the errors that build up from direct posterior approximation). We achieve this by using encodings that provide effective *representations* of the posterior, without over-committing to using precise posterior approximation. Section 4 shows how predictive foundation models provide an appropriate and practical encoder as a by-product of their training.

## 4 Policy Learning with Belief Representations

Section 3 argued that acquisition policies should act on representations of the task-relevant posterior: representations that retain the information in observation histories needed for optimal decisions, while structured enough to make the map to acquisitions easy to learn. Pretrained predictive foundation models are natural candidates for this role: to predict well across diverse tasks from in-context observations alone, such models must internally summarise the observed data into representations that track how beliefs change as data accumulates, precisely the information an acquisition policy needs. We now describe how POLAR instantiates this principle using Tabular Foundation Models (TFMs) as practical belief encoders.

**Foundation models as belief encoders.** A pretrained TFM $F_\psi$ is trained to map context sets to predictive distributions across a broad family of synthetic tasks [27, 51, 23]. Given a dataset $\mathcal{D}_{t-1}$ and a candidate $x \in \mathcal{X}$, a forward pass returns (i) a predictive distribution $q_\psi(\cdot \mid x, \mathcal{D}_{t-1})$ over the outcome $y$, and (ii) a candidate-conditioned hidden representation $r_\psi(x,\mathcal{D}_{t-1}) \in \mathbb{R}^{d_r}$ from which this predictive distribution is decoded. The predictive distribution $q_\psi$ is one particular output decoded from $r_\psi$ through a fixed prediction head. It lives in outcome space, approximating a posterior predictive over $y$ at a candidate $x$, whereas the decision-theoretic object in Section 3 is the posterior over the task-relevant quantity $w$. Thus, although $q_\psi$ is useful for prediction and for some task-specific surrogates, it is not in general itself a belief representation for $w$.

We therefore use the hidden representation $r_\psi$, rather than $q_\psi$ alone, as the interface for policy learning, so that the acquisition policy can act on the candidate-conditioned state before it is compressed into the final predictive output. Although predictive pretraining optimises the decoded distribution $q_\psi$, it does so through hidden representations $r_\psi$ that must support prediction across a broad family of synthetic priors and variable-size target sets. This training pressure encourages $r_\psi$ to organise $\mathcal{D}_{t-1}$ along directions that determine how predictions change with the data—predictive uncertainty, local structure, and context-dependent function behaviour—which are precisely the kinds of information an acquisition policy needs in order to decide where to query next. We therefore treat $r_\psi$ as a belief representation in the sense of Section 3. As with any neural encoder, $r_\psi$ is not an exact sufficient statistic for $w$, but predictive pretraining shapes it to retain the information that determines optimal acquisitions, and we validate its adequacy empirically in Section 6.

While the framework of Section 3 applies to general design spaces, TFMs are far more efficient when the context $\mathcal{D}_{t-1}$ is processed once and reused across a finite candidate set. We therefore work in a pool-based setting: at each round, the policy chooses from a feasible pool $\mathcal{C}_t \subseteq \mathcal{C}$. A single batched pass over $\mathcal{C}_t$ returns the candidate-conditioned representations $\{r_\psi(x,\mathcal{D}_{t-1}) : x \in \mathcal{C}_t\}$. This setting matches many practical problems: hyperparameter optimisation over discrete configuration grids, molecular optimisation over finite molecule libraries, and active learning over finite unlabelled pools. For continuous design spaces, the same architecture can be combined with a candidate generator or evaluated over sampled candidate sets.

**Policy head.** We instantiate the policy head $g$ as a function $g_\eta : \mathbb{R}^{d_r} \to \mathbb{R}$ with learnable parameters $\eta$. For each candidate $x \in \mathcal{C}_t$, $g_\eta$ maps $r_\psi(x,\mathcal{D}_{t-1})$ to a scalar logit, and normalising these logits across the candidate pool with a softmax gives the stochastic policy $\pi_\eta(\cdot \mid \mathcal{D}_{t-1}, \mathcal{C}_t)$. We parameterise $\pi_\eta$ as a distribution over candidates during training, which provides exploration and admits standard stochastic-gradient policy training. The Bayes-optimal policy is deterministic at convergence [44], so at deployment, we select the candidate with the largest learned score.

**Task scoring rules.** The architecture is shared across all tasks; only the task-specific scoring rule changes. For each task, we define $\widehat{\Phi}(\mathcal{D}_T, w)$ as a tractable surrogate of $-s(p_w(\cdot | \mathcal{D}_T), w)$. In some tasks, this score is directly observable, while in others a surrogate estimator is needed. In BED, the natural scoring rule is the log loss, whose expected score is the EIG. Since the EIG is intractable, we approximate it with the sPCE lower bound [14]. Other EIG estimators could be substituted without further change. In BO, the queried function values $y_t$ are directly observed, so the score reduces to a tractable function of the trajectory. We use the terminal best observed value, $\widehat{\Phi}_{\mathrm{BO}}(\mathcal{D}_T, w) = \max_{t\leq T} y_t$, for which no posterior approximation is required. In AL, when the goal is to make future predictions [62, 34], the terminal score is the predictive log-density on a target set, which is also intractable under the true posterior, so we use a variational surrogate built on $q_\psi$.

**Policy training.** We train the policy by rolling out $\pi_\eta$ on simulated tasks and applying a score-function policy-gradient objective. Given a task-specific score $\widehat{\Phi}(\mathcal{D}_T,w)$, to provide dense training signals, we define local utility increments $u_t = \widehat{\Phi}(\mathcal{D}_t,w) - \widehat{\Phi}(\mathcal{D}_{t-1},w)$, so that $\sum_{t=1}^T u_t = \widehat{\Phi}(\mathcal{D}_T,w) - \widehat{\Phi}(\mathcal{D}_0,w)$. With $\tau = (x_{1:T},y_{1:T})$ denoting a sampled trajectory, the policy loss is

$$
\mathcal{L}_{\mathrm{pol}}(\eta) = -\mathbb{E}_{\tau\sim \pi_\eta}\left[\sum_{t=1}^T u_t \log \pi_\eta(x_t;\mathcal{D}_{t-1})\right].
\tag{6}
$$

Here, following [45, 33], we use the one-step increment rather than the full reward-to-go. This objective should be understood as a low-variance local-credit surrogate for the terminal-score objective, rather than an unbiased policy-gradient estimator of the full non-myopic objective.

**Backbone adaptation.** The pretrained TFM backbone $F_\psi$ is trained on synthetic priors that may not perfectly match the downstream task distribution. We therefore optionally adapt $\psi$ using an auxiliary supervised prediction loss, $\mathcal{L}_{\mathrm{pred}}(\psi) = \sum_{t=1}^T \sum_{m=1}^M \ell_{\mathrm{pred}}\left(q_\psi(\cdot \mid x_m^\star,\mathcal{D}_{t-1}), y_m^\star\right)$, where $\{(x_m^\star, y_m^\star)\}_{m=1}^M$ are target points sampled from the same simulated task during training. The form of $\mathcal{L}_{\mathrm{pred}}$ depends on the prediction head: for density-based heads [26], it is negative log-likelihood, while for quantile-based heads [52], it is quantile regression loss. A key design choice is that policy gradients are not propagated into the pretrained backbone: $\mathcal{L}_{\mathrm{pol}}$ updates only the policy head, while $\mathcal{L}_{\mathrm{pred}}$ is the only loss that may update $\psi$. This lets the backbone adapt through supervised prediction signals while preventing high-variance policy-gradient updates from destabilising the pretrained model. Thus, representation refinement and decision learning remain decoupled.

**Limitations and possible extensions.** We close this section by noting the scope of the particular realisation of POLAR described above, and the natural extensions it suggests. First, our policy acts over finite candidate pools. This covers the pool-based settings considered in this work, but problems requiring highly precise continuous designs may benefit from continuous-action variants, for example, by optimising the learned score over generated candidates or by introducing a global summary token that feeds a continuous policy head. Second, our policy-gradient objective uses local credit assignment rather than an explicit long-horizon value function. This keeps training simple and low variance, but tasks with strongly delayed information gains may benefit from value-based extensions or reward-to-go estimators. Third, when the downstream task distribution differs substantially from the synthetic priors used to pretrain the backbone, auxiliary prediction losses may be needed to realign $r_\psi$, adding training cost. Parameter-efficient adaptation strategies, such as LoRA-style adapters [30] or partial-layer finetuning, could reduce this overhead.

## 5 Related Work

Amortised policy-based BED was first proposed by DAD [14], which learns a design policy offline and deploys it in a single forward pass, with iDAD [38] extending the framework to implicit likelihood models. A parallel line recasts the problem as reinforcement learning [5, 42], and Huang et al. [32] amortise designs against an explicit downstream decision utility. Hedman et al. [25] introduce a semi-amortised variant that combines an offline policy with online refinement, while Guo et al. [24] consider constraint-aware BED via online planning, and a recent thread improves training signals through alternative EIG estimators [6, 37, 60, 7]. Huang et al. [33] show that prediction-driven supervision can support policy learning, but its shared backbone is trained from scratch under coupled prediction and policy objectives, making it sample-inefficient. Our work is complementary to all of these, rather than designing a new estimator or training algorithm, we reuse a pretrained predictive model as the belief encoder so that a small policy head can be trained with standard policy gradient.

In BO, one line of work retains the classical surrogate-plus-acquisition template while replacing the surrogate with increasingly powerful neural predictors, including PFN- and transformer-based models [47, 9, 41, 35, 67]. A second line instead learns the acquisition policy end-to-end, mapping observed trajectories directly to the next query [10, 11, 45, 63, 71, 72]. POLAR sits between these threads: it inherits the strong representations of a pretrained surrogate, but learns a policy on top of those representations rather than committing to a fixed acquisition rule.

Our work is also related to the broader family of meta-learned predictors that map context sets to predictive distributions. Originating from the Conditional Neural Processes [19], these architectures evolved into Transformer Neural Processes [50] and early PFNs [48]. Recent tabular foundation models such as TabPFN [26, 27, 23] and TabICL [51, 52] scale this paradigm to broad supervised-learning regimes and have stimulated a growing literature [49, 69, 28, 70]. Finally, since our backbone is adapted with an auxiliary prediction loss, our work is also related to recent studies on fine-tuning TFMs to better match downstream data distributions [55, 65, 18].

> **Image description.** Three panels: two line plots of EIG versus training samples for the 2D and
> 5D location-finding tasks (a, b), and a 2×2 grid of scatter plots comparing design trajectories of
> four methods on a shared source configuration (c).
>
> **Panel (a)** ("Location Finding 2D"): x-axis "Training samples" on a log scale with ticks
> $3\times10^4$, $3\times10^5$, $3\times10^6$, $3\times10^7$, $3\times10^8$; y-axis "EIG($\uparrow$)"
> from about 3 to 14. A gray dashed horizontal line labeled "Random" sits at about 8.2. Four
> methods are plotted as connected markers: DAD (magenta pentagons), RL-BOED (green circles),
> ALINE (light-blue triangles), and POLAR (orange stars), per the legend. DAD, RL-BOED and ALINE
> each have four points spanning $3\times10^5$ to $3\times10^8$, rising from roughly 5.7, 3.2 and 3.4
> respectively up to roughly 10.4, 11.8 and 13.5 at the rightmost point (ALINE ends highest of the
> three). POLAR has only three points, at $3\times10^4$, $3\times10^5$ and $3\times10^6$, rising
> steeply from about 9.5 to about 12.6 to about 13.5 and then stopping — already above all other
> curves' final values while using far fewer samples. Vertical error bars (standard error) are drawn
> at each point but are small relative to the marker size.
>
> **Panel (b)** ("Location Finding 5D"): same layout, x-axis ticks $5\times10^4$, $5\times10^5$,
> $5\times10^6$, $5\times10^7$, $5\times10^8$; y-axis "EIG($\uparrow$)" from about 2 to 12. The
> gray dashed "Random" line sits at about 8. DAD, RL-BOED and ALINE again have four points, at the
> tick positions from $5\times10^5$ to $5\times10^8$, rising from about 1.6, 2.0 and 4.7 respectively
> to about 10.1, 9.1 and 12.0. POLAR again has only three points, at $5\times10^4$, $5\times10^5$
> and $5\times10^6$, rising from about 10.5 to 11.1 to about 12.2, again plateauing at a high value
> far earlier on the x-axis than the other methods.
>
> **Panel (c)**: a 2×2 grid of small square scatter plots, titled "POLAR (ours)" (top left), "DAD"
> (top right), "RL-BOED" (bottom left), and "ALINE" (bottom right), sharing axes "Coordinate 1"
> (x, 0 to 1, bottom row only) and "Coordinate 2" (y, 0 to 1, left column only). Each panel shows
> two orange star markers ("True $\theta$", per the legend in the top-left panel) marking two fixed
> source locations, and a set of circular markers ("Design") colored on a blue sequential scale from
> pale blue to dark navy according to a shared colorbar on the right labeled "Acquisition step"
> (running from about 5 to 30). In the POLAR panel, the design markers cluster tightly around the
> two star locations, with the darkest (latest-step) points sitting essentially on top of the stars. In
> the DAD panel, design markers are spread broadly across the unit square with little visible
> clustering near the stars. In the RL-BOED panel, markers gather near both stars, but several dark
> (late-step) points also sit between them (around (0.45, 0.5)) and a few lighter points are scattered
> away from both stars. In the ALINE panel, markers cluster around both stars similarly to POLAR,
> with several lighter points scattered away from the clusters.

Figure 2: Location finding. (a) EIG against total training samples in the 2D setting. Error bars denote standard error across 1000 runs. (b) The same comparison in the 5D setting. (c) Example design trajectories for all methods on a shared latent source configuration.

## 6 Experiments

We now empirically evaluate POLAR across a range of tasks. Section 6.1 benchmarks POLAR on two standard BED tasks, location finding and constant elasticity of substitution, where we also report a series of ablations isolating the contribution of each component of our method. Section 6.2 evaluates POLAR for BO on a hyperparameter optimisation benchmark, and Section 6.3 turns to a real-world high-dimensional molecular optimisation task. We additionally study loss-driven active learning in Appendix C, which demonstrates that POLAR extends naturally to settings defined by user-specified predictive losses. Unless otherwise stated, all experiments use TabICL v2 [52] as the backbone; additional results with TabPFN v2.5 [23] are reported in Appendix D.

### 6.1 Benchmarking on Bayesian experimental design tasks

We begin with two standard BED benchmarks, location finding and constant elasticity of substitution (CES), which have been used extensively in prior work [15, 16, 5, 33]. The goal of location finding [61] is to infer the locations of $K$ hidden sources in $d_x$-dimensional space from noisy distance measurements. We consider two variants to test scalability: a standard 2D setting ($d_x = 2, K = 2, T = 30$) and a higher-dimensional 5D setting ($d_x = 5, K = 2, T = 50$). CES [2] involves eliciting the parameters of an economic utility function from a series of pairwise preference queries. We compare against Random, DAD [14], RL-BOED [5], and ALINE [33]. For both tasks, we report the EIG lower bound estimated by sPCE, and the upper bound estimated by sNMC [14]. Full details on the task and training setups are provided in Appendix B.1.

**Main results.** To explicitly assess the training efficiency, we report performance as a function of the total number of training samples (see Appendix B.1 for details on how we calculate this metric for a fair comparison). For location finding, Figures 2a and 2b show that POLAR is dramatically more sample-efficient than prior amortised BED policies. In the standard 2D task, our method slightly outperforms ALINE, the current state-of-the-art, using approximately $100\times$ fewer training samples, and surpasses the best reported results of both RL-BOED and DAD using $1{,}000\times$ fewer samples. This dramatic efficiency gap further widens in the more complex 5D environment, where our method eclipses the peak performance of RL-BOED and DAD with $10{,}000\times$ fewer samples, and exceeds ALINE with $100\times$ fewer samples. This sample efficiency is particularly valuable in settings where simulation or real data collection is expensive, as is typical in scientific applications of experimental design. Figure 2c provides visualisations of design trajectories. The sNMC upper bound results are provided in Appendix D.1. The conclusion remains consistent for the CES task. Table 1 shows that POLAR outperforms all amortised baselines at matched training-sample budgets. Results at full convergence for each method are reported in Appendix D.2, where POLAR also retains its lead. Owing to the cost of evaluating a foundation-model backbone, our method is slightly slower to deploy than the other policies, but the absolute overhead remains negligible in practice: at roughly 0.03 seconds per decision, the latency is still effectively imperceptible at the human timescale.

Table 1: CES. Comparison of EIG estimates and deployment efficiency on the CES task. Results are reported as mean $\pm$ s.e. over 1,000 independent runs.

| Methods | $10^5$ training samples: sPCE | $10^5$ training samples: sNMC | $10^6$ training samples: sPCE | $10^6$ training samples: sNMC | Deployment time (s) |
| --- | --- | --- | --- | --- | --- |
| DAD | $9.54 \pm 0.14$ | $9.66 \pm 0.14$ | $11.46 \pm 0.16$ | $12.33 \pm 0.19$ | $0.0003 \pm 0.00$ |
| RL-BOED | $11.04 \pm 0.11$ | $11.21 \pm 0.12$ | $12.32 \pm 0.13$ | $14.73 \pm 1.25$ | $0.0005 \pm 0.00$ |
| ALINE | $8.10 \pm 0.18$ | $9.86 \pm 0.29$ | $9.11 \pm 0.17$ | $11.64 \pm 0.33$ | $0.004 \pm 0.00$ |
| POLAR | $\mathbf{11.50} \pm 0.15$ | $\mathbf{12.87} \pm 0.20$ | $\mathbf{13.06} \pm 0.12$ | $\mathbf{15.53} \pm 0.22$ | $0.03 \pm 0.00$ |

To better understand where these gains come from, we conduct three ablations on the 2D location finding task, all reported in Figure 3. Additional ablations on the choice of pretrained backbone and the representation layer used as the belief state are reported in Figure A3, which shows POLAR is not tied to a single backbone, and later backbone layers provide more useful belief states for acquisition.

**Frozen vs. finetuned backbone.** We first compare our default setting, in which the backbone is adapted by the supervised prediction loss, against a frozen-backbone variant. Even without any weight updates, the frozen variant remains competitive with the peak performance of RL-BOED. However, it clearly falls short of our finetuned variant. Because the backbone is a general-purpose foundation model pretrained on a vast array of synthetic distributions, its representations are not tailored to a specific task. The auxiliary prediction loss acts as an alignment signal, finetuning the backbone to adapt its predictive representations to the target domain, which in turn provides a better belief state for the policy head to act upon.

**Decoupled vs. coupled policy gradients.** Our default method updates the backbone only through the prediction loss. We compare this against a coupled variant in which policy gradients are also allowed to update the backbone. The coupled variant severely degrades performance. Once policy gradients are allowed to modify the backbone, the same parameters must simultaneously satisfy two rather different objectives: a dense, relatively stable prediction objective and a sparse, high-variance reinforcement-learning objective. In practice, this appears to destabilise the learning and weaken the benefits of pretraining. By decoupling the gradients, we effectively assign the optimal learning objective to each component: low-variance supervised learning for belief state estimation, and policy gradients for decision-making on a stable representation manifold.

> **Image description.** A single grouped bar chart titled "Ablation Study". The x-axis "Training samples" has two
> groups, $3\times10^5$ and $3\times10^6$; the y-axis "EIG($\uparrow$)" ranges from 0 to about 13,
> with gridlines every 2.5 units.
>
> Each group contains four adjacent bars, in the same left-to-right order, colored from light to dark
> per the legend: "POLAR" (pale cream), "Frozen backbone" (orange), "From scratch" (darker orange),
> and "Coupled grads" (dark brown). Each bar carries a small vertical error-bar cap at its top.
>
> In the $3\times10^5$ group, the bars stand at roughly 12.7 (POLAR), 10.9 (Frozen backbone), 8.5
> (From scratch), and 5.2 (Coupled grads) — POLAR tallest, Coupled grads shortest. In the
> $3\times10^6$ group, the bars stand at roughly 13.4 (POLAR), 11.8 (Frozen backbone), 8.2 (From
> scratch), and 9.3 (Coupled grads); here Coupled grads is slightly taller than From scratch but still
> well below POLAR and Frozen backbone. POLAR is the tallest bar in both groups, and the ordering
> POLAR > Frozen backbone > {From scratch, Coupled grads} holds throughout.

Figure 3: Ablations in the 2D setting, isolating the impact of our core architectural choices, including gradient decoupling, backbone finetuning and initialisation.

**Pretrained backbone vs. training from scratch.** We next compare our method against the same architecture trained from scratch, without loading the pretrained checkpoint. The difference is substantial: training from scratch converges much more slowly and remains clearly worse within the same sample budget. This supports a central message of the paper: learning underlying belief representations is a critical part of the overall policy training for amortized adaptive design methods. A pretrained foundation model has already significantly alleviated this problem, and starting from it confers the bulk of our sample efficiency.

### 6.2 Hyperparameter optimisation benchmarks

We next evaluate our method on BO using HPO-B [1], a large-scale hyperparameter optimisation benchmark containing the evaluations of a wide range of machine learning models across thousands of hyperparameter configurations and datasets. HPO-B is pre-partitioned into multiple search spaces, each corresponding to a model family with its own hyperparameter parameterisation. Following prior work [45, 41], we report results on six search spaces, spanning input dimensionalities from 2 to 16 and a representative range of model families. For each search space, we train our policy on the provided meta-training tasks and evaluate on the fixed test tasks and seeds defined by the benchmark.

> **Image description.** Two side-by-side line plots sharing an x-axis ("Acquisition step", 0 to 50) and a common legend below, comparing seven methods (Random, GP, Meta-GP, PFNs4BO, NAP, TabICL, POLAR), each drawn in a distinct color and marker with a shaded one-standard-error band around its curve.
>
> **Left panel (Average Regret):** the y-axis is log-scaled, spanning roughly $8\times10^{-3}$ to $3\times10^{-1}$, with labeled gridlines at $10^{-1}$ and $10^{-2}$. All seven curves start together near 0.2–0.3 at step 0 and decrease monotonically thereafter. POLAR (orange star) separates from the rest within the first ~10 steps and falls fastest, reaching about $1.2\times10^{-2}$ by step 50, with a visibly wide shaded band. NAP (green square) is the closest competitor, ending near $2\times10^{-2}$. PFNs4BO (pink diamond) and Meta-GP (light blue plus) trail close together around $3$–$4\times10^{-2}$. TabICL (gold X) ends just above them. GP (dark blue circle) and Random (gray triangle) decline the least, both leveling off near $6$–$7\times10^{-2}$.
>
> **Right panel (Average Rank):** the y-axis spans roughly 2.4 to 5.7, ticked at 3, 4, and 5 (lower is better). All curves start at rank 4 at step 0. Random rises quickly to the worst rank (~5) and stays there for the rest of the trajectory. GP climbs to ~4.7–4.9. TabICL rises to ~4.0–4.3. Meta-GP stays roughly flat near 3.9. NAP and PFNs4BO settle close together around 3.4–3.7. POLAR drops sharply to its best rank (~2.8) by step 10–15, then rises gradually to about 3.1–3.2 by step 50, remaining the lowest (best) rank throughout the trajectory.

Figure 4: Hyperparameter optimisation on HPO-B. Average regret (left) and average rank (right) aggregated across the six search spaces. Shaded regions denote one standard error across the test tasks. POLAR achieves the lowest regret and the best average rank throughout the acquisition trajectory.

We compare against baselines spanning both classical and amortised BO. RANDOM samples configurations uniformly from the candidate pool. GP fits a Gaussian process [54] directly on the test dataset. META-GP, following Maraval et al. [45], pretrains the kernel hyperparameters on the meta-training datasets and initialises the GP with these parameters at test time. PFNS4BO [47] uses a prior-fitted network [48] as a surrogate. NAP [45] is an end-to-end fully amortised method based on transformer neural processes [50] that maps observation histories directly to next queries. Finally, TABICL uses the same finetuned TabICL backbone as POLAR, but pairs it with a hand-designed acquisition function rather than a learned policy head. All surrogate-based baselines use EI as the acquisition strategy. Additional details on baseline configurations and the dataset are provided in Appendix B.2.

**Results.** Figure 4 shows the average normalised regret and average rank aggregated across all search spaces. Our method achieves the lowest regret and the best average rank throughout the acquisition trajectory, with the gap to the strongest baselines opening within the first ten queries and persisting through the full budget. Notably, compared with NAP, our method shows that using pretrained TFMs can yield stronger performance without requiring the policy to learn its own belief representation from scratch. Additional per-search-space regret and rank curves are reported in Appendix D.3.

### 6.3 Molecular docking optimisation

Finally, we evaluate our method on a real-world high-dimensional molecular optimisation benchmark, DOCKSTRING [17], which provides docking scores for over 260,000 molecules against a panel of protein targets. Since the physicochemical interactions that govern binding affinity are inherently correlated across these structurally related targets, this dataset serves as a natural testbed for meta BO.

We formulate the task as identifying the molecule that minimises the docking score for a given protein target. We train our policy on a meta-training set of 8 protein targets and evaluate its zero-shot transfer performance on a held-out test set of 6 distinct targets; full target lists are provided in Appendix B.3. Each molecule is represented by a 512-bit Morgan fingerprint computed from its SMILES string.

> **Image description.** A single line plot of average regret versus acquisition step for five methods on the DOCKSTRING benchmark, with a legend inside the upper-right corner of the plot.
>
> The x-axis is "Acquisition step", ticked at 0, 20, and 40 (extending to about 50). The y-axis is "Average Regret", ticked from 0.5 to 3.0 in steps of 0.5. Five curves are shown, each with a shaded mean ± s.e. band: Random (gray triangle), GP (Tanimoto) (dark blue circle), Random Forest (green circle), TabICL (gold X), and POLAR (ours) (orange star). All five start together near 2.55–2.6 at step 0. POLAR (ours) decreases fastest and separates from the others within the first ~10 steps, reaching about 0.75 by step 50. TabICL declines next-fastest, leveling off around 1.0. GP (Tanimoto) and Random Forest fall more slowly: Random Forest drops faster early on (about 1.85 at step 10), while GP (Tanimoto) initially tracks Random; the two converge after step ~28 and end near 1.2–1.3, with Random Forest slightly below GP (Tanimoto) at the final point. Random declines the least, ending around 1.6.

Figure 5: DOCKSTRING. Average regret (mean $\pm$ s.e.) across the six held-out targets.

Operating in a 512-dimensional space presents significant challenges for traditional BO. Standard GPs equipped with continuous kernels (e.g., RBF) notoriously struggle in high-dimensional, discrete spaces. Instead, we equip the GP with a Tanimoto kernel, which measures set overlap between binary fingerprints and is the standard choice for fingerprint-based molecular BO [22]. We additionally compare against a Random Forest surrogate, a standard baseline in cheminformatics that is commonly paired with molecular fingerprints [64, 68]. For the amortised surrogate baseline, we use exactly the same finetuned TabICL backbone as in our method.

**Results.** Figure 5 reports average regret across the six test targets. POLAR achieves the lowest regret throughout the trajectory. Both POLAR and TabICL substantially outperform the non-amortised GP and Random Forest baselines, indicating that meta-training across targets allows the backbone to capture shared structure. The further gap between POLAR and TabICL confirms that learning the acquisition rule directly from the belief representation outperforms a hand-designed acquisition.

## 7 Conclusions

We introduced POLAR, an amortised framework for adaptive data acquisition based on a decision-theoretic view: policies need not act directly on raw histories or explicit posterior approximations, but can instead act on task-relevant belief representations. By instantiating these representations with pretrained predictive foundation models, POLAR decouples belief formation from decision-making and trains an acquisition policy on them. Across tasks, POLAR surpasses classical non-amortised methods and state-of-the-art amortised baselines while requiring up to $100\times$ fewer training samples.

---

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

# Efficient Adaptive Data Acquisition via Pretrained Belief Representations - Backmatter

---

## Acknowledgments and Disclosure of Funding

DH, LA, and SK were supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI, 359207). CH was supported by Business Finland (VirtualLab4Pharma, grant agreement 3597/31/2023) and the European Union (Horizon Europe, grant agreement 101214398, ELLIOT). LA was also supported by Research Council of Finland grants 358980 and 356498. SK was also supported by the UKRI Turing AI World-Leading Researcher Fellowship, [EP/W002973/1]. ZH is supported by the EPSRC CDT in Statistics and Machine Learning (EP/Y034813/1). TR is supported by the EPSRC grant EP/Y037200/1.

We acknowledge CSC – IT Center for Science, Finland, for computational resources provided by the LUMI supercomputer, owned by the EuroHPC Joint Undertaking and hosted by CSC and the LUMI consortium (LUMI projects 462000943 and 462000874). Access was provided through the Finnish LUMI-OKM allocation. We also acknowledge the computational resources provided by the Aalto Science-IT Project from Computer Science IT.

## References

- [1] Arango, S. P., Jomaa, H. S., Wistuba, M., and Grabocka, J. (2021). Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.
- [2] Arrow, K. J., Chenery, H. B., Minhas, B. S., and Solow, R. M. (1961). Capital-labor substitution and economic efficiency. The review of Economics and Statistics, pages 225–250.
- [3] Berger, J. O. (1985). Statistical decision theory and bayesian analysis. Springer Series in Statistics.
- [4] Bickford Smith, F., Kossen, J., Trollope, E., Van Der Wilk, M., Foster, A., and Rainforth, T. (2025). Rethinking aleatoric and epistemic uncertainty. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 4345–4359. PMLR.
- [5] Blau, T., Bonilla, E. V., Chades, I., and Dezfouli, A. (2022). Optimizing sequential experimental design with deep reinforcement learning. In International conference on machine learning, pages 2107–2128. PMLR.
- [6] Blau, T., Chades, I., Dezfouli, A., Steinberg, D., and Bonilla, E. V. (2023). Statistically efficient bayesian sequential experiment design via reinforcement learning with cross-entropy estimators. arXiv preprint arXiv:2305.18435.
- [7] Bracher, N., Kühmichel, L., Ivanova, D. R., Intes, X., Bürkner, P.-C., and Radev, S. T. (2025). Jadai: Jointly amortizing adaptive design and bayesian inference. arXiv preprint arXiv:2512.22999.
- [8] Chaloner, K. and Verdinelli, I. (1995). Bayesian experimental design: A review. Statistical science, pages 273–304.
- [9] Chang, P. E., Loka, N. R. B. S., Huang, D., Remes, U., Kaski, S., and Acerbi, L. (2025). Amortized probabilistic conditioning for optimization, simulation and inference. In International Conference on Artificial Intelligence and Statistics, pages 703–711. PMLR.
- [10] Chen, Y., Hoffman, M. W., Colmenarejo, S. G., Denil, M., Lillicrap, T. P., Botvinick, M., and Freitas, N. (2017). Learning to learn without gradient descent by gradient descent. In International Conference on Machine Learning, pages 748–756. PMLR.
- [11] Chen, Y., Song, X., Lee, C., Wang, Z., Zhang, R., Dohan, D., Kawakami, K., Kochanski, G., Doucet, A., Ranzato, M., et al. (2022). Towards learning universal hyperparameter optimizers with transformers. Advances in Neural Information Processing Systems, 35:32053–32068.
- [12] Cowen-Rivers, A. I., Lyu, W., Tutunov, R., Wang, Z., Grosnit, A., Griffiths, R. R., Maraval, A. M., Jianye, H., Wang, J., Peters, J., et al. (2022). Hebo: Pushing the limits of sample-efficient hyper-parameter optimisation. Journal of Artificial Intelligence Research, 74:1269–1349.
- [13] Dawid, A. P. (1998). Coherent measures of discrepancy, uncertainty and dependence, with applications to bayesian predictive experimental design. Department of Statistical Science, University College London. http://www. ucl. ac. uk/Stats/research/abs94. html, Tech. Rep, 139.
- [14] Foster, A., Ivanova, D. R., Malik, I., and Rainforth, T. (2021). Deep adaptive design: Amortizing sequential bayesian experimental design. In International conference on machine learning, pages 3384–3395. PMLR.
- [15] Foster, A., Jankowiak, M., Bingham, E., Horsfall, P., Teh, Y. W., Rainforth, T., and Goodman, N. (2019). Variational bayesian optimal experimental design. In Advances in Neural Information Processing Systems, volume 32.
- [16] Foster, A., Jankowiak, M., O’Meara, M., Teh, Y. W., and Rainforth, T. (2020). A unified stochastic gradient approach to designing bayesian-optimal experiments. In International Conference on Artificial Intelligence and Statistics, pages 2959–2969. PMLR.
- [17] García-Ortegón, M., Simm, G. N., Tripp, A. J., Hernández-Lobato, J. M., Bender, A., and Bacallado, S. (2022). Dockstring: easy molecular docking yields better benchmarks for ligand design. Journal of chemical information and modeling, 62(15):3486–3502.
- [18] Garg, A., Ali, M., Hollmann, N., Purucker, L., Müller, S., and Hutter, F. (2025). Real-tabpfn: Improving tabular foundation models via continued pre-training with real-world data. arXiv preprint arXiv:2507.03971.
- [19] Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., and Eslami, S. A. (2018). Conditional neural processes. In International conference on machine learning, pages 1704–1713. PMLR.
- [20] Garnett, R. (2023). Bayesian optimization. Cambridge University Press.
- [21] Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378.
- [22] Griffiths, R.-R., Klarner, L., Moss, H., Ravuri, A., Truong, S., Du, Y., Stanton, S., Tom, G., Rankovic, B., Jamasb, A., et al. (2023). Gauche: a library for gaussian processes in chemistry. Advances in Neural Information Processing Systems, 36:76923–76946.
- [23] Grinsztajn, L., Flöge, K., Key, O., Birkel, F., Jund, P., Roof, B., Jäger, B., Safaric, D., Alessi, S., Hayler, A., et al. (2025). Tabpfn-2.5: Advancing the state of the art in tabular foundation models. arXiv preprint arXiv:2511.08667.
- [24] Guo, Y., Huang, D., Zhang, X., Katt, S., Kaski, S., and Bharti, A. (2026). Constrained bayesian experimental design via online planning. arXiv preprint arXiv:2605.26990.
- [25] Hedman, M., Ivanova, D. R., Guan, C., and Rainforth, T. (2025). Step-dad: Semi-amortized policy-based bayesian experimental design. In International Conference on Machine Learning, pages 22904–22923. PMLR.
- [26] Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2023). Tabpfn: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations.
- [27] Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326.
- [28] Hoo, S. B., Müller, S., Salinas, D., and Hutter, F. (2024). The tabular foundation model tabpfn outperforms specialized time series forecasting models based on simple features. In NeurIPS workshop on time series in the age of large models.
- [29] Houlsby, N., Huszár, F., Ghahramani, Z., and Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745.
- [30] Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. (2022). Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.
- [31] Huan, X. and Marzouk, Y. M. (2016). Sequential bayesian optimal experimental design via approximate dynamic programming. arXiv preprint arXiv:1604.08320.
- [32] Huang, D., Guo, Y., Acerbi, L., and Kaski, S. (2024). Amortized bayesian experimental design for decision-making. Advances in Neural Information Processing Systems, 37:109460–109486.
- [33] Huang, D., Wen, X., Bharti, A., Kaski, S., and Acerbi, L. (2026a). Aline: Joint amortization for bayesian inference and active data acquisition. Advances in Neural Information Processing Systems, 38:54068–54102.
- [34] Huang, Z., Smith, F. B., and Rainforth, T. (2026b). Loss-driven bayesian active learning. In The 29th International Conference on Artificial Intelligence and Statistics.
- [35] Hung, Y. H., Lin, K.-J., Lin, Y.-H., Wang, C.-Y., Sun, C., and Hsieh, P.-C. (2025). Boformer: Learning to solve multi-objective bayesian optimization via non-markovian rl. In The Thirteenth International Conference on Learning Representations.
- [36] Igoe, C. (2025). Efficient Bayesian Experimental Design with Deep Learning. PhD thesis, Carnegie Mellon University.
- [37] Iqbal, S., Corenflos, A., Särkkä, S., and Abdulsamad, H. (2024). Nesting particle filters for experimental design in dynamical systems. In International Conference on Machine Learning, pages 21047–21068. PMLR.
- [38] Ivanova, D. R., Foster, A., Kleinegesse, S., Gutmann, M. U., and Rainforth, T. (2021). Implicit deep adaptive design: Policy-based experimental design without likelihoods. Advances in neural information processing systems, 34:25785–25798.
- [39] Lacoste-Julien, S., Huszár, F., and Ghahramani, Z. (2011). Approximate inference for the loss-calibrated bayesian. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 416–424. JMLR Workshop and Conference Proceedings.
- [40] Li, C.-Y., Toussaint, M., Rakitsch, B., and Zimmer, C. (2025a). Amortized safe active learning for real-time data acquisition: Pretrained neural policies from simulated nonparametric functions. arXiv preprint arXiv:2501.15458.
- [41] Li, D., Cho, K., and Liu, C. (2025b). None to optima in few shots: Bayesian optimization with mdp priors. arXiv preprint arXiv:2511.01006.
- [42] Lim, V., Novoseller, E., Ichnowski, J., Huang, H., and Goldberg, K. (2022). Policy-based bayesian experimental design for non-differentiable implicit models. arXiv preprint arXiv:2203.04272.
- [43] Lindley, D. V. (1956). On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, 27(4):986–1005.
- [44] Lindley, D. V. (1972). Bayesian statistics: A review. SIAM.
- [45] Maraval, A., Zimmer, M., Grosnit, A., and Bou Ammar, H. (2023). End-to-end meta-bayesian optimisation with transformer neural processes. Advances in Neural Information Processing Systems, 36:11246–11260.
- [46] Maus, N., Kim, K., Pleiss, G., Eriksson, D., Cunningham, J. P., and Gardner, J. R. (2024). Approximation-aware bayesian optimization. Advances in Neural Information Processing Systems, 37:21114–21140.
- [47] Müller, S., Feurer, M., Hollmann, N., and Hutter, F. (2023). Pfns4bo: In-context learning for bayesian optimization. In International Conference on Machine Learning, pages 25444–25470. PMLR.
- [48] Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., and Hutter, F. (2021). Transformers can do bayesian inference. arXiv preprint arXiv:2112.10510.
- [49] Müller, S., Reuter, A., Hollmann, N., Rügamer, D., and Hutter, F. (2025). Position: The future of bayesian prediction is prior-fitted. In International Conference on Machine Learning, pages 81861–81875. PMLR.
- [50] Nguyen, T. and Grover, A. (2022). Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In International Conference on Machine Learning, pages 16569–16594. PMLR.
- [51] Qu, J., Holzmüller, D., Varoquaux, G., and Morvan, M. L. (2025). Tabicl: A tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564.
- [52] Qu, J., Holzmüller, D., Varoquaux, G., and Morvan, M. L. (2026). Tabiclv2: A better, faster, scalable, and open tabular foundation model. arXiv preprint arXiv:2602.11139.
- [53] Rainforth, T., Foster, A., Ivanova, D. R., and Bickford Smith, F. (2024). Modern bayesian experimental design. Statistical Science, 39(1):100–114.
- [54] Rasmussen, C. E. (2003). Gaussian processes in machine learning. In Summer school on machine learning, pages 63–71. Springer.
- [55] Rubachev, I., Kotelnikov, A., Kartashev, N., and Babenko, A. (2025). On finetuning tabular foundation models. arXiv preprint arXiv:2506.08982.
- [56] Ryan, E. G., Drovandi, C. C., McGree, J. M., and Pettitt, A. N. (2016). A review of modern computational algorithms for bayesian optimal design. International Statistical Review, 84(1):128–154.
- [57] Savage, L. J. (1951). The theory of statistical decision. Journal of the American Statistical association, 46(253):55–67.
- [58] Savage, L. J. (1971). Elicitation of personal probabilities and expectations. Journal of the American Statistical Association, 66(336):783–801.
- [59] Settles, B. (2012). Active learning. Morgan & Claypool Publishers.
- [60] Shen, W., Dong, J., and Huan, X. (2025). Variational sequential optimal experimental design using reinforcement learning. Computer Methods in Applied Mechanics and Engineering, 444:118068.
- [61] Sheng, X. and Hu, Y.-H. (2005). Maximum likelihood multiple-source localization using acoustic energy measurements with wireless sensor networks. IEEE transactions on signal processing, 53(1):44–53.
- [62] Smith, F. B., Kirsch, A., Farquhar, S., Gal, Y., Foster, A., and Rainforth, T. (2023). Prediction-oriented bayesian active learning. In International conference on artificial intelligence and statistics, pages 7331–7348. PMLR.
- [63] Song, L., Gao, C., Xue, K., Wu, C., Li, D., Hao, J., Zhang, Z., and Qian, C. (2024). Reinforced in-context black-box optimization. arXiv preprint arXiv:2402.17423.
- [64] Svetnik, V., Liaw, A., Tong, C., Culberson, J. C., Sheridan, R. P., and Feuston, B. P. (2003). Random forest: a classification and regression tool for compound classification and qsar modeling. Journal of chemical information and computer sciences, 43(6):1947–1958.
- [65] Tanna, A., Seth, P., Bouadi, M., Avaiya, U., and Sankarapu, V. K. (2025). Tabtune: A unified library for inference and fine-tuning tabular foundation models. arXiv preprint arXiv:2511.02802.
- [66] Trott, O. and Olson, A. J. (2010). Autodock vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. Journal of computational chemistry, 31(2):455–461.
- [67] Viering, T. J., Adriaensen, S., Rakotoarison, H., Müller, S., Hvarfner, C., Hutter, F., and Bakshy, E. (2025). α-pfn: In-context learning entropy search. In Frontiers in Probabilistic Inference: Learning meets Sampling.
- [68] Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., et al. (2019). Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling, 59(8):3370–3388.
- [69] Ye, H.-J., Liu, S.-Y., and Chao, W.-L. (2025). A closer look at tabpfn v2: Understanding its strengths and extending its capabilities. arXiv preprint arXiv:2502.17361.
- [70] Zhang, Q., Tan, Y. S., Tian, Q., and Li, P. (2025a). Tabpfn: One model to rule them all? arXiv preprint arXiv:2505.20003.
- [71] Zhang, X., Hassan, C., Martinelli, J., Huang, D., and Kaski, S. (2026). In-context multi-objective optimization. In The Fourteenth International Conference on Learning Representations.
- [72] Zhang, X., Huang, D., Kaski, S., and Martinelli, J. (2025b). Pabbo: Preferential amortized black-box optimization. In The Thirteenth International Conference on Learning Representations.

---

*Transcribed from the PDF text layer and corrected with LLMs; text, equations, tables, and figure descriptions may contain mistakes.*
