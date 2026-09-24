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

*Transcribed from the PDF text layer and corrected with LLMs; text, equations, tables, and figure descriptions may contain mistakes.*
