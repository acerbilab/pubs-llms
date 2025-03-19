```
@article{norton2019human,
    author={Norton, Elyse H. and Acerbi, Luigi and Ma, Wei Ji and Landy, Michael S.},
    title={Human online adaptation to changes in prior probability},
    journal={PLOS Computational Biology},
    year={2019},
    month={07},
    volume={15},
    pages={1-26},
    number={7},
    publisher={Public Library of Science},
    doi={10.1371/journal.pcbi.1006681}
}
```

---

#### Page 1

# Human online adaptation to changes in prior probability

Elyse H. Norton ${ }^{1}$, Luigi Acerbi ${ }^{1,2 *}$, Wei Ji Ma ${ }^{1,2}$, Michael S. Landy ${ }^{1,2 *}$

#### Abstract

1 Psychology Department, New York University, New York, New York, United States of America, 2 Center for Neural Science, New York University, New York, New York, United States of America

«Current Address: Département des neurosciences fondamentales, Université de Genève, CMU, Genève, Switzerland

- landy @nyu.edu

#### Abstract

Optimal sensory decision-making requires the combination of uncertain sensory signals with prior expectations. The effect of prior probability is often described as a shift in the decision criterion. Can observers track sudden changes in probability? To answer this question, we used a change-point detection paradigm that is frequently used to examine behavior in changing environments. In a pair of orientation-categorization tasks, we investigated the effects of changing probabilities on decision-making. In both tasks, category probability was updated using a sample-and-hold procedure: probability was held constant for a period of time before jumping to another probability state that was randomly selected from a predetermined set of probability states. We developed an ideal Bayesian change-point detection model in which the observer marginalizes over both the current run length (i.e., time since last change) and the current category probability. We compared this model to various alternative models that correspond to different strategies-from approximately Bayesian to simple heuristics-that the observers may have adopted to update their beliefs about probabilities. While a number of models provided decent fits to the data, model comparison favored a model in which probability is estimated following an exponential averaging model with a bias towards equal priors, consistent with a conservative bias, and a flexible variant of the Bayesian change-point detection model with incorrect beliefs. We interpret the former as a simpler, more biologically plausible explanation suggesting that the mechanism underlying change of decision criterion is a combination of on-line estimation of prior probability and a stable, long-term equal-probability prior, thus operating at two very different timescales.

## Author summary

We demonstrate how people learn and adapt to changes to the probability of occurrence of one of two categories on decision-making under uncertainty. The study combined psychophysical behavioral tasks with computational modeling. We used two behavioral tasks: a typical forced-choice categorization task as well as one in which the observer specified the decision criterion to use on each trial before the stimulus was displayed. We formulated an ideal Bayesian change-point detection model and compared it to several

---

#### Page 2

alternative models. We found that the data are explained best by a model that estimates category probability based on recently observed exemplars with a bias towards equal probability. Our results suggest that the brain takes multiple relevant time scales into account when setting category expectations.

# Introduction

Sensory decision-making involves making decisions under uncertainty. Furthermore, optimal sensory decision-making requires the combination of uncertain sensory signals with prior expectations. Perceptual models of decision-making often incorporate prior expectations to describe human behavior. In Bayesian models, priors are combined with likelihoods to compute a posterior [1]. In signal detection theory, the effect of unequal probabilities (signal present vs. absent) is a shift of the decision criterion [2].

The effects of prior probability on the decision criterion have been observed in detection [2-4], line tilt [5], numerosity estimation [6, 7], recognition memory [8], and perceptual categorization [9] tasks, among others. These studies generally use explicit priors, assume a fixed effect, and treat learning as additional noise. However, there are many everyday tasks in which the probability of a set of alternatives needs to be assessed based on one's past experience with the outcomes of the task. The importance of experience has been demonstrated in studies examining differences between experience-based and description-based decisions [10, 11] and in perceptual-categorization tasks with unequal probability, in which response feedback leads to performance that is closer to optimal than observational feedback [12, 13]. While these studies demonstrate the importance of experience on decision-making behavior, they do not describe how experience influences expectation formation. The influence of experience on the formation of expectations has been demonstrated for learning stimulus mean [14-17], variance [14, 18], higher-order statistics [19], likelihood distributions [20], the rate of change of the environment [15-17, 21-23], and prior probability [24, 25]. Here, we add to previous work by investigating how one's previous experience influences probability learning in a changing environment.

In the previous work on probability learning by Behrens et al. [24], participants tracked the probability of a reward to learn the value of two alternatives. This is a classic decision-making task that involves combining prior probability and reward. In contrast, we are interested in perceptual decision-making under uncertainty, in which prior probability is combined with uncertain sensory signals. We might expect differences in strategy between cognitive and perceptual tasks, as cognitive tasks can be susceptible to additional biases. For example, participants often exhibit base-rate neglect (i.e., they ignore prior probability when evaluating evidence) in cognitive tasks [26] but not in perceptual tasks [2]. On the other hand, Behrens et al. [24] found that participants' behavior was well described by an optimal Bayesian model, in that observed learning rates quantitatively matched those of a Bayesian decision-maker carrying out the same task. A more recent study by Zylberberg et al. [25] examined probability learning under uncertainty in a motion-discrimination task. In this study, probability was estimated from a confidence signal rather than explicit feedback. Additionally, probability was changed across blocks, allowing participants to infer a change had occurred. Here, we examine probability learning when feedback is explicit and changes are sudden.

To investigate probability learning in uncertain conditions, we designed a perceptual categorization task in which observers need to learn category probability through experience. Since our goal was to examine low-level perceptual and decision-making processes, we used a

---

#### Page 3

highly controlled experimental environment. To prevent the use of external cues (e.g., the start of a new block indicating a change in probability) probabilities were changed using a sample-and-hold procedure. This approach has been used in decision-making [21, 22, 24] and motor domains [18] to examine behavior in dynamic contexts. Observers completed both a covertand overt-criterion task, in which the decision criterion was implicit or explicit, respectively. The overt task, previously developed by Norton et al. [16], provided a richer dataset upon which to compare models of human behavior. We determined how observers tracked category probability in a changing environment by comparing human behavior to both Bayesian and alternative models. We find that a number of models qualitatively describe human behavior, but that, quantitatively, model comparison favored an exponential averaging model with a bias towards equal priors and a flexible variant of the Bayesian change-point detection model with incorrect beliefs about the generative model. Although model comparison did not distinguish between these models, we interpret the exponential model with a conservative bias as a simpler, more biologically plausible explanation. Our results suggest that changes in the decision criterion are the result of both probability estimates computed on-line and a more stable, longterm prior.

# Results

## Experiment

During each session, observers completed one of two orientation-categorization tasks. On each trial in the covert-criterion task, observers categorized an ellipse as belonging to category A or B by key press (Fig 1A). On each trial in the overt-criterion task, observers used the mouse to rotate a line to indicate their criterion prior to the presentation of an ellipse (Fig 1B). The observer was correct if the ellipse belonged to category A and was clockwise of the criterion line or if the ellipse belonged to category B and was counter-clockwise of the criterion line. The overt-criterion task is an explicit version of the covert-criterion task developed by Norton et al. [16]. The overt-criterion task provides a richer dataset than the covert-criterion task in that it affords a continuous measure and allows us to see trial by trial changes in the reported decision criterion, at the expense of being a more cognitive task. In both tasks, the categories were defined by normal distributions on orientation with different means $\left(\mu_{\mathrm{B}}=\mu_{\mathrm{A}}+\Delta \theta\right)$ and equal standard deviation $\left(\sigma_{\mathrm{C}}\right)$; the mean of category A was set clockwise of the the mean of category B and a neutral criterion would be located halfway between the category means (Fig 1C). The difficulty of the task was equated across participants by setting $\Delta \theta$ to a value that predicted a $d^{\prime}$ value of 1.5 based on the data from the initial measurement session (see Methods). All data are reported relative to the neutral criterion, $z_{\text {neutral }}=\left(\mu_{\mathrm{A}}+\mu_{\mathrm{B}}\right) / 2$.

Prior to testing, observers were trained on the categories. Only the covert-criterion task was used for training (see Section 6 in S1 Appendix). During training, category probability was equal $(\pi=0.5)$ and observers received feedback on every trial that indicated both the correctness of the response (tone) and the generating category (visual). As a measure of category learning, we compute the probability of being correct in the training block and averaged across sessions. All observers learned the categories to the expected level of accuracy ( $p$ (correct) $=$ $0.74 \pm 0.01$; mean $\pm$ SEM across observers), given that the expected fraction of correct responses for an ideal observer with $d^{\prime}=1.5$ and equal priors over categories is 0.77 . As an additional test of category learning, immediately following training observers estimated the mean orientation of each category by rotating an ellipse. Each category mean was estimated once and no feedback was provided. There was a significant correlation between the true and estimated means for each category (category A: $r=0.82, p<0.0001$; category B: $r=0.97$, $p<0.0001$ ), suggesting that categories were learned. However, on average mean estimates

---

#### Page 4

> **Image description.** The image contains five panels, labeled A through E, illustrating an experimental design and related concepts.
>
> Panel A depicts a trial sequence in the covert-criterion task. It shows a sequence of gray squares. The first square shows a white plus sign and the text "500 ms" above it. The second square shows a black oval tilted diagonally and the text "300 ms" above it. The third square shows two small squares labeled "1" and "2". The last square shows two squares with a plus sign in the center. The left square has a green plus sign and the text "Points: +1" above it. The right square has a red plus sign and the text "Points: +0" above it. The text "300 ms" and a sound icon are above these two squares. A diagonal arrow labeled "time" connects the squares.
>
> Panel B depicts a trial sequence in the overt-criterion task. It shows a sequence of gray squares. The first square shows an orange line with a curved arrow indicating rotation. The next squares show a diagonal line with a different color above and below the line. The first square has a green line above and a yellow line below and the text "Points: +1" above it. The second square has a yellow line above and a green line below and the text "Points: +1" above it. The third square has a green line above and a yellow line below and the text "Points: +0" above it. The fourth square has a yellow line above and a red line below and the text "Points: +0" above it. The text "300 ms" and a sound icon are above these four squares. A diagonal arrow labeled "time" connects the squares.
>
> Panel C is a graph showing probability distributions. The x-axis is labeled "stimulus orientation (°)" and ranges from -45 to 45. The y-axis is labeled "probability". There are two curves: a green curve labeled "cat. A" and a red curve labeled "cat. B". The green curve is centered around -15 and the red curve is centered around 15.
>
> Panel D is a line graph showing changes in category probability over trials. The x-axis is labeled "trial number" and ranges from 0 to 800. The y-axis is labeled "πt" and ranges from 0 to 1. The graph shows a stepwise function that changes at various points. Downward pointing triangles are placed at each change point and labeled "change point".
>
> Panel E is a directed acyclic graph representing a generative model. The nodes are labeled with variables: Δt-1, Δt, πt-1, πt, Ct-1, Ct, st-1, st, xt-1, xt. Arrows indicate dependencies between the variables.

Fig 1. Experimental design. (A) Trial sequence in the covert-criterion task. After stimulus offset, observers reported the category by key press and received auditory feedback indicating the correctness of their response. In addition, the fixation cross was displayed in the color of the generating category. (B) Trial sequence in the overt-criterion task. Prior to stimulus onset, observers rotated a line to indicate their criterion by sliding the mouse side-to-side. When the observer clicked the mouse, a stimulus was displayed under the criterion line and auditory feedback was provided. (C) Example stimulus distributions for category A (green) and category B (red). Note that numbers on the $x$-axis are relative to the neutral criterion (i.e., 0 is the neutral criterion). (D) Example of random stepwise changes in probability across a block of trials. Change points occurred every 80-120 trials and are depicted above by the black arrows. Category A probabilities $\pi_{t}$ were sampled from a discrete set, $S_{\pi}=\{0.2,0.35,0.5,0.65,0.8\}$. (E) Generative model for the task in which category probability $\pi$ is updated following a sample-and-hold procedure, a category $C$ is selected based on the category probability, a stimulus $s$ is drawn from the category distribution and is corrupted by visual noise resulting in the noisy measurement $x$. Note that this diagram omits the dependency that leads to change points every $80-120$ trials.

were repelled from the category boundary (average category A error of $11.3^{\circ} \pm 6.3^{\circ}$ and average category B error of $-8.0^{\circ} \pm 2.6^{\circ}$; mean $\pm$ SEM across observers) suggesting a systematic repulsive bias.

To determine how category probability affects decision-making, during testing category A probability $\pi_{t}$ was determined using a sample-and-hold procedure (Fig 1D; category B probability was $1-\pi_{t}$ ). For $t=1$, category A probability was randomly chosen from a set of five probabilities $S_{\pi}=\{0.2,0.35,0.5,0.65,0.8\}$. On most trials, no change occurred $\left(\Delta_{t}=0\right)$, so that $\pi_{t+1}=\pi_{t}$. Every $80-120$ trials there was a change point $\left(\Delta_{t}=1\right)$, with change point sampled uniformly. At each change point, category probability was randomly selected from the $S_{\pi}$ excluding the current probability. On each trial $t$, a category $C_{t}$ was randomly selected (with $P$ (category $\mathrm{A})=\pi_{t}$ ) and a stimulus $s_{t}$ was drawn from the stimulus distribution corresponding to the selected category. We assume that the observer's internal representation of the stimulus

---

#### Page 5

is a noisy measurement $x_{t}$ drawn from a Gaussian distribution with mean $s_{t}$ and standard deviation $\sigma_{\mathrm{v}}$, which represents visual noise (v). The generative model of the task is summarized in Fig 1E.

# Bayesian models

To understand how decision-making behavior is affected by changes in category probability, we compared observer performance to several Bayesian models. To compute the behavior of a Bayesian observer, we developed a Bayesian change-point detection algorithm, based on Adams and MacKay [27], but which also accounts for Markov dependencies in the transition distribution after a change. Specifically, the Bayesian observer estimates the posterior distribution over the current run length (time since the last change point), and the state (category probability) before the last change point, given the data so far observed (category labels until trial $t, \boldsymbol{C}_{t}=\left(C_{1}, \ldots, C_{t}\right)$ ). We denote the current run length at the end of trial $t$ by $r_{t}$, the current state by $\pi_{t}$, and the state before the last change point by $\xi_{t}$, where both $\pi_{t}, \xi_{t} \in S_{n}$. That is, if a change point occurs after trial $t$ (i.e., $r_{t}=0$ ), then the new category A probability will be $\pi_{t}$ and the previous run's category probability $\xi_{t}=\pi_{t-1}$. If no change point occurs, both $\pi$ and $\xi$ remain unchanged. We use the notation $\boldsymbol{C}_{t}^{(r)}$ to indicate the set of observations (category labels) associated with the run $r_{t}$, which is $\boldsymbol{C}_{t-r_{t}+1: t}$ for $r_{t}>0$, and $\emptyset$ for $r_{t}=0$. The range of times with a colon, e.g., $\boldsymbol{C}_{t^{\prime}, t}$, indicates the sub-vector of $\boldsymbol{C}$ with elements from $t^{\prime}$ to $t$ included.

Both of our tasks provide category feedback, so that at the end of trial $t$ the observer has been informed of $\boldsymbol{C}_{1: t}$. In S1 Appendix we derive the iterative Bayesian ideal-observer model. After each trial, the model calculates a posterior distribution over possible run lengths and previous probability states, $P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)$. The generative model makes it easy to calculate the conditional probability of the current state for a given run length and previous state, $P\left(\pi_{0} \mid r_{t}, \xi_{t}, \boldsymbol{C}_{1: t}\right)$. These two distributions may be combined, marginalizing (summing) across the unknown run length and previous states to yield the predictive probability distribution of the current state, $P\left(\pi_{t} \mid \boldsymbol{C}_{1: t}\right)$. Given this distribution over states, in both tasks the observer needs to determine the probability of each category. In particular,

$$
P\left(\boldsymbol{C}_{t+1}=\mathrm{A} \mid \boldsymbol{C}_{1: t}\right)=\mathbb{E}\left[\pi_{t}\right]=\sum_{s_{t} \in S_{n}} \pi_{t} P\left(\pi_{t} \mid \boldsymbol{C}_{1: t}\right)
$$

In the overt task, the ideal observer sets the current criterion to the optimal value $z_{t}^{\text {opt }}$ based on the known category orientation distributions and the current estimate of category probabilities. Further, in the ideal and all subsequent models of the overt task, in addition to early sensory noise $\left(\sigma_{\mathrm{v}}\right)$ we assume the actual setting is perturbed by adjustment noise $\left(z_{t}=z_{t}^{\text {opt }}+\varepsilon_{t}\right.$, where $\left.\varepsilon_{t} \sim \mathcal{N}\left(0, \sigma_{\mathrm{v}}^{2}\right)\right)$.

In the covert task, the observer views a stimulus and makes a noisy measurement $x_{t}$ of its true orientation $s_{t}$ with noise variance $\sigma_{s}^{2}$. The prior category probability is combined with the noisy measurement to compute category A's posterior probability $P\left(\boldsymbol{C}_{t+1}=\mathrm{A} \mid x_{t+1}, \boldsymbol{C}_{1: t}\right)$. The observer responds "A" if that probability is greater than 0.5 .

We consider the ideal-observer model (Bayes ${ }_{\text {ideal }}$ ) and four (suboptimal) variants thereof, which deviate from the ideal observer in terms of their beliefs about specific features of the experiment (Bayes ${ }_{r}$, Bayes $_{s}$, Bayes $_{\beta}$, and Bayes $_{t, \pi, \beta}$ ). Two further variants of the Bayesian model (Bayes $_{\text {bias }}$ and Bayes $_{t, \beta}$ ) are described in S1 Appendix. Crucially, all these models are "Bayesian" in that they compute a posterior over run length and probability state, but they differ with respect to the observer's assumptions about the generative model. Note that these models differ from the model provided by Gallistel and colleagues [28], which was used to model a task in which participants explicitly indicated perceived probability and change points.

---

#### Page 6

Bayes $_{\text {ideal }}$. The ideal Bayesian observer uses knowledge of the generative model to maximize expected gain. This model assumes knowledge of sensory noise, the category distributions, the set of potential probability states, and the run length distribution. Because observers were trained on the categories prior to completing each categorization task, assuming knowledge of the category distributions seems reasonable. Further, it is reasonable to assume knowledge of sensory noise as this is a characteristic of the observer. However, since observers were not told how often probability would change, nor were they told the set of potential probability states, observers may have had incorrect beliefs about the generative model. Thus, we developed additional Bayesian models (described below), in which observers could have potentially incorrect beliefs about different aspects of the generative model.

Bayes $_{r}$. The Bayes ${ }_{r}$ model allows for an observer to have an incorrect belief about the run length distribution. For a given discrete $r$, the observer believes that the run length distribution is drawn from a discrete uniform distribution, $\sim \operatorname{Unif}\left[[\frac{r}{2} r], r\right]$. We chose this particular interval rather than a more general one, because it is simple and includes the true generative distribution. For the ideal observer, $r=120$.

Bayes $_{\alpha}$. The Bayes ${ }_{\alpha}$ model allows for an observer to have an incorrect belief about the set of probability states. The veridical set of experimental states is five values of $\pi$ ranging from 0.2 to 0.8 . The Bayes $_{\alpha}$ model observer also assumes five possible values of $\pi$ evenly spaced, but ranging from $\pi_{\min }$ to $\pi_{\max }=1-\pi_{\min }$, where $\pi_{\min }$ is a free parameter.

Bayes $\boldsymbol{\beta}$. While incorrect assumptions about parameters of the generative model result in suboptimal behavior, suboptimality can also arise from prior biases (that is, incorrect hyperparameters). The Bayes ${ }_{\beta}$ model, like the Bayes ${ }_{\text {ideal }}$ model, assumes knowledge of the generative model, but also includes a hyperprior $\operatorname{Beta}(\beta, \beta)$ that is applied after a change point. Thus, as $\beta$ increases, the posterior belief over category probabilities is biased toward equal probability. For the ideal observer, $\beta=1$ (a uniform distribution).

Bayes $_{r, \pi, \beta}$. The 'incorrect-belief' Bayesian models described above vary one assumption at a time, with at most a single additional free parameter. To get a sense of model performance when multiple assumptions are varied, the Bayes $_{r, \pi, \beta}$ model allows for incorrect beliefs about the run length and probability distributions and a prior bias towards equal probability. Specifically, the Bayes $_{r, \pi, \beta}$ model observer assumes that run length is drawn from the discrete distribution $[r, r+\Delta r]$ and that there are five possible values of $\pi$ evenly spaced ranging from $\pi_{\min }$ to $\pi_{\max }$, where $r \in[1,200], \Delta r \in[1,200], \pi_{\min } \in(0,0.5)$, and $\pi_{\max } \in(0.5,1)$ are all free parameters. The model also includes the hyperprior $\operatorname{Beta}(\beta, \beta)$, with $\beta \in(0,100]$, that is applied after a change point. Because of the model's complexity, excessive flexibility, and differences in the fitting procedure (see Methods), we only compared it to the best-fit model as determined by a separate model-comparison analysis. For the same reasons, note that this model is not included in the following figures or parameter tables. However, the results are summarized in Fig S1 in S1 Appendix.

# Alternative models

In addition to the Bayesian models described above, we tested the following alternative models that do not require the observer to compute a posterior over run length and probability state. In each of the following models, assumptions vary about whether and how probability is estimated. In the Fixed Criterion (Fixed) model the observer assumes fixed probabilities. In the Exponential-Averaging (Exp), Exponential-Averaging with Prior Bias ( $\operatorname{Exp}_{\text {bias }}$ ), and the Wilson et al. (2013) models, probability is estimated based on the recent history of categories. In the Reinforcement-Learning (RL) model, the decision criterion is updated following an errordriven learning rule with no assumptions about probability. Finally, the Behrens et al. (2007)

---

#### Page 7

model is an alternative Bayesian model with fewer assumptions and restrictions than the Bayesian change-point detection model described above. We also tested three additional models that are described in S1 Appendix.

Each of the following models, except for the RL model, computes an estimate of category probability $\left(\hat{\pi}_{\mathrm{A}, t}\right)$ on each trial and the estimated probability of the alternative is $\hat{\pi}_{\mathrm{B}, t}=1-\hat{\pi}_{\mathrm{A}, t}$. On each trial, the optimal criterion $z_{\text {opt }}$ is computed based on these estimated probabilities in the identical manner as for the Bayesian models. To make a categorization decision in the covert-criterion task, the criterion is applied to the noisy observation of the stimulus. In the overt-criterion task, the observer reports the criterion, which we again assume is corrupted by adjustment noise.

Fixed. While incorporating category probability into the decision process maximizes expected gain, a simpler strategy is to ignore changes in probability and fix the decision criterion throughout the block. In the fixed-criterion model, we assume equal category probability and the criterion is set to the neutral criterion:

$$
z=\frac{1}{2}\left(\mu_{\mathrm{B}}+\mu_{\mathrm{A}}\right)
$$

This model is equivalent to a model in which the likelihood ratio, $\frac{P\left(\mathrm{~A}, \mid C_{t}=\mathrm{A}\right)}{P\left(\mathrm{~A}, \mid C_{t}=\mathrm{B}\right)}$, is used to make categorization decisions rather than the posterior odds ratio in the covert-criterion task. This is a reasonable strategy for an observer who wants to make an informed decision, but is unsure about the current probability state and its rate of change.

Exp. The exponential-averaging model computes smoothed estimates of category probability by taking a weighted average of previously experienced category labels, giving more weight to recently experienced labels:

$$
\begin{aligned}
\hat{\pi}_{\mathrm{A}, 1} & =0.5 \\
\hat{\pi}_{\mathrm{A}, t+1} & =\alpha_{\mathrm{Exp}} C_{t}+\left(1-\alpha_{\mathrm{Exp}}\right) \hat{\pi}_{\mathrm{A}, t}
\end{aligned}
$$

where $\alpha_{\text {Exp }}$ is the smoothing factor, $0<\alpha_{\text {Exp }}<1$, and $C_{t}=1$ if category A is observed on trial $t$ and $C_{t}=0$ otherwise. The time constant of memory decay for this model is $\tau=\frac{-1}{\log (1-\alpha_{\text {Exp }})}$. Mathematically this model is equivalent to a delta-rule [29] model based on an "error" that is the difference between the current category and the current probability estimate: $\hat{\pi}_{\mathrm{A}, t+1}=\hat{\pi}_{\mathrm{A}, t}+\alpha_{\text {Exp }}\left(C_{t}-\hat{\pi}_{\mathrm{A}, t}\right)$. The criterion $z$ is then set to the optimal value based on this category-probability estimate.
$\mathbf{E x p}_{\text {bias. }}$ The $\operatorname{Exp}_{\text {bias }}$ model is identical to the Exp model, while also incorporating a common finding in the literature [2] known as conservatism (i.e., observers are biased towards the neutral criterion). On each trial an estimate of probability is computed as described in Eq (3), and averaged with a prior probability of 0.5 :

$$
\hat{\pi}_{\mathrm{A}, t}=w \hat{\pi}_{\mathrm{A}, t}+\frac{1}{2}(1-w)
$$

where $w$ is the weight given to the probability estimate $\hat{\pi}_{\mathrm{A}, t}$ and $(1-w)$ is the weight given to $\pi_{\mathrm{A}}=0.5$. The criterion $z$ is also set to the optimal value based on this category-probability estimate.

Wilson et al. (2013). Due to the complexity of the full Bayesian change-point detection model, Wilson et al. [30] developed an approximation to the full model using a mixture of delta rules. In their reduced model, the full run-length distribution is approximated by maintaining a subset of all possible run lengths. These are referred to as nodes $\left\{l_{1}, \ldots, l_{l}\right\}$. On each

---

#### Page 8

trial, an estimate of probability is computed for each node,

$$
\begin{aligned}
\hat{\pi}_{A, t}^{i_{t}} & =0.5 \\
\hat{\pi}_{A, t+1}^{i_{t}} & =\hat{\pi}_{A, t}^{i_{t}}+\frac{1}{l_{t}+v_{p}}\left(C_{t}-\hat{\pi}_{A, t}^{i_{t}}\right)
\end{aligned}
$$

where $v_{\mathrm{p}}$ is a model parameter that represents the strength of the observer's prior towards equal category probability (pseudocounts of a Beta prior; larger $v_{\mathrm{p}}$ corresponds to stronger conservatism) and $C_{t}$ is the category label. The learning rate for each node is thus $\alpha_{\text {Wilson }, l_{t}}=\frac{1}{l_{t}+v_{p}}$. For a single-node model, this is identical to the Exp model described above Eq (3). To obtain an overall estimate of probability in a multiple-node model, estimates (Eq (5)) are combined by taking a weighted average. The weights, $p\left(l_{t} \mid C_{1: t}\right)$, are updated on every trial. The update is dependent on an approximation to the change-point prior, which in turn depends on the hazard rate $h$ (for details see Wilson et al. [30], Eqs 25-31, along with the later correction [31]). In other words, when the probability of a change is high, more weight is given to $l_{1}$, which results in a greater change in the overall probability estimate. But when the probability of a change is low, more weight is given to nodes greater than $l_{1}$ and the probability estimate is more stable. For a three-node model probability is estimated as

$$
\hat{\pi}_{A, t+1}=p\left(l_{1} \mid C_{1: t}\right) \hat{\pi}_{A, t+1}^{i_{t}}+p\left(l_{2} \mid C_{1: t}\right) \hat{\pi}_{A, t+1}^{i_{t}}+p\left(l_{3} \mid C_{1: t}\right) \hat{\pi}_{A, t+1}^{i_{t}}
$$

For the purpose of the current experiment, we used a three-node model in which the first node was fixed $\left(l_{1}=1\right)$ and $l_{2}$ and $l_{3}$ were allowed to vary. In addition, the prior strength parameter, $v_{\mathrm{p}}$, which modulates the learning rate, was also free. By allowing $v_{\mathrm{p}}$ to vary, this model is equivalent to the three-node model described by Wilson and colleagues [30] in which all nodes were free and $v_{\mathrm{p}}=2$ was fixed. The hazard rate was set to 0.01 , corresponding to an expected block length of 100 trials (the experimental mean), and we assumed a change occurred at $t=1$, so that all the weight was given to $l_{1}$. We also tested an alternative model with a fixed value of $v_{\mathrm{p}}=2$, but it resulted in a worse fit.

RL. While the models described above make assumptions about how probability is estimated, it is also possible that observers simply update the decision criterion without estimating the current probability state. A model-free reinforcement-learning approach assumes the observer does not use knowledge of the environmental structure. Instead, the decision criterion, rather than an estimate of category probability, is updated. The observer updates the internal criterion $(z)$ on each trial according to the following delta rule:

$$
z_{t+1}=\left\{\begin{array}{l}
z_{t}, \text { if correct } \\
z_{t}+\alpha_{\mathrm{RL}}\left(x_{t}-z_{t}\right), \text { if incorrect. }
\end{array}\right.
$$

Thus, the criterion is updated when negative feedback is received by taking a small step in the direction of the difference between the noisy observation $\left(x_{t}\right)$ and current criterion $\left(z_{t}\right)$, where the step size is scaled by the learning rate $\alpha_{\mathrm{RL}}$. Nothing is changed after a correct response. Assuming effective training, we initialize the RL model by setting $z_{1}=\frac{1}{2}\left(\mu_{0}+\mu_{\mathrm{A}}\right)$.

Behrens et al. (2007). Behrens and colleagues [24] developed a model that uses Bayesian learning to update one's belief about the statistics of the environment to use on the next trial. Importantly, a Behrens et al. (2007) observer updates beliefs about the probability of category $\mathrm{A}(\pi)$, volatility $(v)$ and the volatility's rate of change $(k)$ without having to store the entire history of outcomes $\left(C_{1: t}\right)$ and reward statistics. The posterior probability over $\pi, v$, and $k$, or

---

#### Page 9

current belief about the environmental statistics, is computed as follows:

$$
\begin{aligned}
& p\left(\pi_{t+1}, v_{t+1}, k \mid C_{t: t+1}\right) \propto p\left(C_{t+1} \mid \pi_{t+1}\right) \times \\
& \quad \int\left[\int p\left(\pi_{t}, v_{t}, k \mid C_{t: t}\right) p\left(v_{t+1} \mid v_{t}, k\right) d v_{t}\right] p\left(\pi_{t+1} \mid \pi_{t}, v_{t+1}\right) d \pi_{t}
\end{aligned}
$$

where $p\left(\pi_{t+1} \mid \pi_{t}, v_{t+1}\right) \sim \beta\left(\pi_{t}, \exp \left(v_{t+1}\right)\right)$ is the probability transition function, $p\left(v_{t+1} \mid v_{t}, k\right) \sim$ $\mathcal{N}\left(v_{t}, \exp (k)\right)$ is the volatility transition function, and $p\left(C_{t+1} \mid \pi_{t+1}\right)$ is the likelihood of the most recently observed category. The likelihood is equal to $\pi_{t+1}$ when $C_{t+1}=1$ and $1-\pi_{t+1}$ when $C_{t+1}=0$. Integration was done numerically over a five-dimensional grid $\left(\pi_{t}, v_{t}, k, \pi_{t+1}, v_{t+1}\right)$. We assumed the observer adopted non-informative uniform priors over $\pi, v$, and $k$. Thus, each dimension was split into 30 equally spaced bins with $v$ and $k$ spaced evenly in log space (as per positive parameters of unknown scale). To obtain an estimate of the probability of category A , we marginalized over $v_{t+1}$ and $k$ and computed the mean of the probability distribution,

$$
\hat{\pi}_{\mathrm{A}, t+1}=\int \pi_{t+1} p\left(\pi_{t+1}\right) d \pi_{t+1}
$$

Other than sensory or adjustment noise, this model has no additional free parameters. Assuming perfect category knowledge, the criterion $z$ is set to the optimal value based on this category-probability estimate.

Behrens et al. (2007) with bias. For the Behrens et al. (2007) model with a bias towards equal probability, category probability was estimated as in Eq 9 and a weighted average between $\pi_{\mathrm{A}}=0.5$ and the model estimate was computed as in Eq 4. This model has one additional free parameter, $w$, which is the weight given to the probability estimate $\hat{\pi}_{A, t}$ as in the $\mathrm{Exp}_{\text {bias }}$ model. The criterion $z$ is then set to the optimal value based on the weighted categoryprobability estimate.

# Raw data

Fig 2 shows raw data for a single observer in the covert (Fig 2A) and overt (Fig 2C) tasks. For visualization in the covert task, the 'excess' number of A responses is plotted as a function of trial (gray line in Fig 2A). To compute the 'excess' number of A responses, we subtracted $\frac{1}{2}$ from the cumulative number of A responses. Thus, 'excess' A responses are constant for an observer who reported A and B equally often, increase when A is reported more, and decrease when A is reported less. To get a sense of how well the observer performed in the covert task, the number of 'excess' A trials (based on the actual category on each trial rather than the observer's response) is shown in black (Fig 2A, top). For reference, $\pi_{\mathrm{A}}$ is shown as a function of trial (Fig 2A, bottom). From visual inspection, the observer reported A more often when $\pi_{\mathrm{A}, t}>0.5$ and B more often when $\pi_{\mathrm{A}, t}<0.5$. Results for all observers in the covert task can be found in S1 Appendix (gray line in Figs S6A-S17A).

In the overt task, the orientation of the observer's criterion setting, relative to the neutral criterion, is plotted as a function of trial (gray circles). For visualization, a running average was computed over a five-trial moving window (gray line). Here (Fig 2C, top), the black line represents the criterion on each trial, given perfect knowledge of the categories, sensory uncertainty, and category probability. While this is impossible for an observer to attain, we can see that the observer's criterion follows the general trend. This suggests that observers update their criterion appropriately in response to changes in probability. That is, the criterion is set counterclockwise from the neutral criterion when $\pi_{\mathrm{A}, t}>0.5$, and clockwise of neutral when $\pi_{\mathrm{A}, t}<0.5$.

---

#### Page 10

> **Image description.** This image contains four panels labeled A, B, C, and D, each presenting graphs related to behavioral data and model fits.
>
> **Panel A:**
> This panel shows two graphs stacked vertically.
>
> - The top graph displays the "excess 'A' responses" on the y-axis, ranging from -45 to 90, against the "trial number" on the x-axis, ranging from 0 to 800. A gray line represents observer data, while a black line represents the true number of 'excess' A's. Both lines show fluctuations, generally increasing and then decreasing.
> - The bottom graph shows a step function, with the y-axis labeled "πt" ranging from 0 to 1. The x-axis is "trial number" ranging from 0 to 800. The function alternates between values of approximately 0.2 and 0.8 in a stepwise fashion.
>
> **Panel B:**
> This panel contains a grid of 9 smaller graphs. Each graph displays "excess 'A' responses" on the y-axis (ranging from -45 to 90) against "trial number" on the x-axis (ranging from 0 to 800). Each graph shows a gray line representing observer data, and a colored line representing a model fit. Shaded regions around the colored lines indicate the 68% confidence interval. The models are labeled as follows: Bayesideal (dark blue), Bayesr (blue), Bayesπ (light blue), Bayesβ (cyan), Fixed (red), Exp (green), Expbias (light green), Wilson et al. (2013) (yellow), RL (purple), Behrens et al. (2007) (orange), and Behrens et al. (2007) + bias (light yellow).
>
> **Panel C:**
> This panel mirrors the structure of Panel A, with two vertically stacked graphs.
>
> - The top graph plots "criterion orientation (°)" on the y-axis, ranging from -50 to 50, against "trial number" on the x-axis, ranging from 0 to 800. Gray circles represent raw settings, a gray line represents a running average over a 5-trial moving window, and a black line represents the ideal criterion.
> - The bottom graph is a step function similar to that in Panel A, with the y-axis labeled "πt" ranging from 0 to 1, and the x-axis labeled "trial number" ranging from 0 to 800. The step function has varying step heights.
>
> **Panel D:**
> This panel is structured similarly to Panel B, containing a grid of 9 smaller graphs. Each graph plots "criterion orientation (°)" on the y-axis (ranging from -50 to 50) against "trial number" on the x-axis (ranging from 0 to 800). Each graph shows a gray line representing the running average computed from the observer's data, and a colored line representing a model fit. The models are labeled as in panel B, with corresponding colors.

Fig 2. Behavioral data and model fits. (A) Behavioral data from a representative observer (CWG) in the covert-criterion task. Top: The number of 'excess' A responses (i.e., the cumulative number of A responses minus $j$ ) across trials. Gray line: Observer data. Black line: The true number of 'excess' A's. Bottom: The corresponding category probability. (B) Model fits (colored lines) for observer CWG in the covert-criterion task. Shaded regions: $68 \% \mathrm{CI}$ for model fits. Gray: Observer data. (C) Behavioral data from a representative observer (GK) in the overt-criterion task. Top: The orientation of the criterion line relative to the neutral criterion as a function of trial number. Gray circles: raw settings. Gray line: running average over a 5-trial moving window. Black line: ideal criterion for an observer with perfect knowledge of the experimental parameters. Bottom: Category probability across trials. (D) Models fits (colored lines) for observer GK in the overtcriterion task. A running average was computed over a 5-trial window for visualization. Shaded regions: $68 \% \mathrm{Cl}$ on model fits. These are generally smaller than the model-fit line. Gray line: the running average computed from the observer's data.

---

#### Page 11

Fig 2C (bottom) shows $\pi_{\mathrm{A}}$ as a function of trial. Results for all observers in the overt task can be found in S1 Appendix (gray line in Figs S6B-S17B).

# Modeling results

Model predictions. A qualitative comparison of the behavioral data to the ground truth suggests that observers updated their criterion in response to changes in probability. However, it does not tell us how. To explore the mechanism underlying these changes, we compared the observers' data to multiple models. For each task and model, the mean model response for a representative subject is plotted in Fig 2B (covert task) and Fig 2D (overt task). Shaded regions indicate $68 \%$ CIs computed from the posterior over model parameters. Specifically, we sampled parameters from the posterior over model parameters and computed the model response for the given set of parameters. We then computed the standard deviation across model responses for a large sample (see Model visualization). Occasionally, shaded regions computed on fits are narrower than the data line. Qualitatively, most models captured observers' changing criteria, with the fixed model being much worse. Differences across models are especially pronounced in the overt task. Specifically, we see that the Exp, $\operatorname{Exp}_{\text {bias }}$, Wilson et al. (2013), RL, and Behrens et al. (2007) models capture changes in criterion that occur between change points that the ideal Bayesian change-point model generally fails to capture. The MAP model fits for each observer, model, and task can be found in S1 Appendix (colored lines in Figs S6-S17).

Model comparison. We quantitatively compared models by computing the log marginal likelihood (LML) for each subject, model, and task. The marginal likelihood is the probability of the data given the model, marginalized over model parameters (see Methods). Here, we report differences in log marginal likelihood ( $\Delta \mathrm{LML}$, also known as log Bayes factor) from the best model, so that larger $\Delta \mathrm{LML}$ correspond to worse models. We compare models both by looking at average performance across subjects (a fixed-effects analysis), and also via a Bayesian Model Selection approach (BMS; [32]) in which subjects are treated as random variables. With BMS, we estimate for each model its posterior frequency $f$ in the population and its protected exceedance probability $\phi$, which is the probability that a given model is the most frequent model in the population, above and beyond chance [33].

Model comparison (Fig 3) favored the $\operatorname{Exp}_{\text {bias }}$ model, which outperformed the second best model Bayes, (covert task: $\Delta \mathrm{LML}=9.27 \pm 2.86$; overt task: $\Delta \mathrm{LML}=8.96 \pm 4.01$; mean and SEM across observers) in the two tasks (covert task: $t(10)=3.99, p=0.003$; overt task: $t(10)=$ $2.37, p=0.04$ ). Similarly, Bayesian model comparison performed at the group level also favored the $\operatorname{Exp}_{\text {bias }}$ model (covert task: $f=0.42$ and $\phi=0.96$; overt task: $f=0.34$ and $\phi=0.86$ ). These results suggest that observers estimate probability by taking a weighted average of recently experienced categories with a bias towards $\pi=0.5$.

To evaluate the performance of the more complex Bayes ${ }_{r, \pi, \beta}$ model, we computed both the Bayesian Information Criterion (BIC), obtained via maximum-likelihood estimation, and an approximate expected lower bound (ELBO) of the log marginal likelihood, obtained via Variational Bayesian Monte Carlo [34], a recent variational inference technique (see Methods for details). For a fair comparison, we also computed the same metrics for the best-fitting model from the analysis above $\left(\operatorname{Exp}_{\text {bias }}\right)$, using the same fitting procedures. In this analysis, negative $\Delta \mathrm{BIC}$ and positive $\Delta \mathrm{ELBO}$ correspond to evidence in favor of the $\operatorname{Exp}_{\text {bias }}$ model. For the covert task, the two models were indistinguishable in terms of both $\mathrm{BIC}(\Delta \mathrm{BIC}=-2.75 \pm 6.07 ; t(10)=$ $-0.45, p=0.66$ ) and $\operatorname{ELBO}(\Delta \mathrm{ELBO}=0.93 \pm 2.81 ; t(10)=0.33, p=0.75)$. The same finding held for the overt task, in that models performed comparably in terms of both $\mathrm{BIC}(\Delta \mathrm{BIC}=$ $-2.14 \pm 10.58 ; t(10)=-0.20, p=0.84)$ and $\operatorname{ELBO}(\Delta \mathrm{ELBO}=1.54 \pm 4.89 ; t(10)=0.32, p=0.76)$.

---

#### Page 12

> **Image description.** This image contains two sets of bar graphs comparing different models for covert and overt tasks. Each set of graphs is divided into two panels, labeled A and B.
>
> Panel A:
>
> - Two bar graphs are stacked vertically. The top graph represents the "Covert task" and the bottom graph represents the "Overt task."
> - The y-axis of both graphs is labeled "relative log marginal likelihood (LML<sub>Expbias</sub> - LML)". The y-axis scale ranges from 0 to 200 for the covert task and 0 to 100 for the overt task.
> - The x-axis of both graphs is labeled "model" and shows different models: Bayes<sub>ideal</sub>, Bayes<sub>r</sub>, Bayes<sub>π</sub>, Bayes<sub>β</sub>, Fixed, Exp, Exp<sub>bias</sub>, Wilson et al. (2013), RL, Behrens et al. (2007), and Behrens et al. (2007) + bias.
> - Each model is represented by a colored bar, and error bars are shown on top of each bar. The bars are different colors, including dark blue, light blue, red, green, yellow-brown, purple, and orange.
>
> Panel B:
>
> - Two stacked bar graphs are shown vertically, corresponding to the covert and overt tasks.
> - The y-axis is labeled "protected exceedance probability (φ)" and ranges from 0 to 1.
> - Each bar is mostly green, labeled "Exp<sub>bias</sub>" with a small portion at the top representing "other models" in different colors.
> - The height of the green bar indicates the probability of the Exp<sub>bias</sub> model being the best fit.

Fig 3. Model comparison. (A) Average log marginal likelihood (LML) scores relative to the best-fitting model, $\operatorname{Exp}_{\text {bias }}$ (top: covert task; bottom: overt task). Lower scores indicate a better fit. Error bars: $95 \%$ CI (bootstrapped). (B) Bayesian model selection at the group level. The protected exceedance probability $(\phi)$ is plotted for each model. Models are stacked in order of decreasing probability.
https://doi.org/10.1371/journal.pcbi. 1006681 . g003
In short, according to both metrics the $\operatorname{Exp}_{\text {bias }}$ model describes the data at least as well as the much more complex Bayes ${ }_{r, \pi, \beta}$ model. The model fits and results of this comparison can be found in Fig S1 in S1 Appendix.

Fig 4A shows the number of observers that were best fit by each model for each task. To compare LML scores across tasks, and for the purpose of this analysis only, we standardized model scores for each observer and task. Standardized LML scores in the overt task are plotted as a function of standardized LML scores in the covert task in Fig 4B. We found a significant positive correlation, $r=0.60, p<0.01$, indicating that models with higher LML scores in the covert task were also higher in the overt task. This result suggests that strategy was fairly consistent across tasks at the group level. In addition, there was more variance in model scores for worse-fitting models.

Model parameters. We examine here the parameter estimates of the $\operatorname{Exp}_{\text {bias }}$ model, recalling that the two model parameters $\alpha_{\text {Exp }}$ and $w$ represent, respectively, the exponential smoothing factor and the degree to which observers exhibit conservatism. The maximum a posteriori (MAP) parameter estimates are plotted in Fig 5. Converting the smoothing factor to a time constant over trials, we found that the time constant in both tasks was well below the true rate

---

#### Page 13

> **Image description.** The image contains two panels, labeled A and B, showing modeling results for individual observers.
>
> Panel A displays a stacked bar chart. The vertical axis is labeled "model frequency" and ranges from 0 to 12. The horizontal axis is labeled "task" and has two categories: "covert" and "overt". Each category has a stacked bar. The "covert" bar is composed of green, blue, purple, and gold sections, from bottom to top. The "overt" bar is composed of green, gold, blue, light blue sections, from bottom to top.
>
> Panel B is a scatter plot comparing standardized LML (Log Marginal Likelihood) scores across tasks. The vertical axis is labeled "standardized LML scores (overt)" and ranges from -3 to 3. The horizontal axis is labeled "standardized LML scores (covert)" and also ranges from -3 to 3. A dashed black line represents the identity line. Various colored dots represent data points, with each color corresponding to a different model, as indicated by the legend on the right. The legend lists the models: Bayesideal (dark blue), Bayesr (blue), Bayesπ (dark blue), Bayesβ (light blue), Fixed (red), Exp (dark green), Expbias (light green), Wilson et al. (2013) (gold), RL (purple), Behrens et al. (2007) (orange), and Behrens et al. (2007)bias (yellow). The R-squared value is displayed as "R² = 0.38".

Fig 4. Modeling results for individual observers. (A) Model frequency. The number of observers best fit by each model plotted for each task. Models are stacked in order of decreasing frequency. (B) Comparison of LML scores across tasks. LML scores were standardized for each observer and task. Standardized LML scores in the overt task are plotted as a function of standardized LML scores in the covert task (colored data points). Black dashed line: identity line.
https://doi.org/10.1371/journal.pcbi. 1006681 . g004
of change (covert: $\tau=[4.24,7.18]$; overt: $\tau=[3.48,4.75] ; \tau_{\text {true }}=100$ trials on average). We conducted paired-sample $t$-tests to compare the raw parameter estimates in the covert and overt tasks. We found a significant difference in $w(t(10)=-2.55, p=0.03)$, suggesting that observers were more conservative in the covert than the overt task. No significant difference was found for $\alpha_{\text {Exp }}(t(10)=-0.98, p=0.35)$.

To investigate whether there was bias in the parameter-estimation procedure when fitting the $\operatorname{Exp}_{\text {bias }}$ model, we also conducted a parameter-recovery analysis. Most parameters could be recovered correctly, except for adjustment variability $\left(\sigma_{\mathrm{a}}\right)$ in the overt task, which we found to be overestimated on average (see Section 4 in S1 Appendix for details). Note that this bias in estimating $\sigma_{\mathrm{a}}$ in the overt task does not affect our model comparison, which is based on LMLs and not on point estimates.

While we might expect performance to be similar across tasks and observers (i.e., a correlation between the parameter fits in each task), no significant correlations were found ( $\alpha_{\text {Exp }}: r=$ $-0.14, p=0.67 ; w: r=0.16, p=0.64$ ). Parameter estimates for all models, except the Bayes ${ }_{r, \alpha, \beta}$ model, are shown in Tables 1 and 2.

As the Bayes ${ }_{r, \alpha, \beta}$ contained a number of parameters that were unique to the model, parameter estimates for this model were not included in the tables below (covert: $\sigma_{\mathrm{v}}=9.6 \pm 0.7$, $r=56.9 \pm 19.1, \Delta r=37.0 \pm 14.8, \pi=.15 \pm .04, \Delta \pi=.35 \pm .04$, and $\beta=1.68 \pm .6$; overt: $\sigma_{\mathrm{a}}=$ $15.5 \pm 1.1, r=32.3 \pm 10.2, \Delta r=30.7 \pm 9.5, \pi=.13 \pm .04, \Delta \pi=.42 \pm .03$, and $\beta=0.42 \pm .2)$.

# Discussion

Although we know that people update decision criteria in response to explicit changes in prior probability, the effects of implicit changes in prior probability on decision-making behavior are less well known. In the present study, we used model comparison to investigate the mechanisms underlying decision-making behavior in an orientation-categorization task as prior probability changed. We tested a set of models that varied in both computational and memory

---

#### Page 14

> **Image description.** This image contains two panels, A and B, each presenting data related to parameter estimates for the Exp_bias model.
>
> Panel A contains two bar graphs.
>
> - The top graph displays the mean α_exp values across observers for covert and overt tasks. The bars are white with black borders, and error bars are visible on top of each bar. The y-axis ranges from 0 to 1, labeled with 0 and 0.5. The x-axis is labeled "task" with "covert" and "overt" as categories.
> - The bottom graph shows the mean w values across observers for covert and overt tasks. The bars are solid black, and error bars are visible on top of each bar. The y-axis ranges from 0 to 1, labeled with 0 and 0.5. The x-axis is labeled "task" with "covert" and "overt" as categories. A horizontal line with an asterisk above it connects the tops of the two bars, indicating significance.
>
> Panel B contains two scatter plots.
>
> - The top plot shows individual α_exp MAP values from fits in the overt task as a function of MAP values from fits in the covert task. The data points are represented by open circles. The x and y axes are labeled "parameter value (covert)" and "parameter value (overt)", respectively, and both range from 0 to 1, labeled with 0, 0.5, and 1. A dashed black line represents the identity line.
> - The bottom plot shows individual w MAP values from fits in the overt task as a function of MAP values from fits in the covert task. The data points are represented by solid black circles. The x and y axes are labeled "parameter value (covert)" and "parameter value (overt)", respectively, and both range from 0 to 1, labeled with 0, 0.5, and 1. A dashed black line represents the identity line.
> - A legend is located between the two scatter plots, indicating that open circles represent α_exp and solid black circles represent w.

Fig 5. Parameter estimates for the $\operatorname{Exp}_{\text {bias }}$ model. (A) Mean $\alpha_{\text {exp }}$ (top) and $w$ (bottom) MAP values across observers. The \* denotes significance at the $p<0.05$ level. Error bars: $\pm$ SEM. (B) Individual $\alpha_{\text {exp }}$ (top) and $w$ (bottom) MAP values from fits in the overt task as a function of MAP values from fits in the covert task. Black dashed lines: identity line.
https://doi.org/10.1371/journal.pcbi. 1006681 . g005
demands. Models were tested on data from both a covert- and overt-criterion task. A comprehensive approach, consisting of both qualitative and quantitative analysis, was performed to determine the best fitting model. We found that observers updated their decision criterion following changes in probability. Additionally, we observed systematic changes in the decision criterion during periods of stability, which was clearly evident in the overt-criterion data. While most models fit the data reasonably well qualitatively, model comparison slightly favored an exponential-averaging model with a bias towards equal probability, which was indistinguishable from a flexible variant of the Bayesian change-point detection model with

---

#### Page 15

Table 1. Maximum a posteriori parameter estimates $\pm$ S.E. in the covert-criterion task.

| Model                        | $\boldsymbol{\sigma}_{\mathbf{c}}$ | $\boldsymbol{r}$ | $\boldsymbol{\pi}$ | $\boldsymbol{\beta}$ | $\boldsymbol{\alpha}$ | $\boldsymbol{w}$ | $\boldsymbol{l}_{\mathbf{2}}$ | $\boldsymbol{l}_{\mathbf{3}}$ | $\boldsymbol{v}_{\mathbf{p}}$ |
| :--------------------------- | :--------------------------------: | :--------------: | :----------------: | :------------------: | :-------------------: | :--------------: | :---------------------------: | :---------------------------: | :---------------------------: |
| Bayes $_{\text {sibol }}$    |           $9.7 \pm 0.6$            |        -         |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Bayes $_{\text {e }}$        |           $9.9 \pm 0.8$            | $51.5 \pm 11.8$  |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Bayes $_{\pi}$               |           $10.5 \pm 0.8$           |        -         |  $0.32 \pm 0.02$   |          -           |           -           |        -         |               -               |               -               |               -               |
| Bayes $_{\beta}$             |           $10.2 \pm 0.7$           |        -         |         -          |    $18.4 \pm 8.2$    |           -           |        -         |               -               |               -               |               -               |
| Fixed                        |           $10.9 \pm 0.8$           |        -         |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Exp                          |           $9.4 \pm 0.7$            |        -         |         -          |          -           |    $0.04 \pm 0.01$    |        -         |               -               |               -               |               -               |
| Exp $_{\text {bias }}$       |           $10.0 \pm 0.8$           |        -         |         -          |          -           |    $0.17 \pm 0.04$    | $0.58 \pm 0.05$  |               -               |               -               |               -               |
| Wilson et al. (2013)         |           $9.4 \pm 0.7$            |        -         |         -          |          -           |           -           |        -         |        $12.2 \pm 2.8$         |       $118.5 \pm 23.3$        |        $43.3 \pm 14.4$        |
| RL                           |           $9.2 \pm 0.7$            |        -         |         -          |          -           |    $0.26 \pm 0.03$    |        -         |               -               |               -               |               -               |
| Behrens et al. (2007)        |           $8.5 \pm 0.3$            |        -         |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Behrens et al. (2007) + bias |           $10.4 \pm 0.7$           |        -         |         -          |          -           |           -           | $0.31 \pm 0.04$  |               -               |               -               |               -               |

incorrect beliefs and a bias towards equal probability. We can thus interpret the $\operatorname{Exp}_{\text {bias }}$ model as a simpler explanation, in which observers update the decision criterion by combining online estimation of probability with an equal-probability prior. Ultimately, our results help explain decision-making behavior in situations in which people need to assess the probability of an outcome based on previous experience.

# Criterion updates in response to implicit changes in category probability

To determine the influence of prior probability on decision-making behavior, we examined changes in the decision criterion. First, we found that no participant was best fit by a fixed-criterion model. This finding suggests that observers update decision criteria in response to implicit changes in probability. This result is consistent with previous studies in which prior probability was explicit [2-9]. Further, this finding complements recent studies suggesting that individuals can learn and adapt to statistical regularities in changing environments [14-17, 24, $25,35,36]$. Although this finding suggests that observers dynamically adjust decision criteria in response to changes in prior probability, it does not tell us how they do this (e.g., do observers compute on-line estimates of probability?). To uncover the mechanisms underlying changes in decision-making behavior, we compared multiple models ranging from the full Bayesian change-point detection model to a model-free reinforcement-learning (RL) model.

Table 2. Maximum a posteriori parameter estimates $\pm$ S.E. in the overt-criterion task.

| Model                        | $\boldsymbol{\sigma}_{\mathbf{a}}$ | $\boldsymbol{r}$ | $\boldsymbol{\pi}$ | $\boldsymbol{\beta}$ | $\boldsymbol{\alpha}$ | $\boldsymbol{w}$ | $\boldsymbol{l}_{\mathbf{2}}$ | $\boldsymbol{l}_{\mathbf{3}}$ | $\boldsymbol{v}_{\mathbf{p}}$ |
| :--------------------------- | :--------------------------------: | :--------------: | :----------------: | :------------------: | :-------------------: | :--------------: | :---------------------------: | :---------------------------: | :---------------------------: |
| Bayes $_{\text {sibol }}$    |           $17.2 \pm 1.1$           |        -         |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Bayes $_{\text {e }}$        |           $16.4 \pm 1.1$           |   $70 \pm 15$    |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Bayes $_{\pi}$               |           $16.7 \pm 1.1$           |        -         |  $0.17 \pm 0.03$   |          -           |           -           |        -         |               -               |               -               |               -               |
| Bayes $_{\beta}$             |           $17.0 \pm 1.1$           |        -         |         -          |    $13.2 \pm 8.2$    |           -           |        -         |               -               |               -               |               -               |
| Fixed                        |           $19.4 \pm 1.1$           |        -         |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Exp                          |           $16.7 \pm 1.1$           |        -         |         -          |          -           |    $0.09 \pm 0.02$    |        -         |               -               |               -               |               -               |
| Exp $_{\text {bias }}$       |           $16.0 \pm 1.0$           |        -         |         -          |          -           |    $0.22 \pm 0.03$    | $0.74 \pm 0.06$  |               -               |               -               |               -               |
| Wilson et al. (2013)         |           $16.4 \pm 1.1$           |        -         |         -          |          -           |           -           |        -         |        $10.8 \pm 2.0$         |        $46.4 \pm 18.4$        |         $9.2 \pm 5.1$         |
| RL                           |           $18.1 \pm 1.2$           |        -         |         -          |          -           |    $0.21 \pm 0.03$    |        -         |               -               |               -               |               -               |
| Behrens et al. (2007)        |           $18.6 \pm 1.0$           |        -         |         -          |          -           |           -           |        -         |               -               |               -               |               -               |
| Behrens et al. (2007) + bias |           $17.4 \pm 1.0$           |        -         |         -          |          -           |           -           | $0.66 \pm 0.06$  |               -               |               -               |               -               |

---

#### Page 16

# Systematic criterion fluctuations

How is the decision criterion set? Qualitatively, most models appear to fit the data reasonably well in the covert task. However, when we look at data from the overt task, while the Bayesian change-point detection models captured the overall trend, some variants failed to capture local fluctuations in the decision criterion observed during periods of stability (i.e., time intervals between change points). In other words, the criterion predicted by these models stabilized whereas the observers' behavior did not. This was less true when the Bayesian change-point detection model was allowed to vary more freely. In contrast, the exponential-averaging models continually update the observer's estimate of probability based on recently experienced categories. It can be difficult to discriminate a hierarchical model (e.g., the Bayesian change-point detection models) from flat models (e.g., our exponential models). For example, in a similar paradigm, Heilbron and Meyniel [37] found that predictions of change points could be similar for hierarchical and flat models. The addition of confidence judgments allowed them to discriminate the two model forms more readily.

How quickly observers updated this estimate is determined in the model by the decay-rate parameter. From our model fits, we found that observers had an average decay rate that was substantially smaller than the true run length distribution (on average 4.5 vs. 100 trials, respectively), leading to frequent, systematic fluctuations in decision criteria. Although we cannot directly observe these fluctuations in the covert task, because the estimated decay rate was not significantly different across tasks we can assume the fluctuations occurred in a similar manner. Like the exponential models, the RL model was also able to capture local fluctuations in the decision criterion. However, the amplitude of the changes in criterion predicted by the RL model was generally too low compared to the data. This discrepancy was especially clear in the overt task; no participant was best fit by the RL model. As a more fair comparison to the exponential models, we also fit the Bayesian model developed by Behrens and colleagues [24] with and without a bias towards equal probability. These models are more fair in that they do not require specifying a run-length distribution and allowed us to increase the number of possible probability states. While this allowed us to capture local fluctuations, model comparison favored the $\operatorname{Exp}_{\text {bias }}$ model. These results have two important implications. (1) It is important to test alternatives to Bayesian models: observers' behavior might be explained without requiring an internal representation of probability. (2) Using multiple tasks together with rigorous model comparison can provide additional insight into behavior (see also [38]). Here, the fluctuations in decision criteria between change points led to suboptimal behavior. Overall, our findings suggest that suboptimality arose from an incorrect, possibly heuristic inference process, that goes beyond mere sensory noise [39-41].

## A dual mechanism for criterion updating

While the Bayes ${ }_{r, n, \beta}$ and $\operatorname{Exp}_{\text {bias }}$ models fit the data equally well, the $\operatorname{Exp}_{\text {bias }}$ model provides a simpler explanation-considering that our experiment did not necessarily require subjects to build a hierarchical model of the task (see [37]). This suggests that observers compute on-line estimates of category probability based on recent experience. Further, the bias component of the model suggests that observers are conservative, as reflected in a long-term prior that categories are equally likely. The degree to which observers weight this prior varied across individuals and tasks. Taken together, these results suggest a dual mechanism for learning and incorporating prior probability into decisions. That is, there are (at least) two components to decision making that are acquired and updated at very different timescales.

Multiple-mechanism models have been used to describe behavior in decision-making [30] and motor behavior [42]. A model that combines delta rules predicts motor behavior better

---

#### Page 17

than either delta rule alone [42]. Using a combination of delta rules [30], we were able to capture the local fluctuations in criterion that the ideal Bayesian model missed. However, we found that a constant weight on $\pi=0.5$ fit better than the multiple-node model described by Wilson and colleagues [30]. Temporal differences between their task and ours might explain some of the differences we observed, as changes occurred much more slowly in our experiment. Additionally, while fitting Wilson et al.'s model we set the hazard rate to 0.01 (the average rate of change), but observers had to learn this value throughout the experiment and may have had incorrect assumptions about the rate of change [21, 22].

# Explanations of conservatism

Conservatism was an important feature in our models, as it improved model fits each time it was incorporated. While we observed conservatism in both the covert- and overt-criterion tasks, we found that, on average, observers were significantly more conservative in the covert task. To understand why conservatism differs across tasks, we need to understand the differences between the tasks. While the generative model was identical across tasks, the observer's response differed. In the covert task, observers chose between two alternatives. In the overt task, observers selected a decision criterion. This is an important difference because it allows us to potentially rule out previous explanations of conservatism, such as the use of subjective probability [4], misestimation of the relative frequency of events [43, 44], and incorrect assumptions about the sensory distributions [45, 46]; these explanations predict similar levels of conservatism across tasks. On the other hand, conservatism may be due to the use of suboptimal decision rules. Probability matching is a strategy in which participants select alternatives proportional to their probability, and has been used to explain suboptimal behavior in forcedchoice tasks in which observers choose between two or more alternatives [6, 47-50]. Thus, the higher levels of conservatism in the covert task may have been due to the use of a suboptimal decision rule like probability matching, which would effectively smooth the observer's response probability across trials. Probability matching is not applicable to responses in the overt task. Thus, the use of different decision rules may result in different levels of conservatism. These differences may also arise from an increase in uncertainty in the covert task due to less explicit feedback. An observer with greater uncertainty will rely more on the prior. Thus, conservatism may be the result of having a prior over criteria that interacts with task uncertainty. This can be tested by manipulating uncertainty over the generative model and measuring changes in conservatism. It is also possible that our training protocol introduced a bias towards equal probability and, due to the greater similarity between the covert and training tasks, the bias was stronger in the covert task. Finally, it is also possible that conservatism is the result of both the use of suboptimal decision rules and one or more of the previously proposed explanations.

## Incorrect assumptions about the generative model

While we tested a number of Bayesian change-point detection models that explored an array of assumptions about the generative model, clearly one could propose even more variants (e.g., a model with incorrect assumptions about category means and variance). Here, we analyzed one such assumption at a time. A simple way to expand the model space is via a factorial comparison [40, 51], which we did not consider here due to computational intractability and the combinatorial explosion of models. We did however, fit the Bayes ${ }_{\text {c.g.g }}$ model, which simultaneously accounted for incorrect assumptions about the run length and probability-state distributions and a bias towards equal probability. We compared the fit to the $\operatorname{Exp}_{\text {bias }}$ model. Notably these two models explained the data equally well, despite the higher flexibility of the

---

#### Page 18

Bayesian model. We can thus interpret the $\operatorname{Exp}_{\text {bias }}$ model as a simpler explanation for Bayesian change-point detection behavior with largely erroneous beliefs. For all models except the RL model we assumed knowledge of the category distributions. However, Norton et al. [16] found that for the same orientation-categorization task, category means were estimated dynamically, even after prolonged training. Similarly, Gifford et al. [52] observed suboptimality in an audi-tory-categorization task and found that the data were best explained by a model with non-stationary categories and prior probability that was updated using the recent history of category exemplars. This occurred despite holding categories and probability constant within a block. In fact, similar effects of non-stationarity have been observed in several other studies [53-55]. In addition to non-stationary category means, observers may also have misestimated category variance [56], especially since learning category variance takes longer than learning category means [14].

# Conclusion

In sum, our results provide a computational model for how decision-making behavior changes in response to implicit changes in prior probability. Specifically, they suggest a dual mechanism for learning and incorporating prior probability that operate at different timescales. Importantly, this helps explain behavior in situations in which assessment of probability is learned through experience. Further, our results demonstrate the need to compare multiple models and the benefit of using tasks that provide a richer, more informative dataset.

## Methods

## Ethics statement

The Institutional Review Board at New York University approved the experimental procedure and observers gave written informed consent prior to participation.

## Participants

Eleven observers participated in the experiment (mean age 26.6, range 20-31, 8 females). All observers had normal or corrected-to-normal vision. One of the observers (EHN) was also an author.

## Apparatus and stimuli

Stimuli were presented on a gamma-corrected Dell Trinitron P780 CRT monitor with a $31.3 \times$ 23.8" display, a resolution of $1024 \times 768$ pixels, a refresh rate of 85 Hz , and a mean luminance of $40 \mathrm{~cd} / \mathrm{m}^{2}$. Observers viewed the display from a distance of 54.6 cm . The experiment was programmed in MATLAB [57] using the Psychophysics Toolbox [58, 59].

Stimuli were $4.0 \times 1.0^{\prime}$ ellipses presented at the center of the display on a mid-gray background. In both the orientation-discrimination and covert-criterion tasks, trials began with a central white fixation cross (1.2'). In the overt-criterion task, a yellow line with random orientation was presented at the center of the display ( $5.0 \times 0.5^{\prime}$ ).

## Procedure

Categories. In the 'categorization' sessions described below, stimulus orientations were drawn from one of two categories (A or B). Category distributions were Gaussian with different means $\left(\mu_{\mathrm{B}}>\mu_{\mathrm{A}}\right)$ and equal variance $\left(\sigma_{\mathrm{A}}^{2}=\sigma_{\mathrm{B}}^{2}=\sigma_{\mathrm{c}}^{2}=100 \operatorname{deg}^{2}\right)$. The mean of category A was chosen randomly from all possible orientations at the beginning of each session and the mean of category B was set so that $d^{\prime}$ was approximately 1.5 , which was determined for each

---

#### Page 19

observer based on the estimates of sensory uncertainty obtained during a 'measurement session' (Fig 1C).

Sessions. All observers participated in three 1-hour sessions. Observers completed a 'measurement' session first followed by two 'categorization' sessions. Observers completed the covert-criterion task in the first 'categorization' session followed by the overt-criterion task in the second, or vice versa (chosen randomly). At the beginning of each 'categorization' session, observers completed 200 trials of category training followed by 800 experimental trials. Prior to training, observers were provided with a detailed explanation of the category distributions and training task. After training, observers were provided with additional instructions about the subsequent 'categorization' task and told that the categories would remain constant for the remainder of the session but that category probability may change. Observers were not told how often category probability would change or the range of probability states.

# Measurement task

During the 'measurement' session, sensory uncertainty $\left(\sigma_{w}\right)$ was estimated using a two-interval, forced-choice, orientation-discrimination task in which two black ellipses were presented sequentially on a mid-gray background. The observer reported the interval containing the ellipse that was more clockwise by keypress. Once the response was recorded, auditory feedback was provided and the next trial began. An example trial sequence is shown in Fig S3A in S1 Appendix.

The orientation of the ellipse in the first interval was chosen randomly on every trial from a uniform distribution ranging from -90 to $90^{\circ}$. The orientation of the second ellipse was randomly oriented clockwise or counter-clockwise of the first. The difference in orientation between the two ellipses was selected using an adaptive staircase procedure. The minimum step-size was $1^{\prime}$ and the maximum step-size was $32^{\prime}$. Each observer ran two blocks. In each block, four staircases ( 65 trials each) were interleaved (two 1-up, 2-down and two 1-up, 3-down staircases) and randomly selected on each trial. For analyses and results see S1 Appendix.

## Category training

Each training trial was identical to a covert-criterion trial (Fig 1A). During training there was an equal chance that a stimulus was drawn from either category. To assess learning of category distributions, observers were asked to estimate the mean orientation of each category following training. The mean of each category was estimated exactly once. The order in which category means were estimated was randomized. For estimation, a black ellipse with random orientation was displayed in the center of the display. Observers slid the mouse to the right and left to rotate the ellipse clockwise and counterclockwise, respectively and clicked the mouse to indicate they were satisfied with the setting. No feedback was provided. We computed the proportion correct for each observer to ensure category learning by comparing it to the expected proportion correct $(p($ correct $)=0.77)$ for $d^{\prime}=1.5$. Mean estimates are plotted in Fig S4C in S1 Appendix as a function of the true category means. We computed the average estimation error for each category and observer by subtracting the estimate from the true mean. From visual inspection, it appears that training was effective with the exception of one outlier, which we assume was a lapse.

## Categorization tasks

Covert-criterion task. In the covert-criterion task, observers categorized ellipses based on their orientation. The start of each trial $\left(N_{\text {trials }}=800\right)$ was signaled by the appearance of a

---

#### Page 20

central white fixation cross ( 500 ms ). A black oriented ellipse was then displayed at the center of the screen ( 300 ms ). Observers categorized the ellipse as A or B by keypress. Observers received feedback as to whether they were correct on every trial. Observers received a point for every correct response and aggregate points were displayed at the top of the screen to motivate observers. In addition, the fixation cross was displayed at the center of the screen in the color corresponding to the true category (category A: green; category B: red). The next trial began immediately. An example trial sequence is depicted in Fig 1A.

Overt-criterion task. In the overt-criterion task, observers completed an explicit version of the categorization task described above that was developed by Norton et al. [16]. At the beginning of each trial $\left(N_{\text {trials }}=800\right)$, a line was displayed at the center of the screen. The orientation of the line was randomly selected from a uniform distribution ranging from -90 to $90^{\circ}$. The observers' task was to rotate the line to indicate the criterion for that trial. Observers were explicitly instructed to set the criterion so that a subsequent category A stimulus would fall clockwise of the line and category B stimuli would fall counter-clockwise of it. Observers rotated the line clockwise or counterclockwise by sliding the mouse to the right or left and clicked the mouse to indicate their setting. Next, an ellipse was displayed under the criterion line in the color corresponding to the true category for 300 ms . Auditory feedback indicated whether the set criterion correctly categorized the ellipse. That is, observers were correct when a category A stimulus was clockwise of the criterion line or a category B stimulus was counterclockwise of the line. Observers received a point for a correct response and aggregate points were displayed at the top of the screen. The next trial began immediately. An example trial sequence is depicted in Fig 1B.

# Model fitting

For fitting, all models had one free noise parameter. In the covert-criterion task, this was sensory noise $\left(\sigma_{\mathrm{s}}\right)$. In the overt-criterion task, sensory noise was fixed and set to the value obtained in the 'measurement' session, but we included a noise parameter for the adjustment of the criterion line $\left(\sigma_{\mathrm{a}}\right)$. Fixing one noise parameter in the overt-criterion task ameliorated potential issues of lack of parameter identifiability [60], and ensured that models had the same complexity across tasks. The Bayes ${ }_{\text {ideal }}$, Fixed, and Behrens et al. (2007) models had no additional parameters. The following suboptimal Bayesian models had one additional parameter: Bayes ${ }_{r}$ $(r)$; Bayes ${ }_{\pi}\left(\pi_{\min }\right)$; Bayes ${ }_{\beta}(\beta)$. The Bayes ${ }_{r, \pi, \beta}$ model had 5 additional parameters $\left(r, \Delta r, \pi_{\min }, \Delta \pi\right.$, $\beta$ ). The Exp and RL models also only had one additional parameter $(\alpha)$, as did the Behrens et al. (2007) model with a bias towards equal probability $(w)$. The $\operatorname{Exp}_{\text {bias }}$ model had two additional parameters ( $\alpha_{\text {exp }}$ and $w$ ), and the Wilson et al. (2013) model had three ( $l_{2}, l_{3}$, and $v_{\mathrm{p}}$ ).

To fit each model, for each subject and task we computed the logarithm of the unnormalized posterior probability of the parameters,

$$
\log p^{*}(\boldsymbol{\theta} \mid \text { data })=\log p(\text { data } \mid \boldsymbol{\theta}, \mathrm{M})+\log p(\boldsymbol{\theta} \mid \mathrm{M})
$$

where data are category decisions in the covert-criterion task and criterion orientation in the overt-criterion task, $M$ is a specific model, and $\boldsymbol{\theta}$ represents the model parameters (generally, a vector). The first term of Eq 10 is the log likelihood, while the second term is the prior over parameters (see below).

For each model (except Bayes ${ }_{r, \pi, \beta}$ ) we evaluated Eq 10 on a cubic grid, with bounds chosen to contain almost all posterior probability mass. The grid for all models, except Wilson et al. (2013), consisted of 100 equally spaced values for each parameter. Due to the computational demands of the Wilson et al. (2013) model, we reduced the grid to 50 equally spaced values. The grid allowed us to approximate the full posterior distribution over parameters, and also to

---

#### Page 21

evaluate the normalization constant for the posterior, which corresponds to the evidence or marginal likelihood, used as a metric of model comparison (see Model comparison). We reported as parameter estimates the best-fitting model parameters on the grid, that is the maxi-mum-a-posteriori (MAP) values (see Tables 1 and 2). We used the full posterior distributions to compute posterior predictive distributions, that is, model predictions for visualization (see Model visualization), and to generate plausible parameter values for our model-recovery analysis. The Bayes ${ }_{v, \sigma, \beta}$ model had too many parameters to compute the marginal likelihood via brute force on a grid, so we adopted a different procedure for model comparison (see Model comparison).

Priors over parameters. We chose all priors to be uninformative. For noise parameters, we used uniform priors over a reasonably large range ( $\left[1^{\prime}, 30^{\prime}\right]$ ). For $\alpha_{\text {exp }}, \alpha_{\text {RL }}$, and $w$ we used a uniform prior from 0 to 1 . For $r$ we used a uniform prior from 2 to 200 trials. For $\pi_{\min }$ we used a uniform prior from 0 to 0.5 . For $\beta$, we used a uniform prior on the square root of the parameter value, ranging from 0 to 10 . Instead of fitting the individual nodes in the Wilson et al. (2013) model, we fit the difference between nodes, i.e., $\delta_{1}=l_{2}-l_{1}$ and $\delta_{2}=l_{3}-l_{2}$. We used a uniform prior on the square root of $\delta_{1}$ ranging from 1.01 to 5 and on the square root of $\delta_{2}$ ranging from 1.01 to 14 . Finally, for $v_{\mathrm{p}}$ we used a uniform prior in log space from 0 to 5 .

# Response probability

Covert-criterion task. For each model, parameter combination, observer, and trial in the covert-criterion task, we computed the probability of choosing category A on each trial given a stimulus, $s_{t}$, and all previously experienced categories, $\mathbf{C}_{1: t-1}$. In all models, the observer's current decision depends on the noisy measurement, $x_{t}$, so the probability of responding A for a given stimulus $s_{t}$ is

$$
p\left(\hat{C}_{\mathrm{A}} \mid s_{t}, \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)=\int p\left(\hat{C}_{\mathrm{A}} \mid x_{t}, \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right) \mathcal{N}\left(x_{t} \mid s_{t}, \sigma_{\mathrm{e}}^{2}\right) d x_{t}
$$

Because the current criterion setting in the RL model depends on the vector of all previous stimulus measurements, $\boldsymbol{x}_{1: t}$, the probability could not be computed analytically for this model. As an approximation, we used Monte Carlo simulations with 5000 sample measurement vectors. For each measurement vector, we applied the model's decision rule and approximated the probability by computing the proportion of times the model chose A out of all the simulations. For all models, we included a fixed lapse rate, $\lambda=10^{-4}$, that is the probability of a completely random response. The probability of choosing category $\mathrm{A}, \hat{C}_{\mathrm{A}}$, in the presence of lapses was then

$$
p\left(\hat{C}_{\mathrm{A}} \mid s_{t}, \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)=(1-\lambda) p\left(\hat{C}_{\mathrm{A}} \mid s_{t}, \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)+\frac{\lambda}{2}
$$

Effectively, the lapse rate acts as a regularization term that avoids excessive penalties to the likelihood of a model for outlier trials.

Next, assuming conditional independence between trials, we computed the log likelihood across all of the observer's choices, given each model and parameter combination

$$
\log p(\text { data } \mid \mathrm{M}, \theta)=\sum_{t=1}^{N_{\text {trials }}} \log p\left(\hat{C}_{\mathrm{A}} \mid s_{t}, \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)
$$

where $t$ is the trial index and $N_{\text {trials }}$ is the total number of trials.
Overt-criterion task. For each model, parameter combination, observer, and trial in the overt-criterion task, we computed the decision criterion on each trial. For all models except

---

#### Page 22

the RL model, the criterion was computed as in S1 Appendix. For the RL model, the criterion was computed as in Eq 7 for 5000 sample measurement vectors. For all models in the overt-criterion task, the criterion was corrupted by adjustment noise with variance $\sigma_{\alpha}^{2}$, so that $\hat{z}_{t} \sim \mathcal{N}\left(z_{t}, \sigma_{\alpha}^{2}\right)$, where $z_{t}$ was the observer's chosen criterion at trial $t$, and $\hat{z}_{t}$ was the actual reported criterion after adjustment noise. In addition, the observer had a chance of lapsing (e.g., a misclick), in which case the response was uniformly distributed in the range. Therefore, the probability that the observer reports the criterion $\hat{z}_{t}$ was

$$
p\left(\hat{z}_{t} \mid \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)=(1-\lambda) p\left(\hat{z}_{t} \mid \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)+\frac{\lambda}{180}
$$

with $\lambda=5 \times 10^{-5}$. As in the covert-criterion task, we computed the log likelihood across all trials by summing the log probability

$$
\log p(\text { data } \mid \mathrm{M}, \theta)=\sum_{t=1}^{T} \log p\left(\hat{z}_{t} \mid \mathbf{C}_{1: t-1}, \mathrm{M}, \theta\right)
$$

# Model comparison

To obtain a quantitative measure of model fit, for each observer, model (except Bayes $_{r, \pi, \beta}$ ), and task we computed the log marginal likelihood (LML) by integrating over the parameters in Eq 10,

$$
\log p(\text { data } \mid \mathrm{M})=\log \int p(\text { data } \mid \theta, \mathrm{M}) p(\theta \mid \mathrm{M}) d \theta
$$

To approximate the integral in Eq 16, we marginalized across each parameter dimension using the trapezoidal method. Assuming equal probability across models, the marginal likelihood is proportional to the posterior probability of a given model and thus represents a principled metric for comparison that automatically accounts for both goodness of fit and model complexity via Bayesian Occam's razor [61]. Penalizing for model complexity is a desirable feature of a model-comparison metric to reduce overfitting.

In addition to the Bayesian model-comparison metric described above, we computed the Akaike Information Criterion (AIC) [62] for each of our models. AIC is one of many information criteria that penalize the maximum log likelihood by a term that increases with the number of parameters. LML and AIC results were consistent (see S1 Appendix. for model comparison using AIC scores).

For comparison purposes, we report relative model-comparison scores, $\Delta$ LML and $\triangle$ AIC. We used bootstrapping to compute confidence intervals on the mean difference scores. Specifically, we simulated 10,000 sample data sets. For each simulated dataset we sampled, with replacement, 11 difference scores (the same number of difference scores as observers) and calculated the mean. To determine the $95 \% \mathrm{CI}$, we sorted the mean difference scores and determined the scores that corresponded to the 2.5 and 97.5 percentiles.

For an additional analysis at the group level, we used the random-effects Bayesian model selection analysis (BMS) developed by Stephan et al. [32] and expanded on by Rigoux et al. [33]. Specifically, using observers' LML scores we computed the protected exceedance probability $\phi$ and the posterior model frequency for each model. Exceedance probability represents the probability that one model is the most frequent decision-making strategy in the population, given the group data, above and beyond chance. This analysis was conducted using the open-source software package Statistical Parametric Mapping (SPM12; http://www.fil.ion.ucl. ac.uk/spm).

---

#### Page 23

To compare the Bayes $_{\mathrm{r}, \mathrm{r}, \beta}$ and $\operatorname{Exp}_{\text {bias }}$ models, we first fitted the data by maximum likelihood. For each of the two models, subject, and task we minimized the negative log likelihood of the data using Bayesian Adaptive Direct Search (BADS; https://github.com/lacerbi/bads [63]), taking the best result of 20 optimization runs with randomized starting points. Using the negative log likelihood from the fits, we computed the Bayesian Information Criterion (BIC), which penalizes the maximum log likelihood by a term that increases with the number of parameters in a similar way to AIC. We then estimated approximate lower bounds (ELBO) to the log marginal likelihood (and approximate posterior distributions, unused in our current analysis) via Variational Bayesian Monte Carlo (VBMC; https://github.com/lacerbi/vbmc [34]), taking the 'best' out of 10 variational optimization runs. VBMC combines variational inference and active-sampling Bayesian quadrature to perform approximate Bayesian inference in a sample-efficient manner. We validated the technique on the $\operatorname{Exp}_{\text {bias }}$ model, by comparing the ELBO obtained via VBMC and the log marginal likelihood calculated via numerical integration. For all subjects and task, the two metrics differed by $<0.2$ points. As a diagnostic of convergence for VBMC, for each problem instance we also verified that the majority of variational optimization runs returned an ELBO less than 1 point away from the 'best' solution we used in the analysis. For the few datasets that exhibited a slightly larger variability in the ELBOs across optimization runs, such variability was still much lower than the standard error across subjects of the $\triangle \mathrm{ELBO}$, thus not affecting the results of the model comparison.

# Model visualization

For each model, observer, and task, we randomly sampled 1000 parameter combinations from the joint posterior distribution with replacement. For each parameter combination, we simulated model responses using the same stimuli that were presented to the observer. Because the model output in the covert task was the probability of reporting category A, for each trial in a simulated dataset we simulated 10,000 model responses (i.e., category decisions), calculated the cumulative number of A's for each simulated dataset, and averaged the results. The mean and standard deviation were computed across all simulated datasets in both tasks. Model fit plots show the mean response (colored line) with shaded regions representing one standard deviation from the mean. Thus, shaded regions represent a $68 \%$ confidence interval on model fits.

## Model recovery

To ensure that our models were discriminable, we performed a model-recovery analysis, details of which can be found in S1 Appendix. In addition to the model-recovery analysis, we also performed a parameter-recovery analysis for the $\operatorname{Exp}_{\text {bias }}$ model. This was done to determine whether our parameter estimation procedure was biased for each parameter and task.

---

# RESEARCH ARTICLE - Appendix

---

## Supporting information

S1 Appendix. Supplementary information. Ideal observer model derivation; additional models; comparison of the Bayes $_{\mathrm{r}, \mathrm{r}, \beta}$ and $\operatorname{Exp}_{\text {bias }}$ models, model comparison with AIC; recovery analysis; measurement task; category training; individual model fits. (PDF)

## Acknowledgments

We would like to thank Chris Grimmick for helping with data collection. This work utilized the NYU IT High Performance Computing resources and services.

---

#### Page 24

# Author Contributions 

Conceptualization: Elyse H. Norton, Luigi Acerbi, Wei Ji Ma, Michael S. Landy.
Data curation: Elyse H. Norton.
Formal analysis: Elyse H. Norton, Luigi Acerbi, Wei Ji Ma, Michael S. Landy.
Funding acquisition: Michael S. Landy.
Methodology: Elyse H. Norton, Michael S. Landy.
Project administration: Michael S. Landy.
Software: Luigi Acerbi.
Supervision: Michael S. Landy.
Validation: Elyse H. Norton.
Visualization: Elyse H. Norton.
Writing - original draft: Elyse H. Norton, Luigi Acerbi.
Writing - review \& editing: Elyse H. Norton, Luigi Acerbi, Wei Ji Ma, Michael S. Landy.

## References

1. Bernardo J, Smith A. Bayesian theory. New York: Wiley; 1994.
2. Green D, Swets JA. Signal detection theory and psychophysics. New York: Wiley; 1966.
3. Tanner WP, Jr. Theory of recognition. J Acoust Soc Am. 1956; 28:882-888. https://doi.org/10.1121/1. 1908504
4. Ackermann JF, Landy MS. Suboptimal decision criteria are predicted by subjectively weighted probabilities and rewards. Atten Percept Psychophys. 2015; 77:638-658. https://doi.org/10.3758/s13414-014-0779-z PMID: 25366822
5. Ulehla ZJ. Optimality of perceptual decision criteria. J Exp Psychol. 1966; 71:564. https://doi.org/10. 1037/h0023007 PMID: 5909083
6. Healy AF, Kubovy M. Probability matching and the formation of conservative decision rules in a numerical analog of signal detection. J Exp Psychol Hum Learn. 1981; 7:344. https://doi.org/10.1037/02787393.7.5.344
7. Kubovy M, Healy AF. The decision rule in probabilistic categorization: What it is and how it is learned. J Exp Psychol Gen. 1977; 106:427. https://doi.org/10.1037/0096-3445.106.4.427
8. Healy AF, Kubovy M. The effects of payoffs and prior probabilities on indices of performance and cutoff location in recognition memory. Mem Cognit. 1978; 6:544-553. https://doi.org/10.3758/BF03198243 PMID: 24203388
9. Maddox WT. Toward a unified theory of decision criterion learning in perceptual categorization. J Exp Anal Behav. 2002; 78:567-595. https://doi.org/10.1901/jeab.2002.78-567 PMID: 12507020
10. Barron G, Erev I. Small feedback-based decisions and their limited correspondence to descriptionbased decisions. J Behav Decis Mak. 2003; 16:215-233. https://doi.org/10.1002/bdm. 443
11. Hertwig R, Barron G, Weber EU, Erev I. Decisions from experience and the effect of rare events in risky choice. Psychol Sci. 2004; 15:534-539. https://doi.org/10.1111/j.0956-7976.2004.00715.x PMID: 15270998
12. Bohil CJ, Wismer AJ. Implicit learning mediates base rate acquisition in perceptual categorization. Psychon Bull Rev. 2015; 22:586-593. https://doi.org/10.3758/s13423-014-0694-2 PMID: 25037267
13. Wismer AJ, Bohil CJ. Base-rate sensitivity through implicit learning. PLoS One. 2017; 12(6):e0179256. https://doi.org/10.1371/journal.pone.0179256 PMID: 28632779
14. Berniker M, Voss M, Körding K. Learning priors for Bayesian computations in the nervous system. PLoS One. 2010; 5(9):e12686. https://doi.org/10.1371/journal.pone.0012686 PMID: 20844766
15. Summerfield C, Behrens TE, Koechlin E. Perceptual classification in a rapidly changing environment. Neuron. 2011; 71:725-736. https://doi.org/10.1016/j.neuron.2011.06.022 PMID: 21867887

---

#### Page 25

16. Norton EH, Fleming SM, Daw ND, Landy MS. Suboptimal criterion learning in static and dynamic environments. PLoS Comput Biol. 2017; 13(1):e1005304. https://doi.org/10.1371/journal.pcbi. 1005304 PMID: 28046006
17. Nassar MR, Wilson RC, Heasly B, Gold JI. An approximately Bayesian delta-rule model explains the dynamics of belief updating in a changing environment. J Neurosci. 2010; 30:12366-12378. https://doi. org/10.1523/JNEUROSCI.0822-10.2010 PMID: 20844132
18. Landy MS, Trommershäuser J, Daw ND. Dynamic estimation of task-relevant variance in movement under risk. J Neurosci. 2012; 32:12702-12711. https://doi.org/10.1523/JNEUROSCI.6160-11.2012 PMID: 22972994
19. Acerbi L, Wolpert DM, Vijayakumar S. Internal representations of temporal statistics and feedback calibrate motor-sensory interval timing. PLoS Comput Biol. 2012; 8(11):e1002771. https://doi.org/10.1371/ journal.pcbi. 1002771 PMID: 23209386
20. Sato Y, Körding KP. How much to trust the senses: Likelihood learning. J Vis. 2014; 14(13):13. https:// doi.org/10.1167/14.13.13 PMID: 25398975
21. Glaze CM, Kable JW, Gold JI. Normative evidence accumulation in unpredictable environments. eLife. 2015; 4. https://doi.org/10.7554/eLife.08825 PMID: 26322383
22. Wilson RC, Nassar MR, Gold JI. Bayesian online learning of the hazard rate in change-point problems. Neural Comput. 2010; 22:2452-2476. https://doi.org/10.1162/NECO_a_00007 PMID: 20569174
23. Meyniel F, Schlunegger D, Dehaene S. The sense of confidence during probabilistic learning: A normative account. PLoS Comput Biol. 2015; 11(6):e1004305. https://doi.org/10.1371/journal.pcbi. 1004305 PMID: 26076466
24. Behrens TE, Woolrich MW, Walton ME, Rushworth MF. Learning the value of information in an uncertain world. Nat Neurosci. 2007; 10:1214. https://doi.org/10.1038/nn1954 PMID: 17676057
25. Zylberberg A, Wolpert DM, Shadlen MN. Counterfactual reasoning underlies the learning of priors in decision making. Neuron. 2018; 99(5):1083-1097. https://doi.org/10.1016/j.neuron.2018.07.035 PMID: 30122376
26. Tversky A, Kahneman D. Judgment under uncertainty: Heuristics and biases. Science. 1974; 185:1124-1131. https://doi.org/10.1126/science.185.4157.1124 PMID: 17835457
27. Adams RP, MacKay DJ. Bayesian online changepoint detection. arXiv preprint arXiv:07103742. 2007.
28. Gallistel CR, Krishan M, Liu Y, Miller RR, Latham PE. The perception of probability. Psychol Rev. 2014; 121:96-123. https://doi.org/10.1037/a0035232 PMID: 24490790
29. Rescorla RA, Wagner AR. A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In: Black AH, Prokasy WF, editors. Classical conditioning II: Current research and theory. New York: Appleton-Century-Crofts; 1972. p. 64-99.
30. Wilson RC, Nassar MR, Gold JI. A mixture of delta-rules approximation to Bayesian inference in change-point problems. PLoS Comput Biol. 2018; 14(6):e1006210. https://doi.org/10.1371/journal.pcbi. 1003150 PMID: 23935472
31. Wilson RC, Nassar MR, Gold JI. Correction: A mixture of delta-rules approximation to Bayesian inference in change-point problems. PLoS Comput Biol. 2013; 9(7):e1003150. https://doi.org/10.1371/ journal.pcbi. 1006210 PMID: 29944654
32. Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ. Bayesian model selection for group studies. Neuroimage. 2009; 46:1004-1017. https://doi.org/10.1016/j.neuroimage.2009.03.025 PMID: 19306932
33. Rigoux L, Stephan KE, Friston KJ, Daunizeau J. Bayesian model selection for group studies-revisited. Neuroimage. 2014; 84:971-985. https://doi.org/10.1016/j.neuroimage.2013.08.065 https://www.ncbi. nlm.nih.gov/pubmed/24018303
34. Acerbi L. Variational Bayesian Monte Carlo. In: Advances in Neural Information Processing Systems. vol. 31; 2018. p. 8213-8223.
35. Burge J, Ernst MO, Banks MS. The statistical determinants of adaptation rate in human reaching. J Vis. 2008; 8(4):20. https://doi.org/10.1167/8.4.20 PMID: 18484859
36. Qamar AT, Cotton RJ, George RG, Beck JM, Prezhdo E, Laudano A, et al. Trial-to-trial, uncertaintybased adjustment of decision boundaries in visual categorization. Proc Natl Acad Sci USA. 2013; 110:20332-20337. https://doi.org/10.1073/pnas.1219756110 PMID: 24272938
37. Heilbron M, Meyniel F. Confidence resets reveal hierarchical adaptive learning in humans. PLoS Comput Biol. 2019; 15(4). https://doi.org/10.1371/journal.pcbi. 1006972 PMID: 30964861
38. Acerbi L, Dokka K, Angelaki DE, Ma WJ. Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS Comput Biol. 2018; 14(7):e1006110. https://doi. org/10.1371/journal.pcbi. 1006110 PMID: 30052625

---

#### Page 26

39. Beck JM, Ma WJ, Pitkow X, Latham PE, Pouget A. Not noisy, just wrong: The role of suboptimal inference in behavioral variability. Neuron. 2012; 74(1):30-39. https://doi.org/10.1016/j.neuron.2012.03.016 PMID: 22500627
40. Acerbi L, Vijayakumar S, Wolpert DM. On the origins of suboptimality in human probabilistic inference. PLoS Comput Biol. 2014; 10(6):e1003661. https://doi.org/10.1371/journal.pcbi.1003661 PMID: 24945142
41. Drugowitsch J, DeAngelis GC, Angelaki DE, Pouget A. Tuning the speed-accuracy trade-off to maximize reward rate in multisensory decision-making. eLife. 2015; 4:e06678. https://doi.org/10.7554/eLife. 06678 PMID: 26090907
42. Izawa J, Shadmehr R. Learning from sensory and reward prediction errors during motor adaptation. PLoS Comput Biol. 2011; 7(3):e1002012. https://doi.org/10.1371/journal.pcbi.1002012 PMID: 21423711
43. Atmeave F. Psychological probability as a function of experienced frequency. J Exp Psychol. 1953; 46:81. https://doi.org/10.1037/h0057955 PMID: 13084849
44. Varey CA, Mellers BA, Birnbaum MH. Judgments of proportions. J Exp Psychol Hum Percept Perform. 1990; 16:613. https://doi.org/10.1037/0096-1523.16.3.613 PMID: 2144575
45. Maloney LT, Thomas EA. Distributional assumptions and observed conservatism in the theory of signal detectability. J Math Psychol. 1991; 35:443-470. https://doi.org/10.1016/0022-2496(91)90043-S
46. Kubovy M. A possible basis for conservatism in signal detection and probabilistic categorization tasks. Percept Psychophys. 1977; 22:277-281. https://doi.org/10.3758/BF03199690
47. Lee W, Janke M. Categorizing externally distributed stimulus samples for unequal molar probabilities. Psychol Rep. 1965; 17:79-90. https://doi.org/10.2466/pr0.1965.17.1.79 PMID: 5826504
48. Murray RF, Patel K, Yee A. Posterior probability matching and human perceptual decision making. PLoS Comput Biol. 2015; 11(6):e1004342. https://doi.org/10.1371/journal.pcbi.1004342 PMID: 26079134
49. Thomas EA, Legge D. Probability matching as a basis for detection and recognition decisions. Psychol Rev. 1970; 77:65. https://doi.org/10.1037/h0028579
50. Wozny DR, Beierholm UR, Shams L. Probability matching as a computational strategy used in perception. PLoS Comput Biol. 2010; 6(8):e1000871. https://doi.org/10.1371/journal.pcbi.1000871 PMID: 20700493
51. van den Berg R, Awh E, Ma WJ. Factorial comparison of working memory models. Psychol Rev. 2014; 121:124. https://doi.org/10.1037/a0035234 PMID: 24490791
52. Gifford AM, Cohen YE, Stocker AA. Characterizing the impact of category uncertainty on human auditory categorization behavior. PLoS Comput Biol. 2014; 10(7):e1003715. https://doi.org/10.1371/journal. pcbi. 1003715 PMID: 25032683
53. Petzschner FH, Glasauer S. Iterative Bayesian estimation as an explanation for range and regression effects: A study on human path integration. J Neurosci. 2011; 31:17220-17229. https://doi.org/10.1523/ JNEUROSCI.2028-11.2011 PMID: 22114288
54. Raviv O, Ahissar M, Loewenstein Y. How recent history affects perception: The normative approach and its heuristic approximation. PLoS Comput Biol. 2012; 8(10):e1002731. https://doi.org/10.1371/ journal.pcbi. 1002731 PMID: 23133343
55. Fischer J, Whitney D. Serial dependence in visual perception. Nat Neurosci. 2014; 17:738-743. https:// doi.org/10.1038/nn. 3689 PMID: 24686785
56. Zylberberg A, Roelfsema PR, Sigman M. Variance misperception explains illusions of confidence in simple perceptual decisions. Conscious Cogn. 2014; 27:246-253. https://doi.org/10.1016/j.concog. 2014.05.012 PMID: 24951943
57. MATLAB. version 7.10.0 (R2010a). Natick, Massachusetts: The MathWorks Inc.; 2010.
58. Brainard DH, Vision S. The psychophysics toolbox. Spat Vis. 1997; 10:433-436. https://doi.org/10. 1163/156856897X00357 PMID: 9176952
59. Pelli DG. The VideoToolbox software for visual psychophysics: Transforming numbers into movies. Spat Vis. 1997; 10:437-442. https://doi.org/10.1163/156856897X00366 PMID: 9176953
60. Acerbi L, Ma WJ, Vijayakumar S. A framework for testing identifiability of Bayesian models of perception. In: Advances in Neural Information Processing Systems; 2014. p. 1026-1034
61. MacKay DJ. Information Theory, Inference and Learning Algorithms. Cambridge, England: Cambridge University Press; 2003.
62. Akaike H. Information theory and an extension of the maximum likelihood principle. In: Proceedings of the 2nd International Symposium on Information; 1973. p. 267-281.
63. Acerbi L, Ma WJ. Practical Bayesian optimization for model fitting with Bayesian Adaptive Direct Search. In: Advances in Neural Information Processing Systems. vol. 30; 2017. p. 1836-1846.

---

#### Page 1

# Human online adaptation to changes in prior probability - Appendix S1

Elyse H. Norton<br>elyse.norton@nyu.edu<br>Luiqi Acerbi<br>luigi.acerbi@nyu.edu<br>Wei Ji Ma<br>weijima@nyu.edu<br>Michael S. Landy<br>landy@nyu.edu

May 27, 2019

## Contents

1 Ideal observer model derivation ..... 2
1.1 Bayesian online change-point detection ..... 2
1.1.1 Conditional predictive probability ..... 4
1.1.2 The change-point posterior ..... 5
1.1.3 Iterative posterior update and boundary conditions ..... 6
1.2 Task-dependent predictive distributions ..... 6
1.2.1 Covert-criterion task ..... 6
1.2.2 Overt-criterion task ..... 7
1.3 Algorithm ..... 8
2 Additional models ..... 10
2.1 Bayesian ..... 10
2.2 Reinforcement learning - probability updating ..... 11
2.3 Wilson et al. (2013) ..... 11
3 Comparison of the Bayes ${ }_{c \pi, \beta}$ and the $\operatorname{Exp}_{\text {bias }}$ models ..... 12
4 Model comparison with AIC ..... 12

---

#### Page 2

5 Recovery analysis ..... 15
5.1 Model recovery ..... 15
5.2 Parameter recovery ..... 15
6 Measurement task ..... 16
6.1 Procedure ..... 16
6.2 Analysis ..... 16
6.3 Results ..... 17
7 Category training ..... 17
7.1 Procedure ..... 17
7.2 Results ..... 18
8 Individual model fits ..... 19

# 1 Ideal observer model derivation

Our derivation of the Bayesian online change-point detection algorithm for the ideal observer generalizes that of Adams and MacKay [1]. For clarity and ease of reference, we report here the full derivation; only the broad outline is described in the main text.

### 1.1 Bayesian online change-point detection

The Bayesian observer estimates the posterior distribution over the current run length, or time since the last change point, and the state (category probabilities) before the last change point, given the data (category labels) observed so far. We denote the length of the run at the end of trial $t$ by $r_{t}$. Similarly, we denote with $\pi_{t}$ and $\xi_{t}$ the current state and the state before the last change point, both measured at the end of trial $t$. Here, $\pi_{t}$ represents the probability that, on the subsequent trial, the category will be A (the probability of category B is $1-\pi_{t}$ ). Both $\pi_{t}, \xi_{t} \in S_{\pi}$, where $S_{\pi}$ is a discrete set of possible states. In the experiment, $S_{\pi}=\{0.2,0.35,0.5,0.65,0.8\}$. We use the notation $\boldsymbol{C}_{t}^{(r)}$ to indicate the set of observations (category labels) associated with the run $r_{t}$, which is $\boldsymbol{C}_{t-r_{t}+1: t}$ for $r_{t}>0$, and $\emptyset$ for $r_{t}=0$. We use the subscript colon operator $\boldsymbol{C}_{t^{\prime}: t}$ to denote the sub-vector of $\boldsymbol{C}$ (the full sequence of observed categories) with elements from $t^{\prime}$ to $t$ included.

---

#### Page 3

We write the predictive distribution of category by marginalizing over run lengths $r_{t}$ and previous states $\xi_{t}$,

$$
P\left(C_{t+1} \mid \boldsymbol{C}_{1: t}\right)=\sum_{r_{t}} \sum_{\xi_{t}} P\left(C_{t+1} \mid r_{t}, \xi_{t}, \boldsymbol{C}_{t}^{(r)}\right) P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)
$$

We assume that, in the case of a change point at the end of trial $t$, the new state might have Markovian dependence on the previous state, that is $\pi_{t} \sim P\left(\pi_{t} \mid \pi_{t-1}\right)$. This is a generalization of the model of Adams and MacKay [1], in which the distribution parameters were assumed to be independent after change points. In the experiment, $P\left(\pi_{t} \mid \pi_{t-1}\right)=\frac{1}{\left|\pi_{t}\right|-1} \llbracket \pi_{t} \neq \pi_{t-1} \rrbracket$. We use $\llbracket A \rrbracket$ to denote Iverson's bracket which is 1 if the expression $A$ is true, and 0 otherwise [2].

To find the posterior distribution (the second term in Eq S1)

$$
P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)=\frac{P\left(r_{t}, \xi_{t}, \boldsymbol{C}_{1: t}\right)}{P\left(\boldsymbol{C}_{1: t}\right)}
$$

we write the joint distribution over run length, previous state and observed data recursively,

$$
\begin{aligned}
P\left(r_{t}, \xi_{t}, \boldsymbol{C}_{1: t}\right)= & \sum_{r_{t-1}} \sum_{\xi_{t-1}} P\left(r_{t}, r_{t-1}, \xi_{t}, \xi_{t-1}, \boldsymbol{C}_{1: t}\right) \\
= & \sum_{r_{t-1}} \sum_{\xi_{t-1}} P\left(r_{t}, \xi_{t}, C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{1: t-1}\right) P\left(r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{1: t-1}\right) \\
= & \sum_{r_{t-1}} \sum_{\xi_{t-1}} P\left(r_{t}, \xi_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right) \\
& \quad \times P\left(C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) P\left(r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{1: t-1}\right)
\end{aligned}
$$

Note that the justification for specializing from $\boldsymbol{C}_{1: t}$ to $\boldsymbol{C}_{t}^{(r)}$ and $\boldsymbol{C}_{t-1}^{(r)}$ will become clear in the derivations below. We can rewrite Eq S3 in terms of the posterior distribution as

$$
\begin{aligned}
P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)= & \frac{1}{P\left(\boldsymbol{C}_{1: t}\right)} P\left(r_{t}, \xi_{t}, \boldsymbol{C}_{1: t}\right) \\
= & \frac{P\left(\boldsymbol{C}_{1: t-1}\right)}{P\left(\boldsymbol{C}_{1: t}\right)} \sum_{r_{t-1}} \sum_{\xi_{t-1}} P\left(r_{t}, \xi_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right) \\
& \quad \times P\left(C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) P\left(r_{t-1}, \xi_{t-1} \mid \boldsymbol{C}_{1: t-1}\right)
\end{aligned}
$$

---

#### Page 4

Eq S4 is the basis for the iterative Bayesian algorithm, since it allows us to derive the posterior distribution at time $t$ as a function of the posterior distribution and a number of auxiliary variables at time $t-1$.

For computational convenience, we rewrite the posterior from Eq S4 as an unnormalized posterior

$$
U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)=\sum_{r_{t-1}} \sum_{\xi_{t-1}} P\left(r_{t}, \xi_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right) P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)}
$$

where we introduced $P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)}$ to denote the posterior from the previous trial times the conditional predictive probability for the current category,

$$
P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)} \equiv P\left(C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) P\left(r_{t-1}, \xi_{t-1} \mid \boldsymbol{C}_{1: t-1}\right)
$$

To compute the unnormalized posterior in Eq S5, we need:

- the conditional predictive probability, which we compute in the following Section 1.1.1;
- the change-point posterior, which we compute in Section 1.1.2;
- the posterior from the previous trial.

We put everything together in Section 1.1.3.

# 1.1.1 Conditional predictive probability

The posterior over state at the end of trial $t-1$, given the last $r_{t-1}$ trials and the previous state $\xi_{t-1}$, is

$$
P\left(\pi_{t-1} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) \propto\left\|\pi_{t-1} \neq \xi_{t-1}\right\| P\left(\pi_{t-1} \mid r_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right)
$$

For computational convenience, we denote $\Psi_{t}^{(r, \pi)} \equiv P\left(\pi_{t}=\pi \mid r_{t}=r, \boldsymbol{C}_{t}^{(r)}\right)$ and we store it in a table. Clearly, $\Psi_{t}^{(r, \pi)}$ depends only on the length of the run $r$, the category probability $\pi$ and the number of times category A occurs during the run. In the algorithm below, $\Psi$ is computed iteratively trial-by-trial and values of $\Psi$ are computed only for combinations of run length and number of A categories that occur in the sequence. The conditional predictive probability for observing $C_{t}$, using Eq S7, is

$$
\begin{aligned}
P\left(C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) & =\sum_{\pi_{t-1}} P\left(C_{t} \mid \pi_{t-1}\right) P\left(\pi_{t-1} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) \\
& \propto \sum_{\pi_{t-1}} P\left(C_{t} \mid \pi_{t-1}\right)\left\|\pi_{t-1} \neq \xi_{t-1}\right\| \Psi_{t-1}^{\left(r_{t-1}, \pi_{t-1}\right)}
\end{aligned}
$$

---

#### Page 5

# 1.1.2 The change-point posterior

The conditional posterior on the change point (that is, run length) and previous state, $P\left(r_{t}, \xi_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right)$, has a sparse representation since it has only two outcomes: the run length either continues to grow ( $r_{t}=r_{t-1}+1$ and $\xi_{t}=\xi_{t-1}$ ) or a change point occurs ( $r_{t}=0$ and the posterior over $\xi_{t}$ is the posterior over $\pi_{t-1}$ computed from $\left.\boldsymbol{C}_{t}^{(r)}\right)$. We have

$$
P\left(r_{t}, \xi_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right)=P\left(r_{t} \mid r_{t-1}\right) P\left(\xi_{t} \mid r_{t}, r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right)
$$

The probability of a run length after a change point is

$$
P\left(r_{t} \mid r_{t-1}\right)= \begin{cases}H\left(r_{t-1}+1\right) & \text { if } r_{t}=0 \\ 1-H\left(r_{t-1}+1\right) & \text { if } r_{t}=r_{t-1}+1 \\ 0 & \text { otherwise }\end{cases}
$$

where the function $H(\tau)$ is the hazard function,

$$
H(\tau)=\frac{P_{\text {gap }}(g=\tau)}{\sum_{t=\tau}^{\infty} P_{\text {gap }}(g=t)}, \text { for } \tau \geq 1
$$

$P_{\text {gap }}$ is the prior over run lengths. In our experimental setup, $P_{\text {gap }}(g)=$ $\frac{1}{g_{\max }-g_{\min }+1} \llbracket g_{\min } \leq g \leq g_{\max } \rrbracket$, with $g_{\min }=80$ and $g_{\max }=120$.

The conditional posterior over the previous state is

$$
P\left(\xi_{t} \mid r_{t}, r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right)= \begin{cases}P\left(\pi_{t-1}=\xi_{t} \mid r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right) & \text { if } r_{t}=0 \\ \delta\left(\xi_{t}-\xi_{t-1}\right) & \text { otherwise }\end{cases}
$$

where $P\left(\pi_{t-1} \mid r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right)$ is the posterior over state given that $C_{t}$ has just been observed,

$$
\begin{aligned}
P\left(\pi_{t-1} \mid r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right) & \propto P\left(\pi_{t-1}, r_{t}, r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right) \\
& \propto P\left(C_{t} \mid \pi_{t-1}\right) P\left(\pi_{t-1} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) P\left(r_{t} \mid r_{t-1}\right) \\
& \propto P\left(C_{t} \mid \pi_{t-1}\right) \llbracket \pi_{t-1} \neq \xi_{t-1} \rrbracket \Psi_{t-1}^{\left(r_{t-1}, \pi_{t-1}\right)}
\end{aligned}
$$

where in the last step $P\left(r_{t} \mid r_{t-1}\right)$ is constant in $\pi_{t-1}$ and therefore irrelevant.

---

#### Page 6

# 1.1.3 Iterative posterior update and boundary conditions

Using Eqs S5 and S9, the iterative posterior update equation becomes

$$
U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)=\sum_{r_{t-1}} P\left(r_{t} \mid r_{t-1}\right) \sum_{\xi_{t-1}} P\left(\xi_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t}^{(r)}\right) P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)}
$$

which is computed separately for the case $r_{t}=0$ and $r_{t}>0$ via Eqs S7-S13.
We assume as boundary conditions that a change point just occurred and uniform probability across previous states

$$
P\left(r_{0}, \xi_{0} \mid \emptyset\right)=\delta\left(r_{0}\right) \frac{1}{\left|S_{\#}\right|}
$$

Once we have $U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)$, we can easily obtain the normalized posterior $P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)$ by computing the normalization constant via a discrete summation over run lengths $r_{t}$ and previous states $\xi_{t}$,

$$
P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)=\frac{U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)}{\mathcal{Z}\left(\boldsymbol{C}_{1: t}\right)}, \quad \text { with } \mathcal{Z}\left(\boldsymbol{C}_{1: t}\right)=\sum_{r_{t}} \sum_{\xi_{t}} U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)
$$

This result together with Eq S1 allows the observer to compute $P\left(C_{t+1} \mid \boldsymbol{C}_{1: t}\right)$.

### 1.2 Task-dependent predictive distributions

Armed with an expression for the observer's posterior distribution over run lengths and previous states, given all trials experienced so far (Eq S16), we can now compute the predictive distributions for the observer's response at trial $t$ for the covert- and overt-criterion tasks.

### 1.2.1 Covert-criterion task

The probability density of a noisy measurement $x_{t}$ is

$$
p\left(x_{t} \mid C_{t}\right)=\mathcal{N}\left(x_{t} \mid \mu_{C_{t}}, \sigma^{2}\right)
$$

with $\sigma^{2} \equiv \sigma_{\mathrm{v}}^{2}+\sigma_{\mathrm{s}}^{2}$, where $\sigma_{\mathrm{v}}^{2}$ is the observer's visual measurement noise and $\sigma_{s}^{2}$ is the stimulus variance. The conditional posterior for category $C_{t}$, after observing $x_{t}$, is

$$
P\left(C_{t} \mid x_{t}, \boldsymbol{C}_{1: t-1}\right)=\frac{P\left(x_{t} \mid C_{t}\right) P\left(C_{t} \mid \boldsymbol{C}_{1: t-1}\right)}{\sum_{C_{t}^{\prime}} P\left(x_{t} \mid C_{t}^{\prime}\right) P\left(C_{t}^{\prime} \mid \boldsymbol{C}_{1: t-1}\right)}
$$

---

#### Page 7

We assume that for a given noisy measurement the observer responds $\hat{C}_{t}$ if that category is more probable, that is,

$$
P\left(\hat{C}_{t} \mid x_{t}, \boldsymbol{C}_{1: t-1}\right)=\llbracket P\left(C_{t} \mid x_{t}, \boldsymbol{C}_{1: t-1}\right)>0.5 \rrbracket
$$

The probability of observing response $\hat{C}_{t}$ for a given stimulus $s_{t}$ is therefore

$$
P\left(\hat{C}_{t} \mid s_{t}, \boldsymbol{C}_{1: t-1}\right)=\int P\left(\hat{C}_{t} \mid x_{t}, \boldsymbol{C}_{1: t-1}\right) \mathcal{N}\left(x_{t} \mid s_{t}, \sigma_{\mathrm{v}}^{2}\right) d x_{t}
$$

which can be easily computed via 1-D numerical integration over a grid of regularly spaced $x_{t}$ using trapezoidal or Simpson's rule [3].

We can also consider an observer model with lapses that occasionally reports the wrong category with probability $0 \leq \lambda \leq 1$,

$$
P_{\text {lapse }}\left(\hat{C}_{t} \mid s_{t}, \boldsymbol{C}_{1: t-1}\right)=(1-\lambda) P\left(\hat{C}_{t} \mid s_{t}, \boldsymbol{C}_{1: t-1}\right)+\frac{\lambda}{|C|}
$$

where $|C|$ is the number of categories in the task $(|C|=2$ in our case $)$, and we assume equal response probability across categories for lapses.

# 1.2.2 Overt-criterion task

The optimal criterion $z_{\text {opt }}$ is the point at which $P\left(C_{\mathrm{A}} \mid x, t\right)=P\left(C_{\mathrm{B}} \mid x, t\right)$, given the available information at trial $t$. Specifically, noting that $P\left(C, \pi_{t} \mid x\right) \propto$ $P\left(x \mid C\right) P\left(C \mid \pi_{t}\right) P\left(\pi_{t}\right)$, we have

$$
\begin{aligned}
\sum_{\pi_{t}} P\left(x \mid C_{\mathrm{A}}\right) P\left(C_{\mathrm{A}} \mid \pi_{t}\right) P\left(\pi_{t}\right) & =\sum_{\pi_{t}} P\left(x \mid C_{\mathrm{B}}\right) P\left(C_{\mathrm{B}} \mid \pi_{t}\right) P\left(\pi_{t}\right) \\
P\left(x \mid C_{\mathrm{A}}\right) \sum_{\pi_{t}} \pi_{t} P\left(\pi_{t}\right) & =P\left(x \mid C_{\mathrm{B}}\right) \sum_{\pi_{t}}\left(1-\pi_{t}\right) P\left(\pi_{t}\right) \\
\frac{\sum_{\pi_{t}} \pi_{t} P\left(\pi_{t}\right)}{\sum_{\pi_{t}}\left(1-\pi_{t}\right) P\left(\pi_{t}\right)} & =e^{-\frac{\left(x-\mu_{\mathrm{B}}\right)^{2}}{2 \sigma^{2}}+\frac{\left(x-\mu_{\mathrm{A}}\right)^{2}}{2 \sigma^{2}}} \\
\Longrightarrow z_{t}^{\mathrm{opt}} & =\frac{\sigma^{2} \log \Gamma_{t}}{\mu_{\mathrm{B}}-\mu_{\mathrm{A}}}+\frac{1}{2}\left(\mu_{\mathrm{A}}+\mu_{\mathrm{B}}\right)
\end{aligned}
$$

where we have defined $\Gamma_{t}=\frac{\sum_{\pi_{t}} \pi_{t} P\left(\pi_{t}\right)}{\sum_{\pi_{t}}\left(1-\pi_{t}\right) P\left(\pi_{t}\right)}$. We assume that $\mu_{\mathrm{A}}$ and $\mu_{\mathrm{B}}$ are known exactly from the training session.

The probability that the observer reports criterion $\hat{z}_{t}$ at trial $t$ is

$$
P\left(\hat{z}_{t} \mid \boldsymbol{C}_{1: t-1}\right)=\mathcal{N}\left(\hat{z}_{t} \mid z_{t}^{\mathrm{opt}}, \sigma_{\mathrm{a}}^{2}\right)
$$

---

#### Page 8

where $\sigma_{\mathrm{a}}^{2}$ is criterion-placement (adjustment) noise. Note that the likelihood at trial $t$ is based on information gathered through trial $t-1$.

We can also consider an observer model with lapses who occasionally reports a criterion uniformly at random with probability $0 \leq \lambda \leq 1$,

$$
P_{\text {lapse }}\left(\hat{z}_{t} \mid \boldsymbol{C}_{1: t-1}\right)=(1-\lambda) P\left(\hat{z}_{t} \mid \boldsymbol{C}_{1: t-1}\right)+\frac{\lambda}{180}
$$

# 1.3 Algorithm

In the following, we use the notation $P(\boldsymbol{x} \mid \boldsymbol{y}) \propto f(\boldsymbol{x}, \boldsymbol{y})$ to indicate that the user needs to compute $f(\boldsymbol{x}, \boldsymbol{y})$ and then normalize as follows, $P(\boldsymbol{x} \mid \boldsymbol{y})=$ $\frac{f(\boldsymbol{x}, \boldsymbol{y})}{\sum_{\boldsymbol{x}} f(\boldsymbol{x} \boldsymbol{\prime}, \boldsymbol{y})}$.

1. Initialize
   (a) Posterior $P\left(r_{0}, \xi_{0} \mid \emptyset\right)=\delta\left(r_{0}\right) \frac{1}{\left|S_{\pi}\right|}$
   (b) Lookup table $\Psi^{\left(r_{0}, \pi_{0}\right)}=\delta\left(r_{0}\right) \frac{1}{\left|S_{\pi}\right|}$
   (c) Set trial $t=1$
2. Observe new category $C_{t}$
3. Compute auxiliary variables
   (a) Evaluate predictive probability (Eq S8)

$$
P\left(C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) \propto \sum_{\pi_{t-1}} P\left(C_{t} \mid \pi_{t-1}\right)\left\|\pi_{t-1} \neq \xi_{t-1}\right\| \Psi^{\left(r_{t-1}, \pi_{t-1}\right)}
$$

(b) Evaluate the predictive probability times posterior probability (Eq S6)

$$
P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)}=P\left(C_{t} \mid r_{t-1}, \xi_{t-1}, \boldsymbol{C}_{t-1}^{(r)}\right) P\left(r_{t-1}, \xi_{t-1} \mid \boldsymbol{C}_{1: t-1}\right)
$$

(c) Evaluate the posterior probability over state (from Eq S13)

$$
P\left(\pi_{t-1} \mid r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right) \propto P\left(C_{t} \mid \pi_{t-1}\right)\left\|\pi_{t-1} \neq \xi_{t-1}\right\| \Psi^{\left(r_{t-1}, \pi_{t-1}\right)}
$$

4. Update run length and previous-state posterior

---

#### Page 9

(a) Calculate the unnormalized change-point probabilities (Eq S14)

$$
\begin{aligned}
U\left(r_{t}\right. & \left.=0, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)=\sum_{r_{t-1}} H\left(r_{t-1}+1\right) \\
& \times \sum_{\xi_{t-1}} P\left(\pi_{t-1}=\xi_{t} \mid r_{t-1}, \xi_{t-1}, C_{t}, \boldsymbol{C}_{t-1}^{(r)}\right) P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)}
\end{aligned}
$$

(b) Calculate the unnormalized growth probabilities (see Eq S14)

$$
U\left(r_{t}=r_{t-1}+1, \xi_{t}=\xi_{t-1} \mid \boldsymbol{C}_{1: t}\right)=\left[1-H\left(r_{t-1}+1\right)\right] P_{t}^{\left(r_{t-1}, \xi_{t-1}\right)}
$$

(c) Calculate the normalization (Eq S16)

$$
\mathcal{Z}\left(\boldsymbol{C}_{1: t}\right)=\sum_{r_{t}} \sum_{\xi_{t}} U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)
$$

(d) Determine the posterior distribution (Eq S16)

$$
P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)=\frac{U\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)}{\mathcal{Z}\left(\boldsymbol{C}_{1: t}\right)}
$$

5. Bookkeeping and predictions
   (a) Update sufficient statistics for all $r$ and $\pi$

$$
\begin{aligned}
& \tilde{\Psi}_{t}^{(r, \pi)}= \begin{cases}1 & \text { if } r=0 \\
\Psi_{t-1}^{(r, \pi)} P\left(C_{t} \mid \pi_{t-1}=\pi\right) & \text { if } r>0\end{cases} \\
& \Psi_{t}^{(r, \pi)}=\frac{\tilde{\Psi}_{t}^{(r, \pi)}}{\sum_{\pi \prime} \tilde{\Psi}_{t}^{(r, \pi \prime)}}
\end{aligned}
$$

(b) Compute the predictive distribution of category (Eq S1)
(c) Store predictive posterior over $\pi_{t}$

$$
P\left(\pi_{t} \mid \boldsymbol{C}_{1: t}\right) \propto \sum_{r_{t}} \sum_{\xi_{t}} \Psi_{t}^{\left(r_{t}, \pi_{t}\right)} \llbracket \pi_{t} \neq \xi_{t} \rrbracket P\left(r_{t}, \xi_{t} \mid \boldsymbol{C}_{1: t}\right)
$$

6. Increase trial index $t \leftarrow t+1$ and return to step 2

For each trial $t$, the posterior predictive distributions calculated in steps 5 b and 5 c are used to compute the observer's response probabilities in the covertand overt-criterion tasks, respectively, as described in Section 1.2.

---

#### Page 10

# 2 Additional models

We describe here a number of model variants which we did not include in the main text for reasons of space. For all additional models, we report as a model comparison metric the difference in log marginal likelihood ( $\Delta \mathrm{LML}$ ) with respect to a baseline model (mean $\pm$ SEM across subjects). Usually, unless stated otherwise, we take as baseline the best-fitting model described in the main text, $\operatorname{Exp}_{\text {bias }}$. Positive values of $\Delta \mathrm{LML}$ denote a worse-fitting model than baseline.

### 2.1 Bayesian

The main text discusses four Bayesian models. Bayes ${ }_{\text {ideal }}$ is the algorithm above, using the precise generative model for our experiment. Bayes ${ }_{r}$ uses the same algorithm, but adds a free parameter for the run-length distribution (and hence the hazard function) assumed by the observer. Bayes ${ }_{\pi}$ also uses the same algorithm, but adds a parameter for the range of the set of five states assumed by the observer. Bayes ${ }_{\beta}$ assumes the observer uses a betadistributed prior over states. To implement this observer requires minor modifications of the algorithm. In particular, the beta prior is substituted for the uniform distribution in the initialization steps for $P\left(r_{0}, \xi_{0} \mid \emptyset\right)$ and $\Psi^{\left(r_{0}, \pi_{0}\right)}$ as well as the update step for $\tilde{\Psi}_{t}^{(r, \pi)}$ in the case where $r_{t}=0$.

To ensure the robustness of our results we fit two additional suboptimal Bayesian models and compared each model to the winning model ( $\mathrm{Exp}_{\text {bias }}$ ). To capture conservatism as we did in the $\mathrm{Exp}_{\text {bias }}$ model, we fit a model that took a weighted average between the probability predicted by the ideal observer model and $\pi=0.5$ (Bayes ${ }_{\text {bias }}$ ). The weight on the probability computed by an ideal observer was defined by the parameter $w$ with range $0 \leq w \leq 1$, such that 0 indicated the use of a fixed criterion and 1 the optimal. This model is similar to the Bayes ${ }_{\beta}$ model described in the main text with a symmetric hyperprior on $\pi$, in that both result in conservatism. However, we ran it to ensure that the fits did not change when the parameterization was identical to the $\mathrm{Exp}_{\text {bias }}$ model. We also fit a three-parameter model, in which the maximum run length $r$ and the hyperparameter $\beta$ were both free parameters (Bayes ${ }_{r, \beta}$ ). We chose these parameters because the Bayes ${ }_{r}$ model was the best fitting Bayesian model tested, and the Bayes ${ }_{\beta}$ model takes into account conservatism, which we observed in our data. Neither of these additional models fit better than the $\operatorname{Exp}_{\text {bias }}\left(\right.$ Bayes $_{\text {bias }}: \Delta \mathrm{LML}_{\text {covert }}=$

---

#### Page 11

$19.92 \pm 5.00, \Delta \mathrm{LML}_{\text {overt }}=20.49 \pm 3.43 ;$ Bayes $_{r, \beta}: \Delta \mathrm{LML}_{\text {covert }}=5.89 \pm 2.10$, $\Delta \mathrm{LML}_{\text {overt }}=10.54 \pm 4.32)$.

# 2.2 Reinforcement learning - probability updating

The following model ( $\mathrm{RL}_{\text {prob }}$ ) differs from the RL models in the main text in that it updates the category probability (as opposed to updating the decision criterion), which makes it similar to the exponential-averaging model. Similarly to the RL models in the main text (and in contrast with the Exp models), it updates probability according to a delta rule which is applied only after incorrect responses. After each response at trial $t$, the probability estimate for the next trial is updated using the following delta rule,

$$
\hat{\pi}_{\mathrm{A}, t+1}= \begin{cases}\hat{\pi}_{\mathrm{A}, t} & \text { if correct } \\ \hat{\pi}_{\mathrm{A}, t}+\alpha_{\mathrm{prob}}\left(C_{t}-\hat{\pi}_{\mathrm{A}, t}\right) & \text { if incorrect }\end{cases}
$$

where $\hat{\pi}_{\mathrm{A}, t}$ is the observer's estimate of the probability for category A on trial $t$ $\left(\hat{\pi}_{\mathrm{B}, t}=1-\hat{\pi}_{\mathrm{A}, t}\right), \alpha_{\text {prob }}$ is the learning rate, and $C_{t}$ is the current category label. Thus, the probability estimate is updated when negative feedback is received by taking a small step in the direction of the most recently experienced category. This model has two free parameters ( $\alpha_{\text {prob }}$ and either $\sigma_{\mathrm{v}}$ or $\sigma_{\mathrm{a}}$ ).

To capture conservatism, we considered an additional model ( $\mathrm{RL}_{\text {prob, bias }}$ ) in which we took a weighted average between the probability estimate and $\pi=0.5$, which added another free parameter $w$ to the model.

In terms of model comparison, both models were indistinguishable from the fixed criterion model $\left(\mathrm{RL}_{\text {prob }}: \Delta \mathrm{LML}_{\text {covert }}=-2.48 \pm 2.11, \Delta \mathrm{LML}_{\text {overt }}\right.$ $=$ $-2.92 \pm 1.22 ; \mathrm{RL}_{\text {prob, bias }}: \Delta \mathrm{LML}_{\text {covert }}=0.36 \pm 2.18, \Delta \mathrm{LML}_{\text {overt }}=1.09 \pm$ 1.34). Furthermore, the fits were significantly worse than the $\mathrm{Exp}_{\text {bias }}$ model $\left(\mathrm{RL}_{\text {prob }}: \Delta \mathrm{LML}_{\text {covert }}=61.75 \pm 13.44, \Delta \mathrm{LML}_{\text {overt }}=75.53 \pm 10.20 ; \mathrm{RL}_{\text {prob, bias }}\right.$ : $\left.\Delta \mathrm{LML}_{\text {covert }}=58.91 \pm 13.71, \Delta \mathrm{LML}_{\text {overt }}=71.52 \pm 9.91\right)$.

### 2.3 Wilson et al. (2013)

The Wilson et al. model $[4,5]$ was developed as an approximation to the full change-point detection model. Their approximation used a mixture of delta rules, each of which alone is identical to our Exp model with different learning rates. In the main text, we fit a three node model with two free node parameters $\left(l_{2}\right.$ and $\left.l_{3}\right)$ and the hyperparameter on category probability

---

#### Page 12

$\nu_{\mathrm{p}}$ as a free parameter as well. Here, instead we fit the model with $\nu_{\mathrm{p}}=2$, which was determined based on our experimental design. On average, this model provided a worse fit than the Wilson et al. model presented in the main text $\left(\Delta \mathrm{LML}_{\text {covert }}=28.79 \pm 9.91\right.$; overt task: $\left.\Delta \mathrm{LML}_{\text {overt }}=4.18 \pm 2.94\right)$.

# 3 Comparison of the Bayes ${ }_{r, \pi, \beta}$ and the $\mathbf{E x p}_{\text {bias }}$ models

We compared the winning $\operatorname{Exp}_{\text {bias }}$ from our preliminary model-comparison analysis to the Bayes ${ }_{r, \pi, \beta}$, which allowed for incorrect beliefs and a bias towards equal priors. Because of the complexity of the Bayes ${ }_{r, \pi, \beta}$, we fit both models using maximum likelihood and variational Bayes [6], thus computing the Bayesian information criteria (BIC) and ELBO scores for each observer and model. Each of these model-comparison methods penalizes the model for increased complexity. The maximum-likelihood fits for each model and task are shown in Fig S1A (covert) and Fig S1B (overt). The relative modelcomparison scores are shown in Fig S1C. For both BIC and ELBO, we found that the two models were indistinguishable from one another.

## 4 Model comparison with AIC

To ensure the robustness of our model comparison results, in addition to using the log marginal likelihood as a measure of goodness of fit, we calculated the Akaike information criterion (AIC) [7]. Unlike the log marginal likelihood, AIC uses a point estimate and penalizes for complexity by adding a correction for the number of parameters $k: A I C=2 k-2 \ln (\hat{L})$, where $\hat{L}$ is the maximum log likelihood of the dataset. Like the log marginal likelihood, AIC is best interpreted as a relative score. The model comparison results using relative AIC scores (relative to the winning model) are shown in Fig S2. From the plot we see that our results do not change using a different metric (compare with Fig 3 in the main text). Furthermore, the ranks for all models do not change for either task when comparing -0.5 AIC and LML scores ( $\rho=1.0$, $p<0.0001$ ). Note that for historical reason the AIC scores have an additional factor of two.

---

#### Page 13

> **Image description.** The image contains three plots, labeled A, B, and C.
>
> Plot A is a line graph. The x-axis is labeled "trial number" and ranges from 0 to 800. The y-axis is labeled "excess 'A' responses" and ranges from -45 to 90. There are three lines plotted: a gray line labeled "Observer", a dark blue line labeled "Bayesr,π,β", and a green line labeled "Expbias". All three lines start near 0 and fluctuate, reaching a peak around trial number 400 before decreasing again.
>
> Plot B is also a line graph. The x-axis is labeled "trial number" and ranges from 0 to 800. The y-axis is labeled "criterion orientation (deg)" and ranges from -50 to 50. Similar to Plot A, there are three lines plotted: a gray line labeled "Observer", a dark blue line labeled "Bayesr,π,β", and a green line labeled "Expbias". These lines show more rapid fluctuations compared to Plot A.
>
> Plot C is a bar graph. The x-axis is labeled "task" and has two categories: "Covert" and "Overt". The y-axis is labeled "relative model comparison score" and ranges from -25 to 25. For each task, there are two bars: a light gray bar labeled "BIC" and a dark gray bar labeled "ELBO". Error bars are shown for each bar, indicating the standard error. The "Covert" task bars are slightly above 0, while the "Overt" task bars are also slightly above 0.

Figure S1. Maximum-likelihood model comparison. Maximumlikelihood fits in the covert (A) and overt (B) tasks for observers CWG and GK, respectively (green - $\operatorname{Exp}_{\text {bias }}$; dark blue - Bayes $_{\epsilon, \pi, \beta}$ ). The observer's response is shown in gray. The relative model-comparison scores (C) were computed using both BIC and ELBO (an approximate lower bound of the log marginal likelihood) scores. Error bars: $+/-2$ S.E.

---

#### Page 14

> **Image description.** This image contains two bar charts arranged vertically, comparing different models using relative AIC scores. The top chart represents the "Covert" task, while the bottom chart represents the "Overt" task.
>
> Here's a breakdown of the elements:
>
> - **Arrangement:** Two bar charts are stacked vertically. Both charts share the same horizontal axis labels but have different vertical axis scales.
>
> - **Vertical Axis:** The vertical axis on both charts is labeled "relative AIC score (AIC - AICExpbias)". The top chart's scale ranges from 0 to 400, while the bottom chart's scale ranges from 0 to 200.
>
> - **Horizontal Axis:** The horizontal axis is labeled "model" and displays the names of various models. The models are: "BayeSideal", "Bayesr", "Bayesπ", "Bayesβ", "Fixed", "Exp", "Expbias", "Wilson et al. (2013)", "RL", "Behrens et al. (2007)", and "Behrens et al. (2007) + bias". The labels are slightly rotated to fit.
>
> - **Bars:** Each model has a corresponding bar on each chart, representing its relative AIC score. The bars are different colors, including dark blue, light blue, cyan, red, dark green, light green, yellow, purple, and orange.
>
> - **Error Bars:** Each bar has a vertical error bar extending above it, indicating the uncertainty or variability in the AIC score.
>
> - **Titles:** The top chart is labeled "Covert" on the left side, and the bottom chart is labeled "Overt" on the left side.

Figure S2. Model comparison with AIC scores. AIC scores relative to the $\operatorname{Exp}_{\text {bias }}$ model are shown for the covert (top) and overt (bottom) tasks. Higher scores indicate a worse fit. Error bars: $95 \%$ C.I.

---

#### Page 15

# 5 Recovery analysis

### 5.1 Model recovery

We performed a model recovery analysis to validate our model-fitting pipeline and ensure that models were identifiable [8]. For this analysis, we generated ten synthetic datasets from each model, observer, and task (1,980 datasets). Parameters for each simulated dataset were determined by sampling with replacement from the posterior over model parameters. We fitted these datasets with all models ( 17,820 fits), and for each pair of generating and fitting models we calculated the proportion of times each model fit the data best (i.e., had the greatest LML score), producing the confusion matrix in Fig S3. First, the fact that the confusion matrix is mostly diagonal means that most datasets were best fit by their true generating model, suggesting a generally successful recovery.

Across both tasks, we found that the true generating model was the bestfitting model for $70.1 \% \pm 9.0 \%$ of simulated datasets (covert: $66.06 \% \pm 11.36 \%$; overt: $74.0 \% \pm 8.6 \%$; mean and SEM across models). For most simulated datasets, the true generating model was recovered for all models except the Exp model (see diagonal in Fig S3), which was best fit by the Wilson et al. (2013) model. However, this does not affect the results as the Wilson et al. (2013) model was not the best-fitting model across observers. Additionally, in the covert-criterion task (Fig S3B) the RL model simulations were best fit by the $\operatorname{Exp}_{\text {bias }}$ model. This is potentially due to the fact that observers exhibited a greater amount of conservatism in the covert task. Increased conservatism results in smaller, smoother changes of criterion, which is consistent with what we observed in the RL model (see the third row, third column panel in Fig 2D in the main text), so that data from the RL model are also well fit by the $\operatorname{Exp}_{\text {bias }}$ model. However, these models were clearly distinguishable in the overt task (Fig S3C), which allows us to rule out the RL model. These results again provide support for the use of tasks, such as our overt task, that allow the researcher to better distinguish between computational models.

### 5.2 Parameter recovery

To determine whether our parameter estimation procedure was biased, we analyzed the parameter recovery performance for the $\operatorname{Exp}_{\text {bias }}$ model. Specifically, for each observer we created ten synthetic datasets by sampling from

---

#### Page 16

the posterior over model parameters and simulating the experiment with the same experimental parameters as the observer experienced. Each synthetic dataset was then fit to the $\operatorname{Exp}_{\text {bias }}$ model and the best fitting parameters (MAP estimates) were estimated. For each parameter and task, we conducted a paired-samples t-test comparing the average best fitting parameters to the average generating parameters. We did not find a statistically significant difference between the fitted and generating $\alpha_{\text {Exp }}$ and $w$ parameters for either task: $\alpha_{\text {Exp }}$ (covert: $t(10)=1.14, p=0.28$; overt: $t(10)=1.01$, $p=0.34$ ) and $w$ (covert: $t(10)=-2.00, p=0.07$; overt: $t(10)=0.46$, $p=0.66$ ), suggesting good parameter recovery. While there was no significant difference between the fitted and generating noise parameter $\left(\sigma_{\mathrm{v}}\right)$ in the covert task $(t(10)=2.10, p=0.06)$, we found a significant difference in the noise parameter $\left(\sigma_{\mathrm{a}}\right)$ in the overt task $(t(10)=-16.53, p=1.37 e-08)$. This difference remained significant after correcting for multiple comparisons using the Bonferroni cutoff of $p=0.0083$. This result suggests that $\sigma_{\mathrm{a}}$ was overestimated on average.

# 6 Measurement task

### 6.1 Procedure

During the 'measurement' session, observers completed a two-interval forcedchoice, orientation-discrimination task in which two black ellipses were presented sequentially on a mid-gray background and the observer reported the interval containing the more clockwise ellipse (Fig S4A). This allowed us to measure the observer's sensory uncertainty.

### 6.2 Analysis

A cumulative normal distribution was fit to the orientation-discrimination data (probability of choosing interval one as a function of the orientation difference between the first and second ellipse) using a maximum-likelihood criterion with parameters $\mu, \sigma$, and $\lambda$ (the mean, SD, and lapse rate). We define threshold as the underlying measurement $\mathrm{SD} \sigma_{\mathrm{v}}$ (correcting for the 2 IFC task by dividing by $\sqrt{2}$ ).

---

#### Page 17

> **Image description.** This image consists of three heatmaps, labeled A, B, and C, representing model recovery performance. The heatmaps are square matrices where rows represent "fit model" and columns represent "simulated model". Each cell's brightness indicates the proportion of "wins" for a given fit model against a simulated model, with a color bar ranging from 0 (black) to 1 (white) displayed to the left of heatmap A.
>
> - **Heatmap A:** This is the largest heatmap, showing model recovery performance across both tasks. The x-axis (simulated model) and y-axis (fit model) are labeled with the following models: Bayesideal, Bayesr, Bayesπ, Bayesβ, Fixed, Exp, Expbias, Wilson et al. (2013), and RL. The diagonal elements of the matrix are generally brighter than the off-diagonal elements, indicating successful model recovery.
>
> - **Heatmap B:** This smaller heatmap represents model recovery performance for the covert task only. The x and y axes are not explicitly labeled but presumably use the same model order as heatmap A. The heatmap is smaller and positioned above heatmap C.
>
> - **Heatmap C:** This smaller heatmap represents model recovery performance for the overt task only. The x and y axes are not explicitly labeled but presumably use the same model order as heatmap A. It is located below heatmap B, separated by a thin white line.
>
> The overall visual impression is that of a comparative analysis of model fitting performance under different task conditions, with the brightness of cells indicating the success of fitting a particular model to data generated by another model.

Figure S3. Model recovery. For each model, observer, and task, 10 sets of parameters were sampled from the model posterior and used to generate synthetic data ( 1,980 total simulations). The synthetic datasets were then fit to each model ( 17,820 fits) and the goodness of fit was judged by computing the LML. The proportion of "wins" (i.e., the number of times the simulated model outperformed the alternative models) is indicated by brightness. Model recovery performance is shown across both tasks (A), the covert task only (B), and the overt task only (C).

# 6.3 Results

Fig S3B shows a representative psychometric function for one observer. The average threshold across observers was $\sigma_{\mathrm{v}}=6.71^{\circ} \pm 1.23^{\circ}$.

## 7 Category training

### 7.1 Procedure

Category training was completed prior to the covert- and overt-criterion tasks, so observers could learn the category distributions. It was important

---

#### Page 18

> **Image description.** The image contains two panels, labeled A and B.
>
> Panel A shows a sequence of visual stimuli presented over time. The sequence begins with a gray square containing a white plus sign in the center, labeled "500 ms" below. This is followed by a gray square containing a black ellipse, labeled "300 ms" below. Next is another gray square, labeled "500 ms" below. This is followed by a gray square containing a black ellipse, labeled "300 ms" below. Finally, there is a gray square with two outlined boxes labeled "1" and "2" inside, labeled "500 ms" below. An arrow labeled "time" points diagonally downward from the first square to the last square, indicating the temporal order of the stimuli.
>
> Panel B shows a psychometric function plotted on a graph. The x-axis is labeled "orientation difference (deg)" and ranges from -16 to 16. The y-axis is labeled "p(choose "Interval 1")" and ranges from 0 to 1, with a tick mark at 0.5. The graph contains several data points represented by black dots, which are fitted with a black sigmoid curve. The curve starts near 0 on the left side of the graph, rises steeply around x=0, and approaches 1 on the right side of the graph.

Figure S4. Measurement task. A: Trial sequence. Two ellipses were presented sequentially on a mid-gray background. The observer reported the interval containing the more clockwise ellipse. Feedback was provided. B: The best fitting psychometric function for one observer. The area of each data point is proportional to the number of trials.

not to confound category learning with probability learning. Training was identical to the covert-criterion task (Fig S5A). On each trial ( $N_{\text {trials }}=200$ ), a black ellipse was presented on a mid-gray background and observers reported the category to which it belonged. Category probability was equal during training and we provided correctness feedback. To determine how well observers learned the category distributions, observers estimated the mean orientation of each category at the end of the training block by rotating an ellipse to match the mean orientation (Fig S5B). Each category was estimated exactly once.

# 7.2 Results

Observers' estimates of the category means are shown in Fig S5C as a function of the true mean. Data points represents each observer's estimate after each task for each category. There was a significant correlation between category estimates and the true category means (category A: $r=0.82, p<0.0001$; category B: $r=0.97, p<0.0001$ ), suggesting that participants learned the categories reasonably well. On average, estimates were repelled from the

---

#### Page 19

category boundary (average category A error of $11.3^{\circ} \pm 6.3^{\circ}$ and average category B error of $-8.0^{\circ} \pm 2.6^{\circ}$; mean and SEM across observers).

> **Image description.** The image presents a figure with three panels, labeled A, B, and C, illustrating a category training experiment.
>
> Panel A, titled "Part 1: Categorization," depicts a sequence of events over time. It begins with a gray square containing a white plus sign, labeled "500 ms." This is followed by another gray square showing a tilted black ellipse, labeled "300 ms." Next, a gray square shows two outlined boxes labeled "1" and "2." Finally, there are two gray squares, each labeled "300 ms" and containing a speaker icon. The left square displays "Points: +1" above a green plus sign, while the right square shows "Points: +0" above a red plus sign. An arrow labeled "time" indicates the progression of these events.
>
> Panel B, titled "Part 2: Mean Estimation," also shows a sequence of events over time. It starts with a gray square containing the letter "A." This is followed by a series of gray squares, each displaying a tilted black ellipse with varying orientations. A branching line points from the series of squares to the text "Rotate to set the orientation." An arrow labeled "time" indicates the progression of these events.
>
> Panel C is a scatter plot. The x-axis is labeled "true category mean (deg)" and ranges from -100 to 100. The y-axis is labeled "est. category mean (deg)" and also ranges from -100 to 100. The plot contains data points representing "Covert" (black filled circles) and "Overt" (white filled circles) conditions for "Cat. A" (green squares) and "Cat. B" (red squares). A dashed line extends diagonally across the plot.

Figure S5. Category training. A: Trial sequence. After stimulus offset observers reported the category by key press and received feedback. B: Mean estimation task. After completing the training block, observers rotated an ellipse to estimate the category means. C: Estimation results. Observers' category-mean estimates are shown as a function of the true category mean for each category, observer and task.

# 8 Individual model fits

The maximum a posteriori (MAP) model fits for each observer, task, and model are plotted below for all models. Note that the parameter values obtained for the Bayes ${ }_{r, \pi, \beta}$ model via maximum likelihood are equivalent to MAP estimates, since for all parameters we used flat priors in the chosen parameterization.

---

#### Page 20

> **Image description.** The image shows two panels, A and B, each containing a series of line graphs. Each small graph represents data for an individual observer, indicated by a three-letter code above each graph.
>
> Panel A:
>
> - The panel is labeled "A" in the top left corner.
> - The y-axis label on the left side of the panel reads "excess 'A' responses". The y-axis ranges vary across the individual graphs, but generally span from approximately -150 to 100.
> - The x-axis label at the bottom of the panel reads "trial number". The x-axis ranges from 0 to 800 in all graphs.
> - Each graph contains two lines: a dark blue line and a gray line. These lines represent the data for each observer.
> - The observer codes are: CWG, EGC, EHN, ERK, GK, JKT, JYZ, RND, SML, SQC.
>
> Panel B:
>
> - The panel is labeled "B" in the top left corner.
> - The y-axis label on the left side of the panel reads "criterion orientation (deg)". The y-axis ranges from approximately -50 to 50 in all graphs.
> - The x-axis label at the bottom of the panel reads "trial number". The x-axis ranges from 0 to 800 in all graphs.
> - Each graph contains two lines: a dark blue line and a gray line. These lines represent the data for each observer.
> - The observer codes are: CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC.
>
> The arrangement of the graphs is in a grid-like pattern within each panel.

Figure S6. Bayes ${ }_{\text {ideal }}$ fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 21

> **Image description.** The image shows two panels, labeled A and B, each containing a series of line graphs.
>
> Panel A:
>
> - The panel is titled "A".
> - The y-axis label is "excess 'A' responses". The y-axis ranges vary across the individual graphs, but generally span from -100 to 100 or -150 to 50.
> - The x-axis label is "trial number". The x-axis ranges from 0 to 800.
> - There are ten individual line graphs arranged in a grid. Each graph has a title above it: CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, SML, and SQC.
> - Each graph contains two lines: a blue line and a gray line, representing data points across the trial number. The lines fluctuate and show different patterns across the graphs.
>
> Panel B:
>
> - The panel is titled "B".
> - The y-axis label is "criterion orientation (deg)". The y-axis ranges from -50 to 50.
> - The x-axis label is "trial number". The x-axis ranges from 0 to 800.
> - There are ten individual line graphs arranged in a grid, mirroring the arrangement in Panel A. Each graph has the same title as its counterpart in Panel A: CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, SML, and SQC.
> - Each graph contains two lines: a blue line and a gray line, representing data points across the trial number. The lines fluctuate and show different patterns across the graphs. The gray lines appear more jagged than the blue lines.
>
> In summary, the image presents a series of line graphs in two panels, A and B, showing data for different conditions (CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, SML, and SQC) across trial numbers. Panel A displays "excess 'A' responses", while Panel B displays "criterion orientation (deg)". Each graph contains a blue and a gray line.

Figure S7. Bayes ${ }_{c}$ fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 22

> **Image description.** The image contains two panels, labeled A and B, each displaying a series of line graphs. Panel A is titled "excess 'A' responses" on the y-axis and "trial number" on the x-axis, while panel B is titled "criterion orientation (deg)" on the y-axis and "trial number" on the x-axis.
>
> Each panel contains nine individual line graphs arranged in a 3x3 grid. Each of these smaller graphs is labeled with a three-letter identifier (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC).
>
> In panel A, each small graph displays two lines: one in blue and one in gray. The y-axis scales vary slightly between the graphs, but generally range from -100 to 100 or -150 to 50. The x-axis ranges from 0 to 800.
>
> In panel B, each small graph also displays two lines: one in blue and one in gray. The y-axis scale is consistent across all graphs, ranging from -50 to 50. The x-axis ranges from 0 to 800. The lines in panel B appear more jagged and have higher frequency fluctuations compared to the lines in panel A.

Figure S8. Bayes $_{\tau}$ fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 23

> **Image description.** This image contains two panels, labeled A and B, each displaying a series of line graphs. Panel A shows graphs of "excess 'A' responses" versus "trial number," while Panel B shows graphs of "criterion orientation (deg)" versus "trial number."
>
> **Panel A:**
>
> - The panel is labeled "A" in the top left corner.
> - The y-axis label is "excess 'A' responses." The range of the y-axis varies slightly between graphs, but generally spans from -100 to 100 or a similar range.
> - The x-axis label is "trial number." The x-axis ranges from 0 to 800 in all graphs.
> - There are ten individual line graphs arranged in a grid. Each graph is labeled with a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC) above it.
> - Each graph contains two lines: one in light blue and one in gray. The lines fluctuate, showing the change in "excess 'A' responses" over the course of 800 trials.
>
> **Panel B:**
>
> - The panel is labeled "B" in the top left corner.
> - The y-axis label is "criterion orientation (deg)." The y-axis ranges from -50 to 50 in all graphs.
> - The x-axis label is "trial number." The x-axis ranges from 0 to 800 in all graphs.
> - Similar to Panel A, there are ten individual line graphs arranged in a grid, each labeled with the same three-letter code as in Panel A.
> - Each graph contains two lines: one in light blue and one in gray. The lines fluctuate, showing the change in "criterion orientation" over the course of 800 trials. The lines in Panel B appear more jagged and have a higher frequency of fluctuation compared to the lines in Panel A.

Figure S9. Bayes ${ }_{\beta}$ fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 24

> **Image description.** The image contains two panels, labeled A and B, each displaying a series of line graphs. Panel A shows graphs of "excess 'A' responses" and Panel B shows graphs of "criterion orientation (deg)".
>
> **Panel A:**
>
> - The panel is labeled "A" in the top left corner.
> - The y-axis label on the left side reads "excess 'A' responses".
> - There are 9 small line graphs arranged in a 3x3 grid.
> - Each graph has a title above it consisting of a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC).
> - Each graph plots data points connected by lines. There are two lines in each graph, one in grey and one in dark blue, that appear to represent different fits of the data.
> - The x-axis is labeled "trial number" at the bottom of the panel.
> - The x-axis ranges from 0 to 800 in all graphs.
> - The y-axis ranges vary slightly between graphs to accommodate the data, but generally span from -100 to 100 or -150 to 50.
>
> **Panel B:**
>
> - The panel is labeled "B" in the top left corner.
> - The y-axis label on the left side reads "criterion orientation (deg)".
> - There are 9 small line graphs arranged in a 3x3 grid.
> - Each graph has a title above it consisting of a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC), matching the labels in Panel A.
> - Each graph plots data points connected by lines. There are two lines in each graph, one in grey and one in dark blue, that appear to represent different fits of the data.
> - The x-axis is labeled "trial number" at the bottom of the panel.
> - The x-axis ranges from 0 to 800 in all graphs.
> - The y-axis ranges from -50 to 50 in all graphs.

Figure S10. Bayes ${ }_{r, \pi, \beta}$ fits based on maximum-likelihood estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 25

> **Image description.** The image shows two panels, A and B, each containing a grid of smaller plots. Each plot displays a graph with the x-axis labeled "trial number" ranging from 0 to 800, and the y-axis labeled "excess 'A' responses" in panel A and "criterion orientation (deg)" in panel B. Each small plot is also labeled with a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC).
>
> Panel A consists of 10 small plots arranged in a grid. Each plot contains two lines: a gray line that fluctuates and a smoother red line. The y-axis in panel A ranges from -100 to 100 for most plots, but some plots have different ranges (e.g., -150 to 50).
>
> Panel B consists of 10 small plots arranged in a grid. Each plot contains a fluctuating gray line and a relatively flat red line close to zero. The y-axis in panel B ranges from -50 to 50 for all plots.

Figure S11. Fixed fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 26

> **Image description.** This image contains a series of line graphs arranged in two panels, labeled A and B. Each panel contains 10 individual graphs arranged in a 2x5 grid.
>
> **Panel A:**
>
> - The panel is titled "A" in the top left corner.
> - The y-axis label is "excess 'A' responses". The y-axis ranges vary between the individual graphs, with maximum values ranging from 50 to 100 and minimum values ranging from -50 to -150.
> - The x-axis label is "trial number". The x-axis ranges from 0 to 800 in all graphs.
> - Each graph is labeled with a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC) above the plot area.
> - Each graph contains two lines: a green line and a gray line. The lines show trends over the trial number.
>
> **Panel B:**
>
> - The panel is titled "B" in the top left corner.
> - The y-axis label is "criterion orientation (deg)". The y-axis ranges from -50 to 50 in all graphs.
> - The x-axis label is "trial number". The x-axis ranges from 0 to 800 in all graphs.
> - Each graph is labeled with a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC) above the plot area, matching the labels in Panel A.
> - Each graph contains two lines: a green line and a gray line. The gray line is more jagged than the green line. The lines show trends over the trial number.

Figure S12. Exp fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 27

> **Image description.** The image shows a set of line graphs arranged in two panels, labeled A and B. Each panel contains 10 individual graphs, arranged in a 2x5 grid.
>
> Panel A:
> The y-axis label is "excess 'A' responses". The x-axis label is "trial number". Each individual graph is labeled with a three-letter code (CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC). Each graph contains two lines: one green and one gray. The x-axis ranges from 0 to 800. The y-axis ranges vary depending on the individual graph, but are typically between -100 and 100.
>
> Panel B:
> The y-axis label is "criterion orientation (deg)". The x-axis label is "trial number". Each individual graph is labeled with the same three-letter code as in panel A. Each graph contains two lines: one green and one gray. The x-axis ranges from 0 to 800. The y-axis ranges from -50 to 50.

Figure S13. $\operatorname{Exp}_{\text {bias }}$ fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 28

> **Image description.** The image presents a figure with two panels, labeled A and B, each containing a series of line graphs. Each small graph within a panel displays two lines, one in gold and one in gray, plotted against a horizontal axis labeled "trial number".
>
> Panel A:
> The overall title on the y-axis is "excess 'A' responses". The y-axis scales vary slightly between individual graphs, but generally range from -100 to 100, or -150 to 50. The x-axis, "trial number", ranges from 0 to 800 in all subplots. Each subplot is labeled with a three-letter code (e.g., CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, SQC).
>
> Panel B:
> The overall title on the y-axis is "criterion orientation (deg)". The y-axis scales range from -50 to 50. The x-axis, "trial number", ranges from 0 to 800 in all subplots except for the EGC subplot, which ranges to 500. Each subplot is labeled with the same three-letter code as in panel A. The gold line appears smoother than the gray line in most subplots.

Figure S14. Wilson et al. (2013) fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 29

> **Image description.** This image contains two panels, labeled A and B, each displaying a series of line graphs. Each graph within a panel represents data for a different observer, identified by a three-letter code above each graph (e.g., CWG, EGC, EHN).
>
> Panel A:
>
> - The panel is titled "A".
> - The y-axis label on the left side of the panel reads "excess 'A' responses". The y-axis ranges vary slightly between graphs but generally span from approximately -100 to 100 or -150 to 50.
> - The x-axis label at the bottom of the panel reads "trial number". The x-axis ranges from 0 to 800 in all graphs.
> - Each graph contains two lines: a gray line and a purple line. The lines show how the excess 'A' responses vary with the trial number for each observer.
>
> Panel B:
>
> - The panel is titled "B".
> - The y-axis label on the left side of the panel reads "criterion orientation (deg)". The y-axis ranges from -50 to 50 in all graphs.
> - The x-axis label at the bottom of the panel reads "trial number". The x-axis ranges from 0 to 800 in all graphs.
> - Each graph contains two lines: a gray line and a purple line. The gray line is more jagged than the purple line. The lines show how the criterion orientation varies with the trial number for each observer.
>
> The observers are CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, and SQC.

Figure S15. RL fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 30

> **Image description.** This image contains two panels, labeled A and B, each displaying a set of line graphs. Each graph within a panel represents data for a different observer, indicated by a three-letter code above each graph.
>
> Panel A: Each graph in this panel plots "excess 'A' responses" on the y-axis, ranging from -100 to 100 (with some graphs having different ranges, such as -150 to 50 or -50 to 50), against "trial number" on the x-axis, ranging from 0 to 800. Each graph contains two lines: one in gray and one in orange, depicting different conditions or data sets for that observer. The observers are labeled as CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, and SQC.
>
> Panel B: Each graph in this panel plots "criterion orientation (deg)" on the y-axis, ranging from -50 to 50, against "trial number" on the x-axis, ranging from 0 to 800. Similar to Panel A, each graph contains two lines, one in gray and one in orange, representing different data sets for each observer. The observers are the same as in Panel A: CWG, EGC, EHN, ERK, GK, HHL, JKT, JYZ, RND, SML, and SQC.
>
> The arrangement of graphs is consistent across both panels, with the same observers appearing in the same relative positions.

Figure S16. Behrens et al. (2007) fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 31

> **Image description.** The image contains two panels, labeled A and B, each containing a series of line graphs. Each small graph within a panel displays data for a specific observer, identified by a three-letter code above each graph (e.g., CWG, EGC, EHN).
>
> Panel A:
>
> - The y-axis label is "excess 'A' responses."
> - The x-axis label is "trial number."
> - Each graph shows two lines: one in gray and one in orange. The x-axis ranges from 0 to 800, and the y-axis ranges vary among the graphs but generally span from -100 to 100.
>
> Panel B:
>
> - The y-axis label is "criterion orientation (deg)."
> - The x-axis label is "trial number."
> - Each graph shows two lines: one in gray and one in orange. The x-axis ranges from 0 to 800, and the y-axis ranges from -50 to 50.
>
> The graphs in both panels show how the gray and orange lines change over the course of the trials, with varying patterns for each observer. The orange line appears smoother than the gray line in most of the graphs in panel B.

Figure S17. Behrens et al. (2007) + bias fits based on MAP estimation in the covert (A) and overt (B) tasks for each observer.

---

#### Page 32

# References

1. Adams RP, MacKay DJ. Bayesian online changepoint detection. arXiv preprint arXiv:07103742. 2007;
2. Knuth DE. Two notes on notation. Am Math Mon. 1992;99:403-422.
3. Press WH, Teukolsky SA, Vetterling WT, Flannery BP. Numerical recipes 3rd edition: The art of scientific computing. Cambridge, England: Cambridge University Press; 2007.
4. Wilson RC, Nassar MR, Gold JI. A mixture of delta-rules approximation to Bayesian inference in change-point problems. PLoS Comput Biol. 2018;14(6):e1006210.
5. Wilson RC, Nassar MR, Gold JI. Correction: A mixture of delta-rules approximation to Bayesian inference in change-point problems. PLoS Comput Biol. 2013;9(7):e1003150.
6. Acerbi L. Variational Bayesian Monte Carlo. In: Advances in Neural Information Processing Systems. vol. 31; 2018. p. 8213-8223.
7. Akaike H. Information theory and an extension of the maximum likelihood principle. In: Proceedings of the 2nd international symposium on information; 1973. p. 267-281.
8. Acerbi L, Ma WJ, Vijayakumar S. A framework for testing identifiability of Bayesian models of perception. In: Advances in Neural Information Processing Systems; 2014. p. 1026-1034.