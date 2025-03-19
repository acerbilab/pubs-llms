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
