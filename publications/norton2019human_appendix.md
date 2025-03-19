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
