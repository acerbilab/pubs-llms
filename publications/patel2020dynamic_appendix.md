# Dynamic allocation of limited memory resources in reinforcement learning - Appendix

---

#### Page 13

# Appendices 

## A Computing the gradient to maximize the objective function

## A. 1 Gradient of the log policy

In order to compute $\nabla_{\sigma} \log (\pi(a \mid s))$, we first note that we can rewrite draws from the memory distribution, $\bar{q}_{s a} \sim \mathcal{N}\left(\bar{q}_{s a}, \sigma_{s a}^{2}\right)$, as $\bar{q}_{s a}=\bar{q}_{s a}+\zeta_{s a} \sigma_{s a}$, where $\zeta_{s a} \sim \mathcal{N}(0,1)$ [33]. In this section, we abuse the notation slightly to omit the explicit dependence on the state-action pair $(s, a)$ for clarity, and instead place it in the subscript. With this, we can write our policy $\pi$ as a probability vector for all actions $a$ in a given state $s$ :

$$
\begin{aligned}
\pi(a \mid s) & =\delta\left(a, \underset{a^{\prime}}{\arg \max }\left(\bar{q}_{s a^{\prime}}+\zeta_{s a^{\prime}} \sigma_{s a^{\prime}}\right)\right) \\
\pi(\cdot \mid s) & =\lim _{\beta \rightarrow \infty} \operatorname{softmax}\left(\overline{\boldsymbol{q}}_{s}+\zeta_{s} \sigma_{s}, \beta\right) \\
& =\lim _{\beta \rightarrow \infty} \frac{1}{\sum_{a} \exp \beta\left(\bar{q}_{s a}+\zeta_{s a} \sigma_{s a}\right)}\left[\begin{array}{c}
\exp \beta\left(\bar{q}_{s a_{1}}+\zeta_{s a_{1}} \sigma_{s a_{1}}\right) \\
\vdots \\
\exp \beta\left(\bar{q}_{s a_{n}}+\zeta_{s a_{n}} \sigma_{s a_{n}}\right)
\end{array}\right]
\end{aligned}
$$

where in the first line we applied the Thompson sampling rule (that is, pick the action with maximal sampled value), in the second line we rewrote it as the limit of a softmax with inverse temperature $\beta \rightarrow \infty$, and in the last line we wrote the softmax explicitly (as a vector for each entry of $\pi(\cdot \mid s)$ ).
Next, we relax the limit $\beta \rightarrow \infty$ in Eq. A. 9 so as to differentiate the logarithm of the policy $\log \pi$ for $\beta>0$ with respect to the relevant elements of the resource allocation vector $\sigma(s, a)$ as follows:

$$
\begin{aligned}
\frac{\partial}{\partial \sigma_{s a}} \log \pi(\cdot \mid s) & =-\frac{\partial}{\partial \sigma_{s a}} \log \left(\sum_{a} \exp \beta\left(\bar{q}_{s a}+\zeta_{s a} \sigma_{s a}\right)\right)+\frac{\partial}{\partial \sigma_{s a}}\left[\begin{array}{c}
\beta\left(\bar{q}_{s a_{1}}+\zeta_{s a_{1}} \sigma_{s a_{1}}\right) \\
\vdots \\
\beta\left(\bar{q}_{s a_{n}}+\zeta_{s a_{n}} \sigma_{s a_{n}}\right)
\end{array}\right] \\
& =-\frac{\exp \beta\left(\bar{q}_{s a}+\zeta_{s a} \sigma_{s a}\right)}{\sum_{a} \exp \beta\left(\bar{q}_{s a}+\zeta_{s a} \sigma_{s a}\right)} \beta \zeta_{s a}+\beta \zeta_{s a} \delta\left(a, a_{i}\right) \\
& =\beta \zeta_{s a}\left(\delta\left(a, a_{i}\right)-\pi(a \mid s)\right)
\end{aligned}
$$

where the final step follows from rewriting the softmax function as the (soft) policy $\pi(a \mid s)$ in Eq. A.9, i.e. with some $\beta>0$ but not $\beta \rightarrow \infty$.

Thus, the gradient of the logarithm of the policy $\log \pi(a \mid s)$ with respect to the resource allocation vector $\sigma$ can be written as:

$$
\frac{\partial}{\partial \sigma_{s^{\prime} a^{\prime}}} \log \pi(a \mid s)= \begin{cases}\beta \zeta_{s a}(1-\pi(a \mid s)) & \text { for } s^{\prime}=s, a^{\prime}=a \\ -\beta \zeta_{s a^{\prime}} \pi\left(a^{\prime} \mid s\right) & \text { for } s^{\prime}=s, a^{\prime} \neq a \\ 0 & \text { for } s^{\prime} \neq s\end{cases}
$$

which is reported as Eq. 5 in the main text.

## A. 2 Gradient of the cost

In this section, we show how to compute $\nabla_{\sigma} D_{\mathrm{KL}}(Q \| P)$, where $Q=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right)$ and $P=$ $\mathcal{N}\left(\overline{\boldsymbol{q}}, \sigma_{\text {base }}^{2} I\right)$, and $I$ is the identity matrix. Since the covariance matrix is diagonal, we can take the gradient with respect to elements of the resource allocation vector $\sigma$ individually. In other words, we can take the gradient of each memory's marginal normal distribution with its standard deviation:

$$
\begin{aligned}
\frac{\partial}{\partial \sigma} D_{\mathrm{KL}}\left(\mathcal{N}\left(\bar{q}, \sigma^{2}\right) \| \mathcal{N}\left(\bar{q}, \sigma_{\text {base }}^{2}\right)\right) & =\frac{\partial}{\partial \sigma} \mathbb{E}_{Q}\left[\log \left(\frac{Q}{P}\right)\right] \\
& =\frac{\partial}{\partial \sigma} \mathbb{E}_{Q}[\log (Q)]-\frac{\partial}{\partial \sigma} \mathbb{E}_{Q}[\log (P)]
\end{aligned}
$$

---

#### Page 14

We can expand the first of the two terms as:

$$
\begin{aligned}
\frac{\partial}{\partial \sigma} \mathbb{E}_{Q}[\log (Q)] & =\frac{\partial}{\partial \sigma} \mathbb{E}_{x \sim \mathcal{N}\left(\bar{q}, \sigma^{2}\right)}\left[\log \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp -\frac{1}{2}\left(\frac{x-\bar{q}}{\sigma}\right)^{2}\right)\right] \\
& =\frac{\partial}{\partial \sigma} \mathbb{E}_{x \sim \mathcal{N}\left(\bar{q}, \sigma^{2}\right)}\left[-\frac{1}{2} \log (2 \pi)-\log (\sigma)-\frac{1}{2}\left(\frac{x-\bar{q}}{\sigma}\right)^{2}\right] \\
& =-\frac{1}{2} \frac{\partial}{\partial \sigma} \log (2 \pi)-\frac{\partial}{\partial \sigma} \log (\sigma)-\frac{1}{2} \frac{\partial}{\partial \sigma} \mathbb{E}_{x \sim \mathcal{N}\left(\bar{q}, \sigma^{2}\right)}\left[\left(\frac{x-\bar{q}}{\sigma}\right)^{2}\right] \\
& =-\frac{1}{\sigma}-\frac{1}{2} \frac{\partial}{\partial \sigma} \mathbb{E}_{z \sim \mathcal{N}(0,1)}\left[z^{2}\right] \\
& =-\frac{1}{\sigma}
\end{aligned}
$$

where we use the variable transformation $z=(x-\bar{q}) / \sigma$ in the penultimate step. We can follow a similar approach for the second term to get:

$$
\begin{aligned}
\frac{\partial}{\partial \sigma} \mathbb{E}_{Q}[\log (P)] & =\frac{\partial}{\partial \sigma} \mathbb{E}_{x \sim \mathcal{N}\left(\bar{q}, \sigma^{2}\right)}\left[\log \left(\frac{1}{\sqrt{2 \pi \sigma_{\text {base }}^{2}}} \exp -\frac{1}{2}\left(\frac{x-\bar{q}}{\sigma_{\text {base }}}\right)^{2}\right)\right] \\
& =\frac{\partial}{\partial \sigma} \mathbb{E}_{x \sim \mathcal{N}\left(\bar{q}, \sigma^{2}\right)}\left[-\frac{1}{2} \log \left(2 \pi \sigma_{\text {base }}^{2}\right)-\frac{1}{2}\left(\frac{x-\bar{q}}{\sigma}\right)^{2} \frac{\sigma^{2}}{\sigma_{\text {base }}^{2}}\right] \\
& =-\frac{1}{2} \frac{\partial}{\partial \sigma} \log (2 \pi \sigma_{\text {base }}^{2})-\frac{1}{2} \frac{\partial}{\partial \sigma} \mathbb{E}_{x \sim \mathcal{N}\left(\bar{q}, \sigma^{2}\right)}\left[\left(\frac{x-\bar{q}}{\sigma}\right)^{2}\right] \frac{\sigma^{2}}{\sigma_{\text {base }}^{2}} \\
& =-\frac{1}{2} \frac{\partial}{\partial \sigma} \mathbb{E}_{z \sim \mathcal{N}(0,1)}\left[z^{2}\right] \frac{\sigma^{2}}{\sigma_{\text {base }}^{2}} \\
& =-\frac{1}{2} \frac{\partial}{\partial \sigma} \frac{\sigma^{2}}{\sigma_{\text {base }}^{2}} \\
& =-\frac{\sigma}{\sigma_{\text {base }}^{2}}
\end{aligned}
$$

Combining Eqs. A.12, A.13, and A.14, we can write our analytically obtained gradient of the cost term with respect to individual elements of the resource allocation vector $\sigma(s, a)$ as:

$$
\frac{\partial}{\partial \sigma_{s a}} D_{\mathrm{KL}}\left(\mathcal{N}\left(\bar{q}, \sigma^{2} I\right) \| \mathcal{N}\left(\bar{q}, \sigma_{\text {base }}^{2} I\right)\right)=\frac{\sigma_{s a}}{\sigma_{\text {base }}^{2}}-\frac{1}{\sigma_{s a}}
$$

# A. 3 Justification for our choice of the gradient of expected reward 

A potential concern regarding our method of allocating resources may be our choice of the advantage function to compute the gradient of the expected rewards (Eqs. 3-4 in the main text). Crucially, the advantage gradient uses the means of the q-value distributions of the relevant memories. However, our main assumption in the paper is that agents do not have direct access to the mean of the q-value distribution. According to our assumption, the agent could only estimate the mean by averaging over a large number of samples from the distribution, a process which could take a considerable amount of time (because sequential samples from memory would be highly correlated [29]).
This concern is resolved by considering that in DRA the resource allocation vector is not updated during the trial, but rather only offline, i.e. before or after the trial, or potentially during sleep. This way, during the task, the agent draws single (Thompson) samples in order to act and does not waste extra time in order to consolidate and reallocate resources across its memories.
While 'offline sampling' resolves the issue of how agents can access the mean of the distribution to compute policy updates, and it is the approach followed in this work, it represents a binary solution

---

#### Page 15

(i.e., either the agent takes one Thompson sample online, or a very large number of them offline). We could generalize this approach by allowing an agent to take multiple samples from its q-value distribution to get a better estimate of the expected return while performing the task. Taking additional samples would cost them time, which they could potentially use to act in the environment and collect rewards. If the opportunity cost is higher than the potential increase in rewards obtained by taking more samples, they may not want to waste time sampling but instead make their memories (q-value distributions) precise enough that fewer samples suffice to maximize reward given their storage capacity. This is another example of the speed-accuracy trade-off we considered in Section 4.3 in the main text, and which we leave to explore for future work.

# B Task parameters and additional results 

## B. 1 Additional results for the planning task

In the main article, we showed results for the planning task we adapted from Huys et al. [40] where subjects had to plan sequences of $M=3$ moves. More generally, we ran DRA for $M \in\{3,4,5\}$, showing that the algorithm allocates resources differentially depending on $M$ (Fig. B.4).

> **Image description.** This image contains three scatter plots arranged horizontally, each showing the relationship between "Difference in cumulative reward" (x-axis) and "d' (discriminability)" (y-axis) for different values of "Search Budget". The plots are titled "M = 3", "M = 4", and "M = 5" respectively.
> 
> Here's a breakdown of the common elements and differences between the plots:
> 
> *   **Axes:**
>     *   X-axis: "Difference in cumulative reward". The range varies slightly between plots.
>         *   M=3: 20 to 100
>         *   M=4 and M=5: 0 to 120
>     *   Y-axis: "d' (discriminability)". The range varies between plots.
>         *   M=3: 0 to 6
>         *   M=4: 0 to 4
>         *   M=5: 0 to 5
> 
> *   **Data Points:** Each plot displays scattered data points, with each point representing a specific combination of "Difference in cumulative reward" and "d' (discriminability)". The points are color-coded based on the "Search Budget" using a gradient from light pink to dark purple.
> 
> *   **Regression Lines:** Each plot includes multiple regression lines, one for each "Search Budget" value. These lines show the linear relationship between the two variables for a given search budget. Each regression line has a shaded region around it, indicating the confidence interval.
> 
> *   **Legend:** Each plot includes a legend labeled "Search Budget". The legend lists the specific values of the search budget and their corresponding colors. The search budget values and their ranges differ slightly between the plots.
>     *   M=3: 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0
>     *   M=4: 6.25, 12.5, 18.75, 25.0, 37.5, 50.0, 75.0, 100.0
>     *   M=5: 3.125, 6.25, 12.5, 18.75, 25.0, 50.0, 75.0, 100.0
> 
> *   **Plot Titles:** Each plot is labeled with "M = [number]", where the number represents a parameter (likely the number of moves in a planning task, based on the context).
> 
> The plots show how the discriminability of memories (d') changes as a function of their impact on cumulative reward, for different search budgets and different values of the parameter M.

Figure B.4: Linear regression fits for the discriminability of memories as a function of their impact on cumulative reward for the planning task with number of moves $M \in\{3,4,5\}$.

## B. 2 Task parameters

Table B.1: Parameters used for each task

|  | Task |  |  |
| :--: | :--: | :--: | :--: |
| Parameter | Grid-world | Mountain Car | Planning task |
| $\alpha_{1}$ | 0.1 | 0.1 | 0.1 |
| $\alpha_{2}$ | 0.1 | 0.1 | 0.1 |
| $\beta$ | 10 | 10 | 10 |
| $\gamma$ | 1 | 1 | 1 |
| $\lambda$ | 0.2 | 0.1 | 1 |
| $\sigma_{\text {base }}$ | 5 | 5 | 100 |
| $\sigma_{0}$ | 3 | 3 | 50 |
| $N_{\text {traj }}$ | 10 | 10 | 10 |
| $N_{\text {restarts }}$ | 5 | 5 | 5 |

In this section, we report (Table B.1) and briefly describe the (hyper-)parameters chosen for each task. For the present study, we fixed the learning rates for the means and standard deviations of the memory distribution, $\alpha_{1}$ and $\alpha_{2}$ respectively, to reasonably low values. We set the inverse temperature parameter $\beta$ to a reasonably high value (for the softmax approximation to the 'hard' max to hold, as per Section A.1), but not too high to restrict the influence of individual updates on the resource

---

#### Page 16

allocation. As mentioned in the main text, we exclude discounting for all the tasks and thus set $\gamma=1$. Perhaps the most important choice is the parameter $\lambda$ that introduces a trade-off between the expected reward and the cost of being precise. We chose $\lambda$ that we best captured the difficulties faced by memory-limited agents, but a range of nearby values yields qualitatively similar results, e.g. in the mountain car task, $\lambda \in[0,0.4]$ allows agents to perform the task well with enough training. The other equally important parameter would perhaps be $\sigma_{\text {base }}$, which would represent the resources for some base distribution of q-values in memory before training. $\sigma_{\text {base }}$ controls how discriminable different actions would be from a given state, and we chose it appropriately given the reward structure of each task. As mentioned in the main text, starting with a higher resource budget than the base distribution, i.e. with $\sigma_{0}<\sigma_{\text {base }}$, either by means of paying more attention or allocating more neurons, allows agents to accelerate learning. We sampled $N_{\text {traj }}=10$ trajectories to update the resource allocation vector at the end of each trial with adequate precision. As shown in Fig. 3c in the main text, sampling more trajectories does not yield better performance, but less leads to variability in the stochastic estimate of the gradient and thus hurts performance. Finally, we performed $N_{\text {restarts }}=5$ optimization runs for each task to report an estimate of variability across runs (i.e., error bars), as mentioned in the main text. In addition to the above parameters used to display the results, we systematically varied the values of $\lambda$ and $\sigma_{\text {base }}$ in all tasks and report that the qualitative results hold for a large range of values of these parameters with $\sigma_{\text {base }}$ having a slightly stronger effect than $\lambda$.