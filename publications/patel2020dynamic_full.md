```
@article{patel2020dynamic,
  title={Dynamic allocation of limited memory resources in reinforcement learning},
  author={Nisheet Patel and Luigi Acerbi and A. Pouget},
  year={2020},
  journal={The Thirty-fourth Annual Conference on Neural Information Processing Systems (NeurIPS 2020)}
}
```

---

#### Page 1

# Dynamic allocation of limited memory resources in reinforcement learning

Nisheet Patel\*<br>Department of Basic Neurosciences<br>University of Geneva<br>nisheet.patel@unige.ch

Luigi Acerbi<br>Department of Computer Science<br>University of Helsinki<br>luigi.acerbi@helsinki.fi

Alexandre Pouget<br>Department of Basic Neurosciences<br>University of Geneva<br>alexandre.pouget@unige.ch

#### Abstract

Biological brains are inherently limited in their capacity to process and store information, but are nevertheless capable of solving complex tasks with apparent ease. Intelligent behavior is related to these limitations, since resource constraints drive the need to generalize and assign importance differentially to features in the environment or memories of past experiences. Recently, there have been parallel efforts in reinforcement learning and neuroscience to understand strategies adopted by artificial and biological agents to circumvent limitations in information storage. However, the two threads have been largely separate. In this article, we propose a dynamical framework to maximize expected reward under constraints of limited resources, which we implement with a cost function that penalizes precise representations of action-values in memory, each of which may vary in its precision. We derive from first principles an algorithm, Dynamic Resource Allocator (DRA), which we apply to two standard tasks in reinforcement learning and a model-based planning task, and find that it allocates more resources to items in memory that have a higher impact on cumulative rewards. Moreover, DRA learns faster when starting with a higher resource budget than what it eventually allocates for performing well on tasks, which may explain why frontal cortical areas in biological brains appear more engaged in early stages of learning before settling to lower asymptotic levels of activity. Our work provides a normative solution to the problem of learning how to allocate costly resources to a collection of uncertain memories in a manner that is capable of adapting to changes in the environment.

## 1 Introduction

Reinforcement learning (RL) is a powerful form of learning wherein agents interact with their environment by taking actions available to them and observing the outcome of their choices in order to maximize a scalar reward signal. Most RL algorithms use a value function in order to find good policies $[1,2]$ because knowing the optimal value function is sufficient to find an optimal policy [3, 4]. However, RL models typically assume that agents can access and update the value function for each state or state-action pair with nearly infinite precision. In neural circuits, however, this assumption

[^0]
[^0]: \*Current address: Département des neurosciences fondamentales, Université de Genève, CMU, 1 rue Michel-Servet, 1206 Genève, Switzerland. Alternative e-mail: nisheet.pat@gmail.com.

---

#### Page 2

is necessarily violated. The values must be stored in memories which are limited in their precision, especially in smaller brains in which the number of neurons can be severely limited [5].
In this work, we make such constraints explicit and consider the question of what rationality is when computational resources are limited [6, 7]. Specifically, we examine how agents might represent values with limited memory, how they may utilize imprecise memories in order to compute good policies, and whether and how they should prioritize some memories over others by devoting more resources to encode them with higher precision. We are interested in drawing useful abstractions from small-scale models that can be applied generally and in investigating whether brains employ similar mechanisms.
We pursue this idea by considering memories that are imprecise in their representation of values, which are stored as distributions that the agent may only sample from. We construct a novel family of cost functions that can adjudicate between maximizing reward and limited coding resources (Section 2). Crucially, for this general class of resource-limited objective functions, we derive how an RL agent can solve the resource allocation problem by computing stochastic gradients with respect to the coding resources. Further, we combine resource allocation with learning, enabling agents to assign importance to memories dynamically with changes in the environment. We call our proposed algorithm the Dynamic Resource Allocator (DRA), which we apply to standard tasks in RL in Section 3 and a model-based planning task in Section 4. ${ }^{1}$

# 1.1 Related work

Our work is related to previous research within the paradigm of bounded rationality [6, 7] on two accounts. First, previous studies have considered capacity-limited agents that trade reward to gain information [8-11], but did so in a way that abstracts away the underlying costs, and therefore cannot disambiguate the effects of limited storage capacity from other energetic or computational limitations. Second, a separate line of work makes the limitations in memory explicit [12-14] like we do, but such studies restrict their analyses to simple working memory tasks, such as reproducing observed stimuli or delayed recall, and lack a general-purpose, dynamical framework for decision-making. Our work is also ideologically related to that of others looking into prioritized replay of memories, both in RL [15] and neuroscience [16], with the important difference that these studies do not make the uncertainty in agents' memories explicit as we do. Moreover, agents in Mattar and Daw [16] replay memories to update their q-value estimates and hence their policy. Thus, prioritization of memories in [16] stems only due to incomplete learning and their results would not hold for well-trained agents, in contrast to our work where these effects are driven by a limited capacity constraint.
Other groups have proposed alternative approaches to deal with memory limitations in RL, such as using regularization (SAC [17]), or using neural networks for representing policy and value functions, and even compressing state representations with graph-Laplacian [18]. Our work is meant to complement these previous studies. SAC, for instance, directly penalizes the policy entropy while maximizing reward to encourage exploration. In DRA, we penalize precise representations of q-values instead. The use of a compressed graph-Laplacian [18], on the other hand, hints at yet another problem involving efficient use of memory - compact representation of states (e.g., chunking) which we plan to combine with our approach in future work. On the technical side, DRA is related to O'Donoghue et al.'s uncertainty-based exploration in the manner of decoupling updates to different moments of the value distribution [19]. Finally, it is worth noting that our work fundamentally differs from previous work in RL applied to external resource allocation [20, 21] in that we are using RL to optimize the computational resources of the agent itself.

## 2 Background and details

### 2.1 Environment and agent's memories

We consider problems that can be described by a Markov Decision Process (MDP) characterized by a quadruple $\left\langle\mathcal{S}, \mathcal{A}, \mathcal{R}_{a}, \mathcal{P}_{a}\right\rangle$, where $\mathcal{S}$ is a finite set of states that describe the environment, $\mathcal{A}$ is a finite set of actions available to the agent, $\mathcal{R}_{a}\left(s, s^{\prime}\right)$ is the immediate reward received after taking action $a$

[^0]
[^0]: ${ }^{1}$ Code to run DRA and reproduce our results is available at https://github.com/nisheetpatel/DynamicResourceAllocator.

---

#### Page 3

in state $s$ and transitioning to state $s^{\prime}$, and $\mathcal{P}_{a}=p\left(s^{\prime} \mid s, a\right)$ denotes the dynamics of the environment, i.e. probability of transitioning from state $s$ to $s^{\prime}$ after taking action $a$ [22].

For simplicity, we assume that agents have a model of the world, though this assumption is not necessary except in model-based planning tasks such as the one in Section 4. We also assume that the agents perfectly observe the states. However, agents may only represent the value of each state-action pair, the $q$-value, with finite precision. This means that the q-values are represented as distributions rather than point estimates, and agents can only access samples from the distribution stored in memory but cannot access its parameters such as the mean and precision directly. We would like to emphasize that, for such agents, storing items in memory more precisely requires more resources.

Concretely, we define the agent's memory in a tabular form comprising the states, actions, immediate (mean) rewards, next states, and the corresponding imprecise q-value, represented here by a normal distribution with mean $\bar{q}_{\text {est }}$ and variance $\sigma_{\text {est }}^{2}$. Thus, each memory can be written as a quintuple $\left\langle s, a, r, s^{\prime}, \mathcal{N}\left(\bar{q}_{\text {est }}, \sigma_{s a}^{2}\right)\right\rangle$, and the total number of memories equals $|\mathcal{S} \times \mathcal{A}|$. In this work, we assume the overall q-value distribution to be a multivariate normal with diagonal covariance matrix (that is, independent memories). However, our approach for allocating resources across a collection of uncertain memories can be generalized to arbitrary distributions.

# 2.2 Objective and policy

The goal of the agent is to maximize their expected sum of future rewards in a given task episode, subject to a cost of representing the q-values precisely. Biological agents pay such costs for recruiting more neurons for representation and computation of task-relevant statistics, or for creating the relevant synaptic connections between existing neurons for learning [23]. In general, our method can be applied to arbitrarily defined cost functions, but here we focus on a cost derived from neural and information-theoretical principles.

Biological agents must use a pre-existing population of neurons from a region of their brain where q-values are represented [24-26] to store their base q-value distribution in its connections and activity. We will refer to this base distribution as $P:=\mathcal{N}\left(\overline{\boldsymbol{q}}, \sigma_{\text {base }}^{2} I\right)$. As brains learn to perform well on the task, they update the synaptic connections and may recruit more neurons if necessary to represent the q-value distribution in memory as $Q:=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right)$, with higher precision. In this work, we reason that the agent (and the brain) pays a cost proportional to the KL-divergence $D_{\mathrm{KL}}(Q \| P)$, representing the information-theoretical cost for encoding deviations from the base distribution (e.g., due to modified connectivity and neuronal activity).

Thus, the full objective that the agent is trying to maximize in each episode is:

$$
\mathcal{F}:=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t+1} \mid Q=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right)\right]-\lambda D_{\mathrm{KL}}\left(Q=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right) \| P=\mathcal{N}\left(\overline{\boldsymbol{q}}, \sigma_{\text {base }}^{2} I\right)\right)
$$

where $\gamma \in[0,1]$ is the discount factor which we set to 1 in this article, the first term represents the expected reward for the MDP given the agent's memory distribution $Q$ and policy $\pi$, and the second term represents the cost function described above with a cost per nat equal to $\lambda \geq 0$.

Given that their memories are noisy, agents can only draw samples from their memory distribution of q-values. If the agent is greedy, they will then choose the action corresponding to the largest sampled q-value, effectively yielding the policy $\pi$ :

$$
\pi(a \mid s)=\operatorname{Pr}\left(a=\underset{a^{\prime}}{\arg \max } \tilde{q}\left(s, a^{\prime}\right)\right) \quad \text { with } \quad \tilde{q}\left(s, a^{\prime}\right) \sim \mathcal{N}\left(\bar{q}\left(s, a^{\prime}\right), \sigma^{2}\left(s, a^{\prime}\right)\right)
$$

This behavioral policy is also known as Thompson sampling [27, 28], which is often chosen deliberately for efficient exploration, but here, it is a consequence of having imprecise memories. In principle, agents could draw more than one sample from memory to increase the precision of their estimates of q-values, but we assume that drawing multiple independent samples would require additional time [29]. In this paper, we restrict our analyses to single Thompson samples at decision time, leaving a detailed analysis of the speed-accuracy trade-off [30] for future work.
Our key contribution in this work is providing the agent with control over the precision of each of its memories, the resource allocation vector $\boldsymbol{\sigma}$. Thus, the agent may allocate more resources to some memories than others so as to maximize the objective $\mathcal{F}$ defined in Eq. 1.

---

#### Page 4

# 2.3 Maximizing the objective

First, we consider the scenario where the agent wishes to allocate resources optimally - by maximizing the objective $\mathcal{F}$ in Eq. 1 - when the mean q-values, $\overline{\boldsymbol{q}}$, are fixed and known. To do so, we analytically derive a stochastic gradient for $\mathcal{F}$.
To compute the gradient of the first term of $\mathcal{F}$, i.e. the expected reward, we follow the approach as in the policy gradient theorem [31][22, section 13.2], with the difference being that our gradient is with respect to the resource allocation vector, $\sigma$, instead of, for instance, the parameters of a policy network. In general, this gradient may be written as:

$$
\nabla_{\sigma} \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t+1} \mid Q=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right)\right]=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \Psi_{t} \nabla_{\sigma} \log \pi\left(a_{t} \mid s_{t}\right) \mid Q=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right)\right]
$$

where $\Psi_{t}$ can take many forms with different computational properties [32], including but not limited to:

$$
\Psi_{t}= \begin{cases}\sum_{t^{\prime}=t}^{\infty} \gamma^{t-t^{\prime}} r_{t^{\prime}+1} & \text { R-gradient (REINFORCE) } \\ \bar{q}\left(s_{t}, a_{t}\right) & \text { Q-gradient (mean q-value) } \\ A\left(s_{t}, a_{t}\right) & \text { A-gradient (advantage function). }\end{cases}
$$

In our case, we define the advantage function as $A\left(s_{t}, a_{t}\right)=\bar{q}\left(s_{t}, a_{t}\right)-|\mathcal{A}|^{-1} \sum_{a} \bar{q}\left(s_{t}, a\right)$ (see justification for using the mean $\bar{q}$ in Appendix A.3). In more complicated tasks that involve planning (e.g., Section 4), agents may sample multiple future trajectories using the policy $\pi$, followed by performing a non-linear operation such as picking the maximal reward of all sampled trajectories. Thus, it would be inappropriate to use the advantage function, which only characterizes the expected reward until termination for a single trajectory (minus a baseline). In such scenarios, we replace $\Psi_{t}$ with the reward that results from planning, $\max _{i}\left(\sum_{t=0}^{T} r_{t+1, i}\right)$, where $i$ indexes planned trajectories and $T$ is the planning horizon, following the reasoning as in the original REINFORCE algorithm [31].
Eq. 3 allows us to compute an unbiased stochastic estimate of the gradient - the expectation on the right-hand side - via Monte Carlo, i.e. by sampling one trajectory or averaging over multiple sampled trajectories. Note that if the agents do not have a model of the world, they may simply store previously experienced sequences of states in a buffer (episodic memory) and use these stored trajectories instead of generating new ones with their model. Unless otherwise mentioned, we use $N_{\text {traj }}=10$ trajectories to compute the stochastic approximation for the expectation.
We obtain an analytical approximation for $\nabla_{\sigma} \log \pi(a \mid s)$ (Eq. 3) for our policy by first reparametrizing our q-value distribution as Kingma and Welling prescribe for normal distributions [33], and then using soft Thompson sampling [27, 28], i.e. using softmax or soft-arg max instead of arg max in Eq. 2 (see Appendix A. 1 for full derivation). This modification yields:

$$
\frac{\partial}{\partial \sigma\left(s^{\prime}, a^{\prime}\right)} \log \pi(a \mid s)= \begin{cases}\beta \zeta_{s a}(1-\pi(a \mid s)) & \text { for } s^{\prime}=s, a^{\prime}=a \\ -\beta \zeta_{s a^{\prime}} \pi\left(a^{\prime} \mid s\right) & \text { for } s^{\prime}=s, a^{\prime} \neq a \\ 0 & \text { for } s^{\prime} \neq s\end{cases}
$$

where $\zeta_{s a} \stackrel{\text { i.i.d. }}{=} \mathcal{N}(0,1) \forall s \in \mathcal{S}, a \in \mathcal{A}$, and $\beta$ is the inverse-temperature parameter for softmax. Plugging this into Eq. 3 for each step in the sampled trajectories yields the stochastic gradient of the first term of $\mathcal{F}$, the expected reward, with respect to $\sigma$.
Computing the gradient of the second term of $\mathcal{F}$, the cost, is straightforward (see Appendix A.2):

$$
\frac{\partial}{\partial \sigma_{s a}} D_{\mathrm{KL}}\left(Q=\mathcal{N}\left(\overline{\boldsymbol{q}}, \boldsymbol{\sigma}^{2} I\right) \| P=\mathcal{N}\left(\overline{\boldsymbol{q}}, \sigma_{\text {base }}^{2} I\right)\right)=\frac{\sigma_{s a}}{\sigma_{\text {base }}^{2}}-\frac{1}{\sigma_{s a}}
$$

Now, the agent may iteratively update its resource allocation vector, $\boldsymbol{\sigma}$, as:

$$
\boldsymbol{\sigma} \leftarrow \boldsymbol{\sigma}+\alpha \nabla_{\sigma} \mathcal{F}
$$

where $\alpha>0$ is a learning rate and $\mathcal{F}$ is the objective in Eq. 1 that depends on the resource allocation vector $\boldsymbol{\sigma}$, the stochastic gradient for which is given by Eqs. 3, 5, and 6.
Alternatively, we can maximize $\mathcal{F}$ via a black-box optimizer such as Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [34, 35], which works well for up to hundreds of dimensions (memories, in our case). Results from CMA-ES provide us with a baseline for our gradient-based optimization in small-scale environments (Section 4), since both methods are able to find the optimal resource allocation at stationary state, i.e. when the mean q-values, $\overline{\boldsymbol{q}}$, are fixed and known.

---

#### Page 5

# 2.4 Dynamic allocation of limited memory resources

Real-world environments are rarely stationary. Thus, any agent with limited memory resources must assign importance dynamically to items in memory in a time-efficient manner. Our framework makes it is possible to do so by decoupling the updates for the mean and the variance of the memory distribution $Q$ (as, for instance, done by O'Donoghue et al. [19]). Agents can update the mean of $Q$ on the fly with any on-policy learning algorithm (e.g. SARSA [4] or expected SARSA [36]) and simultaneously update the variance as in Eq. 7. Combining these elements, we propose Algorithm 1, Dynamic Resource Allocator (DRA), to enable memory-limited agents to find good policies.

```
Algorithm 1: Dynamic Resource Allocator (DRA)
    Set hyper-parameters \(\boldsymbol{\theta}=\left(\alpha_{1}, \alpha_{2}, \beta, \gamma, \lambda\right)\)
    Initialize \(\boldsymbol{\bar{q}}, \boldsymbol{\sigma}\), table of memories \(=\left\langle s, a, r, s^{\prime}, \mathcal{N}\left(\bar{q}_{s a}, \sigma_{s a}^{2} I\right)\right\rangle \) \mathcal{S} \times \mathcal{A}\)
    for episode \(k=1, \ldots, K\) do
        \(s \leftarrow s_{0}\)
        while \(s\) is not Terminal do
            \(a \leftarrow \pi(s, Q, \beta)\)
            \(s^{\prime}, r \leftarrow \operatorname{Environment}(s, a)\)
            \(\delta \leftarrow r+\gamma \sum_{a^{\prime}} \pi\left(a^{\prime} \mid s^{\prime}\right) q\left(s^{\prime}, a^{\prime}\right)-q(s, a)\)
            \(q(s, a) \leftarrow q(s, a)+\alpha_{1} \delta\)
            \(s \leftarrow s^{\prime}\)
    end
    Sample \(N\) trajectories to compute:
        \(\nabla_{\sigma} \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t+1} \mid Q\right]=\frac{1}{N} \sum_{n=1}^{N}\left[\sum_{t=0}^{\infty} \Psi_{t} \nabla_{\sigma} \log \pi\left(a_{t} \mid s_{t}\right)\right]\)
                                    where \(\nabla_{\sigma} \log \pi\left(a_{t} \mid s_{t}\right) \leftarrow\) Eq. 4
    \(\boldsymbol{\sigma} \leftarrow \boldsymbol{\sigma}+\alpha_{2}\left(\nabla_{\sigma} \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t+1} \mid Q\right]+\lambda \nabla_{\sigma} D_{\mathrm{KL}}(Q \| P)\right)\)
                                    where \(\nabla_{\sigma} D_{\mathrm{KL}}(Q \| P) \leftarrow\) Eq. 6
end
```

## 3 Results on standard RL environments

### 3.1 2D Grid-world

First, we consider the grid-world adapted from Mattar and Daw [16] and depicted in Fig. 1a. The goal of the agent is to find the shortest route from the starting location indicated by the position of the rat to the cheese, since all transitions yield a reward -1 except reaching the cheese, which rewards 10 points. In each state, the agent can only choose to go up, down, left, or right, and if their intended action is blocked by an obstacle, that action leaves their position unchanged.

Our results show that the initial amount of resources agents can afford at the beginning of training has a critical influence on learning speed. We choose four different values of $\sigma_{0}$ (see Fig. 1b legend) to initialize $\sigma(s, a)=\sigma_{0} \forall(s, a)$ for the memory distribution, and for each of them, we perform five optimization runs for 5000 episodes each. As shown in Fig. 1b, all $\sigma_{0}$ conditions eventually converge to the same solution (asymptotic $\sigma_{*}$ is largely independent of the initial $\sigma_{0}$ ), but starting with more resources (low $\sigma_{0}$ ) leads to faster learning, at a higher initial cost. In real-world animal experiments, this is a very common observation: more neurons are responsive to task-relevant variables in the early stages of training than when the animal has been well-trained [37, 38], suggesting that biological brains may indeed deliberately assign more resources early to enable quicker learning.
Further, we test the ability of DRA to dynamically reallocate resources by changing the rewards and transition structure of the task: we remove the obstacle directly adjacent to the cheese and re-position the cheese two states to the left after 3000 episodes in a separate experiment. This modification changes the shortest path to the cheese, and we expect to see a corresponding change in resource allocation and the policy. To depict this change graphically, we compute the entropy of the agent's policy at each state during different stages of training. Namely, we compute the choice probability vector as the normalized histogram of $10^{5}$ actions from each state and then compute the entropy of those vectors. In Fig. 1c, memories corresponding to states to which DRA allocates more resources

---

#### Page 6

> **Image description.** The image presents a multi-panel figure illustrating a 2D grid-world task and the learning behavior of an agent within it. The figure is divided into three sections labeled (a), (b), and (c).
>
> Panel (a), titled "Task," shows a schematic representation of the grid-world environment. The grid consists of several square cells, with a mouse located in the bottom left corner and a piece of cheese in the top right corner. Gray rectangular shapes represent obstacles within the grid.
>
> Panel (b), titled "Convergence," displays a line graph. The x-axis represents the "Number of episodes," ranging from 0 to 4000. The left y-axis represents "Objective (-)" ranging from -50 to 0, and the right y-axis represents "Cost (--)" ranging from 0 to 30. There are four solid lines plotted on the graph, each representing a different value of σ₀ (1, 3, 4, and 5), with error bars indicating standard deviation. Dashed lines also appear on the graph, corresponding to cost values. A vertical dashed blue line is present near the beginning of the x-axis.
>
> Panel (c), titled "Normalized choice entropy as a function of learning," consists of four grid-world visualizations. Each grid-world shows the same environment as in panel (a), with the mouse and cheese present. The cells are colored with a gradient from light orange to dark purple, representing the normalized choice entropy, with darker colors indicating lower entropy (more precise memories). A color bar on the right side of the panel shows the mapping between color and entropy values, ranging from 0 (dark purple) to 1 (light orange). The four visualizations are labeled with the episode number: "Episode 300," "Episode 3000," "Episode 3300" (labeled as "Environment changed after 3000 episodes"), and "Episode 10000."

Figure 1: 2D grid-world. (a) Task (adapted from [16]). (b) Rate of convergence of the objective $\mathcal{F}$ (solid lines, Eq. 1) that the agent aims to maximize and the cost (dashed lines) it pays to encode memories with higher precision. Error bars represent SD across mean of 5 optimization runs. (c) Normalized entropy of the agent's policy through training and re-training after change in the environment. Darker colors indicate lower entropy and thus more precise memories.

are indicated by lower entropy, i.e. less randomness, of the policy. Early on, the agent's memories get more precise for states that are close to the reward (top-left panel), whereas an over-trained agent only remembers the shortest path to the reward (top-right panel). Moreover, immediately after the change in the environment, we expect agents to follow the previously optimal path (top-right panel) and turn left at the end, and eventually follow the shorter paths. The bottom panels show that agents learn to reallocate resources to better paths over time but they still retain traces of the older memories.
We also compare DRA against a model that allocates resources equally to all memories ('equalprecision'), but the precision shared across all memories is otherwise subject to the same optimization procedure as DRA. We find that DRA achieves $2 \times$ improvement in the objective (Eq. 1) over the equal-precision model. Finally, we construct another baseline model by letting $\lambda \rightarrow 0$ in DRA, which reduces it to SARSA, and report that DRA only takes $1.5 \times$ the number of episodes to converge as compared to the baseline model while making efficient use of memory resources. Similar findings hold for all the tested hyperparameters in a wide range (see Appendix B.2).

# 3.2 Mountain car

Next, we test DRA on the mountain car problem [39], where an under-powered car needs to reach the flag on top of the hill (Fig. 2a). At each time-step, the car can accelerate left or right by a fixed amount, or do nothing. It always starts at the bottom of the hill $x=-0.5$ with velocity $v=0$ and must swing left and right, gaining momentum to progressively reach higher. In Figs. 2b-c, we show the mean of the value function for each state, computed as $\bar{v}(s)=\max _{a} \bar{q}(s, a)$, as well as the entropy of the agent's policy for each state computed as described in Section 3.1. We find that DRA sensibly encodes memories corresponding to states that are close to the states in the optimal trajectory for this task with higher precision.
Further, in Fig. 2d we show the performance of alternative gradients (Eq. 4) that may be used to allocate resources with DRA. Our results are in line with previous work in that the advantage gradient outperforms other approaches [32]. Finally, we perform the same analyses as in Section 3.1 by comparing DRA against an 'equal-precision' model that allocates resources equally to all memories, where DRA achieves a $1.3 \times$ improvement in the objective; and by comparing DRA against a baseline

---

#### Page 7

model $(\lambda=0)$, where DRA takes $1.3 \times$ the number of episodes to converge while making efficient use of memory resources.

> **Image description.** The image contains four panels, labeled (a) through (d), depicting aspects of a reinforcement learning task, specifically the "Mountain Car" problem.
>
> Panel (a) shows a schematic of the Mountain Car environment. A gray rectangle labeled "Wall" is on the left. A curved line represents the terrain, with a yellow star labeled "Start" at the bottom of the valley. A small blue car is shown at several points along the terrain, suggesting its movement. A pink flag labeled "Goal" is at the top of the hill on the right. A dashed line approximates the shape of the valley. The text "-sin(3x)" is below the x-axis.
>
> Panel (b) is a 2D heatmap representing the "Value function." The x-axis is labeled "Position" and ranges from approximately -1.2 to 0.6. The y-axis is labeled "Velocity" and ranges from approximately -0.07 to 0.07. The heatmap is colored from yellow to dark blue, with a colorbar on the right ranging from 0 to -120. A white spiral with an arrow at the end is overlaid on the heatmap.
>
> Panel (c) is another 2D heatmap, representing "Choice entropy." The x and y axes are labeled "Position" and "Velocity" respectively, with the same ranges as in panel (b). The heatmap is colored from yellow to dark blue, with a colorbar on the right ranging from -0 to -1.
>
> Panel (d) is a bar chart. The x-axis is labeled "Gradient" and has three categories: "R", "Q", and "A". The y-axis is labeled "Objective" and ranges from -150 to 0. Each category has a colored bar: "R" is green, "Q" is orange, and "A" is blue. Error bars are present on each bar, indicating the standard deviation.

Figure 2: Mountain car. (a) Task. (b) Value function learnt by the agent with a close-to-optimal route indicated by the white arrow. (c) Entropy of the agent's policy indicating precision of corresponding memories. (d) Maximum objective achieved by using the Advantage function (A), the mean q-value $(\mathrm{Q})$, and REINFORCE (R) to compute the stochastic gradient of the first term of $\mathcal{F}$ - the expected reward (Eqs. 1-4). Errorbars represent SD of the mean across 5 optimization runs.

# 4 Results on a model-based planning task

### 4.1 Task details

To study the effect of resource allocation in model-based planning, we consider here the task devised by Huys et al. [40], whose deterministic state transitions and immediate rewards are described in Fig. 3a. Participants never see this underlying structure, but are trained extensively on the transition structure and immediate rewards until they pass a test. Subjects are asked to perform $M \in\{3,4,5\}$ moves on each trial, indicated in advance, with the goal of maximizing cumulative rewards. Crucially, the subjects must plan their sequence of moves in a fixed time-period of 9 s , during which they cannot act, and get a subsequent 2.5 s to execute the entire sequence of actions. With perfect knowledge of the task, the optimal sequence can be found by simply exploring all $2^{M}$ sequences and selecting the one associated with the highest reward (e.g., from state 5 , with $M=3$, one should pick state $6,1, \&$ 2). Because of the time pressure, however, subjects are unable to explore all possible moves.

We construct an MDP for this task by expanding the state space by a factor of $M$, allowing the task to be Markovian. In this section, we consider the MDP for $M=3$ moves, but the results hold true generally (see Appendix B.1). Agents thus have $N_{\text {mem }}=|\mathcal{S}| \times M \times|\mathcal{A}|=36$ memories of the form $\left\langle s, a, s^{\prime}, r, \mathcal{N}\left(q_{s a}, \sigma_{s a}^{2}\right)\right\rangle$, with perfect recollection of $s^{\prime}, r$ given $s, a$, but without prior knowledge of the q-values which are initialized randomly around $\bar{q}(s, a)=0 \forall s, a$ with some precision $\sigma(s, a)=\sigma_{0} \forall s, a$.

In order to plan, agents start in the initial state $s_{0}$ and choose next states according to their policy until they reach a terminal state, which is when they reset $s=s_{0}$ and continue planning. Agents keep track of all paths traversed and rewards accumulated for each path until they reach the time limit, which we implement by limiting the number of states that the agent can explore: the search budget. When this search budget is exhausted, agents choose the sequence of actions corresponding to the most-rewarding path in their working memory as shown by the schematic in Fig. 3b.

### 4.2 Comparison with an alternative model and black-box optimization

We compare the results obtained via our gradient-based resource allocation to a black-box optimization procedure (CMA-ES, described in Section 2.3). For both methods, we fix the mean of the q-value distribution to the optimal point-estimate $\overline{\boldsymbol{q}}^{*}$ obtained via q-learning [3], and maximize the objective $\mathcal{F}$ (Eq. 1) with respect to the resources, $\boldsymbol{\sigma}$. Fig. 3c shows that both methods perform comparably when we sample $N_{\text {traj }} \geq 10$ trajectories to estimate the stochastic gradient of $\mathcal{F}$ (see Section 2.3).

Our agent is limited in its precision of q-values, but only some of them need to be encoded precisely, namely the ones associated with decisions that have a high impact on cumulative rewards, e.g. for

---

#### Page 8

> **Image description.** This image presents a figure with six panels (a-f) that illustrate a planning task and the performance of different optimization strategies.
>
> Panel (a) shows a diagram of the task structure. Six numbered states (1-6) are represented as squares. Arrows indicate possible transitions between states, with solid arrows representing "Left" actions and dashed arrows representing "Right" actions. Numbers adjacent to the arrows indicate the reward associated with that transition. For example, moving from state 2 to state 1 yields a reward of +140.
>
> Panel (b) shows a decision tree representing paths explored in an example trial. The tree starts at state 1. Each node in the tree represents a state, and branches represent possible actions. The chosen path, which maximizes reward, is highlighted with a surrounding box. The rewards associated with the terminal states of the chosen path are shown (r=0, r=50, r=100). The paths are also color coded, Path 1 is red, Path 2 is green, and Path 3 is blue.
>
> Panel (c) is a bar graph comparing the optimal objective found by estimating the stochastic gradient (∇σ) with different numbers of sampled trajectories (Ntraj = 1, 10, 50) and CMA-ES. The y-axis is labeled "Objective". Error bars are present on each bar, representing the standard deviation of the mean across optimization runs.
>
> Panel (d) is a line graph showing the difference in the optimal objective (blue line) found by DRA from the optimal objective found by a model with equally precise memories. An orange line represents the same difference for expected reward. The x-axis is labeled "Search budget (%)", and the y-axis is labeled "Perf. difference (Flexible - Equal)".
>
> Panel (e) contains two scatter plots. The left plot shows the discriminability of memories (d') as a function of their impact on cumulative reward. The right plot shows the average resources allocated to memories as a function of their impact on cumulative reward. Each plot contains data points for different search budgets (25%, 50%, 75%, 100%), with linear regression fits shown for each search budget.
>
> Panel (f) is a line graph showing the optimal objective (F, Eq. 1) and a modified objective (F̃, Eq. 8) as a function of search budget. The x-axis is labeled "Search budget (%)". The left y-axis is labeled "Objective (exp. reward)", and the right y-axis is labeled "Objective (exp. reward rate)". Shaded areas around the lines represent the standard deviation of the mean across optimization runs. A yellow star indicates the peak of the orange line.

Figure 3: Planning task. (a) Task structure. (b) Paths explored and chosen in an example trial. (c) Comparison of the optimal objective $\mathcal{F}_{\star}=\arg \max _{\sigma} \mathcal{F}$ found by estimating the stochastic gradient $\nabla_{\sigma} \mathcal{F}$ with $N_{\text {top }} \in\{1,10,50\}$ sampled trajectories and CMA-ES. (d) Difference in the optimal objective (blue) found by DRA from the optimal objective found by a model that is constrained to have equally precise memories; orange is the same for expected reward. As the search budget increases (planning-time pressure decreases), the advantage diminishes. (e) Linear regression fits for the discriminability of memories ( $\mathrm{d}^{\prime}$, left) and average resources allocated to memories (right) as a function of their impact on cumulative reward. As the search budget increases, DRA gives up differential allocation of resources to items in memory as it is no longer advantageous. (f) $\mathcal{F}_{\star}$ (green, left axis) and $\widetilde{\mathcal{F}}_{\star}$ (orange, right axis) as a function of search budget. Errorbars/shaded areas represent SD of mean across 5 optimization runs in (c),(d), and (f), and across memories in (e).

$M=2, s_{0}=6$, moving to state 1 vs. 3 results in a large difference ( 120 points) in cumulative reward. To show that it is indeed beneficial to prioritize some memories over others by encoding them with higher precision, we also trained an agent that is constrained to have all its memories equally precise ('equal-precision'), though how precise is subject to the same optimization procedure. Fig. 3d shows the advantage in performance of the agent that allocates resources flexibly (DRA) from the one that is constrained to allocate resources equally. This advantage is more pronounced when the agent is under time pressure (lower search budget) to plan its sequence of moves as expected.
Furthermore, this task reveals that the key factor for resource allocation is not memory precision per se, but the discriminability $d^{\prime}$ (a ratio of difference in means divided by the effective standard deviation) between q-values of a state. In Fig. 3e, we see this effect manifested strongly when agents have a low search budget, and it flattens as the search budget increases to $100 \%$, i.e. when agents can explore all possible paths, such that the reward they obtain is unaffected by the precision of q-values. Since $d^{\prime}$ is correlated with difference in cumulative reward, we also plot the mean $\sigma$ of memories (across actions) to show that this effect indeed stems from resource allocation. Intuitively, DRA allocates more precision to memories when larger differences in rewards are at stake (Fig. 3e).

# 4.3 The speed-accuracy trade-off

So far, we have restricted our analyses to cases where agents are given a fixed search budget to plan their moves, and they optimize their objective $\mathcal{F}$ independently for each fixed budget. As expected, their performance increases monotonically with the time they spend planning and eventually saturates as shown in Fig. 3f (green curve). In most real-world scenarios, however, the search budget is unknown and therefore agents need to decide when to stop planning and execute an action. We can incorporate this speed-accuracy trade-off $[30,41]$ in our framework by modifying the objective $\mathcal{F}$ as:

---

#### Page 9

$$
\widetilde{\mathcal{F}}:=\frac{\mathbb{E}_{\pi}\left[\max _{\text {path }=1}^{h M}\left(\sum_{t=1}^{M} r_{t, \text { path }}(Q)\right]\right.}{\left\langle a_{\text {dec }} b+t_{\text {non-dec }}\right\rangle}-\lambda D_{\mathrm{KL}}(Q \| P)
$$

where the numerator of the first term represents the expected reward from planning specific to this task as described in Section 2.3 and Fig. 3b, and the denominator represents the average time the agent spends per trial written as the sum of the average decision-time (proportional to the search budget $b$ with proportionality constant $a_{\text {dec }}$ ) and non-decision time $t_{\text {non-dec }}$ (e.g. due to sensory delay and executing motor output). In many cases such as in this task, agents can follow locally estimated gradients with respect to $b$ as they allocate resources dynamically with DRA to maximize $\widetilde{\mathcal{F}}$ since this turns out to be a convex optimization problem (Fig. 3f, orange curve).
Note that in this analysis, agents draw single Thompson samples from all the memories at each state during planning (see Eq. 2). In principle, they could spend more time at some states than others, in which case, the effective precision of memories would not only be controlled by the number of neurons, but also the time available to plan. However, a detailed analysis of such flexible planning is outside the scope of the current work.

# 5 Discussion

In this article, we propose a framework to model uncertainty in action-values stored in limited memories, show how resource-limited agents can plan and act with such uncertainty, and how they benefit by prioritizing memories differentially. Our work provides a novel, normative approach to dynamically allocate limited memory resources for finding good policies for decision-making. Though we only consider normally distributed and independently encoded action-values, our framework can easily be extended to arbitrary value distributions and resource costs.
Previous work has considered prioritization of memory access guided solely by the value of backups that lead to a change in the agent's policy [16]. However, such a form of prioritization predicts that all memories are equally relevant when animals are well-trained, which contradicts empirical findings [42, 43]. In order to model and understand animal behavior, it is important to consider irreducible sources of uncertainty in the value function besides learning, due either to resource limitations or stochastic rewards and dynamics [19, 44]. In future work, we would like to combine all these sources of uncertainty in order to make experimentally testable predictions for animal behavior.
Our work partly explains a commonly reported phenomenon in neuroscience: while early sensory and somatosensory brain areas recruit more neurons over the course of training, other brain areas responsible for higher-level cognitive processes such as accessing and storing memories (prefrontal cortex) and evidence integration during decision-making (posterior parietal cortex), either commit more neurons or show higher levels of activity during earlier stages of learning than later stages when animals are well-trained [37, 38, 45]. By showing that resource-constrained agents can accelerate learning by starting with more resources (Fig. 1b), we provide a plausible hypothesis for this observation.

Finally, our framework also makes clear predictions about how action-values should be represented probabilistically in the brain. The details of the predictions will depend on the nature of neural code for probability distributions. Given that action-values are scalar variables, probabilistic population codes - in which each neuron or sub-population is tuned to a specific action-value - provide a biologically plausible neural code, for which there exists strong experimental evidence [46, 47]. For this type of code, the amplitude of the neuronal response encoding a specific action-value should be inversely proportional to the variance associated with this action-value [46]. Such predictions could be tested in animals trained on a foraging task while recording in sensorimotor areas like LIP [24], the superior colliculus [48, 49], or the basal ganglia [25] where neurons are known to encode both actions and expected rewards. One might also imagine that, throughout the course of learning, the number of neurons encoding action-values, or the information-limiting correlations among these neurons [50], are modulated so as to reflect the precision with which these action-values are encoded.

---

# Dynamic allocation of limited memory resources in reinforcement learning - Backmatter

---

#### Page 10

# Broader Impact 

We believe that this work has the potential to lead to a net-positive change in the reinforcement learning community and more broadly in society as a whole. Our work enables researchers to represent the uncertainty in memories due to resource constraints and perform well in the face of such constraints by prioritizing the knowledge that really matters. While our work is preliminary, we believe that furthering this line of work may prove to be highly beneficial in reducing the overall carbon footprint of the artificial intelligence (AI) industry, which has recently come under scrutiny for the jarring energy consumption of several common large AI models that produce up to five times as much $\mathrm{CO}_{2}$ than an average American car does in its lifetime [51, 52].
In terms of ethical aspects, our method is neutral per se. The advancement of energy-efficient algorithms may enable autonomous agents to function for long hours in remote areas, the applications for which could be used for both constructive and destructive things alike, e.g. they may be deployed for rescue missions [53] or weaponized for military applications [54, 55], but this holds true for any RL agent.

## Acknowledgments and Disclosure of Funding

We thank Pablo Tano and Reidar Riveland for useful discussions, and Morio Hamada for providing helpful feedback on a previous version of the manuscript. Nisheet Patel was supported by the Swiss National Foundation (grant number 31003A_165831). Luigi Acerbi was partially supported by the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence (FCAI). Alexandre Pouget was supported by the Swiss National Foundation (grant numbers 31003A_165831 and 315230_197296).

## References

[1] Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237-285, 1996.
[2] Richard S Sutton, Andrew G Barto, et al. Introduction to reinforcement learning, volume 135. MIT press Cambridge, 1998.
[3] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3-4):279-292, 1992.
[4] Gavin A Rummery and Mahesan Niranjan. On-line Q-learning using connectionist systems, volume 37. University of Cambridge, Department of Engineering Cambridge, UK, 1994.
[5] Craig Denis Hardman, Jasmine Monica Henderson, David Isaac Finkelstein, Malcolm Kenneth Horne, George Paxinos, and Glenda Margaret Halliday. Comparison of the basal ganglia in rats, marmosets, macaques, baboons, and humans: volume and neuronal number for the output, internal relay, and striatal modulating nuclei. Journal of Comparative Neurology, 445(3):238-255, 2002.
[6] Samuel J Gershman, Eric J Horvitz, and Joshua B Tenenbaum. Computational rationality: A converging paradigm for intelligence in brains, minds, and machines. Science, 349(6245):273-278, 2015.
[7] Falk Lieder and Thomas L Griffiths. Resource-rational analysis: understanding human cognition as the optimal use of limited computational resources. Behavioral and Brain Sciences, 43, 2020.
[8] Susanne Still and Doina Precup. An information-theoretic approach to curiosity-driven reinforcement learning. Theory in Biosciences, 131(3):139-148, 2012.
[9] Jonathan Rubin, Ohad Shamir, and Naftali Tishby. Trading value and information in MDPs. In Decision Making with Imperfect Decision Makers, pages 57-74. Springer, 2012.
[10] Jordi Grau-Moya, Felix Leibfried, Tim Genewein, and Daniel A Braun. Planning with informationprocessing constraints and model uncertainty in Markov decision processes. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 475-491. Springer, 2016.
[11] Daniel Alexander Ortega and Pedro Alejandro Braun. Information, utility and bounded rationality. In International Conference on Artificial General Intelligence, pages 269-274. Springer, 2011.

---

#### Page 11

[12] Michael T Todd, Yael Niv, and Jonathan D Cohen. Learning to use working memory in partially observable environments through dopaminergic reinforcement. In Advances in neural information processing systems, pages 1689-1696, 2009.
[13] Jordan W Suchow and Tom Griffiths. Deciding to remember: Memory maintenance as a Markov decision process. In CogSci, 2016.
[14] Ronald Van den Berg and Wei Ji Ma. A resource-rational theory of set size effects in human visual working memory. ELife, 7:e34963, 2018.
[15] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
[16] Marcelo G Mattar and Nathaniel D Daw. Prioritized memory access explains planning and hippocampal replay. Nature neuroscience, 21(11):1609-1617, 2018.
[17] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.
[18] Yifan Wu, George Tucker, and Ofir Nachum. The laplacian in rl: Learning representations with efficient approximations. arXiv preprint arXiv:1810.04586, 2018.
[19] Brendan O’Donoghue, Ian Osband, Remi Munos, and Volodymyr Mnih. The uncertainty Bellman equation and exploration. arXiv preprint arXiv:1709.05380, 2017.
[20] Hongzi Mao, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula. Resource management with deep reinforcement learning. In Proceedings of the 15th ACM Workshop on Hot Topics in Networks, pages 50-56, 2016.
[21] Hao Ye, Geoffrey Ye Li, and Biing-Hwang Fred Juang. Deep reinforcement learning based resource allocation for V2V communications. IEEE Transactions on Vehicular Technology, 68(4):3163-3173, 2019.
[22] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction 2nd ed, 2018.
[23] Andrea Hasenstaub, Stephani Otte, Edward Callaway, and Terrence J Sejnowski. Metabolic cost as a unifying principle governing neuronal biophysics. Proceedings of the National Academy of Sciences, 107 (27):12329-12334, 2010.
[24] Leo P Sugrue, Greg S Corrado, and William T Newsome. Matching behavior and the representation of value in the parietal cortex. science, 304(5678):1782-1787, 2004.
[25] Kazuyuki Samejima, Yasumasa Ueda, Kenji Doya, and Minoru Kimura. Representation of action-specific reward values in the striatum. Science, 310(5752):1337-1340, 2005.
[26] Matthew R Roesch, Donna J Calu, and Geoffrey Schoenbaum. Dopamine neurons encode the better option in rats deciding between differently delayed or sized rewards. Nature neuroscience, 10(12):1615-1624, 2007.
[27] William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285-294, 1933.
[28] Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial on Thompson sampling. arXiv preprint arXiv:1707.02038, 2017.
[29] Edward Vul, Noah Goodman, Thomas L Griffiths, and Joshua B Tenenbaum. One and done? Optimal decisions from very few samples. Cognitive science, 38(4):599-637, 2014.
[30] Richard P Heitz. The speed-accuracy tradeoff: history, physiology, methodology, and behavior. Frontiers in neuroscience, 8:150, 2014.
[31] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.
[32] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
[33] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.
[34] Nikolaus Hansen. The CMA evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772, 2016.

---

#### Page 12

[35] Nikolaus Hansen, Youhei Akimoto, and Petr Baudis. CMA-ES/pycma on Github. Zenodo, DOI:10.5281/zenodo.2559634, February 2019. URL https://doi.org/10.5281/zenodo.2559634.
[36] Harm Van Seijen, Hado Van Hasselt, Shimon Whiteson, and Marco Wiering. A theoretical and empirical analysis of Expected Sarsa. In 2009 ieee symposium on adaptive dynamic programming and reinforcement learning, pages 177-184. IEEE, 2009.
[37] Leslie G Ungerleider, Julien Doyon, and Avi Karni. Imaging brain plasticity during motor skill learning. Neurobiology of learning and memory, 78(3):553-564, 2002.
[38] Eran Dayan and Leonardo G Cohen. Neuroplasticity subserving motor skill learning. Neuron, 72(3): $443-454,2011$.
[39] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI gym, 2016.
[40] Quentin JM Huys, Niall Lally, Paul Faulkner, Neir Eshel, Erich Seifritz, Samuel J Gershman, Peter Dayan, and Jonathan P Roiser. Interplay of approximate planning strategies. Proceedings of the National Academy of Sciences, 112(10):3098-3103, 2015.
[41] Jan Drugowitsch, Gregory C DeAngelis, Dora E Angelaki, and Alexandre Pouget. Tuning the speedaccuracy trade-off to maximize reward rate in multisensory decision-making. Elife, 4:e06678, 2015.
[42] Danielle Panoz-Brown, Vishakh Iyer, Lawrence M Carey, Christina M Sluka, Gabriela Rajic, Jesse Kestenman, Meredith Gentry, Sydney Brotheridge, Isaac Somekh, Hannah E Corbin, et al. Replay of episodic memories in the rat. Current Biology, 28(10):1628-1634, 2018.
[43] Anoopum S Gupta, Matthijs AA van der Meer, David S Touretzky, and A David Redish. Hippocampal replay is not a simple function of experience. Neuron, 65(5):695-705, 2010.
[44] Will Dabney, Zeb Kurth-Nelson, Naoshige Uchida, Clara Kwon Starkweather, Demis Hassabis, Rémi Munos, and Matthew Botvinick. A distributional code for value in dopamine-based reinforcement learning. Nature, pages 1-5, 2020.
[45] Farzaneh Najafi, Gamaleldin F Elsayed, Robin Cao, Eftychios Pnevmatikakis, Peter E Latham, John P Cunningham, and Anne K Churchland. Excitatory and inhibitory subnetworks are equally selective during decision-making and emerge simultaneously during learning. Neuron, 105(1):165-179, 2020.
[46] Wei Ji Ma, Jeffrey M Beck, Peter E Latham, and Alexandre Pouget. Bayesian inference with probabilistic population codes. Nature neuroscience, 9(11):1432-1438, 2006.
[47] Han Hou, Qihao Zheng, Yuchen Zhao, Alexandre Pouget, and Yong Gu. Neural correlates of optimal multisensory decision making under time-varying reliabilities with an invariant linear probabilistic population code. Neuron, 104(5):1010-1021, 2019.
[48] Dhushan Thevarajah, Ryan Webb, Christopher Ferrall, and Michael C Dorris. Modeling the value of strategic actions in the superior colliculus. Frontiers in behavioral neuroscience, 3:57, 2010.
[49] Takuro Ikeda and Okihide Hikosaka. Positive and negative modulation of motor response in primate superior colliculus by reward expectation. Journal of Neurophysiology, 98(6):3163-3170, 2007.
[50] Rubén Moreno-Bote, Jeffrey Beck, Ingmar Kanitscheider, Xaq Pitkow, Peter Latham, and Alexandre Pouget. Information-limiting correlations. Nature neuroscience, 17(10):1410, 2014.
[51] Karen Hao. Training a single ai model can emit as much carbon as five cars in their lifetimes. MIT Technology Review, 2019.
[52] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP. arXiv preprint arXiv:1906.02243, 2019.
[53] Hiroaki Kitano, Satoshi Tadokoro, Itsuki Noda, Hitoshi Matsubara, Tomoichi Takahashi, Atsuhi Shinjou, and Susumu Shimada. Robocup rescue: Search and rescue in large-scale disasters as a domain for autonomous agents research. In IEEE SMC'99 Conference Proceedings. 1999 IEEE International Conference on Systems, Man, and Cybernetics (Cat. No. 99CH37028), volume 6, pages 739-743. IEEE, 1999.
[54] Javaid Khurshid and Hong Bing-Rong. Military robots-a glimpse from today and tomorrow. In ICARCV 2004 8th Control, Automation, Robotics and Vision Conference, 2004., volume 1, pages 771-777. IEEE, 2004.
[55] Patrick Lin, George Bekey, and Keith Abney. Autonomous military robotics: Risk, ethics, and design. Technical report, California Polytechnic State Univ San Luis Obispo, 2008.

---

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