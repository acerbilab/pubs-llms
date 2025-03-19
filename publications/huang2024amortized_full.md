```
@inproceedings{huang2024amortized,
  title={Amortized Bayesian Experimental Design for Decision-Making},
  author={Huang, Daolang and Guo, Yujia and Acerbi, Luigi and Kaski, Samuel},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024},
}
```

---

#### Page 1

# Amortized Bayesian Experimental Design for Decision-Making

Daolang Huang<br>Aalto University<br>daolang.huang@aalto.fi

Yujia Guo<br>Aalto University<br>yujia.guo@aalto.fi

Luigi Acerbi<br>University of Helsinki<br>luigi.acerbi@helsinki.fi

Samuel Kaski<br>Aalto University<br>University of Manchester<br>samuel.kaski@aalto.fi

#### Abstract

Many critical decisions, such as personalized medical diagnoses and product pricing, are made based on insights gained from designing, observing, and analyzing a series of experiments. This highlights the crucial role of experimental design, which goes beyond merely collecting information on system parameters as in traditional Bayesian experimental design (BED), but also plays a key part in facilitating downstream decision-making. Most recent BED methods use an amortized policy network to rapidly design experiments. However, the information gathered through these methods is suboptimal for down-the-line decision-making, as the experiments are not inherently designed with downstream objectives in mind. In this paper, we present an amortized decision-aware BED framework that prioritizes maximizing downstream decision utility. We introduce a novel architecture, the Transformer Neural Decision Process (TNDP), capable of instantly proposing the next experimental design, whilst inferring the downstream decision, thus effectively amortizing both tasks within a unified workflow. We demonstrate the performance of our method across several tasks, showing that it can deliver informative designs and facilitate accurate decision-making.

## 1 Introduction

In a wide array of disciplines, from clinical trials (Cheng and Shen, 2005) to medical imaging (Burger et al., 2021), a fundamental challenge is the design of experiments to infer the distribution of some unobservable, unknown properties of the systems under study. Bayesian Experimental Design (BED) (Lindley, 1956; Chaloner and Verdinelli, 1995; Ryan et al., 2016; Rainforth et al., 2024) is a powerful framework in this context, guiding and optimizing experimental design by maximizing the expected amount of information about parameters gained from experiments, see Fig. 1(a). However, the ultimate goal in many tasks extends beyond parameter inference to inform a downstream decisionmaking task by leveraging our understanding of these parameters. For example, in personalized medical diagnostics, a model is built based on historical data to facilitate an optimal treatment for a new patient (Bica et al., 2021). This data typically comprises patient covariates, administered treatments, and observed outcomes. Since the query of such data tends to be expensive due to, e.g., privacy issues, we need to actively design queries to optimize resource utilization. Here, when the goal is to improve decisions, the strategy of experimental designs should not focus solely on inferring the parameters of the model, but rather on guiding the final decision-making for the new patient, to ensure that each query contributes maximally to the diagnostic decision.

---

#### Page 2

> **Image description.** This image presents three diagrams illustrating different workflows for Bayesian Experimental Design (BED). Each diagram is enclosed in a rounded rectangle.
>
> Panel (a) depicts "Traditional BED". It shows a cyclical process with the following elements:
>
> - "Design": Represented by a blue and yellow flask with bubbles, connected by an "Optimize" arrow to a "Posterior" element.
> - "Posterior": Shown as a blue curve resembling a probability distribution on a graph, connected by an "Inference" arrow to "Outcome".
> - "Outcome": Illustrated by a molecular structure, connected by an "Observe" arrow back to "Design", completing the cycle.
>
> Panel (b) illustrates "Amortized BED". It also shows a cyclical process:
>
> - "Design": Represented by the same flask icon as in (a). It is connected by a "Generate" arrow from "Policy".
> - "Policy": Shown as a multi-layered neural network icon, located within a dashed-line rectangle labeled "Offline training stage". It is connected by an "Update history" arrow to "Outcome".
> - "Outcome": Illustrated by the same molecular structure as in (a), connected by an "Observe" arrow back to "Design".
>
> Panel (c) presents "Our decision-aware amortized BED". It shows a more complex workflow:
>
> - "Decision Utility": Represented by an icon of a question mark and checkmarks, located within a dashed-line rectangle labeled "Offline training stage".
> - "TNDP": Shown as a multi-layered neural network icon, also within the "Offline training stage". "Decision Utility" is connected to "TNDP" by a "Feedback" arrow.
> - "TNDP" is connected by an "Estimate" arrow to "Decision", represented by a stack of coins icon, with a right-pointing arrow indicating the decision.
> - "Design": Represented by the same flask icon as in (a) and (b), connected by a "Generate" arrow from "TNDP".
> - "Outcome": Illustrated by the same molecular structure as in (a) and (b), connected by an "Observe" arrow back to "Design". "Outcome" is also connected to "TNDP" by an "Update history" arrow.

Figure 1: Overview of BED workflows. (a) Traditional BED, which iterates between optimizing designs, running experiments, and updating the model via Bayesian inference. (b) Amortized BED, which uses a policy network for rapid experimental design generation. (c) Our decision-aware amortized BED integrates decision utility in training to facilitate downstream decision-making.

Traditional BED methods (Rainforth et al., 2018; Foster et al., 2019, 2020; Kleinegesse and Gutmann, 2020) do not take down-the-line decision-making tasks into account during the experimental design phase. As a result, inference and decision-making processes are carried out separately, which is suboptimal for decision-making in scenarios where experiments can be adaptively designed. Losscalibrated inference, which was originally introduced by Lacoste-Julien et al. (Lacoste-Julien et al., 2011) for variational approximations in Bayesian inference, provides a concept that adjusts the inference process to capture posterior regions critical for decision-making tasks. Rather than focusing on parameter estimation, the idea is to directly maximize the expected downstream utility, recognizing that accurate decision-making can proceed without exact knowledge of the posterior as not all regions of the posterior contribute equally to the downstream task. Inspired by this concept, we could consider integrating decision-making directly into the experimental design process to align the proposed designs more closely with the ultimate decision-making task.
To pick the next optimal design, standard BED methods require estimating and optimizing the expected information gain (EIG) over the design space, which can be extremely time-consuming. This limitation has led to the development of amortized BED (Foster et al., 2021; Ivanova et al., 2021; Blau et al., 2022, 2023), a policy-based method which leverages a neural network policy trained on simulated experimental trajectories to quickly generate designs, as illustrated in Fig. 1(b). Given an experiment history, these policy-based methods determine the next experimental design through a single forward pass, significantly speeding up the design process. Another significant advantage of amortized BED methods is their capacity to extract and utilize unstructured domain knowledge embedded in historical data. Unlike traditional methods, which never reuse the information from past data, amortized methods can integrate this knowledge to refine and improve design strategies for new experiments. In our problem setting, the benefits of amortization are also valuable where decisions must be made swiftly, such as when determining optimal treatment for patients in urgent settings.
In this paper, we propose an amortized decision-making-aware BED framework, see Fig. 1(c). We identify two key aspects where previous amortized BED methods fall short when applied to downstream decision-making tasks. First, the training objective of the existing methods does not consider downstream decision tasks. Therefore, we introduce the concept of Decision Utility Gain (DUG) to guide experimental design to better align with the downstream objective. DUG is designed to measure the improvement in the maximum expected utility derived from the new experiment. Second, to obtain the optimal decision, we still need to explicitly approximate the predictive distribution of the outcomes to estimate the utility. Current amortized methods learn this distribution only implicitly and therefore require extra computation for the decision-making process. To address this, we propose

---

#### Page 3

a novel Transformer neural decision process (TNDP) architecture with dual output heads: one acting as a policy network to propose new designs, and another to approximate the predictive distribution to support the downstream decision-making. This setup allows an iterative approach where the system autonomously proposes informative experimental designs and makes optimal final decisions. Finally, since our ultimate goal is to make optimal decisions at the final stage, which may involve multiple experiments, it is crucial that our experimental designs are not myopic or overly greedy by only maximizing next-step decision utility. Therefore, we develop a non-myopic objective function that ensures decisions are made with a comprehensive consideration of future outcomes.

Contributions. In summary, the contributions of our work include:

- We propose the concept of decision utility gain (DUG) for guiding the next experimental design with a direct focus on optimizing down-the-line decision-making tasks.
- We present a novel architecture - Transformer neural decision process (TNDP) - designed to amortize the experimental designs directly for downstream decision-making.
- We empirically show the effectiveness of TNDP across a variety of experimental design tasks involving decision-making, where it significantly outperforms other methods that do not consider downstream decisions.

# 2 Preliminaries

### 2.1 Bayesian experimental design

BED (Lindley, 1956; Ryan et al., 2016; Rainforth et al., 2024) is a powerful statistical framework that optimizes the experimental design process. The primary goal of BED is to sequentially select a set of experimental designs $\xi \in \Xi$ and gather outcomes $y$, to maximize the amount of information obtained about the parameters of interest, denoted as $\theta$. Essentially, BED seeks to minimize the entropy of the posterior distribution of $\theta$ or, equivalently, to maximize the information that the experimental outcomes provide about $\theta$.

At the heart of BED lies the concept of Expected Information Gain (EIG), which quantifies the value of different experimental designs based on how much they are expected to reduce uncertainty about $\theta$, measured in terms of expected entropy $(H[\cdot])$ reduction:

$$
\operatorname{EIG}(\xi)=\mathbb{E}_{p(y \mid \xi)}[H[p(\theta)]-H[p(\theta \mid \xi, y)]]
$$

The optimal design is then defined as $\xi^{*}=\arg \max _{\xi \in \Xi} \operatorname{EIG}(\xi)$. In practice, calculating EIG is a computationally challenging task because it involves integrations over $p(y \mid \xi)$ and $p(\theta \mid \xi, y)$, which are both intractable. In recent years, various methods have been proposed to make the computation of the EIG feasible in practical scenarios, such as nested Monte Carlo estimators (Rainforth et al., 2018) and variational approximations (Foster et al., 2019). However, even with these advancements, the computational load remains significant, hindering feasibility in tasks that demand rapid designs. This limitation has pushed forward the development of amortized BED methods, which significantly reduce computational demands during the deployment stage.

### 2.2 Amortized BED

Amortized BED methods represent a significant shift from traditional experimental optimization techniques. Instead of optimizing for each experimental design separately, Foster et al. (2021) developed a parameterized policy $\pi$, which directly maps the experimental history $h_{1: t}=\left\{\left(\xi_{1}, y_{1}\right), \ldots,\left(\xi_{t}, y_{t}\right)\right\}$ to the next design $\xi_{t+1}=\pi\left(h_{1: t}\right)$. To train such a policy network, Foster et al. (2021) proposed using sequential Prior Contrastive Estimation (sPCE) to optimize the lower bound of the total EIG across the entire $T$-step experiments trajectory:

$$
s P C E(\pi, L)=\mathbb{E}_{p\left(\theta_{0: L}\right) p\left(h_{1: T} \mid \theta_{0}, \pi\right)}\left[\log \frac{p\left(h_{1: T} \mid \theta_{0}, \pi\right)}{\frac{1}{L+1} \sum_{\ell=0}^{L} p\left(h_{1: T} \mid \theta_{\ell}, \pi\right)}\right]
$$

where $\theta_{1: L}$ are contrastive samples drawn from the prior distribution. Although the training of such a policy network is computationally expensive, once trained, the network can act as an oracle to quickly propose the next design through a single forward pass, thus amortizing the initial training cost over numerous deployments.

---

#### Page 4

# 2.3 Bayesian decision theory

Bayesian decision theory (Berger, 2013) provides an axiomatic framework for decision-making under uncertainty, systematically incorporating probabilistic beliefs about unknown parameters into decision-making processes. It introduces a task-specific utility function $u(\theta, a)$, which quantifies the value of the outcomes from different decisions $a \in \mathcal{A}$ when the system is in state $\theta$. The optimal decision is then determined by maximizing the expected utility, which integrates the utility function over the unknown system parameters, given the available knowledge $h_{1: t}$ :

$$
a^{*}=\underset{a \in A}{\arg \max } \mathbb{E}_{p\left(\theta \mid h_{1: t}\right)}[u(\theta, a)]
$$

In many scenarios, outcomes are stochastic and it is more typical to make decisions based on their predictive distribution $p\left(y \mid \xi, h_{1: t}\right)=\mathbb{E}_{p\left(\theta \mid h_{1: t}\right)}[p\left(y \mid \xi, \theta, h_{1: t}\right)]$, such as in clinical trials where the optimal treatment is chosen based on predicted patient responses rather than solely on underlying biological mechanisms. A similar setup can be found in (Kuśmierczyk et al., 2019; Vadera et al., 2021). As we switch the belief about the state of the system to the outcomes and to keep as much information as possible, we need to evaluate the effect of $\theta$ on all points of the design space. Thus, instead of the posterior over the latent state $\theta$, we represent our belief directly as $p\left(y_{\Xi} \mid h_{1: t}\right) \equiv\left\{p\left(y \mid \xi, h_{1: t}\right)\right\}_{\xi \in \Xi}$, i.e. a joint predictive (posterior) distribution of outcomes over all possible designs given the current information $h_{1: t}$. ${ }^{1}$ The utility is then expressed as $u\left(y_{\Xi}, a\right)$, which relies on the decision $a$ and all possible predicted outcomes $y_{\Xi}$. It is a natural extension of the traditional definition of utility by marginalizing out the posterior distribution of $\theta$. The rule of making the optimal decision is reformulated in terms of the predictive distribution as:

$$
a^{*}=\underset{a \in A}{\arg \max } \mathbb{E}_{p\left(y_{\Xi} \mid h_{1: t}\right)}[u\left(y_{\Xi}, a\right)]
$$

Traditional methods usually separate the inference and decision-making steps, which are optimal when the true posterior or the predictive distribution can be computed exactly. However, in most cases the posteriors are not directly accessible, and we often resort to using approximate distributions. This necessity results in a suboptimal decision-making process as the approximate posteriors often focus on representing the full posterior yet fail to ensure high accuracy in regions crucial for decisionmaking. Loss-calibrated inference (Lacoste-Julien et al., 2011) emerges as an effective solution to address this problem. It calibrates the inference by focusing on utility rather than mere accuracy of the approximation, thereby ensuring a more targeted posterior estimation. This method has been applied to improving Markov chain Monte Carlo (MCMC) methods (Abbasnejad et al., 2015), Bayesian neural networks (Cobb et al., 2018) and expectation propagation (Morais and Pillow, 2022).

## 3 Decision-aware BED

### 3.1 Problem setup

Our objective is to optimize the experimental design process for down-the-line decision-making. In this paper, we consider scenarios in which we design a series of experiments $\xi \in \Xi$ and observe corresponding outcomes $y$ to inform a final decision-making step. We assume we have a fixed experimental budget with $T$ query steps. For decision-making, we consider a set of possible decisions, denoted as $a \in \mathcal{A}$, with the objective of identifying an optimal decision $a^{*}$ that maximizes a predefined prediction-based utility function $u\left(y_{\Xi}, a\right)$.

### 3.2 Decision Utility Gain

Our method focuses on designing the experiments to directly improve the quality of the final decisionmaking. To quantify the effectiveness of each experimental design in terms of decision-making, we introduce Decision Utility Gain (DUG), which is defined as the difference in the expected utility of the best decision, with the new information obtained from the current experimental design, versus the best decision with the information obtained from previous experiments.

[^0]
[^0]: ${ }^{1}$ This definition assumes conditional independence of the outcomes given the design. More generally, $p\left(y_{\Xi} \mid h_{1: t}\right)$ defines a joint distribution or a stochastic process indexed by the set $\Xi$ (Parzen, 1999), where a familiar example could be a Gaussian process posterior defined on $\Xi \subseteq \mathbb{R}^{d}$ (Rasmussen and Williams, 2006).

---

#### Page 5

Definition 3.1. Given a historical experimental trajectory $h_{1: t-1}$, the Decision Utility Gain (DUG) for a given design $\xi_{t}$ and its corresponding outcome $y_{t}$ at step $t$ is defined as follows:

$$
\operatorname{DUG}\left(\xi_{t}, y_{t}\right)=\max _{a \in A} \mathbb{E}_{p\left(y_{\Xi} \mid h_{1: t-1} \cup\left\{\left(\xi_{t}, y_{t}\right)\right\}\right)}[u\left(y_{\Xi}, a\right)]-\max _{a \in A} \mathbb{E}_{p\left(y_{\Xi} \mid h_{1: t-1}\right)}[u\left(y_{\Xi}, a\right)]
$$

DUG measures the improvement in the maximum expected utility from observing a new experimental design, differing in this from standard marginal utility gain (see e.g., Garnett, 2023). The optimal design is the one that provides the largest increase in maximal expected utility. Shifting from parameter-centric to utility-centric evaluation, we directly evaluate the design's influence on the decision utility, bypassing the need to reduce the uncertainty of unknown latent parameters.
At the time we choose the design $\xi_{t}$, the outcome remains uncertain. Therefore, we should consider the Expected Decision Utility Gain (EDUG) with respect to the marginal distribution of the outcomes to select the next design.
Definition 3.2. The Expected Decision Utility Gain (EDUG) for a design $\xi_{t}$, given the historical experimental trajectory $h_{1: t-1}$, is defined as:

$$
\operatorname{EDUG}\left(\xi_{t}\right)=\mathbb{E}_{p\left(y_{t} \mid \xi_{t}, h_{1: t-1}\right)}\left[\operatorname{DUG}\left(\xi_{t}, y_{t}\right)\right]
$$

With EDUG, we can guide the experimental design without calculating the posterior distribution. If we knew the true predictive distribution, we could always determine the one-step lookahead optimal design by maximizing EDUG across the design space with $\xi^{*}=\arg \max _{\xi \in \mathbb{R}} \operatorname{EDUG}(\xi)$. However, in practice, the true predictive distributions are often unknown, making the optimization of EDUG exceptionally challenging. This difficulty arises due to the inherent bi-level optimization problem and the need to evaluate two layers of expectations, both over the unknown predictive distribution.
To avoid the expensive computational cost of optimizing EDUG, we propose leveraging a policy network, inspired by Foster et al. (2021), that directly maps historical data to the next design. This approach sidesteps the continuous need to optimize EDUG by learning a design strategy over many simulated experiment trajectories during the training phase. It can dramatically reduce computational demands at deployment, allowing for more efficient real-time decisions.

# 4 Amortizing decision-aware BED

A fully amortized BED framework for decision-making requires not only amortizing the experimental design but also the predictive distribution to approximate the expected utility. Moreover, permutation invariance is often assumed in sequential BED (Foster et al., 2021) ${ }^{2}$, meaning that the sequence of experiments does not influence the cumulative information gained. Conditional neural processes (CNPs) (Garnelo et al., 2018; Nguyen and Grover, 2022; Huang et al., 2023b) provide a suitable basis for developing our framework due to their design, which not only respects the permutation invariance of the inputs by treating them as an unordered set but also amortizes modeling of the predictive distributions. See Appendix A for a brief introduction to CNPs and TNPs.

### 4.1 Transformer Neural Decision Processes

The architecture of our model, termed Transformer Neural Decision Process (TNDP), is a novel architecture building upon the Transformer neural process (TNP) (Nguyen and Grover, 2022). It aims to amortize both the experimental design and the predictive distributions for the subsequent decisionmaking process. The data architecture of our system comprises four parts $D=\left\{D^{(c)}, D^{(p)}, D^{(q)}, \mathrm{GI}\right\}$ :

- A context set $D^{(c)}=h_{1: t}=\left\{\left(\xi_{t}^{(c)}, y_{t}^{(c)}\right)\right\}_{i=1}^{t}$ contains all past $t$-step designs and outcomes.
- A prediction set $D^{(p)}=\left\{\left(\xi_{t}^{(p)}, y_{t}^{(p)}\right)\right\}_{i=1}^{n_{p}}$ consists of $n_{p}$ design-outcome pairs used for approximating $p\left(y_{\Xi} \mid h_{1: t}\right)$, which is closely related to the training objective of the CNPs. The output from this head can then be used to estimate the expected utility.
- A query set $D^{(q)}=\left\{\xi_{t}^{(q)}\right\}_{i=1}^{n_{q}}$ consists of $n_{q}$ candidate experimental designs being considered for the next step. In scenarios where the design space $\Xi$ is continuous, we randomly sample a set of query points for each iteration during training. In the deployment phase, optimal experimental designs can be obtained by optimizing the model's output.

[^0]
[^0]: ${ }^{2}$ When permutation invariance does not hold in some cases, our model can be easily adapted by adding positional encoding to the input.

---

#### Page 6

> **Image description.** The image contains two diagrams, labeled (a) and (b), illustrating the architecture of a system called TNDP.
>
> **(a) Architecture Diagram:**
>
> - At the bottom, a horizontal orange rectangle represents the "Data Embedding Block f_emb". Below this block are labels indicating inputs: "t" and "γ" for "Global Info GI", "ξ₁⁽ᶜ⁾, y₁⁽ᶜ⁾" and "ξ₂⁽ᶜ⁾, y₂⁽ᶜ⁾" for "Context Set D⁽ᶜ⁾", "ξ₁⁽ᵖ⁾" and "ξ₂⁽ᵖ⁾" for "Prediction Set D⁽ᵖ⁾", and "ξ₁⁽q⁾" and "ξ₂⁽q⁾" for "Query Set D⁽q⁾".
> - Vertical arrows point upwards from these inputs to the Data Embedding Block. Above the Data Embedding Block, there are corresponding outputs labeled "Eᵗ", "Eᵞ", "E₁⁽ᶜ⁾", "E₂⁽ᶜ⁾", "E₁⁽ᵖ⁾", "E₂⁽ᵖ⁾", "E₁⁽q⁾", and "E₂⁽q⁾". The arrows from "E₁⁽ᶜ⁾" and "E₂⁽ᶜ⁾" are combined with a plus sign inside a circle before reaching the next layer.
> - These outputs feed into a larger, horizontal blue rectangle labeled "Transformer Block f_tfm".
> - Above the Transformer Block, vertical arrows lead to two separate blocks: a peach-colored rectangle labeled "Prediction Head fₚ" with output "q(y⁽ᵖ⁾|p)" and a green rectangle labeled "Query Head f_q" with output "πₜ(.|h₁:t-1)". The arrows leading to these blocks are labeled "λ⁽ᵖ⁾" and "λ⁽q⁾" respectively.
>
> **(b) Attention Mask Diagram:**
>
> - This diagram is a grid of squares, with rows and columns labeled to represent different data components.
> - The rows are labeled "GI", "D⁽ᶜ⁾" (split into "ξ₁⁽ᶜ⁾, y₁⁽ᶜ⁾" and "ξ₂⁽ᶜ⁾, y₂⁽ᶜ⁾"), "D⁽ᵖ⁾" (split into "ξ₁⁽ᵖ⁾" and "ξ₂⁽ᵖ⁾"), and "D⁽q⁾" (split into "ξ₁⁽q⁾" and "ξ₂⁽q⁾").
> - The columns are similarly labeled "GI", "D⁽ᶜ⁾" (split into "ξ₁⁽ᶜ⁾, y₁⁽ᶜ⁾" and "ξ₂⁽ᶜ⁾, y₂⁽ᶜ⁾"), "D⁽ᵖ⁾" (split into "ξ₁⁽ᵖ⁾" and "ξ₂⁽ᵖ⁾"), and "D⁽q⁾" (split into "ξ₁⁽q⁾" and "ξ₂⁽q⁾").
> - Some of the squares in the grid are filled with a blue color, while others are white. The blue squares indicate that the element on the left can attend to the element on the top in the self-attention layer of f_ffm. Specifically, the first column is filled with blue squares. The second and third columns have blue squares only in the rows corresponding to D⁽ᶜ⁾. The fourth and fifth columns have blue squares only in the rows corresponding to D⁽ᵖ⁾. The sixth and seventh columns have blue squares only in the rows corresponding to D⁽q⁾.

Figure 2: Illustration of TNDP. (a) An overview of TNDP architecture with input consisting of 2 observed design-outcome pairs from $D^{(\mathrm{c})}, 2$ designs from $D^{(\mathrm{p})}$ for prediction, and 2 candidate designs from $D^{(\mathrm{q})}$ for query. (b) The corresponding attention mask. The colored squares indicate that the elements on the left can attend to the elements on the top in the self-attention layer of $f_{\mathrm{ffm}}$.

- Global information $\mathrm{GI}=[t, \gamma]$ where $t$ represents the current step in the experimental sequence, and $\gamma$ encapsulates task-related information, which could include contextual data relevant to the decision-making process. We will further explain the choice of $\gamma$ in Section 6.

TNDP comprises four main components: the data embedder block $f_{\text {emb }}$, the Transformer block $f_{\text {ffm }}$, the prediction head $f_{\mathrm{p}}$, and the query head $f_{\mathrm{q}}$. Each component plays a distinct role in the overall decision-aware BED process. The full architecture is shown in Fig. 2(a).
At first, each set of $D$ is processed by the data embedder block $f_{\text {emb }}$ to map to an aligned embedding space. These embeddings are then concatenated to form a unified representation $\boldsymbol{E}=\operatorname{concat}\left(\boldsymbol{E}^{(\mathrm{c})}, \boldsymbol{E}^{(\mathrm{p})}, \boldsymbol{E}^{(\mathrm{q})}, \boldsymbol{E}^{\mathrm{GI}}\right)$. Please refer to Appendix B for a detailed explanation of how we embed the data. After the initial embedding, the Transformer block $f_{\text {ffm }}$ processes $\boldsymbol{E}$ using self-attention mechanisms to produce a single attention matrix, which is subsequently processed by an attention mask (see Fig. 2(b)) that allows for selective interactions between different data components, ensuring that each part contributes appropriately to the final output. To explain, each design from the prediction set $D^{(\mathrm{p})}$ is configured to attend to itself, the global information, and the historical data, reflecting the dependence of the predictions on the historical data and the independence from other designs. Similarly, each $\xi^{(\mathrm{q})}$ in the query set $D^{(\mathrm{q})}$ is also restricted to attend only to itself, the global information, and the historical data. This setup preserves the independence of each candidate design, ensuring that the evaluation of one design neither influences nor is influenced by others. The output of $f_{\text {ffm }}$ is then split according to the specific needs of the query and prediction head $\boldsymbol{\lambda}=\left[\boldsymbol{\lambda}^{(\mathrm{p})}, \boldsymbol{\lambda}^{(\mathrm{q})}\right]=f_{\text {ffm }}(\boldsymbol{E})$.
The primary role of the prediction head $f_{\mathrm{p}}$ is to approximate $p\left(y_{\mathbb{E}} \mid h_{1: t}\right)$ with a family of parameterized distributions $q\left(y_{\mathbb{E}} \mid \boldsymbol{p}_{t}\right)$, where $\boldsymbol{p}_{t}=f_{\mathrm{p}}\left(\boldsymbol{\lambda}_{t}^{(\mathrm{p})}\right)$ is the output of $f_{\mathrm{p}}$ at the step $t$. The training of $f_{\mathrm{p}}$ is by minimizing the negative log-likelihood of the predicted probabilities:

$$
\mathcal{L}^{(\mathrm{p})}=-\sum_{t=1}^{T} \sum_{i=1}^{n_{\mathrm{p}}} \log q\left(y_{t}^{(\mathrm{p})} \mid \boldsymbol{p}_{i, t}\right)=-\sum_{t=1}^{T} \sum_{i=1}^{n_{\mathrm{p}}} \log \mathcal{N}\left(y_{t}^{(\mathrm{p})} \mid \boldsymbol{\mu}_{i, t}, \boldsymbol{\sigma}_{i, t}^{2}\right)
$$

where $\boldsymbol{p}_{i, t}$ represents the output of design $\xi_{i}^{(\mathrm{p})}$ at step $t$. Here we choose a Gaussian likelihood with $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ representing the predicted mean and standard deviation split from $\boldsymbol{p}$.
The query head $f_{\mathrm{q}}$ processes the transformed embeddings $\boldsymbol{\lambda}^{(\mathrm{q})}$ from the Transformer block to generate a policy distribution over possible experimental designs. Specifically, it converts the embeddings into a probability distribution used to select the next experimental design. The outputs of the query head, $\boldsymbol{q}=f_{\mathrm{q}}\left(\boldsymbol{\lambda}^{(\mathrm{q})}\right)$, are mapped to a probability distribution via a Softmax function:

$$
\pi\left(\xi_{j, t}^{(\mathrm{q})} h_{1: t-1}\right)=\frac{\exp \left(\boldsymbol{q}_{j, t}\right)}{\sum_{i=0}^{n_{\mathrm{q}}} \exp \left(\boldsymbol{q}_{i, t}\right)}
$$

---

#### Page 7

where $\boldsymbol{q}_{j, t}$ represents the $t$-step's output for the candidate design $\xi_{t}^{(\mathrm{q})}$.
To design a reward signal that guides $f_{\mathrm{q}}$ in proposing informative designs, we first define a singlestep immediate reward based on DUG (Eq. (5)), replacing the true predictive distribution with our approximated distribution:

$$
r_{t}\left(\xi_{t}^{(\mathrm{q})}\right)=\max _{a \in A} \mathbb{E}_{q\left(y_{\Xi} \mid \boldsymbol{p}_{t}\right)}[u\left(y_{\Xi}, a\right)]-\max _{a \in A} \mathbb{E}_{q\left(y_{\Xi} \mid \boldsymbol{p}_{t-1}\right)}[u\left(y_{\Xi}, a\right)]
$$

This reward quantifies how the experimental design influences our decision-making by estimating the improvement in expected utility that results from incorporating new experimental outcomes. However, this objective remains myopic, as it does not account for the future or the final decision-making. To address this, we employ the REINFORCE algorithm (Williams, 1992), which allows us to consider the impact of the current design on future rewards. The final loss of $f_{\mathrm{q}}$ can be written as the negative expected reward for the complete experimental trajectory:

$$
\mathcal{L}^{(\mathrm{q})}=-\sum_{t=1}^{T} R_{t} \log \pi\left(\xi_{t}^{(\mathrm{q})} \mid h_{1: t-1}\right)
$$

where $R_{t}=\sum_{k=t}^{T} \alpha^{k-t} r_{k}\left(\xi_{k}^{(\mathrm{q})}\right)$ represents the non-myopic discounted reward. The discount factor $\alpha$ is used to decrease the importance of rewards received at later time step. $\xi_{t}^{(\mathrm{q})}$ is obtained through sampling from the policy distribution $\xi_{t}^{(\mathrm{q})} \sim \pi\left(\cdot \mid h_{1: t-1}\right)$.
The update of $f_{\mathrm{q}}$ depends critically on the accuracy with which $f_{\mathrm{p}}$ approximates the predictive distribution. Ultimately, the effectiveness of decision-making relies on the informativeness of the designs proposed by $f_{\mathrm{q}}$, ensuring that every step in the experimental trajectory is optimally aligned with the overarching goal of maximizing the decision utility. The full algorithm of our method is shown in Appendix C.

# 5 Related work

Lindley (1972) proposes the first decision-theoretic BED framework, later reiterated by Chaloner and Verdinelli (1995). However, their utility is defined based on individual designs, while our utility is formulated in terms of a stochastic process and is designed for the final decision-making task after multiple rounds of experimental design. Recently, several other BED frameworks that focus on different downstream properties have been proposed. Bayesian Algorithm Execution (BAX) (Neiswanger et al., 2021) develops a BED framework that optimizes experiments based on downstream properties of interest. BAX introduces a new metric that queries the next experiment by maximizing the mutual information between the property and the outcome. CO-BED (Ivanova et al., 2023) introduces a contextual optimization method within the BED framework, where the design phase incorporates information-theoretic objectives specifically targeted at optimizing contextual rewards. Neiswanger et al. (2022) presents an information-based acquisition function for Bayesian optimization which explicitly considers the downstream task. Zhong et al. (2024) proposes a goaloriented BED framework for nonlinear models using MCMC to optimize the EIG on predictive quantities of interest. Filstroff et al. (2024) presents a framework for active learning that queries data to reduce the uncertainty on the posterior distribution of the optimal downstream decision.
In recent years, various amortized BED methods have emerged. Foster et al. (2021) is the first to introduce this framework; subsequent work extends it to scenarios with unknown likelihood (Ivanova et al., 2021) and improved performance using reinforcement learning (Blau et al., 2022; Lim et al., 2022). The latest research proposes a semi-amortized framework that periodically updates the policy during the experiment to improve adaptability (Ivanova et al., 2024). Maraval et al. (2024) proposes a fully amortized Bayesian optimization (BO) framework that employs a similar TNP architecture, while their work focuses specifically on BO objectives, our approach addresses general downstream decision-making tasks. Additionally, our framework introduces a novel coupled training objective between query and prediction heads, providing a more integrated architecture for decision-making.
Our proposed architecture is based on pre-trained Transformer models. Transformer-based neural processes (Müller et al., 2021; Nguyen and Grover, 2022; Chang et al., 2024) serve as the foundational structure for our approach, but they have not considered experimental design. Decision Transformers (Chen et al., 2021; Zheng et al., 2022) can be used for sequentially designing experiments. However, we additionally amortize the predictive distribution, making the learning process more challenging.

---

#### Page 8

> **Image description.** The image consists of two plots stacked vertically, related to a 1D synthetic regression task. The plots share a common horizontal axis labeled "x". The image is labeled "(a) 1D synthetic regression" at the bottom.
>
> The top plot shows a curve representing the "target function" plotted as a solid black line. The y-axis is labeled "y" and ranges from 0.0 to 2.0. Several data points are marked with pink "x" symbols, labeled as "queried data". A vertical dashed red line, labeled "x\*", intersects the target function. A cyan star symbol marks the "next query" point on the target function, located at the intersection with the red dashed line. A legend in the upper right corner identifies the symbols and line styles.
>
> The bottom plot displays a green curve resembling a Gaussian distribution. The y-axis is labeled "π" and ranges from 0.000 to 0.005. The area under the green curve is shaded in a lighter green. The peak of the curve aligns with the vertical red dashed line from the top plot.

(a) 1D synthetic regression

> **Image description.** This is a line graph comparing the performance of different methods in a decision-aware active learning task.
>
> The graph has the following features:
>
> - **Axes:** The x-axis is labeled "Design steps _t_" and ranges from 0 to 10. The y-axis is labeled "Proportion of correct decisions (%)" and ranges from 0.4 to 0.9.
> - **Lines:** There are six lines plotted on the graph, each representing a different method. The methods are:
>   - GP-RS (red line with square markers)
>   - GP-TEIG (brown line with diamond markers)
>   - GP-US (purple line with triangle markers)
>   - GP-DEIG (pink line with cross markers)
>   - GP-DUS (green line with triangle markers)
>   - TNDP (ours) (orange line with star markers)
> - **Shaded Regions:** Each line has a shaded region around it, representing the standard error. The colors of the shaded regions correspond to the colors of the lines.
> - **Legend:** A legend is provided in the lower right corner of the graph, identifying each line with its corresponding method.
> - **Title:** The graph is labeled "(b) Decision-aware active learning" at the bottom.
>
> The TNDP (ours) method (orange line with star markers) appears to outperform the other methods, achieving a higher proportion of correct decisions as the number of design steps increases.

(b) Decision-aware active learning

Figure 3: Results of synthetic regression and decision-aware active learning. (a) The top figure represents the true function and the initial known points. The red line indicates the location of $x^{*}$. The blue star marks the next query point, sampled from the policy's predicted distribution shown in the bottom figure. (b) Mean and standard error of the proportion of correct decisions on 100 test points w.r.t. the acquisition steps. Our TNDP significantly outperforms other methods.

## 6 Experiments

In this section, we evaluate our proposed framework on several tasks. Our experimental approach is detailed in Appendix B. In Appendix F.3, we provide additional ablation studies of TNDP to show the effectiveness of our query head and the non-myopic objective function. The code to reproduce our experiments is available at https://github.com/huangdaolang/ amortized-decision-aware-bed.

### 6.1 Toy example: synthetic regression

We begin with an illustrative example to show how our TNDP works. We consider a 1D synthetic regression task where the goal is to perform regression at a specific test point $x^{*}$ on an unknown function. To accurately predict this point, we need to sequentially design some new points to query. This example can be viewed as a prediction-oriented active learning (AL) task (Smith et al., 2023).
The design space $\Xi=\mathcal{X}$ is the domain of $x$, and $y$ is the corresponding noisy observations of the function. Let $\mathcal{Q}(\mathcal{X})$ denote the set of combinations of distributions that can be output by TNDP, we can then define decision space to be $\mathcal{A}=\mathcal{Q}(\mathcal{X})$. The downstream decision is to output a predictive distribution for $y^{*}$ given a test point $x^{*}$, and the utility function $u\left(y_{\Xi}, a\right)=\log q\left(y^{*} \mid x^{*}, h_{1: t}\right)$ is the log probability of $y$ under the predicted distribution, given the queried historical data $h_{t}$.
During training, we sample functions from Gaussian Processes (GPs) (Rasmussen and Williams, 2006) with squared exponential kernels of varying output variances and lengthscales and randomly sample a point as the test point $x^{*}$. We set the global contextual information $\lambda$ as the test point $x^{*}$. For illustration purposes, we consider only the case where $T=1$. Additional details for the data generation can be found in Appendix E.
Results. From Fig. 3(a), we can observe that the values of $\pi$ concentrate near $x^{*}$, meaning our query head $f_{\mathrm{q}}$ tends to query points close to $x^{*}$ to maximize the DUG. This is an intuitive example demonstrating that our TNDP can adjust its design strategy based on the downstream task.

### 6.2 Decision-aware active learning

We now show another application of our method in a case of decision-aware AL studied by Filstroff et al. (2024). In this experiment, the model will be used for downstream decision-making after performing AL, i.e., we will use the learned information to take an action towards a specific target. A practical application of this problem is personalized medical diagnosis introduced in Section 1, where a doctor needs to query historical patient data to decide on a treatment for a new patient.
We use the same problem setup as in Filstroff et al. (2024). The decision space consists of $N_{d}$ available decisions, $a \in \mathcal{A} \in\left\{1, \ldots, N_{d}\right\}$. The design space $\Xi=\mathcal{X} \times \mathcal{A}$ is composed of the patient's

---

#### Page 9

> **Image description.** The image consists of four line graphs arranged horizontally. Each graph displays the "Utility" on the y-axis versus "Step t" on the x-axis. The y-axis ranges vary slightly between the graphs, but generally span from around 2.7 to 2.8. The x-axis ranges from 0 to 50 in all graphs. Each graph contains multiple lines, each representing a different algorithm. The algorithms are labeled as "RS", "UCB", "EI", "PI", "PFNs4BO", and "TNDP". Each line has a different color and marker style (e.g., dotted, dashed, solid). Shaded regions around each line indicate the standard deviation. The titles of the graphs, from left to right, are "ranger", "rpart", "svm", and "xgboost". A legend is placed below the graphs, indicating the color and marker style associated with each algorithm.

Figure 4: Results on Top- $k$ HPO task. For each meta-dataset, we calculated the average utility across all available test sets. The error bars represent the standard deviation over five runs. TNDP consistently achieved the best performance in terms of utility.
covariates $x$ and the decisions $a$ they receive. The outcome $y$ is the treatment effect after the patient receives the decision, which is influenced by real-world unknown parameters $\theta$ such as medical conditions. For historical data, each patient is associated with only one decision. The utility function $u\left(y_{\Xi}, a\right)=\mathbb{I}\left(\hat{a}^{*}, a^{*}\right)$ is a binary accuracy score that measures whether we can make the correct decision for a new patient $x^{*}$ based on the queried history, where $\hat{a}^{*}$ is the predicted Bayesian optimal decision and $a^{*}$ the true optimal decision. Here, $u\left(y_{\Xi}, a\right)=1$ if and only if $\hat{a}^{*}=a^{*}$.
In our experiment, we use the synthetic dataset from Filstroff et al. (2024), the details of the data generating process can be found in Appendix F. We set $N_{d}=4$ and use independent GPs to generate different outcomes. Each data point is randomly assigned a decision, and the outcome is the corresponding $y$ value from the associated GP. We randomly select a test point $x^{*}$ and determine the optimal decision $a^{*}$ based on the GP that provides the maximum $y$ value at $x^{*}$. We set global contextual information $\lambda$ as the covariates of the test point $x^{*}$.
We compare TNDP with other non-amortized AL methods: random sampling (GP-RS), uncertainty sampling (GP-US), decision uncertainty sampling (GP-DUS), targeted information (GP-TEIG) introduced by Sundin et al. (2018), and decision EIG (GP-DEIG) proposed by Filstroff et al. (2024). A detailed description of each method can be found in Appendix F. Each method is tested on 100 different $x^{*}$ points, and the average utility, i.e., the proportion of correct decisions, is calculated.
Results. The results are shown in Fig. 3(b), where we can see that TNDP achieves significantly better average accuracy than other methods. Additionally, we conduct an ablation study of TNDP in Appendix F. 3 to verify the effectiveness of $f_{q}$. We further analyze the deployment running time to show the advantage of amortization, see Appendix D.1.

# 6.3 Top- $k$ hyperparameter optimization

In traditional optimization tasks, we typically only aim to find a single point that maximizes the underlying function $f$. However, instead of identifying a single optimal point, there are scenarios where we wish to estimate a set of top- $k$ distinct optima. For example, this is the case in robust optimization, where selecting multiple points can safeguard against variations in data or model performance.
In this experiment, we choose hyperparameter optimization (HPO) as our task and conduct experiments on the HPO-B datasets (Arango et al., 2021). The design space $\Xi \subseteq \mathcal{X}$ is a finite set defined over the hyperparameter space and the outcome $y$ is the accuracy of a given configuration on a specific dataset. Our decision is to find $k$ hyperparameter sets, denoted as $a=\left(a_{1}, \ldots, a_{k}\right) \in A \subseteq \mathcal{X}^{k}$, with $a_{i} \neq a_{j}$. The utility function is then defined as $u\left(y_{\Xi}, a\right)=\sum_{i=1}^{k} y_{a_{i}}$, where $y_{a_{i}}$ is the accuracy corresponding to the hyperparameter configuration $a_{i}$. In this experiment, the global contextual information $\lambda=\emptyset$.
We compare our methods with five different BO methods: random sampling (RS), Upper Confidence Bound (UCB), Expected Improvement (EI), Probability of Improvement (PI), and an amortized method PFNs4BO (Müller et al., 2023), which is a transformer-based model designed for hyperparameter optimization. We set $k=3$ and $T=50$, starting with an initial dataset of 5 points. Our experiments are conducted on four search spaces selected from the HPO-B benchmark. All results

---

#### Page 10

are evaluated on a predefined test set, ensuring that TNDP does not encounter these test sets during training. For more details, see Appendix G.
Results. From the experimental results (Fig. 4), our method demonstrates superior performance across all four meta-datasets, particularly during the first 10 queries, achieving clearly better utility gains. This indicates that our TNDP can effectively identify high-performing hyperparameter configurations early in the optimization process.
Finally, we included a real-world experiment on retrosynthesis planning. Specifically, our task is to assist chemists in identifying the top- $k$ synthetic routes for a novel molecule, as selecting the most practical routes from many random routes generated by the retrosynthesis software can be troublesome. The detailed description of the task and the results are shown in Appendix G.3.

# 7 Discussion

### 7.1 Limitations \& future work

We recognize that the training of the query head inherently poses a reinforcement learning (RL) (Li, 2017) problem. Currently, we employ a basic REINFORCE algorithm, which can result in unstable training, particularly for tasks with sparse reward signals. For more complex problems in the future, we could deploy advanced RL methods, such as Proximal Policy Optimization (PPO) (Schulman et al., 2017); the trade-offs include the introduction of additional hyperparameters and increased computational cost. Like all amortized approaches, our method requires a large amount of data and upfront training time to develop a reliable model. Besides, our architecture is based on the Transformer, which suffers from quadratic complexity with respect to the input sequence length. This can become a bottleneck when the query set is very large. Future work could focus on designing more sample-efficient methods to reduce the data and training requirements. Our TNDP follows the common practice in the neural processes literature (Garnelo et al., 2018) of using independent Gaussian likelihoods. If modeling correlations between points is crucial for the downstream task, we can replace the output with a joint multivariate normal distribution (Markou et al., 2022) or predict the output autoregressively (Bruinsma et al., 2023). Following most BED approaches, our work assumes that the model is well-specified. However, model misspecification or shifts in the utility function during deployment could impact the performance of the amortized model (Rainforth et al., 2024). Future work could address the challenge of robust experimental design under model misspecification (Huang et al., 2023a). Another limitation is that our system is currently constrained to accepting designs of the same dimensionality. Future work could focus on developing dimension-agnostic methods to expand the scope of amortization. Lastly, our model is trained on a fixed-step length, assuming a finite horizon for the experimental design process. Future research could explore the design of systems that can handle infinite horizon cases, potentially improving the applicability of TNDP to a broader range of real-world problems.

### 7.2 Conclusions

In this paper, we proposed an amortized framework for decision-aware Bayesian experimental design (BED). We introduced the concept of Decision Utility Gain (DUG) to guide experimental design more effectively toward optimizing decision outcomes. Towards amortization, we developed a novel Transformer Neural Decision Process (TNDP) architecture with dual output heads: one for proposing new experimental designs and another for approximating the predictive distribution to facilitate optimal decision-making. Our experimental results demonstrated that TNDP significantly outperforms traditional BED methods across a variety of tasks. By integrating decision-making considerations directly into the experimental design process, TNDP not only accelerates the design of experiments but also improves the quality of the decisions derived from these experiments.

---

# Amortized Bayesian Experimental Design for Decision-Making - Backmatter

---

## Acknowledgements

DH, LA and SK were supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI). YG was supported by Academy of Finland grant 345604. LA was also supported by Research Council of Finland grants 358980 and 356498. SK was also supported by the UKRI Turing AI World-Leading Researcher Fellowship, [EP/W002973/1]. The authors wish to thank Aalto Science-IT project, and CSC-IT Center for Science, Finland, for the computational and data storage resources provided.

---

#### Page 11

# References

Abbasnejad, E., Domke, J., and Sanner, S. (2015). Loss-calibrated monte carlo action selection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29.

Arango, S. P., Jomaa, H. S., Wistuba, M., and Grabocka, J. (2021). Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Balandat, M., Karrer, B., Jiang, D., Daulton, S., Letham, B., Wilson, A. G., and Bakshy, E. (2020). Botorch: A framework for efficient monte-carlo bayesian optimization. Advances in neural information processing systems, 33.

Berger, J. O. (2013). Statistical decision theory and Bayesian analysis. Springer Science \& Business Media.

Bica, I., Alaa, A. M., Lambert, C., and Van Der Schaar, M. (2021). From real-world patient data to individualized treatment effects using machine learning: current and future methods to address underlying challenges. Clinical Pharmacology \& Therapeutics, 109(1):87-100.

Bille, P. (2005). A survey on tree edit distance and related problems. Theoretical computer science, $337(1-3): 217-239$.

Blacker, A. J., Williams, M. T., and Williams, M. T. (2011). Pharmaceutical process development: current chemical and engineering challenges, volume 9. Royal Society of Chemistry.

Blau, T., Bonilla, E., Chades, I., and Dezfouli, A. (2023). Cross-entropy estimators for sequential experiment design with reinforcement learning. arXiv preprint arXiv:2305.18435.

Blau, T., Bonilla, E. V., Chades, I., and Dezfouli, A. (2022). Optimizing sequential experimental design with deep reinforcement learning. In International conference on machine learning, pages 2107-2128. PMLR.

Bruinsma, W., Markou, S., Requeima, J., Foong, A. Y., Andersson, T., Vaughan, A., Buonomo, A., Hosking, S., and Turner, R. E. (2023). Autoregressive conditional neural processes. In The Eleventh International Conference on Learning Representations.

Burger, M., Hauptmann, A., Helin, T., Hyvönen, N., and Puska, J.-P. (2021). Sequentially optimized projections in x-ray imaging. Inverse Problems, 37(7):075006.

Chaloner, K. and Verdinelli, I. (1995). Bayesian experimental design: A review. Statistical science, pages 273-304.

Chang, P. E., Loka, N., Huang, D., Remes, U., Kaski, S., and Acerbi, L. (2024). Amortized probabilistic conditioning for optimization, simulation and inference. arXiv preprint arXiv:2410.15320.

Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., and Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34.

Cheng, Y. and Shen, Y. (2005). Bayesian adaptive designs for clinical trials. Biometrika, 92(3):633646 .

Cobb, A. D., Roberts, S. J., and Gal, Y. (2018). Loss-calibrated approximate inference in bayesian neural networks. arXiv preprint arXiv:1805.03901.

Filstroff, L., Sundin, I., Mikkola, P., Tiulpin, A., Kylmäoja, J., and Kaski, S. (2024). Targeted active learning for bayesian decision-making. Transactions on Machine Learning Research.

Foster, A., Ivanova, D. R., Malik, I., and Rainforth, T. (2021). Deep adaptive design: Amortizing sequential bayesian experimental design. In International Conference on Machine Learning, pages 3384-3395. PMLR.

Foster, A., Jankowiak, M., Bingham, E., Horsfall, P., Teh, Y. W., Rainforth, T., and Goodman, N. (2019). Variational bayesian optimal experimental design. Advances in Neural Information Processing Systems, 32.

---

#### Page 12

Foster, A., Jankowiak, M., O’Meara, M., Teh, Y. W., and Rainforth, T. (2020). A unified stochastic gradient approach to designing bayesian-optimal experiments. In International Conference on Artificial Intelligence and Statistics, pages 2959-2969. PMLR.

Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., and Eslami, S. A. (2018). Conditional neural processes. In International conference on machine learning, pages 1704-1713. PMLR.

Garnett, R. (2023). Bayesian optimization. Cambridge University Press.
Huang, D., Bharti, A., Souza, A., Acerbi, L., and Kaski, S. (2023a). Learning robust statistics for simulation-based inference under model misspecification. Advances in Neural Information Processing Systems, 36.

Huang, D., Haussmann, M., Remes, U., John, S., Clarté, G., Luck, K., Kaski, S., and Acerbi, L. (2023b). Practical equivariances via relational conditional neural processes. Advances in Neural Information Processing Systems, 36.

Ivanova, D. R., Foster, A., Kleinegesse, S., Gutmann, M. U., and Rainforth, T. (2021). Implicit deep adaptive design: Policy-based experimental design without likelihoods. Advances in Neural Information Processing Systems, 34.

Ivanova, D. R., Hedman, M., Guan, C., and Rainforth, T. (2024). Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design. ICLR 2024 Workshop on Data-centric Machine Learning Research (DMLR).

Ivanova, D. R., Jennings, J., Rainforth, T., Zhang, C., and Foster, A. (2023). Co-bed: informationtheoretic contextual optimization via bayesian experimental design. In International Conference on Machine Learning. PMLR.

Kleinegesse, S. and Gutmann, M. U. (2020). Bayesian experimental design for implicit models by mutual information neural estimation. In International conference on machine learning, pages 5316-5326. PMLR.

Kuśmierczyk, T., Sakaya, J., and Klami, A. (2019). Variational bayesian decision-making for continuous utilities. Advances in Neural Information Processing Systems, 32.

Lacoste-Julien, S., Huszár, F., and Ghahramani, Z. (2011). Approximate inference for the losscalibrated bayesian. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 416-424. JMLR Workshop and Conference Proceedings.

Li, Y. (2017). Deep reinforcement learning: An overview. arXiv preprint arXiv:1701.07274.
Lim, V., Novoseller, E., Ichnowski, J., Huang, H., and Goldberg, K. (2022). Policy-based bayesian experimental design for non-differentiable implicit models. arXiv preprint arXiv:2203.04272.

Lindley, D. V. (1956). On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, 27(4):986-1005.

Lindley, D. V. (1972). Bayesian statistics: A review. SIAM.
Maraval, A., Zimmer, M., Grosnit, A., and Bou Ammar, H. (2024). End-to-end meta-bayesian optimisation with transformer neural processes. Advances in Neural Information Processing Systems, 36 .

Markou, S., Requeima, J., Bruinsma, W., Vaughan, A., and Turner, R. E. (2022). Practical conditional neural process via tractable dependent predictions. In International Conference on Learning Representations.

Mo, Y., Guan, Y., Verma, P., Guo, J., Fortunato, M. E., Lu, Z., Coley, C. W., and Jensen, K. F. (2021). Evaluating and clustering retrosynthesis pathways with learned strategy. Chemical science, 12(4):1469-1478.

Morais, M. J. and Pillow, J. W. (2022). Loss-calibrated expectation propagation for approximate bayesian decision-making. arXiv preprint arXiv:2201.03128.

---

#### Page 13

Müller, S., Feurer, M., Hollmann, N., and Hutter, F. (2023). Pfns4bo: In-context learning for bayesian optimization. In International Conference on Machine Learning, pages 25444-25470. PMLR.

Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., and Hutter, F. (2021). Transformers can do bayesian inference. In International Conference on Learning Representations.

Neiswanger, W., Wang, K. A., and Ermon, S. (2021). Bayesian algorithm execution: Estimating computable properties of black-box functions using mutual information. In International Conference on Machine Learning, pages 8005-8015. PMLR.

Neiswanger, W., Yu, L., Zhao, S., Meng, C., and Ermon, S. (2022). Generalizing bayesian optimization with decision-theoretic entropies. Advances in Neural Information Processing Systems, 35 .

Nguyen, T. and Grover, A. (2022). Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In International Conference on Machine Learning, pages 16569-16594. PMLR.

Parzen, E. (1999). Stochastic processes. SIAM.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Rainforth, T., Cornish, R., Yang, H., Warrington, A., and Wood, F. (2018). On nesting monte carlo estimators. In International Conference on Machine Learning, pages 4267-4276. PMLR.

Rainforth, T., Foster, A., Ivanova, D. R., and Bickford Smith, F. (2024). Modern bayesian experimental design. Statistical Science, 39(1):100-114.

Rasmussen, C. E. and Williams, C. K. (2006). Gaussian Processes for Machine Learning. MIT Press.
Ryan, E. G., Drovandi, C. C., McGree, J. M., and Pettitt, A. N. (2016). A review of modern computational algorithms for bayesian optimal design. International Statistical Review, 84(1):128154 .

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Smith, F. B., Kirsch, A., Farquhar, S., Gal, Y., Foster, A., and Rainforth, T. (2023). Prediction-oriented bayesian active learning. In International Conference on Artificial Intelligence and Statistics, pages 7331-7348. PMLR.

Stevens, S. J. (2011). Progress toward the synthesis of providencin. PhD thesis, Colorado State University.

Sundin, I., Peltola, T., Micallef, L., Afrabandpey, H., Soare, M., Mamun Majumder, M., Daee, P., He, C., Serim, B., Havulinna, A., et al. (2018). Improving genomics-based predictions for precision medicine through active elicitation of expert knowledge. Bioinformatics, 34(13):i395-i403.

Szymkuć, S., Gajewska, E. P., Klucznik, T., Molga, K., Dittwald, P., Startek, M., Bajczyk, M., and Grzybowski, B. A. (2016). Computer-assisted synthetic planning: the end of the beginning. Angewandte Chemie International Edition, 55(20):5904-5937.

Vadera, M. P., Ghosh, S., Ng, K., and Marlin, B. M. (2021). Post-hoc loss-calibration for bayesian neural networks. In Uncertainty in Artificial Intelligence, pages 1403-1412. PMLR.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30 .

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229-256.

---

#### Page 14

Zheng, Q., Zhang, A., and Grover, A. (2022). Online decision transformer. In international conference on machine learning. PMLR.

Zhong, S., Shen, W., Catanach, T., and Huan, X. (2024). Goal-oriented bayesian optimal experimental design for nonlinear models using markov chain monte carlo. arXiv preprint arXiv:2403.18072.

---

# Amortized Bayesian Experimental Design for Decision-Making - Appendix

---

#### Page 15

# Appendix

The appendix is organized as follows:

- In Appendix A, we provide a brief introduction to conditional neural processes (CNPs) and Transformer neural processes (TNPs).
- In Appendix B, we describe the details of our model architecture and the training setups.
- In Appendix C, we present the full algorithm for training our TNDP architecture.
- In Appendix D, we compare the inference time with other methods and show the overall training time of TNDP.
- In Appendix E, we describe the details of our toy example.
- In Appendix F, we describe the details of the decision-aware active learning example.
- In Appendix G, we describe the details of the top- $k$ hyperparameter optimization task, along with additional results on the retrosynthesis planning task.

## A Conditional neural processes

CNPs (Garnelo et al., 2018) are designed to model complex stochastic processes through a flexible architecture that utilizes a context set and a target set. The context set consists of observed data points that the model uses to form its understanding, while the target set includes the points to be predicted. The traditional CNP architecture includes an encoder and a decoder. The encoder is a DeepSet architecture to ensure permutation invariance, it transforms each context point individually and then aggregates these transformations into a single representation that captures the overall context. The decoder then uses this representation to generate predictions for the target set, typically employing a Gaussian likelihood for approximation of the true predictive distributions. Due to the analytically tractable likelihood, CNPs can be efficiently trained through maximum likelihood estimation.

## A. 1 Transformer neural processes

Transformer Neural Processes (TNPs), introduced by Nguyen and Grover (2022), improve the flexibility and expressiveness of CNPs by incorporating the Transformer's attention mechanism (Vaswani et al., 2017). In TNPs, the transformer architecture uses self-attention to process the context set, dynamically weighting the importance of each point. This allows the model to create a rich representation of the context, which is then used by the decoder to generate predictions for the target set. The attention mechanism in TNPs facilitates the handling of large and variable-sized context sets, improving the model's performance on tasks with complex input-output relationships. The Transformer architecture is also useful in our setups where certain designs may have a more significant impact on the decision-making process than others. For more details about TNPs, please refer to Nguyen and Grover (2022).

## B Implementation details

## B. 1 Embedders

The embedder $f_{\text {emb }}$ is responsible for mapping the raw data to a space of the same dimension. For the toy example and the top- $k$ hyperparameter task, we use three embedders: a design embedder $f_{\text {emb }}^{(k)}$, an outcome embedder $f_{\text {emb }}^{(u)}$, and a time step embedder $f_{\text {emb }}^{(t)}$. Both $f_{\text {emb }}^{(k)}$ and $f_{\text {emb }}^{(u)}$ are multi-layer perceptions (MLPs) with the following architecture:

- Hidden dimension: the dimension of the hidden layers, set to 32.
- Output dimension: the dimension of the output space, set to 32 .
- Depth: the number of layers in the neural network, set to 4.
- Activation function: ReLU is used as the activation function for the hidden layers.

The time step embedder $f_{\text {emb }}^{(t)}$ is a discrete embedding layer that maps time steps to a continuous embedding space of dimension 32 .

---

#### Page 16

For the decision-aware active learning task, since the design space contains both the covariates and the decision, we use four embedders: a covariate embedder $f_{\text {emb }}^{(x)}$, a decision embedder $f_{\text {emb }}^{(\text {d }}$, an outcome embedder $f_{\text {emb }}^{(\text {a })}$, and a time step embedder $f_{\text {emb }}^{(t)} \cdot f_{\text {emb }}^{(x)} \cdot f_{\text {emb }}^{(y)}$ and $f_{\text {emb }}^{(t)}$ are MLPs which use the same settings as described above. The decision embedder $f_{\text {emb }}^{(\text {d }}$ is another discrete embedding layer.
For context embedding $\boldsymbol{E}^{(\mathrm{c})}$, we first map each $\xi_{i}^{(\mathrm{c})}$ and $y_{i}^{(\mathrm{c})}$ to the same dimension using their respective embedders, and then sum them to obtain the final embedding. For prediction embedding $\boldsymbol{E}^{(\mathrm{p})}$ and query embedding $\boldsymbol{E}^{(\mathrm{q})}$, we only encode the designs. For $\boldsymbol{E}^{(\mathrm{d})}$, except the embeddings of the time step, we also encode the global contextual information $\lambda$ using $f_{\text {emb }}^{(\mathrm{c})}$ in the toy example and the decision-aware active learning task. All the embeddings are then concatenated together to form our final embedding $\boldsymbol{E}$.

# B. 2 Transformer blocks

We utilize the official TransformerEncoder layer of PyTorch (Paszke et al., 2019) (https:// pytorch.org) for our transformer architecture. For all experiments, we use the same configuration: the model has 6 Transformer layers, with 8 heads per layer, the MLP block has a hidden dimension of 128 , and the embedding dimension size is set to 32 .

## B. 3 Output heads

The prediction head, $f_{\mathrm{p}}$ is an MLP that maps the Transformer's output embeddings of the query set to the predicted outcomes. It consists of an input layer with 32 hidden units, a ReLU activation function, and an output layer. The output layer predicts the mean and variance of a Gaussian likelihood, similar to CNPs.

For the query head $f_{\mathrm{q}}$, all candidate experimental designs are first mapped to embeddings $\boldsymbol{\lambda}^{(\mathrm{q})}$ by the Transformer, and these embeddings are then passed through $f_{\mathrm{q}}$ to obtain individual outputs. We then apply a Softmax function to these outputs to ensure a proper probability distribution. $f_{\mathrm{q}}$ is an MLP consisting of an input layer with 32 hidden units, a ReLU activation function, and an output layer.

## B. 4 Training details

For all experiments, we use the same configuration to train our model. We set the initial learning rate to $5 \mathrm{e}-4$ and employ the cosine annealing learning rate scheduler. The number of training epochs is set to 50,000 for top- $k$ tasks and 100,000 for other tasks, and the batch size is 16 . For the REINFORCE, we use a discount factor of $\alpha=0.99$.

## C Full algorithm for training TNDP

```
Algorithm 1 Transformer Neural Decision Processes (TNDP)
    Input: Utility function \(u\left(y_{\Xi}, a\right)\), prior \(p(\theta)\), likelihood \(p(y \mid \theta, \xi)\), query horizon \(T\)
    Output: Trained TNDP
    while within the training budget do
        Sample \(\theta \sim p(\theta)\) and initialize \(D\)
        for \(t=1\) to \(T\) do
            \(\xi_{t}^{(\mathrm{q})} \sim \pi_{t}\left(\cdot \mid h_{1: t-1}\right) \quad \triangleright\) sample next design from policy
            Sample \(y_{t} \sim p(y \mid \theta, \xi) \quad \triangleright\) observe outcome
            Set \(h_{1: t}=h_{1: t-1} \cup\left\{\left(\xi_{t}^{(\mathrm{q})}, y_{t}\right)\right\} \quad \triangleright\) update history
            Set \(D^{(\mathrm{c})}=h_{1: t}, D^{(\mathrm{q})}=D^{(\mathrm{q})} \backslash\left\{\xi_{t}^{(\mathrm{q})}\right\} \quad \triangleright\) update \(D\)
            Calculate \(r_{t}\left(\xi_{t}^{(\mathrm{q})}\right)\) with \(u\left(y_{\Xi}, a\right)\) using Eq. (9) \(\triangleright\) calculate reward
        end for
            \(R_{t}=\sum_{k=t}^{T} \alpha^{k-t} r_{k}\left(\xi_{k}^{(\mathrm{q})}\right) \quad \triangleright\) calculate cumulative reward
        Update TNDP using \(\mathcal{L}^{(p)}\) (Eq. (7)) and \(\mathcal{L}^{(\mathrm{q})}\) (Eq. (10))
    end while
    At deployment, we can use \(f^{(\mathrm{q})}\) to sequentially query \(T\) designs. Afterward, based on the queried
        experiments, we perform one-step final decision-making using the prediction from \(f^{(p)}\).
```

---

#### Page 17

# D Computational cost analysis

## D. 1 Inference time analysis

We evaluate the inference time of our algorithm during the deployment stage. We select decisionaware active learning as the experiment for our time comparison. All experiments are evaluated on an Intel Core i7-12700K CPU. We measure both the acquisition time and the total time. The acquisition time refers to the time required to compute one next design, while the total time refers to the time required to complete 10 rounds of design. The final results are presented in Table A1, with the mean and standard deviation calculated over 10 runs.

Traditional methods rely on updating the GP and optimizing the acquisition function, which is computationally expensive. D-EIG and T-EIG require many model retraining steps to get the next design, which is not tolerable in applications requiring fast decision-making. However, since our model is fully amortized, once it is trained, it only requires a single forward pass to design the experiments, resulting in significantly faster inference times.

| Method                         | Acquisition time (s) | Total time (s) |
| :----------------------------- | :------------------- | :------------- |
| GP-RS                          | $0.00002(0.00001)$   | $28(7)$        |
| GP-US                          | $0.07(0.01)$         | $29(7)$        |
| GP-DUS                         | $0.38(0.02)$         | $44(5)$        |
| T-EIG (Sundin et al., 2018)    | $1558(376)$          | $15613(3627)$  |
| D-EIG (Filstroff et al., 2024) | $572(105)$           | $5746(1002)$   |
| TDNP (ours)                    | $0.015(0.004)$       | $0.31(0.06)$   |

Table A1: Comparison of computational costs across different methods. We report the mean value and (standard deviation) derived from 10 runs with different seeds.

## D. 2 Overall training time

Throughout this paper, we carried out all experiments, including baseline model computations and preliminary experiments not included in the final paper, on a GPU cluster featuring a combination of Tesla P100, Tesla V100, and Tesla A100 GPUs. We estimate the total computational usage to be roughly 5000 GPU hours. For each experiment, it takes around 10 GPU hours on a Tesla V100 GPU with 32 GB memory to reproduce the result, with an average memory consumption of 8 GB .

## E Details of toy example

## E. 1 Data generation

In our toy example, we generate data using a GP with the Squared Exponential (SE) kernel, which is defined as:

$$
k\left(x, x^{\prime}\right)=v \exp \left(-\frac{\left(x-x^{\prime}\right)^{2}}{2 \ell^{2}}\right)
$$

where $v$ is the variance, and $\ell$ is the lengthscale. Specifically, in each training iteration, we draw a random lengthscale $\ell \sim 0.25+0.75 \times U(0,1)$ and the variance $v \sim 0.1+U(0,1)$, where $U(0,1)$ denotes a uniform random variable between 0 and 1 .

## F Details of decision-aware active learning experiments

## F. 1 Data generation

For this experiment, we use a GP with a Squared Exponential (SE) kernel to generate our data. The covariates $x$ are drawn from a standard normal distribution. For each decision, we use an independent GP to simulate different outcomes. In each training iteration, the lengthscale for each GP is randomly sampled as $\ell \sim 0.25+0.75 \times U(0,1)$ and the variance as $v \sim 0.1+U(0,1)$, where $U(0,1)$ denotes a uniform random variable between 0 and 1 .

---

#### Page 18

# F. 2 Other methods description

We compare our method with other non-amortized approaches, all of which use GPs as the functional prior. Each model is equipped with an SE kernel with automatic relevance determination. GP hyperparameters are estimated with maximum marginal likelihood.

Our method is compared with the following methods:

- Random sampling (GP-RS): randomly choose the next design $\xi_{t}$ from the query set.
- Uncertainty sampling (GP-US): choose the next design $\xi_{t}$ for which the predictive distribution $p\left(y_{t} \mid \xi_{t}, h_{t-1}\right)$ has the largest variance.
- Decision uncertainty sampling (GP-DUS): choose the next design $\xi_{t}$ such that the predictive distribution of the optimal decision corresponding to this design is the most uncertain.
- Targeted information (GP-TEIG) (Sundin et al., 2018): a targeted active learning criterion, introduced by (Sundin et al., 2018), selects the next design $\xi_{t}$ that provides the highest EIG on $p\left(y^{*} \mid x^{*}, h_{t-1} \cup\left\{\left(\xi_{t}, y_{t}\right)\right\}\right)$.
- Decision EIG (GP-DEIG) (Filstroff et al., 2024): choose the next design $\xi_{t}$ which directly aims at reducing the uncertainty on the posterior distribution of the optimal decision. See Filstroff et al. (2024) for a detailed explanation.

## F. 3 Ablation study

We also carry out an ablation study to verify the effectiveness of our query head and the non-myopic objective function. We first compare TNDP with TNDP using random sampling (TNDP-RS), and the results are shown in Fig. A1(a). We observe that the designs proposed by the query head significantly improve accuracy, demonstrating that the query head can propose informative designs based on downstream decisions.

We also evaluate the impact of the non-myopic objective by comparing TNDP with a myopic version that only optimizes for immediate utility rather than long-term gains $(\alpha=0)$. The results, presented in Fig. A1(b), show that TNDP with the non-myopic objective function achieves higher accuracy across iterations compared to using the myopic objective. This indicates that our non-myopic objective effectively captures the long-term benefits of each design choice, leading to improved overall performance.

> **Image description.** This is a line graph comparing the performance of two methods, TNDP and TNDP-RS, over 10 design steps.
>
> The graph has the following elements:
>
> - **Axes:** The x-axis is labeled "Design steps t" and ranges from 1 to 10. The y-axis is labeled "Proportion of correct decisions (%)" and ranges from 0.5 to 0.9. Grid lines are visible in the background of the plot.
> - **Data:** There are two lines plotted on the graph:
>   - A green line with triangle markers, labeled "TNDP-RS". This line starts around 0.59 at design step 1 and gradually increases to approximately 0.72 at design step 10. A shaded green area surrounds the line, representing the uncertainty or variability in the data.
>   - An orange line with star markers, labeled "TNDP". This line starts around 0.63 at design step 1, increases more rapidly than the green line, and plateaus around 0.85 at design step 10. A shaded orange area surrounds the line, representing the uncertainty or variability in the data.
> - **Title:** Below the graph, there is a title: "(a) Effect query head".
> - **Legend:** A legend is present at the bottom right of the plot, associating the green line with "TNDP-RS" and the orange line with "TNDP".
>
> The graph visually demonstrates that TNDP generally achieves a higher proportion of correct decisions compared to TNDP-RS across the design steps.

(a) Effect query head

> **Image description.** This is a line graph comparing the "Proportion of correct decisions (%)" against "Design steps t".
>
> The graph has the following characteristics:
>
> - **Axes:** The x-axis is labeled "Design steps t" and ranges from 1 to 10. The y-axis is labeled "Proportion of correct decisions (%)" and ranges from 0.5 to 0.9.
> - **Data:** There are two lines plotted on the graph:
>   - A red line with square markers, labeled "myopic".
>   - An orange line with star markers, labeled "non-myopic".
> - **Shaded Regions:** Each line has a shaded region around it, indicating some measure of variance or uncertainty. The "myopic" line has a red shaded region, and the "non-myopic" line has an orange shaded region.
> - **Trends:** The "non-myopic" line generally shows a higher proportion of correct decisions than the "myopic" line. Both lines show an increasing trend as the number of design steps increases.

(b) Impact of non-myopic objective

Figure A1: Comparison of TNDP variants on the decision-aware active learning task. (a) Shows the effect of the query head, where TNDP outperforms TNDP-RS, demonstrating its ability to generate informative designs. (b) Illustrates the impact of the non-myopic objective, with TNDP achieving higher accuracy than the myopic version.

---

#### Page 19

# G Details of top- $k$ hyperparameter optimization experiments

## G. 1 Data

In this task, we use HPO-B benchmark datasets (Arango et al., 2021). The HPO-B dataset is a large-scale benchmark for HPO tasks, derived from the OpenML repository. It consists of 176 search spaces (algorithms) evaluated on 196 datasets, with a total of 6.4 million hyperparameter evaluations. This dataset is designed to facilitate reproducible and fair comparisons of HPO methods by providing explicit experimental protocols, splits, and evaluation measures.
We extract four meta-datasets from the HPOB dataset: ranger ( $\mathrm{id}=7609, d_{x}=9$ ), svm ( $\mathrm{id}=5891, d_{x}=8$ ), rpart ( $\mathrm{id}=5859, d_{x}=6$ ), and xgboost ( $\mathrm{id}=5971, d_{x}=16$ ). In the test stage, the initial context set is chosen based on their pre-defined indices. For detailed information on the datasets, please refer to https://github.com/releaunifreiburg/HPO-B.

## G. 2 Other methods description

In our experiments, we compare our method with several common acquisition functions used in HPO. We use GPs as surrogate models for these acquisition functions. All the implementations are based on BoTorch (Balandat et al., 2020) (https://botorch.org/). The acquisition functions compared are as follows:

- Random Sampling (RS): This method selects hyperparameters randomly from the search space, without using any surrogate model or acquisition function.
- Upper Confidence Bound (UCB): This acquisition function balances exploration and exploitation by selecting points that maximize the upper confidence bound. The UCB is defined as:

$$
\alpha_{\mathrm{UCB}}(\mathbf{x})=\mu(\mathbf{x})+\kappa \sigma(\mathbf{x})
$$

where $\mu(\mathbf{x})$ is the predicted mean, $\sigma(\mathbf{x})$ is the predicted standard deviation, and $\kappa$ is a parameter that controls the trade-off between exploration and exploitation.

- Expected Improvement (EI): This acquisition function selects points that are expected to yield the greatest improvement over the current best observation. The EI is defined as:

$$
\alpha_{\mathrm{EI}}(\mathbf{x})=\mathbb{E}\left[\max \left(0, f(\mathbf{x})-f\left(\mathbf{x}^{+}\right)\right)\right]
$$

where $f\left(\mathbf{x}^{+}\right)$is the current best value observed, and the expectation is taken over the predictive distribution of $f(\mathbf{x})$.

- Probability of Improvement (PI): This acquisition function selects points that have the highest probability of improving over the current best observation. The PI is defined as:

$$
\alpha_{\mathrm{PI}}(\mathbf{x})=\Phi\left(\frac{\mu(\mathbf{x})-f\left(\mathbf{x}^{+}\right)-\omega}{\sigma(\mathbf{x})}\right)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution, $f\left(\mathbf{x}^{+}\right)$ is the current best value observed, and $\omega$ is a parameter that encourages exploration.

In addition to those non-amortized GP-based methods, we also compare our method with an amortized surrogate model PFNs4BO (Müller et al., 2023). It is a Transformer-based model designed for hyperparameter optimization which does not consider the downstream task. We use the pre-trained PFNs4BO-BNN model which is trained on HPO-B datasets and choose PI as the acquisition function, the model and the training details can be found in their official implementation (https://github. com/automl/PFNs4BO).

## G. 3 Additional experiment on retrosynthesis planning

We now show a real-world experiment on retrosynthesis planning (Blacker et al., 2011). Specifically, our task is to help chemists identify the top- $k$ synthetic routes for a novel molecule (Mo et al., 2021), as it can be challenging to select the most practical routes from many random options generated by the retrosynthesis software (Stevens, 2011; Szymkuć et al., 2016). In this task, the design space for each molecule $m$ is a finite set of routes that can synthesize the molecule. The sequential experimental

---

#### Page 20

design is to select a route for a specific molecule to query its score $y$, which is calculated based on the tree edit distance (Bille, 2005) from the best route. The downstream task is to recommend the top- $k$ routes with the highest target-specific scores based on the collected information.

In this experiment, we choose $k=3$ and $T=10$, and set the global information $\gamma=m$. We train our TNDP on a novel non-public metadataset, including 1500 molecules with 70 synthetic routes for each molecule. The representation dimension of the molecule is 64 and that of the route is 264 , both of which are learned through a neural network. Given the highdimensional nature of the data representation, it is challenging to directly compare TNDP with other GP-based methods, as GPs typically struggle with scalability and performance in such high-dimensional settings. Therefore, we only compare TNDP with TNDP using random sampling. The final results are evaluated on 50 test molecules that are not seen during training, as shown in Fig. A2.

> **Image description.** This is a line graph comparing the performance of two methods, TNDP and Random search, over a series of design steps.
>
> The graph has the following characteristics:
>
> - **Axes:** The x-axis is labeled "Design steps t" and ranges from 0 to 10. The y-axis is labeled "Utility" and ranges from 6 to 18.
> - **Data:** There are two lines plotted on the graph:
>   - One line, in orange, represents the TNDP method. This line is marked with star-shaped data points.
>   - The other line, in red, represents the Random search method. This line is marked with square-shaped data points.
> - **Error Bands:** Each line has a shaded area around it, representing the standard deviation or confidence interval. The TNDP line has a light orange shaded area, while the Random search line has a light red shaded area.
> - **Legend:** A legend is present in the lower part of the graph, labeling the orange line as "TNDP" and the red line as "Random search".
> - **Overall Trend:** Both lines show an increasing trend, indicating that utility increases with the number of design steps. The TNDP method generally outperforms the Random search method, as its line is consistently higher on the graph.

Figure A2: Results of retrosynthesis planning experiment. The utility is the sum of the quality scores of top- $k$ routes and is calculated with 50 molecules. Our TNDP outperforms the random search baseline.