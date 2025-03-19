```
@article{huang2024bamortized,
  title={Amortized Decision-Aware Bayesian Experimental Design},
  author={Daolang Huang and Yujia Guo and Luigi Acerbi and Samuel Kaski},
  year={2024},
  journal={NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty},
  doi={10.48550/arXiv.2411.02064}
}
```

---

#### Page 1

# Amortized Decision-Aware Bayesian Experimental Design

Daolang Huang<br>Aalto University<br>daolang.huang@aalto.fi<br>Luigi Acerbi<br>University of Helsinki<br>luigi.acerbi@helsinki.fi

Yujia Guo<br>Aalto University<br>yujia.guo@aalto.fi<br>Samuel Kaski<br>Aalto University<br>University of Manchester<br>samuel.kaski@aalto.fi

#### Abstract

Many critical decisions are made based on insights gained from designing, observing, and analyzing a series of experiments. This highlights the crucial role of experimental design, which goes beyond merely collecting information on system parameters as in traditional Bayesian experimental design (BED), but also plays a key part in facilitating downstream decision-making. Most recent BED methods use an amortized policy network to rapidly design experiments. However, the information gathered through these methods is suboptimal for down-the-line decision-making, as the experiments are not inherently designed with downstream objectives in mind. In this paper, we present an amortized decision-aware BED framework that prioritizes maximizing downstream decision utility. We introduce a novel architecture, the Transformer Neural Decision Process (TNDP), capable of instantly proposing the next experimental design, whilst inferring the downstream decision, thus effectively amortizing both tasks within a unified workflow. We demonstrate the performance of our method across two tasks, showing that it can deliver informative designs and facilitate accurate decision-making ${ }^{1}$.

## 1 Introduction

A fundamental challenge in a wide array of disciplines is the design of experiments to infer unknown properties of the systems under study [9, 7]. Bayesian Experimental Design (BED) [17, 8, 24, 22] is a powerful framework to guide and optimize experiments by maximizing the expected amount of information about parameters gained from experiments, see Fig. 1(a). To pick the next optimal design, standard BED methods require estimating and optimizing the expected information gain (EIG) over the design space, which can be extremely time-consuming. This limitation has led to the development of amortized BED [10, 14, 5, 6], a policy-based method which leverages a neural network policy trained on simulated experimental trajectories to quickly generate designs, as illustrated in Fig. 1(b).

However, the ultimate goal in many tasks extends beyond parameter inference to inform a downstream decision-making task by leveraging our understanding of these parameters, such as in personalized medical diagnostics [4]. Previous amortized BED methods do not take down-the-line decision-making tasks into account, which is suboptimal for decision-making in scenarios where experiments can be adaptively designed. Loss-calibrated inference, which was originally introduced by Lacoste-Julien

[^0]
[^0]: ${ }^{1}$ The full version of this work can be found at: https://arxiv.org/abs/2411.02064.

---

#### Page 2

> **Image description.** This image presents a diagram comparing three different workflows for Bayesian Experimental Design (BED). The diagram is divided into three panels, labeled (a), (b), and (c), each depicting a different approach.
>
> Panel (a) illustrates "Traditional BED". It shows a cycle of "Design" (represented by a beaker), "Observe" leading to "Outcome" (represented by a molecule structure), and "Inference" leading to "Posterior" (represented by a bell curve). An arrow labeled "Optimize" leads from the "Posterior" back to "Design", completing the cycle.
>
> Panel (b) illustrates "Amortized BED". It shows a cycle of "Design" (beaker), "Observe" leading to "Outcome" (molecule structure). An arrow labeled "Generate" leads from "Policy" (represented by a neural network diagram) to "Design". An arrow labeled "Update history" leads from "Outcome" back to "Policy". The "Policy" box is enclosed in a dashed-line box labeled "Offline training stage".
>
> Panel (c) illustrates "Our decision-aware amortized BED". It shows a cycle of "Design" (beaker), "Observe" leading to "Outcome" (molecule structure). An arrow labeled "Generate" leads from "TNDP" (represented by a neural network diagram) to "Design". An arrow labeled "Update history" leads from "Outcome" back to "TNDP". An arrow labeled "Feedback" leads from "Decision" (represented by a signpost icon) back to "Decision Utility" (represented by a question mark and checkmark/X mark icons). An arrow labeled "Estimate" leads from "TNDP" to "Decision Utility". An arrow leads from "Decision Utility" to "Decision". The "TNDP" and "Decision Utility" boxes are enclosed in a dashed-line box labeled "Offline training stage".

Figure 1: Overview of BED workflows. (a) Traditional BED, which iterates between optimizing designs, running experiments, and updating the model via Bayesian inference. (b) Amortized BED, which uses a policy network for rapid experimental design generation. (c) Our decision-aware amortized BED integrates decision utility in the training to facilitate downstream decision-making.
et al. [16] for variational approximations in Bayesian inference, provides a concept that adjusts the inference process to capture posterior regions critical for decision-making tasks. Inspired by this concept, we consider integrating decision-making directly into the experimental design process to align the proposed experimental designs more closely with the ultimate decision-making task.
In this paper, we propose an amortized decision-making-aware BED framework, see Fig. 1(c). We identify two key aspects where previous amortized BED methods fall short when applied to downstream decision-making tasks. First, the training objective of the existing methods does not consider downstream decision tasks. Therefore, we introduce the concept of Decision Utility Gain (DUG) to guide experimental design to better align with the downstream objective. Second, to obtain the optimal decision, we still need to explicitly approximate the predictive distribution of the outcomes to estimate the utility. Current amortized methods learn this distribution only implicitly and therefore do not fully amortize the decision-making process. To address this, we propose a novel Transformer neural decision process (TNDP) architecture, where the system can instantly propose informative experimental designs and make final decisions. Finally, we train under a non-myopic objective function that ensures decisions are made with consideration of future outcomes. We empirically show the effectiveness of our method through two tasks.

## 2 Decision-aware BED

### 2.1 Preliminaries and problem setup

In this paper, we consider scenarios in which we design a series of experiments $\xi \in \Xi$ and observe corresponding outcomes $y$ to inform a final decision-making step. The experimental history is denoted as $h_{1: t}=\left\{\left(\xi_{1}, y_{1}\right), \ldots,\left(\xi_{t}, y_{t}\right)\right\}$ and we assume a fixed experimental budget with $T$ query steps. Our objective is to identify an optimal decision $a^{*}$ from a set of possible decisions $\mathcal{A}$ at time $T$.

To make decisions under uncertainty, Bayesian decision theory [3] provides an axiomatic framework by incorporating probabilistic beliefs about unknown parameters into decision-making. Given a taskspecific utility function $u(\theta, a)$, which quantifies the value of the outcomes from different decisions $a \in \mathcal{A}$ when the system is in state $\theta$, the optimal decision is then determined by maximizing the expected utility under the posterior distribution of the parameters $p\left(\theta \mid h_{1: t}\right)$.
In many real-world scenarios, outcomes are stochastic and it is more typical to make decisions based on their predictive distribution $p\left(y \mid \xi, h_{1: t}\right)=\mathbb{E}_{p\left(\theta \mid h_{1: t}\right)}[p\left(y \mid \xi, \theta, h_{1: t}\right)]$, such as in clinical

---

#### Page 3

trials where the optimal treatment is chosen based on predicted patient responses. A similar setup can be found in $[15,28]$. Thus, we can represent our belief directly as $p\left(y_{\Xi} \mid h_{1: t}\right) \equiv\left\{p(y \mid \xi, h_{1: t})\right\}_{\xi \in \Xi}$, which is a stochastic process that defines a joint predictive distribution of outcomes indexed by the elements of the design set $\Xi$, given the current information $h_{1: t}$. Our utility function is then expressed as $u\left(y_{\Xi}, a\right)$, which is a natural extension of the traditional definition of utility by marginalizing out the posterior distribution of $\theta$. The rule for making the optimal decision is then reformulated as:

$$
a^{*}=\underset{a \in A}{\arg \max } \mathbb{E}_{p\left(y_{\Xi} \mid h_{1: t}\right)}\left[u\left(y_{\Xi}, a\right)\right]
$$

# 2.2 Decision utility gain

To quantify the effectiveness of each experimental design in terms of decision-making, we introduce Decision Utility Gain (DUG), which is defined as the difference in the expected utility of the best decision, with the new information obtained from the current experimental design, versus the best decision with the information obtained from previous experiments.
Definition 2.1. Given a historical experimental trajectory $h_{1: t-1}$, the Decision Utility Gain (DUG) for a given design $\xi_{t}$ and its corresponding outcome $y_{t}$ at step $t$ is defined as follows:

$$
\operatorname{DUG}\left(\xi_{t}, y_{t}\right)=\max _{a \in A} \mathbb{E}_{p\left(y_{\Xi} \mid h_{1: t-1} \cup\left\{\left(\xi_{t}, y_{t}\right)\right\}\right)}\left[u\left(y_{\Xi}, a\right)\right]-\max _{a \in A} \mathbb{E}_{p\left(y_{\Xi} \mid h_{1: t-1}\right)}\left[u\left(y_{\Xi}, a\right)\right]
$$

DUG measures the improvement in the maximum expected utility from observing a new experimental design, differing in this from standard marginal utility gain (see e.g., [12]). The optimal design is the one that provides the largest increase in maximal expected utility. At the time we choose the design $\xi_{t}$, the outcome remains uncertain. Therefore, we should consider the Expected Decision Utility Gain (EDUG) to select the next design, which is defined as $\operatorname{EDUG}\left(\xi_{t}\right)=\mathbb{E}_{p\left(y_{t} \mid \xi_{t}, h_{1: t-1}\right)}\left[\operatorname{DUG}\left(\xi_{t}, y_{t}\right)\right]$. The one-step lookahead optimal design can be determined by maximizing EDUG with $\xi^{*}=\arg \max _{\xi \in \Xi} \operatorname{EDUG}(\xi)$. However, in practice, the true predictive distributions are often unknown, making the optimization of EDUG exceptionally challenging. This difficulty arises due to the inherent bi-level optimization problem and the need to evaluate two layers of expectations.
To avoid the expensive cost of optimizing EDUG, we propose using a policy network that directly maps historical data to the next design. This approach sidesteps the need to iteratively optimize EDUG by learning a design strategy over many simulated experiment trajectories beforehand.

### 2.3 Amortization with TNDP

Our architecture, termed Transformer Neural Decision Process (TNDP), is a novel architecture building upon the Transformer neural process (TNP) [20]. It aims to amortize both the experimental design and the subsequent decision-making. A general introduction to TNP can be found in Appendix A.
The data architecture of our system comprises four parts: A context set $D^{(\mathrm{c})}=\left\{\left(\xi_{i}^{(\mathrm{c})}, y_{i}^{(\mathrm{c})}\right)\right\}_{i=1}^{t}$ contains all past $t$-step designs and outcomes; A prediction set $D^{(\mathrm{p})}=\left\{\left(\xi_{i}^{(\mathrm{p})}, y_{i}^{(\mathrm{p})}\right)\right\}_{i=1}^{n_{\mathrm{p}}}$ consists of $n_{\mathrm{p}}$ design-outcome pairs used for approximating $p\left(y_{\Xi} \mid h_{1: t}\right)$. The output from this head can then be used to estimate the expected utility; A query set $D^{(\mathrm{q})}=\left\{\xi_{i}^{(\mathrm{q})}\right\}_{i=1}^{n_{\mathrm{q}}}$ consists of $n_{\mathrm{q}}$ candidate experimental designs being considered for the next step; Global information $\mathrm{GI}=[t, \gamma]$ where $t$ represents the current step, and $\gamma$ encapsulates task-related information, which could include contextual data relevant to the decision-making process.
TNDP comprises four main components, the full architecture is shown in Fig. 2(a). At first, the data embedder block $f_{\text {emb }}$ maps each set of $D$ to an aligned embedding space. The embeddings are then concatenated to form a unified representation $\boldsymbol{E}=\operatorname{concat}\left(\boldsymbol{E}^{(\mathrm{c})}, \boldsymbol{E}^{(\mathrm{p})}, \boldsymbol{E}^{(\mathrm{q})}, \boldsymbol{E}^{(\mathrm{d})}\right)$. After the initial embedding, the Transformer block $f_{\text {tfm }}$ processes $\boldsymbol{E}$ using attention mechanisms that allow for selective interactions between different data components, ensuring that each part contributes appropriately to the final output. Fig. 2(b) shows an example attention mask. The output of $f_{\text {tfm }}$ is then split according to the specific needs of the query and prediction head $\boldsymbol{\lambda}=\left[\boldsymbol{\lambda}^{(\mathrm{p})}, \boldsymbol{\lambda}^{(\mathrm{q})}\right]=f_{\text {tfm }}(\boldsymbol{E})$.

The primary role of the prediction head $f_{\mathrm{p}}$ is to approximate $p\left(y_{\Xi} \mid h_{1: t}\right)$ at any step $t$ with a family of parameterized distributions $q\left(y_{\Xi} \mid \boldsymbol{p}_{t}\right)$, where $\boldsymbol{p}_{t}=f_{\mathrm{p}}\left(\boldsymbol{\lambda}_{t}^{(\mathrm{p})}\right)$ is the output of $f_{\mathrm{p}}$. We choose a Gaussian

---

#### Page 4

> **Image description.** This image contains two diagrams, labeled (a) and (b), illustrating the architecture of TNDP (likely a machine learning model).
>
> **Diagram (a):**
>
> - It shows a flow diagram representing the data processing pipeline.
> - At the bottom, there's a horizontal "Data Embedding Block f_emb" represented as a light orange rectangle. Below this block are labels: "t Global Info GI", "γ", "ξ_1^(c), y_1^(c) Context Set D^(c)", "ξ_2^(c), y_2^(c)", "ξ_1^(p) Prediction Set D^(p)", "ξ_2^(p)", "ξ_1^(q) Query Set D^(q)", and "ξ_2^(q)".
> - Above the "Data Embedding Block", there are upward arrows leading to a "Transformer Block f_tfm" represented as a blue rectangle. The arrows are labeled "E^t", "E^γ", "E_1^(c)", "E_2^(c)", "E_1^(p)", "E_2^(p)", "E_1^(q)", and "E_2^(q)". There are "+" symbols above the arrows from "E_1^(c)" and "E_2^(c)".
> - Above the "Transformer Block", there are two upward arrows leading to two separate blocks. The left block is labeled "Prediction Head f_p" and is colored orange, with an arrow pointing up from it labeled "q(y^(p)|p)". The right block is labeled "Query Head f_q" and is colored green, with an arrow pointing up from it labeled "π_t(.|h_1:t-1)". The arrows leading into the Prediction and Query Head are labeled "λ^(p)" and "λ^(q)" respectively.
>
> **Diagram (b):**
>
> - It shows a grid of squares, some filled with a light blue color and some left white.
> - The rows are labeled on the left with "GI", "D^(c)", "D^(p)", and "D^(q)" with brackets indicating the range of elements belonging to that category. Specifically, the D^(c), D^(p), and D^(q) labels have brackets indicating that they refer to two rows each.
> - The columns are labeled at the top with "D^(c)", "D^(p)", and "D^(q)" with brackets indicating the range of elements belonging to that category. The D^(c) label has "ξ_1^(c), y_1^(c)" and "ξ_2^(c), y_2^(c)" above two columns, while D^(p) has "ξ_1^(p)" and "ξ_2^(p)" above two columns, and D^(q) has "ξ_1^(q)" and "ξ_2^(q)" above two columns.
> - The grid represents an attention mask, where the colored squares likely indicate attention between elements.

Figure 2: Illustration of TNDP. (a) An overview of TNDP architecture with input consisting of 2 observed design-outcome pairs from $D^{(\mathrm{c})}, 2$ designs from $D^{(\mathrm{p})}$ for prediction, and 2 candidate designs from $D^{(\mathrm{q})}$ for query. (b) The corresponding attention mask. The colored squares indicate that the elements on the left can attend to the elements on the top in the self-attention layer of $f_{\mathrm{ffm}}$.
likelihood and train $f_{\mathrm{p}}$ by minimizing the negative log-likelihood of the predicted probabilities:

$$
\mathcal{L}^{(\mathrm{p})}=-\sum_{t=1}^{T} \sum_{i=1}^{n_{\mathrm{p}}} \log q\left(y_{i}^{(\mathrm{p})} \mid \boldsymbol{p}_{i, t}\right)=-\sum_{t=1}^{T} \sum_{i=1}^{n_{\mathrm{p}}} \log \mathcal{N}\left(y_{i}^{(\mathrm{p})} \mid \boldsymbol{\mu}_{i, t}, \boldsymbol{\sigma}_{i, t}^{2}\right)
$$

where $\boldsymbol{p}_{i, t}$ represents the output of design $\xi_{i}^{(\mathrm{p})}$ at step $t, \boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are the predicted mean and standard deviation split from $\boldsymbol{p}$.
Lastly, the query head $f_{\mathrm{q}}$ processes the embeddings $\boldsymbol{\lambda}^{(\mathrm{q})}$ from the Transformer block to generate a policy distribution over possible experimental designs. The outputs of the query head, $\boldsymbol{q}=f_{\mathrm{q}}\left(\boldsymbol{\lambda}^{(\mathrm{q})}\right)$, are mapped to a probability distribution $\pi\left(\xi_{t}^{(\mathrm{q})} \mid h_{1: t-1}\right)$ via a Softmax function. To design a reward signal that guides the query head $f_{\mathrm{q}}$ in proposing informative designs, we first define a singlestep immediate reward based on DUG (Eq. (2)), replacing the true predictive distribution with our approximated distribution:

$$
r_{t}\left(\xi_{t}^{(\mathrm{q})}\right)=\max _{a \in A} \mathbb{E}_{q\left(y_{\mathbb{E}} \mid \boldsymbol{p}_{t}\right)}[u\left(y_{\mathbb{E}}, a\right)]-\max _{a \in A} \mathbb{E}_{q\left(y_{\mathbb{E}} \mid \boldsymbol{p}_{t-1}\right)}[u\left(y_{\mathbb{E}}, a\right)]
$$

This reward quantifies how the experimental design influences our decision-making by estimating the improvement in expected utility that results from incorporating new experimental outcomes. However, this objective remains myopic, as it does not account for the future or the final decision-making. To address this, we employ the REINFORCE algorithm [30]. The final loss of $f_{\mathrm{q}}$ can be written as:

$$
\mathcal{L}^{(\mathrm{q})}=-\sum_{t=1}^{T} R_{t} \log \pi\left(\xi_{t}^{(\mathrm{q})} \mid h_{1: t-1}\right)
$$

where $R_{t}=\sum_{k=t}^{T} \alpha^{k-t} r_{k}\left(\xi_{k}^{(\mathrm{q})}\right)$ represents the non-myopic discounted reward. The discount factor $\alpha$ is used to decrease the importance of rewards received at later time step. $\xi_{t}^{(\mathrm{q})}$ is obtained through sampling from the policy distribution $\xi_{t}^{(\mathrm{q})} \sim \pi\left(\cdot \mid h_{1: t-1}\right)$. The details of implementing and training TNDP are shown in Appendix B.

## 3 Experiments

### 3.1 Toy example: targeted active learning

We begin with an illustrative example to show how our TNDP works. We consider a synthetic regression task where the goal is to perform regression at a specific test point $x^{*}$ on an unknown function. To accurately predict this point, we need to actively collect some new points to query.

---

#### Page 5

The design space $\Xi=\mathcal{X}$ is the domain of $x$, and $y$ is the corresponding noisy observations of the function. Let $\mathcal{Q}(\mathcal{X})$ denote the set of combinations of distributions that can be output by TNDP, we can then define decision space to be $\mathcal{A}=\mathcal{Q}(\mathcal{X})$. The downstream decision is to output a predictive distribution for $y^{*}$ given a test point $x^{*}$, and the utility function $u\left(y_{\Xi}, a\right)=\log q\left(y^{*} \mid x^{*}, h_{1: t}\right)$ is the log probability of $y$ under the predicted distribution.
During training, we sample functions from Gaussian Processes (GPs) [23] with squared exponential kernels of varying output variances and lengthscales and randomly sample a point as the test point $x^{*}$. We set the global contextual information $\lambda$ as the test point $x^{*}$. For illustration purposes, we consider only the case where $T=1$. Additional details for the data generation can be found in Appendix C.

> **Image description.** The image consists of two plots stacked vertically, both related to a toy example.
>
> The top plot shows a 2D graph with the x-axis ranging from 0 to 1 and the y-axis ranging from 0 to 2. A black curve, labeled "target function" in the legend, represents the underlying function. Several pink crosses, labeled "queried data" in the legend, are scattered along the curve. A blue star, labeled "next query" in the legend, is positioned on the curve near x=0.5 and y=1.7. A vertical dashed red line, labeled "x\*" in the legend, is located at approximately x=0.45.
>
> The bottom plot is a distribution plot, with the x-axis aligned with the top plot, ranging from 0 to 1. The y-axis, labeled "π", ranges from 0 to 0.005. A green filled curve, resembling a Gaussian distribution, is centered around x=0.45, with the area under the curve filled in green. The vertical dashed red line from the top plot extends into the bottom plot, intersecting the peak of the green curve. The x-axis is labeled "x".

Figure 3: Results of toy example. The top figure represents the true function and the initial known points. The red line indicates the location of $x^{*}$. The blue star marks the next query point, sampled from the policy's predicted distribution shown in the bottom figure.

Results. From Fig. 3, we can observe that the values of $\pi$ concentrate near $x^{*}$, meaning our query head $f_{\mathrm{q}}$ tends to query points close to $x^{*}$ to maximize the DUG. This is an intuitive example demonstrating that our TNDP can adjust its design strategy based on the downstream task.

> **Image description.** The image consists of four line graphs arranged horizontally, each representing the performance of different hyperparameter optimization algorithms on a specific machine learning model.
>
> - **Overall Structure:** Each graph has the same basic structure:
>
>   - X-axis: Labeled "Step t," ranging from 0 to 50.
>   - Y-axis: Labeled "Utility," with different ranges for each graph to accommodate the specific utility values.
>   - Multiple colored lines representing different optimization algorithms. Shaded regions around each line indicate the standard deviation.
>   - A legend at the bottom of the image identifies each color with a specific algorithm: RS (yellow dotted line), UCB (green solid line), EI (red dashed line), PI (purple dash-dotted line), PFNs4BO (cyan solid line), and TNDP (orange solid line with star markers).
>
> - **Individual Graphs:**
>
>   - **ranger:** Y-axis ranges from 2.74 to 2.82.
>   - **rpart:** Y-axis ranges from 2.28 to 2.52.
>   - **svm:** Y-axis ranges from 2.72 to 2.88.
>   - **xgboost:** Y-axis ranges from 2.60 to 2.80.
>
> - **General Trends:**
>   - Most algorithms show a rapid increase in utility during the initial steps, followed by a plateau.
>   - The TNDP algorithm (orange line) generally achieves the highest utility values across all four models.
>   - The RS algorithm (yellow dotted line) often shows the lowest utility values.
>
> The titles above each graph indicate the machine learning model being optimized: "ranger," "rpart," "svm," and "xgboost."

Figure 4: Average utility on Top- $k$ HPO task. The error bars represent the standard deviation over five runs. TNDP consistently achieved the best performance regarding the utility.

# 3.2 Top- $k$ hyperparameter optimization

In traditional optimization tasks, we typically only aim to find a single point that maximizes the underlying function $f$. However, instead of identifying a single optimal point, there are scenarios where we wish to estimate a set of top- $k$ distinct optima, such as in materials discovery [18, 27].
In this experiment, we choose hyperparameter optimization (HPO) tasks and conduct experiments on the HPO-B datasets [1]. The design space $\Xi \subseteq \mathcal{X}$ is a finite set defined over the hyperparameter space and the outcome $y$ is the accuracy of a given configuration. Our decision is to find $k$ hyperparameter sets, denoted as $a=\left(a_{1}, \ldots, a_{k}\right) \in A \subseteq \mathcal{X}^{k}$, with $a_{i} \neq a_{j}$. The utility function is then defined as $u\left(y_{\Xi}, a\right)=\sum_{i=1}^{k} y_{a_{i}}$, where $y_{a_{i}}$ is the accuracy corresponding to the configuration $a_{i}$.
We compare our methods with five different BO acquisition functions: random sampling (RS), Upper Confidence Bound (UCB), Expected Improvement (EI), Probability of Improvement (PI), and an amortized method PFNs4BO [19]. We set $k=3$ and $T=50$. Our experiments are conducted on four search spaces. All results are evaluated on a predefined test set. For more details, see Appendix D.

Results. From the results (Fig. 4), our method demonstrates superior performance across all four meta-datasets, particularly during the first 10 queries.

## 4 Discussion and conclusion

In this paper, we introduced a decision-aware amortized BED framework with a novel TNDP architecture to optimize experimental designs for better decision-making. Future work includes conducting more extensive empirical tests and ablation studies, deploying more advanced RL algorithms [26] to enhance training stability, addressing robust experimental design under model misspecification [22, 13, 25], and developing dimension-agnostic methods to expand the scope of amortization.
