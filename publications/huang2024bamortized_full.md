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

---

# Amortized Decision-Aware Bayesian Experimental Design - Backmatter

---

#### Page 6

# References 

[1] Sebastian Pineda Arango, Hadi Samer Jomaa, Martin Wistuba, and Josif Grabocka. Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.
[2] Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. Botorch: A framework for efficient monte-carlo bayesian optimization. Advances in neural information processing systems, 33, 2020.
[3] James O Berger. Statistical decision theory and Bayesian analysis. Springer Science \& Business Media, 2013.
[4] Ioana Bica, Ahmed M Alaa, Craig Lambert, and Mihaela Van Der Schaar. From real-world patient data to individualized treatment effects using machine learning: current and future methods to address underlying challenges. Clinical Pharmacology \& Therapeutics, 109(1): $87-100,2021$.
[5] Tom Blau, Edwin V Bonilla, Iadine Chades, and Amir Dezfouli. Optimizing sequential experimental design with deep reinforcement learning. In International conference on machine learning, pages 2107-2128. PMLR, 2022.
[6] Tom Blau, Edwin Bonilla, Iadine Chades, and Amir Dezfouli. Cross-entropy estimators for sequential experiment design with reinforcement learning. arXiv preprint arXiv:2305.18435, 2023.
[7] Martin Burger, Andreas Hauptmann, Tapio Helin, Nuutti Hyvönen, and Juha-Pekka Puska. Sequentially optimized projections in x-ray imaging. Inverse Problems, 37(7):075006, 2021.
[8] Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. Statistical science, pages 273-304, 1995.
[9] Yi Cheng and Yu Shen. Bayesian adaptive designs for clinical trials. Biometrika, 92(3):633-646, 2005.
[10] Adam Foster, Desi R Ivanova, Ilyas Malik, and Tom Rainforth. Deep adaptive design: Amortizing sequential bayesian experimental design. In International Conference on Machine Learning, pages 3384-3395. PMLR, 2021.
[11] Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. Conditional neural processes. In International conference on machine learning, pages 1704-1713. PMLR, 2018.
[12] Roman Garnett. Bayesian optimization. Cambridge University Press, 2023.
[13] Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, and Samuel Kaski. Learning robust statistics for simulation-based inference under model misspecification. Advances in Neural Information Processing Systems, 36, 2023.
[14] Desi R Ivanova, Adam Foster, Steven Kleinegesse, Michael U Gutmann, and Thomas Rainforth. Implicit deep adaptive design: Policy-based experimental design without likelihoods. Advances in Neural Information Processing Systems, 34, 2021.
[15] Tomasz Kuśmierczyk, Joseph Sakaya, and Arto Klami. Variational bayesian decision-making for continuous utilities. Advances in Neural Information Processing Systems, 32, 2019.
[16] Simon Lacoste-Julien, Ferenc Huszár, and Zoubin Ghahramani. Approximate inference for the loss-calibrated bayesian. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 416-424. JMLR Workshop and Conference Proceedings, 2011.
[17] Dennis V Lindley. On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, 27(4):986-1005, 1956.

---

#### Page 7

[18] Yue Liu, Tianlu Zhao, Wangwei Ju, and Siqi Shi. Materials discovery and design using machine learning. Journal of Materiomics, 3(3):159-177, 2017.
[19] Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. Pfns4bo: In-context learning for bayesian optimization. In International Conference on Machine Learning, pages 25444-25470. PMLR, 2023.
[20] Tung Nguyen and Aditya Grover. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In International Conference on Machine Learning, pages 16569-16594. PMLR, 2022.
[21] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.
[22] Tom Rainforth, Adam Foster, Desi R Ivanova, and Freddie Bickford Smith. Modern bayesian experimental design. Statistical Science, 39(1):100-114, 2024.
[23] Carl Edward Rasmussen and Christopher KI Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.
[24] Elizabeth G Ryan, Christopher C Drovandi, James M McGree, and Anthony N Pettitt. A review of modern computational algorithms for bayesian optimal design. International Statistical Review, 84(1):128-154, 2016.
[25] Marvin Schmitt, Paul-Christian Bürkner, Ullrich Köthe, and Stefan T Radev. Detecting model misspecification in amortized bayesian inference with neural networks: An extended investigation. arXiv preprint arXiv:2406.03154, 2024.
[26] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
[27] Kei Terayama, Masato Sumita, Ryo Tamura, and Koji Tsuda. Black-box optimization for automated discovery. Accounts of Chemical Research, 54(6):1334-1346, 2021.
[28] Meet P Vadera, Soumya Ghosh, Kenney Ng, and Benjamin M Marlin. Post-hoc loss-calibration for bayesian neural networks. In Uncertainty in Artificial Intelligence, pages 1403-1412. PMLR, 2021.
[29] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
[30] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229-256, 1992.

---

# Amortized Decision-Aware Bayesian Experimental Design - Appendix

---

#### Page 8

# Appendix

## A Conditional neural processes

CNPs [11] are designed to model complex stochastic processes through a flexible architecture that utilizes a context set and a target set. The context set consists of observed data points that the model uses to form its understanding, while the target set includes the points to be predicted. The traditional CNP architecture includes an encoder and a decoder. The encoder is a DeepSet architecture to ensure permutation invariance, it transforms each context point individually and then aggregates these transformations into a single representation that captures the overall context. The decoder then uses this representation to generate predictions for the target set, typically employing a Gaussian likelihood for approximation of the true predictive distributions. Due to the analytically tractable likelihood, CNPs can be efficiently trained through maximum likelihood estimation.

## A. 1 Transformer neural processes

Transformer Neural Processes (TNPs), introduced by [20], enhance the flexibility and expressiveness of CNPs by incorporating the Transformer's attention mechanism [29]. In TNPs, the transformer architecture uses self-attention to process the context set, dynamically weighting the importance of each point. This allows the model to create a rich representation of the context, which is then used by the decoder to generate predictions for the target set. The attention mechanism in TNPs facilitates the handling of large and variable-sized context sets, improving the model's performance on tasks with complex input-output relationships. The Transformer architecture is also useful in our setups where certain designs may have a more significant impact on the decision-making process than others. For more details about TNPs, please refer to [20].

## B Additional details of TNDP

## B. 1 Full algorithm for training TNDP

```
Algorithm 1 Transformer Neural Decision Processes (TNDP)
    Input: Utility function \(u\left(y_{\Xi}, a\right)\), prior \(p(\theta)\), likelihood \(p(y \mid \theta, \xi)\), query horizon \(T\)
    Output: Trained TNDP
    while within the training budget do
        Sample \(\theta \sim p(\theta)\) and initialize \(D\)
        for \(t=1\) to \(T\) do
            \(\xi_{t}^{(q)} \sim \pi_{t}\left(\cdot \mid h_{1: t-1}\right) \quad \triangleright\) sample next design from policy
            Sample \(y_{t} \sim p(y \mid \theta, \xi) \quad \triangleright\) observe outcome
            Set \(h_{t}=h_{t-1} \cup\left\{\left(\xi_{t}^{(q)}, y_{t}\right)\right\} \quad \triangleright\) update history
            Set \(D^{(\mathrm{c})}=h_{1: t}, D^{(\mathrm{q})}=D^{(\mathrm{q})} \backslash\left\{\xi_{t}^{(\mathrm{q})}\right\} \quad \triangleright\) update \(D\)
            Calculate \(r_{t}\left(\xi_{t}^{(q)}\right)\) with \(u\left(y_{\Xi}, a\right)\) using Eq. (4)
            end for
            \(R_{t}=\sum_{k=t}^{T} \alpha^{k-t} r_{k}\left(\xi_{k}^{(q)}\right) \quad \triangleright\) calculate cumulative reward
        Update TNDP using \(\mathcal{L}^{(p)}(\mathrm{Eq} .(3))\) and \(\mathcal{L}^{(\mathrm{q})}\) (Eq. (5))
    end while
    At deployment, we can use \(f^{(\mathrm{q})}\) to sequentially query \(T\) designs. Afterward, based on the queried
        experiments, we perform one-step final decision-making using the prediction from \(f^{(p)}\).
```

## B. 2 Embedders

The embedder $f_{\text {emb }}$ is responsible for mapping the raw data to a space of the same dimension. For the toy example and the top- $k$ hyperparameter task, we use three embedders: a design embedder $f_{\text {emb }}^{(\xi)}$, an outcome embedder $f_{\text {emb }}^{(y)}$, and a time step embedder $f_{\text {emb }}^{(t)}$. Both $f_{\text {emb }}^{(\xi)}$ and $f_{\text {emb }}^{(y)}$ are multi-layer perceptions (MLPs) with the following architecture:

---

#### Page 9

- Hidden dimension: the dimension of the hidden layers, set to 32.
- Output dimension: the dimension of the output space, set to 32 .
- Depth: the number of layers in the neural network, set to 4.
- Activation function: ReLU is used as the activation function for the hidden layers.

The time step embedder $f_{\text {emb }}^{(t)}$ is a discrete embedding layer that maps time steps to a continuous embedding space of dimension 32 .

For the decision-aware active learning task, since the design space contains both the covariates and the decision, we use four embedders: a covariate embedder $f_{\text {emb }}^{(x)}$, a decision embedder $f_{\text {emb }}^{(d)}$, an outcome embedder $f_{\text {emb }}^{(y)}$, and a time step embedder $f_{\text {emb }}^{(t)} \cdot f_{\text {emb }}^{(x)}, f_{\text {emb }}^{(y)}$ and $f_{\text {emb }}^{(t)}$ are MLPs which use the same settings as described above. The decision embedder $f_{\text {emb }}^{(d)}$ is another discrete embedding layer.
For context embedding $\boldsymbol{E}^{(c)}$, we first map each $\xi_{i}^{(c)}$ and $y_{i}^{(c)}$ to the same dimension using their respective embedders, and then sum them to obtain the final embedding. For prediction embedding $\boldsymbol{E}^{(p)}$ and query embedding $\boldsymbol{E}^{(q)}$, we only encode the designs. For $\boldsymbol{E}^{(\mathrm{d})}$, except the embeddings of the time step, we also encode the global contextual information $\lambda$ using $f_{\text {emb }}^{(x)}$ in the toy example and the decision-aware active learning task. All the embeddings are then concatenated together to form our final embedding $\boldsymbol{E}$.

# B. 3 Transformer blocks

We utilize the official TransformerEncoder layer of PyTorch [21] (https://pytorch.org) for our transformer architecture. For all experiments, we use the same configuration, which is as follows:

- Number of layers: 6
- Number of heads: 8
- Dimension of feedforward layer: 128
- Dropout rate: 0.0
- Dimension of embedding: 32

## B. 4 Output heads

The prediction head, $f_{\mathrm{p}}$ is an MLP that maps the Transformer's output embeddings of the query set to the predicted outcomes. It consists of an input layer with 32 hidden units, a ReLU activation function, and an output layer. The output layer predicts the mean and variance of a Gaussian likelihood, similar to CNPs.

For the query head $f_{\mathrm{q}}$, all candidate experimental designs are first mapped to embeddings $\boldsymbol{\lambda}^{(\mathrm{q})}$ by the Transformer, and these embeddings are then passed through $f_{\mathrm{q}}$ to obtain individual outputs. We then apply a Softmax function to these outputs to ensure a proper probability distribution. $f_{\mathrm{q}}$ is an MLP consisting of an input layer with 32 hidden units, a ReLU activation function, and an output layer.

## B. 5 Training details

For all experiments, we use the same configuration to train our model. We set the initial learning rate to $5 \mathrm{e}-4$ and employ the cosine annealing learning rate scheduler. The number of training epochs is set to 50,000 . For the REINFORCE algorithm, we select a discount factor of $\alpha=0.99$.

## C Details of toy example

## C. 1 Data generation

In our toy example, we generate data using a GP with the Squared Exponential (SE) kernel, which is defined as:

---

#### Page 10

$$
k\left(x, x^{\prime}\right)=v \exp \left(-\frac{\left(x-x^{\prime}\right)^{2}}{2 \ell^{2}}\right)
$$

where $v$ is the variance, and $\ell$ is the lengthscale. Specifically, in each training iteration, we draw a random lengthscale $\ell \sim 0.25+0.75 \times U(0,1)$ and the variance $v \sim 0.1+U(0,1)$, where $U(0,1)$ denotes a uniform random variable between 0 and 1 .

# D Details of top- $k$ hyperparameter optimization experiments

## D. 1 Data

In this task, we use HPO-B benchmark datasets [1]. The HPO-B dataset is a large-scale benchmark for HPO tasks, derived from the OpenML repository. It consists of 176 search spaces (algorithms) evaluated on 196 datasets, with a total of 6.4 million hyperparameter evaluations. This dataset is designed to facilitate reproducible and fair comparisons of HPO methods by providing explicit experimental protocols, splits, and evaluation measures.
We extracted four meta-datasets from the HPOB dataset: ranger (7609), svm (5891), rpart (5859), and xgboost (5971). For detailed information on the datasets, please refer to https://github.com/ releaunifreiburg/HPO-B.

## D. 2 Other methods description

In our experiments, we compare our method with several common acquisition functions used in HPO. We use GPs as surrogate models for these acquisition functions. All the implementations are based on BoTorch [2] (https://botorch.org/). The acquisition functions compared are as follows:

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
\alpha_{\mathrm{PI}}(\mathbf{x})=\Phi\left(\frac{\mu(\mathbf{x})-f\left(\mathbf{x}^{+}\right)-\xi}{\sigma(\mathbf{x})}\right)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution, $f\left(\mathbf{x}^{+}\right)$ is the current best value observed, and $\xi$ is a parameter that encourages exploration.

We also compared our method with an amortized method PFNs4BO [19]. It is a Transformer-based model designed for hyperparameter optimization which does not consider the downstream task. We used the pre-trained PFNs4BO-BNN model and chose PI as the acquisition function. We used the PFNs4BO's official implementation (https://github.com/automl/PFNs4BO).