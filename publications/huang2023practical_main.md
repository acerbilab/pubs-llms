```
@inproceedings{huang2023practical,
  title={Practical Equivariances via Relational Conditional Neural Processes},
  author={Huang, Daolang and Hausmann, Manuel and Remes, Ulpu and Clarté, Grégoire and Luck, Kevin Sebastian and Kaski, Samuel and Acerbi, Luigi},
  booktitle={The Thirty-seventh Annual Conference on Neural Information Processing Systems (NeurIPS 2023)},
  year={2023},
}
```

---

#### Page 1

# Practical Equivariances via Relational Conditional Neural Processes

Daolang Huang ${ }^{1}$ Manuel Haussmann ${ }^{1}$ Ulpu Remes ${ }^{2}$ ST John ${ }^{1}$<br>Grégoire Clarté ${ }^{3} \quad$ Kevin Sebastian Luck ${ }^{4,6}$ Samuel Kaski ${ }^{1,5}$ Luigi Acerbi ${ }^{3}$<br>${ }^{1}$ Department of Computer Science, Aalto University, Finland<br>${ }^{2}$ Department of Mathematics and Statistics, University of Helsinki<br>${ }^{3}$ Department of Computer Science, University of Helsinki<br>${ }^{4}$ Department of Electrical Engineering and Automation (EEA), Aalto University, Finland<br>${ }^{5}$ Department of Computer Science, University of Manchester<br>${ }^{6}$ Department of Computer Science, Vrije Universiteit Amsterdam, The Netherlands<br>\{daolang.huang, manuel.haussmann, ti.john, samuel.kaski\}@aalto.fi<br>k.s.luck@vu.nl<br>\{ulpu.remes, gregoire.clarte, luigi.acerbi\}@helsinki.fi

#### Abstract

Conditional Neural Processes (CNPs) are a class of metalearning models popular for combining the runtime efficiency of amortized inference with reliable uncertainty quantification. Many relevant machine learning tasks, such as in spatiotemporal modeling, Bayesian Optimization and continuous control, inherently contain equivariances - for example to translation - which the model can exploit for maximal performance. However, prior attempts to include equivariances in CNPs do not scale effectively beyond two input dimensions. In this work, we propose Relational Conditional Neural Processes (RCNPs), an effective approach to incorporate equivariances into any neural process model. Our proposed method extends the applicability and impact of equivariant neural processes to higher dimensions. We empirically demonstrate the competitive performance of RCNPs on a large array of tasks naturally containing equivariances.

## 1 Introduction

Conditional Neural Processes (CNPs; [10]) have emerged as a powerful family of metalearning models, offering the flexibility of deep learning along with well-calibrated uncertainty estimates and a tractable training objective. CNPs can naturally handle irregular and missing data, making them suitable for a wide range of applications. Various advancements, such as attentive (ACNP; [20]) and Gaussian (GNP; [30]) variants, have further broadened the applicability of CNPs. In principle, CNPs can be trained on other general-purpose stochastic processes, such as Gaussian Processes (GPs; [34]), and be used as an amortized, drop-in replacement for those, with minimal computational cost at runtime.

However, despite their numerous advantages, CNPs face substantial challenges when attempting to model equivariances, such as translation equivariance, which are essential for problems involving spatio-temporal components or for emulating widely used GP kernels in tasks such as Bayesian Optimization (BayesOpt; [12]). In the context of CNPs, kernel properties like stationarity and isotropy would correspond to, respectively, translational equivariance and equivariance to rigid transformations. Lacking such equivariances, CNPs struggle to scale effectively and emulate (equivariant) GPs even in moderate higher-dimensional input spaces (i.e., above two). Follow-up work has introduced Convolutional CNPs (ConvCNP; [15]), which leverage a convolutional deep sets construction to

---

#### Page 2

> **Image description.** The image consists of five separate plots arranged horizontally. The plots are labeled (a) through (e) and depict 1D regression models. The first three plots, (a), (b), and (c), are grouped under the label "INT", while the last two, (d) and (e), are grouped under the label "OOID".
>
> Each plot has an x-axis and a y-axis. The y-axis ranges from -2 to 2 in plots (a), (b), and (c). The x-axis ranges from -2 to 2 in plots (a), (b), and (c), and from 2 to 6 in plots (d) and (e). The axes are labeled 'x' and 'y'.
>
> Plot (a) is labeled "CNP" and shows an orange line representing the model's prediction, surrounded by a shaded orange region representing the uncertainty. Black dots represent the context data.
>
> Plot (b) is labeled "RCNP" and shows a blue line representing the model's prediction, surrounded by a shaded blue region representing the uncertainty. Black dots represent the context data.
>
> Plot (c) is labeled "GP" and shows a red line representing the model's prediction, surrounded by a shaded red region representing the uncertainty. Black dots represent the context data.
>
> Plot (d) is labeled "CNP" and shows an orange line representing the model's prediction, surrounded by a shaded orange region representing the uncertainty. Black dots represent the context data.
>
> Plot (e) is labeled "RCNP" and shows a blue line representing the model's prediction, surrounded by a shaded blue region representing the uncertainty. Black dots represent the context data.

Figure 1: Equivariance in 1D regression. Left: Predictions for a CNP (a) and RCNP (b) in an interpolation (INT) task, trained for 20 epochs to emulate a GP (c) with Matérn- $\frac{5}{2}$ kernel and noiseless observations. The CNP underfits the context data (black dots), while the RCNP leverages translation equivariance to learn faster and yield better predictions. Right: The CNP (d) fails to predict in an out-of-input-distribution (OOID) task, where the input context is outside the training range (note the shifted $x$ axis); whereas the RCNP (e) generalizes by means of translational equivariance.

induce translational-equivariant embeddings. However, the requirement of defining an input grid and performing convolutions severely limits the applicability of ConvCNPs and variants thereof (ConvGNP [30]; FullConvGNP [2]) to one- or two-dimensional equivariant inputs; both because higher-dimensional implementations of convolutions are poorly supported by most deep learning libraries, and for the prohibitive cost of performing convolutions in three or more dimensions. Thus, the problem of efficiently scaling equivariances in CNPs above two input dimensions remains open.
In this paper, we introduce Relational Conditional Neural Processes (RCNPs), a novel approach that offers a simple yet powerful technique for including a large class of equivariances into any neural process model. By leveraging the existing equivariances of a problem, RCNPs can achieve improved sample efficiency, predictive performance, and generalization (see Figure 1). The basic idea in RCNPs is to enforce equivariances via a relational encoding that only stores appropriately chosen relative information of the data. By stripping away absolute information, equivariance is automatically satisfied. Surpassing the complex approach of previous methods (e.g., the ConvCNP family for translational equivariance), RCNPs provide a practical solution that scales to higher dimensions, while maintaining strong performance and extending to other equivariances. The cost to pay is increased computational complexity in terms of context size (size of the dataset we are conditioning on at runtime); though often not a bottleneck for the typical metalearning small-context setting of CNPs. Our proposed method works for equivariances that can be expressed relationally via comparison between pairs of points (e.g., their difference or distance); in this paper, we focus on translational equivariance and equivariance to rigid transformations.

Contributions. In summary, our contributions in this work are:

- We introduce a simple and effective way - relational encoding - to encode exact equivariances directly into CNPs, in a way that easily scales to higher input dimensions.
- We propose two variants of relational encoding: one that works more generally ('Full'); and one which is simpler and more computationally efficient ('Simple'), and is best suited for implementing translation equivariance.
- We provide theoretical foundations and proofs to support our approach.
- We empirically demonstrate the competitive performance of RCNPs on a variety of tasks that naturally contain different equivariances, highlighting their practicality and effectiveness.

Outline of the paper. The remainder of this paper is organized as follows. In Section 2, we review the foundational work of CNPs and their variants. This is followed by the introduction of our proposed relational encoding approach to equivariances at the basis of our RCNP models (Section 3). We then provide in Section 4 theoretical proof that relational encoding achieves equivariance without losing essential information; followed in Section 5 by a thorough empirical validation of our claims in various tasks requiring equivariances, demonstrating the generalization capabilities and predictive performance of RCNPs. We discuss other related work in Section 6, and the limitations of our approach, including its computational complexity, in Section 7. We conclude in Section 8 with an overview of the current work and future directions.
Our code is available at https://github.com/acerbilab/relational-neural-processes.

---

#### Page 3

# 2 Background: the Conditional Neural Process family

In this section, we review the Conditional Neural Process (CNP) family of stochastic processes and the key concept of equivariance at the basis of this work. Following [30], we present these notions within the framework of prediction maps [8]. We denote with $\mathbf{x} \in \mathcal{X} \subseteq \mathbb{R}^{d_{x}}$ input vectors and $\mathbf{y} \in \mathcal{Y} \subseteq \mathbb{R}^{d_{y}}$ output vectors, with $d_{x}, d_{y} \geq 1$ their dimensionality. If $f(\mathbf{z})$ is a function that takes as input elements of a set $\mathbf{Z}$, we denote with $f(\mathbf{Z})$ the set $\{f(\mathbf{z})\}_{\mathbf{z} \in \mathbf{Z}}$.

Prediction maps. A prediction map $\pi$ is a function that maps (1) a context set $(\mathbf{X}, \mathbf{Y})$ comprising input/output pairs $\left\{\left(\mathbf{x}_{1}, \mathbf{y}_{1}\right), \ldots,\left(\mathbf{x}_{N}, \mathbf{y}_{N}\right)\right\}$ and (2) a collection of target inputs $\mathbf{X}^{\star}=\left(\mathbf{x}_{1}^{\star}, \ldots, \mathbf{x}_{M}^{\star}\right)$ to a distribution over the corresponding target outputs $\mathbf{Y}^{\star}=\left(\mathbf{y}_{1}^{\star}, \ldots, \mathbf{y}_{M}^{\star}\right)$ :

$$
\pi\left(\mathbf{Y}^{\star} \mid(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=p\left(\mathbf{Y}^{\star} \mid \mathbf{r}\right)
$$

where $\mathbf{r}=r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)$ is the representation vector that parameterizes the distribution over $\mathbf{Y}^{\star}$ via the representation function $r$. Bayesian posteriors are prediction maps, a well-known example being the Gaussian Process (GP) posterior:

$$
\pi\left(\mathbf{Y}^{\star} \mid(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=\mathcal{N}\left(\mathbf{Y}^{\star} \mid \mathbf{m}, \mathbf{K}\right)
$$

where the prediction map takes the form of a multivariate normal with representation vector $\mathbf{r}=$ $(\mathbf{m}, \mathbf{K})$. The mean $\mathbf{m}=m_{\text {post }}\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)$ and covariance matrix $\mathbf{K}=k_{\text {post }}\left(\mathbf{X}, \mathbf{X}^{\star}\right)$ of the multivariate normal are determined by the conventional GP posterior predictive expressions [34].

Equivariance. A prediction map $\pi$ with representation function $r$ is $\mathcal{T}$-equivariant with respect to a group $\mathcal{T}$ of transformations ${ }^{1}$ of the input space, $\tau: \mathcal{X} \rightarrow \mathcal{X}$, if and only if for all $\tau \in \mathcal{T}$ :

$$
r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=r\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{\star}\right)
$$

where $\tau \mathbf{x} \equiv \tau(\mathbf{x})$ and $\tau \mathbf{X}$ is the set obtained by applying $\tau$ to all elements of $\mathbf{X}$. Eq. 3 defines equivariance of a prediction map based on its representation function, and can be shown to be equivalent to the common definition of an equivariant map; see Appendix A. Intuitively, equivariance means that if the data (the context inputs) are transformed in a certain way, the predictions (the target inputs) transform correspondingly. Common groups of transformations include translations, rotations, reflections - all examples of rigid transformations. In kernel methods and specifically in GPs, equivariances are incorporated in the prior kernel function $k\left(\mathbf{x}, \mathbf{x}^{\star}\right)$. For example, translational equivariance corresponds to stationarity $k_{\text {sta }}=k\left(\mathbf{x}-\mathbf{x}^{\star}\right)$, and equivariance to all rigid transformations corresponds to isotropy, $k_{\text {iso }}=k\left(\left\|\mathbf{x}-\mathbf{x}^{\star}\right\|_{2}\right)$, where $\|\cdot\|_{2}$ denotes the Euclidean norm of a vector. A crucial question we address in this work is how to implement equivariances in other prediction maps, and specifically in the CNP family.

Conditional Neural Processes. A CNP [10] uses an encoder ${ }^{2} f_{c}$ to produce an embedding of the context set, $\mathbf{e}=f_{c}(\mathbf{X}, \mathbf{Y})$. The encoder uses a DeepSet architecture [48] to ensure invariance with respect to permutation of the order of data points, a key property of stochastic processes. We denote with $\mathbf{r}_{m}=\left(\mathbf{e}, \mathbf{x}_{m}^{\star}\right)$ the local representation of the $m$-th point of the target set $\mathbf{X}^{\star}$, for $1 \leq m \leq M$. CNPs yield a prediction map with representation $\mathbf{r}=\left(\mathbf{r}_{1}, \ldots, \mathbf{r}_{M}\right)$ :

$$
\pi\left(\mathbf{Y}^{\star} \mid(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=p\left(\mathbf{Y}^{\star} \mid \mathbf{r}\right)=\prod_{m=1}^{M} q\left(\mathbf{y}_{m}^{\star} \mid \lambda\left(\mathbf{r}_{m}\right)\right)
$$

where $q(\cdot \mid \boldsymbol{\lambda})$ belongs to a family of distributions parameterized by $\boldsymbol{\lambda}$, and $\boldsymbol{\lambda}=f_{\beta}\left(\mathbf{r}_{m}\right)$ is decoded in parallel for each $\mathbf{r}_{m}$. In the standard CNP, the decoder network $f_{\beta}$ is a multi-layer perceptron. A common choice for CNPs is a Gaussian likelihood, $q\left(\mathbf{y}_{m}^{\star} \mid \boldsymbol{\lambda}\right)=\mathcal{N}\left(\mathbf{y}_{m}^{\star} \mid \mu\left(\mathbf{r}_{m}\right), \Sigma\left(\mathbf{r}_{m}\right)\right)$, where $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ represent the predictive mean and covariance of each output, independently for each target (a mean field approach). Given the closed-form likelihood, CNPs are easily trainable via maximumlikelihood optimization of parameters of encoder and decoder networks, by sampling batches of context and target sets from the training data.

[^0]
[^0]: ${ }^{1}$ A group of transformations is a family of composable, invertible functions with identity $\tau_{\mathrm{st}} \mathbf{x}=\mathbf{x}$.
${ }^{2}$ We use purple to highlight parametrized functions (neural networks) whose parameters will be learned.

---

#### Page 4

Gaussian Neural Processes. Notably, standard CNPs do not model dependencies between distinct target outputs $\mathbf{y}_{m}^{\star}$ and $\mathbf{y}_{m^{\prime}}^{\star}$, for $m \neq m^{\prime}$. Gaussian Neural Processes (GNPs [30]) are a variant of CNPs that remedy this limitation, by assuming a joint multivariate normal structure over the outputs for the target set, $\pi\left(\mathbf{Y}^{\star} \mid(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=\mathcal{N}\left(\mathbf{Y} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}\right)$. For ease of presentation, we consider now scalar outputs ( $d_{y}=1$ ), but the model generalizes to the multi-output case. GNPs parameterize the mean as $\mu_{m}=f_{\mu}\left(\mathbf{r}_{m}\right)$ and covariance matrix $\Sigma_{m, m^{\prime}}=k\left(f_{\Sigma}\left(\mathbf{r}_{m}\right), f_{\Sigma}\left(\mathbf{r}_{m^{\prime}}\right)\right) f_{c}\left(\mathbf{r}_{m}\right) f_{c}\left(\mathbf{r}_{m^{\prime}}\right)$, for target points $\mathbf{x}_{m}^{\star}, \mathbf{x}_{m^{\prime}}^{\star}$, where $f_{\mu}, f_{\Sigma}$, and $f_{c}$ are neural networks with outputs, respectively, in $\mathbb{R}, \mathbb{R}^{d_{\Sigma}}$, and $\mathbb{R}^{+}, k(\cdot, \cdot)$ is a positive-definite kernel function, and $d_{\Sigma} \in \mathbb{N}^{+}$denotes the dimensionality of the space in which the covariance kernel is evaluated. Standard GNP models use the linear covariance (where $f_{\nu}=1$ and $k$ is the linear kernel) or the kvv covariance (where $k$ is the exponentiated quadratic kernel with unit lengthscale), as described in [30].

Convolutional Conditional Neural Processes. The Convolutional CNP family includes the ConvCNP [15], ConvGNP [30], and FullConvGNP [2]. These CNP models are built to implement translational equivariance via a ConvDeepSet architecture [15]. For example, the ConvCNP is a prediction map $p\left(\mathbf{Y}^{\star} \mid(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=\prod_{m=1}^{M} q\left(\mathbf{y}_{m}^{\star} \mid \Phi_{\mathbf{X}, \mathbf{Y}}\left(\mathbf{x}_{m}^{\star}\right)\right)$, where $\Phi_{\mathbf{X}, \mathbf{Y}}(\cdot)$ is a ConvDeepSet. The construction of ConvDeepSets involves, among other steps, gridding of the data if not already on the grid and application of $d_{x}$-dimensional convolutional neural networks ( $2 d_{x}$ for FullConvGNP). Due to the limited scaling and availability of convolutional operators above two dimensions, ConvCNPs do not scale in practice for $d_{x}>2$ translationally-equivariant input dimensions.

Other Neural Processes. The neural process family includes several other members, such as latent NPs (LNP; [11]) which model dependencies in the predictions via a latent variable - however, LNPs lack a tractable training objective, which impairs their practical performance. Attentive (C)NPs (A(C)NPs; [20]) implement an attention mechanism instead of the simpler DeepSet architecture. Transformer NPs [31] combine a transformer-based architecture with a causal mask to construct an autoregressive likelihood. Finally, Autoregressive CNPs (AR-CNPs [3]) provide a novel technique to deploy existing CNP models via autoregressive sampling without architectural changes.

# 3 Relational Conditional Neural Processes

We introduce now our Relational Conditional Neural Processes (RCNPs), an effective solution for embedding equivariances into any CNP model. Through relational encoding, we encode selected relative information and discard absolute information, inducing the desired equivariance.

Relational encoding. In RCNPs, the (full) relational encoding of a target point $\mathbf{x}_{m}^{\star} \in \mathbf{X}^{\star}$ with respect to the context set $(\mathbf{X}, \mathbf{Y})$ is defined as:

$$
\rho_{\text {full }}\left(\mathbf{x}_{m}^{\star},(\mathbf{X}, \mathbf{Y})\right)=\bigoplus_{n, n^{\prime}=1}^{N} f_{r}\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right), \mathbf{R}_{n n^{\prime}}\right), \quad \mathbf{R}_{n n^{\prime}} \equiv\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{n^{\prime}}\right), \mathbf{y}_{n}, \mathbf{y}_{n^{\prime}}\right)
$$

where $g: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^{d_{\text {comp }}}$ is a chosen comparison function ${ }^{3}$ that specifies how a pair $\mathbf{x}, \mathbf{x}^{\prime}$ should be compared; $\mathbf{R}$ is the relational matrix, comparing all pairs of the context set; $f_{r}: \mathbb{R}^{d_{\text {comp }}} \times \mathbb{R}^{d_{\text {comp }}+2 d_{z}} \rightarrow \mathbb{R}^{d_{\text {set }}}$ is the relational encoder, a neural network that maps a comparison vector and element of the relational matrix into a high-dimensional space $\mathbb{R}^{d_{\text {set }}}$; $\bigoplus$ is a commutative aggregation operation (sum in this work) ensuring permutation invariance of the context set [48]. From Eq. 5, a point $\mathbf{x}^{\star}$ is encoded based on how it compares to the entire context set.
Intuitively, the comparison function $g(\cdot, \cdot)$ should be chosen to remove all information that does not matter to impose the desired equivariance. For example, if we want to encode translational equivariance, the comparison function should be the difference of the inputs, $g_{\text {diff }}\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right)=\mathbf{x}_{m}^{\star}-\mathbf{x}_{n}$ (with $d_{\text {comp }}=d_{\mathrm{x}}$ ). Similarly, isotropy (invariance to rigid transformations, i.e. rotations, translations, and reflections) can be encoded via the Euclidean distance $g_{\text {dist }}\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right)=\left\|\mathbf{x}_{m}^{\star}-\mathbf{x}_{n}\right\|_{2}$ (with $d_{\text {comp }}=1$ ). We will prove these statements formally in Section 4.

[^0]
[^0]: ${ }^{3}$ We use green to highlight the selected comparison function that encodes a specific set of equivariances.

---

#### Page 5

Full RCNP. The full-context RCNP, or FullRCNP, is a prediction map with representation $\mathbf{r}=\left(\boldsymbol{\rho}_{1}, \ldots, \boldsymbol{\rho}_{M}\right)$, with $\boldsymbol{\rho}_{m}=\rho_{\text {full }}\left(\mathbf{x}_{m}^{\star},(\mathbf{X}, \mathbf{Y})\right)$ the relational encoding defined in Eq. 5:

$$
\pi\left(\mathbf{Y}^{\star} \mid(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=p\left(\mathbf{Y}^{\star} \mid \mathbf{r}\right)=\prod_{m=1}^{M} q\left(\mathbf{y}_{m}^{\star} \mid \lambda\left(\boldsymbol{\rho}_{m}\right)\right)
$$

where $q(\cdot \mid \boldsymbol{\lambda})$ belongs to a family of distributions parameterized by $\boldsymbol{\lambda}$, where $\boldsymbol{\lambda}=f_{d}\left(\boldsymbol{\rho}_{m}\right)$ is decoded from the relational encoding $\boldsymbol{\rho}_{m}$ of the $m$-th target. As usual, we often choose a Gaussian likelihood, whose mean and covariance (variance, for scalar outputs) are produced by the decoder network.

Note how Eq. 6 (FullRCNP) is nearly identical to Eq. 4 (CNP), the difference being that we replaced the representation $\mathbf{r}_{m}=\left(\mathbf{e}, \mathbf{x}_{m}^{\star}\right)$ with the relational encoding $\boldsymbol{\rho}_{m}$ from Eq. 5 . Unlike CNPs, in RCNPs there is no separate encoding of the context set alone. The RCNP construction generalizes easily to other members of the CNP family by plug-in replacement of $\mathbf{r}_{m}$ with $\boldsymbol{\rho}_{m}$. For example, a relational GNP (RGNP) describes a multivariate normal prediction map whose mean is parameterized as $\mu_{m}=$ $f_{\mu}\left(\boldsymbol{\rho}_{m}\right)$ and whose covariance matrix is given by $\Sigma_{m, m^{\prime}}=k\left(f_{\Sigma}\left(\boldsymbol{\rho}_{m}\right), f_{\Sigma}\left(\boldsymbol{\rho}_{m^{\prime}}\right)\right) f_{c}\left(\boldsymbol{\rho}_{m}\right) f_{c}\left(\boldsymbol{\rho}_{m^{\prime}}\right)$.

Simple RCNP. The full relational encoding in Eq. 5 is cumbersome as it asks to build and aggregate over a full relational matrix. Instead, we can consider the simple or 'diagonal' relational encoding:

$$
\rho_{\text {diag }}\left(\mathbf{x}_{m}^{\star},(\mathbf{X}, \mathbf{Y})\right)=\bigoplus_{n=1}^{N} f_{r}\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right), g\left(\mathbf{x}_{n}, \mathbf{x}_{n}\right), \mathbf{y}_{n}\right)
$$

Eq. 7 is functionally equivalent to Eq. 5 restricted to the diagonal $n=n^{\prime}$, and further simplifies in the common case $g\left(\mathbf{x}_{n}, \mathbf{x}_{n}\right)=\mathbf{0}$, whereby the argument of the aggregation becomes $f_{r}\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right), \mathbf{y}_{n}\right)$.
We obtain the simple RCNP model (from now on, just RCNP) by using the diagonal relational encoding $\rho_{\text {diag }}$ instead of the full one, $\rho_{\text {full }}$. Otherwise, the simple RCNP model follows Eq. 6. We will prove, both in theory and empirically, that the simple RCNP is best for encoding translational equivariance. Like the FullRCNP, the RCNP easily extends to other members of the CNP family.
In this paper, we consider the FullRCNP, FullRGNP, RCNP and RGNP models for translations and rigid transformations, leaving examination of other RCNP variants and equivariances to future work.

## 4 RCNPs are equivariant and context-preserving prediction maps

In this section, we demonstrate that RCNPs are $\mathcal{T}$-equivariant prediction maps, where $\mathcal{T}$ is a transformation group of interest (e.g., translations), for an appropriately chosen comparison function $g: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^{d_{\text {comp }}}$. Then, we formalize the statement that RCNPs strip away only enough information to achieve equivariance, but no more. We prove this by showing that the RCNP representation preserves information in the context set. Full proofs are given in Appendix A.

### 4.1 RCNPs are equivariant

Definition 4.1. Let $g$ be a comparison function and $\mathcal{T}$ a group of transformations $\tau: \mathcal{X} \rightarrow \mathcal{X}$. We say that $g$ is $\mathcal{T}$-invariant if and only if $g\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=g\left(\tau \mathbf{x}, \tau \mathbf{x}^{\prime}\right)$ for any $\mathbf{x}, \mathbf{x}^{\prime} \in \mathcal{X}$ and $\tau \in \mathcal{T}$.
Definition 4.2. Given a comparison function $g$, we define the comparison sets:

$$
\begin{aligned}
g((\mathbf{X}, \mathbf{Y}),(\mathbf{X}, \mathbf{Y})) & =\left\{\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{n^{\prime}}\right), \mathbf{y}_{n}, \mathbf{y}_{n^{\prime}}\right)\right\}_{1 \leq n, n^{\prime} \leq N} \\
g((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}) & =\left\{\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right), \mathbf{y}_{n}\right)\right\}_{1 \leq n \leq N, 1 \leq m \leq M} \\
g\left(\mathbf{X}^{\star}, \mathbf{X}^{\star}\right) & =\left\{g\left(\mathbf{x}_{m}^{\star}, \mathbf{x}_{m^{\prime}}^{\star}\right)\right\}_{1 \leq m, m^{\prime} \leq M}
\end{aligned}
$$

If $g$ is not symmetric, we can also denote $g\left(\mathbf{X}^{\star},(\mathbf{X}, \mathbf{Y})\right)=\left\{\left(g\left(\mathbf{x}_{m}^{\star}, \mathbf{x}_{n}\right), \mathbf{y}_{n}\right)\right\}_{1 \leq n \leq N, 1 \leq m \leq M}$.
Definition 4.3. A prediction map $\pi$ and its representation function $r$ are relational with respect to a comparison function $g$ if and only if $r$ can be written solely through set comparisons:

$$
r((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star})=r\left(g((\mathbf{X}, \mathbf{Y}),(\mathbf{X}, \mathbf{Y})), g\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right), g\left(\mathbf{X}^{\star},(\mathbf{X}, \mathbf{Y})\right), g\left(\mathbf{X}^{\star}, \mathbf{X}^{\star}\right)\right)
$$

Lemma 4.4. Let $\pi$ be a prediction map, $\mathcal{T}$ a transformation group, and $g$ a comparison function. If $\pi$ is relational with respect to $g$ and $g$ is $\mathcal{T}$-invariant, then $\pi$ is $\mathcal{T}$-equivariant.

---

#### Page 6

From Lemma 4.4 and previous definitions, we derive the main result about equivariance of RCNPs.
Proposition 4.5. Let $g$ be the comparison function used in a RCNP, and $\mathcal{T}$ a group of transformations. If $g$ is $\mathcal{T}$-invariant, the RCNP is $\mathcal{T}$-equivariant.

As useful examples, the difference comparison function $g_{\text {diff }}\left(\mathbf{x}, \mathbf{x}^{*}\right)=\mathbf{x}^{*}-\mathbf{x}$ is invariant to translations of the inputs, and the distance comparison function $g_{\text {dist }}\left(\mathbf{x}, \mathbf{x}^{*}\right)=\left\|\mathbf{x}^{*}-\mathbf{x}\right\|_{2}$ is invariant to rigid transformations; thus yielding appropriately equivariant RCNPs.

### 4.2 RCNPs are context-preserving

The previous section demonstrates that any RCNP is $\mathcal{T}$-equivariant, for an appropriate choice of $g$. However, a trivial comparison function $g\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\mathbf{0}$ would also satisfy the requirements, yielding a trivial representation. We need to guarantee that, at least in principle, the encoding procedure removes only information required to induce $\mathcal{T}$-equivariance, and no more. A minimal request is that the context set is preserved in the prediction map representation $\mathbf{r}$, modulo equivariances.
Definition 4.6. A comparison function $g$ is context-preserving with respect to a transformation group $\mathcal{T}$ if for any context set $(\mathbf{X}, \mathbf{Y})$ and target set $\mathbf{X}^{*}$, there is a submatrix $\mathbf{Q}^{\prime} \subseteq \mathbf{Q}$ of the matrix $\mathbf{Q}_{m+n^{\prime}}=\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{*}\right), g\left(\mathbf{x}_{n}, \mathbf{x}_{n^{\prime}}\right), \mathbf{y}_{n}, \mathbf{y}_{n^{\prime}}\right)$, a reconstruction function $\gamma$, and a transformation $\tau \in \mathcal{T}$ such that $\gamma\left(\mathbf{Q}^{\prime}\right)=(\tau \mathbf{X}, \mathbf{Y})$.

For example, $g_{\text {dist }}$ is context-preserving with respect to the group of rigid transformations. For any $m, \mathbf{Q}^{\prime}=\mathbf{Q}_{m}$ : is the set of pairwise distances between points, indexed by their output values. Reconstructing the positions of a set of points given their pairwise distances is known as the Euclidean distance geometry problem [38], which can be solved uniquely up to rigid transformations with traditional multidimensional scaling techniques [39]. Similarly, $g_{\text {diff }}$ is context-preserving with respect to translations. For any $m$ and $n^{\prime}, \mathbf{Q}^{\prime}=\mathbf{Q}_{m: n^{\prime}}$ can be projected to the vector $\left(\begin{array}{ll}\mathbf{x}_{1}- \\ \mathbf{x}_{m}^{*}, \mathbf{y}_{1}\end{array}\right), \ldots,\left(\mathbf{x}_{N}-\mathbf{x}_{m}^{*}, \mathbf{y}_{N}\right)$ ), which is equal to $\left(\tau_{m} \mathbf{X}, \mathbf{Y}\right)$ with the translation $\tau_{m}(\cdot)=\cdot-\mathbf{x}_{m}^{*}$.
Definition 4.7. For any $(\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}$, a family of functions $h_{\boldsymbol{\theta}}\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right) \rightarrow \mathbf{r} \in \mathbb{R}^{d_{\text {rep }}}$ is contextpreserving under a transformation group $\mathcal{T}$ if there exists $\boldsymbol{\theta} \in \boldsymbol{\Theta}, d_{\text {rep }} \in \mathbb{N}$, a reconstruction function $\gamma$, and a transformation $\tau \in \mathcal{T}$ such that $\gamma\left(h_{\boldsymbol{\theta}}\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)\right) \equiv \gamma(\mathbf{r})=(\tau \mathbf{X}, \mathbf{Y})$.

Thus, an encoding is context-preserving if it is possible at least in principle to fully recover the context set from $\mathbf{r}$, implying that no relevant context is lost. This is indeed the case for RCNPs.
Proposition 4.8. Let $\mathcal{T}$ be a transformation group and $g$ the comparison function used in a FullRCNP. If $g$ is context-preserving with respect to $\mathcal{T}$, then the representation function $r$ of the FullRCNP is context-preserving with respect to $\mathcal{T}$.
Proposition 4.9. Let $\mathcal{T}$ be the translation group and $g_{\text {diff }}$ the difference comparison function. The representation of the simple RCNP model with $g_{\text {diff }}$ is context-preserving with respect to $\mathcal{T}$.

Given the convenience of the RCNP compared to FullRCNPs, Proposition 4.9 shows that we can use simple RCNPs to incorporate translation-equivariance with no loss of information. However, the simple RCNP model is not context-preserving for other equivariances, for which we ought to use the FullRCNP. Our theoretical results are confirmed by the empirical validation in the next section.

## 5 Experiments

In this section, we evaluate the proposed relational models on several tasks and compare their performance with other conditional neural process models. For this, we used publicly available reference implementations of the neuralprocesses software package [1, 3]. We detail our experimental approach in Appendix B, and we empirically analyze computational costs in Appendix C.

### 5.1 Synthetic Gaussian and non-Gaussian functions

We first provide a thorough comparison of our methods with other CNP models using a diverse array of Gaussian and non-Gaussian synthetic regression tasks. We consider tasks characterized by functions derived from (i) a range of GPs, where each GP is sampled using one of three different kernels (Exponentiated Quadratic (EQ), Matérn- $\frac{5}{2}$, and Weakly-Periodic); (ii) a non-Gaussian sawtooth

---

#### Page 7

process; (iii) a non-Gaussian mixture task. In the mixture task, the function is randomly selected from either one of the three aforementioned distinct GPs or the sawtooth process, each chosen with probability $\frac{1}{4}$. Apart from evaluating simple cases with $d_{x}=\{1,2\}$, we also expand our experiments to higher dimensions, $d_{x}=\{3,5,10\}$. In these higher-dimensional scenarios, applying ConvCNP and ConvGNP models is not considered feasible. We assess the performance of the models in two distinct ways. The first one, interpolation (INT), uses the data generated from a range identical to that employed during the training phase. The second one, out-of-input-distribution (OOID), uses data generated from a range that extends beyond the scope of the training data.

Results. We first compare our translation-equivariant ('stationary') versions of RCNP and RGNP with other baseline models from the CNP family (Table 1). Comprehensive results, including all five regression problems and five dimensions, are available in Appendix D. Firstly, relational encoding of the translational equivariance intrinsic to the task improves performance, as both RCNP and RGNP models surpass their CNP and GNP counterparts in terms of INT results. Furthermore, the OOID results demonstrate significant improvement of our models, as they can leverage translationalequivariance to generalize outside the training range. RCNPs and RGNPs are competitive with convolutional models (ConvCNP, ConvGNP) when applied to 1D data and continue performing well in higher dimension, whereas models in the ConvCNP family are inapplicable for $d_{x}>2$.

Table 1: Comparison of interpolation (INT) and out-of-input-distribution (OOID) performance of our RCNP models and CNP baselines on synthetic regression tasks with varying input dimensions. We show mean and (standard deviation) across 10 runs with different seeds. "F" denotes failed attempts that yielded very bad results. Missing entries could not be run. Statistically significantly best results are bolded. Our methods (RCNP and RGNP) perform better than their CNP and GNP counterparts in terms of both INT and OOID, and scale to higher dimension compared to ConvCNP and ConvGNP.

|                                                     |            | Weakly-periodic KL divergence( $\downarrow$ ) |             |             | Sawtooth log-likelihood( $\dagger$ ) |               |               | Mixture log-likelihood( $\dagger$ ) |               |               |
| :-------------------------------------------------: | :--------: | :-------------------------------------------: | :---------: | :---------: | :----------------------------------: | :-----------: | :-----------: | :---------------------------------: | :-----------: | :-----------: |
|                                                     |            |                   $d_{x}=1$                   |  $d_{x}=3$  |  $d_{x}=5$  |              $d_{x}=1$               |   $d_{x}=3$   |   $d_{x}=5$   |              $d_{x}=1$              |   $d_{x}=3$   |   $d_{x}=5$   |
| $\underset{\sim}{\stackrel{\rightharpoonup}{\sim}}$ | RCNP (sta) |                  0.24 (0.00)                  | 0.28 (0.00) | 0.31 (0.00) |             3.03 (0.06)              |  0.85 (0.01)  |  0.44 (0.00)  |             0.20 (0.01)             | $-0.10(0.00)$ | $-0.31(0.03)$ |
|                                                     | RGNP (sta) |                  0.03 (0.00)                  | 0.05 (0.00) | 0.08 (0.00) |             3.90 (0.09)              |  1.09 (0.01)  |  1.13 (0.05)  |             0.34 (0.03)             |  0.37 (0.01)  |  0.04 (0.02)  |
|                                                     |  ConvCNP   |                  0.21 (0.00)                  |      -      |      -      |             3.64 (0.04)              |       -       |       -       |             0.38 (0.02)             |       -       |       -       |
|                                                     |  ConvGNP   |                  0.01 (0.00)                  |      -      |      -      |             3.94 (0.11)              |       -       |       -       |             0.49 (0.15)             |       -       |       -       |
|                                                     |    CNP     |                  0.31 (0.00)                  | 0.39 (0.00) | 0.42 (0.00) |             2.25 (0.02)              |  0.36 (0.28)  | $-0.03(0.10)$ |             0.01 (0.01)             | $-0.57(0.11)$ | $-0.72(0.08)$ |
|                                                     |    GNP     |                  0.06 (0.00)                  | 0.08 (0.01) | 0.11 (0.01) |             0.83 (0.04)              |  0.23 (0.13)  |  0.02 (0.05)  |             0.17 (0.01)             | $-0.17(0.00)$ | $-0.32(0.00)$ |
| $\underset{\sim}{\stackrel{\rightharpoonup}{\sim}}$ | RCNP (sta) |                  0.24 (0.00)                  | 0.28 (0.01) | 0.31 (0.00) |             3.04 (0.06)              |  0.85 (0.01)  |  0.44 (0.00)  |             0.20 (0.01)             | $-0.10(0.00)$ | $-0.31(0.03)$ |
|                                                     | RGNP (sta) |                  0.03 (0.00)                  | 0.05 (0.01) | 0.08 (0.00) |             3.90 (0.10)              |  1.09 (0.01)  |  1.13 (0.05)  |             0.34 (0.03)             |  0.37 (0.01)  |  0.04 (0.02)  |
|                                                     |  ConvCNP   |                  0.21 (0.00)                  |      -      |      -      |             3.64 (0.04)              |       -       |       -       |             0.38 (0.02)             |       -       |       -       |
|                                                     |  ConvGNP   |                  0.01 (0.00)                  |      -      |      -      |             3.97 (0.08)              |       -       |       -       |             0.49 (0.15)             |       -       |       -       |
|                                                     |    CNP     |                  2.88 (0.91)                  | 1.58 (0.50) | 2.20 (0.81) |                  F                   | $-0.37(0.12)$ | $-0.22(0.03)$ |                  F                  | $-2.55(1.15)$ | $-1.71(0.55)$ |
|                                                     |    GNP     |                       F                       | 1.47 (0.27) | 0.62 (0.04) |                  F                   |       F       |       F       |                  F                  | $-0.67(0.05)$ | $-0.72(0.03)$ |

We further consider two GP tasks with isotropic EQ and Matérn- $\frac{5}{2}$ kernels (invariant to rigid transformations). Within this set of experiments, we include the FullRCNP and FullRGNP models, each equipped with the 'isotropic' distance comparison function. The results (Table 2) indicate that RCNPs and FullRCNPs consistently outperform CNPs across both tasks. Additionally, we notice that FullRCNPs exhibit better performance compared to RCNPs as the dimension increases. When $d_{x}=2$, the performance of our RGNPs is on par with that of ConvGNPs, and achieves the best results in terms of both INT and OOID when $d_{x}>2$, which again highlights the effectiveness of our models in handling high-dimensional tasks by leveraging existing equivariances.

### 5.2 Bayesian optimization

We explore the extent our proposed models can be used for a higher-dimensional meta-learning task, using Bayesian optimization (BayesOpt) as our application [12]. The neural processes, ours as well as the baselines, serve as surrogates to find the global minimum $f_{\min }=f\left(\mathbf{x}_{\min }\right)$ of a black-box function. For this task, we train the models by generating random functions from a GP kernel sampled from a set of base kernels-EQ, Matérn- $\left\{\frac{1}{2}, \frac{3}{2}, \frac{5}{2}\right\}$ as well as their sums and products-with randomly sampled hyperparameters. By training on a large distribution over kernels, we aim to exploit the metalearning capabilities of neural processes. The trained CNP models are then used as surrogates to minimize the Hartmann function [40, p.185] in three and six dimensions, a common BayesOpt test function. We use the expected improvement acquisition function, which we can evaluate analytically. Specifics on the experimental setup and further evaluations can be found in Appendix E.

---

#### Page 8

Table 2: Comparison of the interpolation (INT) and out-of-input-distribution (OOID) performance of our RCNP models with different CNP baselines on two GP synthetic regression tasks with isotropic kernels of varying input dimensions.

|                                    |               |              EQ               |             |             |             |     Matīrn- $\frac{1}{2}$     |             |     |
| :--------------------------------: | :-----------: | :---------------------------: | :---------: | :---------: | :---------: | :---------------------------: | :---------: | :-: |
|                                    |               | KL divergence( $\downarrow$ ) |             |             |             | KL divergence( $\downarrow$ ) |             |     |
|                                    |               |           $d_{n}=2$           |  $d_{s}=3$  |  $d_{a}=5$  |  $d_{n}=2$  |           $d_{n}=3$           |  $d_{a}=5$  |     |
|                                    |  RCNP (sta)   |          0.26 (0.00)          | 0.40 (0.01) | 0.45 (0.00) | 0.30 (0.00) |          0.39 (0.00)          | 0.35 (0.00) |     |
|                                    |  RGNP (sta)   |          0.03 (0.00)          | 0.05 (0.00) | 0.11 (0.00) | 0.03 (0.00) |          0.05 (0.00)          | 0.11 (0.00) |     |
| $\stackrel{\rightharpoonup}{\sim}$ | FuDRCNP (iso) |          0.26 (0.00)          | 0.31 (0.00) | 0.35 (0.00) | 0.30 (0.00) |          0.32 (0.00)          | 0.29 (0.00) |     |
|                                    | FuDRGNP (iso) |          0.08 (0.00)          | 0.14 (0.00) | 0.25 (0.00) | 0.09 (0.00) |          0.16 (0.00)          | 0.21 (0.00) |     |
|                                    |    ConvCNP    |          0.22 (0.00)          |      -      |      -      | 0.26 (0.00) |               -               |      -      |     |
|                                    |    ConvGNP    |          0.01 (0.00)          |      -      |      -      | 0.01 (0.00) |               -               |      -      |     |
|                                    |      CNP      |          0.33 (0.00)          | 0.44 (0.00) | 0.57 (0.00) | 0.39 (0.00) |          0.46 (0.00)          | 0.47 (0.00) |     |
|                                    |      GNP      |          0.05 (0.00)          | 0.09 (0.01) | 0.19 (0.00) | 0.07 (0.00) |          0.11 (0.00)          | 0.19 (0.00) |     |
|                                    |  RCNP (sta)   |          0.26 (0.00)          | 0.40 (0.01) | 0.45 (0.00) | 0.30 (0.00) |          0.39 (0.00)          | 0.35 (0.00) |     |
|                                    |  RGNP (sta)   |          0.03 (0.00)          | 0.05 (0.00) | 0.11 (0.00) | 0.03 (0.00) |          0.05 (0.00)          | 0.11 (0.00) |     |
|                                    | FuDRCNP (iso) |          0.26 (0.00)          | 0.31 (0.00) | 0.35 (0.00) | 0.30 (0.00) |          0.32 (0.00)          | 0.29 (0.00) |     |
|                                    | FuDRGNP (iso) |          0.08 (0.00)          | 0.14 (0.00) | 0.25 (0.00) | 0.09 (0.00) |          0.16 (0.00)          | 0.21 (0.00) |     |
|                                    |    ConvCNP    |          0.22 (0.00)          |      -      |      -      | 0.26 (0.00) |               -               |      -      |     |
|                                    |    ConvGNP    |          0.01 (0.00)          |      -      |      -      | 0.01 (0.00) |               -               |      -      |     |
|                                    |      CNP      |          4.54 (1.76)          | 3.30 (1.55) | 1.22 (0.09) | 6.75 (2.72) |          1.75 (0.42)          | 0.93 (0.02) |     |
|                                    |      GNP      |          2.25 (0.61)          | 2.54 (1.44) | 0.74 (0.02) | 1.06 (0.26) |          1.23 (0.17)          | 0.62 (0.02) |     |

> **Image description.** This image contains two line graphs comparing the performance of different models on the Hartmann function with different input dimensions. The left graph is labeled "Hartmann 3d" and the right graph is labeled "Hartmann 6d".
>
> Both graphs share the same y-axis label: "error |min f(xt) - fmin|" ranging from 0.0 to 2.5 on the left graph and from 0.0 to 3.0 on the right graph. The x-axis label for both graphs is "number of queries". The x-axis ranges from 0 to 50 on the left graph and from 0 to 100 on the right graph.
>
> Each graph displays multiple lines, each representing a different model: GP (dotted black line), CNP (solid blue line), GNP (solid green line), ACNP (dotted orange line), AGNP (solid orange line), RCNP (dotted green line), and RGNP (solid green line). Each line is surrounded by a shaded region of the same color, representing the uncertainty or variance of the model's performance.
>
> The legend on the right side of the image lists the models and their corresponding line styles and colors.

Figure 2: Bayesian Optimization. Error during optimization of a 3D/6D Hartmann function (lower is better). RCNP/RGNP improve upon the baselines, approaching the GP performance.

Results. As shown in Figure 2, RCNPs and RGNPs are able to learn from the random functions and come close to the performance of a Gaussian Process (GP), the most common surrogate model for BayesOpt. Note that the GP is refit after every new observation, while the CNP models can condition on the new observation added to the context set, without any retraining. CNPs and GNPs struggle with the diversity provided by the random kernel function samples and fail at the task. In order to remain competitive, they need to be extended with an attention mechanism (ACNP, AGNP).

### 5.3 Lotka-Volterra model

Neural process models excel in the so-called sim-to-real task where a model is trained using simulated data and then applied to real-world contexts. Previous studies have demonstrated this capability by training neural process models with simulated data generated from the stochastic Lotka-Volterra predator-prey equations and evaluating them with the famous hare-lynx dataset [32]. We run this benchmark evaluation using the simulator and experimental setup proposed by [3]. Here the CNP models are trained with simulated data and evaluated with both simulated and real data; the learning tasks represented in the training and evaluation data include interpolation, forecasting, and reconstruction. The evaluation results presented in Table 3 indicate that the best model depends on the task type, but overall our proposed relational CNP models with translational equivariance perform comparably to their convolutional and attentive counterparts, showing that our simpler approach does not hamper performance on real data. Full results with baseline CNPs are provided in Appendix F.

### 5.4 Reaction-Diffusion model

The Reaction-Diffusion (RD) model is a large class of state-space models originating from chemistry [45] with several applications in medicine [13] and biology [22]. We consider here a reduced model representing the evolution of cancerous cells [13], which interact with healthy cells through the

---

#### Page 9

Table 3: Normalized log-likelihood scores in the Lotka-Volterra experiments (higher is better). The mean and (standard deviation) reported for each model are calculated based on 10 training outcomes evaluated with the same simulated (S) and real (R) learning tasks. The tasks include interpolation (INT), forecasting (FOR), and reconstruction (REC). Statistically significantly (see Appendix B.1) best results are bolded. RCNP and RGNP models perform on par with convolutional and attentive baselines.

|         |    INT (S)    |    FOR (S)    |    REC (S)    |    INT (R)    |    FOR (R)    |    REC (R)    |
| :-----: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
|  RCNP   | $-3.57(0.02)$ | $-4.85(0.00)$ | $-4.20(0.01)$ | $-4.24(0.02)$ | $-4.83(0.03)$ | $-4.55(0.05)$ |
|  RGNP   | $-3.51(0.01)$ | $-4.27(0.00)$ | $-3.76(0.00)$ | $-4.31(0.06)$ | $-4.47(0.03)$ | $-4.39(0.11)$ |
| ConvCNP | $-3.47(0.01)$ | $-4.85(0.00)$ | $-4.06(0.00)$ | $-4.21(0.04)$ | $-5.01(0.02)$ | $-4.75(0.05)$ |
| ConvGNP | $-3.46(0.00)$ | $-4.30(0.00)$ | $-3.67(0.01)$ | $-4.19(0.02)$ | $-4.61(0.03)$ | $-4.62(0.11)$ |
|  ACNP   | $-4.04(0.06)$ | $-4.87(0.01)$ | $-4.36(0.03)$ | $-4.18(0.05)$ | $-4.79(0.03)$ | $-4.48(0.02)$ |
|  AGNP   | $-4.12(0.17)$ | $-4.35(0.09)$ | $-4.05(0.19)$ | $-4.33(0.15)$ | $-4.48(0.06)$ | $-4.29(0.10)$ |

production of acid. These three quantities (healthy cells, cancerous cells, acid concentration) are defined on a discretized space-time grid $(2+1$ dimensions, $d_{z}=3)$. We assume we only observe the difference in number between healthy and cancerous cells, which makes this model a hidden Markov model. Using realistic parameters inspired by [13], we simulated $4 \cdot 10^{3}$ full trajectories, from which we subsample observations to generate training data for the models; another set of $10^{3}$ trajectories is used for testing. More details can be found in Appendix G.
We propose two tasks: first, a completion task, where target points at time $t$ are inferred through the context at different spatial locations at time $t-1, t$ and $t+1$; secondly, a forecasting task, where the target at time $t$ is inferred from the context at $t-1$ and $t-2$. These tasks, along with the form of the equation describing the model, induce translation invariance in both space and time, which requires the models to incorporate translational equivariance for all three dimensions.

Results. We compare our translational-equivariant RCNP models to ACNP, CNP, and their GNP variants in Table 4. Comparison with ConvCNP is not feasible, as this problem is three-dimensional. Our methods outperform the others on both the completion and forecasting tasks, showing the advantage of leveraging translational equivariance in this complex spatio-temporal modeling problem.

Table 4: Normalized log-likelihood scores in the Reaction-Diffusion problem for both tasks (higher is better). Mean and (standard deviation) from 10 training runs evaluated on a separate test dataset.

|             |     RCNP     |     RGNP      |     ACNP      |     AGNP      |      CNP      |      GNP      |
| :---------: | :----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| Completion  | $0.22(0.33)$ | $1.38(0.62)$  | $0.17(0.03)$  | $0.20(0.03)$  | $0.10(0.01)$  | $0.13(0.03)$  |
| Forecasting | $0.07(0.18)$ | $-0.18(0.58)$ | $-0.65(0.31)$ | $-1.58(1.05)$ | $-0.51(0.20)$ | $-0.50(0.30)$ |

### 5.5 Additional experiments

As further empirical tests of our method, we study our technique in the context of autoregressive CNPs in Appendix H.1, present a proof-of-concept of incorporating rotational symmetry in Appendix H.2, and examine the performance of RCNPs on image regression in Appendix H.3.

## 6 Related work

This work builds upon the foundation laid by CNPs [10] and other members of the CNP family, covered at length in Section 2. A significant body of work has focused on incorporating equivariances into neural network models [33, 4, 23, 41]. The concept of equivariance has been explored in Convolutional Neural Networks (CNNs; [24]) with translational equivariance, and more generally in Group Equivariant CNNs [5], where rotations and reflections are also considered. Work on DeepSets laid out the conditions for permutation invariance and equivariance [48]. Set Transformers [25] extend

---

#### Page 10

this approach with an attention mechanism to learn higher-order interaction terms among instances of a set. Our work focuses on incorporating equivariances into prediction maps, and specifically CNPs.

Prior work on incorporating equivariances into CNPs requires a regular discrete lattice of the input space for their convolutional operations [15, 30]. EquivCNPs [19] build on work by [7] which operates on irregular point clouds, but they still require a constructed lattice over the input space. SteerCNPs [18] generalize ConvCNPs to other equivariances, but still suffer from the same scaling issues. These methods are therefore in practice limited to low-dimensional (one to two equivariant dimensions), whereas our proposal does not suffer from this constraint.
Our approach is also related to metric-based meta-learning, such as Prototypical Networks [36] and Relation Networks [37]. These methods learn an embedding space where classification can be performed by computing distances to prototype representations. While effective for few-shot classification tasks, they may not be suitable for more complex tasks or those requiring uncertainty quantification. GSSM [46] aims to learn relational biases via a graph structure on the context set, while we directly build exact equivariances into the CNP architecture.

Kernel methods and Gaussian processes (GPs) have long addressed issues of equivariance by customizing kernel designs to encode specific equivariances [14]. For instance, stationary kernels are used to capture globally consistent patterns [34], with applications in many areas, notably Bayesian Optimization [12]. However, despite recent computational advances [9], kernel methods and GPs still struggle with high-dimensional, complex data, with open challenges in deep kernel learning [47] and amortized kernel learning (or metalearning) [26, 35], motivating our proposal of RCNPs.

## 7 Limitations

RCNPs crucially rely on a comparison function $g\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ to encode equivariances. The comparison functions we described (e.g., for isotropy and translational equivariance) already represent a large class of useful equivariances. Notably, key contributions to the neural process literature focus only on translation equivariance (e.g., ConvCNP [15], ConvGNP [30], FullConvGNP [2]). Extending our method to other equivariances will require the construction of new comparison functions.

The main limitation of the RCNP class is its increased computational complexity in terms of context and target set sizes (respectively, $N$ and $M$ ). The FullRCNP model can be cumbersome, with $O\left(N^{2} M\right)$ cost for training and deployment. However, we showed that the simple or 'diagonal' RCNP variant can fully implement translational invariance with a $O(N M)$ cost. Still, this cost is larger than $O(N+M)$ of basic CNPs. Given the typical metalearning setting of small-data regime (small context sets), the increased complexity is often acceptable, outweighed by the large performance improvement obtained by leveraging available equivariances. This is shown in our empirical validation, in which RCNPs almost always outperformed their CNP counterparts.

## 8 Conclusion

In this paper, we introduced Relational Conditional Neural Processes (RCNPs), a new member of the neural process family which incorporates equivariances through relational encoding of the context and target sets. Our method applies to equivariances that can be induced via an appropriate comparison function; here we focused on translational equivariances (induced by the difference comparison) and equivariances to rigid transformations (induced by the distance comparison). How to express other equivariances via our relational approach is an interesting direction for future work.
We demonstrated with both theoretical results and extensive empirical validation that our method successfully introduces equivariances in the CNP model class, performing comparably to the translationalequivariant ConvCNP models in low dimension, but with a simpler construction that allows RCNPs to scale to larger equivariant input dimensions $\left(d_{x}>2\right)$ and outperform other CNP models.
In summary, we showed that the RCNP model class provides a simple and effective way to implement translational and other equivariances into the CNP model family. Exploiting equivariances intrinsic to a problem can significantly improve performance. Open problems remain in extending the current approach to other equivariances which are not expressible via a comparison function, and making the existing relational approach more efficient and scalable to larger context datasets.
