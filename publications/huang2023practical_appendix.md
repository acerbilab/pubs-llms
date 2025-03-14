# Practical Equivariances via Relational Conditional Neural Processes - Appendix

---

#### Page 14

# Appendix 

In this Appendix, we include our methodology, mathematical proofs, implementation details for reproducibility, additional results and extended explanations omitted from the main text.

## Contents

A Theoretical proofs ..... 15
A. 1 Definition of equivariance for prediction maps ..... 15
A. 2 Proof that RCNPs are equivariant ..... 16
A. 3 Proof that RCNPs are context-preserving ..... 16
B Methods ..... 18
B. 1 Training procedure, error bars and statistical significance ..... 18
B. 2 Experimental details and reproducibility ..... 19
B. 3 Details on Figure 1 ..... 19
C Computational cost analysis ..... 19
C. 1 Inference time analysis ..... 19
C. 2 Overall training time ..... 20
D Details of synthetic regression experiments ..... 20
D. 1 Models ..... 21
D. 2 Data ..... 21
D. 3 Full results ..... 22
E Details of Bayesian optimization experiments ..... 23
E. 1 Models ..... 23
E. 2 Data ..... 24
E. 3 Bayesian optimization ..... 25
E. 4 Computation time ..... 26
F Details of Lotka-Volterra model experiments ..... 26
F. 1 Models ..... 26
F. 2 Data ..... 28
F. 3 Full results ..... 28
F. 4 Computation time ..... 29
G Details of Reaction-Diffusion model experiments ..... 29
G. 1 Models ..... 29
G. 2 Data ..... 29
G. 3 Additional results ..... 31
G. 4 Computation time ..... 31
H Additional experiments ..... 31

---

#### Page 15

H. 1 Autoregressive CNPs ..... 31
H.1.1 Computation time ..... 32
H. 2 Rotation equivariance ..... 32
H.2.1 Computation time ..... 33
H. 3 Image datasets ..... 34
H.3.1 Setup ..... 34
H.3.2 Experiment 1: Centered images (no translation) ..... 34
H.3.3 Experiment 2: Translated images ..... 35
H.3.4 Computation time ..... 38

# A Theoretical proofs 

In this section, we provide extended proofs for statements and theorems in the main text. First, we show that our definition of equivariance for the prediction map is equivalent to the common definition of equivalent maps (Section A.1). Then, we provide full proofs that RCNPs are equivariant (Section A.2) and context-preserving (Section A.3) prediction maps, as presented in the theorems in Section 4 of the main text.

## A. 1 Definition of equivariance for prediction maps

We show that the definition of equivariance of a prediction map, which is introduced in the main text based on the invariance of the representation function (Eq. 3 in the main text), is equivalent to the common definition of equivariance for a generic mapping.
First, we need to extend the notion of group action of a group $\mathcal{T}$ into $\mathbb{R}^{d}$, for $d \in \mathbb{N}$. We define two additional actions, (i) for $\tau \in \mathcal{T}$ and $f$ a function from $\mathbb{R}^{d}$ into $\mathbb{R}^{p}$, and (ii) for a couple $(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{d} \times \mathbb{R}^{p}$. Following [15]:

$$
\tau(\mathbf{x}, \mathbf{y})=(\tau \mathbf{x}, \mathbf{y}), \quad \tau f(\mathbf{x})=f\left(\tau^{-1} \mathbf{x}\right)
$$

These two definitions can be extended for $n$-uplets $\mathbf{X}$ and $\mathbf{Y}$ :

$$
\tau(\mathbf{X}, \mathbf{Y})=(\tau \mathbf{X}, \mathbf{Y}), \quad \tau f(\mathbf{X})=f\left(\tau^{-1} \mathbf{X}\right)
$$

In the literature, equivariance is primarily defined for a map $F: \mathbf{Z} \in \mathcal{X} \times \mathbb{R}^{d} \mapsto F_{\mathbf{Z}} \in \mathcal{F}\left(\mathcal{X}, \mathbb{R}^{d}\right)$ by:

$$
\tau F_{\mathbf{Z}}=F_{\tau \mathbf{Z}}
$$

Conversely, in the main text (Section 2), we defined equivariance for a prediction map through the representation function: a prediction map $\pi$ with representation function $r$ is $\mathcal{T}$-equivariant if and only if for all $\tau \in \mathcal{T}$,

$$
r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)=r\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{*}\right)
$$

Eqs. S1 and S2 differ in that they deal with distinct objects. The former corresponds to the formal definition of an equivariant map, while the latter uses the representation function, which is simpler to work with for the scope of our paper.
To show the equivalence of the two definitions (Eqs. S1 and S2), we recall that a prediction map $\pi$ is defined through its representation $r$ :

$$
P_{\mathbf{Z}}\left(\mathbf{X}^{*}\right)=\pi\left(\cdot \mid \mathbf{Z}, \mathbf{X}^{*}\right)=p\left(\cdot \mid r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)\right)
$$

where $\mathbf{Z} \equiv(\mathbf{X}, \mathbf{Y})$. Then, starting from Eq. S1 applied to the prediction map,

$$
\begin{aligned}
P_{\tau(\mathbf{X}, \mathbf{Y})}=\tau P_{(\mathbf{X}, \mathbf{Y})} & \Longleftrightarrow \forall \mathbf{X}^{*}, p\left(\cdot \mid r\left((\tau \mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)\right)=p\left(\cdot \mid r\left((\mathbf{X}, \mathbf{Y}), \tau^{-1} \mathbf{X}^{*}\right)\right) \\
& \Longleftrightarrow \forall \mathbf{X}^{*}, r\left((\tau \mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)=r\left((\mathbf{X}, \mathbf{Y}), \tau^{-1} \mathbf{X}^{*}\right)
\end{aligned}
$$

Noting that $\tau^{-1} r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)=r\left((\mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{*}\right)$, by applying $\tau^{-1}$ to each side of the last line of the equation above, and swapping sides, we get:

$$
P_{\tau(\mathbf{X}, \mathbf{Y})}=\tau P_{(\mathbf{X}, \mathbf{Y})} \Longleftrightarrow \forall \mathbf{X}^{*}, r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)=r\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{*}\right)
$$

that is, Eqs. S1 and S2 are equivalent.

---

#### Page 16

# A. 2 Proof that RCNPs are equivariant 

We recall from Eq. 3 in the main text that a prediction map $\pi$ with representation function $r$ is $\mathcal{T}$-equivariant with respect to a group $\mathcal{T}$ of transformations, $\tau: \mathcal{X} \rightarrow \mathcal{X}$, if and only if for all $\tau \in \mathcal{T}$ :

$$
r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=r\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{\star}\right)
$$

where $\tau \mathbf{x} \equiv \tau(\mathbf{x})$ and $\tau \mathbf{X}$ is the set obtained by applying $\tau$ to all elements of $\mathbf{X}$.
Also recall that a prediction map $\pi$ and its representation function $r$ are relational with respect to a comparison function $g: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^{d_{\text {comp }}}$ if and only if $r$ can be written solely through the set comparisons defined in Eq. 8 of the main text:

$$
r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=r(g((\mathbf{X}, \mathbf{Y}),(\mathbf{X}, \mathbf{Y})), g\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right), g\left(\mathbf{X}^{\star},(\mathbf{X}, \mathbf{Y})\right), g\left(\mathbf{X}^{\star}, \mathbf{X}^{\star}\right)\right)
$$

We can now prove the following Lemma from the main text.
Lemma (Lemma 4.4). Let $\pi$ be a prediction map, $\mathcal{T}$ a transformation group, and $g$ a comparison function. If $\pi$ is relational with respect to $g$ and $g$ is $\mathcal{T}$-invariant, then $\pi$ is $\mathcal{T}$-equivariant.

Proof. From Eq. 8 in the main text, if $g$ is $\mathcal{T}$-invariant, then for a target set $\mathbf{X}^{\star}$ and $\tau \in \mathcal{T}$ :

$$
g\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)=\left\{\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{n}^{\star}\right), \mathbf{y}_{n}\right)\right\}=\left\{\left(g\left(\tau \mathbf{x}_{n}, \tau \mathbf{x}_{n}^{\star}\right), \mathbf{y}_{n}\right)\right\}=g\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{\star}\right)
$$

for all $1 \leq n \leq N$ and $1 \leq m \leq M$. A similar equality holds for all the comparison sets in the definition of a relational prediction map (see above). Since $\pi$ is relational, and using the equality above, we can write its representation function $r$ as:

$$
\begin{aligned}
r\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right) & =r(g((\mathbf{X}, \mathbf{Y}),(\mathbf{X}, \mathbf{Y})), g\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right), g\left(\mathbf{X}^{\star},(\mathbf{X}, \mathbf{Y})\right), g\left(\mathbf{X}^{\star}, \mathbf{X}^{\star}\right)\right) \\
& =r(g((\tau \mathbf{X}, \mathbf{Y}),(\tau \mathbf{X}, \mathbf{Y})), g\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{\star}\right), g\left(\tau \mathbf{X}^{\star},(\tau \mathbf{X}, \mathbf{Y})\right), g\left(\tau \mathbf{X}^{\star}, \tau \mathbf{X}^{\star}\right)\right) \\
& =r\left((\tau \mathbf{X}, \mathbf{Y}), \tau \mathbf{X}^{\star}\right)
\end{aligned}
$$

Thus, $\pi$ is $\mathcal{T}$-equivariant.
From the Lemma, we can prove the first main result of the paper, that RCNPs are equivariant prediction maps for a transformation group $\mathcal{T}$ provided a suitable comparison function $g$.
Proposition (Proposition 4.5). Let $g$ be the comparison function used in a RCNP, and $\mathcal{T}$ a group of transformations. If $g$ is $\mathcal{T}$-invariant, the RCNP is $\mathcal{T}$-equivariant.

Proof. In all RCNP models, elements of the context and target sets are always processed via a relational encoding. By definition (Eqs. 5 and 7 in the main text), the relational encoding is written solely as a function of $g((\mathbf{X}, \mathbf{Y}),(\mathbf{X}, \mathbf{Y}))$ and $g\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{\star}\right)$, so the same holds for the representation function $r$. Thus, any RCNP is a relational prediction map with respect to $g$, and it follows that it is also a $\mathcal{T}$-equivariant prediction map according to Lemma 4.4.

## A. 3 Proof that RCNPs are context-preserving

We prove here that RCNPs preserve information in the context set. We will do so by providing a construction for a function that reconstructs the context set given the relationally encoded vector $\boldsymbol{\rho}_{m}$ (up to transformations). As a disclaimer, the proofs in this section should be taken as 'existence proofs' but the provided construction is not practical. There are likely better context-preserving encodings, as shown by the empirical performance of RCNPs with moderately sized networks. Future theoretical work should provide stronger bounds on the size of the relational encoding, as recently established for DeepSets [43, 44].
For convenience, we recall Eq. 5 of the main text for the relational encoding provided a comparison function $g$ :

$$
\boldsymbol{\rho}_{m}=\rho_{\text {full }}\left(\mathbf{x}_{m}^{\star},(\mathbf{X}, \mathbf{Y})\right)=\bigoplus_{n, n^{\prime}=1}^{N} f_{r}\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right), \mathbf{R}_{n n^{\prime}}\right), \quad \mathbf{R}_{n n^{\prime}} \equiv\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{n^{\prime}}\right), \mathbf{y}_{n}, \mathbf{y}_{n^{\prime}}\right)
$$

where $\mathbf{R} \equiv g((\mathbf{X}, \mathbf{Y}), g(\mathbf{X}, \mathbf{Y}))$ is the relational matrix. First, we show that from the relational encoding $\rho_{\text {full }}$ we can reconstruct the matrix $\mathbf{R}$, modulo permutations of rows or columns as defined below.

---

#### Page 17

Definition A.1. Let $\mathbf{R}, \mathbf{R}^{\prime} \in \mathbb{R}^{N \times N}$. The two matrices are equal modulo permutations, denoted $\mathbf{R} \cong \mathbf{R}^{\prime}$, if there is a permutation $\sigma \in \operatorname{Sym}(N)$ such that $\mathbf{R}_{n n^{\prime}}=\mathbf{R}_{\sigma(n) \sigma\left(n^{\prime}\right)}^{\prime}$, for all $1 \leq n, n^{\prime} \leq N$.

In other words, $\mathbf{R} \cong \mathbf{R}^{\prime}$ if $\mathbf{R}^{\prime}$ is equal to $\mathbf{R}$ after an appropriate permutation of the indices of both rows and columns. Now we proceed with the reconstruction proof.
Lemma A.2. Let $g: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^{d_{\text {comp }}}$ be a comparison function, $(\mathbf{X}, \mathbf{Y})$ a context set, $\mathbf{X}^{*}$ a target set, $\mathbf{R}$ the relational matrix and $\rho_{\text {full }}$ the relational encoding as per Eq. 5 in the main text. Then there is a reconstruction function $h$, for $\mathbf{r}=\left(\boldsymbol{\rho}_{1}, \ldots, \boldsymbol{\rho}_{M}\right)$, such that $h(\mathbf{r})=\mathbf{R}^{\prime}$ where $\mathbf{R}^{\prime} \cong \mathbf{R}$.

Proof. For this proof, it is sufficient to show that there is a function $h$ applied to a local representation of any context point, $\boldsymbol{\rho}_{m}=\rho_{\text {full }}\left(\mathbf{x}_{m}^{*},(\mathbf{X}, \mathbf{Y})\right)$, such that $h\left(\boldsymbol{\rho}_{m}\right)=\mathbf{R}^{\prime}$, with $\mathbf{R}^{\prime} \cong \mathbf{R}$. In this proof, we also consider that all numbers in a computer are represented up to numerical precision, so with $\mathbb{R}$ we denote the set of real numbers physically representable in a chosen floating point representation. As a final assumption, without loss of generality, we take $\mathbf{y}_{n} \neq \mathbf{y}_{n^{\prime}}$ for $n \neq n^{\prime}$. If there are non-unique elements $\mathbf{y}_{n}$ in the context set, in practice we can always disambiguate them by adding a small jitter.
Let $f_{r}: \mathcal{R} \rightarrow \mathbb{R}^{d_{\text {rel }}}$ be the relational encoder of Eq. 5 of the main text, a neural network parametrized by $\boldsymbol{\theta}_{r}$, where $\mathcal{R}=\mathbb{R}^{d_{\text {comp }}} \times \mathbb{R}^{d_{\text {comp }}+2 d_{\mathrm{rg}}}$. Here, we also include in $\boldsymbol{\theta}_{r}$ hyperparameters defining the network architecture, such as the number of layers and number of hidden units per layer. We pick $d_{\text {rel }}$ large enough such that we can build a one-hot encoding of the elements of $\mathcal{R}$, i.e., there is an injective mapping from $\mathcal{R}$ to $\{0,1\}^{d_{\text {rel }}}$ such that one and only one element of the output vector is 1 . This injective mapping exists, since $\mathcal{R}$ is discrete due to finite numerical precision. We select $\boldsymbol{\theta}_{r}$ such that $f_{r}$ approximates the one-hot encoding up to arbitrary precision, which is achievable thanks to universal approximation theorems for neural networks [29]. Thus, $\boldsymbol{\rho}_{m}=\bigoplus_{n, n^{\prime}} f_{r}\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{*}\right), \mathbf{R}_{n n^{\prime}}\right)$ is a (astronomically large) vector with ones at the elements that denote $\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{*}\right), \mathbf{R}_{n n^{\prime}}\right)$ for $1 \leq n, n^{\prime} \leq N$.
Finally, we can build a reconstruction function $h$ that reads out $\boldsymbol{\rho}_{m}$, the sum of one-hot encoded vectors, mapping each 'one', based on its location, back to the corresponding input $\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{*}\right), \mathbf{R}_{n n^{\prime}}\right)$. We have then all the information to build a matrix $\mathbf{R}^{\prime}$ indexed by $\mathbf{y}$, that is

$$
\mathbf{R}_{\mathbf{y} \mathbf{y}^{\prime}}^{\prime}=\mathbf{R}_{n n^{\prime}}, \quad \text { with } \mathbf{y}=\mathbf{y}_{n}, \mathbf{y}^{\prime}=\mathbf{y}_{n^{\prime}}, \quad \text { for some } \mathbf{y}, \mathbf{y}^{\prime} \in \mathbf{Y}, 1 \leq n, n^{\prime} \leq N
$$

Since we assumed $\mathbf{y}_{n} \neq \mathbf{y}_{n^{\prime}}$ for $n \neq n^{\prime}, \mathbf{R}^{\prime}$ is unique up to permutations of the rows and columns, and by construction $\mathbf{R}^{\prime} \cong \mathbf{R}$.

We now define the matrix

$$
\mathbf{Q}_{m n n^{\prime}}=\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{*}\right), g\left(\mathbf{x}_{n}, \mathbf{x}_{n^{\prime}}\right), \mathbf{y}_{n}, \mathbf{y}_{n^{\prime}}\right)=\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{*}\right), \mathbf{R}_{n n^{\prime}}\right)
$$

for $1 \leq m \leq M$ and $1 \leq n, n^{\prime} \leq N$. We recall that a comparison function $g$ is context-preserving with respect to a transformation group $\mathcal{T}$ if for any context set $(\mathbf{X}, \mathbf{Y})$ and target set $\mathbf{X}^{*}$, for all $1 \leq m \leq M$, there is a submatrix $\mathbf{Q}^{\prime} \subseteq \mathbf{Q}_{m \cup}$, a reconstruction function $\gamma$, and a transformation $\tau \in \mathcal{T}$ such that $\gamma\left(\mathbf{Q}^{\prime}\right)=(\tau \mathbf{X}, \mathbf{Y})$. We showed in Section 4.2 of the main text that the distance comparison function $g_{\text {dist }}$ is context-preserving with respect to the group of rigid transformations; and that the difference comparison function $g_{\text {diff }}$ is context-preserving with respect to the translation group. Similarly, a family of functions $h_{\boldsymbol{\theta}}\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right) \rightarrow \mathbb{R}^{d_{\text {rep }}}$ is context-preserving under $\mathcal{T}$ if there exists $\boldsymbol{\theta} \in \boldsymbol{\Theta}, d_{\text {rep }} \in \mathbb{N}$, a reconstruction function $\gamma$, and a transformation $\tau \in \mathcal{T}$ such that $\gamma\left(h_{\boldsymbol{\theta}}\left((\mathbf{X}, \mathbf{Y}), \mathbf{X}^{*}\right)\right)=(\tau \mathbf{X}, \mathbf{Y})$
With the construction and definitions above, we can now show that RCNPs are in principle contextpreserving. In particular, we will apply the definition of context-preserving functions $h$ to the class of representation functions $r$ of the RCNP models introduced in the paper, for which $d_{\text {rep }}=M d_{\text {rel }}$ since $\mathbf{r}=\left(\boldsymbol{\rho}_{1}, \ldots, \boldsymbol{\rho}_{M}\right)$. The proof applies both to the full RCNP and the simple (diagonal) RCNP, for an appropriate choice of the comparison function $g$.
Proposition (Proposition 4.8). Let $\mathcal{T}$ be a transformation group and $g$ the comparison function used in a FullRCNP. If $g$ is context-preserving with respect to $\mathcal{T}$, then the representation function $r$ of the FullRCNP is context-preserving with respect to $\mathcal{T}$.

Proof. Given Lemma A.2, for any given $m$, we can reconstruct $\mathbf{Q}_{m \cup}=(\ldots, \mathbf{R})$ (modulo permutations) from the relational encoding. Since $g$ is context-preserving with respect to $\mathcal{T}$, we can

---

#### Page 18

reconstruct $(\mathbf{X}, \mathbf{Y})$ from $\mathbf{R}$ modulo a transformation $\tau \in \mathcal{T}$. Thus, $\rho_{\text {full }}$ is context-preserving with respect to $\mathcal{T}$ and so is $r$.

Proposition (Proposition 4.9). Let $\mathcal{T}$ be the translation group and $g_{\text {diff }}$ the difference comparison function. The representation of the simple RCNP model with $g_{\text {diff }}$ is context-preserving with respect to $\mathcal{T}$.

Proof. As shown in the main text, the difference comparison function is context-preserving for $\mathbf{Q}_{m:}^{\prime}=\left.\mathbf{Q}_{m n n}\right|_{n=1} ^{N}=\left(g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right), \mathbf{R}_{n n}\right)_{n=1}^{N}$, for a given $m$. In other words, we can reconstruct $(\mathbf{X}, \mathbf{Y})$ given only the diagonal of the $\mathbf{R}$ matrix and the comparison between a single target input $\mathbf{x}_{m}^{\star}$ and the elements of the context set, $g\left(\mathbf{x}_{n}, \mathbf{x}_{m}^{\star}\right)$. Following a construction similar to the one in Lemma A.2, we can map $\boldsymbol{\rho}_{m}$ back to $\mathbf{Q}_{m:}^{\prime}$. Then since $g$ is context-preserving with respect to $\mathbf{Q}_{m:}^{\prime}$, we can reconstruct $(\mathbf{X}, \mathbf{Y})$ up to translations.

A potentially surprising consequence of our context-preservation theorems is that in principle no information is lost about the entire context set. Thus, in principle an RCNP with a sufficiently large network is able to reconstruct any high-order interaction of the context set, even though the RCNP construction only builds upon interactions of pairs of points. However, higher-order interactions might be harder to encode in practice. Since the RCNP representation is built on two-point interactions, depending on the network size, it may be harder for the network to effectively encode the simultaneous interaction of many context points.

# B Methods 

## B. 1 Training procedure, error bars and statistical significance

This section describes the training and evaluation procedures used in our regression experiments. The training procedure follows [3, Appendix G]. Stochastic gradient descent is performed via the Adam algorithm [21] with a learning rate specified in each experiment. After each epoch during model training, each model undergoes validation using a pre-generated validation set. The validation score is a confidence bound based on the log-likelihood values. Specifically, the mean ( $\mu_{\text {val }}$ ) and standard deviation ( $\sigma_{\text {val }}$ ) of the log-likelihood values over the entire validation set are used to calculate the validation score as $\mu_{\text {val }}-1.96 \sigma_{\text {val }} / \sqrt{N_{\text {val }}}$ where $N_{\text {val }}$ is the validation dataset size. The validation score is compared to the previous best score observed within the training run, and if the current validation score is higher, the current model parameters (e.g., weights and biases for neural networks) are saved. The models are trained for a set number of epochs in each experiment, and when the training is over, the model is returned with the parameters that resulted in the highest validation score.
To ensure reliable statistical analysis and fair comparisons, we repeat model training using the above procedure ten times in each experiment, with each run utilizing different seeds for training and validation dataset generation and model training. In practice, we generated ten seed sets so that each training run with the same model utilizes different seeds, but we maintain consistency by using the same seeds across different models in the same experiment. Each model is then represented with ten training outcomes in each experiment, and we evaluate all training outcomes with a fixed evaluation set or evaluation sets.
We evaluate the methods based on two performance measures: (i) the log-likelihood, normalized with respect to the number of target points, and (ii) the Kullback-Leibler (KL) divergence between the posterior prediction map $\pi$ and a target $\pi^{*}$, defined as

$$
\mathrm{KL}\left(\pi \| \pi^{*}\right)=\int \pi(x) \log \frac{\pi(x)}{\pi^{*}(x)} \mathrm{d} x
$$

assuming that $\pi$ is absolutely continuous to $\pi^{*}$ with respect to a measure $\lambda$ (usually, the Lebesgue measure). We report the KL normalized with respect to the number of target points.
Throughout our experiments, we rely on the KL divergence to evaluate the experiments with Gaussian process (GP) generated functions ('Gaussian'), for which we can compute the ground-truth target $\pi^{*}$ (the multivariate normal prediction of the posterior GP). We use the normalized log-likelihood for all the other ('non-Gaussian') cases. For each model and evaluation set, we report both the average score

---

#### Page 19

calculated across training runs and the standard deviation between scores observed in each training run.

We compare models based on the average scores and also run pairwise statistical tests to identify models that resulted in comparable performance. The results from pairwise comparisons are used throughout the paper in the results tables to highlight in bold the models that are considered best in each experiment. Specifically, we use the following method to determine the best models:

- First, we highlight in bold the model (A) that has the best empirical mean metric.
- Then, for each alternative model (B), we run a one-sided Student's t-test for paired samples, with the alternative hypothesis that model (B) has a higher (better) mean than model (A), and the null hypothesis that the two models have the same mean. Samples are paired since they share a random seed (i.e., the same training data).
- The models that do not lead to statistically significant ( $p<0.05$ ) rejection of the null hypothesis are highlighted in bold as well.

# B. 2 Experimental details and reproducibility 

The experiments carried out in this work used the open-source neural processes package released with previous work [3]. The package is distributed under the MIT license and available at https: //github.com/wesselb/neuralprocesses [1]. We extended the package with the proposed relational model architectures and the new experiments considered in this work. Our implementation of RCNP is available at https://github.com/acerbilab/relational-neural-processes.
For the purpose of open science and reproducibility, all training and evaluation details are provided in the following sections of this Appendix as well as in the Github repository linked above.

## B. 3 Details on Figure 1

In Figure 1 in the main text, we compare CNP and RCNP models. The RCNP model used in this example utilizes a stationary kernel to encode translation equivariance. The encoding in the CNP model and the relational encoding in the RCNP model are 256-dimensional, and both models used an encoder network with three hidden layers and 256 hidden units per layer. In addition, both models used a decoder network with six hidden layers and 256 hidden units per layer. We used the CNP architecture from previous work [3] and modeled the RCNP architecture on the CNP architecture. The encoder and decoder networks use ReLU activation functions.

We trained the neural process models with datasets generated based on random functions sampled from a noiseless Gaussian process prior with a Matérn- $\frac{5}{2}$ covariance function with lengthscale $\ell=0.5$. The datasets were generated by sampling $1-30$ context points and 50 target points from interval $[-2,2]$. The neural process models were trained for 20 epochs with $2^{14}$ datasets in each epoch and learning rate $3 \cdot 10^{-4}$. The datasets used in training the CNP and RCNP models were generated with the same seed.

Figure 1 shows model predictions in two regression tasks. First, we visualized the predicted mean and credible bounds calculated as 1.96 times the predicted standard deviation in the range $[-2,2]$ given 10 context points. This corresponds to the standard interpolation (INT) task. Then we shifted the context points and visualized the predicted mean and credible bounds in the range $[2,6]$. This corresponds to an out-of-input-distribution (OOID) task. The Gaussian process model used as a reference had Matérn- $\frac{5}{2}$ covariance function with lengthscale $\ell=0.5$, the same process used to train the CNP and RCNP models.

## C Computational cost analysis

## C. 1 Inference time analysis

In this section, we provide a quantitative analysis of the computational cost associated with running various models, reporting the inference time under different input dimensions $d_{x}$ and varying context set sizes $N$. We compare our simple and full RCNP models with the CNP and ConvCNP families.

---

#### Page 20

All results are calculated on an Intel Core i7-12700K CPU, under the assumption that these models can be deployed on devices or local machines without GPU access.

Table S1 shows the results with different input dimensions $d_{x}$. We set the size of both context and target sets to 20 . Firstly, we can see the runtime of convolutional models increase significantly as they involve costly convolutional operations, and cannot be applied in practice to $d_{x}>2$. Conversely, the runtime cost remains approximately constant across $d_{x}$ for the other models. Secondly, our RCNP models have inference time close to that of CNP models, while FullRCNP models are slower (although still constant in terms of $d_{x}$ ). Since context set size is the main factor affecting the inference speed of our RCNP models, we delve further into the computational costs associated with varying context set sizes $N$, keeping the input dimension $d_{x}=1$ and target set size $M=20$ constant (Table S2). Given that RCNP has a cost of $O(N M)$, we observe a small, steady increase in runtime. Conversely, the FullRCNP models, with their $O\left(N^{2} M\right)$ cost, show an approximately quadratic surge in runtime as $N$ increases, as expected.

Table S1: Comparison of computational costs across different models under different input dimensions $d_{x}$. The size of both context sets $(N)$ and target sets $(M)$ is set to 20 . We report the mean value and (standard deviation) derived from 50 forward passes executed on various randomly generated datasets.

|  | Runtime $\left(\times 10^{-3} \mathrm{~s}\right)$ |  |  |  |
| :--: | :--: | :--: | :--: | :--: |
|  | $d_{x}=1$ | $d_{x}=2$ | $d_{x}=3$ | $d_{x}=5$ |
| RCNP | 2.35 (0.39) | 2.29 (0.28) | 2.28 (0.32) | 2.25 (0.27) |
| RGNP | 2.31 (0.30) | 2.41 (0.34) | 2.31 (0.26) | 2.50 (0.40) |
| FullRCNP | 8.24 (0.46) | 9.09 (1.78) | 8.27 (0.56) | 9.08 (1.72) |
| FullRGNP | 8.40 (0.56) | 9.25 (1.85) | 8.37 (0.62) | 9.20 (1.76) |
| ConvCNP | 4.46 (0.38) | 31.19 (2.43) | - | - |
| ConvGNP | 4.51 (0.48) | 38.04 (2.31) | - | - |
| CNP | 1.86 (0.30) | 2.10 (0.38) | 2.14 (0.30) | 2.09 (0.29) |
| GNP | 2.01 (0.27) | 2.09 (0.26) | 2.20 (0.26) | 2.10 (0.32) |

Table S2: Comparison of computational costs across different models under different context set sizes $N$. We maintain the input dimension at $d_{x}=1$, and a fixed target set size of $M=20$. We report the mean value and (standard deviation) derived from 50 forward passes executed on various randomly generated datasets.

|  | Runtime $\left(\times 10^{-3} \mathrm{~s}\right)$ |  |  |  |
| :--: | :--: | :--: | :--: | :--: |
|  | $N=10$ | $N=20$ | $N=50$ | $N=100$ |
| RCNP | 1.90 (0.26) | 2.34 (0.48) | 2.40 (0.30) | 2.90 (0.46) |
| RGNP | 2.10 (0.26) | 2.53 (0.34) | 2.55 (0.22) | 2.93 (0.37) |
| FullRCNP | 2.83 (0.33) | 8.80 (1.59) | 38.03 (4.71) | 154.17 (16.04) |
| FullRGNP | 3.08 (0.34) | 8.68 (1.02) | 38.33 (5.88) | 161.07 (19.54) |
| ConvCNP | 4.47 (0.43) | 4.47 (0.51) | 5.10 (0.62) | 5.83 (0.88) |
| ConvGNP | 5.10 (0.54) | 4.55 (0.35) | 5.26 (0.50) | 4.70 (0.46) |
| CNP | 1.95 (0.27) | 1.86 (0.28) | 1.87 (0.21) | 1.97 (0.19) |
| GNP | 2.48 (0.70) | 2.03 (0.19) | 2.17 (0.48) | 2.11 (0.23) |

# C. 2 Overall training time 

For the entire paper, we conducted all experiments, including baseline model computations and preliminary experiments not included in the paper, on a GPU cluster consisting of a mix of Tesla P100, Tesla V100, and Tesla A100 GPUs. We roughly estimate the total computational consumption to be around 25000 GPU hours. A more detailed evaluation of computing for each set of experiments is reported when available in the following sections.

## D Details of synthetic regression experiments

We report details and additional results for the synthetic regression experiments from Section 5.1 of the main text. The experiments compare neural process models trained on data sampled from both Gaussian and non-Gaussian synthetic functions, where 'Gaussian' refers to functions sampled from a

---

#### Page 21

Gaussian process (GP). Our procedure follows closely the synthetic regression experiments presented in [3].

# D. 1 Models 

We compare our RCNP models with the CNP and ConvCNP model families in regression tasks with input dimensions $d_{x}=\{1,2,3,5,10\}$. The CNP and GNP models used in this experiment encode the context sets as 256 -dimensional vectors when $d_{x}<5$ and as 128 -dimensional vectors when $d_{x} \geq 5$. Similarly, all relational models including RCNP, RGNP, FullRCNP, and FullRGNP produce relational encodings with dimension 256 when $d_{x}<5$ and dimension 128 when $d_{x} \geq 5$. The encoding network used in both model families to produce the encoding or relational encoding is a three-layer MLP, featuring 256 hidden units per layer for $d_{x}<5$ and 128 for $d_{x} \geq 5$. We also maintain the same setting across all CNP and RCNP models in terms of the decoder network architecture, using a six-layer MLP with 256 hidden units per layer for $d_{x}<5$ and 128 for $d_{x} \geq 5$. The encoder and decoder networks use ReLU activation functions. Finally, the convolutional models ConvCNP and ConvGNP, which are included in experiments where $d_{x}=\{1,2\}$, are employed with the configuration detailed in [3, Appendix F], and GNP, RGNP, FullRGNP, and ConvGNP models all use linear covariance with 64 basis functions.

Neural process models are trained with datasets representing a regression task with context and target features. The datasets used in these experiments were generated based on random functions sampled from Gaussian and non-Gaussian stochastic processes. The models were trained for 100 epochs with $2^{14}$ datasets in each epoch and learning rate $3 \cdot 10^{-4}$. The validation sets used in training included $2^{12}$ datasets and the evaluation sets used to compare the models in interpolation (INT) and out-of-input-distribution (OOID) tasks included $2^{12}$ datasets each.

## D. 2 Data

We used the following functions to generate synthetic Gaussian and non-Gaussian data:

- Exponentiated quadratic (EQ). We sample data from a GP with an EQ covariance function:

$$
f \sim \mathcal{G} \mathcal{P}\left(\mathbf{0}, \exp \left(-\frac{\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|_{2}^{2}}{2 \ell^{2}}\right)\right)
$$

where $\ell$ is the lengthscale.

- Matérn- $\frac{5}{2}$. We sample data from a GP with a Matérn- $\frac{5}{2}$ covariance function:

$$
f \sim \mathcal{G} \mathcal{P}\left(\mathbf{0},\left(1+\sqrt{5} r+\frac{5}{3} r^{2}\right) \exp (-\sqrt{5} r)\right)
$$

where $r=\frac{\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|_{2}}{\ell}$ and $\ell$ is the lengthscale.

- Weakly periodic. We sample data from a GP with a weakly periodic covariance function:

$$
f \sim \mathcal{G} \mathcal{P}\left(\mathbf{0}, \exp \left(-\frac{1}{2 \ell_{d}^{2}}\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|_{2}^{2}-\frac{2}{\ell_{p}^{2}}\left\|\sin \left(\frac{\pi}{p}\left(\mathbf{x}-\mathbf{x}^{\prime}\right)\right)\right\|_{2}^{2}\right)\right)
$$

where $\ell_{d}$ is the lengthscale that decides how quickly the similarity between points in the output of the function decays as their inputs move apart, $\ell_{p}$ determines the lengthscale of periodic variations, and $p$ denotes the period.

- Sawtooth. We sample data from a sawtooth function characterized by a stochastic frequency, orientation, and phase as presented by:

$$
f \sim \omega\langle\mathbf{x}, \mathbf{u}\rangle_{2}+\phi \quad \bmod 1
$$

where $\omega \sim \mathcal{U}(\Omega)$ is the frequency of the sawtooth wave, the direction of the wave is given as $\mathbf{u} \sim \mathcal{U}\left(\left\{\mathbf{x} \in \mathbb{R}^{d_{x}}:\|\mathbf{x}\|_{2}=1\right\}\right)$, and $\phi \sim \mathcal{U}(0,1)$ determines the phase.

- Mixture. We sample data randomly from either one of the three GPs or the sawtooth process, with each having an equal probability to be chosen.

---

#### Page 22

In this set of experiments, we evaluate the models using varying input dimensions $d_{x}=\{1,2,3,5,10\}$. To maintain a roughly equal level of difficulty across data with varying input dimensions $d_{x}$, the hyperparameters for the above data generation processes are selected in accordance with $d_{x}$ :

$$
\ell=\sqrt{d_{x}}, \quad \ell_{d}=2 \cdot \sqrt{d_{x}}, \quad \ell_{p}=4 \cdot \sqrt{d_{x}}, \quad p=\sqrt{d_{x}}, \quad \Omega=\left[\frac{1}{2 \sqrt{d_{x}}}, \frac{1}{\sqrt{d_{x}}}\right]
$$

For the EQ, Matérn- $\frac{5}{2}$, and weakly periodic functions, we additionally add independent Gaussian noise with $\sigma^{2}=0.05$.
The datasets representing regression tasks were generated by sampling context and target points from the synthetic data as follows. The number of context points sampled from each EQ, Matérn- $\frac{5}{2}$, or weakly periodic function varied uniformly between 1 and $30 \cdot d_{x}$, and the number of target points was fixed to $50 \cdot d_{x}$. Since the datasets sampled from the sawtooth and mixture functions represent more difficult regression problems, we sampled these functions for 1-30 context points when $d_{x}=1$ and $1-50 \cdot d_{x}$ context points when $d_{x}>1$, and we also fixed the number of target points to $100 \cdot d_{x}$. All training and validation datasets were sampled from the range $\mathcal{X}=[-2,2]^{d_{x}}$ while evaluation sets were sampled in two ways. To evaluate the models in an interpolation (INT) task, we generated evaluation datasets by sampling context and target points from the same range that was used during training, $\mathcal{X}_{\text {test }}=[-2,2]^{d_{x}}$. To evaluate the models in an out-of-input-distribution (OOID) task, we generated evaluation datasets with context and target points sampled from a range that is outside the boundaries of the training range, specifically $\mathcal{X}_{\text {test }}=[2,6]^{d_{x}}$.

# D. 3 Full results 

The results presented in the main text (Section 5.1) compared the proposed RCNP and RGNP models to baseline and convolutional CNP and GNP models in selected synthetic regression tasks. The full results encompassing a wider selection of tasks and an extended set of models are presented in Table S3, S4, S5, S6, S7. We constrained the experiments with EQ and Matérn- $\frac{5}{2}$ tasks (Table S3 and S4) to input dimensions $d_{x}=\{1,2,3,5\}$ owing to the prohibitive computational memory costs associated with the FullRCNP model included in these experiments. The other experiments (Table S5S7) were repeated with input dimensions $d_{x}=\{1,2,3,5,10\}$. Our RCNP models demonstrate consistently competitive performance when compared with ConvCNP models in scenarios with low dimensional data, and significantly outperform the CNP family of models when dealing with data of higher dimensions.

Table S3: Synthetic (EQ). Comparison of the interpolation (INT) and out-of-input-distribution (OOID) performance of our RCNP models with different CNP baselines on EQ function with varying input dimensions. We use normalized Kullback-Leibler divergences as our metric and show mean and (standard deviation) obtained from 10 runs with different seeds. Trivial refers to a model that predicts a Gaussian distribution utilizing the empirical mean and standard deviation derived from the context outputs. "F" denotes failed attempts that yielded very bad results. Missing entries could not be run. Statistically significantly best results are bolded.

|  | $d_{x}=1$ |  | $d_{x}=2$ |  | $d_{x}=3$ |  | $d_{x}=5$ |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | INT | OOID | INT | OOID | INT | OOID | INT | OOID |
| RCNP (sta) | 0.22 (0.00) | 0.22 (0.00) | 0.26 (0.00) | 0.26 (0.00) | 0.40 (0.01) | 0.40 (0.01) | 0.45 (0.00) | 0.45 (0.00) |
| RGNP (sta) | 0.01 (0.00) | 0.01 (0.00) | 0.03 (0.00) | 0.03 (0.00) | 0.05 (0.00) | 0.05 (0.00) | 0.11 (0.00) | 0.11 (0.00) |
| FullRCNP (iso) | 0.22 (0.00) | 0.22 (0.00) | 0.26 (0.00) | 0.26 (0.00) | 0.31 (0.00) | 0.31 (0.00) | 0.35 (0.00) | 0.35 (0.00) |
| FullRGNP (iso) | 0.03 (0.00) | 0.03 (0.00) | 0.08 (0.00) | 0.08 (0.00) | 0.14 (0.00) | 0.14 (0.00) | 0.25 (0.00) | 0.25 (0.00) |
| ConvCNP | 0.21 (0.00) | 0.21 (0.00) | 0.22 (0.00) | 0.22 (0.00) | - | - | - | - |
| ConvGNP | 0.00 (0.00) | 0.00 (0.00) | 0.01 (0.00) | 0.01 (0.00) | - | - | - | - |
| CNP | 0.25 (0.00) | 2.21 (0.70) | 0.33 (0.00) | 4.54 (1.76) | 0.44 (0.00) | 3.30 (1.55) | 0.57 (0.00) | 1.22 (0.09) |
| GNP | 0.02 (0.00) | F | 0.05 (0.00) | 2.25 (0.61) | 0.09 (0.01) | 2.54 (1.44) | 0.19 (0.00) | 0.74 (0.02) |
| Trivial | 1.03 (0.00) | 1.03 (0.00) | 1.13 (0.00) | 1.13 (0.00) | 1.12 (0.00) | 1.12 (0.00) | 1.03 (0.00) | 1.03 (0.00) |

We also conducted an additional experiment to explore all the possible combinations of comparison functions with the RCNP and FullRCNP models on EQ and Matérn- $\frac{5}{2}$ tasks. Table S8 provides a comprehensive view of the results for EQ tasks across three input dimensions. The empirical results support our previous demonstration that RCNP models are capable of incorporating translation-equivariance with no loss of information: the stationary versions of the RCNP model

---

#### Page 23

Table S4: Synthetic (Matérn- $\frac{5}{2}$ ). Comparison of the interpolation (INT) and out-of-input-distribution (OOID) performance of our RCNP models with different CNP baselines on Matérn- $\frac{5}{2}$ function with varying input dimensions. We use normalized Kullback-Leibler divergences as our metric and show mean and (standard deviation) obtained from 10 runs with different seeds. Trivial refers to a model that predicts a Gaussian distribution utilizing the empirical mean and standard deviation derived from the context outputs. "F" denotes failed attempts that yielded very bad results. Missing entries could not be run. Statistically significantly best results are bolded.

|  | $d_{x}=1$ |  | $d_{x}=2$ |  | $d_{x}=3$ |  | $d_{x}=5$ |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | INT | OOID | INT | OOID | INT | OOID | INT | OOID |
| RCNP (sta) | 0.25 (0.00) | 0.25 (0.00) | 0.30 (0.00) | 0.30 (0.00) | 0.39 (0.00) | 0.39 (0.00) | 0.35 (0.00) | 0.35 (0.00) |
| RGNP (sta) | 0.01 (0.00) | 0.01 (0.00) | 0.03 (0.00) | 0.03 (0.00) | 0.05 (0.00) | 0.05 (0.00) | 0.11 (0.00) | 0.11 (0.00) |
| FullRCNP (iso) | 0.25 (0.00) | 0.25 (0.00) | 0.30 (0.00) | 0.30 (0.00) | 0.32 (0.00) | 0.32 (0.00) | 0.29 (0.00) | 0.29 (0.00) |
| FullRGNP (iso) | 0.03 (0.00) | 0.03 (0.00) | 0.09 (0.00) | 0.09 (0.00) | 0.16 (0.00) | 0.16 (0.00) | 0.21 (0.00) | 0.21 (0.00) |
| CoreCNP | 0.24 (0.00) | 0.24 (0.00) | 0.26 (0.00) | 0.26 (0.00) | - | - | - | - |
| CoreGNP | 0.00 (0.00) | 0.00 (0.00) | 0.01 (0.00) | 0.01 (0.00) | - | - | - | - |
| CNP | 0.29 (0.00) | F | 0.39 (0.00) | 6.75 (2.72) | 0.46 (0.00) | 1.75 (0.42) | 0.47 (0.00) | 0.93 (0.02) |
| GNP | 0.02 (0.00) | 2.96 (1.77) | 0.07 (0.00) | 1.86 (0.26) | 0.11 (0.00) | 1.23 (0.17) | 0.19 (0.00) | 0.62 (0.02) |
| Trivial | 1.04 (0.00) | 1.04 (0.00) | 1.06 (0.00) | 1.06 (0.00) | 0.98 (0.00) | 0.98 (0.00) | 0.79 (0.00) | 0.79 (0.00) |

Table S5: Synthetic (weakly-periodic). Comparison of the interpolation (INT) and out-of-inputdistribution (OOID) performance of our RCNP models with different CNP baselines on weaklyperiodic function with varying input dimensions. We use normalized Kullback-Leibler divergences as our metric and show mean and (standard deviation) obtained from 10 runs with different seeds. Trivial refers to a model that predicts a Gaussian distribution utilizing the empirical mean and standard deviation derived from the context outputs. "F" denotes failed attempts that yielded very bad results. Missing entries could not be run. Statistically significantly best results are bolded.

|  | $d_{x}=1$ |  | $d_{x}=2$ |  | $d_{x}=3$ |  | $d_{x}=5$ |  | $d_{x}=10$ |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | INT | OOID | INT | OOID | INT | OOID | INT | OOID | INT | OOID |
| RCNP (sta) | 0.24 (0.00) | 0.24 (0.00) | 0.24 (0.00) | 0.24 (0.00) | 0.28 (0.00) | 0.28 (0.00) | 0.31 (0.00) | 0.31 (0.00) | 0.31 (0.00) | 0.31 (0.00) |
| RGNP (sta) | 0.03 (0.00) | 0.03 (0.00) | 0.05 (0.00) | 0.05 (0.00) | 0.05 (0.00) | 0.05 (0.00) | 0.08 (0.00) | 0.08 (0.00) | 0.11 (0.00) | 0.11 (0.00) |
| CoreCNP | 0.21 (0.00) | 0.21 (0.00) | 0.20 (0.00) | 0.20 (0.00) | - | - | - | - | - | - |
| CoreGNP | 0.01 (0.00) | 0.01 (0.00) | 0.02 (0.00) | 0.02 (0.00) | - | - | - | - | - | - |
| CNP | 0.31 (0.00) | 2.88 (0.91) | 0.39 (0.00) | 1.81 (0.43) | 0.39 (0.00) | 1.58 (0.50) | 0.42 (0.00) | 2.20 (0.81) | 0.75 (0.00) | 1.03 (0.11) |
| GNP | 0.06 (0.00) | F | 0.07 (0.00) | 2.57 (0.76) | 0.08 (0.01) | 1.47 (0.27) | 0.11 (0.01) | 0.62 (0.04) | 0.22 (0.01) | 0.49 (0.05) |
| Trivial | 0.78 (0.00) | 0.78 (0.00) | 0.81 (0.00) | 0.81 (0.00) | 0.80 (0.00) | 0.80 (0.00) | 0.77 (0.00) | 0.77 (0.00) | 0.76 (0.00) | 0.76 (0.00) |

deliver performance comparable to the stationary FullRCNP models. Conversely, since the RCNP model is not context-preserving for rigid transformations, the isotropic versions of the RCNP model exhibit inferior performance in comparison to the isotropic FullRCNPs.

# E Details of Bayesian optimization experiments 

We report here details and additional results for the Bayesian optimization experiments from Section 5.2 of the main text. We evaluated our proposed approach on a common synthetic optimization test function, the Hartmann function in its three and six dimensional versions [40, p.185]. ${ }^{1}$ Each of two test functions is evaluated on a $d_{x}$-dimensional hypercube $\mathcal{X}=[0,1]^{d_{x}}$.

## E. 1 Models

We compared our proposals, RCNP and RGNP, against CNP and GNP as well as their attentive variants ACNP and AGNP. A GP with a Matérn- $\frac{5}{2}$ kernel additionally served as the "gold standard" to be compared against. We mostly followed the architectural choices discussed in Section D.1. However, we kept the number of hidden units per layer fixed at 256 and the relational encoding dimension at 128 , irrespective of the dimensionality $d_{x}$. The models were trained with a learning rate of $3 \cdot 10^{-4}$ over up to 500 epochs with $2^{14}$ datasets in each epoch. A validation set with $2^{12}$ datasets was used to track the training progress and early stopping was performed if the validation score had not improved for 150 epochs.

[^0]
[^0]:    ${ }^{1}$ See https://www.sfu.ca/ ssurjano/optimization.html for their precise definitions and minima.

---

#### Page 24

Table S6: Synthetic (sawtooth). Comparison of the interpolation (INT) and out-of-input-distribution (OOID) performance of our RCNP models with different CNP baselines on sawtooth function with varying input dimensions. We use normalized log-likelihoods as our metric and show mean and (standard deviation) obtained from 10 runs with different seeds. Trivial refers to a model that predicts a Gaussian distribution utilizing the empirical mean and standard deviation derived from the context outputs. "F" denotes failed attempts that yielded very bad results. Missing entries could not be run. Statistically significantly best results are bolded.

|  | $d_{x}=1$ |  | $d_{z}=2$ |  | $d_{z}=3$ |  | $d_{x}=5$ |  | $d_{z}=10$ |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | INT | OOID | INT | OOID | INT | OOID | INT | OOID | INT | OOID |
| RCNP (sta) | 3.03 (0.06) | 3.04 (0.06) | 1.73 (0.03) | 1.74 (0.03) | 0.85 (0.01) | 0.85 (0.01) | 0.44 (0.00) | 0.44 (0.00) | 0.75 (0.00) | 0.75 (0.00) |
| RCNP (sta) | 3.98 (0.09) | 3.90 (0.10) | 2.13 (0.33) | 2.13 (0.32) | 1.09 (0.01) | 1.09 (0.01) | 1.13 (0.05) | 1.13 (0.05) | 1.33 (0.03) | 1.32 (0.03) |
| CoreCNP | 3.64 (0.04) | 3.64 (0.04) | 3.66 (0.04) | 3.66 (0.04) | - | - | - | - | - | - |
| CoreGNP | 3.94 (0.11) | 3.97 (0.08) | 4.11 (0.03) | 3.99 (0.11) | - | - | - | - | - | - |
| CNP | 2.25 (0.02) | F | 1.15 (0.45) | $-3.27(4.72)$ | 0.36 (0.28) | $-0.37(0.12)$ | $-0.03(0.10)$ | $-0.22(0.03)$ | 0.27 (0.00) | $-2.29(0.67)$ |
| GNP | 0.83 (0.04) | F | 1.04 (0.09) | F | 0.23 (0.13) | F | 0.02 (0.05) | F | 0.03 (0.04) | F |
| Trivial | $-0.27(0.00)$ | $-0.27(0.00)$ | $-0.19(0.00)$ | $-0.19(0.00)$ | $-0.19(0.00)$ | $-0.19(0.00)$ | $-0.18(0.00)$ | $-0.18(0.00)$ | $-0.14(0.00)$ | $-0.14(0.00)$ |

Table S7: Synthetic (mixture). Comparison of the interpolation (INT) and out-of-input-distribution (OOID) performance of our RCNP models with different CNP baselines on mixture function with varying input dimensions. We use normalized log-likelihoods as our metric and show mean and (standard deviation) obtained from 10 runs with different seeds. Trivial refers to a model that predicts a Gaussian distribution utilizing the empirical mean and standard deviation derived from the context outputs. "F" denotes failed attempts that yielded very bad results. Missing entries could not be run. Statistically significantly best results are bolded.

|  | $d_{x}=1$ |  | $d_{z}=2$ |  | $d_{z}=3$ |  | $d_{x}=5$ |  | $d_{z}=10$ |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | INT | OOID | INT | OOID | INT | OOID | INT | OOID | INT | OOID |
| RCNP (sta) | 0.20 (0.01) | 0.20 (0.01) | 0.17 (0.00) | 0.17 (0.00) | $-0.10(0.00)$ | $-0.10(0.00)$ | $-0.31(0.03)$ | $-0.31(0.03)$ | $-0.32(0.00)$ | $-0.32(0.00)$ |
| RCNP (sta) | 0.34 (0.03) | 0.34 (0.03) | 0.46 (0.02) | 0.46 (0.02) | 0.37 (0.01) | 0.37 (0.01) | 0.84 (0.02) | 0.84 (0.02) | $-0.11(0.02)$ | $-0.11(0.02)$ |
| CoreCNP | 0.38 (0.02) | 0.38 (0.02) | 0.63 (0.01) | 0.63 (0.01) | - | - | - | - | - | - |
| CoreGNP | 0.49 (0.15) | 0.49 (0.15) | 0.87 (0.03) | 0.87 (0.03) | - | - | - | - | - | - |
| CNP | 0.01 (0.01) | F | $-0.22(0.01)$ | $-7.16(3.61)$ | $-0.57(0.11)$ | $-2.55(1.15)$ | $-0.72(0.08)$ | $-1.71(0.55)$ | $-0.88(0.00)$ | $-1.15(0.06)$ |
| GNP | 0.17 (0.01) | F | $-0.01(0.01)$ | F | $-0.17(0.00)$ | $-0.67(0.05)$ | $-0.32(0.00)$ | $-0.72(0.03)$ | $-0.38(0.10)$ | $-0.75(0.09)$ |
| Trivial | $-0.78(0.00)$ | $-0.78(0.00)$ | $-0.81(0.00)$ | $-0.81(0.00)$ | $-0.84(0.00)$ | $-0.84(0.00)$ | $-0.86(0.00)$ | $-0.86(0.00)$ | $-0.87(0.00)$ | $-0.87(0.00)$ |

# E. 2 Data 

The datasets used to train the neural process models were generated based on synthetic functions sampled from a Gaussian process model with kernel $k$, with context and target set sizes as described in Section D.2. However, while the datasets used in the synthetic regression experiments were generated using fixed kernel setups, the current experiment explored sampling from a set of base kernels. To explore four different training regimes for each $d_{x}=\{3,6\}$, we changed how the GP kernel $k(z)=\theta k_{0}(z / \ell)$ with an output scale $\theta$ and a lengthscale $\ell$ was chosen as follows.
(i) Matérn-fixed. The kernel $k$ is a Matérn- $\frac{5}{2}$ kernel with fixed $\ell=\sqrt{d_{x}} / 4$ and $\theta=1$.
(ii) Matérn-sampled. The kernel $k$ is a Matérn- $\frac{5}{2}$ kernel, with

$$
\ell \sim \mathcal{L N}\left(\log \left(\sqrt{d_{x}} / 4\right), 0.5^{2}\right), \quad \theta \sim \mathcal{L N}(0,1)
$$

where $\mathcal{L N}$ is a log-Normal distribution, i.e., $\log (\ell)$ follows a standard-Normal distribution.
(iii) Kernel-single. The kernel $k$ is sampled as

$$
\begin{aligned}
& k_{0} \sim \mathcal{U}\left(\left\{\text { EQ, Matérn- } \frac{1}{2}, \text { Matérn- } \frac{3}{2}, \text { Matérn- } \frac{5}{2}\right\}\right) \\
& \ell \sim \mathcal{L N}\left(\log \left(\sqrt{d_{x}} / 4\right), 0.5^{2}\right) \\
& \theta \sim \mathcal{L N}(0,1)
\end{aligned}
$$

where $\mathcal{U}$ is a uniform distribution over the set.
(iv) Kernel-multiple. The kernel $k$ is sampled as

$$
\begin{aligned}
k_{1}, k_{2}, k_{3}, k_{4} & \sim \mathcal{U}\left(\left\{\text { NA, EQ, Matérn- } \frac{1}{2}, \text { Matérn- } \frac{3}{2}, \text { Matérn- } \frac{5}{2}\right\}\right) \\
\ell_{i} & \sim \mathcal{L N}\left(\log \left(\sqrt{d_{x}} / 4\right), 0.5^{2}\right), \quad i \in\{1, \ldots 4\} \\
\theta_{j} & \sim \mathcal{L N}(0,1), \quad j \in\{1,2\} \\
k & =\theta_{1} k_{1}\left(\ell_{1}\right) \cdot k_{2}\left(\ell_{2}\right)+\theta_{2} k_{3}\left(\ell_{3}\right) \cdot k_{4}\left(\ell_{4}\right)
\end{aligned}
$$

---

#### Page 25

Table S8: An ablation study conducted on RCNPs and FullRCNPs using synthetic regression data. We evaluated both the 'stationary' (sta) version, which employs the difference comparison function, and the 'isotropic' (iso) version, utilizing the distance comparison function, across both RCNP and FullRCNP models. The table provides metrics for both the interpolation (INT) and out-of-inputdistribution (OOID) performance. We show mean and (standard deviation) obtained from 10 runs with different seeds. Statistically significantly best results are bolded.

|  |  | EQ |  |  | Matĕrn- $\frac{5}{8}$ |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  |  | KL divergence( $\downarrow$ ) |  |  | KL divergence( $\downarrow$ ) |  |
|  |  | $d_{x}=2$ | $d_{x}=3$ | $d_{x}=5$ | $d_{x}=2$ | $d_{x}=3$ | $d_{x}=5$ |
|  | RCNP (sta) | 0.26 (0.00) | 0.40 (0.01) | 0.45 (0.00) | 0.30 (0.00) | 0.39 (0.00) | 0.35 (0.00) |
|  | RGNP (sta) | 0.03 (0.00) | 0.05 (0.00) | 0.11 (0.00) | 0.03 (0.00) | 0.05 (0.00) | 0.11 (0.00) |
|  | RCNP (iso) | 0.32 (0.00) | 0.41 (0.00) | 0.46 (0.00) | 0.34 (0.00) | 0.39 (0.00) | 0.36 (0.00) |
| $\stackrel{\rightharpoonup}{E}$ | RGNP (iso) | 0.12 (0.00) | 0.17 (0.00) | 0.30 (0.00) | 0.12 (0.00) | 0.17 (0.00) | 0.24 (0.00) |
|  | FullRCNP (sta) | 0.25 (0.00) | 0.30 (0.00) | 0.36 (0.00) | 0.30 (0.00) | 0.31 (0.00) | 0.29 (0.00) |
|  | FullRGNP (sta) | 0.03 (0.00) | 0.04 (0.00) | 0.09 (0.00) | 0.03 (0.00) | 0.04 (0.00) | 0.10 (0.00) |
|  | FullRCNP (iso) | 0.26 (0.00) | 0.31 (0.00) | 0.35 (0.00) | 0.30 (0.00) | 0.32 (0.00) | 0.29 (0.00) |
|  | FullRGNP (iso) | 0.08 (0.00) | 0.14 (0.00) | 0.25 (0.00) | 0.09 (0.00) | 0.16 (0.00) | 0.21 (0.00) |
|  | RCNP (sta) | 0.26 (0.00) | 0.40 (0.01) | 0.45 (0.00) | 0.30 (0.00) | 0.39 (0.00) | 0.35 (0.00) |
|  | RGNP (sta) | 0.03 (0.00) | 0.05 (0.00) | 0.11 (0.00) | 0.03 (0.00) | 0.05 (0.00) | 0.11 (0.00) |
|  | RCNP (iso) | 0.32 (0.00) | 0.41 (0.00) | 0.46 (0.00) | 0.34 (0.00) | 0.39 (0.00) | 0.36 (0.00) |
|  | RGNP (iso) | 0.12 (0.00) | 0.17 (0.00) | 0.30 (0.00) | 0.12 (0.00) | 0.17 (0.00) | 0.24 (0.00) |
|  | FullRCNP (sta) | 0.25 (0.00) | 0.30 (0.00) | 0.36 (0.00) | 0.30 (0.00) | 0.31 (0.00) | 0.29 (0.00) |
|  | FullRGNP (sta) | 0.03 (0.00) | 0.04 (0.00) | 0.09 (0.00) | 0.03 (0.00) | 0.04 (0.00) | 0.10 (0.00) |
|  | FullRCNP (iso) | 0.26 (0.00) | 0.31 (0.00) | 0.35 (0.00) | 0.30 (0.00) | 0.32 (0.00) | 0.29 (0.00) |
|  | FullRGNP (iso) | 0.08 (0.00) | 0.14 (0.00) | 0.25 (0.00) | 0.09 (0.00) | 0.16 (0.00) | 0.21 (0.00) |

where NA indicates that no kernel is chosen, i.e., a term in the sum can consist of a single kernel or be missing completely. This setup is based on the kernel learning setup in [35].

See Section D. 2 for the precise definition of each kernel. Note that each setting is strictly more general than the one before and entails its predecessor as a special case. Setting (iv) was reported in the main text.

# E. 3 Bayesian optimization 

After training, the neural process models were evaluated in a Bayesian optimization task. Starting from a common set of five random initial observations, $\mathbf{x}_{0} \in \mathcal{U}(0,1)^{d_{x}}$, each optimization run performed fifty steps of requesting queries by maximizing the expected improvement acquisition function:

$$
\mathrm{EI}(x)=\mathbb{E}\left[\max \left(f(x)-f_{\text {best }}, 0\right)\right]
$$

where $f(\cdot)$ is the negative target function (as we wanted to minimize $f$ ), $f_{\text {best }}$ is the current observed optimum, and the expectation is calculated based on the neural process or GP model predictions. The acquisition function was optimized via PyCMA [17], the Python implementation of Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [16]. Each CMA-ES run was initialized from the best acquisition function value estimated over a random subset of 100 locations from the hypercube. After each step, the newly queried point and its function evaluation were used to update the surrogate model. For neural process models, this means that the point was added to the context set, whereas for the the GP model, this means that the model was also reinitialized and its hyperparameters refit via type-II maximum likelihood estimation. The neural process model parameters were kept fixed without retraining throughout the Bayesian optimization task.

The results reported in Figure 2 in the main paper were computed using a single pretrained neural process model and ten random Bayesian optimization restarts for each model. For the full results presented here, each neural process model was pretrained three times and used as a surrogate model in ten optimization runs initialized with different sets of five observations, giving us a total of thirty optimization runs with each model. See Figure S1 and Figure S2 for the results and their respective discussion.

---

#### Page 26

# E. 4 Computation time 

Each model was trained on a single GPU, requiring about $90-120 \mathrm{~min}$ per run. The subsequent CMA-ES optimization and analysis took about $60-90 \mathrm{~min}$ running on a CPU with eight cores per experiment. The total runtime was approximately 280 h of GPU time and approximately 7 h on the CPUs.
> **Image description.** This image contains four line graphs arranged in a 2x2 grid, each displaying error (minimum f(x_t) - f_min) versus the number of queries. The graphs compare the performance of different models: GP (Gaussian Process), CNP (Conditional Neural Process), GNP (Generative Neural Process), ACNP (Attentive CNP), AGNP (Attentive GNP), RCNP (Relational CNP), and RGNP (Relational GNP). Each graph has the same x and y axis labels and ranges.
> 
> The x-axis is labeled "number of queries" and ranges from 0 to 50. The y-axis is labeled "error |min f(x_t) - f_min|" and ranges from 0.0 to 2.5.
> 
> Each graph contains multiple lines, each representing a different model. The models are distinguished by line style and color. The GP model is represented by a dotted black line. The CNP model is represented by a dashed blue line. The GNP model is represented by a solid blue line. The ACNP model is represented by a solid orange line. The AGNP model is represented by a dashed green line. The RCNP model is represented by a solid red line (though this is not visible in the graphs). The RGNP model is represented by a solid green line. Shaded regions around some of the lines indicate the variance or uncertainty associated with the model's performance.
> 
> The titles of the graphs are:
> 1. Top left: "Matérn-fixed"
> 2. Top right: "Matérn-sampled"
> 3. Bottom left: "Kernel-single"
> 4. Bottom right: "Kernel-multiple"
> 
> A legend in the bottom right graph identifies the models and their corresponding line styles and colors.


Figure S1: Ablations on Hartmann 3d. GNP struggled throughout all four pretraining scenarios, while its attentive and relational variants (AGNP, RGNP) were able to benefit from the additional flexibility, greatly improving their performance. CNP struggled to learn as the flexibility in the context sets increases, while its attentive and relational counterparts adapted themselves and were able to converge to optimal performance even in the most restrictive pretraining scenario.

## F Details of Lotka-Volterra model experiments

We report here details and additional results for the Lotka-Volterra model from Section 5.3 of the main text.

## F. 1 Models

The RCNP architecture used in this experiment is modeled on the multi-output CNP architecture proposed in previous work [3, Appendix F]. The multi-output CNP encodes the context set associated with each output as a 256-dimensional vector and concatenates the vectors. The encoder network used in this model is an MLP with three hidden layers and 256 hidden units per layer, and the decoder network is an MLP with six hidden layers and 512 hidden units per layer. The RCNP model replicates this architecture by encoding the targets with respect to the context set associated with each output as 256-dimensional vectors, concatenating the vectors, and using encoder and decoder networks that are the same as the encoder and decoder networks used in the CNP model. We note that the encoder

---

#### Page 27

> **Image description.** This image contains four line graphs arranged in a 2x2 grid, comparing the performance of different neural process models. Each graph plots "error |min f(x_t) - f_min|" on the y-axis against "number of queries" on the x-axis.
> 
> Here's a breakdown of the common elements and differences between the graphs:
> 
> *   **Common Elements:**
>     *   **Axes:** Each graph has a y-axis labeled "error |min f(x_t) - f_min|" ranging from 0.0 to 3.0, and an x-axis labeled "number of queries" ranging from 0 to 80.
>     *   **Lines:** Each graph displays multiple lines, each representing a different model: GP (dotted black), CNP (solid blue), GNP (dashed dark blue), ACNP (solid orange), AGNP (dashed orange), RCNP (solid green), and RGNP (dashed green).
>     *   **Shaded Regions:** Each line has a shaded region around it, presumably indicating the standard deviation or confidence interval. The colors of the shaded regions correspond to the line colors.
>     *   **Overall Trend:** In each graph, most lines show a decreasing trend, indicating that the error decreases as the number of queries increases.
> 
> *   **Differences:**
>     *   **Titles:** The graphs are titled differently: "Matérn-fixed", "Matérn-sampled", "Kernel-single", and "Kernel-multiple". These titles likely refer to different configurations or settings of the experiment.
>     *   **Line Behavior:** The specific behavior of each line varies between the graphs, indicating that the performance of the models depends on the experimental setup. For example, the CNP model (solid blue line) consistently performs poorly, staying at a high error level across all graphs, while the GP model (dotted black line) consistently performs the best, reaching the lowest error levels.
>     *   **Legend:** The "Kernel-multiple" graph has a legend on the right side, listing the models (GP, CNP, GNP, ACNP, AGNP, RCNP, RGNP) and their corresponding line styles and colors.
> 
> In summary, the image presents a comparative analysis of different neural process models under varying experimental conditions, visualizing their performance in terms of error reduction with an increasing number of queries.


Figure S2: Ablations on Hartmann 6d. As in the 3d case (see Figure S1), CNP and GNP struggled to learn at all. Similarly, the advanced attentive and relational CNP models converged to their optimum even with the most restrictive prior kernel set. As the flexibility increased, RGNP was able to benefit the most and improved both upon AGNP as well as their conditional NP variants (ACNP, RCNP).
network dimensions are the same as used in the synthetic regression experiments (Section D.1) while the decoder network has more hidden units per layer in the multi-output architectures. The encoder and decoder networks use ReLU activation functions.

The models evaluated in these experiments include RCNP, CNP, ACNP, and ConvCNP as well as RGNP, GNP, AGNP, and ConvGNP. RCNP and RGNP models share the architecture described above while the other models are evaluated with the architectures proposed in previous work [3]. However, in contrast to previous work, where CNP, GNP, ACNP, and AGNP models encoded the context associated with each output with a separate encoder, here we used a shared encoder in RCNP, RGNP, CNP, GNP, ACNP, and AGNP. We made this choice based on preliminary experiments carried out with synthetic multi-output data, where we observed that using a shared encoder improved both RCNP and CNP results. We note that all models are extended with a nonlinear transformation to take into account that the population counts modeled in this experiment are positive [3].
The neural process models were trained with simulated data and evaluated with simulated and real data. The models were trained for 200 epochs with $2^{14}$ datasets in each epoch and learning rate $10^{-4}$. The validation sets used in training included $2^{12}$ datasets also generated based on the simulated data. The evaluation sets generated based on simulated data included $2^{12}$ datasets each and the evaluation sets generated based on real data $2^{8}$ dataset each, and both simulated and real data were used to generate three separate evaluation sets to represent the three task types considered in this experiment: interpolation, forecasting, and reconstruction.

---

#### Page 28

# F. 2 Data 

The datasets used in this task included simulated and real datasets. The simulated datasets were constructed based on simulated predator and prey time series generated using stochastic LotkaVolterra equations. The Lotka-Volterra equations introduced independently by Lotka [28] and Volterra [42] model the joint evolution between prey and predator populations as

$$
\left\{\begin{array}{l}
\dot{u}=\alpha u-\beta u v \\
\dot{v}=\delta u v-\gamma v
\end{array}\right.
$$

where $u$ denotes the prey, $v$ denotes the predators, $\alpha$ is the birth rate of the prey, $\beta$ is the rate at which predators kill prey, $\delta$ the rate at which predators reproduce when killing prey, and $\gamma$ is the death rate of the predators; the dot denotes a temporal derivative, i.e. $\dot{u} \equiv \frac{\mathrm{~d}}{\mathrm{~d} t} u$. In this work, we generated simulated data with a stochastic version proposed by [3].

$$
\left\{\begin{array}{l}
\mathrm{d} u=\alpha u \mathrm{~d} t-\beta u v \mathrm{~d} t+u^{1 / 6} \sigma \mathrm{~d} W^{u} \\
\mathrm{~d} v=\delta u v \mathrm{~d} t-\gamma v \mathrm{~d} t+v^{1 / 6} \sigma \mathrm{~d} W^{v}
\end{array}\right.
$$

where $W^{u}$ and $W^{v}$ are two independent Brownian motions that add noise to the population trajectories and $\sigma>0$ controls the noise magnitude.

We simulated trajectories from the above equation using the code released with the previous work [3]. The parameters used in each simulation were sampled from: $\alpha \sim \mathcal{U}(0.2,0.8), \beta \sim \mathcal{U}(0.04,0.08)$, $\delta \sim \mathcal{U}(0.8,1.2), \gamma \sim \mathcal{U}(0.04,0.08)$, and $\sigma \sim \mathcal{U}(0.5,10)$. The initial population sizes were sampled from $u, v \sim \mathcal{U}(5,100)$, and the simulated trajectories were also scaled by a factor $r \sim \mathcal{U}(1,5)$. To construct a dataset, 150-250 observations were sampled from the simulated prey time series and 150-250 observations from the simulated predator series. The selected observations were then divided into context and target observations based on the selected task type which was fixed in the evaluation sets and selected at random in the training and validation data.
The datasets representing interpolation, forecasting, or reconstruction tasks were created based on the selected simulated observations as follows. To create an interpolation task, 100 predator and 100 prey observations, selected at random, were included in the target set, and the rest were included in the context set. To create a forecasting task, a time point was selected at random between 25 and 75 years, and all observations prior to the selected time point were included in the context set and the rest were included in the target set. To create a reconstruction task, a random choice was made between prey and predators such that the observations from the selected population were all included in the context set while the observations from the other population were divided between the context and target sets by choosing at random a time point between 25 and 75 years and including all observations prior to the selected time point in the context set and the rest in the target set.

The real datasets used to evaluate the neural process models in the sim-to-real tasks were generated based on the famous hare-lynx data [32]. The hare and lynx time series include 100 observations at regular intervals. The observations were divided into context and target sets based on selected task types, and the same observations were used to generate separate evaluation sets for each task. The forecasting and reconstruction task was created with the same approach that was used with simulated data, while the interpolation task was created by choosing at random 1-20 predator observations and 1-20 prey observations that were included in the target set and including the rest in the context set.

## F. 3 Full results

We evaluate and compare the neural process models based on normalized log-likelihood scores calculated based on each evaluation set and across training runs. We compare RCNP, CNP, ACNP, and ConvCNP models in Table S9 (a) and RGNP, GNP, AGNP, and ConvGNP models with 32 basis functions in Table S9 (b) and with 64 basis functions in Table S9 (c). We observe that the best model depends on the task, and comparison across the tasks indicates that RCNP or RGNP are not consistently better or worse than their convolutional and attentive counterparts. This is not the case with CNP and GNP, and we observe that the other models including RCNP and RGNP are almost always substantially better than CNP or GNP. The models with 64 basis functions performed on average better than the models with 32 basis functions, but we observed that some RGNP, GNP, and AGNP training runs did not converge and most GNP runs did not complete 200 epochs due to numerical errors when 64 basis functions were used. We therefore reported results with 32 basis functions in the main text.

---

#### Page 29

Table S9: Normalized log-likelihood scores in the Lotka-Volterra experiments (higher is better). We compared models that predict (a) mean and variance and models that also predict a low-rank covariance matrix with (b) 32 basis functions or (c) 64 basis functions. The mean and (standard deviation) reported for each model are calculated based on 10 training outcomes evaluated with the same simulated (S) and real (R) learning tasks. The tasks include interpolation (INT), forecasting (FOR), and reconstruction (REC). Statistically significantly best results are bolded. RCNP and RGNP models perform on par with their convolutional and attentive counterparts, and nearly always substantially better than the standard CNPs and GNPs, respectively.

|  |  | INT (S) | FOR (S) | REC (S) | INT (R) | FOR (R) | REC (R) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| (a) | RCNP | $-3.57(0.02)$ | $-4.85(0.00)$ | $-4.20(0.01)$ | $-4.24(0.02)$ | $-4.83(0.03)$ | $-4.55(0.05)$ |
|  | ConvCNP | $-3.47(0.01)$ | $-4.85(0.00)$ | $-4.06(0.00)$ | $-4.21(0.04)$ | $-5.01(0.02)$ | $-4.75(0.05)$ |
|  | ACNP | $-4.04(0.06)$ | $-4.87(0.01)$ | $-4.36(0.03)$ | $-4.18(0.05)$ | $-4.79(0.03)$ | $-4.48(0.02)$ |
|  | CNP | $-4.78(0.00)$ | $-4.88(0.00)$ | $-4.86(0.00)$ | $-4.74(0.01)$ | $-4.81(0.02)$ | $-4.70(0.01)$ |
| (b) | RGNP | $-3.51(0.01)$ | $-4.27(0.00)$ | $-3.76(0.00)$ | $-4.31(0.06)$ | $-4.47(0.03)$ | $-4.39(0.11)$ |
|  | ConvGNP | $-3.46(0.00)$ | $-4.30(0.00)$ | $-3.67(0.01)$ | $-4.19(0.02)$ | $-4.61(0.03)$ | $-4.62(0.11)$ |
|  | AGNP | $-4.12(0.17)$ | $-4.35(0.09)$ | $-4.05(0.19)$ | $-4.33(0.15)$ | $-4.48(0.06)$ | $-4.29(0.10)$ |
|  | GNP | $-4.62(0.03)$ | $-4.38(0.04)$ | $-4.36(0.07)$ | $-4.79(0.03)$ | $-4.72(0.04)$ | $-4.72(0.04)$ |
| (c) | RGNP | $-3.54(0.06)$ | $-4.15(0.10)$ | $-3.75(0.10)$ | $-4.30(0.04)$ | $-4.44(0.02)$ | $-4.38(0.10)$ |
|  | ConvGNP | $-3.46(0.00)$ | $-4.15(0.01)$ | $-3.66(0.01)$ | $-4.20(0.03)$ | $-4.57(0.05)$ | $-4.64(0.12)$ |
|  | AGNP | $-4.06(0.10)$ | $-4.16(0.04)$ | $-3.95(0.11)$ | $-4.34(0.10)$ | $-4.41(0.06)$ | $-4.30(0.08)$ |
|  | GNP | $-4.63(0.03)$ | $-4.36(0.05)$ | $-4.40(0.05)$ | $-4.82(0.01)$ | $-4.75(0.04)$ | $-4.75(0.03)$ |

# F. 4 Computation time 

When the models were trained on a single GPU, each training run with RCNP and RGNP took around $6.5-7$ hours and each training run with the reference models around $2-3$ hours depending on the model. The results presented in this supplement and the main text required 120 training runs, and $10-15$ training runs had been carried out earlier to confirm as a sanity check our ability to reproduce the results presented in previous work. The multi-output models used in this experiment were additionally studied in preliminary experiments using synthetic regression data. The computation time used in all the experiments is included in the estimated total reported in Section C.2.

## G Details of Reaction-Diffusion model experiments

We report here details and additional results for the Reaction-Diffusion model from Section 5.4 of the main text.

## G. 1 Models

We compared RCNP, RGNP, ACNP, AGNP, CNP, and GNP models trained using simulated data with input dimensions $d_{x}=\{3,4\}$. We mostly used the architectures described in Section D.1, but since less training data was used in this experiment, we reduced the number of hidden layers in the decoder network to 4 . The models were trained for 100 epochs with $2^{10}$ datasets in each epoch and learning rate $10^{-4}$. The validation sets used in training included $2^{8}$ datasets and the evaluation sets used to compare the models in completion and forecasting tasks included $2^{8}$ datasets each.

## G. 2 Data

We generated simulated data using a simplified equation inspired by the cancer evolution equation from Gatenby and Gawlinski [13]. The equation involves three quantities: the healthy cells $(\mathbf{z}, t) \mapsto u(\mathbf{z}, t)$, the cancerous cells $(\mathbf{z}, t) \mapsto v(\mathbf{z}, t)$, and an acidity measure $(\mathbf{z}, t) \mapsto w(\mathbf{z}, t)$, with

---

#### Page 30

$t \in[0, T]$ and $\mathbf{z} \in E \subset \mathbb{R}^{2}$. Their temporal and spatial dynamics are described by:

$$
\left\{\begin{array}{l}
\dot{u}=r_{u} u\left(1-u / k_{u}\right)-d_{u} w \\
\dot{v}=r_{v} v\left(1-v / k_{v}\right)+D_{v} \nabla^{2} v \\
\dot{w}=r_{w} v-d_{w} w+D_{w} \nabla^{2} v
\end{array}\right.
$$

where $r_{u}, r_{v}$, and $r_{w}$ are apparition rates, $k_{u}$ and $k_{v}$ control the maximum for the number of healthy and cancerous cells, $d_{u}$ and $d_{w}$ are death rates relative to other species, and $D_{v}$ and $D_{w}$ are dispersion rates.
We simulate trajectories from Eq. S8 on a discretized space-time grid using the SpatialPy simulator [6]. For the purpose of our experiment, we selected realistic parameter ranges that would also produce quick evolution. The selected parameter ranges are therefore close to [13], but for example the diffusion rates were increased. The parameters for the simulations are sampled according to the following distributions: $k_{u}=10, k_{v}=100, r_{u} \sim \mathcal{U}(0.0027,0.0033), r_{v} \sim \mathcal{U}(0.027,0.033)$, $r_{w} \sim \mathcal{U}(1.8,2.2), D_{v} \sim U(0.0009,0.0011), D_{w} \sim \mathcal{U}(0.009,0.011), d_{w}=0, d_{u} \sim \mathcal{U}(0.45,0.55)$. These parameters lead to a relatively slowly-growing number of cancerous cells, with a fast decay in healthy cells due to acid spread, as depicted in Figure S3. The process starts with the healthy cells count $u$ constant across the whole space, cancerous cells count $v$ zero except for one cancerous cell at a uniformly random position, and acid count $w$ zero across the whole space.
> **Image description.** The image shows a set of simulations depicting the evolution of healthy cells, cancerous cells, and acid count over time in a spatial grid. The image is organized into three rows, each representing a different quantity: "Sane cells count," "Cancer cells count," and "Acid count" from top to bottom. Each row contains three subplots corresponding to time points t=0, t=1, and t=2.
> 
> Each subplot represents a 11x11 spatial grid, with axes labeled from 0 to 10. The color of each cell in the grid indicates the quantity's value at that location, according to a color scale on the right of the first and second rows, and on the right of the third row.
> 
> *   **Sane cells count (Top Row):**
>     *   At t=0, the grid is uniformly colored in a light peach color, indicating a high initial count of healthy cells across the entire space.
>     *   At t=1, the grid shows a region of darker colors in the center-left, indicating a decrease in healthy cells in that area.
>     *   At t=2, the darker region has expanded, showing further reduction in healthy cells.
> 
> *   **Cancer cells count (Middle Row):**
>     *   At t=0, there is a single light peach colored cell in the center, indicating the initial presence of one cancerous cell. The rest of the grid is dark.
>     *   At t=1, the cancerous cell has multiplied and spread to neighboring cells, indicated by the appearance of a cluster of light peach colored cells.
>     *   At t=2, the cluster has grown further, showing the continued spread of cancerous cells.
> 
> *   **Acid count (Bottom Row):**
>     *   At t=0, the grid is uniformly dark, indicating no acid present initially.
>     *   At t=1, a small cluster of light peach colored cells appears, indicating the presence of acid.
>     *   At t=2, the cluster has expanded, showing the spread of acid.
> 
> The color scales on the right of each row indicate the quantity values, ranging from 0 to 10 for the first two rows, and 0 to 4 for the third row.


Figure S3: Example simulation of the Reaction-Diffusion model at times $t=0,1,2$. Each row represent a different quantity in the model (from top to bottom, healthy cells, cancerous cells, and acid). At time $t=0$, the population of healthy cells is 10 at each point of space, and a single point is occupied by additional cancerous cells. After this time, cancerous cells start spreading and slowly multiplicating, while acid spreads faster, killing healthy cells along its way.

---

#### Page 31

We generated a total $5 \cdot 10^{3}$ trajectories on a $11 \times 11$ grid at times $t=0, \ldots, 5$. We used the first $4 \cdot 10^{3}$ trajectories to generate datasets for model training and the other $10^{3}$ to generate datasets for evaluation. The datasets were generated by sampling from the trajectories as follows. First, we extracted the population counts at three consecutive time points from a trajectory and time point selected at random. We then sampled 50-200 context points and 100-200 target points from this subsample to create a dataset representing either a completion or forecasting task. For the completion task, the context and target points were sampled from the subsample at random, and for the forecasting task, all context points were sampled from the first two time points, and all target points from the third time point were included in the subsample. All datasets included in the training and validation data represented the completion task.

Finally, the input and output variables at each context and target point were determined as follows. In the experiments reported in the main text, we considered the case of partial observations, where only the difference $u-v$ is observed. This makes the simulator model a hidden Markov model and thus intractable. Predicting future states of the quantity $u-v$ without running inference and without having access to $w$ is a challenging problem, similar to several application cases of interest. In this appendix, we additionally consider a scenario where we include the acid count in the input variables and take the healthy cells count as the output variable.

# G. 3 Additional results 

A promising scientific application of neural processes is inferring latent variables given other measured quantities. To explore this capability, we attempt to infer the quantity of healthy cells $(u(\mathbf{z}, t))$ given as input the acidity concentration $(w(\mathbf{z}, t))$ for some $\mathbf{z}, t$. For this study, we augmented the inputs to $(\mathbf{z}, t, w(\mathbf{z}, t))$. Notably, the regression problem is not equivariant in the last input dimension, so we encoded $w(\mathbf{z}, t)$ in a non-relational manner also in the RCNP and RGNP models. The results are presented in Table S10. The difference between the completion and forecasting tasks is small in this experiment, and the best results are observed with RGNP in both tasks. AGNP performance is surprisingly poor and will require further investigation.

Table S10: Normalized log-likelihood scores in the additional experiments of the Reaction-Diffusion problem for both tasks (higher is better). Mean and (standard deviation) from 10 training runs evaluated on a separate test dataset.

|  | RCNP | RGNP | ACNP | AGNP | CNP | GNP |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| Completion | $\mathbf{0 . 3 5}(0.12)$ | $\mathbf{0 . 7 4}(0.18)$ | $\mathbf{0 . 2 4}(0.25)$ | $-\mathbf{2 . 2 0}(3.67)$ | $\mathbf{0 . 4 4}(0.17)$ | $\mathbf{0 . 3 2}(0.26)$ |
| Forecasting | $\mathbf{0 . 2 9}(0.11)$ | $\mathbf{0 . 6 3}(0.15)$ | $\mathbf{0 . 3 8}(0.21)$ | $-\mathbf{1 . 7}(3.0)$ | $\mathbf{0 . 4 6}(0.13)$ | $\mathbf{0 . 4 1}(0.20)$ |

## G. 4 Computation time

On GPU, for all methods the training runs of this experiment were particularly fast, each one lasting no more than 5 minutes. We carried out a total of 10 training runs for each method, with a total of 120 training runs. We also performed around 60 exploratory training runs (with longer training time, larger networks, etc.). The total training time comes to less than 15 hours.

## H Additional experiments

In addition to the experimental evaluation in the main paper and their extended discussion in the sections above, we explored several further experimental setups to demonstrate the wide applicability of our proposal. These are (i) an application of the autoregressive structure proposed by [3] to our proposed approach (Section H.1), (ii) a demonstration of how to incorporate rotational equivariance into the model (Section H.2), and (iii) an exploration of the method's performance on two popular image data regression tasks (Section H.3).

## H. 1 Autoregressive CNPs

To further demonstrate the effectiveness of our RCNP architectures, we ran a set of experiments adopting an autoregressive approach to both our RCNP and other CNP models. Autoregressive

---

#### Page 32

CNPs (AR-CNPs) are a new family of CNP models which were recently proposed by [3]. Unlike traditional CNPs, which generate predictions independently for each target point, or GNPs, which produce a joint multivariate normal distribution over the target set, AR-CNPs define a joint predictive distribution autoregressively, leveraging the chain rule of probability. Without changing the CNP architectures and the training procedure, AR-CNPs feed earlier output predictions back into the model autoregressively to predict new points. This adjustment enables the AR-CNPs to capture highly dependent, non-Gaussian predictive distributions, resulting in better performance compared to traditional CNPs.

In this experiment, we compare AR-RCNP, AR-CNP, and AR-ConvCNP on synthetic regression data. The AR approach is applied at the evaluation stage to generate the target predictions autoregressively. When applying AR, the training method for the models remains unchanged. Thus, for this experiment, we use the same models that were trained with synthetic data. Details about the data generation and model configurations can be found in Section D. We chose the translational-equivariant version of RCNP for each task.

The full results are reported in Table S11. As we maintain consistent experimental settings with previous synthetic experiments (Section D), we can directly compare the performance of models without the AR approach (Table S3-S7) to those enhanced with the AR approach. We observe that the AR approach improves RCNP performance in the Gaussian experiments in all dimensions and in the non-Gaussian sawtooth and mixture experiments when $d_{g}>1$. When $d_{g}=1$, AR-RCNP results are comparable to RCNP results in the mixture task and worse in the sawtooth task. We also observe that AR-CNP results are worse than CNP results in most sawtooth and mixture experiments. Overall we can see that AR-RCNPs are better than AR-CNPs across all dimensions and tasks under both INT and OOID conditions, and achieve close performance compared with AR-ConvCNPs under $d_{g}=1$ in the Gaussian experiments.

Table S11: Comparison of the interpolation (INT) and out-of-input-distribution (OOID) performance of AR-RCNP model with other AR baselines on various synthetic regression tasks with varying input dimensions. We show the mean and (standard deviation) obtained from 10 runs with different seeds. "F" denotes failed attempts that yielded very bad results. Our AR-RCNP method performs competitively in low dimension, and scales to higher input dimensions $\left(d_{g}>2\right)$, where AR-ConvCNP is not applicable.
> **Image description.** This image is a table presenting numerical results comparing different models on synthetic regression data.
> 
> The table is organized as follows:
> 
> *   **Columns:** The columns are grouped into five main categories: "EQ", "Matérn-5/2", "Weakly-periodic", "Sawtooth", and "Mixture". Each of these categories is further divided into two sub-columns labeled "INT" and "OOID". Underneath each of the five main categories is a label indicating the type of metric being reported: "KL divergence(↓)" for the first three categories, and "log-likelihood(↑)" for the last two. The arrows indicate whether lower or higher values are better.
> *   **Rows:** The rows are grouped by the value of "d_x", which takes on the values 1, 3, and 5. Within each group, there are two or three rows corresponding to different models: "AR-RCNP", "AR-ConvCNP", and "AR-CNP".
> *   **Cells:** Each cell contains a numerical value and a value in parentheses. The numerical value is displayed in bold font for the best performing model in each INT/OOID column. Some cells contain the letter "F" instead of a number.
> 
> The table is visually separated by horizontal lines between the different values of "d_x" and between the column headers and the data.


# H.1.1 Computation time 

The AR procedure is not used in model training, so the training procedure aligns with that of standard CNP models. The experiments reported in this section used AR with models trained for the synthetic regression experiments in Section 5.1. If the models had been trained from scratch, we estimate that the total computational cost for these experiments would have been approximately 2000 GPU-hours. AR is applied in the evaluation phase, where we predict target points autoregressively. This process is considerably lengthier than standard evaluation, with the duration influenced by both the target set size and data dimension. Nonetheless, since only a single pass is required during evaluation, the overall computational time remains comparable to standard CNP models.

## H. 2 Rotation equivariance

The experiments presented in this section use a two-dimensional toy example to investigate modeling datasets that are rotation equivariant but not translation equivariant. We created regression tasks based

---

#### Page 33

on synthetic data sampled from a GP model with an exponentiated quadratic covariance function and mean function $m(\mathbf{x})=\|\mathbf{A R} \mathbf{x}\|_{2}^{2}$, where $\mathbf{A}=\operatorname{diag}(\mathbf{a})$ is a fixed diagonal matrix with unequal entries and $\mathbf{R}$ is a random rotation matrix. We considered an isotropic-kernel model version with the standard EQ covariance function (Equation S3) and an anisotropic-kernel model version with the covariance function $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\exp \left(-||\mathbf{B R} \mathbf{x}-\mathbf{B R} \mathbf{x}^{\prime}||_{2}^{2}\right)$, where $\mathbf{B}=\operatorname{diag}(\mathbf{b})$ is a fixed diagonal matrix with unequal entries and $\mathbf{R}$ is the same random rotation matrix as in the mean function. Both models use the anisotropic GP mean function defined previously.
We generated the datasets representing regression tasks by sampling context and target points from the synthetic data generated with isotropic-kernel or anisotropic-kernel model version as follows. The number of context points sampled varied uniformly between 1 and 70, while the number of target points was fixed at 70 . All training and validation datasets were sampled from the range $\mathcal{X}=[-4,0] \times[-4,0]$. To evaluate the models in an interpolation (INT) task, we generated evaluation datasets by sampling context and target points from the training range. To evaluate the models in an out-of-input distribution (OOID) task, we generated evaluation datasets with context and target points sampled in the range $[0,4] \times[0,4]$.
We chose a comparison function $g$ that is rotation invariant but not translation invariant defined as ${ }^{2}$

$$
g\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\left(\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|_{2},\|\mathbf{x}\|_{2},\left\|\mathbf{x}^{\prime}\right\|_{2}\right)
$$

Since 'simple' RCNPs are only context preserving for translation-equivariance and not for other equivariances (Proposition 4.9), we combine the rotation-equivariant comparison function with a FullRCNP model.
We compared FullRCNP with the rotation-equivariant comparison function, FullRCNP (rot), with CNP and ConvCNP models in both the isotropic-kernel and anisotropic-kernel test condition. We also used isotropic-kernel data to train and evaluate a translation-equivariant ('stationary') RCNP version, RCNP (sta), that uses the difference comparison function. The model architectures and training details were the same as in the two-dimensional synthetic regression experiments (Section D.1).
The results reported in Table S12 show that, as expected, the isotropic-kernel test condition is easier than the anisotropic-kernel test condition where the random rotations change the covariance structure. The best results in both conditions are observed with ConvCNP when the models are evaluated in the interpolation task and with FullRCNP (rot) when the models are evaluated in the OOID task. Since the input range in the OOID task is rotated compared to the training range, FullRCNP (rot) results are on the same level in both tasks while the other models are not able to make reasonable predictions in the OOID task. We also observe that that the FullRCNP (rot) results in the interpolation task are better than CNP or RCNP (sta) results in the isotropic-kernel test condition and similar to CNP in the anisotropic-kernel test condition.

Table S12: Normalized log-likelihood scores in the experiments using synthetic data generated with isotropic and anisotropic GP kernels and an anisotropic GP mean function. The mean and (standard deviation) reported for each model are calculated based on 10 training outcomes evaluated with separate datasets representing interpolation (INT) and out-of-input-distribution (OOID) tasks. "F" denotes failed attempts with log-likelihood scores below -60 .

|  | Isotropic GP kernel |  | Anisotropic GP kernel |  |
| :-- | :--: | :--: | :--: | :--: |
|  | INT | OOID | INT | OOID |
| FullRCNP (rot) | -0.89 (0.04) | $\mathbf{- 0 . 9 0}(0.04)$ | -6.37 (0.34) | $\mathbf{- 6 . 3 8}(0.31)$ |
| RCNP (sta) | -1.16 (0.03) | F | - | - |
| ConvCNP | $\mathbf{- 0 . 6 0}(0.02)$ | F | $\mathbf{- 2 . 8 3}(0.12)$ | F |
| CNP | -1.42 (0.03) | F | -6.43 (1.13) | F |

# H.2.1 Computation time 

The models were trained on a single GPU, and each training run took around 6 hours with FullRCNP, 2 hours with CNP, 4 hours with ConvCNP and 3 hours with RCNP. The results provided in this

[^0]
[^0]:    ${ }^{2}$ This comparison function is invariant to proper and improper rotations around the origin (i.e., including mirroring).

---

#### Page 34

appendix required 70 training runs and consumed around 250 GPU hours. Additional experiments related to this example consumed around 2500 GPU hours.

# H. 3 Image datasets 

The experiments reported in the main paper demonstrate how RCNP can leverage prior information about equivariances in regression tasks including tasks with more than 1-2 input dimensions. In this section, we provide details and results from additional experiments that used MNIST [24] and CelebA [27] image data to create regression tasks. The neural process models were used in these experiments to predict the mean and variance over pixel values across the two-dimensional image given some pixels as observed context. The experiments investigate how RCNP compares to other conditional neural process models when the assumption about task equivariance is incorrect.

## H.3.1 Setup

The experiments reported in this section compared RCNP, CNP, and ConvCNP models using interpolation tasks generated based on $16 \times 16$ pixel images. The pixels were viewed as data points with $d_{x}=2$ input variables that indicate the pixel location on image canvas and $d_{y}$ output variables that encode the pixel value. The output dimension depended on the image type. Pixels in grayscale images were viewed as data points with one output variable $\left(d_{y}=1\right)$ that takes values between 0 and 1 while pixels in RGB images were viewed as data points with three output variables $\left(d_{y}=3\right)$ that each take values between 0 and 1 to encode one RGB channel. We evaluated the selected models using standard MNIST and CelebA images (Section H.3.2) and examined modeling translation equivariance with toroidal MNIST images (Section H.3.3).
The model architectures used in the image experiments match the architectures used in the synthetic regression experiments (Section D.1) extended with an output transformation to bound the predicted mean values between 0 and 1 . While we used images with multiple output channels, there was no need to use a multi-output architecture such as discussed in Section F.1. This is because we assumed that all output channels are observed when a pixel is included in the context set and unobserved otherwise, meaning that the output channels in image data are not associated with separate context sets. RCNP models were used with the standard stationary comparison function unless otherwise mentioned.
The models were trained for 200 epochs with $2^{14}$ datasets in each epoch and learning rate $3 \cdot 10^{-4}$, and tested with validation and evaluation sets that included $2^{12}$ datasets each. The log-likelihood score that is used as training objective is usually calculated based on the predicted mean and variance, but when the models were trained on MNIST data, we started training with the predicted variance fixed to a small value. This was done to prevent the neural process models from learning to predict the constant background values with low variance and ignoring the image details. We trained with the fixed variance between epochs $1-150$ and with the predicted variance as usual between epochs 151-200.
The datasets used to train and evaluate the neural process models were generated as follows. Each dataset was generated based on an image that was sampled from the selected image set and downscaled to $16 \times 16$ pixels. The downscaled images were either used as such or transformed with random translations as explained in Section H.3.3. The context sets generated based on the image data included 2-128 pixels sampled at random while the target sets included all 256 pixels. The evaluation sets used in each experiment were generated with the context set size fixed to $2,20,40$, and 100 pixels.

## H.3.2 Experiment 1: Centered images (no translation)

The experiments reported in this section used MNIST and CelebA images scaled to the size $16 \times 16$ (Figure S4). The normalized log-likelihood scores calculated based on each evaluation set and across training runs for each neural process model are reported in Table S13. The best results are observed with ConvCNP, while the order between RCNP and CNP depends on the context set size. CNP results are better than RCNP results when predictions are generated based on $N=2$ context points while RCNP results are better when more context data is available.
We believe CNP works better than RCNP when $N=2$ because the absolute pixel locations are generally informative about the possible pixel values in the MNIST and CelebA images. This means

---

#### Page 35

> **Image description.** The image shows two rows of small images. The top row is labeled "(a)" and contains five grayscale images of handwritten digits on a black background. From left to right, the digits appear to be "6", "9", "6", "3", and "7". The images are pixelated, suggesting a low resolution.
> 
> The bottom row is labeled "(b)" and contains five color images of faces. These images are also pixelated, similar to the digits above. The first image on the left shows a woman with red hair and a blue background. The second image appears to show a person with dark hair and a white background. The third image shows a young boy with brown hair and a blue shirt. The fourth image shows a person with brown hair and a grey background, and the fifth image shows a woman with brown hair and a blue background.


Figure S4: Example (a) MNIST and (b) CelebA $16 \times 16$ images.
Table S13: Normalized log-likelihood scores in the image experiments using (a) $16 \times 16$ MNIST images or (b) $16 \times 16$ CelebA images (higher is better). The mean and (standard deviation) reported for each model are calculated based on 10 training outcomes evaluated in interpolation task with context sizes $2,10,20$, and 100 . RCNP results are the lowest when the context set is small but improve when the context size increases.

|  |  | 2 | 20 | 40 | 100 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| (a) | RCNP | 5.52 (0.30) | 7.13 (0.10) | 8.08 (0.13) | 9.35 (0.15) |
|  | ConvCNP | 6.43 (0.02) | 7.35 (0.04) | 8.27 (0.04) | 9.87 (0.06) |
|  | CNP | 6.29 (0.12) | 7.04 (0.03) | 7.47 (0.03) | 8.00 (0.05) |
| (b) | RCNP | 0.18 (0.00) | 1.19 (0.01) | 1.79 (0.01) | 3.01 (0.03) |
|  | ConvCNP | 0.33 (0.00) | 1.36 (0.01) | 2.03 (0.01) | 3.37 (0.03) |
|  | CNP | 0.31 (0.00) | 0.90 (0.01) | 1.09 (0.01) | 1.25 (0.01) |

that the regression tasks generated based on these images are not translation equivariant and the assumption encoded in the RCNP comparison function is incorrect. Since RCNP does not encode the absolute target location, it needs context data to make predictions that CNP and ConvCNP can make based on the target location alone (Figure S5).
While RCNP needs context data to work around the incorrect assumption about translation equivariance, RCNP results are better than CNP results when more context data is available. We believe this is because the context set representation in CNP does not depend on the current target location while RCNP can learn to preserve context information that is relevant to the current target. For example, while CNP looses information about the exact observed values when the context set is encoded, RCNP generally learns to reproduce the observed context. RCNP may also learn to emphasize context points that are close to the current target and may learn local features that are translation equivariant.

# H.3.3 Experiment 2: Translated images 

The interpolation tasks in the previous image experiments do not exhibit translation equivariance since the input range is fixed and the image content is centered on the canvas. To run an image experiment with translation equivariance, we converted the $16 \times 16$ MNIST images into toruses and applied random translations to move image content around. The translations do not change the relative differences between image pixels on the torus, but when the torus is viewed as an image, both the absolute pixel locations and relative differences between pixel locations are transformed. This means that to capture the translation equivariance, we needed to calculate differences between pixel locations on the torus.

The experiments reported in this section compare models trained using either centered image data (Figure S4 (a)) or image data with random translations on the torus (Figure S6). We ran experiments with both RCNP (sta) that calculates the difference between pixel locations on the image canvas and RCNP (tor) that calculates the difference between pixel locations on the torus. The models trained with centered image data were evaluated with both centered and translated data while the models

---

#### Page 36

> **Image description.** The image shows a figure with two panels, (a) and (b), each containing a 4x4 grid of images. Each row in the grid represents a different model (Context, RCNP, ConvCNP, CNP), and each column represents a different value of N (N=2, N=20, N=40, N=100).
> 
> Panel (a) displays the results of an image experiment using MNIST data. The images in this panel are grayscale and depict handwritten digits.
> - The "Context" row shows images that are mostly blue with a few scattered gray or white pixels. As N increases, the number of non-blue pixels increases, and they start to resemble the shape of a digit.
> - The "RCNP", "ConvCNP", and "CNP" rows display the predicted mean values for the digit. As N increases, the predicted digits become clearer and more defined. The digits appear to be the number "3" or "9".
> 
> Panel (b) displays the results of an image experiment using Celeba 16x16 data. The images in this panel are in color and depict faces.
> - The "Context" row shows images that are mostly blue with a few scattered colored pixels. As N increases, the number of non-blue pixels increases, and they start to resemble the shape of a face.
> - The "RCNP", "ConvCNP", and "CNP" rows display the predicted mean values for the face. As N increases, the predicted faces become clearer and more defined.


Figure S5: Example context sets and predicted mean values in the image experiment using (a) MNIST and (b) Celeba $16 \times 16$ data.

---

#### Page 37

> **Image description.** The image shows five grayscale images of handwritten digits, each presented in a square frame with a black background. The digits are rendered in varying shades of gray, appearing pixelated due to the low resolution.
> 
> *   **Panel 1:** Shows a partial digit, possibly a '2' or '7', with the lower portion visible. The upper part of the digit is cut off.
> *   **Panel 2:** Displays another partial digit, with only a fragment of the shape visible, making it difficult to identify.
> *   **Panel 3:** Clearly presents the digit '4'.
> *   **Panel 4:** Shows the digit '3', though a portion of it is cut off.
> *   **Panel 5:** Shows a mostly blank square with a few gray pixels in the corners.


Figure S6: MNIST $16 \times 16$ example images with random translations on a torus.
trained with translated data were evaluated with translated data. The normalized log-likelihood scores are reported in Table S14.

Table S14: Normalized log-likelihood scores in the image experiments using $16 \times 16$ MNIST images with random translations on a torus (higher is better). We compare (a) models trained on centered image data evaluated on centered image data, (b) models trained on centered image data evaluated on image data with random translations, and (c) models trained on image data with random translations evaluated on image data with random translations. The mean and (standard deviation) reported for each model are calculated based on 10 training outcomes evaluated in interpolation task with context sizes $2,10,20$, and 100. "F" denotes very bad results with negative log-likelihood scores. RCNP (tor) trained on centered image data is able to make predictions about images with random translations because the translation mechanism is modeled in the comparison function.

|  |  | 2 | 20 | 40 | 100 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| (a) | RCNP (sta) | 5.52 (0.30) | 7.13 (0.10) | 8.08 (0.13) | 9.35 (0.15) |
|  | RCNP (tor) | 5.51 (0.01) | 6.63 (0.04) | 7.59 (0.03) | 9.15 (0.09) |
|  | ConvCNP | 6.43 (0.02) | 7.35 (0.04) | 8.27 (0.04) | 9.87 (0.06) |
|  | CNP | 6.29 (0.12) | 7.04 (0.03) | 7.47 (0.03) | 8.00 (0.05) |
| (b) | RCNP (sta) | F | F | F | F |
|  | RCNP (tor) | 5.62 (0.01) | 6.68 (0.04) | 7.61 (0.06) | 9.13 (0.09) |
|  | ConvCNP | F | F | F | F |
|  | CNP | F | F | F | F |
| (c) | RCNP (sta) | 5.51 (0.21) | 6.66 (0.11) | 7.67 (0.09) | 9.35 (0.11) |
|  | RCNP (tor) | 5.56 (0.10) | 6.72 (0.06) | 7.70 (0.09) | 9.31 (0.14) |
|  | ConvCNP | 5.61 (0.01) | 6.72 (0.03) | 7.78 (0.04) | 9.60 (0.08) |
|  | CNP | 5.55 (0.03) | 6.29 (0.03) | 6.80 (0.02) | 7.52 (0.09) |

The results in Table S14 (a) extend the results discussed in the previous section (Table S13 (a)). The results indicate that when the models are trained and tested with interpolation tasks generated based on centered image data, RCNP (sta) works better than RCNP (tor). We believe this is because the relative locations calculated based on pixel coordinates on the image canvas are more informative about the absolute location than relative locations calculated on the torus. In other words, while RCNP (sta) and RCNP (tor) do not encode the absolute target location on the image canvas, RCNP (sta) can learn to derive equivalent information based on regularities between context sets.
The image interpolation task becomes translation equivariant on the torus when the models are trained or tested using the image data with random translations. The results reported in Table S14 (b) indicate that using differences calculated on the torus allows RCNP (tor) to generalize between centered and translated images while the other models are not able to make sensible predictions in this OOID task. All models work well when trained with matched data (Table S14 (c)), meaning that training on image data with random translations allows the other models to learn the equivariance which is encoded in RCNP (tor). The best results in Table S14 (c) are observed with ConvCNP, while RCNP (tor) and RCNP (sta) are similar to CNP when $N=2$ and better than CNP when more context data is available.

---

#### Page 38

# H.3.4 Computation time 

The models were trained on a single GPU, and each training run with RCNP took around 3-4 hours, each training run with CNP around 1-2 hours, and each training run with ConvCNP around 4.5-5.5 hours depending on the training data. The results presented in this supplement required 110 training runs with total runtime around 350 hours. In addition other experiments carried out with image data took around 2000 hours.