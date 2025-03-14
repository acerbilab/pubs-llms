```
@inproceedings{huang2023learning,
  title={Learning Robust Statistics for Simulation-based Inference under Model Misspecification},
  author={Daolang Huang and Ayush Bharti and A. Souza and Luigi Acerbi and Samuel Kaski},
  booktitle={The Thirty-seventh Annual Conference on Neural Information Processing Systems (NeurIPS 2023)},
  year={2023},
  doi={10.48550/arXiv.2305.15871}
}
```

---

#### Page 1

# Learning Robust Statistics for Simulation-based Inference under Model Misspecification

Daolang Huang*<br>Aalto University<br>daolang.huang@aalto.fi<br>Ayush Bharti*<br>Aalto University<br>ayush.bharti@aalto.fi<br>Amauri H. Souza<br>Aalto University<br>Federal Institute of Ceará<br>amauri.souza@aalto.fi<br>Luigi Acerbi<br>University of Helsinki<br>luigi.acerbi@helsinki.fi<br>Samuel Kaski<br>Aalto University<br>University of Manchester<br>samuel.kaski@aalto.fi

#### Abstract

Simulation-based inference (SBI) methods such as approximate Bayesian computation (ABC), synthetic likelihood, and neural posterior estimation (NPE) rely on simulating statistics to infer parameters of intractable likelihood models. However, such methods are known to yield untrustworthy and misleading inference outcomes under model misspecification, thus hindering their widespread applicability. In this work, we propose the first general approach to handle model misspecification that works across different classes of SBI methods. Leveraging the fact that the choice of statistics determines the degree of misspecification in SBI, we introduce a regularized loss function that penalises those statistics that increase the mismatch between the data and the model. Taking NPE and ABC as use cases, we demonstrate the superior performance of our method on high-dimensional time-series models that are artificially misspecified. We also apply our method to real data from the field of radio propagation where the model is known to be misspecified. We show empirically that the method yields robust inference in misspecified scenarios, whilst still being accurate when the model is well-specified.

## 1 Introduction

Bayesian inference traditionally entails characterising the posterior distribution of parameters assuming that the observed data came from the chosen model family [7]. However, in practice, the true data-generating process rarely lies within the family of distributions defined by the model, leading to model misspecification. This can be caused by, among other things, measurement errors or contamination in the observed data that are not included in the model, or when the model fails to capture the true nature of the physical phenomenon under study. Misspecified scenarios are especially likely for simulator-based models, where the goal is to describe some complex real-world phenomenon. Such simulators, also known as implicit generative models [25], have become prevalent in many domains of science and engineering such as genetics [71], ecology [3], astrophysics [40], economics [29], telecommunications [8], cognitive science [79], and agent-based modeling [87]. The growing field of simulation-based inference (SBI) [21, 51] tackles inference for such intractable likelihood models, where the approach relies on forward simulations from the model, instead of likelihood evaluations, to estimate the posterior distribution.

Traditional SBI methods include approximate Bayesian computation (ABC) [4, 51, 76], synthetic likelihood [67, 85] and minimum distance estimation [15]. More recently, the use of neural networks

[^0]
[^0]: \*Equal contribution.

---

#### Page 2

> **Image description.** The image consists of three panels arranged horizontally, displaying scatter plots and density plots related to statistical inference.
>
> - **Panel 1 (Left):** Titled "NPE", this panel shows a scatter plot. The x-axis is labeled "Statistic 1" and ranges from approximately -2000 to 0. The y-axis is labeled "Statistic 2" and ranges from -800 to 0. Many small black dots, labeled "Simulated", are clustered along a diagonal line extending from the top-left to the bottom-right of the plot. A single red "X", labeled "Observed", is located in the bottom-right corner of the plot, distinctly separated from the cluster of black dots.
>
> - **Panel 2 (Middle):** Titled "Our method", this panel also displays a scatter plot. The x-axis is labeled "Statistic 1" and ranges from approximately -200 to 0. The y-axis is labeled "Statistic 2" and ranges from 0 to 3000. Similar to the first panel, many small black dots, labeled "Simulated", are clustered, but in this case, they are concentrated in the bottom-right corner and along a diagonal line extending from the top-left to the bottom-right. A single red "X", labeled "Observed", is located within the cluster of black dots in the bottom-right corner.
>
> - **Panel 3 (Right):** This panel presents a scatter plot with density plots along the axes. The x-axis is labeled "θ1" and ranges from approximately 2.5 to 7.5. The y-axis is labeled "θ2" and ranges from 0 to 25. A horizontal black line, labeled "True θ", is positioned at y = 10, and a vertical black line is positioned at x = 4.5. Two shaded regions are overlaid on the scatter plot: a blue region labeled "Ours" and an orange region labeled "NPE". The blue region is concentrated around the intersection of the black lines, while the orange region is located higher up and to the left. Density plots are displayed along the top and right edges of the scatter plot, corresponding to the marginal distributions of θ1 and θ2. The density plots are color-coded to match the shaded regions, with blue representing "Ours" and orange representing "NPE".

Figure 1: Inference on misspecified Ricker model with two parameters, see Section 4 for details. Left: Learned statistics obtained using NPE; each point (in black) corresponds to a parameter value sampled from the prior. The observed statistic (in red) is outside the set of statistics the model can simulate. Middle: Statistics learned using our robust method; the observed statistic is inside the distribution of simulated statistics. Right: The resulting posteriors obtained from NPE and our method. The posterior obtained from our method is close to the true parameter value, while that from NPE is completely off, going outside the prior range (denoted by dashed grey lines).

as conditional density estimators [72] has fuelled the development of new SBI methods that target either the likelihood (neural likelihood estimation) [52, 61, 75, 83], the likelihood-to-evidence ratio (neural ratio estimation) [22, 23, 28, 30, 39, 46, 77], or the posterior (neural posterior estimation) $[38,41,53,58,59,70,75,83]$. A common feature of all these SBI methods is the choice of summary statistics of the data, which readily impacts their performance [14]. The summary statistics are either manually handcrafted by domain experts, or learned automatically from simulated data using neural networks $[1,17,18,26,31,47,84]$.
The unreliability of Bayesian posteriors under misspecification is exacerbated for SBI methods as the inference relies on simulations from the misspecified model. Moreover, for SBI methods using conditional density estimators [5, 13, 41, 61, 69], the resulting posteriors can be wildly inaccurate and may even go outside the prior range $[9,16,35]$; see [27] for an instance of this problem in the study of molecules. This behaviour is exemplified in Figure 1 for the posterior-targeting neural SBI method called neural posterior estimation (NPE). We see that when the model is misspecified, the NPE posterior goes outside the prior range. This is because the model is unable to match the observed statistic for any value of the parameters, as shown in Figure 1(left), and hence, NPE is forced to generalize outside its training distribution, yielding a posterior that is outside the prior range.
We argue that the choice of statistics is crucial to the notion of misspecification in SBI. Suppose we wish to fit a Gaussian model to data from some skewed distribution. If we choose, for instance, the sample mean and the skewness as statistics, then the Gaussian model would fail to match these statistics for any choice of parameters, and we would end up in the same situation as in Figure 1 where the SBI method is forced to generalize outside its training distribution. However, choosing the sample mean and sample variance as statistics instead may solve this issue, as the model may be able to replicate the chosen statistics for some value of parameters, even though the Gaussian model itself is misspecified for any skewed dataset. This problem of identifying a low-dimensional statistic of a high-dimensional dataset for which the model of interest is correctly specified is termed the data selection problem [82]. We refer to the statistics that solve the data selection problem as being robust.

Contributions. In this paper, we learn robust statistics using neural networks to solve the data selection problem. To that end, we introduce a regularized loss function that balances between learning statistics that are informative about the parameters, and penalising those choices of statistics or features of the data that the model is unable to replicate. By doing so, the observed statistic does not go outside the set of statistics that the model can simulate, see Figure 1(middle), thereby yielding a posterior around the true parameter even under misspecification (Figure 1(right)). As our method relies on learning appropriate statistics and not on the subsequent inference procedure, it is applicable to all the statistics-based SBI methods, unlike other robust methods in the literature [24, 48, 81]. We substantiate our claim in Section 4 through extensive numerical studies on NPE and ABC - two fundamentally different SBI methods. Additionally in Section 5, we apply our method to a real case of model misspecification in the field of radio propagation [8].

---

#### Page 3

Related works. Model misspecification has been studied in the context of different SBI methods such as ABC [9, 33, 35, 36, 74], synthetic likelihood [32, 34, 57], minimum distance estimation [24], and neural estimators [48, 81]. It was first analysed for ABC methods such as the regressionadjustment ABC [5] in [35], and a robust approach was proposed in [33]. In [9], the authors proposed to handle model misspecification in ABC by incorporating the domain expert while selecting the statistics. Generalized Bayesian inference [12, 43, 49], which is a popular framework for designing methods that are robust to misspecification, has also been linked to ABC [74] and synthetic likelihood [57]. In [24], the authors propose a robust SBI method based on Bayesian nonparametric learning and the posterior bootstrap. More recently, robustness to model misspecification has been tackled for neural likelihood [48] and posterior estimators [81]. We use the latter method, called robust neural posterior estimation (RNPE) [81], as a baseline for comparing our proposed approach. Our method differs from these previous robust approaches in that it is applicable across distinct SBI methods.

# 2 Preliminaries

Simulation-based inference (SBI). Consider a simulator $\left\{\mathbb{P}_{\theta}: \theta \in \Theta\right\}$, which is a family of distributions parameterized by $\theta$ on data space $\mathcal{X} \subseteq \mathbb{R}^{d}$. We assume that $\mathbb{P}_{\theta}$ is intractable, but sampling independent and identically distributed (iid) data $\mathbf{x}_{1: n}=\left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\} \sim \mathbb{P}_{\theta}$ is straightforward for $\mathbf{x}^{(i)} \in \mathcal{X}, 1 \leq i \leq n$. Given iid observed data $\mathbf{y}_{1: n} \sim \mathbb{Q}, \mathbf{y}^{(i)} \in \mathcal{X}$, from some true data-generating process $\mathbb{Q}$ and a prior distribution of $\theta$, we are interested in estimating a $\theta^{*} \in \Theta$ such that $\mathbb{P}_{\theta^{*}}=\mathbb{Q}$, or alternatively, in characterizing the posterior distribution $p\left(\theta \mid \mathbf{y}_{1: n}\right)$. Since $\mathcal{X}$ is a high-dimensional space in most practical cases, it is common practice to project both the simulated and the observed data onto a low-dimensional space of summary statistics $\mathcal{S}$ via a mapping $\eta: \mathcal{X}^{n} \rightarrow \mathcal{S}$, such that $\mathbf{s}=\eta\left(\mathbf{x}_{1: n}\right)$ is the vector of simulated statistics, and $\mathbf{s}_{\text {obs }}=\eta\left(\mathbf{y}_{1: n}\right)$ the observed statistic. In the following, we assume that this deterministic map $\eta$ is not known a priori, but needs to be learned using a neural network, henceforth called the summary network $\eta_{\psi}$, with trainable parameters $\psi$. We now introduce two different paradigms for learning the statistics.

Learning statistics in neural posterior estimation (NPE) framework. NPEs are a class of SBI methods which are becoming popular in astrophysics [40] and nuclear fusion [37, 80], among other fields. They map each statistic vector $\mathbf{s}$ to an estimate of the posterior $p(\theta \mid \mathbf{s})$ using conditional density estimators such as normalizing flows [41, 53, 59, 69]. The posterior is assumed to be a member of a family of distributions $q_{\nu}$, parameterised by $\nu$. NPEs map the statistics to distribution parameters $\nu$ via an inference network $h_{\phi}$, where $\phi$ constitutes the weights and biases of $h$, such that $q_{h_{\phi}(\mathbf{s})}(\theta) \approx p(\theta \mid \mathbf{s})$. The inference network is trained on dataset $\left\{\left(\theta_{i}, \mathbf{s}_{i}\right)\right\}_{i=1}^{m}$, simulated from the model using prior samples $\left\{\theta_{i}\right\}_{i=1}^{m} \sim p(\theta)$, by minimising the loss $\mathcal{L}(\phi)=-\mathbb{E}_{p(\theta, \mathbf{s})}\left[\log q_{h_{\phi}(\mathbf{s})}(\theta)\right]$. Once trained, the posterior estimate is then obtained by simply passing the observed statistic $\mathbf{s}_{\text {obs }}$ through the trained network. Hence, NPEs enjoy the benefit of amortization, wherein inference on new observed dataset is straightforward after a computationally expensive training phase. In cases where the statistics are not known a priori, the NPE framework allows for joint training of both the summary and the inference networks by minimising the loss function

$$
\mathcal{L}_{\mathrm{NPE}}(\phi, \psi)=-\mathbb{E}_{p\left(\theta, \mathbf{x}_{1: n}\right)}\left[\log q_{h_{\phi}\left(\eta_{\psi}\left(\mathbf{x}_{1: n}\right)\right)}(\theta)\right]
$$

on the training dataset $\left\{\left(\theta_{i}, \mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{m}$ [69]. Even though NPEs are flexible and efficient SBI methods, they are known to perform poorly when the model is misspecified [16, 81]. This is because the observed statistic $\mathbf{s}_{\text {obs }}$ becomes an out-of-distribution sample under misspecification (see Figure 1 for an example), forcing the inference network in NPE to generalize outside its training distribution.

Learning statistics using autoencoders. Unlike NPEs, statistics for other SBI methods are learned prior to carrying out the inference procedure. This is achieved, for instance, by training an autoencoder with the reconstruction loss [1]

$$
\mathcal{L}_{\mathrm{AE}}\left(\psi, \psi_{d}\right)=\mathbb{E}_{p\left(\theta, \mathbf{x}_{1: n}\right)}\left[\left(\mathbf{x}_{1: n}-\tilde{\eta}_{\psi_{d}}\left(\eta_{\psi}\left(\mathbf{x}_{1: n}\right)\right)\right)^{2}\right]
$$

where $\psi$ and $\psi_{d}$ are the parameters of the encoder $\eta$ and the decoder $\tilde{\eta}$, respectively. The trained encoder $\eta_{\psi}$ is then taken as the summarizing function in the SBI method. In this paper, we will use the encoder $\eta_{\psi}$ to perform inference in an ABC framework, which we now recall.

---

#### Page 4

Approximate Bayesian computation (ABC). ABC is arguably the most popular SBI method, and is widely used in many research domains [64]. ABC relies on computing the distance between the simulated and the observed statistics to obtain samples from the approximate posterior of a simulator. Given a discrepancy $\rho(\cdot, \cdot)$ and a tolerance threshold $\delta$, the basic rejection-ABC method involves repeating the following algorithm: (i) sample $\theta^{\prime} \sim p(\theta)$, (ii) simulate $\mathbf{x}_{1: n}^{\prime} \sim \mathbb{P}_{\theta^{\prime}}$, (iii) compute $\mathbf{s}^{\prime}=\eta\left(\mathbf{x}_{1: n}^{\prime}\right)$, and (iv) if $\rho\left(\mathbf{s}^{\prime}, \mathbf{s}_{\text {obs }}\right)<\delta$, accept $\theta^{\prime}$, until a sample $\left\{\theta_{i}^{\prime}\right\}_{i=1}^{n_{s}^{\prime}}$ is obtained from the approximate posterior. Conditional density estimators have also been used in ABC to improve the posterior approximation of the rejection-ABC method [5, 11, 13, 14]. These so-called regression adjustment ABC methods involve fitting a function $g$ between the accepted parameters and statistics pairs $\left\{\left(\theta_{i}^{\prime}, \mathbf{s}_{i}^{\prime}\right)\right\}_{i=1}^{n_{s}}$ as $\theta_{i}^{\prime}=g\left(\mathbf{s}_{i}^{\prime}\right)+\omega_{i}, 1 \leq i \leq n_{\delta}$, where $\left\{\omega_{i}\right\}_{i=1}^{n_{\delta}}$ are the residuals. Once fitted, the accepted parameter samples are then adjusted as $\hat{\theta}_{i}^{\prime}=\hat{g}\left(\mathbf{s}_{\text {obs }}\right)+\hat{\omega}_{i}$, where $\hat{g}(\mathbf{s})$ is the estimate of $\mathbb{E}[\theta \mid \mathbf{s}]$ and $\hat{\omega}_{i}$ is the empirical residual. Despite the popularity of the regression adjustment ABC methods, they have also been shown to behave wildly under misspecification [9, 35], similar to NPEs.

# 3 Methodology

Model misspecification in SBI. Bayesian inference assumes that there exists a $\theta^{\star} \in \Theta$ such that $\mathbb{P}_{\theta^{\star}}=\mathbb{Q}$. Under model misspecification, $\mathbb{P}_{\theta} \neq \mathbb{Q}$ for any $\theta \in \Theta$, which can lead to unreliable predictive distributions. However, this definition does not apply for models whose inference is carried out in light of certain summary statistics. For instance, a Gaussian model is clearly misspecified for a bimodal data sample. However, if inference is based on, say, the sample mean and the sample variance of the data, the model may still match the observed statistics accurately. Hence, even when a simulator is misspecified with respect to the true data-generating process, it may still be well-specified with respect to the observed statistic. The choice of statistics therefore dictates the notion of misspecification in SBI. We propose an analogous definition for misspecification in SBI:
Definition 3.1 (Misspecification of model-summarizer pair in SBI). Let $\eta_{\#} \mathbb{P}_{\#}^{n}$ be a pushforward of the probability measure $\mathbb{P}_{\#}^{n}=\mathbb{P}_{\#} \times \cdots \times \mathbb{P}_{\theta}$ on $\mathcal{X}^{n}$ under $\eta: \mathcal{X}^{n} \rightarrow \mathcal{S}$, meaning that for $A \subset \mathcal{S}$, $\eta_{\#} \mathbb{P}_{\#}^{n}(A)=\mathbb{P}_{\#}^{n}\left(\eta^{-1}(A)\right)=\mathbb{P}_{\theta}\left(\eta^{-1}(A)_{1}\right) \times \cdots \times \mathbb{P}_{\theta}\left(\eta^{-1}(A)_{n}\right)$. Then, $\left\{\eta_{\#} \mathbb{P}_{\#}^{n}: \theta \in \Theta\right\}$ is a set of distributions on the statistics space $\mathcal{S}$ induced by the model for a given summarizer $\eta$, and $\eta_{\#} \mathbb{Q}^{n}$ is the distribution of the statistics from the true data-generating process. Then, the model-summarizer pair is misspecified if, $\forall \theta \in \Theta, \eta_{\#} \mathbb{Q}^{n} \neq \eta_{\#} \mathbb{P}_{\#}^{n}$.

It is trivial to see that if the model is well-specified in the data space $\mathcal{X}$, it will also be well-specified in the statistic space $\mathcal{S}$ for any choice of $\eta$. The converse, however, is not true, as previously mentioned. Using this definition, we quantify the level of misspecification of a model-summarizer pair as follows:
Definition 3.2 (Misspecification margin in SBI). Let $\mathcal{D}$ be a distance defined on the set of all Borel probability measures on $\mathcal{S}$. Then, for a given $\eta$, we define the misspecification margin $\varepsilon_{\eta}$ in SBI as

$$
\varepsilon_{\eta}=\inf _{\theta \in \Theta} \mathcal{D}\left(\eta_{\#} \mathbb{P}_{\#}^{n}, \eta_{\#} \mathbb{Q}^{n}\right)
$$

Note that the margin is equal to zero for the well-specified case. Moreover, the larger the margin, the bigger the mismatch between the simulated and the observed statistics w.r.t $\eta$. Our aim is to learn an $\eta$ such that $\mathbb{P}_{\theta}$ is represented well whilst the model-summarizer pair is no longer misspecified as per Definition 3.1, or alternatively, has zero margin as per Definition 3.2. We formulate this as a constrained optimization problem in Equation (6).

Learning robust statistics for SBI. Suppose we have the flexibility to choose the summarizer $\eta$ from a family of functions parameterized by $\psi$. Then, a potential approach for tackling model misspecification in SBI is to select the summarizer $\eta$ that minimizes the misspecification margin $\varepsilon_{\eta}$. However, that involves computing the infimum for all possible choices of $\psi$. To avoid that, we minimize an upper bound on the margin instead, given as

$$
\varepsilon_{\eta_{\psi}} \leq \varepsilon_{\eta_{\psi}}^{\text {upper }}=\mathbb{E}_{p(\theta)}\left[\mathcal{D}\left(\eta_{\psi \#} \mathbb{P}_{\#}^{n}, \eta_{\psi \#} \mathbb{Q}^{n}\right)\right]
$$

Note that Equation (4) is valid for any choice of $p(\theta)$, and we call $\varepsilon_{\eta_{\psi}}^{\text {upper }}$ the margin upper bound.
Minimizing the margin upper bound alone would lead to $\eta$ being a constant function, which when used for inference, would yield the prior distribution. Hence, there is a trade-off between choosing an $\eta$ that is informative about the parameters and an $\eta$ that minimizes the margin upper bound. We

---

#### Page 5

propose to navigate this trade-off by including the margin upper bound as a regularization term in the loss function for learning $\eta$. To that end, let $\mathcal{L}(\omega, \psi)$ be the loss function used to learn $\eta_{\psi}$ (along with additional parameters $\omega$ ) which, for instance, can be either $\mathcal{L}_{\mathrm{NPE}}$ from Equation (1) or $\mathcal{L}_{\mathrm{AE}}$ from Equation (2). Then, our proposed loss with robust statistics (RS) is written as

$$
\mathcal{L}_{\mathrm{RS}}(\omega, \psi)=\mathcal{L}(\omega, \psi)+\lambda \underbrace{\mathbb{E}_{p(\theta)}\left[\mathcal{D}\left(\eta_{\psi \#} \mathbb{P}_{\theta}^{n}, \eta_{\psi \#} \mathbb{Q}^{n}\right)\right]}_{\text {regularization }}
$$

where $\lambda \geq 0$ is the weight we place on the regularization term relative to the standard loss $\mathcal{L}$. Minimizing the loss function in Equation (5) corresponds to solving the Lagrangian relaxation of the optimisation problem:

$$
\begin{array}{ll}
\min _{\omega, \psi} & \mathcal{L}(\omega, \psi) \\
\text { s.t. } & \mathbb{E}_{p(\theta)}\left[\mathcal{D}\left(\eta_{\psi \#} \mathbb{P}_{\theta}^{n}, \eta_{\psi \#} \mathbb{Q}^{n}\right)\right] \leq \xi, \quad \xi>0
\end{array}
$$

with $\lambda \geq 0$ being the Lagrangian multiplier fixed to a constant value.
Estimating the margin upper bound. In order to implement the proposed loss from Equation (5), we need to estimate the margin upper bound using the training dataset $\left\{\left(\theta_{i}, \mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{m} \sim p\left(\theta, \mathbf{x}_{1: n}\right)$ sampled from the prior and the model. However, we only have one sample each from the distributions $\eta_{\psi \#} \mathbb{P}_{\theta_{i}}^{n}, i=1, \ldots, m$, and one sample of the observed statistic from $\eta_{\psi \#} \mathbb{Q}^{n}$. Taking $\mathcal{D}$ to be the Euclidean distance $\|\cdot\|$, we can estimate the margin upper bound as $\frac{1}{m} \sum_{i=1}^{m}\left\|\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)-\eta_{\psi}\left(\mathbf{y}_{1: n}\right)\right\|$. Although easy to compute, we found this choice of $\mathcal{D}$ to yield overly conservative posteriors even in the well-specified case, see Appendix B. 3 for the results. This is because the Euclidean distance is large even if only a handful of simulated statistics - corresponding to different parameter values in the training data - are far from the observed statistic. As a result, the regularizer dominates the loss function while the standard loss term $\mathcal{L}$ related to $\theta$ is not minimized, leading to underconfident posteriors. Hence, we need a distance $\mathcal{D}$ that is robust to outliers and can be computed between the set $\left\{\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{m}$ and $\eta_{\psi}\left(\mathbf{y}_{1: n}\right)$, instead of computing the distance point-wise.
To that end, we take $\mathcal{D}$ to be the maximum mean discrepancy (MMD) [42], which is a notion of distance between probability distributions or datasets. MMD has a number of attractive properties that make it suitable for our method: (i) it can be estimated efficiently using samples [15], (ii) it is robust against a few outliers (unlike the KL divergence), and (iii) it can be computed for unequal sizes of datasets. The MMD has been used in a number of frameworks such as ABC [8, 10, 50, 54, 62], minimum distance estimation [2, 15, 19, 56], generalised Bayesian inference [20, 57], and Bayesian nonparametric learning [24]. Similar to our method, the MMD has previously been used as a regularization term to train Bayesian neural networks [66], and in the NPE framework to detect model misspecification [73]. While we include the mmD-based regularizer to ensure that the observed statistic is not an out-of-distribution sample in the summary space, the regularizer in [73] involves computing the MMD between the simulated statistics and samples from a standard Gaussian, thus ensuring that the learned statistics are jointly Gaussian. They then conduct a goodness-of-fit test [42] to detect if the model is misspecified. Their method is complementary to ours, such that our method can be used once misspecification has been detected using [73].

Maximum mean discrepancy (MMD) as $\mathcal{D}$. The MMD is a kernel-based distance between probability distributions, computed by mapping the distributions to a function space called the reproducing kernel Hilbert space (RKHS). Let $\mathcal{H}_{k}$ be the RKHS associated with the symmetric and positive definite function $k: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{R}$, called a reproducing kernel [6], defined on the space of statistics $\mathcal{S}$ such that $f(\mathbf{s})=\langle f, k(\cdot, \mathbf{s})\rangle_{\mathcal{H}_{k}} \forall f \in \mathcal{H}_{k}$ (called the reproducing property). We denote the norm and inner product of $\mathcal{H}_{k}$ by $\|\cdot\|_{\mathcal{H}_{k}}$ and $\langle\cdot, \cdot\rangle_{\mathcal{H}_{k}}$, respectively. Any distribution $\mathbb{P}$ can be mapped to $\mathcal{H}_{k}$ via its kernel-mean embedding $\mu_{k, \mathbb{P}}=\int_{\mathcal{S}} k(\cdot, \mathbf{s}) \mathbb{P}(\mathrm{d} \mathbf{s})$. The MMD between two arbitrary probability measures $\mathbb{P}$ and $\mathbb{Q}$ on $\mathcal{S}$ is defined as the distance between their embeddings in $\mathcal{H}_{k}$, i.e., $\operatorname{MMD}_{k}(\mathbb{P}, \mathbb{Q})=\left\|\mu_{k, \mathbb{P}}-\mu_{k, \mathbb{Q}}\right\|_{\mathcal{H}_{k}}$ [55]. Using the reproducing property, we can express the squared-MMD as

$$
\operatorname{MMD}_{k}^{2}[\mathbb{P}, \mathbb{Q}]=\mathbb{E}_{\mathbf{s}, \mathbf{s}^{\prime} \sim \mathbb{P}}\left[k\left(\mathbf{s}, \mathbf{s}^{\prime}\right)\right]-2 \mathbb{E}_{\mathbf{s} \sim \mathbb{P}, \mathbf{s}^{\prime} \sim \mathbb{Q}}\left[k\left(\mathbf{s}, \mathbf{s}^{\prime}\right)\right]+\mathbb{E}_{\mathbf{s}, \mathbf{s}^{\prime} \sim \mathbb{Q}}\left[k\left(\mathbf{s}, \mathbf{s}^{\prime}\right)\right]
$$

which is computationally convenient as the expectations of the kernel can be estimated using iid samples from $\mathbb{P}$ and $\mathbb{Q}$. Given samples of simulated statistics $\left\{\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{t}$ and the observed

---

#### Page 6

statistic $\eta_{\psi}\left(\mathbf{y}_{1: n}\right)$, we estimate the squared-MMD between them using the V-statistic estimator [42]:

$$
\begin{aligned}
\operatorname{MMD}_{k}^{2}\left[\eta_{\psi \#} \mathbb{P}_{\theta}^{n}, \eta_{\psi \#} \mathbb{Q}^{n}\right] & \approx \operatorname{MMD}_{k}^{2}\left[\left\{\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{l}, \eta_{\psi}\left(\mathbf{y}_{1: n}\right)\right] \\
& =\frac{1}{l^{2}} \sum_{i, j=1}^{l} k\left(\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right), \eta_{\psi}\left(\mathbf{x}_{1: n, j}\right)\right)-\frac{2}{l} \sum_{i=1}^{l} k\left(\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right), \eta_{\psi}\left(\mathbf{y}_{1: n}\right)\right)
\end{aligned}
$$

Note that the last term in Equation (7) is always constant for the estimator above as we only have one data-point of the observed statistic, hence we disregard it. Equation (8) therefore corresponds to estimating the MMD between the distribution of the simulated statistics for a given $\psi$ and a Dirac measure on $\eta_{\psi}\left(\mathbf{y}_{1: n}\right)$. The computational cost of estimating the squared-MMD is $\mathcal{O}\left(l^{2}\right)$ and its rate of convergence is $\mathcal{O}\left(l^{-\frac{1}{2}}\right)$. The NPE loss with robust statistics (NPE-RS) can then be estimated as

$$
\hat{\mathcal{L}}_{\mathrm{NPE}-\mathrm{RS}}(\phi, \psi)=-\frac{1}{m} \sum_{i=1}^{m} \log q_{h_{\phi}\left(\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right)}\left(\theta_{i}\right)+\lambda \mathrm{MMD}_{k}^{2}\left[\left\{\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{l}, \eta_{\psi}\left(\mathbf{y}_{1: n}\right)\right]
$$

Similarly, the autoencoder loss with robust statistics (AE-RS) reads

$$
\hat{\mathcal{L}}_{\mathrm{AE}-\mathrm{RS}}\left(\psi, \psi_{d}\right)=\frac{1}{m} \sum_{i=1}^{m}\left(\mathbf{x}_{1: n, i}-\widehat{\eta}_{\psi_{d}}\left(\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right)\right)^{2}+\lambda \mathrm{MMD}_{k}^{2}\left[\left\{\eta_{\psi}\left(\mathbf{x}_{1: n, i}\right)\right\}_{i=1}^{l}, \eta_{\psi}\left(\mathbf{y}_{1: n}\right)\right]
$$

We use a subset of the training dataset of size $l<m$ to compute the MMD instead of $m$ to avoid incurring additional computational cost in case $m$ is large. The MMD-based regularizer can also be used as a score in a classification task to detect model misspecification, as shown in Appendix A.

Role of the regularizer $\lambda$. The regularizer $\lambda$ penalizes learning those statistics for which the observed statistic $\eta_{\psi}\left(\mathbf{y}_{1: n}\right)$ is far from the set of statistics that the model can simulate given a prior distribution. When $\lambda$ tends to zero, maximization of the likelihood dictates learning of both $\phi$ and $\psi$ in Equation (9), and our method converges to the NPE method. On the other hand, the regularization term is minimized when the summary network outputs the same statistics for both simulated and observed data, i.e., when $\mathcal{D}$ is zero. In this case, the inference network can only rely on information from the prior $p(\theta)$. As a result, for large values of $\lambda$, we expect the regularization term to dominate the loss and the resulting posterior to converge to the prior distribution. Similar argument holds for the autoencoder loss in Equation (10). We empirically observe this behavior in Section 4. Hence, $\lambda$ encodes the trade-off between efficiency and robustness of our inference method. Choosing $\lambda$ can be cast as a hyperparameter selection problem, for which we can leverage additional data (if available) as a validation dataset, or use post-hoc qualitative analysis of the posterior predictive distribution.

# 4 Numerical experiments

We apply our method of learning robust statistics to two different SBI frameworks — NPE [41] and ABC [5] (see Appendix B. 6 for results on applying our method to neural likelihood estimator [52]). We also compare the performance of our method against RNPE [81], by using the output of the trained summary network in NPE as statistics for RNPE. Experiments are conducted on synthetic data from two time-series models, namely the Ricker model from population ecology [67, 85] and the Ornstein-Uhlenbeck process [18]. Real data experiment on a radio propagation model is presented in Section 5. Analysis of the computational cost of our method and results on a 10-dimensional Gaussian model is presented in Appendix B. 4 and B.5, respectively. The code to reproduce our experiments is available at https://github.com/huangdaolang/robust-sbi.

Ricker model simulates the evolution of population size $N_{t}$ over the course of time $t$ as $N_{t+1}=$ $\exp \left(\theta_{1}\right) N_{t} \exp \left(-N_{t}+e_{t}\right), t=1, \ldots, T$, where $\exp \left(\theta_{1}\right)$ is the growth rate parameter, $e_{t}$ are zeromean iid Gaussian noise terms with variance $\sigma_{e}^{2}$, and $N_{0}=1$. The observations $x_{t}$ are assumed to be Poisson random variables such that $x_{t} \sim \operatorname{Poiss}\left(\theta_{2} N_{t}\right)$. For simplicity, we fix $\sigma_{e}^{2}=0.09$ and estimate $\theta=\left[\theta_{1}, \theta_{2}\right]^{\top}$ using the prior distribution $\mathcal{U}([2,8] \times[0,20])$, and $T=100$ time-steps.

Ornstein-Uhlenbeck process (OUP) is a stochastic differential equation model widely used in financial mathematics and evolutionary biology. The OU process $x_{t}$ is defined as:

$$
\begin{aligned}
x_{t+1} & =x_{t}+\Delta x_{t}, \quad t=1, \ldots, T \\
\Delta x_{t} & =\theta_{1}\left[\exp \left(\theta_{2}\right)-x_{t}\right] \Delta t+0.5 w
\end{aligned}
$$

---

#### Page 7

> **Image description.** The image consists of eight boxplot figures arranged in a 2x4 grid, comparing the performance of different statistical methods.
>
> - **Overall Structure**: The plots are arranged in two rows. The top row displays RMSE (Root Mean Squared Error) values, while the bottom row displays MMD (Maximum Mean Discrepancy) values. The columns represent different models and methods: Ricker (NPE), OUP (NPE), Ricker (ABC), and OUP (ABC).
>
> - **Axes and Labels**:
>   - The vertical axes of the top row are labeled "RMSE" and range from 0.0 to 15.0 (Ricker NPE), 0 to 5 (OUP NPE), 0 to 12 (Ricker ABC), and 0 to 8 (OUP ABC).
>   - The vertical axes of the bottom row are labeled "MMD" and range from 0.00 to 1.25.
>   - The horizontal axes are labeled "$\epsilon$" and show values 0%, 10%, and 20%.
> - **Boxplots**: Each plot contains boxplots representing the distribution of results for different methods at each value of $\epsilon$.
>
>   - In the Ricker (NPE) and OUP (NPE) plots, the methods are NPE (yellow), RNPE (green), and NPE-RS (ours) (blue).
>   - In the Ricker (ABC) and OUP (ABC) plots, the methods are ABC (purple), and ABC-RS (ours) (red).
>   - The boxplots show the median, interquartile range (IQR), and whiskers extending to 1.5 times the IQR.
>
> - **Titles**: Each plot has a title indicating the model and method used, such as "Ricker (NPE)", "OUP (NPE)", "Ricker (ABC)", and "OUP (ABC)".
>
> - **Legend**: Each of the Ricker (NPE) and Ricker (ABC) plots includes a legend indicating the method corresponding to each color.

Figure 2: Performance of the SBI methods in terms of RMSE and MMD for both the Ricker model and OUP. Each box represents the median and interquartile range (IQR), while the whiskers extend to the furthest points within 1.5 times the IQR from the edges of the box. For the well-specified case $(\epsilon=0 \%)$, the proposed NPE-RS and ABC-RS methods perform similar to their counterpart NPE and ABC, respectively. Under misspecification $(\epsilon>0 \%)$, NPE-RS and ABC-RS achieve lower RMSE and MMD values, demonstrating robustness to model misspecification.
where $T=25, \Delta t=0.2, x_{0}=10$, and $w \sim \mathcal{N}(0, \Delta t)$. A uniform prior $\mathcal{U}([0,2] \times[-2,2])$ is placed on the parameters $\theta=\left[\theta_{1}, \theta_{2}\right]^{\top}$.

Contamination model. We test for robustness relative to the $\epsilon$-contamination model [65], similar to [24]. For both the Ricker model and the OUP, the observed data comes from the distribution $\mathbb{Q}=(1-\epsilon) \mathbb{P}_{\theta_{\text {true }}}+\epsilon \mathbb{P}_{\theta_{c}}$, where a large proportion of the data comes from the model with $\theta_{\text {true }}$, while $\epsilon$ proportion of the data comes from the distribution $\mathbb{P}_{\theta_{c}}$. Hence, $\epsilon \in[0,1]$ denotes the level of misspecification within both the models, with $\epsilon=0$ resulting in the well-specified case. We take $\theta_{\text {true }}=[0.5,1.0]^{\top}, \theta_{c}=[-0.5,1.0]^{\top}$ for OUP, and $\theta_{\text {true }}=[4,10]^{\top}, \theta_{c}=[4,100]^{\top}$ for the Ricker model.

Implementation details. We take $m=1000$ samples for the training data and $n=100$ realizations of both the observed and simulated data for each $\theta$. We set $\lambda$ using an additional dataset simulated from the models using $\theta_{\text {true }}$. For the MMD, we take the kernel to be the exponentiated-quadratic kernel $k\left(\mathbf{s}, \mathbf{s}^{\prime}\right)=\exp \left(-\left\|\mathbf{s}-\mathbf{s}^{\prime}\right\|_{2}^{2} / \beta^{2}\right)$, and set its lengthscale $\beta$ using the median heuristic $\beta=\sqrt{\text { med } / 2}$ [42], where med denotes the median of the set of squared two-norm distances $\left\|\mathbf{s}_{i}-\mathbf{s}_{j}\right\|_{2}^{2}$ for all pairs of distinct data points in $\left\{\mathbf{s}_{i}\right\}_{i=1}^{m}$. We use $l=200$ samples of the simulated statistics to estimate the MMD. For the Ricker model, the summary network $\eta_{\psi}$ is composed of 1D convolutional layers, whereas for the OUP, $\eta_{\psi}$ is a combination of bidirectional long short-term memory (LSTM) recurrent modules and 1D convolutional layers. The dimension of the statistic space is set to four for both the models. We take $q$ to be a conditional normalizing flow for all the three NPE methods. We take the tolerance $\delta$ in ABC to be the top $5 \%$ of samples that yield the smallest distance.

Performance metrics. We evaluate the accuracy of the posterior, as well as the posterior predictive distribution of the SBI methods. For the posterior distribution, we compute the root mean squared error (RMSE) as $\left(1 / N \sum_{i=1}^{N}\left(\theta_{i}-\theta_{\text {true }}\right)^{2}\right)^{\frac{1}{2}}$ where $\left\{\theta_{i}\right\}_{i=1}^{N}$ are posterior samples. For the predictive accuracy, we compute the MMD between the observed data and samples from the posterior predictive distribution of each method. As the models simulate high-dimensional data, we use the kernel specialised for time-series from [8]. The lengthscale of this kernel is set to $\beta=1$ for all misspecification levels to facilitate fair comparison.

Results. Figure 2 presents the results for both the Ricker model and the OUP across 100 runs. We observe that both NPE and ABC with our robust statistics (RS) outperform their counterparts, including RNPE, under misspecification $(\epsilon=10 \%$ and $20 \%)$. Moreover, our performance is similar

---

#### Page 8

> **Image description.** This image presents a figure with three panels, each displaying a two-dimensional scatter plot with density estimations along the axes. The panels are arranged horizontally and are labeled (a), (b), and (c). Each panel visualizes the posterior distributions of two parameters, theta1 and theta2, obtained from different methods (NPE, RNPE, and NPE-RS) under varying degrees of model misspecification.
>
> Each panel features a primary scatter plot with theta1 on the x-axis and theta2 on the y-axis. The axes range from approximately 2 to 8 for theta1 and 0 to 20 or 25 for theta2, with numerical labels at intervals. A black horizontal line and a black vertical line intersect at approximately theta1 = 4 and theta2 = 10, representing the true parameter values (theta_true). The scatter plots display density estimations using different colors: orange for NPE, green for RNPE, and blue for NPE-RS (labeled as "ours"). The density estimations are represented by shaded regions, with darker shades indicating higher density.
>
> Above each scatter plot is a one-dimensional density plot of theta1, and to the right is a one-dimensional density plot of theta2, both using the same color scheme to represent the different methods. Dashed gray lines indicate the prior range.
>
> The panels are labeled as follows:
>
> - (a) Well-specified (epsilon = 0)
> - (b) Misspecified (epsilon = 10%)
> - (c) Misspecified (epsilon = 20%)
>
> The legend is located in the upper left panel, indicating the color coding for each method.

Figure 3: Posteriors obtained from NPE, RNPE, and our NPE-RS method for the Ricker model. We perform similar to NPE in the well-specified case, unlike RNPE. Under misspecification, NPE and RNPE posteriors drift away from $\theta_{\text {true }}$, going even beyond the prior range (denoted by dashed gray lines) in the case of NPE. Our method is robust to model misspecification.

to NPE in the well-specified case $(\epsilon=0 \%)$. An instance of the NPE posteriors is shown in Figure 3 for the Ricker model. Our NPE-RS posterior is very similar to the NPE posterior for the well-specified case, whereas the RNPE posterior is underconfident, as noted in [81]. Although RNPE is more robust than NPE under misspecification, its posterior still drifts away from $\theta_{\text {true }}$, while NPE-RS posterior stays around $\theta_{\text {true }}$ even for $\epsilon=20 \%$. This is because our method has the flexibility to choose appropriate statistics based on misspecification level, while RNPE is bound to a fixed choice of pre-selected statistics. Posterior plots for ABC and OUP are given in Appendix B.2.

Prior misspecification. So far our discussion has pertained to the case of likelihood misspecification. However, in Bayesian inference, the prior is also a part of the model. Hence, a potential form of misspecification is prior misspecification, in which the prior gives low or even zero probability to the "true" data generating parameters. We investigate the issue of prior misspecification using the Ricker model as an example. To that end, we modify the prior distribution of $\theta_{2}$ by setting it to $\log \operatorname{Normal}(0.5,1.0)$ while keeping the prior of $\theta_{1}$ unchanged. We use $\theta_{\text {true }}=[4,25]^{\top}$ to generate the observed data, as $\theta_{2}=25$ has very low density under the lognormal prior, and the probability that a value of $\theta_{2} \geq 25$ will be sampled is 0.00372 . The result in Figure 4 shows that NPE and RNPE yield posteriors that do not include the true value of $\theta_{2}$, whereas our NPE-RS posterior is still around $\theta_{\text {true }}$. This highlights the effectiveness of our method to handle cases of prior misspecification as well.

> **Image description.** This image is a scatter plot with density plots along the axes, comparing different methods for estimating parameters in a model.
>
> The main scatter plot shows the relationship between two parameters, theta1 (θ₁) on the x-axis and theta2 (θ₂) on the y-axis. The range of theta1 is approximately 2 to 8, and the range of theta2 is 0 to 80.
>
> Several elements are overlaid on the scatter plot:
>
> - A solid black horizontal line at approximately theta2 = 25 and a solid black vertical line at approximately theta1 = 4. These are labeled in the legend as "θtrue" and represent the true values of the parameters.
> - A dashed gray vertical line at approximately theta1 = 2 and a dashed gray horizontal line at approximately theta2 = 0. These likely represent the prior range.
> - Density plots, shown as filled contours, representing the estimated distributions of the parameters using different methods. These are color-coded:
>   - Orange: NPE
>   - Green: RNPE
>   - Blue: NPE-RS (ours)
>
> The density plots indicate the regions where the estimated parameters are most likely to fall according to each method. The NPE (orange) density is concentrated in a region that is shifted upwards from the true value, while the RNPE (green) density is concentrated in a region that is shifted downwards from the true value. The NPE-RS (blue) density is concentrated around the true value.
>
> Along the top of the scatter plot is a density plot showing the distribution of theta1 for each method. Along the right side of the scatter plot is a density plot showing the distribution of theta2 for each method.

Figure 4: Posteriors from NPE, RNPE, and NPE-RS for Ricker model. NPE-RS is robust to prior misspecification.

Varying regularizer $\lambda$. As mentioned in Section 3, our method converges to NPE as $\lambda$ tends to zero, and to the prior distribution as $\lambda$ becomes high. To investigate this effect, we vary $\lambda$ and measure the distance between the NPE-RS and the NPE posteriors, and between the NPE-RS posterior and the prior. We use the MMD from Equation (7) to measure this distance on the Ricker model in the misspecified case $(\epsilon=20 \%)$. We see in Figure 5(left) that the MMD values between the prior and the NPE-RS posteriors go close to zero for $\lambda=10^{3}$, indicating that our method essentially returns the prior for high values of $\lambda$. On the other hand, the MMD values between NPE and NPE-RS posteriors in Figure 5(middle) are smallest for $\lambda=0.01$ and $\lambda=0.1$. For non-extreme values of $\lambda$ (i.e. $\lambda=1$ or 10), we observe maximum difference between NPE-RS and NPE. This is because the NPE posteriors tend to move out of the prior support under misspecification, whereas the NPE-RS posteriors remain around $\theta_{\text {true }}$, as shown in Figure 3. Finally, the NPE-RS posteriors are also close to the NPE posteriors for small values of $\lambda$ in the well-specified case, as shown in Figure 5(right).

---

#### Page 9

> **Image description.** This image presents three scatter plots arranged horizontally, each displaying the relationship between a variable lambda (λ) on the x-axis and MMD (Maximum Mean Discrepancy) on the y-axis. Each plot also includes a dashed line connecting the mean MMD values for each lambda value.
>
> - **General Layout:** The plots are visually similar, sharing the same y-axis scale (0.0 to 1.5) and a logarithmic x-axis scale with values 0.01, 0.1, 1.0, 10.0, 100.0, and 1000.0. Each plot has a title indicating what is being plotted.
>
> - **Plot 1: "Dist. to prior (misspecified)":** The title indicates this plot shows the distance to the prior in a misspecified setting. The data points are scattered in gray, with a higher density at lower lambda values (0.01 and 0.1), where the MMD values are around 1.0. As lambda increases, the MMD values generally decrease, with most points clustering near 0.0 for lambda values of 100.0 and 1000.0. The dashed black line connects the average MMD values for each lambda, showing a clear downward trend as lambda increases. The average values are marked with blue circles.
>
> - **Plot 2: "Dist. to NPE posterior (misspecified)":** This plot shows the distance to the NPE posterior in a misspecified setting. The scattered gray data points show a more uniform distribution across the lambda values compared to the first plot. The MMD values range from approximately 0.0 to 1.5 for all lambda values. The dashed black line connecting the average MMD values shows a slight increase from lambda 0.01 to 1.0, followed by a gradual decrease as lambda increases further. The average values are marked with blue circles.
>
> - **Plot 3: "Dist. to NPE posterior (well-specified)":** This plot shows the distance to the NPE posterior in a well-specified setting. The scattered gray data points are clustered more tightly compared to the other two plots, with MMD values generally below 0.5. The dashed black line connecting the average MMD values shows a relatively flat trend, with a slight increase as lambda increases. The average values are marked with blue circles.
>
> - **Axes Labels:** All three plots share the same axis labels. The y-axis is labeled "MMD" and the x-axis is labeled "λ".

Figure 5: MMD between the NPE-RS posteriors and (left) the prior in the misspecified setting, (middle) the NPE posteriors in the misspecified setting, (right) the NPE posteriors in the wellspecified setting, for different values of $\lambda$.

# 5 Application to real data: A radio propagation example

Our final experiment involves a stochastic radio channel model, namely the Turin model [45, 63, 78], which is used to simulate radio propagation phenomena in order to test and design wireless communication systems. The model is driven by an underlying point process which is unobservable, thereby rendering its likelihood function intractable. The model simulates high-dimensional complexvalued time-series data, and has four parameters. The parameters govern the starting point, the slope of the decay, the rate of the point-process, and the noise floor of the time-series, as shown in Figure 6. We attempt to fit this model to a real dataset from a measurement campaign [44] for which the model is known to be misspecified [8], on account of the data samples being non-iid. The averaged power of the data (square of the absolute value) as a function of time is shown by the black curve in Figure 6. The data dimension is 801 and we have $n=100$ realizations. We take the power of the complex data in decibels\* as input to the summary network, which consists of 1D convolutional layers, and we set uniform priors for all the four parameters. Descriptions of the model, the data, and the prior ranges are provided in Appendix C.
We fit the model to the real data using NPE, RNPE and our NPE-RS methods. As there is no notion of ground truth in this case, we plot the resulting posterior predictive distributions in Figure 6. Note that the multiple peaks present in the data are not replicated by the Turin model for any method, which is due to the model being misspecified for this dataset. Despite that, our NPE-RS method appears to fit the data well, while NPE performs the worst. Moreover, our method is better than RNPE at matching the starting point of the data, the slope, and the noise floor, with RNPE underestimating all three aspects on average. The MMD between the observed data and the posterior predictive distribution of NPE, RNPE and NPE-RS is $0.11,0.09$, and 0.03 , respectively. Hence, NPE-RS provides a reasonable fit of the model even under misspecification.

## 6 Conclusion

We proposed a simple and elegant solution for tackling misspecification of simulator-based models. As our method relies on learning robust statistics and not on the subsequent inference procedure, it is applicable to any SBI method that utilizes summary statistics. Apart from achieving robust inference under misspecified scenarios, the method performs reasonably well even when the model is well-specified. The proposed method only has one hyperparameter that encodes the trade-off between efficiency and robustness, which can be selected like other neural network hyperparameters, for instance, via a validation set.

[^0]
[^0]: \*In wireless communications, the power of the signal is measured in decibels (dB). A power $P$ in linear scale corresponds to $10 \log _{10}(P) \mathrm{dB}$.

---

#### Page 10

Limitations and future work. A limitation of our method is the increased computational complexity due to the cost of estimating the MMD, which can be alleviated using the sample-efficient MMD estimator from [10] or quasi-Monte Carlo points [56]. Moreover, as our method utilizes the observed statistic during the training procedure, the corresponding NPE is not amortized anymore a limitation we share with RNPE. Thus, working on robust NPE methods which are still amortized (to some extent) is an interesting direction for future research. An obvious extension of the work could be to investigate if the ideas translate to likelihood-free model selection methods [68, 76] which can suffer from similar problems as SBI if all (or some) of the candidate models are misspecified.
