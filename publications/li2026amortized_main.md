```
@article{li2026amortized,
  title={Amortized Bayesian Workflow},
  author={Chengkun Li and Aki Vehtari and Paul-Christian B{\"u}rkner and Stefan T. Radev and Luigi Acerbi and Marvin Schmitt},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=osV7adJlKD}
}
```

---

# Amortized Bayesian Workflow

Chengkun Li University of Helsinki

chengkun.li@helsinki.fi

Aki Vehtari ELLIS Institute Finland, Aalto University

aki.vehtari@aalto.fi

Paul- Christian Burkner TU Dortmund University

paul.buerkner@tu- dortmund.de

Stefan T. Radev Rensselaer Polytechnic Institute

radevs@rpi.edu

Luigi Acerbi University of Helsinki

luigi.acerbi@helsinki.fi

Marvin Schmitt Independent Scientist

mail.marvinschmitt@gmail.com

## Abstract

Bayesian inference often faces a trade- off between computational speed and sampling accuracy. We propose an adaptive workflow that integrates rapid amortized inference with gold- standard MCMC techniques to achieve a favorable combination of both speed and accuracy when performing inference on many observed datasets. Our approach uses principled diagnostics to guide the choice of inference method for each dataset, moving along the Pareto front from fast amortized sampling via generative neural networks to slower but guaranteed- accurate MCMC when needed. By reusing computations across steps, our workflow synergizes amortized and MCMC- based inference. We demonstrate the effectiveness of this integrated approach on several synthetic and real- world problems with tens of thousands of datasets, showing efficiency gains while maintaining high posterior quality.

## 1 Introduction

In many statistical modeling applications, from finance to biology and neuroscience, we often aim to infer unknown parameters \(\theta\) from observables \(y\) modeled as a joint distribution \(p(\theta ,y)\) (e.g., Raulo et al., 2023; Seaton et al., 2023; George et al., 2022; Landmeyer et al., 2020; Chen et al., 2019; Malen et al., 2022; Schneider et al., 2018; Tsilifis & Ghosh, 2022). The posterior \(p(\theta |y)\) is the statistically optimal solution to this inverse problem, and there are different computational approaches to approximate this target distribution.

Markov chain Monte Carlo (MCMC) methods constitute the most popular family of posterior sampling algorithms and still remain the gold standard for modern Bayesian inference due to their theoretical guarantees and powerful diagnostics (Gelman et al., 2013; 2020). MCMC methods yield autocorrelated draws conditional on a fixed dataset \(y_{\mathrm{obs}}\) . As a consequence, the probabilistic model has to be re- fit for each new dataset, which involves repeating the entire MCMC procedure from scratch. Modern implementations equip MCMC with state- of- the- art extensions, for example, through Hamiltonian dynamics (HMC; Neal, 2011), by minimizing the required tuning by users (NUTS; Hoffman & Gelman, 2014), or by parallelizing thousands of chains on GPU hardware (ChEES- HMC; Hoffman et al., 2021). The well- established Bayesian workflow (Gelman et al., 2020) leverages these tools in an iterative process of model specification, fitting, evaluation, and revision. While powerful, this approach becomes computationally burdensome when applied independently to large collections of datasets.

Differently, amortized Bayesian inference (ABI) aims to learn a direct mapping from observables \(y\) to the corresponding posterior \(p(\theta | y)\) , using flexible function approximators such as deep neural networks (Cranmer et al., 2020; Radev et al., 2020; Greenberg et al., 2019; Papamakarios et al., 2021; Wildberger et al., 2023; Sharrock et al., 2024; Zammit- Mangion et al., 2025). Amortized inference typically follows a two- stage approach: (i) a training stage, where neural networks learn to distill information from the probabilistic model based on simulated examples of observations and parameters \((\theta , y) \sim p(\theta) p(y | \theta)\) ; and (ii) an inference stage where the neural networks approximate the posterior distribution for an unseen dataset \(y_{\mathrm{obs}}\) in near- instant time without repeating the training stage. In other words: The upfront training cost is amortized by negligible inference cost on arbitrary amounts of unseen test data. Owing to its reliance on simulated data, amortized inference in this form overlaps with simulation- based inference (Cranmer et al., 2020), which originated from posterior computations for models with intractable likelihood.

However, amortized inference lacks the powerful diagnostics and gold- standard guarantees associated with MCMC samplers in the standard Bayesian workflow (Gelman et al., 2020). Yet, applying a standard workflow is computationally prohibitive at scale. In modern Bayesian computation, MCMC and ABI occupy different ends of a Pareto frontier (see Figure 1): the former provides reliable accuracy at high cost, while the latter offers near- instant inference speed with limited per- dataset reliability (Hermans et al., 2022; Schmitt et al., 2023; Lueckmann et al., 2021).

> **Image description.** A two-dimensional scatter plot titled "Figure 1" that illustrates the trade-off between accuracy guarantees and inference speed for various computational methods.
>
> The graph is structured with two axes:
> *   **Y-axis:** Labeled "Accuracy guarantees," ranging from "weak" at the bottom to "strong" at the top.
> *   **X-axis:** Labeled "Inference speed per data set," ranging from "slow" on the left to "instant" on the right.
>
> The visual elements include:
> *   **Data Points:** A cluster of numerous gray circles represents various computational approaches.
> *   **Pareto Front:** A distinct, curved line labeled "Pareto front" connects the most efficient points, representing the optimal balance between high accuracy and high speed.
> *   **Labeled Methods:** Several specific methods are marked with labels and arrows pointing to their respective positions on the graph:
>     *   **MCMC:** Located in the upper-left quadrant, indicating high accuracy guarantees but slow inference speed.
>     *   **Amortized inference:** Located in the lower-right quadrant, indicating near-instant inference speed but weaker accuracy guarantees.
>     *   **PSI:** A point located near the Pareto front, positioned between the MCMC and Amortized inference extremes.
>     *   **Amortized inference (second point):** Another point labeled "Amortized inference" is shown slightly higher and slightly slower than the extreme right point, still maintaining a high speed relative to the MCMC methods.
>
> The overall arrangement visually demonstrates that methods like MCMC achieve the highest accuracy but at the cost of slow inference, while Amortized Inference achieves the fastest inference but with lower accuracy guarantees. The Pareto front highlights the set of methods that offer the best possible trade-off between these two competing metrics.

<center>Figure 1: Our workflow adaptively moves along the Pareto front and reuses previous computations. </center>

In this paper, we propose an adaptive workflow that consistently yields high- quality posterior draws while remaining computationally efficient. Our proposed workflow moves along the Pareto front, enabling fast- and- accurate inference when possible, and

slow- but- guaranteed- accurate inference when necessary (see Figure 1). It combines the strengths of ABI and MCMC by incorporating diagnostic checks to guide inference decisions and reuse computations wherever possible. The resulting amortized Bayesian workflow therefore offers a principled, scalable, and diagnostic- driven approach for efficient posterior inference on many observed datasets; see Figure 2 for a conceptual overview.1 To summarize, our contributions are:

- Design of—and systematic guidance through—an adaptive Bayesian workflow for accelerating Bayesian inference, which combines the strengths of amortized inference, importance sampling, and MCMC in a theoretically motivated and modular manner.

- Empirical validation of the workflow and of its inference speedup, demonstrating the applicability of the workflow on both synthetic and large-scale, real-world problems.

## 2 Integrating amortized inference into the Bayesian workflow

Our adaptive workflow starts with neural network training to enable subsequent amortized inference on a large number of unseen datasets—typically well into tens of thousands. This training phase is conceptually identical to standalone amortized inference training (e.g., Radev et al., 2020; Cranmer et al., 2020). For the inference phase, however, we develop a principled control flow that guides the analysis. Based on state- of- the- art diagnostics that are tailored to each step along the workflow, we propose decision criteria to select the appropriate inference algorithm for each observed dataset. In order to optimize the overall efficiency, our workflow contains mechanisms to reuse previous computations along the way.

> **Image description.** A complex technical flow diagram illustrating an adaptive workflow for Bayesian sampling, titled "Figure 2." The diagram is structured into three sequential steps (Step 1, Step 2, and Step 3), showing how data is processed and reused, transitioning from fast, approximate methods to slower, more accurate ones.
>
> **Overall Structure and Data Flow:**
> The workflow proceeds horizontally from left to right. The primary data representation throughout the steps is a cloud of blue dots, labeled "Example draws." Blue dashed arrows labeled "reuse draws" indicate that the output of the previous step's draws is fed as input into the next step, demonstrating the adaptive nature of the process. Red arrows labeled "no" indicate failure points where the workflow might need to be revised or halted.
>
> **Step 1: Amortized Inference**
> *   **Input:** The process begins with a block labeled "One dataset $y_{obs}$."
> *   **Process:** The main box is labeled "Step 1 Amortized Inference." Inside, a sub-block identifies the "Neural density estimator $q_{\phi}$."
> *   **Output:** The process generates a cloud of blue dots ("Example draws").
> *   **Decision:** A diamond shape asks "Diagnostics OK?".
> *   **Outcome:** If successful, a green grid block labeled "Accept draws" is generated. If unsuccessful, a red arrow labeled "no" exits the step.
>
> **Step 2: Pareto-smoothed Importance Sampling (PSIS)**
> *   **Input:** This step receives the blue dot cloud from Step 1.
> *   **Process:** The main box is labeled "Step 2 Pareto-smoothed Importance Sampling." It contains a sub-block for "Importance weights."
> *   **Output:** It generates a new cloud of blue dots ("Example draws").
> *   **Decision:** A diamond shape asks "Diagnostics OK?".
> *   **Outcome:** If successful, a green grid block labeled "Accept draws" is generated. If unsuccessful, a red arrow labeled "no" exits the step.
>
> **Step 3: ChEES-HMC**
> *   **Input:** This step receives the blue dot cloud from Step 2.
> *   **Process:** The main box is labeled "Step 3 ChEES-HMC with amortized init." It contains a sub-block for "Initialize $S$ superchains."
> *   **Output:** It generates a final cloud of blue dots ("Example draws").
> *   **Decision:** A diamond shape asks "Diagnostics OK?".
> *   **Outcome:** If successful, a green grid block labeled "Accept draws" is generated. If unsuccessful, a red box labeled "Use NUTS or revise model" is triggered.
>
> **Data Quantities and Acceptance Rates:**
> Below the main flow, three rows of green grids provide quantitative data regarding the number of datasets processed and accepted at each stage:
> 1.  **Step 1:** "K=256 observed datasets" leads to "Accept amortized draws for 192/256 data sets."
> 2.  **Step 2:** "PSIS on remaining 64 datasets" leads to "Accept PSIS for 56/64 data sets."
> 3.  **Step 3:** "ChEES-HMC on remaining 8 data sets" leads to "Accept ChEES-HMC for 8/8 data sets."
>
> The caption below the figure summarizes the workflow: "Figure 2: Our adaptive workflow leverages near-instant amortized posterior sampling when possible and gradually resorts to slower—but more accurate—sampling algorithms. As indicated by the blue dashed arrows, we reuse the $S$ draws from the amortized posterior in Step 1 for the subsequent steps in the form of PSIS proposals (Step 2) and initial values in ChEES-HMC (Step 3)."

<center>Figure 2: Our adaptive workflow leverages near-instant amortized posterior sampling when possible and gradually resorts to slower—but more accurate—sampling algorithms. As indicated by the blue dashed arrows, we reuse the \(S\) draws from the amortized posterior in Step 1 for the subsequent steps in the form of PSIS proposals (Step 2) and initial values in ChEES-HMC (Step 3). </center>

### 2.1 Training phase: simulation-based optimization

In ABI, a neural estimator \(q_{\phi}\) with trainable parameters \(\phi\) typically minimizes a strictly proper scoring rule \(\mathcal{S}\) (Gneiting & Raftery, 2007; Pacchiardi & Dutta, 2022) in expectation over the joint model \(p(\theta , y) = p(\theta)p(y | \theta)\) ,

\[\phi = \arg \min_{\phi}\mathbb{E}_{(\theta ,y)\sim p(\theta ,y)}\left[\mathcal{S}\big(q_{\phi}(\cdot |y),\theta \big)\right]. \quad (1)\]

A popular choice is the logarithmic scoring rule, \(\mathcal{S}\big(q_{\phi}(\cdot |y),\theta \big)\coloneqq - \log q_{\phi}(\theta |y)\) , which amounts to the forward Kullback- Leibler (KL) objective used for training normalizing flows in ABI (Greenberg et al., 2019; Radev et al., 2020). Score- based formulations that target a time- dependent gradient \(\nabla_{\theta_{t}}\log p(\theta_{t}|y)\) are also possible (Sharrock et al., 2024; Gloeckler et al., 2024). Since most Bayesian models are generative by design, we can readily simulate \(M\) synthetic tuples of parameters and corresponding observations from the joint probabilistic model,

\[(\theta^{(m)},y^{(m)})\sim p(\theta ,y)\quad \Leftrightarrow \quad \theta^{(m)}\sim p(\theta),y^{(m)}\sim p(y|\theta)\mathrm{~for~}m = 1,\ldots ,M, \quad (2)\]

which results in the training set \(\{(\theta^{(m)},y^{(m)})\}_{m = 1}^{M}\) for optimizing Eq. 1. Throughout this paper, we use coupling- based normalizing flows (Durkan et al., 2019; Papamakarios et al., 2021) as a flexible conditional density estimator \(q_{\phi}\) and the forward KL divergence as the training objective. However, our proposed workflow is agnostic to the specific choice of generative backbone used for amortization, as long as the model supports efficient sampling (see Section 2.2.1) and density evaluations (see Section 2.2.2).

Diagnostics. Since the neural network training algorithm hinges on simulated data, we cannot evaluate the amortized posterior estimator on real data just yet. However, we can easily simulate a synthetic test set \(\{(\theta_{\phi}^{(j)},y^{(j)})\}_{j = 1}^{J}\) of size \(J\) from the joint model via Eq. 2. In this closed- world setting, we know which "true" parameter vector \(\theta_{\phi}^{(j)}\) generated each simulated test dataset \(y^{(j)}\) . A key diagnostic for evaluating the amortized posterior estimator is simulation- based calibration checking (SBC; Talts et al., 2018; Sáilynoja et al., 2022; Modrák et al., 2025; Yao & Domke, 2023). Formally, SBC involves (1) defining a test quantity \(f:\Theta \times Y\to \mathbb{R}\) (e.g., marginal projections \(\theta\) or the log likelihood \(p(y|\theta)\) ), (2) computing this statistic for the true data- generating parameter \(\theta_{\phi}^{(j)}\) , and (3) comparing it to the empirical distribution of the same statistic derived from amortized posterior draws given \(y^{(j)}\) (Modrák et al., 2025). The rank of the true statistic within the posterior draws should be uniformly distributed if the amortized posterior estimator is well- calibrated.

We recommend assessing uniformity using the graphical approach by Sáilynoja et al. (2022), which reveals the type of miscalibration present (e.g., bias or over- /under- dispersion) and is therefore useful for guiding

improvements to amortized training. The choice of test quantity in SBC determines the sensitivity of the check; for example, the log- likelihood test quantity is typically more sensitive at detecting discrepancies than marginal projections (Modrák et al., 2025); using expressive neural classifiers is also possible (Yao & Domke, 2023). We further note a trade- off: imposing stricter criteria can improve the fidelity of the amortized estimator but will also tend to reject otherwise practically useful amortized estimators.

By default, we use marginal projections as the test quantities for SBC and complement SBC checking with parameter recovery checking, where parameter estimates are compared against known ground- truth parameters via direct visualization (Radev et al., 2020; 2023). Parameter recovery checking provides practical insight into whether the learned inverse mapping from \(y\) to \(\theta\) is effective and helps mitigate a known failure mode of SBC with marginal projections as test quantities, in which the posterior approximation simply recovers the prior. We refer to Appendix A for further details and corresponding pseudocodes.

Note. Amortized inference lies at the intersection of Bayesian modeling and deep learning, unlocking massive potential for scalable posterior inference. However, this also comes with the practical challenges inherent to training deep neural networks. While a detailed treatment of neural architecture design and optimization exceeds the scope of this paper, practitioners can use established simulation- based inference libraries like sbi (Boelts et al., 2025) or BayesFlow (Radev et al., 2023), which provide modern plug- and- play components as well as sensible defaults for a wide range of applications. We summarize a set of best practices and actionable recommendations for training amortized posterior estimators in Appendix B.

Training phase: If simulation- based calibration checking and parameter recovery diagnostics pass, proceed to Step 1. Otherwise, tune the training hyperparameters (e.g., simulation budget, training epochs, learning rate, or neural network architecture) and re- train the amortized network.

### 2.2 Inference phase: posterior approximation on observed datasets

Once the amortized estimator is capable of yielding sufficiently accurate posterior draws in closed- world settings (i.e., in- distribution), we use the pre- trained neural network to achieve rapid amortized posterior inference on a total of \(K\) observed datasets \(\{y_{\mathrm{obs}}^{(k)}\}_{k = 1}^{K}\) . Recall that a given pre- trained amortized neural estimator may be perfectly suitable for some real datasets while it is utterly untrustworthy for others. Therefore, we want to assess on a per- dataset basis whether the amortized posterior draws are trustworthy and should be accepted, or whether we should proceed to a slower algorithm with stronger accuracy guarantees. The diagnostics in the inference phase are evaluated conditionally on each observed dataset, with the ultimate goal of determining whether the set of current posterior draws is acceptable for that specific dataset.

#### 2.2.1 Step 1: Amortized posterior draws

We want to exploit the rapid sampling capabilities of the amortized posterior estimator \(q_{\phi}\) as much as possible, as long as the sampled posteriors are trustworthy according to a set of principled diagnostics. Therefore, the natural first step for each observed dataset \(y_{\mathrm{obs}}^{(k)}\) is to query the amortized posterior and sample \(S\) posterior draws \(\hat{\theta}_{1}^{(k)},\ldots ,\hat{\theta}_{S}^{(k)}\sim q_{\phi}(\theta |y^{(k)})\) in near- instant time (see Figure 2, first panel).

Diagnostics. Like other neural network approaches (Yang et al., 2024), amortized inference may yield unfaithful results under distribution shifts (Schmitt et al., 2023; Ward et al., 2022; Huang et al., 2023). To address this, we detect whether an observed dataset \(y_{\mathrm{obs}}\) is out- of- distribution (OOD) relative to the data- generating process

> **Image description.** A technical line graph illustrating probability density functions (PDF) for different data sets based on Mahalanobis distance. The graph is titled "Figure 3: Illustration of our sampling-based hy-" (the caption is truncated).
>
> The visualization uses a standard Cartesian coordinate system:
> *   **Y-axis:** Labeled "Density," ranging from 0.0 to 0.4, representing the probability density.
> *   **X-axis:** Labeled "Mahalanobis distance," ranging from 0 to 25.
>
> Three distinct visual elements are plotted on the graph, each corresponding to an entry in the legend:
>
> 1.  **Training samples (null distribution):** Represented by a smooth, yellow/orange curve. This distribution is centered at a low Mahalanobis distance (approximately 2) and reaches a peak density of about 0.35.
> 2.  **Test samples:** Represented by a smooth, blue curve. This distribution is slightly wider and centered at a higher Mahalanobis distance (approximately 3.5), with a peak density of around 0.25.
> 3.  **OOD cut-off:** Represented by a sharp, vertical red line. This line acts as a threshold, positioned at a Mahalanobis distance of approximately 7.
>
> The legend, located in the upper right corner, clearly identifies these three elements:
> *   Yellow box: "Training samples (null distribution)"
> *   Blue box: "Test samples"
> *   Red box: "OOD cut-off"
>
> The overall visual pattern demonstrates how the density of the training and test data distributions is compared against a defined threshold (the OOD cut-off) to identify potential outliers or out-of-distribution data points. The red line at Mahalanobis distance 7 serves as the boundary for classification.

<center>Figure 3: Illustration of our sampling-based hypothesis test that flags OOD datasets (to the right of the OOD cut-off). </center>

\(p(\theta ,y)\) . We first compute a low- dimensional summary statistic \(s(y)\in \mathbb{R}^{d}\) for each dataset.2 The summary statistics from the training dataset \(\{y^{(m)}\}_{m = 1}^{M}\) are used to approximate the Mahalanobis distance by estimating their empirical mean \(\mu_{s}\) and covariance \(\Sigma_{s}\) . Then, for any test dataset \(y\) , its Mahalanobis distance to the training set is:

\[D_{M}(y) = \sqrt{(s(y) - \mu_{s})^{\top}\Sigma_{s}^{-1}(s(y) - \mu_{s})}. \quad (3)\]

We compute \(\{D_{M}(y^{(m)})\}_{m = 1}^{M}\) for all training datasets to establish a frequentist sampling distribution of distances under the null hypothesis (i.e., of in- distribution datasets). Given a new observed dataset \(y_{\mathrm{obs}}\) , we compare its Mahalanobis distance \(D_{M}(y_{\mathrm{obs}})\) to the empirical distribution of training distances. We define the OOD rejection rule as:

\[\mathrm{Reject}_{\mathrm{OOD}}(y_{\mathrm{obs}}) = \mathbb{I}\left\{D_{M}(y_{\mathrm{obs}}) > \mathrm{Quantile}_{1 - \alpha}\left(\{D_{M}(y^{(m)})\}_{m = 1}^{M}\right)\right\} , \quad (4)\]

where \(\alpha\) is by default set to 0.05 and we flag datasets whose Mahalanobis distances fall in the right \(\alpha\) tail of the empirical training distances as out- of- distribution (see Figure 3). The type- I error rate \(\alpha\) (false rejection) of this test can be set relatively high to obtain a conservative test that will flag many datasets for detailed investigation in further steps of our workflow.

In a nutshell, this is a sampling- based hypothesis test for distribution shifts, similar in spirit to the kernel- based test proposed by Schmitt et al. (2023). Since the amortized estimator has no guarantees nor known error bounds for data outside of the empirical support of the joint model \(p(\theta , y)\) (Elsemüller et al., 2024; Schmitt et al., 2023; Frazier et al., 2024; Elsemüller et al., 2025), we propagate such out- of- distribution datasets to Step 2. It is worth noting that a smaller Mahalanobis distance does not necessarily imply better posterior quality and that this OOD test is only intended to filter out datasets that are most likely to be problematic for the amortized estimator—specifically, those requiring extrapolation outside the ellipsoid defined by the training summary statistics.

Alternative diagnostics. In addition to the proposed out- of- distribution test, more sophisticated data- conditional diagnostics can further assess the accuracy of amortized posterior draws for individual datasets and enhance the reliability of accepted amortized draws. Examples include posterior simulation- based calibration checking (posterior SBC; Sailyhoja et al., 2025) or the local classifier two- sample test (L- C2ST; Linhart et al., 2023), to name a few. These diagnostics each offer distinct advantages and limitations, but typically require substantially more computation than the OOD test.

Posterior SBC is conceptually straightforward and offers necessary conditions for the accuracy of amortized posterior samples by assessing consistency. However, it requires additional simulations for each test dataset and requires training the amortized estimator on inputs that effectively double the size of the original observations. L- C2ST, which trains classifiers to distinguish between \(q_{\phi}(\theta |y)p(y)\) and the joint distribution \(p(\theta , y)\) , provides theoretically sufficient and necessary conditions for amortized inference accuracy. In practice, however, its effectiveness can be very sensitive to several factors, including classifier design choices (e.g., data pre- processing and optimization strategies), classifier calibration, and the relative sizes of the simulation budgets allocated to classifier training and amortized estimator training.

The choice to apply these additional diagnostics depends on context- specific factors, including the number of observed datasets, the relative computational cost of simulations versus likelihood evaluations,3 and the dimensionality of the observations. Ultimately, whether amortized posterior draws are deemed acceptable hinges on the accuracy requirements of the specific application. By default, we recommend the OOD test for its simplicity, efficiency, and suitability as a first- line diagnostic.

Step 1: If the observed dataset passes the OOD test (i.e., Mahalanobis distance is below the threshold), accept the amortized draws; otherwise, proceed to Step 2.

#### 2.2.2 Step 2: Pareto-smoothed importance sampling

In this step, we use Pareto- smoothed importance sampling (PSIS) (Vehtari et al., 2024) to both improve and assess the quality of the amortized posterior draws of datasets which have previously been rejected (see Figure 2, second panel). Based on the amortized posterior draws from Step 1, PSIS computes importance weights \(w_{s}^{(k)} = p(y^{(k)} | \hat{\theta}_{s}) p(\hat{\theta}_{s}) / q_{\phi}(\hat{\theta}_{s} | y^{(k)})\) for each observed dataset \(y^{(k)}\) (as in default importance sampling). Then, PSIS fits a generalized Pareto distribution to the largest importance weights, which in turn is used to smooth the tail of the weight distribution (Vehtari et al., 2024). Finally, these smoothed importance weights are used for computing posterior expectations and for improving the posterior draws with the sampling importance resampling (SIR) scheme (Rubin, 1988). While the utility of standard importance sampling for improving neural posterior draws has previously been investigated (Dax et al., 2023), we specifically use the PSIS algorithm, which is self- diagnosing (see Diagnostics below) and therefore better suited for a principled workflow. Further details of PSIS are provided in Appendix A.

Diagnostics. We use the Pareto- \(\hat{k}\) diagnostic to gauge the fidelity of the PSIS- refined posterior draws. Pareto- \(\hat{k}\) is the estimated shape parameter of the generalized Pareto distribution and quantifies the tail heaviness of the largest importance weights. According to Vehtari et al. (2024), for moderate sample size ( \(S > 2000\) ), Pareto- \(\hat{k} \leq 0.7\) indicates that PSIS estimates are reliable; \(4\) when \(\hat{k} > 0.7\) , the minimum sample size for obtaining a reliable Monte Carlo estimate through (Pareto- smoothed) importance sampling rapidly grows infeasibly large in practice, implying that the amortized posterior is a poor proposal for importance sampling correction and the corresponding dataset should be routed to Step 3. This \(\hat{k}\) threshold is consistent with the established practice of using PSIS to improve and assess the quality of posterior approximations obtained from variational inference (Yao et al., 2018; Dhaka et al., 2021; Zhang et al., 2022).

Note. The posterior estimator in ABI is typically mode- covering since it optimizes the forward KL divergence in Eq. 1. When the neural network training is insufficient (e.g., small simulation budget or poorly optimized network), this may lead to overdispersed posteriors. Fortunately, this tends to err in the right direction, and PSIS can generally mitigate overdispersed mode- covering draws in low to moderate dimensions (Dhaka et al., 2021). In contrast, variational inference typically optimizes the reverse KL divergence (Rezende & Mohamed, 2015), which implies mode- seeking behavior that is less favorable for importance sampling.

Step 2: If Pareto- \(\hat{k} \leq 0.7\) , accept the importance sampling results; otherwise, proceed to Step 3.

#### 2.2.3 Step 3: Many-chains MCMC with amortized initializations

If PSIS does not yield satisfactory results, we resort to an MCMC sampling scheme as a safe fallback option. In our amortized workflow, the MCMC step is augmented by reusing computations from the previous steps as initialization values. In principle, this step can incorporate any MCMC algorithm suited to the problem at hand. Examples include slice sampling for models with non- differentiable likelihoods (Neal, 2003), or HMC (Neal, 2011) samplers when gradients are available.

In this work, we use the ChEES- HMC algorithm (Hoffman et al., 2021) as an instantiation of MCMC. Most notably, ChEES- HMC supports the execution of thousands of parallel chains on a GPU for high- throughput sampling (Soutsov et al., 2024). Amortized posterior draws from previous steps provide a natural and convenient choice for initializing MCMC chains to accelerate convergence (Figure 4). This approach is conceptually similar to using methods like parallel quasi- Newton variational inference (i.e., Pathfinder; Zhang et al., 2022) to obtain initial values for MCMC chains. However, the amortized initial values are drawn in parallel in near- instant time, while Pathfinder requires re- fitting the variational approximation for each new observed dataset. For the purpose of ChEES- HMC initializations

with multimodal posterior distributions, it is again desirable that the amortized posterior draws are typically mass- covering (cf. Step 2). See Appendix A for additional details on the ChEES- HMC algorithm.

Diagnostics. In this last step, we use the nested \(\widehat{R}\) diagnostic (Margossian et al., 2024), which is specifically designed to assess the convergence of the many- but- short MCMC chains.5 If the diagnostics in this step indicate unreliable inference, we recommend resorting to the overarching Bayesian workflow (Gelman et al., 2020) and addressing the computational issues that even persist when using the (ChEES- )HMC algorithm. This could involve increasing the number of warmup iterations, using the established NUTS- HMC algorithm (Hoffman & Gelman, 2014; Carpenter et al., 2017), or revising the Bayesian model specification and parametrization.

> **Image description.** A 2D scatter plot titled "Figure 4" that visualizes the distribution of samples within a parameter space defined by two variables, $\theta_1$ and $\theta_2$.
>
> The plot features a horizontal axis labeled $\theta_1$ and a vertical axis labeled $\theta_2$. The data points are clustered around a central, curved, black contour, which represents the "Target posterior." This posterior distribution is elongated and elliptical, suggesting a specific, concentrated region of high probability in the parameter space.
>
> Two distinct sets of data points are plotted:
> 1.  **Amortized initializations:** These are represented by numerous purple 'x' marks. These points are scattered across the plot, both inside and outside the central target posterior, indicating the starting points for the sampling process.
> 2.  **ChEES-HMC samples:** These are represented by numerous small green dots. These samples are concentrated primarily within the boundaries of the black target posterior, demonstrating the successful sampling of the target distribution.
>
> A legend is provided in the upper right corner of the plot, clearly defining the visual elements:
> *   A purple 'x' symbol is labeled "Amortized initializations."
> *   A green dot symbol is labeled "ChEES-HMC samples."
> *   A solid black line/contour is labeled "Target posterior."
>
> The overall visual arrangement illustrates the process of using initial draws (purple 'x's) to generate samples (green dots) that converge toward the true underlying distribution (the black target posterior). The caption below the figure reads: "Figure 4: We initialize many ChEES-HMC chains with amortized draws."

<center>Figure 4: We initialize many ChEES-HMC chains with amortized draws. </center>

Step 3: If (nested) \(\widehat{R}\) is below the convergence threshold (e.g., 1.01), accept the MCMC draws. Otherwise, increase warm- up or revise the model according to the standard Bayesian workflow (Gelman et al., 2020).

### 2.3 Related work

Both simulation- based inference and amortized inference have seen rapid progress over the past decade (Zammit- Mangion et al., 2025; Cranmer et al., 2020; Lavin et al., 2021), driven by the need to perform Bayesian inference in complex models with intractable likelihoods (e.g., Dingeldein et al., 2024; Wehenkel et al., 2024; Zhou et al., 2024; Ghaderi- Kangavari et al., 2023; von Krause et al., 2022; Bieringer et al., 2021; Radev et al., 2021). These advances have been fueled by modern generative modeling, such as normalizing flows (Papamakarios et al., 2021; Radev et al., 2020; Greenberg et al., 2019), transformers (Müller et al., 2022; Chang et al., 2025; Whittle et al., 2025), diffusion models (Song et al., 2021; Sharrock et al., 2024; Linhart et al., 2024; Geffner et al., 2023; Gloeckler et al., 2024), consistency models (Song et al., 2023; Schmitt et al., 2024b), and flow matching (Lipman et al., 2023; Wildberger et al., 2023). Practical software toolkits such as BayesFlow (Radev et al., 2023) and sbi (Boelts et al., 2025) further make these simulation- based inference techniques accessible to practitioners in user- friendly interfaces.

To address the potential systematic errors of (amortized) neural posteriors, several works propose corrections using importance reweighting schemes (Dax et al., 2023; Starostin et al., 2025), augmented training objectives (Delaunoy et al., 2022; Mishra et al., 2025; Orozco et al., 2025; Schmitt et al., 2024a), or post- hoc corrections (Siahkoohi et al., 2023). Simultaneously, hybrid approaches that combine density estimators with MCMC have gained traction (Salimans et al., 2015; Hoffman et al., 2019; Gabriel et al., 2022; Midgley et al., 2022; Arbel et al., 2021; Cabezas et al., 2024; Grenioux et al., 2023). These include using variational approximations or learned flows as preconditioners for MCMC (Hoffman et al., 2019; Cabezas & Nemeth, 2023), adaptive proposal mechanisms (Parno & Marzouk, 2018; Gabriel et al., 2022), and initialization strategies to accelerate convergence or improve diagnostics (Zhang et al., 2022; Wang et al., 2023; Starostin et al., 2025).

More broadly, automated Bayesian inference has been a central design goal of probabilistic programming systems such as Stan (Carpenter et al., 2017), PyMC (Oriol et al., 2023), (Num)Pyro (Bingham et al., 2019; Phan et al., 2019). These libraries provide general- purpose inference engines—typically gradient- based MCMC and variational inference—that can be applied to a wide range of likelihood- based models and are accompanied by well- developed diagnostic recommendations and workflow guidelines (Gelman et al., 2020). However, they do not natively support amortized inference across many datasets, and inference must be rerun from scratch for each dataset.

Our proposed workflow builds on and complements these lines of work by integrating amortized inference, likelihood- based correction, and many- chain MCMC into a unified, modular, and diagnostic- driven pipeline for accelerating Bayesian inference. It dynamically adapts the inference strategy to the dataset at hand—using

amortized posterior draws when they are adequate and escalating to PSIS and MCMC otherwise—thereby improving the robustness of amortized inference and the overall efficiency of posterior computation. This modular design provides a practical foundation for principled amortized inference across diverse data regimes.

## 3 Experiments

In this section, we empirically evaluate the effectiveness of our proposed amortized Bayesian workflow across various synthetic and real- world problems. We also examine how reusing amortized posterior draws in subsequent steps can improve the downstream sampling performance. The source code to reproduce all experiments is available in the supplementary material.

### 3.1 Procedure

Training settings. For each problem, we begin by training the amortized posterior estimator on simulated parameter- observation pairs (i.e., simulation- based training). We verify that the model performance is satisfactory in a closed- world setting, as diagnosed by simulation- based calibration and parameter recovery checking (see Section 2.1). Details on diagnostic results, simulation budgets, and training hyperparameters are provided in Appendix C.

Inference settings. For the out- of- distribution diagnostics in Step 1, we use the \(\alpha = 0.05\) as the rejection threshold. We compute Mahalanobis distances in the summary statistics using 10,000 training simulations. We draw 2,000 posterior samples from the amortized posterior \(q_{\phi}\) at Step 1. In Step 2, we correct the amortized draws using PSIS, rejecting draws if Pareto- \(\hat{k} > 0.7\) . Step 3 uses ChEES- HMC with convergence determined by nested \(\hat{R} < 1.01\) . We run 2048 chains in parallel (16 superchains, each with 128 subchains), with 200 warmup steps and a single sampling step, for a total of 2048 posterior draws.

Evaluation metrics. To assess the quality of posterior draws from our workflow, we compare them to reference posterior draws using two evaluation metrics: the 1- Wasserstein distance (W1) and the mean marginal total variation distance (MMTV). The W1 distance quantifies the overall discrepancy between full joint distributions. MMTV measures the lack of overlap between marginal distributions and takes value in the range \([0,1]\) ; for example, an MMTV value of 0.2 implies that, on average, the approximate posterior draws and reference draws share an 80% overlap for their marginal distributions. For both metrics, lower values indicate better posterior approximation quality. As a rule of thumb, MMTV values below 0.2 indicate good posterior approximation fidelity (Acerbi, 2020; Li et al., 2025).

### 3.2 Applications

We apply the proposed workflow to four posterior inference problems, including both simulated benchmarks and real- world experimental datasets. These case studies were chosen to reflect a range of commonly encountered statistical inference scenarios, including classical distributional parameter estimation and analyses of large- scale datasets arising in psychology and cognitive modeling. We describe each problem briefly below, with further details provided in Appendix C.

Generalized extreme value distribution (GEV). We consider parameter inference for the generalized extreme value (GEV) distribution, which models the maxima of samples from a distribution family. Each observation \(y_{i}\) is modeled as:

\[y_{i}\sim \mathrm{GEV}(\mu ,\sigma ,\xi), \quad (5)\]

where \(\mu \in \mathbb{R}\) is the location, \(\sigma \in \mathbb{R}_{>0}\) is the scale, and \(\xi \in \mathbb{R}\) is the shape parameter. We follow the prior specification from Caprani (2021). For each dataset, we collect \(N = 65\) i.i.d. observations and infer the posterior distribution over the parameter vector \(\theta = (\mu ,\sigma ,\xi)\) . We generate a total of \(K = 1000\) test datasets by deliberately simulating from a model with a \(2\times\) wider prior distribution to emulate out- of- distribution settings in real applications (see Appendix C for details).

Bernoulli GLM. The Bernoulli generalized linear model (GLM) is a classical model with binary outcomes, included in the SBI benchmark suite (Lueckmann et al., 2021). Each observation \(y_{i} \in \{0,1\}\) is modeled as:

\[y_{i}\sim \mathrm{Bernoulli}(\sigma (v_{i}^{\top}\theta)), \quad (6)\]

where \(v_{i}\in \mathbb{R}^{10}\) is a fixed input vector, \(\theta \in \mathbb{R}^{10}\) is the parameter vector, and \(\sigma (\cdot)\) denotes the logistic function. We generate \(K = 10,000\) in- distribution test datasets by sampling parameters from the model prior and simulating corresponding observations \(\{y_{i}\}_{i = 1}^{100}\) (Lueckmann et al., 2021).

Psychometric curve fitting. Psychometric functions are widely used in perceptual and cognitive science to characterize the relationship between stimulus intensity and the probability of a specific response (Wichmann & Hill, 2001). We use the overdispersed hierarchical model from Schütt et al. (2016), where the number of correct trials \(y_{i}\) at stimuli level \(x_{i}\) is modeled as:

\[y_{i}\sim \mathrm{Binomial}(n_{i},p_{i}),\quad p_{i}\sim \mathrm{Beta}\left(\left(\frac{1}{\eta^{2}} -1\right)\bar{p}_{i},\left(\frac{1}{\eta^{2}} -1\right)(1 - \bar{p}_{i})\right), \quad (7)\]

where \(n_{i}\) is the number of trials, \(\eta \in [0,1]\) controls overdispersion, and \(\bar{p}_{i} = \psi (x_{i};m,w,\lambda ,\gamma)\) is the expected success probability given by the psychometric function \(\psi (x;m,w,\lambda ,\gamma) = \gamma +(1 - \lambda -\gamma)S(x;m,w)\) , where \(S\) is a sigmoid function (e.g., cumulative normal), \(m\) is the threshold, \(w\) is the width, \(\lambda\) is the lapse rate for infinitely high stimulus levels, and \(\gamma\) is the guess rate for infinitely low stimulus levels. In total, the model parameters are \(\theta = (m,w,\lambda ,\gamma ,\eta)\) . Our empirical evaluation uses 8,526 mouse behavioral datasets from the International Brain Laboratory public database (The International Brain Laboratory et al., 2021).

Decision model. The drift- diffusion model (DDM) is a popular evidence accumulation model for psychological models of human decision making (Ratcliff & McKoon, 2008). It describes a two- choice decision task as a stochastic process in which noisy evidence accumulates over time until it reaches one of the decision boundaries. The evolution of the decision variable \(z(t)\) is modeled as

\[\mathrm{d}z(t) = v\mathrm{d}t + \sigma \mathrm{d}W(t), \quad (8)\]

where \(v\) is the drift rate (the average rate of evidence accumulation), \(\sigma\) is the noise scale, and \(W(t)\) denotes a standard Wiener process. A decision is made when \(z(t)\) reaches either a positive or negative boundary, typically placed symmetrically at \(\pm a\) , where \(a\) is the boundary separation. The model also includes a non- decision time parameter \(\tau\) , capturing processes that are not part of the decision process. We adopt the model specification from von Krause et al. (2022), which extends the standard DDM to incorporate experimental condition effects via six parameters: \(\theta = (v_{1},v_{2},a_{1},a_{2},\tau_{c},\tau_{n})\) . The test datasets consist of 15,000 participants from the online implicit association test (IAT) database (Xu et al., 2014; von Krause et al., 2022), providing a large- scale, real- world benchmark for Bayesian inference in cognitive modeling.

### 3.3 Main Results

Table 1 summarizes the performance of the proposed amortized Bayesian workflow across the four problems described in Section 3.2. Step 1 (ABI) exhibits extremely low time per accepted dataset (TPA), with most of the cost incurred as a one- time expense during the training phase—including prior simulation, model training, and diagnostic evaluation. Once trained, ABI incurs negligible marginal cost ( \(\ll 1\) sec) when applied to a new dataset. Datasets flagged as out- of- distribution in Step 1 are forwarded to Step 2 for correction via PSIS. PSIS is highly effective, successfully correcting most rejected amortized draws and substantially reducing the number of datasets requiring full MCMC. Only a small subset of datasets progresses to Step 3, where ChEES- HMC is used for high- fidelity sampling. As the most computationally expensive component, ChEES- HMC is applied selectively, allowing the workflow to retain both accuracy and efficiency. Overall, the amortized workflow completes inference for nearly all datasets. Compared to using ChEES- HMC for all datasets, our workflow achieves substantial computational savings—approximately over \(5 \times\) , \(120 \times\) , \(60 \times\) , and \(15 \times\) faster for the GEV, Bernoulli GLM, psychometric curve, and decision model tasks, respectively.

Table 1: Summary of our amortized Bayesian workflow across four problems. For each step, we report the number of accepted datasets, wall-clock time (minutes), and time per accepted dataset (TPA) in seconds. The time for the training phase includes amortized estimator training, simulations, and diagnostics evaluations. The time for Step 1 includes amortized posterior draws and the OOD test. The TPA for Step 1 accounts for both the training phase and Step 1. "Workflow total" aggregates the results of our method across all steps. As a baseline reference, "Baseline workflow total" is an estimate of the total required runtime for ChEES-HMC on all datasets.

| Problem | Step | Accepted datasets | Time (min) | TPA (s) |
| :--- | :--- | :--- | :--- | :--- |
| GEV | Training phase | — | 3 | 0.4 |
| GEV | Step 1: Amortized inference | 523/1000 | 0.1 | — |
| GEV | Step 2: Amortized + PSIS | 357/477 | 0.8 | 0.1 |
| GEV | Step 3: ChEES-HMC w/ inits | 87/120 | 11 | 7 |
| GEV | Workflow total (ours) | 967/1000 | 15 | 0.9 |
| GEV | Baseline workflow total | — | 85 | — |
| Bernoulli GLM | Training phase | — | 0.8 | 0.007 |
| Bernoulli GLM | Step 1: Amortized inference | 9519/10000 | 0.3 | — |
| Bernoulli GLM | Step 2: Amortized + PSIS | 425/481 | 0.4 | 0.06 |
| Bernoulli GLM | Step 3: ChEES-HMC w/ inits | 56/56 | 4 | 4 |
| Bernoulli GLM | Workflow total (ours) | 10000/10000 | 5 | 0.03 |
| Bernoulli GLM | Baseline workflow total | — | 688 | — |
| Psychometric curve | Training phase | — | 6 | 0.06 |
| Psychometric curve | Step 1: Amortized inference | 7213/8526 | 0.4 | — |
| Psychometric curve | Step 2: Amortized + PSIS | 1215/1313 | 4 | 0.2 |
| Psychometric curve | Step 3: ChEES-HMC w/ inits | 69/98 | 26 | 22 |
| Psychometric curve | Workflow total (ours) | 8497/8526 | 37 | 0.3 |
| Psychometric curve | Baseline workflow total | — | 2217 | — |
| Decision model | Training phase | — | 85 | 0.4 |
| Decision model | Step 1: Amortized inference | 13498/15000 | 1 | — |
| Decision model | Step 2: Amortized + PSIS | 827/1502 | 47 | 3 |
| Decision model | Step 3: ChEES-HMC w/ inits | 554/675 | 526 | 57 |
| Decision model | Workflow total (ours) | 14879/15000 | 659 | 3 |
| Decision model | Baseline workflow total | — | 11594 | — |

Figure 5 presents the quality of posterior draws using the W1 distance (top row) and MMTV distance (bottom row), comparing draws from each step of the workflow against reference posteriors obtained via well- tuned NUTS. Rejected amortized draws (ABI✗) exhibit markedly worse performance than accepted ones (ABI✓), confirming the effectiveness of the OOD diagnostics. PSIS- corrected draws offer accuracy comparable to ChEES- HMC samples, with only a slight decrease in quality. While amortized draws accepted in Step 1 are less accurate than those produced by PSIS or ChEES- HMC, they still provide high- quality approximations across the majority of datasets, as implied by the W1 and MMTV metrics. These results demonstrate that the proposed workflow not only scales efficiently but also consistently produces high- quality posterior estimates.

### 3.4 Advantage of amortized initializations for MCMC

One major goal of our workflow is to minimize reliance on expensive MCMC by maximizing the reuse of computations. Even when ABI and the PSIS refinement fail to yield acceptable posterior draws after Step 2, we can still leverage the amortized outputs to accelerate MCMC in Step 3.

To evaluate whether amortized posterior estimates remain useful in such cases, we test their effectiveness as initializations for ChEES- HMC chains. We conduct experiments on 20 randomly selected test datasets that progress to Step 3 of the workflow. This indicates that both the amortized posterior draws and their Pareto

> **Image description.** This image is a complex scientific visualization consisting of two rows of box-and-whisker plots, arranged in a 2x4 grid, used to evaluate posterior draws across four different statistical problems.
>
> **Overall Structure and Axes:**
> The figure is divided into two main rows, each representing a different distance metric:
> 1.  **Top Row:** Labeled "W1" on the vertical axis, measuring W1 distance. The scale ranges from 0.0 to 2.4.
> 2.  **Bottom Row:** Labeled "MMTV" on the vertical axis, measuring MMTV distance. The scale ranges from 0.0 to 1.0.
>
> The horizontal axis (X-axis) is shared across both rows and is divided into four columns, each representing a specific statistical problem: GEV, Bernoulli GLM, Psychometric curve, and Decision model.
>
> **Data Categories (X-axis Labels):**
> Under each of the four problem columns, there are four distinct categories of posterior draws, represented by box plots:
> *   $\mathrm{ABI}(\pmb {\mathscr{V}})$
> *   $\mathrm{ABI}(\pmb {\mathscr{V}})$ (Note: This label appears twice in the sequence)
> *   $\mathrm{PSIS}$
> *   $\mathrm{C-HMC}$
>
> **Visual Analysis of the Plots:**
> Each box plot displays the distribution of the distance metric for a specific draw type. The box represents the interquartile range (IQR), the line inside the box indicates the median, and the vertical lines (whiskers) extend to the minimum and maximum observed values.
>
> *   **W1 Distance (Top Row):** Across all four problems, the median W1 distances are generally low, but the distributions are wide, indicating high variance. The $\mathrm{ABI}(\pmb {\mathscr{V}})$ draws often exhibit the largest range and highest median values, particularly in the GEV and Decision model columns.
> *   **MMTV Distance (Bottom Row):** The MMTV distances are generally lower than the W1 distances. A consistent visual pattern across all four problems is that the $\mathrm{C-HMC}$ draws (the rightmost box in each group) tend to have the lowest median values and the tightest distributions, suggesting that this method provides the best posterior approximation.
>
> The overall visual presentation is a comparative analysis, allowing the viewer to assess how different methods of generating posterior draws ($\mathrm{ABI}(\pmb {\mathscr{V}})$, $\mathrm{PSIS}$, and $\mathrm{C-HMC}$) perform relative to each other across various statistical models using two different measures of approximation error.

<center>Figure 5: Evaluation of posterior draws across four problems based on two metrics: W1 distance (top row) and MMTV distance (bottom row). Lower values indicate better posterior approximation. \(\mathrm{ABI}(\pmb {\mathscr{V}})\) and \(\mathrm{ABI}(\pmb {\mathscr{V}})\) denote accepted and rejected draws, respectively, from amortized Bayesian inference in Step 1. PSIS denotes importance-weighted draws accepted in Step 2, and C-HMC denotes draws accepted via ChEES-HMC in Step 3. Metrics are computed on up to 100 datasets for each type of draws. </center>

> **Image description.** A multi-panel line graph, arranged in a 2x2 grid, illustrating the convergence of ChEES-HMC chains across four different statistical models. The graphs compare the performance of three initialization methods: Amortized (blue), Amortized + PSIS (yellow/gold), and Random Init. (red).
>
> **General Graph Elements:**
> *   **Y-Axis:** Labeled "Nested $\hat{R} - 1$," this axis uses a logarithmic scale, ranging from $10^{-3}$ to $10^{-1}$. Lower values indicate better posterior approximation.
> *   **X-Axis:** Labeled "Warmup iterations," this axis shows discrete values: 10, 50, 100, 200, 300, and 500.
> *   **Data Representation:** Each data point is represented by a marker connected by a line, and all points include vertical error bars, representing the median $\pm$ IQR across 20 test datasets.
>
> **Panel Descriptions (Top Row):**
> *   **GEV (Top Left):** This panel shows that all three initialization methods (Amortized, Amortized + PSIS, and Random Init.) exhibit rapid convergence. All lines start near $10^{-1}$ at 10 iterations and drop quickly, clustering tightly near $10^{-3}$ by 200 iterations. The performance of the three methods is visually very similar in this task.
> *   **Bernoulli GLM (Top Right):** Similar to the GEV panel, all three methods show a rapid decrease in the Nested $\hat{R} - 1$ value. The lines converge quickly, settling around $10^{-2}$ or slightly below by 100 iterations, and remain stable at low values through 500 iterations.
>
> **Panel Descriptions (Bottom Row):**
> *   **Psychometric curve (Bottom Left):** This panel demonstrates a clear difference in convergence speed. The Amortized (blue) and Amortized + PSIS (yellow) lines drop sharply and reach the lowest values (near $10^{-3}$) by 100 iterations. In contrast, the Random Init. (red) line drops more slowly, though it still shows significant improvement, reaching values near $10^{-2}$ by 300 iterations.
> *   **Decision model (Bottom Right):** The trend in this panel mirrors the Psychometric curve. The Amortized and Amortized + PSIS methods converge much faster than the Random Init. method. The blue and yellow lines quickly reach the lowest values (near $10^{-3}$) by 100 iterations, while the red line requires more warmup iterations to achieve a similar level of convergence.
>
> Overall, the visual data suggests that the Amortized and Amortized + PSIS initialization methods generally lead to faster convergence (lower Nested $\hat{R} - 1$ values) compared to Random Initialization, particularly in the Psychometric curve and Decision model tasks.

<center>Figure 6: Using amortized posterior draws as initializations for ChEES-HMC reduces the required warmup in the GEV and decision model tasks. We show median \(\pm\) IQR across 20 test datasets in Step 3. </center>

smoothed refinement are deemed unacceptable, as quantified by Pareto- \(\hat{k} >0.7\) in Step 2. We compare three initialization methods for ChEES- HMC chains: (1) amortized posterior draws, (2) PSIS- refined amortized draws, and (3) a random initialization scheme similar to Stan (Carpenter et al., 2017). We run the chains for varying numbers of warmup iterations, followed by a single sampling iteration. As described in Section 2, we use the nested \(\hat{R}\) value to gauge whether the chains converged appropriately during the warmup stage, as quantified by the common \(\hat{R} - 1\) threshold of 0.01 (Vehtari et al., 2021).

Figure 6 shows that amortized posterior draws (and their PSIS- refined counterparts) can significantly reduce the required number of warmup iterations to achieve ChEES- HMC chain convergence, even though the draws themselves have previously been flagged as unacceptable. For the GEV problem and the decision model, chains initialized with amortized draws converge faster than those using random initialization. In the Bernoulli GLM, all methods perform similarly. For the psychometric curve model, random initialization leads to faster convergence for the early stage, but amortized draws still reach the convergence threshold at a similar speed at iteration 200, indicating competitive performance. These findings are particularly relevant in the many- short- chains regime, where computational cost is dominated by the warmup phase. For instance,

with 2048 parallel chains, every single post- warmup step yields 2048 posterior samples, leading to enormous efficiency gains from shorter warmup.

Overall, these results demonstrate that amortized inference may provide suitable initializations for ChEESHMC. However, the added benefit of initializing chains with PSIS- refined amortized draws (Step 2) instead of raw amortized draws (Step 1) remains unclear. While PSIS often accelerates convergence, it occasionally degrades worst- case performance (see upper error bounds for GEV task in Figure 6). We further study the impact of initialization for the popular NUTS sampler (Hoffman & Gelman, 2014), with similar results: amortized initializations reduce the required warmup in most cases (see Appendix E).

## 4 Discussion

We presented an adaptive Bayesian workflow to combine the rapid speed of amortized inference with the undisputed sampling quality of MCMC. Our amortized workflow enables a fundamental shift in the scale and feasibility of Bayesian inference. Applying traditional MCMC (e.g., ChEES- HMC) within a standard Bayesian workflow to every dataset independently would require approximately 10 days of GPU computation across our experimental suite. In contrast, our amortized workflow completes inference in half a day, achieving speedups ranging from over \(5 \times\) to \(120 \times\) depending on the problem. Crucially, high- quality posterior draws are retained through a cascade of diagnostics and selective escalation to PSIS and MCMC. In conclusion, our workflow efficiently uses resources by (i) applying fast amortized inference when the results are accurate; (ii) refining draws with PSIS when possible; and (iii) amortized initializations of slower but accurate MCMC chains when needed.

Modularity and practical flexibility. A key strength of the proposed workflow lies in its modular structure, which allows practitioners to tailor each component to the specific constraints and objectives of their application. In cases where preliminary analysis or low- latency decision- making is essential (e.g., real- time experimental pipelines) or where likelihood evaluations are computationally expensive, the workflow can operate in a lightweight mode using amortized inference with out- of- distribution rejection alone (i.e., Step 1 in our workflow). Conversely, in high- stakes applications where accuracy is paramount, analysts can enforce escalation of all amortized draws through PSIS and, if needed, proceed to full MCMC to guarantee statistical robustness. The choice of MCMC sampler in Step 3 is also fully interchangeable: alternative algorithms such as slice sampling, ensemble samplers (e.g., emcee; Foreman- Mackey et al., 2013), or NUTS can be substituted if the model is non- differentiable, multimodal, or requires richer exploration.

Furthermore, while our paper focuses on the trade- off between wall- clock inference speed and posterior quality, practical deployments may also involve additional factors, such as inference cost (e.g., monetary expense for GPU/CPU hours) and hardware availability. Consequently, the most suitable workflow variant can differ across settings. For example, when GPU resources are limited, launching parallel MCMC chains on CPUs offers a practical alternative, making the workflow more accessible for a broader range of users.

Applicability, limitations and future directions. Our proposed workflow targets likelihood- based Bayesian models for which prior predictive simulation and likelihood evaluation are possible. It is most beneficial in repeated- inference regimes (many datasets or frequent re- fits), with moderate effective dimensionality, and when a good amortized estimator can be trained once and subsequently reused. Hence, it is not universally suitable and does not yield inference speedup gains for all Bayesian models. Training amortized models requires upfront investment in optimization and simulation. In our experiments, we found that default neural network hyperparameter settings, such as normalizing flow architectures, summary network configurations, and optimizer settings, generally yield good performance.

However, in more challenging cases, such as the GEV problem, adjustments may be necessary, guided by training- phase diagnostics. The simulation burden can be exacerbated in high- dimensional ( \(\gtrsim 10\) parameters) or weakly identifiable models, where neural estimators may struggle to approximate complex inverse maps. Alternative amortized inference approaches (see, e.g., Mittal et al., 2025) could be explored in future work to complement simulation- based amortized inference in such scenarios. In settings where likelihood evaluations

are particularly expensive, iterative refinements of the amortized estimator on individual datasets (Glöckler et al., 2022) may also be a practical alternative to likelihood- based corrections in Steps 2 and 3.

Moreover, while our diagnostic for the amortized posterior draws in Step 1 is effective and highly efficient in practice, it remains an imperfect proxy for the true posterior approximation error and can occasionally result in the acceptance of poor- quality amortized draws (cf. Figure 5). An additional empirical study in Appendix D shows that (1) the Mahalanobis distance and the posterior quality metrics are positively correlated when OOD datasets are present; (2) however, some low- distance datasets still yield poor metrics, highlighting the limitation that the OOD diagnostic cannot fully guarantee accuracy and motivating enforced escalation to PSIS (Step 2) when higher accuracy is a requirement. Future work could explore even more effective discrepancy measures, potentially tailored to the task at hand.

More broadly, the workflow supports a compelling vision of training amortized models once and reusing them across tasks or studies—a strategy well suited to applications ranging from psychology to computational biology, among others. In such settings, our layered diagnostics and selective escalation are crucial for maintaining reliability and efficiency. This positions the workflow as a practical bridge between amortized inference and traditional Bayesian rigor, enabling scalable yet trustworthy inference.

---

*Transcribed with OCR and VLMs; text, equations, tables, and figure descriptions may contain mistakes.*
