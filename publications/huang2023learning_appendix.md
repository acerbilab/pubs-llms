# Learning Robust Statistics for Simulation-based Inference under Model Misspecification - Appendix

---

#### Page 15

# Supplementary Materials 

The appendix is organized as follows:

- In Appendix A, we present the results on detecting model misspecification.
- In Appendix B, we provide further implementation details and results for the numerical experiment in Section 4.
- Appendix B.1: Implementation details
- Appendix B.2: Additional posterior plots
- Appendix B.3: Results for $\mathcal{D}$ being the Euclidean distance
- Appendix B.4: Computational cost analysis
- Appendix B.5: Adversarial training on Gaussian linear model
- Appendix B.6: Experiment with neural likelihood estimator
- In Appendix C, we provide details of the radio propagation experiment of Section 5.

## A Detecting misspecification of simulators

Considering that existing SBI methods can yield unreliable results under misspecification and that real-world simulators are probably not able to fully replicate observed data in most cases, detecting whether the simulator is misspecified becomes necessary for generating confidence in the results given by these methods. As misspecification can lead to observed statistics or features falling outside the distribution of training statistics, detecting for it essentially boils down to a class of out-of-distribution detection problems known as novelty detection, where the aim is to detect if the test sample $\mathbf{s}_{\text {obs }}$ come from the training distribution induced by $\left\{s_{i}\right\}_{i=1}^{m}$. This two-label classification problem can potentially be solved by adapting any of the numerous novelty detection methods from the literature. We propose the following two simple novelty detection techniques for detecting misspecification:

Distance-based approach. We assign a score to the observed statistic based on the value of the margin upper bound, as introduced in the main text. We use the MMD as the choice of distance $\mathcal{D}$, and estimate the MMD between the set of simulated statistics $\left\{s_{i}\right\}_{i=1}^{m}$ and the observed statistic $\mathbf{s}_{\text {obs }}$. This MMD-based score can be used in a classification method to detect misspecification.

Density-based approach. In this method, the training samples $\left\{s_{i}\right\}_{i=1}^{m}$ are used to fit a generative model $q$, and the log-likelihood of the observed statistics under $q$ are used as the classification score. We use a Gaussian mixture model (GMM) with $k$ components as $q$, having the distribution

$$
q(s)=\sum_{i=1}^{k} \xi_{i} \varphi\left(s \mid \mu_{i}, \Sigma_{i}\right)
$$

where $\xi_{i}, \mu_{i}$, and $\Sigma_{i}$ are the weight, the mean and the covariance matrix associated with the $i^{\text {th }}$ component, and $\varphi$ denotes the Gaussian pdf. The score $\ln q\left(\mathbf{s}_{\text {obs }}\right)$ can then be used to classify it as either being from in or out of the training distribution.

Experimental set-up. We test the performance of the proposed detection methods on the Ricker model and the OUP with the same contamination model as given in the main text. For each of these simulators, we first train the NPE method on $m=1000$ training data points, and fit a GMM with $k=2$ components to them. We then generate 1000 test datasets or points, half of them from the well-specified model and the other half from the misspecified model, and compute their score. The area under the receiver operating characteristic (AUROC) is used as the performance metric.

Baseline. We construct a baseline for comparing performance of the proposed detection methods. The baseline is based on the insight that under model misspecification, the NPE posterior moves away from the true parameter value (even going outside the prior range). Therefore, we take the root mean squared error (RMSE), defined as $\left(1 / N \sum_{i=1}^{N}\left(\theta_{i}-\theta_{\text {true }}\right)^{2}\right)^{\frac{1}{2}}$ where $\left\{\theta_{i}\right\}_{i=1}^{N}$ are posterior samples, as the classification score.

---

#### Page 16

> **Image description.** The image consists of two line graphs side-by-side, titled "Ricker" on the left and "OUP" on the right. Both graphs share the same axes. The y-axis is labeled "AUROC" and ranges from 0.4 to 1.0 in increments of 0.1. The x-axis is labeled "Misspecification level" and ranges from 0.01 to 0.2, with values 0.01, 0.05, 0.1, 0.15, and 0.2.
> 
> Each graph contains three lines, each representing a different method: RMSE (blue squares), GMM (orange triangles), and MMD (green circles).
> 
> *   **Ricker Graph:**
>     *   The RMSE line starts at approximately 0.53 at 0.01, dips to around 0.38 at 0.05, and then increases to approximately 0.71 at 0.15 and reaches 1.0 at 0.2.
>     *   The GMM line starts at approximately 0.72 at 0.01, increases to around 0.83 at 0.05, and then continues to increase to 0.97 at 0.15 and reaches 1.0 at 0.2.
>     *   The MMD line starts at approximately 0.48 at 0.01, increases to around 0.53 at 0.05, and then continues to increase to 0.66 at 0.1 and 0.85 at 0.15 and reaches 0.92 at 0.2.
> 
> *   **OUP Graph:**
>     *   The RMSE line starts at approximately 0.39 at 0.01, increases to around 0.41 at 0.05, and then continues to increase to 0.71 at 0.1 and 0.95 at 0.15 and reaches 0.99 at 0.2.
>     *   The GMM line starts at approximately 0.77 at 0.01, increases to 1.0 at 0.05 and remains at 1.0 for the rest of the x-axis values.
>     *   The MMD line starts at approximately 0.50 at 0.01, increases to around 0.57 at 0.05, and then continues to increase to 0.77 at 0.1 and 0.95 at 0.15 and reaches 1.0 at 0.2.
> 
> A legend is present between the two graphs, indicating the shapes and colors corresponding to each method: RMSE (blue squares), GMM (orange triangles), and MMD (green circles).


Figure 7: Misspecification detection experiment. AUROC of the proposed detection methods (GMM and MMD) versus misspecification level for the Ricker model and the OUP. The RMSE-based baseline is shown in blue.

Results. The AUROC of the classifiers for different levels of misspecification ( $\epsilon$ in the main text) is shown in Fig. 7 for both the models. The proposed GMM-based detection method performs the best, followed by the MMD-based method. The RMSE-based baseline performs the worst at the classification task. We conclude that it is possible to detect model misspecification in the space of summary statistics using simple to use novelty detection methods.

## B Additional details and results of the numerical experiments

## B. 1 Implementation details

We implement our NPE-RS models based on publicly available implementations from https: //github.com/mackelab/sbi. We use the NPE-C model [41] with Masked Autoregressive Flow (MAF) [60] as the backbone inference network, and adopt the default configuration with 50 hidden units and 5 transforms for MAF. The batch size is set to 50, and we maintain a fixed learning rate of $5 \times 10^{-4}$. The implementation for RNPE is sourced directly from the original repository at https://github.com/danielward27/rnpe.

Regarding the summary network in NPE tasks, for the Ricker model, we employ three 1D convolutional layers with 4 hidden channels, and we set the kernel size to 3 . For the OUP model, we combine three 1D convolutional layers with one bidirectional LSTM layer. The convolutional layers have 8 hidden channels and a kernel size equal to 3 , while the LSTM layer has 2 hidden dimensions. We pass the data separately through the convolutional layers and the LSTM layer and then concatenate the resulting representations to obtain our summary statistics. For the Turin model in Section 5, we utilize five 1D convolutional layers with hidden units set to $[8,16,32,64,8]$, and the kernel size is set to 3 . Across all three summary networks, we employ the mean operation as our aggregator to ensure permutation invariance among realizations.

In ABC tasks, we incorporate autoencoders as our summary network. For the Ricker model, the encoder consists of three 1D convolutional layers with 4 hidden channels, where the kernel size is set to 3 . The decoder comprises of three 1D transposed convolutional layers with the same settings as the encoder's convolutional layers, allowing for data reconstruction. For the OUP model, we adopt a similar summary network as the one used for the Ricker model but with a smaller stride.

In NPE tasks, we use 1000 samples for the training data, along with 100 realizations of both observed and simulated data for each value of $\theta$. We also use 1000 samples for training the autoencoders. For ABC , we use 4000 samples from the prior and accept $n_{\delta}=200$ samples giving a tolerance rate of $5 \%$. We take $\rho$ to be Euclidean distance in the rejection ABC and normalize the statistics by the median absolute deviation before computing the distance to account for the difference in their magnitude.

## B. 2 Additional posterior plots

We now present examples of the remaining posterior plots, apart from the one shown in the main text. The posterior plots for OUP using the NPE-based methods is shown in Figure 8. The observations

---

#### Page 17

are similar to the Ricker model example in the main text: we see that our NPE-RS method yields similar posterior as NPE in the well-specified case, whereas RNPE posteriors are underconfident. When the model is misspecified, NPE posterior goes far from the true parameter value. The NPE-Rs posteriors, however, are still around $\theta_{\text {true }}$, demonstrating robustness to misspecification.
> **Image description.** This image displays three scatter plots, each with marginal probability density functions along the axes, comparing different methods for parameter estimation under varying degrees of model misspecification. The plots are arranged horizontally.
> 
> Each plot features:
> 
> *   **Scatter Plot:** A two-dimensional scatter plot with the x-axis labeled "θ1" and the y-axis labeled "θ2". The data points are represented by a density map, with color intensity indicating the concentration of points. The color of the density map varies across the plots.
> *   **Marginal Probability Density Functions:** Above and to the right of each scatter plot are marginal probability density functions. The function above the scatter plot represents the marginal distribution of θ1, and the function to the right represents the marginal distribution of θ2. These functions are displayed as curves.
> *   **True Parameter Values:** Each scatter plot contains a vertical and horizontal black line intersecting at (0.5, 1), representing the true parameter values (θtrue).
> *   **Legend:** A legend is present only in the leftmost plot, indicating the color scheme for different methods: orange for NPE, green for RNPE, and blue for "Ours".
> *   **Axes:** The x and y axes range from approximately -2 to 3 and -2 to 2, respectively.
> *   **Titles:** Each plot has a title below it:
>     *   (a) Well-specified (ε = 0)
>     *   (b) Misspecified (ε = 10%)
>     *   (c) Misspecified (ε = 20%)
> 
> The plots show how the estimated parameter distributions change as the model becomes more misspecified. In the "Well-specified" case, all methods produce relatively concentrated posteriors near the true parameter values. As misspecification increases, the NPE posterior diverges from the true value, while the "Ours" method maintains a posterior closer to θtrue.


Figure 8: Ornstein-Uhlenbeck process. Posteriors obtained from our method (NPE-RS), RNPE, and NPE for different degrees of model misspecification.

Similar behavior is observed in the ABC case for both the Ricker model and OUP in Figure 9 and Figure 10, respectively. The ABC posteriors go outside the prior range under misspecification, while ABC with our robust statistics yields posteriors closer to $\theta_{\text {true }}$. In Table 1, we report the sample mean and standard deviations for the results shown in Figure 2 of the main text.
> **Image description.** This image presents a set of three scatter plots, each accompanied by marginal density plots. The plots are arranged horizontally, showing the results of different model specifications.
> 
> Each scatter plot visualizes the relationship between two parameters, theta1 (θ₁) on the x-axis and theta2 (θ₂) on the y-axis. The x-axis ranges from approximately 2 to 8, while the y-axis ranges from 0 to 25. Horizontal and vertical lines indicate the true values (θtrue) of the parameters. Two overlapping density plots are shown, one in purple labeled "ABC" and one in red labeled "Ours". The scatter plots contain shaded regions, also in purple and red, representing the joint posterior distributions obtained by different methods.
> 
> Marginal density plots are positioned above and to the right of each scatter plot. The top plot shows the marginal distribution of theta1, while the right plot shows the marginal distribution of theta2. The density plots are colored consistently with the scatter plots, allowing for easy comparison of the different methods.
> 
> The three panels are labeled as follows:
> *   (a) Well-specified (ε = 0)
> *   (b) Misspecified (ε = 10%)
> *   (c) Misspecified (ε = 20%)
> 
> These labels indicate the degree of model misspecification in each case.


Figure 9: Ricker model. Posteriors obtained from our method (ABC-RS) and ABC for different degrees of model misspecification.

# B. 3 Results for $\mathcal{D}$ being the Euclidean distance 

We present results for $\mathcal{D}$ being the Euclidean distance in the well-specified case of the Ricker model in Figure 11(a). As mentioned in Section 3 of the main text, this choice leads to very underconfident posteriors. This is because the Euclidean distance is not a robust distance: it becomes large even if a few points are far from the observed statistic. As a result, using this as the regularization term penalises most choices of summarizer $\eta$, and we learn statistics that are very concentrated around the observed statistic (orange dot). Although a good choice for being robust, Euclidean distance leads to statistics that are not informative about the model parameters, yielding posterior that is similar to the uniform prior. Hence, we used the MMD as the distance in the margin upper bound, which provides better a trade-off between robustness and efficiency (in terms of learning about model parameters).

---

#### Page 18

> **Image description.** This image contains three scatter plots, each accompanied by marginal density plots, comparing the performance of two methods (ABC and "Ours") under different levels of model misspecification.
> 
> Each scatter plot displays the joint distribution of two parameters, theta1 and theta2, with theta1 on the x-axis and theta2 on the y-axis. The plots are overlaid with density estimations for both ABC (in purple) and "Ours" (in red). Horizontal and vertical solid black lines indicate the true values of theta2 and theta1, respectively. Dashed grey lines indicate the same true values.
> 
> Above each scatter plot is a marginal density plot for theta1, showing the distribution of values along the x-axis. Similarly, to the right of each scatter plot is a marginal density plot for theta2, showing the distribution of values along the y-axis.
> 
> The three scatter plots represent different levels of model misspecification:
> *   **(a) Well-specified (epsilon = 0):** The first plot represents the well-specified case.
> *   **(b) Misspecified (epsilon = 10%):** The second plot represents a 10% misspecification.
> *   **(c) Misspecified (epsilon = 20%):** The third plot represents a 20% misspecification.
> 
> The y-axis scales vary across the plots, ranging from -2 to 2 in the first plot, -2 to 4 in the second, and 0 to 15 in the third. The x-axis scale is consistent across all plots, ranging from 0 to 2.
> 
> A legend in the first plot identifies the black line as "theta true", the purple density as "ABC", and the red density as "Ours".


Figure 10: Ornstein-Uhlenbeck process. Posteriors obtained from our method (ABC-RS) and ABC for different degrees of model misspecification.

Table 1: Performance of the SBI methods in terms of RMSE and MMD for both Ricker and OUP. We report the average ( $\pm 1$ std. deviation) values across 100 runs for varying levels of misspecification.

|  |  | RMSE $(\downarrow)$ |  |  | MMD $(\downarrow)$ |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  |  | $\epsilon=0 \%$ | $\epsilon=10 \%$ | $\epsilon=20 \%$ | $\epsilon=0 \%$ | $\epsilon=10 \%$ | $\epsilon=20 \%$ |
| $\begin{aligned} & \text { 刃 } \\ & \text { ㅁ } \end{aligned}$ | NPE | 2.16 (3.07) | 7.86 (1.57) | 11.2 (1.70) | 0.04 (0.07) | 0.74 (0.09) | 1.06 (0.17) |
|  | RNPE | 3.27 (0.35) | 5.51 (0.58) | 7.14 (1.15) | 0.06 (0.05) | 0.51 (0.19) | 0.79 (0.25) |
|  | NPE-RS (ours) | 2.18 (2.66) | 2.19 (1.01) | 4.66 (4.15) | 0.09 (0.14) | 0.21 (0.16) | 0.42 (0.37) |
|  | ABC | 1.46 (0.44) | 6.95 (0.25) | 9.79 (0.96) | 0.01 (0.01) | 0.85 (0.02) | 1.18 (0.04) |
|  | ABC-RS (ours) | 1.20 (0.51) | 3.16 (1.08) | 2.99 (1.28) | 0.01 (0.02) | 0.17 (0.15) | 0.18 (0.16) |
|  | NPE | 0.79 (0.62) | 1.26 (1.18) | 2.59 (2.75) | 0.01 (0.01) | 0.34 (0.15) | 0.63 (0.29) |
|  | RNPE | 0.78 (0.09) | 0.87 (0.10) | 0.98 (0.15) | 0.01 (0.01) | 0.22 (0.13) | 0.49 (0.26) |
|  | NPE-RS (ours) | 0.74 (0.70) | 0.62 (0.33) | 0.63 (0.36) | 0.02 (0.05) | 0.09 (0.09) | 0.21 (0.17) |
|  | ABC | 0.50 (0.07) | 1.20 (0.40) | 5.16 (2.39) | 0.05 (0.03) | 0.88 (0.21) | 0.92 (0.23) |
|  | ABC-RS (ours) | 0.44 (0.06) | 0.62 (0.23) | 0.88 (0.48) | 0.02 (0.02) | 0.26 (0.17) | 0.50 (0.38) |

# B. 4 Computational cost analysis 

We now present a quantitative analysis of the computational cost of training with and without our MMD regularization term. The results, presented in Table 2, are calculated on an Apple M1 Pro CPU. As expected, we observe a higher runtime for our method due to the computational cost of estimating the MMD from 200 samples of simulated data. The total runtime also depends on the number of batchsize $N_{\text {batch }}$, hence, as $N_{\text {batch }}$ increases, the proportion of runtime used for estimating MMD reduces. As a result, we see that for large $N_{\text {batch }}$, the increase in the computational cost of our method with robust statistics is not significant.

## B. 5 Adversarial training on Gaussian linear model

To verify the robustness of our method on higher dimensional parameter space, we run an experiment on the Gaussian linear model, where the data $\mathbf{x} \in \mathbb{R}^{10}$ is sampled from $\mathcal{N}\left(\theta, 0.1 \cdot \boldsymbol{I}_{10}\right)$. A uniform prior $\mathcal{U}([-1,1])^{10}$ is placed on the parameters $\theta \in \mathbb{R}^{10}$. We take $\theta_{\text {true }}=[0.5, \ldots, 0.5]^{\top}, \theta_{c}=[2, \ldots, 2]^{\top}$. To introduce contamination to the observed data, we employ the same approach as outlined in the main text of our paper. However, there is a slight divergence in our experimental setup. In this example, we employ adversarial training, meaning that the model is trained on observed data with a high degree of misspecification $(\epsilon=20 \%)$, while we perform inference on data with a lower misspecification degree $(\epsilon=10 \%)$. For the summary network, we utilize the DeepSet [86] architecture. The encoder comprises two linear layers, each with a width of 20 hidden units, paired with the ReLU activation function. The decoder is constructed with a single linear layer of 20 hidden units.
The results are shown in Table 3. NPE-RS outperforms NPE in terms of MMD between the posterior predictive distribution and the observed data, which highlights the effectiveness of our approach in

---

#### Page 19

> **Image description.** The image consists of two panels, (a) and (b), displaying results related to the Ricker model.
> 
> Panel (a), labeled "Posteriors," shows a two-dimensional density plot with axes labeled "θ1" (horizontal, ranging from 2 to 8) and "θ2" (vertical, ranging from 0 to 20). The density is represented by blue contours, indicating the posterior distribution of the parameters. The plot also includes:
> *   A black horizontal line at θ2 ≈ 10 and a black vertical line at θ1 ≈ 4, labeled as "θtrue," presumably indicating the true parameter values.
> *   Two probability density curves on the top and right sides of the 2D density plot. The top curve is a combination of orange and gray, while the right curve is orange. These curves show the marginal distributions of θ1 and θ2, respectively.
> *   A legend indicating that the blue contours represent "NPE-RS (Euclidean)" and the orange color represents "NPE." The gray color in the top curve represents "θtrue".
> 
> Panel (b), labeled "Summary statistics," consists of a 4x3 grid of scatter plots. Each plot shows the relationship between two summary statistics, "s1", "s2", and "s3", which are displayed on the horizontal axes. The vertical axes are labeled "S1", "S2", "S3", and "S4". Each scatter plot contains numerous blue dots, representing data points, and a single, larger orange dot, possibly indicating a mean or median value. The arrangement of the plots suggests a pairwise comparison of the summary statistics.


Figure 11: Ricker model. Posteriors and summary statistics for $\mathcal{D}$ being the Euclidean distance.
Table 2: Comparison of computational costs across different models on Ricker model. We report the mean value (standard deviation) derived from 20 updates. We use different batch size $N_{\text {batch }}$ and generate 100 realizations for each $\theta$.

|  | Runtime (seconds) |  |  |
| :--: | :--: | :--: | :--: |
|  | $N_{\text {batch }}=50$ | $N_{\text {batch }}=100$ | $N_{\text {batch }}=200$ |
| NPE | $0.22(0.03)$ | $0.46(0.04)$ | $0.87(0.03)$ |
| NPE-RS (ours) | $1.26(0.05)$ | $1.53(0.14)$ | $1.92(0.10)$ |
| ABC | $0.68(0.04)$ | $1.41(0.04)$ | $3.29(0.27)$ |
| ABC-RS (ours) | $1.79(0.04)$ | $2.71(0.25)$ | $4.25(0.46)$ |

high-dimensional parameter spaces, even though the observed data was not used in the training of NPE-RS. This points towards the possibility that by employing adversarial training, we might achieve robustness against lower levels of misspecification whilst still being amortized.

Table 3: Performance comparison between NPE and NPE-RS for Gaussian linear model. We use MMD between the posterior predictive distribution and the observed data as our metric. We report the average ( $\pm 1$ std. deviation) values across 100 runs.

|  | NPE | NPE-RS |  |  |
| :--: | :--: | :--: | :--: | :--: |
| $\lambda$ | - | 20 | 50 | 100 |
| MMD | $0.26(0.02)$ | $0.19(0.04)$ | $\mathbf{0 . 1 8}(0.06)$ | $0.21(0.08)$ |

# B. 6 Experiment with neural likelihood estimators 

In this section, we explore the performance of our method when paired with Neural Likelihood Estimators (NLE). NLE are a class of methods that leverage neural density estimators to directly estimate likelihood functions, bridging the gap between simulators and statistical models.

For this experiment, we adopt NLE-A as proposed by [61]. The original implementation can be found at https://github.com/mackelab/sbi. Similar to our approach with ABC, we adapt our method to NLE by pre-emptively training an autoencoder with our regularization term to learn the summary statistics. We refer to our adapted method as NLE-RS. The configurations for our summary network and simulator are consistent with those described in Appendix B.1.

Figure 12 presents the posterior plots for the Ricker model using the NLE-based methods. Consistent with our observations in the previous experiments, NLE-RS still demonstrates robustness to misspecification, while the NLE posterior tends to deviate away from the true parameters.

---

#### Page 20

> **Image description.** The image is a figure displaying posterior distributions related to a model, likely a statistical or machine learning model based on the context. It consists of a central scatter plot with marginal distributions shown along the top and right edges.
> 
> The central plot is a two-dimensional scatter plot. The x-axis is labeled "$\theta_1$" and ranges from approximately 2.5 to 8. The y-axis is labeled "$\theta_2$" and ranges from 0 to 30. There are two distinct areas of density plotted on this scatter plot. One is a blue shaded region labeled "Ours", and the other is an orange shaded region labeled "NLE". A vertical black line labeled "True $\theta$" is positioned at approximately $\theta_1 = 4.7$. A horizontal black line is positioned at $\theta_2 = 10$. Dashed gray lines are present at $\theta_2 = 0$ and $\theta_1 = 2.5$ and $\theta_1 = 7.5$.
> 
> Above the scatter plot is a one-dimensional distribution plot along the x-axis ($\theta_1$). It shows two overlapping distributions, one blue and one orange, corresponding to "Ours" and "NLE" respectively. The blue distribution is centered slightly to the right of the orange distribution.
> 
> To the right of the scatter plot is a one-dimensional distribution plot along the y-axis ($\theta_2$). It also shows two overlapping distributions, one blue and one orange, corresponding to "Ours" and "NLE" respectively. The blue distribution is centered slightly below the orange distribution.


Figure 12: Ricker model. Posteriors obtained from our method (NLE-RS) and NLE. We set $\epsilon=10 \%$ for this experiment.

# C Details of the radio propagation experiment 

In this section, we describe the data and the Turin model used in Section 5 of the main text.
Data and model description. Let $B$ be the frequency bandwidth used to measure radio channel data at $K$ equidistant points, leading to a frequency separation of $\Delta f=B /(K-1)$. The measured transfer function at $k$ th point, $Y_{k}$, is modelled as

$$
Y_{k}=H_{k}+W_{k}, \quad k=0,1, \ldots, K-1
$$

where $H_{k}$ is the transfer function at the $k$ th frequency, and $W_{k}$ is additive zero-mean complex circular symmetric Gaussian noise with variance $\sigma_{W}^{2}$. Taking the inverse Fourier transform, the time-domain signal $y(t)$ can be obtained as

$$
y(t)=\frac{1}{K} \sum_{k=0}^{K-1} Y_{i} \exp (j 2 \pi k \Delta f t)
$$

The Turin model defines the transfer function as $H_{k}=\sum_{l} \alpha_{l} \exp \left(-j 2 \pi \Delta f k \tau_{l}\right)$, where $\tau_{l}$ is the time-delay and $\alpha_{l}$ is the complex gain of the $l^{\text {th }}$ component. The arrival time of the delays is modelled as one-dimensional homogeneous Poisson point processes, i.e., $\tau_{l} \sim \operatorname{PPP}\left(\mathbb{R}_{+}, \nu\right)$, with $\nu>0$. The gains conditioned on the delays are modelled as iid zero-mean complex Gaussian random variables with conditional variance $\mathbb{E}\left[\left|\alpha_{l}\right|^{2} \mid \tau_{l}\right]=G_{0} \exp \left(-\tau_{l} / T\right) / \nu$. The parameters of the model are $\theta=\left[G_{0}, T, \nu, \sigma_{W}^{2}\right]^{\top}$. The prior ranges used for the parameters are given in Table 4.

Table 4: Prior distributions for the parameters of the Turin model.

|  | $G_{0}$ | $T$ | $\nu$ | $\sigma_{W}^{2}$ |
| :--: | :--: | :--: | :--: | :--: |
| Prior | $\mathcal{U}\left(10^{-9}, 10^{-8}\right)$ | $\mathcal{U}\left(10^{-9}, 10^{-8}\right)$ | $\mathcal{U}\left(10^{7}, 5 \times 10^{9}\right)$ | $\mathcal{U}\left(10^{-10}, 10^{-9}\right)$ |

The radio channel data from [44] is collected in a small conference room of dimensions $3 \times 4 \times 3 \mathrm{~m}^{3}$, using a vector network analyzer. The measurement was performed with a bandwidth of $B=4$ GHz , and $K=801$. Denote each complex-valued time-series by $\hat{\mathbf{y}} \in \mathbb{R}^{K}$, and the whole dataset by $\hat{\mathbf{y}}_{1: n}$, where $n=100$ realizations. We take the input to the summary network to be $\mathbf{y}_{1: n}=$ $10 \log _{10}\left(\left|\hat{\mathbf{y}}_{1: n}\right|^{2}\right)$.

Scatter-plot of learned statistics. In Figure 13 and Figure 14, we show the scatter-plots of the learned statistics using the NPE and our NPE-RS method, respectively. We observe that the observed statistics (shown in orange) is often outside the set of simulated statistics (shown in blue) for the NPE method. Hence, the inference network is forced to generalize outside its training distribution, which leads to poor fit of the model, as shown in Section 5 of the main text. On the other hand, the observed statistic is always inside the set of simulated statistics (or the training distribution) for our method in Figure 14, which leads to robustness against model misspecification.

---

#### Page 21

> **Image description.** This image is a matrix of scatter plots. It contains 8 rows and 7 columns of individual scatter plots, each enclosed in a rectangular frame.
> 
> *   **Arrangement:** The plots are arranged in a grid. The rows are labeled S1 through S8 vertically along the left side of the matrix, and the columns are labeled S1 through S7 horizontally along the bottom of the matrix.
> 
> *   **Scatter Plots:** Each individual plot displays a scatter of blue points. The distribution of these points varies across the plots, with some showing linear correlations, clusters, or more dispersed patterns.
> 
> *   **Orange Dots:** Each scatter plot also contains a single orange dot. The position of the orange dot varies within each plot.
> 
> *   **Labels:** The labels "S1", "S2", "S3", "S4", "S5", "S6", "S7", and "S8" are clearly visible, indicating the variable being plotted on the x and y axes of each scatter plot.


Figure 13: Pairwise scatter-plots of summary statistics learned using NPE method for the Turin model. Each blue dot corresponds to simulated statistic obtained from a parameter value sampled from the prior. The orange dot represents the observed statistic.

---

#### Page 22

> **Image description.** The image is a matrix of pairwise scatter plots. There are 8 rows and 7 columns of scatter plots, for a total of 56 plots. Each plot is enclosed in a black rectangular frame.
> 
> The x and y axes of each plot are unlabeled, but the rows and columns are labeled with "S1" through "S8" along the left edge and "S1" through "S7" along the bottom edge. The labels are in black font.
> 
> Each scatter plot contains many small blue dots, forming a cloud of points. The shape and distribution of the points vary from plot to plot. In addition to the blue dots, each plot contains a single larger orange dot. The position of the orange dot varies from plot to plot.
> 
> The plots on the diagonal (S1 vs S1, S2 vs S2, etc.) are not present.


Figure 14: Pairwise scatter-plots of summary statistics learned using our NPE-RS method for the Turin model. Each blue dot corresponds to simulated statistic obtained from a parameter value sampled from the prior. The orange dot represents the observed statistic.