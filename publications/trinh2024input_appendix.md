# INPUT-GRADIENT SPACE PARTICLE INFERENCE FOR NEURAL NETWORK ENSEMBLES - Appendix

---

#### Page 14

# A RESULTS ON TINYIMAGENET

Table 4: FoRDE-PCA performs best under corruptions while having competitive performance on clean data. Results of PRACTRESNET18 on TINYIMAGENET evaluated over 5 seeds. Each ensemble has 10 members. cA, cNLL and cECE are accuracy, NLL, and ECE on TINYIMAGENET-C.

| Method           |      NLL $\downarrow$       |  Accuracy (\%) $\uparrow$  |        ECE $\downarrow$        |                     CA / CNLL / CECE                     |
| :--------------- | :-------------------------: | :------------------------: | :----------------------------: | :------------------------------------------------------: |
| NODE-BNNs        |       $1.39 \pm 0.01$       |       $67.6 \pm 0.3$       |       $0.114 \pm 0.004$        |                   $30.4 / 3.40 / 0.05$                   |
| SWAG             |       $1.39 \pm 0.01$       |       $66.6 \pm 0.3$       | $\mathbf{0 . 0 2 0} \pm 0.005$ |                   $28.4 / 3.72 / 0.11$                   |
| DEEP ENSEMBLES   | $\mathbf{1 . 1 5} \pm 0.00$ |       $71.6 \pm 0.0$       |       $0.035 \pm 0.002$        |                   $31.8 / 3.38 / 0.09$                   |
| WEIGHT-RDE       | $\mathbf{1 . 1 5} \pm 0.01$ |       $71.5 \pm 0.0$       |       $0.036 \pm 0.003$        |                   $31.7 / 3.39 / 0.09$                   |
| FUNCTION-RDE     |       $1.21 \pm 0.02$       |       $70.2 \pm 0.5$       |       $0.036 \pm 0.004$        |                   $31.1 / 3.43 / 0.10$                   |
| FEATURE-RDE      |       $1.24 \pm 0.01$       | $\mathbf{7 2 . 0} \pm 0.1$ |       $0.100 \pm 0.003$        |                   $31.9 / 3.35 / 0.09$                   |
| LIT              | $\mathbf{1 . 1 5} \pm 0.00$ |       $71.5 \pm 0.0$       |       $0.035 \pm 0.002$        |                   $31.2 / 3.40 / 0.11$                   |
| FoRDE-PCA (OURS) |       $1.16 \pm 0.00$       |       $71.4 \pm 0.0$       |       $0.033 \pm 0.002$        | $\mathbf{3 2 . 2} / \mathbf{3 . 2 8} / \mathbf{0 . 0 8}$ |

## B PERFORMANCE UNDER DIFFERENT ENSEMBLE SIZES

We report the NLL of FoRDE and DE under different ensemble sizes on CIFAR-10/100 and CIFAR10/100-C in Figs. 5-6. We use the WIDERESNET 16x4 (Zagoruyko \& Komodakis, 2016) architecture for this experiment. These figures show that both methods enjoy significant improvements in performance as the ensemble size increases. While Fig. 5a and Fig. 6a show that FoRDE underperforms DE on clean images, Fig. 5b and Fig. 6b show that FoRDE significantly outperforms DE on corrupted images, such that a FoRDE with 10 members has the same or better corruption robustness of a DE with 30 members.

> **Image description.** This image contains two line graphs comparing the performance of two methods, FORDE and DE, under different conditions.
>
> - **Overall Structure:** The image is divided into two panels, (a) and (b), arranged horizontally. Each panel contains a line graph.
>
> - **Panel (a): CIFAR-100 (clean):**
>
>   - **Axes:** The x-axis is labeled "Ensemble size" and has tick marks at 10, 20, and 30. The y-axis is labeled "NLL" with a downward arrow, indicating lower values are better, and has tick marks at 0.64, 0.65, and 0.66.
>   - **Lines:** There are two lines on the graph. A blue line represents "FORDE," and an orange line represents "DE." Both lines show a downward trend as the ensemble size increases. Shaded regions around each line indicate the confidence interval.
>   - **Legend:** A legend in the upper right corner identifies the lines.
>
> - **Panel (b): CIFAR-100-C (corrupted):**
>
>   - **Axes:** The x-axis is labeled "Ensemble size" and has tick marks at 10, 20, and 30. The y-axis is labeled "NLL" with a downward arrow, indicating lower values are better, and has tick marks at 2.05, 2.10, 2.15, and 2.20.
>   - **Lines:** Similar to panel (a), there are two lines: a blue line for "FORDE" and an orange line for "DE." Both lines show a downward trend as the ensemble size increases. The blue line is consistently below the orange line. Shaded regions around each line indicate the confidence interval.
>
> - **General Observations:** Both graphs show that increasing the ensemble size generally decreases the NLL (Negative Log-Likelihood) for both methods. The relative performance of FORDE and DE differs between the clean (a) and corrupted (b) datasets.

Figure 5: FoRDE is competitive on in-distribution and outperforms DEs under domain shifts by corruption. Performance of WIDERESNET16X4 on CIFAR-100 over 5 seeds.

## C Training ProcEDURE

## C. 1 Training Algorithm For FoRDE

We describe the training algorithm of FoRDE in Algorithm 1.

## C. 2 EXPERIMENTAL DETAILS FOR IMAGE CLASSIFICATION EXPERIMENTS

For all the experiments, we used SGD with Nesterov momentum as our optimizer, and we set the momentum coefficient to 0.9 . We used a weight decay $\lambda$ of $5 \times 10^{-4}$ and we set the learning rate $\eta$ to $10^{-1}$. We used a batch size of 128 and we set $\epsilon$ in Algorithm 1 to $10^{-12}$. We used 15 bins to calculate ECE during evaluation.

---

#### Page 15

# Algorithm 1 FoRDE

1: Input: training data $\mathcal{D}$, orthonormal basis $\mathbf{U}$, diagonal matrix of squared lengthscales $\boldsymbol{\Sigma}$, a neural network ensemble $\left\{f\left(\cdot ; \theta_{i}\right)\right\}_{i=1}^{M}$ of size $M$, positive scalar $\epsilon$, number of iterations $T$, step sizes $\left\{\eta_{t}\right\}_{t=1}^{T}$, weight decay $\lambda$
2: Output: optimized parameters $\left\{\theta_{i}^{(T)}\right\}_{i=1}^{M}$
3: Initialize parameters $\left\{\theta_{i}^{(0)}\right\}_{i=1}^{M}$
4: for $t=1$ to $T$ do
5: Draw a mini-batch $\left\{\mathbf{x}_{b}, y_{b}\right\}_{b=1}^{B} \sim \mathcal{D}$.
6: for $b=1$ to $B$ do
7: $\quad$ for $i=1$ to $M$ do $\quad \triangleright$ Calculate the normalized input gradients for each $\theta_{i}$ (Eq. (11))
8 :

$$
\mathbf{s}_{i, b} \leftarrow \frac{\nabla_{\mathbf{x}_{b}} f\left(\mathbf{x}_{b} ; \theta_{i}^{(t)}\right)_{y_{b}}}{\sqrt{\left\|\nabla_{\mathbf{x}_{b}} f\left(\mathbf{x}_{b} ; \theta_{i}^{(t)}\right)_{y_{b}}\right\|_{2}^{2}+\epsilon^{2}}}
$$

9: end for
10: for $i=1$ to $M$ do $\quad \triangleright$ Calculate the pairwise squared distance in Eq. (14)
11: for $j=1$ to $M$ do

$$
d_{i, j, b} \leftarrow \frac{1}{2}\left(\mathbf{s}_{i, b}-\mathbf{s}_{j, b}\right)^{\top} \mathbf{U} \boldsymbol{\Sigma} \mathbf{U}^{\top}\left(\mathbf{s}_{i, b}-\mathbf{s}_{j, b}\right)
$$

12: end for
13: end for
14: Calculate the global bandwidth per batch sample using the median heuristic (Eq. (17)):

$$
h_{b} \leftarrow \operatorname{median}\left(\left\{d_{i, j, b}\right\}_{i=1, j=1}^{M, M}\right) /(2 \ln M)
$$

15: end for
16: for $i=1$ to $M$ do $\quad \triangleright$ Calculate the pairwise kernel similarity using Eq. (16) and Eq. (17)
17: for $j=1$ to $M$ do

$$
k_{i, j} \leftarrow \frac{1}{B} \sum_{b=1}^{B} \exp \left(-d_{i, j, b} / h_{b}\right)
$$

18: end for
19: end for
20: for $i=1$ to $M$ do
21: Calculate the gradient of the repulsion term using Eq. (7):

$$
\mathbf{g}_{i}^{\text {rep }} \leftarrow \frac{\sum_{j=1}^{M} \nabla_{\theta_{i}^{(t)}} k_{i, j}}{\sum_{j=1}^{M} k_{i, j}}
$$

22: Calculate the gradient $\mathbf{g}_{i}^{\text {data }}$ of the cross-entropy loss with respect to $\theta_{i}$.
23: Calculate the update vector in Eq. (8):

$$
\mathbf{v}_{i}^{(t)} \leftarrow \frac{1}{B}\left(\mathbf{g}_{i}^{\text {data }}-\mathbf{g}_{i}^{\text {rep }}\right)
$$

24: $\quad$ Update the parameters and apply weight decay:

$$
\theta_{i}^{(t+1)} \leftarrow \theta_{i}^{(t)}+\eta_{t}\left(\mathbf{v}_{i}^{(t)}-\lambda \theta_{i}^{(t)}\right)
$$

25: end for
26: end for

---

#### Page 16

> **Image description.** This image consists of two line graphs side-by-side, comparing the performance of two methods, FORDE and DE, on CIFAR-10 datasets.
>
> The left graph, labeled "(a) CIFAR-10 (clean)", displays the Negative Log-Likelihood (NLL) on the y-axis (labeled "NLL ↓") against the Ensemble size on the x-axis. The x-axis ranges from 10 to 30. Two lines are plotted: a blue line representing FORDE and an orange line representing DE. Both lines are surrounded by shaded regions of the same color, indicating the standard deviation or confidence interval. The FORDE line is relatively flat, hovering around 0.125, while the DE line decreases from approximately 0.120 to 0.116. A legend in the upper right corner identifies the lines.
>
> The right graph, labeled "(b) CIFAR-10-C (corrupted)", also displays NLL on the y-axis (labeled "NLL ↓") against Ensemble size on the x-axis, with the x-axis ranging from 10 to 30. Again, a blue line represents FORDE and an orange line represents DE, each with a surrounding shaded region. In this graph, both lines decrease as the ensemble size increases. The FORDE line decreases from approximately 0.74 to 0.66, while the DE line decreases from approximately 0.84 to 0.81.

Figure 6: FoRDE is competitive on in-distribution and outperforms DEs under domain shifts by corruption. Performance of WIDERESNET16X4 on CIFAR-10 over 5 seeds.

On CIFAR-10 and CIFAR-100, we use the standard data augmentation procedure, which includes input normalization, random cropping and random horizontal flipping. We ran each experiments for 300 epochs. We decreased the learning rate $\eta$ linearly from $10^{-1}$ to $10^{-3}$ from epoch 150 to epoch 270. For evaluation, we used all available types for corruptions and all levels of severity in CIFAR-10/100-C.

On TINYIMAGENET, we use the standard data augmentation procedure, which includes input normalization, random cropping and random horizontal flipping. We ran each experiments for 150 epochs. We decreased the learning rate $\eta$ linearly from $10^{-1}$ to $10^{-3}$ from epoch 75 to epoch 135 . For evaluation, we used all available types for corruptions and all levels of severity in TINYIMAGENET-C.

For weight-RDE and FoRDE, we only imposed a prior on the weights via the weight decay parameter. For feature-RDE and function-RDE, we followed the recommended priors in Yashima et al. (2022). For feature-RDE, we used Cauchy prior with a prior scale of $10^{-3}$ for CIFAR-10 and a prior scale of $5 \times 10^{-3}$ for both CIFAR-100 and TINYIMAGENET, and we used a projection dimension of 5 . For function-RDE, we used Cauchy prior with a prior scale of $10^{-6}$ for all datasets.

# C. 3 EXPERIMENTAL DETAILS FOR TRANSFER LEARNING EXPERIMENTS

We extracted the outputs of the last hidden layer of a Vision Transformer model pretrained on the ImageNet-21k dataset (google/vit-base-patch16-224-in21k checkpoint in the transformers package from huggingface) and use them as input features, and we trained ensembles of 10 ReLU networks with 3 hidden layers and batch normalization.

For all the experiments, we used SGD with Nesterov momentum as our optimizer, and we set the momentum coefficient to 0.9 . We used a batch size of 256 , and we annealed the learning rate from 0.2 to 0.002 during training. We used a weight decay of $5 \times 10^{-4}$. We used 15 bins to calculate ECE for evaluation. For OOD experiments, we calculated epistemic uncertainty on the test sets of CIFAR-10/100 and CINIC10. For evaluation on natural corruptions, we used all available types for corruptions and all levels of severity in CIFAR-10/100-C.

## D ADDITIONAL RESULTS

## D. 1 INPUT GRADIENT DIVERSITY AND FUNCTIONAL DIVERSITY

To show that FoRDE indeed produces ensembles with higher input gradient diversity among member models, which in turn leads to higher functional diversity than DE, we visualize the input gradient distance and epistemic uncertainty of FoRDE and DE in Fig. 7. To measure the differences between input gradients, we use cosine distance, defined as $1-\cos (\mathbf{u}, \mathbf{v})$ where $\cos (\mathbf{u}, \mathbf{v})$ is the cosine similarity between two vectors $\mathbf{u}$ and $\mathbf{v}$. To quantify functional diversity, we calculate the epistemic uncertainty using the formula in Depeweg et al. (2018), similar to the transfer learning experiments. Fig. 7 shows that FoRDE has higher gradient distances among members compared to DE, while also having higher epistemic uncertainty across all levels of corruption severity. Intuitively, as the test

---

#### Page 17

inputs become more corrupted, epistemic uncertainty of both FoRDE and DE increases, and the input gradients between member models become more dissimilar for both methods. These results suggest that there could be a connection between input gradient diversity and functional diversity in neural network ensembles.

> **Image description.** This image consists of two bar charts side-by-side, comparing two methods, "DE" and "FoRDE-PCA," under different levels of corruption. The title above both charts reads "ResNet18 / CIFAR-100".
>
> The left chart is titled "Gradient cosine distance ↑". The y-axis ranges from 0.0 to 0.6 in increments of 0.2. The x-axis is labeled "Severity" and ranges from 0 to 5. For each severity level (0 to 5), there are two bars: a blue bar representing "DE" and a brown bar representing "FoRDE-PCA". The height of the bars represents the gradient cosine distance. Both bars are very similar in height for each severity level, showing a slight increase as the severity increases. Each bar has a small black error bar on top.
>
> The right chart is titled "Epistemic uncertainty ↑". The y-axis ranges from 0.0 to 0.3 in increments of 0.1. The x-axis is labeled "Severity" and ranges from 0 to 5. Similar to the left chart, there are two bars for each severity level, representing "DE" (blue) and "FoRDE-PCA" (brown). The height of the bars represents the epistemic uncertainty. In this chart, the bars show a clear increasing trend as the severity increases. The brown bars ("FoRDE-PCA") are consistently slightly higher than the blue bars ("DE") for each severity level. Each bar has a small black error bar on top.
>
> A legend at the bottom of both charts indicates that the blue bars represent "DE" and the brown bars represent "FoRDE-PCA".

Figure 7: FoRDE has higher gradient distance as well as higher epistemic uncertainty Results of RESNET18 on CIFAR100 over 5 seeds under different levels of corruption severity, where 0 mean no corruption.

# D. 2 PERFORMANCE UNDER CORRUPTIONS

We plot performance of all methods under the RESNET18/CIFAR-C setting in Figs. 8 and 9. These figures show that FoRDE achieves the best performance across all metrics under all corruption severities.

> **Image description.** The image consists of three box plot charts arranged horizontally. Each chart depicts the performance of four different methods (DE, DE-EmpCov, feature-RDE, and FORDE-PCA) under varying levels of corruption severity.
>
> - **Chart 1:** The first chart displays the Negative Log-Likelihood (NLL) on the y-axis, labeled "NLL ↑", ranging from 1 to 5. The x-axis represents "Corruption severity," with values from 1 to 5. Each corruption severity level has four box plots corresponding to the four methods, colored blue (DE), orange (DE-EmpCov), green (feature-RDE), and red (FORDE-PCA). The NLL generally increases with corruption severity for all methods.
>
> - **Chart 2:** The second chart shows "Accuracy(%) ↑" on the y-axis, ranging from 20 to 80. The x-axis again represents "Corruption severity" from 1 to 5. Similar to the first chart, each corruption severity level has four box plots for the four methods, using the same color scheme. Accuracy generally decreases as corruption severity increases.
>
> - **Chart 3:** The third chart displays "ECE ↓" (Expected Calibration Error) on the y-axis, ranging from 0.0 to 0.4. The x-axis represents "Corruption severity" from 1 to 5. The four methods are represented by box plots with the same color scheme as the previous charts. ECE generally increases with corruption severity, with a notable jump at severity level 5.
>
> A legend is placed below the charts, mapping the colors to the corresponding methods: blue for DE, orange for DE-EmpCov, green for feature-RDE, and red for FORDE-PCA.

Figure 8: FoRDE performs better than baselines across all metrics and under all corruption severities. Results for RESNET18/CIFAR-100-C. Each ensemble has 10 members.

## D. 3 COMPARISON BETWEEN FORDE-PCA AND EMPCOV PRIOR

In Section 3.3, we discussed a possible connection between FoRDE-PCA and the EmpCov prior (Izmailov et al., 2021a). Here, we further visualize performance of FoRDE-PCA, DE with EmpCov prior and vanilla DE on different types of corruptions and levels of severity for the RESNET18/CIFAR10 setting in Fig. 10. This figure also includes results of FoRDE-PCA with EmpCov prior to show that these two approaches can be combined together to further boost corruption robustness of an ensemble. Overall, Fig. 10 shows that FoRDE-PCA and DE-EmpCov have similar behaviors on the majority of the corruption types, meaning that if DE-EmpCov is more or less robust than DE on a corruption type then so does FoRDE-PCA. The exceptions are the blur corruption types (\{motion, glass, zoom,

---

#### Page 18

> **Image description.** This image contains three box plot charts arranged horizontally. Each chart displays the performance of different methods across varying levels of corruption severity.
>
> - **Overall Structure:** The three charts share a common x-axis labeled "Corruption severity," with values ranging from 1 to 5. Each chart represents a different performance metric: NLL (Negative Log-Likelihood), Accuracy (%), and ECE (Expected Calibration Error).
>
> - **Chart 1 (Left): NLL vs. Corruption Severity**
>
>   - Y-axis is labeled "NLL ↓" and ranges from 0 to 3. The downward arrow indicates that lower NLL values are better.
>   - The chart contains box plots for four different methods: DE (blue), DE-EmpCov (orange), feature-RDE (green), and FORDE-PCA (red).
>   - For each corruption severity level (1 to 5), there are four box plots representing the distribution of NLL values for each method.
>   - As corruption severity increases, the NLL values generally increase for all methods.
>
> - **Chart 2 (Middle): Accuracy vs. Corruption Severity**
>
>   - Y-axis is labeled "Accuracy(%) ↑" and ranges from 20 to 100. The upward arrow indicates that higher accuracy values are better.
>   - The chart contains box plots for the same four methods as the first chart: DE (blue), DE-EmpCov (orange), feature-RDE (green), and FORDE-PCA (red).
>   - For each corruption severity level (1 to 5), there are four box plots representing the distribution of accuracy values for each method.
>   - As corruption severity increases, the accuracy values generally decrease for all methods.
>
> - **Chart 3 (Right): ECE vs. Corruption Severity**
>
>   - Y-axis is labeled "ECE ↓" and ranges from 0.0 to 0.4. The downward arrow indicates that lower ECE values are better.
>   - The chart contains box plots for the same four methods as the previous charts: DE (blue), DE-EmpCov (orange), feature-RDE (green), and FORDE-PCA (red).
>   - For each corruption severity level (1 to 5), there are four box plots representing the distribution of ECE values for each method.
>   - As corruption severity increases, the ECE values generally increase for all methods.
>
> - **Legend:** A legend is located below the charts, associating each color with the corresponding method: DE (blue), DE-EmpCov (orange), feature-RDE (green), and FORDE-PCA (red).

Figure 9: FoRDE performs better than baselines across all metrics and under all corruption severities. Results for RESNET18/CIFAR-10-C. Each ensemble has 10 members.
defocus, gaussian\}-blur), where DE-EmpCov is less robust than vanilla DE while FoRDE-PCA exhibits better robustness than DE. Finally, by combining FoRDE-PCA and EmpCov prior together, we achieve the best robustness on average.

> **Image description.** This image contains an array of bar charts comparing the performance of different methods on the ResNet18/CIFAR-10-C dataset under various corruption types and severities.
>
> - **Overall Structure:** The image is arranged as a grid of 20 small bar chart panels. Each panel represents a different type of corruption or the average performance across all corruptions. The title "ResNet18 / CIFAR-10-C" is displayed at the top of the image.
> - **Panel Titles:** Each panel is labeled with a specific corruption type, including "gaussian_noise", "pixelate", "jpeg_compression", "impulse_noise", "spatter", "brightness", "frost", "speckle_noise", "motion_blur", "contrast", "glass_blur", "zoom_blur", "snow", "defocus_blur", "saturate", "fog", "shot_noise", "gaussian_blur", "elastic_transform", and "Average".
> - **Axes:** Each bar chart has a vertical axis labeled "Accuracy (%)" ranging from 0 to 100. The horizontal axis is labeled "Severity" and ranges from 1 to 5, representing different levels of corruption.
> - **Bar Charts:** Each chart displays four bars for each severity level, representing the performance of four different methods: "DE" (blue), "DE-EmpCov" (orange), "FoRDE-PCA" (green), and "FoRDE-PCA-EmpCov" (red). Error bars are visible on top of some of the bars, indicating the variance in the results.
> - **Legend:** A legend is placed at the bottom of the image, associating each method with a specific color: blue for "DE", orange for "DE-EmpCov", green for "FoRDE-PCA", and red for "FoRDE-PCA-EmpCov".

Figure 10: FoRDE-PCA and EmpCov prior behave similarly in most of the corruption types Here we visualize accuracy for each of the 19 corruption types in CIFAR-10-C in the first 19 panels, while the last panel (bottom right) shows the average accuracy. Both FoRDE-PCA and DE-EmpCov are more robust than plain DE on most of the corruption types, with the exception of contrast where both FoRDE-PCA and DE-EmpCov are less robust than DE. On the other hand, on the blur corruption types ( $\{$ motion, glass, zoom, defocus, gaussian $\}$-blur), DE-EmpCov is less robust than vanilla DE while FoRDE-PCA exhibits better robustness than DE.

# D. 4 Tuning the Lengthscales for the RBF kernel

In this section, we show how to tune the lengthscales for the RBF kernel by taking the weighted average of the identity lengthscales and the PCA lengthscales introduced in Section 3.3. Particularly,

---

#### Page 19

using the notation of Section 3.3, we define the diagonal lengthscale matrix $\boldsymbol{\Sigma}_{\alpha}$ :

$$
\boldsymbol{\Sigma}_{\alpha}=\alpha \boldsymbol{\Lambda}^{-1}+(1-\alpha) \mathbf{I}
$$

where $\boldsymbol{\Lambda}$ is a diagonal matrix containing the eigenvalues from applying PCA on the training data as defined in Section 3.3. We then visualize the accuracy of FoRDE-PCA trained under different $\alpha \in\{0.0,0.1,0.2,0.4,0.8,1.0\}$ in Fig. 11 for the RESNET18/CIFAR-100 setting and in Fig. 12 for the RESNET18/CIFAR-10 setting. Fig. 11 shows that indeed we can achieve good performance on both clean and corrupted data by choosing a lengthscale setting somewhere between the identity lengthscales and the PCA lengthscales, which is at $\alpha=0.4$ in this experiment. A similar phenomenon is observed in Fig. 12, where $\alpha=0.2$ achieves the best results on both clean and corrupted data.

> **Image description.** This image consists of three line graphs arranged horizontally, displaying the performance of a model on different datasets. The title above the graphs reads "ResNet18 / CIFAR-100".
>
> Each graph has a similar structure:
>
> - The x-axis is labeled "α" and ranges from 0.0 to 1.0.
> - The y-axis is labeled "Accuracy (%)". The y-axis scales differ between the graphs.
> - Each graph shows a blue line with circular data points, representing the model's performance. A shaded light blue area surrounds the line, likely indicating the standard deviation or confidence interval.
> - A dashed orange line runs horizontally across each graph, representing a baseline performance for comparison.
>
> The graphs are titled as follows:
>
> 1.  "Clean data": The y-axis ranges from 81.5 to 82.0. The blue line starts high, slightly decreases, then drops significantly towards the end.
> 2.  "Severity level 1, 2, 3": The y-axis ranges from 62.5 to 63.5. The blue line shows an increasing trend as α increases.
> 3.  "Severity level 4, 5": The y-axis ranges from 43.0 to 44.5. The blue line shows an increasing trend as α increases.

Figure 11: When moving from the identity lengthscales to the PCA lengthscales, FoRDE becomes more robust against natural corruptions, while exhibiting small performance degradation on clean data. Results are averaged over 3 seeds. Blue lines show performance of FoRDE, while orange dotted lines indicate the average accuracy of DE for comparison. At the identity lengthscales, FoRDE has higher accuracy than DE on in-distribution data but are slightly less robust against corruptions than DE. As we move from the identity lengthscales to the PCA lengthscales, FoRDE becomes more and more robust against corruptions, while showing a small decrease in in-distribution performance. Here we can see that $\alpha=0.4$ achieves good balance between in-distribution accuracy and corruption robustness.

> **Image description.** The image consists of three line graphs arranged horizontally, each displaying the relationship between a variable "alpha" (x-axis) and "Accuracy (%)" (y-axis). The title above the graphs reads "ResNet18 / CIFAR-10".
>
> - **Graph 1 (Left):** Titled "Clean data". The y-axis ranges from 96.1% to 96.4%. The x-axis, labeled "alpha", ranges from 0.0 to 1.0. A blue line with circular data points shows the accuracy fluctuating with different alpha values. A shaded blue area surrounds the line, indicating the standard deviation or confidence interval. A horizontal dashed orange line represents a baseline accuracy.
>
> - **Graph 2 (Middle):** Titled "Severity level 1, 2, 3". The y-axis ranges from 86 to 87%. The x-axis, labeled "alpha", ranges from 0.0 to 1.0. A blue line with circular data points shows the accuracy increasing with different alpha values. A shaded blue area surrounds the line. A horizontal dashed orange line represents a baseline accuracy.
>
> - **Graph 3 (Right):** Titled "Severity level 4, 5". The y-axis ranges from 68 to 70%. The x-axis, labeled "alpha", ranges from 0.0 to 1.0. A blue line with circular data points shows the accuracy increasing with different alpha values. A shaded blue area surrounds the line. A horizontal dashed orange line represents a baseline accuracy.
>
> All three graphs share a similar structure with grid lines, axis labels, and the visual representation of data using lines, data points, and shaded areas. The key difference lies in the range of the y-axis (Accuracy) and the specific trend of the blue line, reflecting different data conditions (clean data vs. different severity levels of corruption).

Figure 12: When moving from the identity lengthscales to the PCA lengthscales, FoRDE becomes more robust against natural corruptions, while exhibiting small performance degradation on clean data. Results are averaged over 3 seeds. Blue lines show performance of FoRDE, while orange dotted lines indicate the average accuracy of DE for comparison. At the identity lengthscales, FoRDE has higher accuracy than DE on in-distribution data but are slightly less robust against corruptions than DE. As we move from the identity lengthscales to the PCA lengthscales, FoRDE becomes more and more robust against corruptions, while showing a small decrease in in-distribution performance. Here we can see that $\alpha=0.2$ achieves good balance between in-distribution accuracy and corruption robustness.
