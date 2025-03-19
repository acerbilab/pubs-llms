# Tackling covariate shift with node-based Bayesian neural networks - Appendix

---

#### Page 12

# A. Original ELBO derivation

Here we provide a detail derivation of the ELBO in Eq. (9). We assume a prior $p(\theta, \mathcal{Z})=p(\theta) p(\mathcal{Z})$ for the parameters $\theta$ and latent variables $\mathcal{Z}$, and we assume a variational posterior $q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})=\delta(\theta-\hat{\theta}) q_{\phi}(\mathcal{Z})$ where $\delta($.$) is a Dirac delta$ distribution. We arrive at the ELBO in Eq. (9) by minimizing the KL divergence between the variational approximation and the true posterior with respect to the variational parameters $(\hat{\theta}, \phi)$ :

$$
\begin{aligned}
& \underset{\phi, \hat{\theta}}{\arg \min } \mathrm{KL}\left[q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})| | p(\theta, \mathcal{Z} \mid \mathcal{D})\right] \\
& =\underset{\phi, \hat{\theta}}{\arg \min } \mathbb{E}_{q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})}\left[\log q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})-\log p(\mathcal{D} \mid \theta, \mathcal{Z})-\log p(\theta, \mathcal{Z})+\log p(\mathcal{D})\right] \\
& =\underset{\phi, \hat{\theta}}{\arg \min } \mathbb{E}_{q_{\phi}(\mathcal{Z})}[-\log p(\mathcal{D} \mid \hat{\theta}, \mathcal{Z})]+\mathrm{KL}\left[q_{\phi}(\mathcal{Z})| | p(\mathcal{Z})\right]-\log p(\hat{\theta})+\log p(\mathcal{D}) \\
& =\underset{\phi, \hat{\theta}}{\arg \min }-\mathcal{L}(\hat{\theta}, \phi)
\end{aligned}
$$

## B. Tempered ELBO derivation

Here we show a connection between the tempered posterior with temperature $\tau=1 /(\gamma+1)$ in Eq. (20) and the augmented ELBO in Section 4.1:

$$
\begin{aligned}
& \underset{\phi, \hat{\theta}}{\arg \min } \frac{1}{\tau} \mathrm{KL}\left[q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})| | p_{\gamma}(\theta, \mathcal{Z} \mid \mathcal{D})\right] \\
& =\underset{\phi, \hat{\theta}}{\arg \min } \frac{1}{\tau} \mathbb{E}_{q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})}\left[\log q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})-\tau \log p(\mathcal{D} \mid \theta, \mathcal{Z})-\tau \log p(\theta, \mathcal{Z})+\log p_{\gamma}(\mathcal{D})\right] \\
& =\underset{\phi, \hat{\theta}}{\arg \min } \mathbb{E}_{q_{\phi, \hat{\theta}}(z, \theta)}\left[\frac{1}{\tau} \log q_{\phi, \hat{\theta}}(\theta, \mathcal{Z})-\log p(\mathcal{D} \mid \theta, \mathcal{Z})-\log p(\theta)-\log p(\mathcal{Z})\right]+\frac{1}{\tau} \log p_{\gamma}(\mathcal{D}) \\
& =\underset{\phi, \hat{\theta}}{\arg \min }-\mathbb{E}_{q_{\phi}(\mathcal{Z})}\left[\log p(\mathcal{D} \mid \hat{\theta}, \mathcal{Z})\right]+\mathrm{KL}\left[q_{\phi}(\mathcal{Z})| | p(\mathcal{Z})\right]-\gamma \mathbb{H}\left[q_{\phi}(\mathcal{Z})\right]-\log p(\hat{\theta})+\frac{1}{\tau} \log p_{\gamma}(\mathcal{D}) \\
& =\underset{\phi, \hat{\theta}}{\arg \min }-\mathcal{L}_{\gamma}(\hat{\theta}, \phi)+\log p_{\gamma}(\mathcal{D})^{\frac{1}{\tau}}
\end{aligned}
$$

## C. Derivation of layer-wise activation shifts due to input corruptions

Here we explain in detail the approximation of layer-wise activation shifts in Eq. (12). To simulate covariate shift, one can take an input $\mathbf{x}$ assumed to come from the same distribution as the training samples and apply a corruption $\mathbf{g}^{0}$ to form a shifted version $\mathbf{x}^{c}$ of $\mathbf{x}$ :

$$
\mathbf{x}^{c} \triangleq \mathbf{x}+\mathbf{g}^{0}(\mathbf{x})
$$

For instance, $\mathbf{x}$ could be an image and $\mathbf{g}^{0}$ can represent the shot noise corruption as seen in Hendrycks \& Dietterich (2019). The corruption $\mathbf{g}^{0}(\mathbf{x})$ creates a shift in the activation of the first layer $\mathbf{f}^{1}$ which can be approximated using the first-order Taylor expansion:

$$
\begin{aligned}
\mathbf{g}^{1}(\mathbf{x}) & =\mathbf{f}^{1}\left(\mathbf{x}^{c}\right)-\mathbf{f}^{1}(\mathbf{x}) \\
& =\sigma\left(\mathbf{W}^{1}\left(\mathbf{x}+\mathbf{g}^{0}(\mathbf{x})\right)+\mathbf{b}^{1}\right)-\sigma\left(\mathbf{W}^{1} \mathbf{x}+\mathbf{b}^{1}\right) \\
& \approx \mathbf{J}_{\sigma}\left[\mathbf{h}^{1}(\mathbf{x})\right]\left(\mathbf{W}^{1} \mathbf{g}^{0}(\mathbf{x})\right)
\end{aligned}
$$

where $\mathbf{J}_{\sigma}=\partial \sigma / \partial \mathbf{h}$ denotes the Jacobian of the activation $\sigma$ with respect to pre-activation outputs $\mathbf{h}$. Similarly, the approximation of the activation shift in the second layer is:

$$
\begin{aligned}
\mathbf{g}^{2}(\mathbf{x}) & =\mathbf{f}^{2}\left(\mathbf{x}^{c}\right)-\mathbf{f}^{2}(\mathbf{x}) \\
& =\sigma\left(\mathbf{W}^{2} \mathbf{f}^{1}\left(\mathbf{x}^{c}\right)+\mathbf{b}^{2}\right)-\sigma\left(\mathbf{W}^{2} \mathbf{f}^{1}(\mathbf{x})+\mathbf{b}^{2}\right) \\
& =\sigma\left(\mathbf{W}^{2}\left(\mathbf{f}^{1}(\mathbf{x})+\mathbf{g}^{1}(\mathbf{x})\right)+\mathbf{b}^{2}\right)-\sigma\left(\mathbf{W}^{2} \mathbf{f}^{1}(\mathbf{x})+\mathbf{b}^{2}\right) \\
& \approx \mathbf{J}_{\sigma}\left[\mathbf{h}^{2}(\mathbf{x})\right]\left(\mathbf{W}^{2} \mathbf{g}^{1}(\mathbf{x})\right)
\end{aligned}
$$

---

#### Page 13

Table 1. The ALL-CNN-C architecture

|                             ALL-CNN-C                             |
| :---------------------------------------------------------------: |
|                  Input $32 \times 32$ RGB images                  |
|          $3 \times 3$ conv. with 96 output filters, ReLU          |
|          $3 \times 3$ conv. with 96 output filters, ReLU          |
| $3 \times 3$ conv. with 96 output filters and stride $r=2$, ReLU  |
|         $3 \times 3$ conv. with 192 output filters, ReLU          |
|         $3 \times 3$ conv. with 192 output filters, ReLU          |
| $3 \times 3$ conv. with 192 output filters and stride $r=2$, ReLU |
|         $3 \times 3$ conv. with 192 output filters, ReLU          |
|          $1 \times 1$ conv. with 10 output filters, ReLU          |
|                      Global average pooling                       |
|                          10-way softmax                           |

Generally, one can approximate the shift in the output of the $\ell$-th layer caused by $\mathbf{g}(\mathbf{x})$ as:

$$
\mathbf{g}^{\ell}(\mathbf{x})=\mathbf{f}^{\ell}\left(\mathbf{x}^{c}\right)-\mathbf{f}^{\ell}(\mathbf{x}) \approx \mathbf{J}_{\sigma}\left[\mathbf{h}^{\ell}(\mathbf{x})\right]\left(\mathbf{W}^{\ell} \mathbf{g}^{\ell-1}(\mathbf{x})\right)
$$

# D. Details on small-scale experiments

For the small-scale experiments in Section 3, we use the ALL-CNN-C architecture from Springenberg et al. (2014). We describe this architecture in Table 1. We train the model for 90 epochs, and only use the output latent variables and a posterior with 1 Gaussian component for this experiment

## E. Additional visualization of outputs at each layer

In Section 3, we provide a PCA visualization of the outputs from the last layer of a node-based ALL-CNN-C BNN on one sample of CIFAR 10. Here we also provide the same visualizations for the first two and the last two layers of the network. We use the same input image as Fig. 2.

## F. Additional details on the experiments and hyperparameters

## F.1. Approximation for the KL divergence with mixture variational posterior

We use a mixture of Gaussians (MoG) distribution with $K$ equally-weighted components to provide a flexible approximation of the true posterior in the latent space:

$$
\begin{gathered}
q(\mathcal{Z})=\frac{1}{K} \sum_{k=1}^{K} q_{k}(\mathcal{Z}) \\
q_{k}(\mathcal{Z})=\prod_{\ell=1}^{L} q_{k, \ell}\left(\mathcal{Z}^{\ell}\right) \\
q_{k, \ell}\left(\mathcal{Z}^{\ell}\right)=\mathcal{N}\left(\boldsymbol{\mu}_{k, \ell}, \operatorname{diag} \boldsymbol{\sigma}_{k, \ell}^{2}\right)
\end{gathered}
$$

where $L$ is the number of layers. We use a Gaussian prior with global scalar variance $s^{2}$ for the latent prior,

$$
p(\mathcal{Z})=\mathcal{N}\left(\mathbf{1}, s^{2} I\right)
$$

---

#### Page 14

The KL divergence decomposes into cross-entropy and entropy terms,

$$
\mathrm{KL}[q(\mathcal{Z}) \| p(\mathcal{Z})]=\mathbb{H}[q, p]-\mathbb{H}[q]=\frac{1}{K} \sum_{k=1}^{K} \mathbb{H}\left[q_{k}, p\right]-\mathbb{H}[q]
$$

where the cross-entropy reduces into tractable terms $\mathbb{H}\left[q_{k}, p\right]$ for Gaussians. The mixture entropy $\mathbb{H}[q]$ remains intractable, but admits a lower bound (Kolchinsky \& Tracey, 2017),

$$
\mathbb{H}[q] \geq \frac{1}{K} \sum_{k=1}^{K} \mathbb{H}\left[q_{k}\right]-\frac{1}{K} \sum_{k=1}^{K} \log \left(\frac{1}{K} \sum_{r=1}^{K} \mathrm{BC}\left(q_{k}, q_{r}\right)\right) \triangleq \widehat{\mathbb{H}}[q]
$$

where

$$
\mathrm{BC}\left(q, q^{\prime}\right)=\int \sqrt{q(\mathbf{z})} \sqrt{q^{\prime}(\mathbf{z})} d \mathbf{z} \quad \leq 1
$$

is the Bhattacharyya kernel of overlap between two distributions (Jebara \& Kondor, 2003; Jebara et al., 2004), and has a closed form solution for a pair of Gaussians $q, q^{\prime}$. The Bhattacharyya kernel has the convenient normalization property $\mathrm{BC}(q, q)=1$. The lower bound considers unary and pairwise component entropies.

# F.2. Experimental details and hyperparameters

We actually maximizes the following objective to train the node-based BNNs on large-scale experiments:

$$
\mathcal{L}_{\gamma, \beta}(\hat{\theta}, \phi)=\mathbb{E}_{q_{\phi}(\mathcal{Z})}[\log p(\mathcal{D} \mid \hat{\theta}, \mathcal{Z})]+\log p(\hat{\theta})+\beta\left(-\mathbb{H}\left[q_{\phi}(\mathcal{Z}), p(\mathcal{Z})\right]+(\gamma+1) \widehat{\mathbb{H}}\left[q_{\phi}(\mathcal{Z})\right]\right)
$$

which is the augmented ELBO in Eq. (19) with additional coefficient $\beta$ for the cross-entropy and variational entropy term. We also replace the intractable mixture entropy $\mathbb{H}[q]$ with its tractable lower bound $\widehat{\mathbb{H}}[q]$ presented in Eq. (47). During training, we will anneal $\beta$ from 0 to 1 . We found this to have ease optimization and produce better final results. For all experiments, we estimate the expected log-likelihood in the loss function using 4 samples.

For all the experiments on CIFAR10/CIFAR100, we run each experiment for 300 epochs, where we increase $\beta$ from 0 to 1 for the first 200 epochs. We use SGD as our optimizer, and we use a weight decay of 0.0005 for the parameters $\theta$. We use a batch size of 128. For all the experiments on TINYIMAGENET, we run each experiment for 150 epochs, where we increase $\beta$ from 0 to 1 for the first 100 epochs. We use a batch size of 256 . Bellow, we use $\lambda_{1}$ and $\lambda_{2}$ to denote the learning rate of the parameters $\theta$ and $\phi$ respectively.
For VGG16, we set the initial learning rate $\lambda_{1}=\lambda_{2}=0.05$, and we decrease $\lambda_{1}$ linearly from 0.05 to 0.0005 from epoch 150 to epoch 270 , while keeping $\lambda_{2}$ fixed throughout training. We initialize the standard deviations with $\mathcal{N}^{+}(0.30,0.02)$ and set the standard deviation of the prior to 0.30 .

For RESNET18, we set the initial learning rate $\lambda_{1}=\lambda_{2}=0.10$, and we decrease $\lambda_{1}$ linearly from 0.10 to 0.001 from epoch 150 to epoch 270 , while keeping $\lambda_{2}$ fixed throughout training. We initialize the standard deviations with $\mathcal{N}^{+}(0.40,0.02)$ and set the standard deviation of the prior to 0.40 .

For PRACTRESNET18, we set the initial learning rate $\lambda_{1}=\lambda_{2}=0.10$, and we decrease $\lambda_{1}$ linearly from 0.10 to 0.001 from epoch 75 to epoch 135 , while keeping $\lambda_{2}$ fixed throughout training. We initialize the standard deviations with $\mathcal{N}^{+}(0.30,0.02)$ and set the standard deviation of the prior to 0.30 .

## F.3. Runtime

We report the average running times of different methods in Table 2. We used similar number of epochs for all methods in each experiment. All experiment were performed on one Tesla V100 GPU. Overall, node BNNs took 4 times longer to train than SWAG since we use 4 Monte Carlo samples per training sample to estimate the expected log-likelihood in the $\gamma$-ELBO. ASAM took 2 times longer to train than SWAG since they require two forward-backward passes per minibatch.

## G. Additional benchmark results

Here we include the benchmark results of VGG16 on CIFAR10 and CIFAR100 in Fig. 13. We also include Fig. 14 and Fig. 15 as larger versions of Fig. 10 and Fig. 11.

---

#### Page 15

Table 2. Average running times of different methods measured in seconds. All experiments were performed on one Tesla V100 GPU.

| Model          | Dataset      | Node-BNN | SWAG  | ASAM  |
| :------------- | :----------- | :------- | :---- | :---- |
| VGG16          | CIFAR100     | 13274    | 3384  | 6870  |
|                | CIFAR10      | 12941    | 3251  | 6539  |
| ResNet18       | CIFAR100     | 18093    | 4528  | 9086  |
|                | CIFAR10      | 17733    | 4474  | 8921  |
| PreActResNet18 | TinyImagenet | 54892    | 13830 | 26564 |

# H. The evolution of variational entropy during training

We visualize the progression of the variational entropy when trained using the original ELBO (without the $\gamma$-entropy term) under different settings in Figs. 16-19. We can observe the typical behaviour of variational inference that it tends to reduce the entropy of the variational posterior over time.

## I. Additional results on the effect of $\gamma$ on performance of node-based BNNs

Here we include Figs. 20-24 to show the effect of $\gamma$ on performance of node-based BNNs under different architectures and datasets.

---

#### Page 16

> **Image description.** This image contains eight scatter plots arranged in a 2x4 grid. Each plot displays data points, an ellipse, and a legend.
>
> Here's a breakdown of the common elements and variations within each plot:
>
> - **Overall Structure:** Each plot has a title (conv1, conv2, conv7, conv8), x and y axes labeled as "Component [number] - [percentage]%", data points represented by circles, a red ellipse encompassing a portion of the data points, and a legend indicating the date associated with different colored data points. The grey unfilled circles represent samples from the output distribution induced by the latent variables.
>
> - **Axes:** The axes vary in scale and range across the plots. The x-axis is labeled "Component 1" in the top row and "Component 3" in the bottom row. The y-axis is labeled "Component 2" in the top row and "Component 4" in the bottom row. The percentage values in the axis labels also differ between plots.
>
> - **Data Points:** The data points are represented as circles and are colored according to the legend. The colors range from dark blue/purple to light green. Many data points are also shown as unfilled grey circles.
>
> - **Ellipse:** A red ellipse is present in each plot, enclosing a cluster of data points near the center of the plot. The shape and orientation of the ellipse vary slightly between plots.
>
> - **Legend:** Each plot includes a legend that maps specific dates (in the format "day-month/year") to different colors of data points. The dates vary slightly between plots.
>
> - **Titles:** The titles of the plots are "conv1", "conv2", "conv7", and "conv8", repeated in the same order for the top and bottom rows.
>
> In summary, the image presents a series of scatter plots visualizing data related to different convolutional layers (conv1, conv2, conv7, conv8), with each plot showing the distribution of data points in a two-dimensional space defined by different component axes. The red ellipse visually highlights a central cluster of data points, and the legend provides a mapping between data point colors and specific dates.
> (a) The outputs of the first two and last two layer in $\mathcal{M}_{16} . q(\mathcal{Z})$ is a single Gaussian with the standard deviations initialized from a half normal $\mathcal{N}^{+}(0.16,0.02)$

> **Image description.** The image contains eight scatter plots arranged in a 2x4 grid. Each plot displays data points in a two-dimensional space, with an ellipse encompassing a subset of the points. Each plot has a title indicating a layer name ("conv1", "conv2", "conv7", "conv8"), and labeled axes representing components and their explained variance ratio.
>
> Each scatter plot shows a distribution of data points, primarily colored in shades of purple, blue, and green, with some points appearing as unfilled grey circles. A red ellipse is drawn on each plot, enclosing a significant portion of the colored data points. The axes are labeled as "Component X - Y.YZ%", where X is a number (1, 2, 3, or 4) and Y.YZ is a percentage value. Each plot also includes a legend indicating the mapping between colors/shapes and date ranges in the format "N - DD/MM", where N is a number and DD/MM represents the day and month.
>
> The plots are arranged as follows:
>
> - Top row: "conv1", "conv2", "conv7", "conv8"
> - Bottom row: "conv1", "conv2", "conv7", "conv8"
>   The top row plots use Component 1 on the x-axis and Component 2 on the y-axis. The bottom row plots use Component 3 on the x-axis and Component 4 on the y-axis.
>   (b) The outputs of the first two and last two layer in $\mathcal{M}_{32}$ whose posterior $q(\mathcal{Z})$ is a single Gaussian with the standard deviations initialized from a half normal $\mathcal{N}^{+}(0.32,0.02)$.

Figure 12. PCA plots of the outputs for the first two and last two layers on a node-based ALL-CNN-C BNN with respect to one image from CIFAR10. Grey unfilled circle are samples from the output distribution induced by the latent variables, while the red ellipse is the 99 percentile of this distribution. The color circle represents the expected output $\mathbf{f}^{i}$ under input corruptions, where we fill the circle if it lies within the ellipse. Each axis label is the component index and its explained variance ratio. In the legend, we denote the severity of the corruptions and the ratio between number of points lie within the 99 percentile of the output distribution and the total number of corruption types. We use the corruptions from (Hendrycks \& Dietterich, 2019) containing 5 levels of severity and 19 types. For the model with larger $H[q(\mathcal{Z})]$ in 12 b , the number of points lie within the ellipse is higher than the model with smaller $H[q(\mathcal{Z})]$ in 12a.

---

#### Page 17

> **Image description.** The image consists of two rows of three scatter plots each, displaying the performance of different machine learning models under varying levels of data corruption. The top row represents results on CIFAR10, while the bottom row represents results on CIFAR100. Each plot in a row shares the same x-axis, labeled "Corruption level" with values ranging from 0 to 5. The y-axes represent different performance metrics: "ECE" (Expected Calibration Error), "NLL" (Negative Log-Likelihood), and "Error (%)". Arrows next to each metric indicate that lower values are better.
>
> Each scatter plot displays data points for several models: "node-BNN" (blue circles), "ens node-BNN" (blue circles with white fill), "SWAG" (orange squares), "ens SWAG" (orange squares with white fill), "ASAM" (grey diamonds), and "ens ASAM" (grey diamonds with white fill). Error bars are visible on some data points, indicating standard deviations.
>
> The first row of plots shows ECE ranging from 0 to 0.2, NLL ranging from 0 to 1.5, and Error (%) ranging from 5 to 35. The second row of plots shows ECE ranging from 0 to 0.3, NLL ranging from 0 to 3.5, and Error (%) ranging from 20 to 60.
>
> Vertical grey lines are present at each integer value on the x-axis to aid in visual comparison. The legend is located below the first row of plots.

Figure 13. Results of vGG16 on CIFAR10 (top) and CIFAR100 (bottom). We use $K=4$ and only the latent output variables for node-based BNNs. We plot ECE, NLL and error for different corruption levels, where level 0 indicates no corruption. We report the average performance over 19 corruption types for level 1 to 5 . We denote the ensemble of a method using the shorthand ens in front of the name. Each result is the average over 25 runs for non-ens versions and 5 runs for ens versions. The error bars represent the standard deviations across different runs. Node-based BNNs and their ensembles (blue) perform best in term of ECE and NLL on OOD data, while having similar accuracy to other methods.

---

#### Page 18

> **Image description.** The image contains two rows of three scatter plots each, showing the performance of different machine learning models under varying levels of data corruption. The top row represents results on CIFAR10, and the bottom row represents results on CIFAR100.
>
> Each plot has "Corruption level" on the x-axis, ranging from 0 to 5. The y-axes vary across the columns. The first column displays "ECE" (Expected Calibration Error), the second displays "NLL" (Negative Log-Likelihood), and the third displays "Error (%)". An arrow pointing downwards next to each y-axis label indicates that lower values are better.
>
> Each plot contains data points for six different models, distinguished by color and shape:
>
> - node-BNN (blue circles)
> - ens node-BNN (open blue circles)
> - SWAG (orange squares)
> - ens SWAG (open orange squares)
> - ASAM (gray diamonds)
> - ens ASAM (open gray diamonds)
> - cSG-HMC (brown downwards-pointing triangles)
> - ens cSG-HMC (open brown downwards-pointing triangles)
>
> Each data point has a small vertical error bar, indicating the standard deviation. The corruption level increases from left to right, and generally, the ECE, NLL, and Error increase with the corruption level.

Figure 14. Results of RESNET18 on CIFAR10 (top) and CIFAR100 (bottom). We use $K=4$ and only the latent output variables for node-based BNNs. We plot ECE, NLL and error for different corruption levels, where level 0 indicates no corruption. We report the average performance over 19 corruption types for level 1 to 5 . We denote the ensemble of a method using the shorthand ens in front of the name. Each result is the average over 25 runs for non-ens versions and 5 runs for ens versions. The error bars represent the standard deviations across different runs. Node-based BNNs and their ensembles (blue) perform best across all metrics on OOD data of CIFAR100, while having competitive results on CIFAR 10 .

---

#### Page 19

> **Image description.** The image consists of three scatter plots arranged horizontally. Each plot shows the performance of different machine learning methods under varying levels of data corruption.
>
> - **Plot 1 (ECE ↓):** The leftmost plot displays the Expected Calibration Error (ECE) on the y-axis, decreasing upwards, against the corruption level on the x-axis, ranging from 0 to 5. Several machine learning methods are compared, each represented by a different colored marker:
>
>   - node-BNN (blue circle)
>   - ens node-BNN (blue open circle)
>   - SWAG (orange square)
>   - ens SWAG (orange open square)
>   - ASAM (gray diamond)
>   - ens ASAM (gray open diamond)
>   - cSG-HMC (brown triangle pointing down)
>   - ens cSG-HMC (brown open triangle pointing down)
>     Error bars are visible for some data points, particularly for SWAG and ens SWAG.
>
> - **Plot 2 (NLL ↓):** The middle plot shows the Negative Log-Likelihood (NLL) on the y-axis, decreasing upwards, against the corruption level on the x-axis, ranging from 0 to 5. The same machine learning methods and markers are used as in the first plot.
>
> - **Plot 3 (Error (%) ↓):** The rightmost plot displays the Error (%) on the y-axis, decreasing upwards, against the corruption level on the x-axis, ranging from 0 to 5. The same machine learning methods and markers are used as in the first two plots.
>
> All three plots share the same x-axis label "Corruption level" and have vertical grid lines at each integer x-axis value. The y-axis scales differ between the plots to accommodate the different ranges of values for ECE, NLL, and Error (%). A legend is provided below all three plots, mapping the marker shapes and colors to the corresponding machine learning methods.

Figure 15. Results of PRACTRESNET18 on TINYIMAGENET. We use $K=4$ and only the latent output variables for node-based BNNs. We plot ECE, NLL and error for different corruption levels, where level 0 indicates no corruption. We report the average performance over 19 corruption types for level 1 to 5 . We denote the ensemble of a method using the shorthand ens in front of the name. Each result is the average over 25 runs for non-ens versions and 5 runs for ens versions. The error bars represent the standard deviations across different runs. Node-based BNNs and their ensembles (blue) perform best accross all metrics on OOD data, while having competitive performance on ID data.

VGG16 / CIFAR10

> **Image description.** The image consists of three line graphs arranged horizontally. Each graph shares the same y-axis label "H[q(Z)]" and x-axis label "Epoch". The x-axis ranges from 0 to 300 in each graph. The y-axis ranges from approximately 750 to 2250. Each graph contains three lines: a solid blue line labeled "out", a dashed orange line labeled "in", and a dash-dotted gray line labeled "both".
>
> - **Panel 1 (K=1):** The title above the first graph is "K = 1". The "out" and "in" lines are nearly overlapping and decrease from approximately 1100 to about 800 as Epoch increases. The "both" line starts at approximately 2200 and decreases to about 1600 as Epoch increases.
> - **Panel 2 (K=2):** The title above the second graph is "K = 2". Similar to the first graph, the "out" and "in" lines are nearly overlapping and decrease from approximately 1100 to about 900 as Epoch increases. The "both" line starts at approximately 2200 and decreases to about 1800 as Epoch increases.
> - **Panel 3 (K=4):** The title above the third graph is "K = 4". The "out" and "in" lines are nearly overlapping and decrease from approximately 1100 to about 900 as Epoch increases. The "both" line starts at approximately 2200 and decreases to about 2000 as Epoch increases.
>
> All three graphs have a light gray grid in the background.

Figure 16. The evolution of entropy during training for VGG16 / CIFAR10 when trained using the original ELBO. Each result is averaged over 5 runs. Each error bar represents one standard deviation but it is too small to be seen.

---

#### Page 20

VGG16 / CIFAR100

> **Image description.** This image consists of three line graphs arranged side-by-side. Each graph displays the evolution of entropy during training.
>
> - **Overall Structure:** The three graphs share a similar structure. Each has a horizontal axis labeled "Epoch" ranging from 0 to 300, and a vertical axis labeled "H[q(z)]" ranging from 0 to 2000. The graphs are plotted on a light gray grid.
>
> - **Graph 1 (K=1):** The title above this graph is "K = 1". Three lines are plotted:
>
>   - A solid blue line labeled "out".
>   - A dashed orange line labeled "in".
>   - A dash-dotted gray line labeled "both".
>     The "out" and "in" lines start at approximately the same y-value (around 1100) and gradually decrease to around 100. The "both" line starts at a much higher y-value (around 2200) and decreases more rapidly, eventually reaching a y-value around 300.
>
> - **Graph 2 (K=2):** The title above this graph is "K = 2". Three lines are plotted, with the same colors and labels as in Graph 1 ("out", "in", "both"). The "out" and "in" lines start at approximately the same y-value (around 1100) and gradually decrease to around 500. The "both" line starts at a much higher y-value (around 2200) and decreases more rapidly, eventually reaching a y-value around 500.
>
> - **Graph 3 (K=4):** The title above this graph is "K = 4". Three lines are plotted, with the same colors and labels as in the previous graphs ("out", "in", "both"). The "out" and "in" lines start at approximately the same y-value (around 1100) and gradually decrease to around 700. The "both" line starts at a much higher y-value (around 2200) and decreases more rapidly, eventually reaching a y-value around 1500.

Figure 17. The evolution of entropy during training for VGG16 / CIFAR100 when trained using the original ELBO. Each result is averaged over 5 runs. Each error bar represents one standard deviation but it is too small to be seen.

ResNet18 / CIFAR10

> **Image description.** The image shows three line graphs arranged horizontally. Each graph represents the evolution of entropy during training, with the x-axis labeled "Epoch" ranging from 0 to 300, and the y-axis labeled "H[q(z)]" ranging from approximately 2000 to 4000. Each graph contains three lines: a solid blue line labeled "out", a dashed orange line labeled "in", and a dash-dotted gray line labeled "both". The graphs differ in their titles: the first is labeled "K = 1", the second "K = 2", and the third "K = 4". All three graphs show a decreasing trend for all three lines as the number of epochs increases.

Figure 18. The evolution of entropy during training for RESNET18 / CIFAR10 when trained using the original ELBO. Each result is averaged over 5 runs. Each error bar represents one standard deviation but it is too small to be seen.

ResNet18 / CIFAR100

> **Image description.** This image consists of three line graphs arranged side-by-side. Each graph depicts the evolution of entropy during training, with the x-axis labeled "Epoch" ranging from 0 to 300, and the y-axis labeled "H[q(z)]" ranging from 0 to 4000.
>
> Each graph contains three lines representing different data sets. A solid blue line is labeled "out", a dashed orange line is labeled "in", and a dash-dotted gray line is labeled "both".
>
> The graphs differ in the value of "K", with the leftmost graph labeled "K = 1", the center graph labeled "K = 2", and the rightmost graph labeled "K = 4".
>
> In each graph, all three lines show a decreasing trend as the epoch increases, indicating a reduction in entropy during training. The "both" line consistently starts at a higher entropy value than the "out" and "in" lines.

Figure 19. The evolution of entropy during training for RESNET18 / CIFAR100 when trained using the original ELBO. Each result is averaged over 5 runs. Each error bar represents one standard deviation but it is too small to be seen.

---

#### Page 21

VGG16 / CIFAR10

> **Image description.** The image is a figure displaying results of VGG16 on CIFAR10 under different gamma values. It consists of 12 line graphs arranged in a 3x4 grid. Each row represents a different latent variable structure, and each graph shows the relationship between "γ/K" on the x-axis and "NLL" (Negative Log-Likelihood) on the y-axis.
>
> Here's a breakdown of the visual elements:
>
> - **Overall Structure:** The figure is titled "VGG16 / CIFAR10" at the top. The graphs are arranged in three rows, labeled "NLL (in)", "NLL (out)", and "NLL (both)" from top to bottom. The columns are labeled "Validation", "Test", "Corruption level 1,2,3", and "Corruption level 4,5" from left to right.
>
> - **Individual Graphs:** Each graph has the following characteristics:
>   - **Axes:** The x-axis is labeled "γ/K". The y-axis is labeled "NLL (in)", "NLL (out)", or "NLL (both)" depending on the row. The y-axis scales vary across the columns, with ranges of approximately 0.175 to 0.275 for "Validation" and "Test", 0.45 to 0.65 for "Corruption level 1,2,3", and 0.9 to 1.4 for "Corruption level 4,5". The x-axis range varies from 0 to 40 for the first two rows and 0 to 20 for the last row.
>   - **Data Lines:** Each graph contains three lines, representing different values of "K" (number of components): K=1 (solid blue line), K=2 (dashed orange line), and K=4 (dash-dotted gray line). Each line is surrounded by a shaded region, presumably indicating the standard deviation.
>   - **Legend:** A legend is present in the "Corruption level 4,5" column, indicating the line styles for K=1, K=2, and K=4.
>   - **Curves:** The lines generally show a U-shaped curve, indicating a minimum NLL value at a certain γ/K value. The exact shape and position of the minimum vary depending on the row and column.

Figure 20. Results of VGG16 on CIFAR 10 under different $\gamma$ value. $K$ is the number of components. Each row corresponds a different latent variable structure. We report the mean and standard deviation over 5 runs for each result.

---

#### Page 22

VGG16 / CIFAR100

> **Image description.** The image is a figure containing 12 line graphs arranged in a 3x4 grid. Each graph plots a variable labeled "NLL" on the y-axis against "γ/K" on the x-axis. The graphs are organized into three rows and four columns.
>
> - **Rows:** The rows are labeled "NLL (in)", "NLL (out)", and "NLL (both)" from top to bottom.
> - **Columns:** The columns are labeled "Validation", "Test", "Corruption level 1,2,3", and "Corruption level 4,5" from left to right.
> - **Axes:** All graphs have similar axes. The y-axis represents "NLL" and ranges from approximately 0.8 to 1.2 for the "Validation" and "Test" columns, 1.6 to 2.4 for "Corruption level 1,2,3", and 2.5 to 4.0 for "Corruption level 4,5". The x-axis represents "γ/K" and ranges from 0 to 75 for the top two rows, and 0 to 30 for the bottom row.
> - **Lines:** Each graph contains three lines representing different values of "K": 1 (solid blue), 2 (dashed orange), and 4 (dash-dotted gray). A legend in the top right corner of the "Corruption level 4,5" graphs identifies these lines. The lines generally show a U-shaped curve, indicating a minimum NLL value at a specific γ/K value. The "Corruption level" graphs show decreasing values of NLL as γ/K increases.
> - **Overall:** The figure seems to present the results of an experiment, comparing the performance of a model under different conditions (validation, test, corruption levels) and with different values of a parameter "K". The NLL value is likely a measure of error or loss, and the goal is to find the optimal γ/K value that minimizes this loss.

Figure 21. Results of VGG16 on CIFAR 100 under different $\gamma$ value. $K$ is the number of components. Each row corresponds a different latent variable structure. We report the mean and standard deviation over 5 runs for each result.

---

#### Page 23

> **Image description.** A set of twelve line graphs arranged in a 3x4 grid, displaying results of ResNet18 on CIFAR10 under varying gamma values (γ/K). Each row corresponds to a different latent variable structure, and the columns represent different evaluation conditions: "Validation", "Test", "Corruption level 1,2,3", and "Corruption level 4,5".
>
> Each individual graph has the following characteristics:
>
> - **Axes:** The x-axis is labeled "γ/K" and ranges from 0 to 20 for the first two columns and 0 to 10 for the last two columns. The y-axis represents the "NLL" (Negative Log-Likelihood) and varies in scale depending on the column. The first row is labeled "NLL (in)", the second "NLL (out)", and the third "NLL (both)".
> - **Data:** Each graph contains three lines representing different values of "K" (number of components): K=1 (solid blue line with a shaded area around it), K=2 (dashed orange line), and K=4 (dash-dotted black line).
> - **Legend:** A legend is present in the top-left graph of each row, indicating the line styles for K=1, K=2, and K=4.
> - **Titles:** Each graph has a title indicating the evaluation condition (Validation, Test, Corruption level 1,2,3, Corruption level 4,5). The title of the entire figure is "ResNet18 / CIFAR10".
>
> The general trend across the graphs is that NLL tends to increase with increasing γ/K for the "Test" condition, while the "Validation" condition shows a more complex relationship. The "Corruption level" graphs show a decrease in NLL as γ/K increases, then it plateaus.

Figure 22. Results of RESNET18 on CIFAR10 under different $\gamma$ value. $K$ is the number of components. Each row corresponds a different latent variable structure. We report the mean and standard deviation over 5 runs for each result.

---

#### Page 24

> **Image description.** The image is a figure containing twelve line graphs arranged in a 3x4 grid. Each row represents a different latent variable structure, labeled as "NLL (in)", "NLL (out)", and "NLL (both)" on the left side. The columns represent different conditions: "Validation", "Test", "Corruption level 1,2,3", and "Corruption level 4,5".
>
> Each graph plots the value of NLL (Negative Log-Likelihood) on the y-axis against "γ/K" on the x-axis. The y-axis ranges vary slightly between the columns, but generally span from around 0.75 to 0.95 for the "Validation" and "Test" columns, 1.5 to 1.8 for "Corruption level 1,2,3", and 2.4 to 3.0 for "Corruption level 4,5". The x-axis ranges from 0 to 20 for the top and bottom rows, and from 0 to 40 for the middle row.
>
> Each graph contains three lines, representing different values of K (number of components): K=1 (solid blue line), K=2 (dashed orange line), and K=4 (dash-dotted gray line). Each line is surrounded by a shaded area of the same color, representing the standard deviation over 5 runs.
>
> The legend indicating the K values is present in the "Validation" column of each row.

Figure 23. Results of RESNET18 on CIFAR 100 under different $\gamma$ value. $K$ is the number of components. Each row corresponds a different latent variable structure. We report the mean and standard deviation over 5 runs for each result.

---

#### Page 25

PreActResNet18 / TinylmageNet

> **Image description.** This image contains a set of line graphs arranged in a 3x4 grid, displaying the results of a machine learning model (PreActResNet18) on the TinyImageNet dataset. Each graph shows the relationship between "γ/K" (gamma/K) on the x-axis and "NLL" (Negative Log-Likelihood) on the y-axis for different values of "K," where K represents the number of components.
>
> Here's a breakdown of the visual elements:
>
> - **Overall Structure:** The image is divided into 12 subplots arranged in three rows and four columns.
>
> - **Titles:** Each column has a title indicating the data type: "Validation," "Test," "Corruption level 1,2,3," and "Corruption level 4,5." Each row has a title indicating the type of NLL: "NLL (in)", "NLL (out)", and "NLL (both)".
>
> - **Axes:**
>
>   - The x-axis of each graph is labeled "γ/K." The range of x-axis values varies between the top two rows (0-60) and the bottom row (0-30).
>   - The y-axis is labeled "NLL (in)", "NLL (out)", and "NLL (both)" for the first, second, and third rows, respectively. The y-axis range varies depending on the column. The first two columns range from approximately 1.4 to 1.7. The third column ranges from 3.0 to 4.0. The fourth column ranges from 4.0 to 5.5.
>
> - **Lines:** Each graph contains three lines, each representing a different value of "K":
>
>   - K = 1 (solid blue line)
>   - K = 2 (dashed orange line)
>   - K = 4 (dash-dotted gray line)
>     Each line is surrounded by a shaded region of the same color, presumably representing the standard deviation.
>
> - **Legend:** A legend is present in the top-right subplot (Corruption level 4,5), indicating the correspondence between line style and "K" value.
>
> - **Data Trends:** The lines generally show a decreasing trend in NLL as γ/K increases, although the exact shape and rate of decrease vary depending on the data type (Validation, Test, Corruption level) and the value of K.
>
> - **Text:** The following text is visible in the image:
>   - "PreActResNet18 / TinyImageNet" (at the top)
>   - "Validation" (column titles)
>   - "Test" (column titles)
>   - "Corruption level 1,2,3" (column titles)
>   - "Corruption level 4,5" (column titles)
>   - "NLL (in)" (row titles)
>   - "NLL (out)" (row titles)
>   - "NLL (both)" (row titles)
>   - "γ/K" (x-axis labels)
>   - "K" (legend title)
>   - "1", "2", "4" (legend labels)
>   - Numerical values on the axes.

Figure 24. Results of PRACTRESNET18 on TINYIMAGENET under different $\gamma$ value. $K$ is the number of components. Each row corresponds a different latent variable structure. We report the mean and standard deviation over 5 runs for each result.
