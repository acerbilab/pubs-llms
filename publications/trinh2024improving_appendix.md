# Improving robustness to corruptions with multiplicative weight perturbations - Appendix

---

#### Page 14

# A Proof of Lemma 1

Proof. Here we note that:

$$
\begin{aligned}
\mathbf{x}^{(h)} & :=\mathbf{f}^{(h)}(\mathbf{x}) \\
\mathbf{x}_{\mathbf{g}}^{(h)} & :=\mathbf{f}^{(h)}(\mathbf{g}(\mathbf{x})) \\
\boldsymbol{\delta}_{\mathbf{g}} \ell(\boldsymbol{\omega}, \mathbf{x}, y) & :=\ell\left(\boldsymbol{\omega}, \mathbf{x}_{\mathbf{g}}, y\right)-\ell(\boldsymbol{\omega}, \mathbf{x}, y) \\
\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)} & :=\mathbf{x}_{\mathbf{g}}^{(h)}-\mathbf{x}^{(h)}
\end{aligned}
$$

We first notice that the per-sample loss $\ell(\boldsymbol{\omega}, \mathbf{x}, y)$ can be viewed as a function of the intermediate activation $\mathbf{x}^{(h)}$ of layer $h$ (see Fig. 2). From Assumption 3, there exists a constant $L_{h}>0$ such that:

$$
\left\|\nabla_{\mathbf{x}_{\mathbf{g}}^{(h)}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{\mathbf{g}}, y\right)-\nabla_{\mathbf{x}^{(h)}} \ell(\boldsymbol{\omega}, \mathbf{x}, y)\right\|_{2} \leq L_{h}\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\|_{2}
$$

which gives us the following quadratic bound:

$$
\ell\left(\boldsymbol{\omega}, \mathbf{x}_{\mathbf{g}}, y\right) \leq \ell(\boldsymbol{\omega}, \mathbf{x}, y)+\left\langle\nabla_{\mathbf{x}^{(h)}} \ell(\boldsymbol{\omega}, \mathbf{x}, y), \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\rangle+\frac{L_{h}}{2}\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\|_{2}^{2}
$$

where $\langle\cdot, \cdot\rangle$ denotes the dot product between two vectors. The results in the equation above have been proven in Böhning and Lindsay (1988). Subtracting $\ell(\boldsymbol{\omega}, \mathbf{x}, y)$ from both side of Eq. (24) gives us:

$$
\boldsymbol{\delta}_{\mathbf{g}} \ell(\boldsymbol{\omega}, \mathbf{x}, y) \leq\left\langle\nabla_{\mathbf{x}^{(h)}} \ell(\boldsymbol{\omega}, \mathbf{x}, y), \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\rangle+\frac{L_{h}}{2}\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\|_{2}^{2}
$$

Since the pre-activation output of layer $h+1$ is $\mathbf{z}^{(h+1)}(\mathbf{x})=\mathbf{W}^{(h+1)} \mathbf{f}^{(h)}(\mathbf{x})=\mathbf{W}^{(h+1)} \mathbf{x}^{(h)}$, we can rewrite the inequality above as:

$$
\boldsymbol{\delta}_{\mathbf{g}} \ell(\boldsymbol{\omega}, \mathbf{x}, y) \leq\left\langle\nabla_{\mathbf{z}^{(h+1)}} \ell(\boldsymbol{\omega}, \mathbf{x}, y) \otimes \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}, \mathbf{W}^{(h+1)}\right\rangle_{F}+\frac{L_{h}}{2}\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\|_{2}^{2}
$$

where $\otimes$ denotes the outer product of two vectors and $\langle\cdot, \cdot\rangle_{F}$ denotes the Frobenius inner product of two matrices of similar dimension.
From Assumption 1, we have that there exists a constant $M>0$ such that:

$$
\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(0)}\right\|_{2}^{2}=\left\|\mathbf{x}_{\mathbf{g}}^{(0)}-\mathbf{x}^{(0)}\right\|_{2}^{2}=\|\mathbf{g}(\mathbf{x})-\mathbf{x}\|_{2}^{2} \leq M
$$

Given that $\mathbf{x}^{(1)}=\boldsymbol{\sigma}^{(1)}\left(\mathbf{W}^{(1)} \mathbf{x}^{(0)}\right)$, we have:

$$
\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(1)}\right\|_{2}^{2}=\left\|\mathbf{x}_{\mathbf{g}}^{(1)}-\mathbf{x}^{(1)}\right\|_{2}^{2} \leq\left\|\mathbf{W}^{(1)} \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(0)}\right\|_{2}^{2}
$$

Here we assume that the activate $\boldsymbol{\sigma}$ satisfies $\|\boldsymbol{\sigma}(\mathbf{x})-\boldsymbol{\sigma}(\mathbf{y})\|_{2} \leq\|\mathbf{x}-\mathbf{y}\|_{2}$, which is true for modern activation functions such as ReLU. Since $\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(0)}\right\|_{2}^{2}$ is bounded, there exists a constant $\hat{C}_{\mathbf{g}}^{(1)}(\mathbf{x})$ such that:

$$
\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(1)}\right\|_{2}^{2}=\left\|\mathbf{x}_{\mathbf{g}}^{(1)}-\mathbf{x}^{(1)}\right\|_{2}^{2} \leq\left\|\mathbf{W}^{(1)} \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(0)}\right\|_{2}^{2} \leq \frac{\hat{C}_{\mathbf{g}}^{(1)}(\mathbf{x})}{2}\left\|\mathbf{W}^{(1)}\right\|_{F}^{2}
$$

where $\|\cdot\|_{F}$ denotes the Frobenius norm. Similarly, as we have proven that $\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(1)}\right\|_{2}^{2}$ is bounded, there exists a constant $\hat{C}_{\mathbf{g}}^{(2)}(\mathbf{x})$ such that:

$$
\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(2)}\right\|_{2}^{2}=\left\|\mathbf{x}_{\mathbf{g}}^{(2)}-\mathbf{x}^{(2)}\right\|_{2}^{2} \leq\left\|\mathbf{W}^{(2)} \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(1)}\right\|_{2}^{2} \leq \frac{\hat{C}_{\mathbf{g}}^{(2)}(\mathbf{x})}{2}\left\|\mathbf{W}^{(2)}\right\|_{F}^{2}
$$

Thus we have proven that for all $h=1, \ldots, H$, there exists a constant $\hat{C}_{\mathbf{g}}^{(h)}(\mathbf{x})$ such that:

$$
\left\|\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}\right\|_{2}^{2} \leq \frac{\hat{C}_{\mathbf{g}}^{(h)}(\mathbf{x})}{2}\left\|\mathbf{W}^{(h)}\right\|_{F}^{2}
$$

By combining Eqs. (26) and (31) and setting $C_{\mathbf{g}}^{(h)}(\mathbf{x})=L_{h} \hat{C}_{\mathbf{g}}^{(h)}(\mathbf{x})$, we arrive at Eq. (4).

---

#### Page 15

# B Proof of Theorem 1

Proof. From Lemma 1, we have for all $h=0, \ldots, H-1$ :

$$
\begin{aligned}
& \mathcal{L}(\boldsymbol{\omega} ; \mathbf{g}(\mathcal{S}))=\frac{1}{N} \sum_{k=1}^{N} \ell\left(\boldsymbol{\omega}, \mathbf{g}\left(\mathbf{x}_{k}\right), y_{k}\right)=\frac{1}{N} \sum_{k=1}^{N}\left(\ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right)+\boldsymbol{\delta}_{\mathbf{g}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right)\right) \\
& \leq \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})+\frac{1}{N} \sum_{k=1}^{N}\left\langle\nabla_{\mathbf{z}^{(h+1)}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right) \otimes \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}_{k}^{(h)}, \mathbf{W}^{(h+1)}\right\rangle_{F}+\frac{\hat{C}_{\mathbf{g}}^{(h)}}{\left.\frac{1}{2} \|\mathbf{W}^{(h)}\|_{F}^{2}\right.}
\end{aligned}
$$

where $\hat{C}_{\mathbf{g}}^{(h)}=\max _{\mathbf{x} \in \mathcal{S}} C_{\mathbf{g}}^{(h)}(\mathbf{x})$. Since this bound is true for all $h$, we can take the average:

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\omega} ; \mathbf{g}(\mathcal{S})) \leq \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})+\frac{1}{H} \sum_{h=1}^{H} \frac{1}{N} \sum_{k=1}^{N}\left\langle\nabla_{\mathbf{z}^{(h)}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right) \otimes \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}_{k}^{(h-1)}, \mathbf{W}^{(h)}\right\rangle_{F} \\
+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2}
\end{aligned}
$$

where $C_{\mathbf{g}}=\frac{1}{H} \sum_{h=1}^{H} \hat{C}_{\mathbf{g}}^{(h)}$. The right-hand side of Eq. (34) can be written as:

$$
\begin{aligned}
& \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})+\frac{1}{H} \sum_{h=1}^{H}\left\langle\frac{1}{N} \sum_{k=1}^{N} \nabla_{\mathbf{z}^{(h)}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right) \otimes \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}_{k}^{(h-1)}, \mathbf{W}^{(h)}\right\rangle_{F}+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2} \\
& =\mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})+\sum_{h=1}^{H}\left\langle\nabla_{\mathbf{W}^{(h)}} \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S}), \mathbf{W}^{(h)} \circ \boldsymbol{\xi}^{(h)}(\mathbf{g})\right\rangle_{F}+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2} \\
& \leq \mathcal{L}(\boldsymbol{\omega}+\boldsymbol{\omega} \circ \boldsymbol{\xi}(\mathbf{g}) ; \mathcal{S})+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2}=\mathcal{L}(\boldsymbol{\omega} \circ(1+\boldsymbol{\xi}(\mathbf{g})) ; \mathcal{S})+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2}
\end{aligned}
$$

where $\boldsymbol{\xi}^{(h)}(\mathbf{g})$ is a matrix of the same dimension as $\mathbf{W}^{(h)}$ whose each entry is defined as:

$$
\left[\boldsymbol{\xi}^{(h)}(\mathbf{g})\right]_{i, j}=\frac{1}{H} \frac{\left[\sum_{k=1}^{N} \nabla_{\mathbf{z}^{(h)}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right) \otimes \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}_{k}^{(h-1)}\right]_{i, j}}{\left[\sum_{k=1}^{N} \nabla_{\mathbf{z}^{(h)}} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right) \otimes \mathbf{x}_{k}^{(h-1)}\right]_{i, j}}
$$

The inequality in Eq. (37) is due to the first-order Taylor expansion and the assumption that the training loss is locally convex at $\boldsymbol{\omega}$. This assumption is expected to hold for the final solution but does not necessarily hold for any $\boldsymbol{\omega}$. Eq. (5) is obtained by combining Eq. (34) and Eq. (37).

## C Training with corruption

Here we present Algorithm 2 which uses corruptions as data augmentation during training, as well as the experiment results of Section 4.1 for ResNet18/CIFAR-10 and PreActResNet18/TinyImageNet settings in Figs. 5 and 6.

## D Training with random additive weight perturbations

Here, we present Algorithm 3 used in Section 4.2 which trains DNNs under random additive weight perturbations and Fig. 7 comparing performance between DAMP and DAAP.

## E Corruption datasets

CIFAR-10/100-C (Hendrycks and Dietterich, 2019) These datasets contain the corrupted versions of the CIFAR-10/100 test sets. They contain 19 types of corruption, each divided into 5 levels of severity.

TinyImageNet-C (Hendrycks and Dietterich, 2019) This dataset contains the corrupted versions of the TinyImageNet test set. It contains 19 types of corruption, each divided into 5 levels of severity.

---

#### Page 16

Algorithm 2 Training with corruption
1: Input: training data $\mathcal{S}=\left\{\left(\mathbf{x}_{k}, y_{k}\right)\right\}_{k=1}^{N}$, a neural network $\mathbf{f}(\cdot ; \boldsymbol{\omega})$ parameterized by $\boldsymbol{\omega} \in \mathbb{R}^{P}$, number of iterations $T$, step sizes $\left\{\eta_{t}\right\}_{t=1}^{T}$, batch size $B$, a corruption $\mathbf{g}$ such as Gaussian noise, weight decay coefficient $\lambda$, a loss function $\mathcal{L}: \mathbb{R}^{P} \rightarrow \mathbb{R}_{+}$.
2: Output: Optimized parameter $\boldsymbol{\omega}^{(T)}$.t
3: Initialize parameter $\boldsymbol{\omega}^{(0)}$.
4: for $t=1$ to $T$ do
5: $\quad$ Draw a mini-batch $\mathcal{B}=\left\{\left(\mathbf{x}_{k}, y_{k}\right)\right\}_{k=1}^{B} \sim \mathcal{S}$.
6: Divide the mini-batch into two disjoint sub-batches of equal size $\mathcal{B}_{1}$ and $\mathcal{B}_{2}$.
7: Apply the corruption $\mathbf{g}$ to all samples in $\mathcal{B}_{1}: \mathbf{g}\left(\mathcal{B}_{1}\right)=\{(\mathbf{g}(\mathbf{x}), y)\}_{(\mathbf{x}, y) \in \mathcal{B}_{1}}$.
8: $\quad$ Compute the gradient $\mathbf{g}=\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega} ; \mathbf{g}\left(\mathcal{B}_{1}\right) \cup \mathcal{B}_{2}\right)$.
9: $\quad$ Update the weights: $\boldsymbol{\omega}^{(t+1)}=\boldsymbol{\omega}^{(t)}-\eta_{t}\left(\mathbf{g}+\lambda \boldsymbol{\omega}^{(t)}\right)$.
10: end for

> **Image description.** A heatmap displays the results of experiments on image corruption.
>
> The heatmap is a grid of colored cells, with rows and columns labeled with text. The rows are labeled with tuples of training methods and corruptions, specifically: (SGD, none), (DAMP, none), (SGD, zoom_blur), (SGD, shot_noise), (SGD, gaussian_noise), (SGD, motion_blur), and (SGD, pixelate). The columns are labeled with different types of image corruption: zoom_blur, shot_noise, gaussian_noise, motion_blur, pixelate, fog, snow, frost, glass_blur, impulse_noise, brightness, defocus_blur, elastic_transform, contrast, jpeg_compression, none, and Avg.
>
> Each cell in the grid contains a numerical value, and the color of the cell corresponds to the value according to a color scale on the right side of the image. The scale ranges from blue to red, with blue representing lower values (around 0.8) and red representing higher values (around 1.2). The first row, corresponding to (SGD, none), is uniformly gray, with all values equal to 1.00. Most other cells are shades of blue, but some cells, particularly in the columns for pixelate, brightness, jpeg_compression, and the Avg column, are red. The numerical values are overlaid on each cell in white.
>
> The column labels are rotated diagonally.

Figure 5: DAMP improves robustness to all corruptions while preserving accuracy on clean images. Results of ResNet18/CIFAR-10 experiments averaged over 3 seeds. The heatmap shows $\mathrm{CE}_{c}^{f}$ described in Eq. (18), where each row corresponds to a tuple of of training (method, corruption), while each column corresponds to the test corruption. The Avg column shows the average of the results of the previous columns. none indicates no corruption. We use the models trained under the SGD/none setting (first row) as baselines to calculate the $\mathrm{CE}_{c}^{f}$. The last five rows are the 5 best training corruptions ranked by the results in the Avg column.

> **Image description.** This is a heatmap displaying numerical data with a color gradient.
>
> - The heatmap is structured as a table. The rows are labeled on the left with text strings such as "(SGD, none)", "(DAMP, none)", "(SGD, zoom_blur)", "(SGD, pixelate)", "(SGD, jpeg_compression)", "(SGD, gaussian_noise)", and "(SGD, shot_noise)".
> - The columns are labeled at the top with text strings such as "zoom_blur", "pixelate", "jpeg_compression", "gaussian_noise", "shot_noise", "motion_blur", "brightness", "defocus_blur", "impulse_noise", "contrast", "fog", "elastic_transform", "snow", "frost", "none", and "Avg". The labels are rotated diagonally.
> - Each cell in the table contains a numerical value, ranging approximately from 0.70 to 1.17. The cells are colored based on these values, with a color scale shown on the right side of the heatmap.
> - The color scale ranges from blue at the bottom (corresponding to lower values around 0.90) to red at the top (corresponding to higher values around 1.10), with white in the middle (around 1.00).
> - The numerical values are formatted to two decimal places.
> - The table has gridlines separating the cells.

Figure 6: DAMP improves robustness to all corruptions while preserving accuracy on clean images. Results of PreActResNet18/TinyImageNet experiments averaged over 3 seeds. The heatmap shows $\mathrm{CE}_{c}^{f}$ described in Eq. (18), where each row corresponds to a tuple of training (method, corruption), while each column corresponds to the test corruption. The Avg column shows the average of the results of the previous columns. none indicates no corruption. We use the models trained under the SGD/none setting (first row) as baselines to calculate the $\mathrm{CE}_{c}^{f}$. The last five rows are the 5 best training corruptions ranked by the results in the Avg column.

ImageNet-C (Hendrycks and Dietterich, 2019) This dataset contains the corrupted versions of the ImageNet validation set, as the labels of the true ImageNet test set was never released. It contains 15 types of corruption, each divided into 5 levels of severity.

---

#### Page 17

Algorithm 3 DAAP: Data Augmentation via Additive Perturbations
1: Input: training data $\mathcal{S}=\left\{\left(\mathbf{x}_{k}, y_{k}\right)\right\}_{k=1}^{N}$, a neural network $\mathbf{f}(\cdot ; \boldsymbol{\omega})$ parameterized by $\boldsymbol{\omega} \in \mathbb{R}^{P}$, number of iterations $T$, step sizes $\left\{\eta_{t}\right\}_{t=1}^{T}$, number of sub-batch $M$, batch size $B$ divisible by $M$, a noise distribution $\boldsymbol{\Xi}=\mathcal{N}\left(\mathbf{0}, \sigma^{2} \mathbf{I}_{P}\right)$, weight decay coefficient $\lambda$, a loss function $\mathcal{L}: \mathbb{R}^{P} \rightarrow \mathbb{R}_{+}$.
2: Output: Optimized parameter $\boldsymbol{\omega}^{(T)}$.
3: Initialize parameter $\boldsymbol{\omega}^{(0)}$.
4: for $t=1$ to $T$ do
5: $\quad$ Draw a mini-batch $\mathcal{B}=\left\{\left(\mathbf{x}_{b}, y_{b}\right)\right\}_{b=1}^{B} \sim \mathcal{S}$.
6: Divide the mini-batch into $M$ disjoint sub-batches $\left\{\mathcal{B}_{m}\right\}_{m=1}^{M}$ of equal size.
7: for $m=1$ to $M$ in parallel do
8: Draw a noise sample $\boldsymbol{\xi}_{m} \sim \boldsymbol{\Xi}$.
9: $\quad$ Compute the gradient $\left.\mathbf{g}_{m}=\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega} ; \mathcal{B}_{m}\right)\right|_{\boldsymbol{\omega}^{(t)}+\boldsymbol{\xi}}$.
10: end for
11: Compute the average gradient: $\mathbf{g}=\frac{1}{M} \sum_{m=1}^{M} \mathbf{g}_{m}$.
12: Update the weights: $\boldsymbol{\omega}^{(t+1)}=\boldsymbol{\omega}^{(t)}-\eta_{t}\left(\mathbf{g}+\lambda \boldsymbol{\omega}^{(t)}\right)$.
13: end for

> **Image description.** The image consists of three horizontal bar charts comparing the performance of different data augmentation techniques under varying levels of data corruption.
>
> - **Overall Structure:** The image is divided into three panels, each representing a different model and dataset combination. From left to right, the panels are labeled "ResNet18 / CIFAR-10", "ResNet18 / CIFAR-100", and "PreActResNet18 / TinyImageNet".
>
> - **Chart Structure:** Each panel contains a horizontal bar chart. The y-axis represents the "Corruption intensity" with three levels: "None", "Mild", and "Severe". The x-axis represents "Error (%) ↓", with values ranging from 0 to 40 in the first panel, 0 to 60 in the second, and 0 to 80 in the third.
>
> - **Data Representation:** Each corruption intensity level has multiple horizontal bars representing different data augmentation techniques and their corresponding error rates.
>
>   - The bars are color-coded to represent different algorithms and noise standard deviations (sigma).
>   - "DAMP" is represented by shades of blue, with increasing darkness corresponding to higher sigma values (0.100, 0.200, 0.300, 0.400).
>   - "DAAP" is represented by shades of red, with increasing darkness corresponding to higher sigma values (0.005, 0.010, 0.020, 0.040).
>   - "SGD" is represented by a green bar.
>   - Each bar has a small error bar at its end.
>
> - **Legend:** A legend is located below the charts, mapping the colors to the corresponding algorithms and sigma values. The legend reads: "DAMP, σ = 0.100", "DAMP, σ = 0.200", "DAMP, σ = 0.300", "DAMP, σ = 0.400", "DAAP, σ = 0.005", "DAAP, σ = 0.010", "DAAP, σ = 0.020", "DAAP, σ = 0.040", and "SGD".

Figure 7: DAMP has better corruption robustness than DAAP. We report the predictive errors (lower is better) averaged over 5 seeds. None indicates no corruption. Mild includes severity levels 1, 2 and 3. Severe includes severity levels 4 and 5. We evaluate DAMP and DAAP under different noise standard deviations $\sigma$. These results imply that the multiplicative weight perturbations of DAMP are more effective than the additive perturbations of DAAP in improving robustness to corruptions.

ImageNet- $\overline{\mathbf{C}}$ (Mintun et al., 2021) This dataset contains the corrupted versions of the ImageNet validation set, as the labels of the true ImageNet test set was never released. It contains 10 types of corruption, each divided into 5 levels of severity. The types of corruption in ImageNet- $\overline{\text { C }}$ differ from those in ImageNet-C.

ImageNet-A (Hendrycks et al., 2021) This dataset contains natural adversarial examples, which are real-world, unmodified, and naturally occurring examples that cause machine learning model performance to significantly degrade. The images contain in this dataset, while differ from those in the ImageNet validation set, stills belong to the same set of classes.

ImageNet-D (Zhang et al., 2024) This dataset contains images belong to the classes of ImageNet but they are modified by diffusion models to change the background, material, and texture.

ImageNet-Cartoon and ImageNet-Drawing (Salvador and Oberman, 2022) This dataset contains the drawing and cartoon versions of the images in the ImageNet validation set.

ImageNet-Sketch (Wang et al., 2019) This dataset contains sketch images belonging to the classes of the ImageNet dataset.

ImageNet-Hard (Taesiri et al., 2023) This dataset comprises an array of challenging images, curated from several validation datasets of ImageNet.

---

#### Page 18

# F Training details

For each method and each setting, we tune the important hyperparameters ( $\sigma$ for DAMP, $\rho$ for SAM and ASAM) using $10 \%$ of the training set as validation set.

CIFAR-10/100 For each setting, we train a ResNet18 for 300 epochs. We use a batch size of 128. We use a learning rate of 0.1 and a weight decay coefficient of $5 \times 10^{-4}$. We use SGD with Nesterov momentum as the optimizer with a momentum coefficient of 0.9 . The learning rate is kept at 0.1 until epoch 150 , then is linearly annealed to 0.001 from epoch 150 to epoch 270 , then kept at 0.001 for the rest of the training. We use basic data preprocessing, which includes channel-wise normalization, random cropping after padding and random horizontal flipping. On CIFAR-10, we set $\sigma=0.2$ for DAMP, $\rho=0.045$ for SAM and $\rho=1.0$ for ASAM. On CIFAR-100, we set $\sigma=0.1$ for DAMP, $\rho=0.06$ for SAM and $\rho=2.0$ for ASAM. Each method is trained on a single host with 8 Nvidia V100 GPUs where the data batch is evenly distributed among the GPUs at each iteration (data parallelism). This means we use the number of sub-batches $M=8$ for DAMP.

TinyImageNet For each setting, we train a PreActResNet18 for 150 epochs. We use a batch size of 128 . We use a learning rate of 0.1 and a weight decay coefficient of $2.5 \times 10^{-4}$. We use SGD with Nesterov momentum as the optimizer with a momentum coefficient of 0.9 . The learning rate is kept at 0.1 until epoch 75 , then is linearly annealed to 0.001 from epoch 75 to epoch 135 , then kept at 0.001 for the rest of the training. We use basic data preprocessing, which includes channel-wise normalization, random cropping after padding and random horizontal flipping. We set $\sigma=0.2$ for DAMP, $\rho=0.2$ for SAM and $\rho=3.0$ for ASAM. Each method is trained on a single host with 8 Nvidia V100 GPUs where the data batch is evenly distributed among the GPUs at each iteration (data parallelism). This means we use the number of sub-batches $M=8$ for DAMP.

ResNet50 / ImageNet We train each experiment for 90 epochs. We use a batch size of 2048. We use a weight decay coefficient of $1 \times 10^{-4}$. We use SGD with Nesterov momentum as the optimizer with a momentum coefficient of 0.9 . We use basic Inception-style data preprocessing, which includes random cropping, resizing to the resolution of $224 \times 224$, random horizontal flipping and channel-wise normalization. We increase the learning rate linearly from $8 \times 10^{-4}$ to 0.8 for the first 5 epochs then decrease the learning rate from 0.8 to $8 \times 10^{-4}$ using a cosine schedule for the remaining epochs. All experiments were run on a single host with 8 Nvidia V100 GPUs and we set $M=8$ for DAMP. We use $p=0.05$ for Dropout, $\sigma=0.1$ for DAMP, $\rho=0.05$ for SAM, and $\rho=1.5$ for ASAM. We also use the image resolution of $224 \times 224$ during evaluation.

ViT-S16 / ImageNet / Basic augmentations We follow the training setup of Beyer et al. (2022) with one difference is that we only use basic Inception-style data processing similar to the ResNet50/ImageNet experiments. We use AdamW as the optimizer with $\beta_{1}=0.9, \beta_{2}=0.999$ and $\epsilon=10^{-8}$. We clip the gradient norm to 1.0 . We use a weight decay coefficient of 0.1 . We use a batch size of 1024 . We increase the learning rate linearly from $10^{-6}$ to $10^{-3}$ for the first 10000 iterations, then we anneal the learning rate from $10^{-3}$ to 0 using a cosine schedule for the remaining iterations. We use the image resolution of $224 \times 224$ for both training and testing. Following Beyer et al. (2022), we make 2 minor modifications to the original ViT-S16 architecture: (1) We change the position embedding layer from learnable to sincos2d; (2) We change the input of the final classification layer from the embedding of the [cls] token to global average-pooling. All experiments were run on a single host with 8 Nvidia V100 GPUs and we set $M=8$ for DAMP. We use $p=0.10$ for Dropout, $\sigma=0.25$ for DAMP, $\rho=0.6$ for SAM, and $\rho=3.0$ for ASAM.

ViT-S16 and B16 / ImageNet / MixUp and RandAugment Most of the hyperparameters are identical to the ViT-S16 / ImageNet / Basic augmentations setting. With ViT-S16, we use $p=0.1$ for Dropout, $\sigma=0.10$ for DAMP, $\rho=0.015$ for SAM, and $\rho=0.4$ for ASAM. With ViT-B16, we use $p=0.1$ for Dropout, $\sigma=0.15$ for DAMP, $\rho=0.025$ for SAM, and $\rho=0.6$ for ASAM.
