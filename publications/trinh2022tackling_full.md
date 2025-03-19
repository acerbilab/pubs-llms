```
@article{trinh2022tackling,
  title={Tackling covariate shift with node-based Bayesian neural networks},
  author={Trung Trinh and Markus Heinonen and Luigi Acerbi and Samuel Kaski},
  year={2022},
  journal={International Conference on Machine Learning},
  doi={10.48550/arXiv.2206.02435}
}
```

---

#### Page 1

# Tackling covariate shift with node-based Bayesian neural networks

Trung Trinh ${ }^{1}$ Markus Heinonen ${ }^{1}$ Luigi Acerbi ${ }^{2}$ Samuel Kaski ${ }^{13}$

#### Abstract

Bayesian neural networks (BNNs) promise improved generalization under covariate shift by providing principled probabilistic representations of epistemic uncertainty. However, weightbased BNNs often struggle with high computational complexity of large-scale architectures and datasets. Node-based BNNs have recently been introduced as scalable alternatives, which induce epistemic uncertainty by multiplying each hidden node with latent random variables, while learning a point-estimate of the weights. In this paper, we interpret these latent noise variables as implicit representations of simple and domainagnostic data perturbations during training, producing BNNs that perform well under covariate shift due to input corruptions. We observe that the diversity of the implicit corruptions depends on the entropy of the latent variables, and propose a straightforward approach to increase the entropy of these variables during training. We evaluate the method on out-of-distribution image classification benchmarks, and show improved uncertainty estimation of node-based BNNs under covariate shift due to input perturbations. As a side effect, the method also provides robustness against noisy training labels.

## 1. Introduction

Bayesian neural networks (BNNs) induce epistemic uncertainty over predictions by placing a distribution over the weights (MacKay, 1992; 1995; Hinton \& van Camp, 1993; Neal, 1996). However, it is challenging to infer the weight posterior due to the high dimensionality and multi-modality of this distribution (Wenzel et al., 2020; Izmailov et al., 2021b). Alternative BNN methods have been introduced

[^0]to avoid the complexity of weight-space inference, which include combining multiple maximum-a-posteriori (MAP) solutions (Lakshminarayanan et al., 2017), performing inference in the function-space (Sun et al., 2019), or performing inference in a lower dimensional latent space (Karaletsos et al., 2018; Pradier et al., 2018; Izmailov et al., 2020; Dusenberry et al., 2020).

A recent approach to simplify BNNs is node stochasticity, which assigns latent noise variables to hidden nodes of the network (Kingma et al., 2015; Gal \& Ghahramani, 2016; Karaletsos et al., 2018; Karaletsos \& Bui, 2020; Dusenberry et al., 2020; Nguyen et al., 2021). By restricting inference to the node-based latent variables, node stochasticity greatly reduces the dimension of the posterior, as the number of nodes is orders of magnitude smaller than the number of weights in a neural network (Dusenberry et al., 2020). Within this framework, multiplying each hidden node with its own random variable has been shown to produce great predictive performance, while having dramatically smaller computational complexity compared to weight-space BNNs (Gal \& Ghahramani, 2016; Kingma et al., 2015; Dusenberry et al., 2020; Nguyen et al., 2021).

In this paper, we focus on node-based BNNs, which represent epistemic uncertainty by inferring the posterior distribution of the multiplicative latent node variables while learning a point-estimate of the weight posterior (Dusenberry et al., 2020; Trinh et al., 2020). We show that node stochasticity simulates a set of implicit corruptions in the data space during training, and by learning in the presence of such corruptions, node-based BNNs achieve natural robustness against some real-world input corruptions. This is an important property because one of the key promises of BNNs is robustness under covariate shift (Ovadia et al., 2019; Izmailov et al., 2021b), defined as a change in the distribution of input features at test time with respect to that of the training data. Based on our findings, we derive an entropy regularization approach to improve out-of-distribution generalization for node-based BNNs.

In summary, our contributions are:

1. We demonstrate that node stochasticity simulates dataspace corruptions during training. We show that the diversity of these corruptions corresponds to the entropy

[^0]:
    ${ }^{1}$ Department of Computer Science, Aalto University, Finland
    ${ }^{2}$ Department of Computer Science, University of Helsinki, Finland
    ${ }^{3}$ Department of Computer Science, University of Manchester, UK. Correspondence to: Trung Trinh <trung.trinh@aalto.fi>.

---

#### Page 2

of the latent node variables, and training on more diverse generated corruptions produce node-based BNNs that are robust against a wider range of corruptions. 2. We derive an entropy-regularized variational inference formulation for node-based BNNs. 3. We demonstrate excellent empirical results in predictive uncertainty estimation under covariate shift due to corruptions compared to strong baselines on largescale image classification tasks. 4. We show that, as a side effect, our approach provides robust learning in the presence of noisy training labels.

Our code is available at https://github.com/ AaltoPML/node-BNN-covariate-shift.

## 2. Background

Neural networks. We define a standard neural network $\mathbf{f}(\mathbf{x})$ with $L$ layers for an input $\mathbf{x}$ as follows:

$$
\begin{aligned}
\mathbf{f}^{0}(\mathbf{x}) & =\mathbf{x} \\
\mathbf{h}^{\ell}(\mathbf{x}) & =\mathbf{W}^{\ell} \mathbf{f}^{\ell-1}(\mathbf{x})+\mathbf{b}^{\ell} \\
\mathbf{f}^{\ell}(\mathbf{x}) & =\sigma^{\ell}\left(\mathbf{h}^{\ell}(\mathbf{x})\right), \quad \forall \ell=1, \ldots, L \\
\mathbf{f}(\mathbf{x}) & =\mathbf{f}^{L}(\mathbf{x})
\end{aligned}
$$

where the parameters $\theta=\left\{\mathbf{W}^{\ell}, \mathbf{b}^{\ell}\right\}_{\ell=1}^{L}$ consist of the weights and biases, and the $\left\{\sigma^{\ell}\right\}_{\ell=1}^{L}$ are the activation functions. For the $\ell$-th layer, $\mathbf{h}^{\ell}$ and $\mathbf{f}^{\ell}$ are the pre- and postactivations, respectively.

Node-based Bayesian neural networks. Probabilistic neural networks constructed using node stochasticity have been studied by Gal \& Ghahramani (2016); Kingma et al. (2015); Louizos \& Welling (2017); Karaletsos et al. (2018); Karaletsos \& Bui (2020); Dusenberry et al. (2020); Trinh et al. (2020); Nguyen et al. (2021). We focus on inducing node stochasticity by multiplying each hidden node with its own random latent variables, and follow the framework of Dusenberry et al. (2020) for optimization. A node-based $\mathrm{BNN} \mathbf{f}_{\mathcal{Z}}(\mathbf{x})$ is defined as:

$$
\begin{aligned}
\mathbf{f}_{\mathcal{Z}}^{0}(\mathbf{x}) & =\mathbf{x} \\
\mathbf{h}_{\mathcal{Z}}^{\ell}(\mathbf{x}) & =\left(\mathbf{W}^{\ell}\left(\mathbf{f}_{\mathcal{Z}}^{\ell-1}(\mathbf{x}) \circ \mathbf{z}^{\ell}\right)+\mathbf{b}^{\ell}\right) \circ \mathbf{s}^{\ell} \\
\mathbf{f}_{\mathcal{Z}}^{\ell}(\mathbf{x}) & =\sigma^{\ell}\left(\mathbf{h}_{\mathcal{Z}}^{\ell}(\mathbf{x})\right), \quad \forall \ell=1, \ldots, L \\
\mathbf{f}_{\mathcal{Z}}(\mathbf{x}) & =\mathbf{f}_{\mathcal{Z}}^{\ell}(\mathbf{x})
\end{aligned}
$$

where $\mathbf{z}^{\ell}$ and $\mathbf{s}^{\ell}$ are the multiplicative latent random variables of the incoming and outgoing signal of the nodes of

> **Image description.** The image is a diagram illustrating a process, possibly related to neural networks or machine learning.
>
> - **Central Shape:** A large, amorphous shape outlined in light purple dominates the center. It has an irregular boundary with several curves and indentations. Inside the shape are two small circles, one light red and one light blue.
>
> - **Text and Arrows:**
>
>   - To the left of the shape, "p(f_z^l)" is written in black. An arrow in light purple points from this text towards the shape, terminating at the light red circle.
>   - Above the shape, "f^l = E[f_z^l]" is written in black. A red arrow points from this text towards the light red circle inside the shape.
>   - Below the light blue circle inside the shape, "f^l + g_0^l" is written in black. A light blue arrow points from the light blue circle towards this text.
>   - To the right of the shape, "f^l + g_1^l" is written in black. A light green arrow points from the light green circle towards this text.
>
> - **Images:**
>
>   - To the left of the central shape, a small, square image shows a blurry object, possibly a dog or other animal, in brown and white tones.
>   - To the upper right of the central shape, two small, square images are labeled "g_0" and "g_1" respectively. Both show blurry objects similar to the one on the left, but with added noise or distortions in the form of colored dots.
>
> - **Connections:**
>   - A dashed line extends from the light red circle inside the shape to the "g_0" image. A small label "g_0^l" is placed near this line.
>   - A dashed line extends from the "g_0" image to a light blue circle.
>   - A dashed line extends from the light red circle inside the shape to the "g_1" image. A small label "g_1^l" is placed near this line.
>   - A dashed line extends from the "g_1" image to a light green circle.
>
> The arrangement suggests a flow or transformation process, where the initial image on the left is processed through the central shape, resulting in two modified images on the right. The text labels and arrows indicate mathematical operations or relationships between these elements.

Figure 1. A sketch depicting the connection between the output distribution at the $\ell$-th layer induced by node stochasticity (purple) centered on the average output ( $\odot$ ), and the output shifts generated by input corruptions ( $\odot, \odot$ ). We expect good performance under mild corruption $\mathbf{g}_{0}$, as the resulting shift remains inside the highdensity region of $p\left(\mathbf{f}_{\mathcal{Z}}^{\ell}\right)$, and worse results under severe corruption $\mathbf{g}_{1}$.

the $\ell$-th layer, and $\circ$ denotes the Hadamard (element-wise) product. We collect all latent variables to $\mathcal{Z}=\left\{\mathbf{z}^{\ell}, \mathbf{s}^{\ell}\right\}_{\ell=1}^{L}$. ${ }^{1}$
To learn the network parameters, we follow Dusenberry et al. (2020) and perform variational inference (Blei et al., 2017) over the weight parameters $\theta$ and latent node variables $\mathcal{Z}$. We begin by defining a prior $p(\theta, \mathcal{Z})=p(\theta) p(\mathcal{Z})$. We set a variational posterior approximation $q_{\hat{\theta}, \phi}(\theta, \mathcal{Z})=$ $q_{\hat{\theta}}(\theta) q_{\phi}(\mathcal{Z})$, where $q_{\hat{\theta}}(\theta)=\delta(\theta-\hat{\theta})$ is a Dirac delta distribution and $q_{\phi}(\mathcal{Z})$ is a Gaussian or a mixture of Gaussians distribution. We infer the posterior by minimizing the Kullback-Leibler (KL) divergence between variational approximation $q$ and true posterior $p(\theta, \mathcal{Z} \mid \mathcal{D})$. This is equivalent to maximizing the evidence lower bound (ELBO):

$$
\begin{aligned}
\mathcal{L}(\hat{\theta}, \phi)=\mathbb{E}_{q_{\phi}(\mathcal{Z})} & {\left[\log p(\mathcal{D} \mid \hat{\theta}, \mathcal{Z})\right] } \\
& -\mathrm{KL}\left[q_{\phi}(\mathcal{Z}) \| p(\mathcal{Z})\right]+\log p(\hat{\theta})
\end{aligned}
$$

In essence, we find a MAP solution for the more numerous weights $\theta$, while inferring the posterior distribution of the latent variables $\mathcal{Z}$. We refer the reader to Appendix A for detailed derivations.

Neural networks under covariate shift. In this paper, we focus on covariate shift from input corruptions, following the setting of Hendrycks \& Dietterich (2019). To simulate covariate shift, one can take an input $\mathbf{x}$ assumed to come from the same distribution as the training samples and apply

[^0]
[^0]: ${ }^{1}$ In this paper, we use a slightly more general definition of node-based BNN with two noise variables per node, and compare it with single-variable variants in Section 5.

---

#### Page 3

an input corruption $\mathbf{g}^{0}$ to form a shifted version $\mathbf{x}^{c}$ of $\mathbf{x}$ :

$$
\mathbf{x}^{c}=\mathbf{x}+\mathbf{g}^{0}(\mathbf{x})
$$

For instance, $\mathbf{x}$ could be an image and $\mathbf{g}^{0}$ can represent the shot noise corruption (Hendrycks \& Dietterich, 2019). The input corruption $\mathbf{g}^{0}(\mathbf{x})$ creates a shift in the output of each layer $\mathbf{g}^{\ell}(\mathbf{x})$ (see Fig. 1). We can approximate these shifts by first-order Taylor expansion (see Appendix C for full derivation),

$$
\begin{aligned}
\frac{\text { shift }}{\mathbf{g}^{\ell}(\mathbf{x})} & =\overbrace{\mathbf{f}^{\ell}\left(\mathbf{x}^{c}\right)}^{\text {corrupted output }} \cdot \overbrace{\mathbf{f}^{\ell}(\mathbf{x})}^{\text {clean output }} \\
& \approx \mathbf{J}_{\sigma}\left[\mathbf{h}^{\ell}(\mathbf{x})\right]\left(\mathbf{W}^{\ell} \mathbf{g}^{\ell-1}(\mathbf{x})\right)
\end{aligned}
$$

where $\mathbf{J}_{\sigma^{\ell}}=\partial \sigma^{\ell} / \partial \mathbf{h}^{\ell}$ denotes the (diagonal) Jacobian of the activation $\sigma^{\ell}$ with respect to $\mathbf{h}^{\ell}$. While $\mathbf{g}^{0}$ causes activation shifts in every layer of the network, we focus on the shift in the final output layer $\mathbf{g}^{L}$. The approximation in Eq. (12) shows that this shift depends on the input $\mathbf{x}$, the network's architecture (e.g., choice of activation functions) and parameters $\theta$. We measure the robustness of a network with respect to a corruption $\mathbf{g}^{0}(\cdot)$ on the dataset $\mathcal{D}=\left\{\mathbf{x}_{n}, \mathbf{y}_{n}\right\}_{n=1}^{N}$ by the induced mean square shift,

$$
\mathrm{MSS}_{g}=\frac{1}{N} \sum_{n=1}^{N}\left\|\mathbf{g}^{L}\left(\mathbf{x}_{n}\right)\right\|_{2}^{2}
$$

where $\mathrm{MSS}_{g}$ is the average shift on the data. Ideally, we want $\mathrm{MSS}_{g}$ to be small for the network to still provide nearly correct predictions given corrupted inputs. When the training data and the architecture are fixed, $\mathrm{MSS}_{g}$ depends on the parameters $\theta$. A direct approach to find $\theta$ minimizing $\mathrm{MSS}_{g}$ is to apply the input corruption $\mathbf{g}^{0}$ to each input $\mathbf{x}_{n}$ during training to teach the network to output the correct label $\mathbf{y}_{n}$ given $\mathbf{g}^{0}\left(\mathbf{x}_{n}\right)$. However, this approach is not domain-agnostic and requires defining a list of corruptions beforehand. In the next sections, we discuss the usage of multiplicative latent node variables as an implicit way to simulate covariate shifts during training.

## 3. Characterizing implicit corruptions

In this section, we demonstrate that multiplicative node variables correspond to implicit input corruptions. We show how to extract and visualize these new corruptions.

### 3.1. Relating input corruptions and multiplicative nodes

The node-based BNN of Eqs. (5)-(8) induces the predictive posterior $p\left(\mathbf{f}_{\mathcal{S}}^{\ell}(\mathbf{x})\right)$ over the $\ell$-th layer outputs by marginalizing over the variational latent parameter posterior $q\left(\mathcal{Z}_{\leq \ell}\right)$. Optimization of the variational objective in Eq. (9) enforces the model to achieve low loss on the training data despite
each layer output being corrupted by noise from $q(\mathcal{Z})$, represented by the expected log likelihood term of the ELBO. Let $\hat{\mathbf{f}}^{\ell}(\mathbf{x})$ denote the mean predictive posterior,

$$
\hat{\mathbf{f}}^{\ell}(\mathbf{x})=\mathbb{E}_{q(\mathcal{Z})}\left[\mathbf{f}_{\mathcal{Z}}^{\ell}(\mathbf{x})\right], \quad \forall \ell=1, \ldots, L
$$

and where we denote the final output $\hat{\mathbf{f}}(\mathbf{x})=\hat{\mathbf{f}}^{L}(\mathbf{x})$. If the shifted output $\hat{\mathbf{f}}^{\ell}\left(\mathbf{x}+\mathbf{g}^{0}(\mathbf{x})\right)=\hat{\mathbf{f}}^{\ell}(\mathbf{x})+\mathbf{g}^{\ell}(\mathbf{x})$ caused by corrupting a training sample $\mathbf{x}$ using $\mathbf{g}^{0}$ lies within the predictive distribution of $\mathbf{f}_{\mathcal{S}}^{\ell}(\mathbf{x})$ (blue dot in Fig. 1), then the model can map this corrupted version of $\mathbf{x}$ to its correct label. This implies robustness against the space of implicit corruptions generated by $q(\mathcal{Z})$, which indirectly leads to robustness against real corruptions. However, standard variational inference will converge to a posterior whose entropy is calibrated for the variability in the training data, but does not necessarily account for corruptions caused by covariate shifts. Thus, the posterior might cover the corruption with low severity $\mathbf{g}_{0}$ (blue dot in Fig. 1), but not the one with higher severity $\mathbf{g}_{1}$ (green dot in Fig. 1). To promote predictive distributions that are more robust to perturbations, we propose to increase the entropy of $p\left(\mathbf{f}_{\mathcal{Z}}^{\ell}(\mathbf{x})\right)$ by increasing the entropy of the variational posterior $q(\mathcal{Z})$.

Empirical demonstration. To illustrate our intuition, we present an example with two node-based BNNs, one with high entropy and one with lower entropy. We use the ALL-CNN-C architecture of Springenberg et al. (2014) and CIFAR10 (Krizhevsky et al., 2009). We initialize the standard deviations of $q(\mathcal{Z})$ for the low-entropy model using the half-normal $\mathcal{N}^{+}(0.16,0.02)$, while we use $\mathcal{N}^{+}(0.32,0.02)$ for the high-entropy model. For brevity, we refer to the former model as $\mathcal{M}_{16}$ and the latter model as $\mathcal{M}_{32}$. In the left plot of Fig. 3, we show that, after training, $\mathcal{M}_{32}$ retains higher variational posterior entropy than $\mathcal{M}_{16}$ due to having higher initial standard deviations for $q(\mathcal{Z}) .^{2}$ We use principal component analysis (PCA) to visualize the samples from the output distribution $p\left(\mathbf{f}_{\mathcal{Z}}^{\ell}(\mathbf{x})\right)$ of the $\ell$-th layer with respect to one input image $\mathbf{x}$, as well as the output $\left\{\hat{\mathbf{f}}^{\ell}\left(\mathbf{x}+\mathbf{g}_{i}(\mathbf{x})\right)\right\}_{i=1}^{95}$ under the real image corruptions $\left\{\mathbf{g}_{i}\right\}_{i=1}^{95}$ from Hendrycks \& Dietterich (2019). There are 19 corruption types with 5 levels of severity, totalling 95 corruption functions. Fig. 2 shows the activations of the last layer, projected into a two-dimensional subspace with PCA for visualization. From this figure, we can see that there is more overlap between samples from $p\left(\mathbf{f}_{\mathcal{Z}}^{\ell}(\mathbf{x})\right)$ and the shifted outputs $\left\{\hat{\mathbf{f}}^{\ell}\left(\mathbf{x}+\mathbf{g}_{i}(\mathbf{x})\right)\right\}_{i=1}^{95}$ for $\mathcal{M}_{32}$ in Fig. 2b than for $\mathcal{M}_{16}$ in Fig. 2a. This indicates that during training the posterior of $\mathcal{M}_{32}$ is able to simulate a larger number of implicit corruptions bearing resemblance to the real-world corruptions than the posterior of $\mathcal{M}_{16}$, leading to better neg-

[^0]
[^0]: ${ }^{2}$ Obtaining high-entropy models by starting with high-entropy initializations is a simple heuristic for the purpose of this example. We introduce a principled approach in Section 4.

---

#### Page 4

> **Image description.** The image contains four scatter plots arranged in a 2x2 grid. Each plot displays data points with varying colors, grey circles, a red ellipse, and a red circle. The plots are labeled (a) M16 and (b) M32 at the bottom.
>
> - **Overall Structure:** The image is divided into two columns and two rows. The top row contains two scatter plots, and the bottom row contains two scatter plots. The left column is labeled (a) M16 and the right column is labeled (b) M32.
>
> - **Scatter Plots:** Each scatter plot has an x-axis and a y-axis. The x-axis is labeled "Component 1" for the top plots and "Component 3" for the bottom plots. The y-axis is labeled "Component 2" for the top plots and "Component 4" for the bottom plots. The percentage of variance explained by each component is indicated on the axis labels (e.g., "Component 1 - 18.25%").
>
> - **Data Points:**
>
>   - Grey Circles: Each plot contains a large number of grey circles, which appear to be clustered around the center.
>   - Colored Circles: Each plot contains colored circles. The colors and corresponding labels are listed in a legend in the upper left corner of each plot. The legends differ slightly between the plots. For example, in plot (a) M16, the legend includes "1 - 14/19", "2 - 09/19", "3 - 05/19", "4 - 04/19", and "5 - 01/19".
>   - Red Circle: A single red circle is present in the center of the cluster of grey circles in each plot.
>   - Red Ellipse: A red ellipse encompasses the majority of the grey circles and some of the colored circles in each plot.
>
> - **Additional Elements:**
>
>   - Top Left Plot: The top left plot (corresponding to (a) M16) includes a small inset image in the lower left corner. This image appears to be a blurry photograph of an object.
>
> - **Labels:**
>   - (a) M16: Located below the two plots in the left column.
>   - (b) M32: Located below the two plots in the right column.

Figure 2. PCA plots of the last layer's outputs of models (a) $\mathcal{M}_{16}$ and (b) $\mathcal{M}_{32}$ with respect to one sample from CIFAR-10 (included in the top left panel). Grey circles are samples from the output distribution induced by $q(\mathcal{Z})$, while the red ellipse shows their 99 percentile. The red circle denotes the expected output $\overline{\mathbf{f}}^{\prime}(\mathbf{x})=$ $\mathbb{E}_{q(\mathcal{Z})}\left[\overline{\mathbf{f}}_{\mathcal{Z}}^{\prime}(\mathbf{x})\right]$ of the test point. Other colored circles represents the expected output $\overline{\mathbf{f}}^{\prime}$ of the 19 corrupted versions of the test point under 5 levels of severity Hendrycks \& Dietterich (2019). Most of the mild corruptions reside inside the predictive posterior of both models (filled color circles). By contrast, only the higher-entropy $\mathcal{M}_{32}$ model encapsulates a large fraction of the severe corruptions - empirically demonstrating the intuition sketched in Fig. 1 and described in Section 3.1.

ative log-likelihood (NLL) accross all level of corruptions as well as on the clean test set in Fig. 3. This example supports our intuition that increasing the entropy of the latent variables $\mathcal{Z}$ allows them to simulate more diverse implicit corruptions, thereby boosting the model's robustness against a wider range of input corruptions.

Why latent variables at every layer? In principle, we could have introduced latent variables only to the first layer of the network, as the shift simulated in the first layer will propagate to subsequent layers. However, modern NNs contain asymmetric activation functions such as ReLU or Softplus, which can attenuate the signal of the shift in the later layers. Thus, the latent variables in every layer (after the first one) maintain the strength of the shift throughout the network during the forward pass. Moreover, by using latent variables at every layer - as opposed to only the first layer - we can simulate a more diverse set of input corruptions, since we can map each sample $\mathcal{Z}$ from $q(\mathcal{Z})$ to an input

> **Image description.** The image contains two line graphs side-by-side.
>
> The left graph plots "H[q(Z)]" on the y-axis and "Epoch" on the x-axis. The x-axis ranges from 0 to 90. The y-axis ranges from -400 to 200. Two lines are plotted: one dashed and light brown, labeled "M16," and another solid and dark blue, labeled "M32." The "M16" line has a negative slope and starts at approximately -450. The "M32" line also has a negative slope, but starts at approximately 250.
>
> The right graph plots "NLL ->" on the y-axis and "Corruption intensity" on the x-axis. The x-axis ranges from 0 to 5. The y-axis ranges from 0.0 to 2.0. Two lines are plotted: one dashed and light brown, labeled "M16," and another solid and dark blue, labeled "M32." Both lines have a positive slope. The "M16" line is slightly above the "M32" line for most of the graph, but they converge at the start.

Figure 3. (Left) Example evolution of $\mathbb{H}[q(\mathcal{Z})]$ during training, which shows that the variational entropy decreases over time. (Right) Performance of two models under different corruption levels (level 0 indicates no corruption). The model with higher entropy $\mathcal{M}_{32}$ performs better than the one with lower entropy $\mathcal{M}_{16}$ across all corruption levels. For each result in both plots, we report the mean and standard deviation over 25 runs. The error bars in the left plot are too small to be seen.

> **Image description.** The image contains six pixelated images arranged in a 2x3 grid. Each column of the grid depicts a pair of images, with the top image appearing to be a corrupted version of a photograph and the bottom image showing the corresponding noise pattern. The columns are labeled (a), (b), and (c) along the bottom edge.
>
> - **Column (a):** The top image is heavily pixelated and noisy, but appears to show a bird sitting on a structure against a blurry background. The noise is a mix of many colors. The bottom image is a field of random colored pixels, with no discernible pattern. The label below reads "(a) λ = 0.03".
> - **Column (b):** The top image is less noisy than in column (a), but still pixelated. The bird and the structure are more clearly visible. The noise is less colorful, with more gray and brown tones. The bottom image is a field of mostly gray pixels with some faint color variations. The label below reads "(b) λ = 0.10".
> - **Column (c):** The top image is the least noisy of the three, with the bird and structure being the most recognizable. The noise is predominantly gray. The bottom image is a field of uniform gray pixels with very little variation. The label below reads "(c) λ = 0.30".
>
> The images appear to be demonstrating how different levels of a parameter, lambda (λ), affect the visibility of the original image and the corresponding noise pattern. As lambda increases from 0.03 to 0.30, the noise decreases, and the original image becomes clearer.

Figure 4. Implicit corruptions generated from model $\mathcal{M}_{32}$ with respect to one image by minimizing the loss in Eq. (16) under varying $\lambda$. Top row are the resulting images from the corruptions below. We can see that $\lambda$ controls the severity of the generated corruptions.

corruption as shown in the following section.

### 3.2. Visualizing the implicit corruptions

Next, we show how to find the explicit image corruptions that correspond to the stochasticity of the predictive posterior. Let $\mathcal{Z}$ be a sample drawn from $q(\mathcal{Z})$. If we assume that $\mathcal{Z}$ corresponds to an input corruption $\mathbf{g}(\mathbf{x})$ :

$$
\mathbf{f}_{\mathcal{Z}}(\mathbf{x})=\overline{\mathbf{f}}(\mathbf{x}+\mathbf{g}(\mathbf{x}))
$$

then we can approximately solve for $\mathbf{g}(\mathbf{x})=\mathbf{x}^{c}-\mathbf{x}$ by finding $\mathbf{x}^{c}$ that minimizes

$$
\mathcal{L}\left(\mathbf{x}^{c}\right)=\frac{1}{2}\left\|\mathbf{f}_{\mathcal{Z}}(\mathbf{x})-\overline{\mathbf{f}}\left(\mathbf{x}^{c}\right)\right\|_{2}^{2}+\frac{\lambda}{2}\|\mathbf{g}(\mathbf{x})\|_{2}^{2}
$$

using gradient descent. The second term with a coefficent $\lambda \geq 0$ regularizes the norm of $\mathbf{g}(\mathbf{x})$. This approach is simi-

---

#### Page 5

> **Image description.** The image contains two line graphs, side-by-side, displaying the Negative Log-Likelihood (NLL) on the y-axis against lambda (λ) on the x-axis.
>
> **Left Graph:**
>
> - X-axis: Labeled "λ" with values 0.01, 0.03, 0.1, and 0.3.
> - Y-axis: Labeled "NLL →" with values 0.25, 0.30, and 0.35.
> - Two solid lines are present:
>   - An orange line with circular markers.
>   - A blue line with downward triangle markers.
> - Two dashed lines are present:
>   - An orange dashed line with circular markers.
>   - A blue dashed line with downward triangle markers.
> - Shaded regions around each solid line indicate standard deviation. The orange shaded region is lighter than the blue shaded region.
>
> **Right Graph:**
>
> - X-axis: Labeled "λ" with values 0.01, 0.03, 0.1, and 0.3.
> - Y-axis: Labeled "NLL →" with values 0.25, 0.30, and 0.35.
> - Two solid lines are present:
>   - An orange line with circular markers.
>   - A blue line with downward triangle markers.
> - Shaded regions around each line indicate standard deviation. The orange shaded region is lighter than the blue shaded region.

> **Image description.** The image contains two line graphs, labeled "Left" and "Right" (though the labels themselves are not visible in the cropped image). Both graphs depict the Negative Log-Likelihood (NLL) on the y-axis against a variable lambda (λ) on the x-axis.
>
> **Left Graph:**
>
> - The x-axis (λ) ranges from 0.01 to 0.5, with tick marks at approximately 0.01, 0.03, 0.1, and 0.5.
> - The y-axis (NLL) ranges from 0.3 to 2.7, with tick marks at 0.3, 0.9, 1.5, 2.1, and 2.7. An arrow points to the y-axis label, "NLL."
> - Two lines are plotted:
>   - A solid orange line with circular markers, labeled "M16" in the legend. This line starts at a high NLL value around 2.7 at λ = 0.01 and decreases as λ increases. A shaded orange region surrounds the line, indicating a standard deviation.
>   - A solid blue line with downward triangle markers, labeled "M32" in the legend. This line starts at a lower NLL value around 1.3 at λ = 0.01 and decreases as λ increases. A shaded blue region surrounds the line, indicating a standard deviation.
> - The legend is located at the top right of the graph and is labeled "Model".
>
> **Right Graph:**
>
> - The x-axis (λ) ranges from 0.01 to 0.5, with tick marks at approximately 0.01, 0.03, 0.1, and 0.5.
> - The y-axis (NLL) ranges from 0.3 to 2.7, with tick marks at 0.3, 0.9, 1.5, 2.1, and 2.7.
> - Two lines are plotted:
>   - A solid orange line with circular markers, labeled "M16" in the legend. This line starts at a high NLL value and decreases as λ increases. A shaded orange region surrounds the line, indicating a standard deviation.
>   - A solid blue line with downward triangle markers, labeled "M32" in the legend. This line starts at a lower NLL value and decreases as λ increases. A shaded blue region surrounds the line, indicating a standard deviation.
> - The legend is located at the top right of the graph and is labeled "Model".
>
> The overall visual impression is that both graphs show a decreasing trend of NLL as λ increases for both models, with M32 generally having a lower NLL than M16. The shaded regions indicate the variability in the results.

Figure 5. Negative log-likelihood (NLL) on 1024 test images of CIFAR-10 corrupted by the implicit corruptions generated by $\mathcal{M}_{16}$ and $\mathcal{M}_{32}$, whose intensities are controlled by $\lambda$ in Eq. (16). For each result, we report the mean and standard deviation over 10 runs. (Left) Each model is tested on the corruptions that it generated. Dashed lines are results on the clean images for reference. Each model is resistant to its own corruptions, as evidenced by the slight decrease in performance under different $\lambda$. (Right) Each model is tested on the corruptions produced by the other. The model with higher entropy $\mathcal{M}_{32}$ is more robust against the corruptions of the one with lower entropy $\mathcal{M}_{16}$ than the reverse, which further supports the notion that higher entropy provides better robustness against input corruptions.

lar to the method of finding adversarial examples of Goodfellow et al. (2014). Fig. 4 visualizes the corruptions generated by $\mathcal{M}_{32}$ on a test image of CIFAR10 under different $\lambda$. We can see that $\lambda$ controls the severity of the corruptions, with smaller $\lambda$ corresponding to higher severity.

Is a model robust against its own corruptions? We use both models $\mathcal{M}_{16}$ and $\mathcal{M}_{32}$ to generate corruptions on a subset of 1024 test images of CIFAR10. We generate 8 corruptions per test image. The left plot of Fig. 5 shows that each model is robust against its own implicit corruptions even when the corruption is severe, as evidenced by the small performance degradation under different $\lambda$. By comparing the right plot to the left plot, we can see that each model is less resistant to the corruptions generated by the other model than its own corruptions. Crucially, however, the performance of $\mathcal{M}_{32}$ under the corruptions generated by $\mathcal{M}_{16}$ is better than the reverse. This example thus suggests that while each model is resistant to its own corruptions, the model with higher entropy shows better robustness against the corruptions created by the other model. ${ }^{3}$

## 4. Maximizing variational entropy

The previous sections motivated the usage of variational posteriors with high entropy from the perspective of simulating a diverse set of input corruptions. In this section, we discuss a simple method to increase the variational entropy.

[^0]

### 4.1. The augmented ELBO

Our goal is to find posterior approximations that have high entropy. In the previous section, we considered a heuristic approach of initializing $q(\mathcal{Z})$ with high entropy (Fig. 3). However, if the initial entropy of $q(\mathcal{Z})$ is too high, training will converge slowly due to high variance in the gradients.
Here we consider the approach of augmenting the original ELBO in Eq. (9) with an extra $\gamma$-weighted entropy term, adapting Mandt et al. (2016). The augmented $\gamma$-ELBO is

$$
\begin{aligned}
\mathcal{L}_{\gamma}(\hat{\theta}, \phi)= & \mathcal{L}(\hat{\theta}, \phi)+\gamma \mathbb{H}\left[q_{\phi}(\mathcal{Z})\right] \\
= & \underbrace{\mathbb{E}_{q_{\phi}(\mathcal{Z})}_{\text {expected log-likelihood }}[\log p(\mathcal{D} \mid \hat{\theta}, \mathcal{Z})]}_{\text {expe }}-\underbrace{\mathbb{H}\left[q_{\phi}(\mathcal{Z}), p(\mathcal{Z})\right]}_{\text {cross-entropy }} \\
& +\underbrace{(\gamma+1) \mathbb{H}\left[q_{\phi}(\mathcal{Z})\right]}_{\text {variational entropy }}+\underbrace{\log p(\hat{\theta})}_{\text {weight prior }}
\end{aligned}
$$

where we decompose the KL into its cross-entropy and entropy terms. $\gamma \geq 0$ controls the amount of extra entropy, with $\gamma=0$ reducing to the classic ELBO in Eq. (9). We can interpret the terms in Eq. (19) as follows: the first term fits the variational parameters to the dataset; the second and fourth terms regularize $\phi$ and $\hat{\theta}$ respectively; the third term increases the entropy of the variational posterior.

### 4.2. Tempered posterior inference

One could also arrive at Eq. (19) by minimizing the KL divergence between the approximate posterior $q_{\phi}(\hat{\theta}, \mathcal{Z})$ and the tempered posterior $p_{\gamma}(\theta, \mathcal{Z} \mid \mathcal{D})$ (Mandt et al., 2016):

$$
\begin{aligned}
p_{\gamma}(\theta, \mathcal{Z} \mid \mathcal{D}) & =\frac{p(\mathcal{D} \mid \theta, \mathcal{Z})^{\tau} p(\mathcal{Z}, \theta)^{\tau}}{p_{\gamma}(\mathcal{D})} \\
p_{\gamma}(\mathcal{D}) & =\int_{\theta} \int_{\mathcal{Z}} p(\mathcal{D} \mid \theta, \mathcal{Z})^{\tau} p(\mathcal{Z}, \theta)^{\tau} d \mathcal{Z} d \theta
\end{aligned}
$$

where the temperature $\tau=1 /(\gamma+1)$. The tempered posterior variational approximation

$$
\underset{\hat{\theta}, \phi}{\arg \min } \frac{1}{\tau} \mathrm{KL}\left[q_{\phi}(\hat{\theta}, \mathcal{Z}) \| p_{\gamma}(\theta, \mathcal{Z} \mid \mathcal{D})\right]
$$

is equivalent to tempered ELBO maximization

$$
\underset{\hat{\theta}, \phi}{\arg \max } \mathcal{L}_{\gamma}(\hat{\theta}, \phi)-\log p_{\gamma}(\mathcal{D})^{\frac{1}{\tau}}
$$

We refer the reader to Appendix B for detailed derivations. The entropy-regularized $\gamma$-ELBO thus corresponds to the family of tempered variational inference, and with positive $\gamma>0$, to 'hot' posteriors (Wenzel et al., 2020). In the next section, we will demonstrate empirically the benefits of such hot posteriors in node-based BNNs.

[^0]: ${ }^{3}$ We note that this a proof of concept and more experiments are needed to verify if these results hold true in general.

---

#### Page 6

> **Image description.** The image consists of four line graphs arranged in a 2x2 grid. Each graph plots the relationship between "γ/K" on the x-axis and "NLL" (Negative Log-Likelihood) on the y-axis. The y-axis label includes a downward-pointing arrow, indicating that lower NLL values are better.
>
> Each graph contains three lines, each representing a different value of "K":
>
> - A solid blue line for K=1
> - A dashed orange line for K=2
> - A dash-dotted grey line for K=4
>
> A legend in the top right graph (panel b) clarifies the line styles. The x-axis ranges from 0 to 80 in all four graphs. The y-axis ranges vary across the graphs.
>
> - Panel (a), labeled "Validation", has a y-axis ranging from approximately 0.85 to 1.2.
> - Panel (b), labeled "Test", has a y-axis ranging from approximately 0.85 to 1.2.
> - Panel (c), labeled "Corruption 1, 2, 3", has a y-axis ranging from approximately 1.6 to 2.2.
> - Panel (d), labeled "Corruption 4, 5", has a y-axis ranging from approximately 2.5 to 3.7.
>
> The lines in each graph generally show a U-shaped curve, indicating a minimum NLL value at some intermediate value of "γ/K". The position and depth of this minimum vary depending on the value of K and the specific graph (validation, test, or corruption level).

Figure 6. Results of (VGG16 / CIFAR 100 / out) with different $K$. The results in (c) are averaged over the first three levels of corruption, and those in (d) are averaged over the last two levels. Notice that we rescale $\gamma$ by $K$ in the x -axis to provide better visualization, as we find that larger $K$ requires higher optimal $\gamma$. We report the mean and standard deviation over 5 runs for each result. Overall, more components provide better optimal performance on OOD data. Higher $\gamma$ provides better OOD performance as the cost of ID performance.

## 5. Experiments

In this section, we present experimental results of nodebased BNNs on image classification tasks. For the datasets, we use CIFAR (Krizhevsky et al., 2009) and TINYIMAGENET (Le \& Yang, 2015), which have corrupted versions of the test set provided by Hendrycks \& Dietterich (2019). We use VGG16 (Simonyan \& Zisserman, 2014), RESNET18 (He et al., 2016a) and PRACTRESNET18 (He et al., 2016b) for the architectures. We test three structures of latent variables: in, where we only use the input latent variables $\left\{\mathbf{z}^{t}\right\}_{t=1}^{L}$; out, where we only use the output latent variables $\left\{\mathbf{s}^{t}\right\}_{t=1}^{L}$; and both, where we use both $\left\{\mathbf{z}^{t}\right\}_{t=1}^{L}$ and $\left\{\mathbf{s}^{t}\right\}_{t=1}^{L}$. We use $K \in\{1,2,4\}$ Gaussian component(s) in the variational posterior. For each result, we report the mean and standard deviation over multiple runs.

### 5.1. Effects of $\gamma$ on covariate shift

In this section, we study the changes in performance of the model trained with the $\gamma$-ELBO objective as we increase $\gamma$. We perform experiments with VGG16 on CIFAR100, and use the corrupted test set of CIFAR100 provided by Hendrycks \& Dietterich (2019). In Fig. 6, we show the out model's behaviour under a different number of Gaussian components $K$. In Fig. 7, we show the results of a model with $K=4$ components under the different latent variable

> **Image description.** The image contains four line graphs arranged in a 2x2 grid. Each graph plots "NLL" (Negative Log-Likelihood) on the y-axis against "γ/K" on the x-axis. The y-axis label "NLL" has an upward-pointing arrow next to it in the first two graphs, indicating that lower values are better. Each graph contains three lines representing different model configurations: "in" (solid blue line), "out" (dashed orange line), and "both" (dash-dotted gray line).
>
> - **Panel (a) Validation:** The x-axis ranges from 0 to 40. The y-axis ranges from approximately 0.9 to 1.1. The "in" and "out" lines are very close, forming a U-shaped curve with a minimum around x=20. The "both" line has a more erratic shape, initially decreasing rapidly, then increasing.
>
> - **Panel (b) Test:** The x-axis ranges from 0 to 40. The y-axis ranges from approximately 0.9 to 1.1. The "in" and "out" lines are very close, forming a U-shaped curve with a minimum around x=20. The "both" line has a more erratic shape, initially decreasing rapidly, then increasing.
>
> - **Panel (c) Corruption 1, 2, 3:** The x-axis ranges from 0 to 40. The y-axis ranges from approximately 1.6 to 2.0. The "in" and "out" lines are U-shaped. The "both" line has a more erratic shape.
>
> - **Panel (d) Corruption 4, 5:** The x-axis ranges from 0 to 40. The y-axis ranges from approximately 2.5 to 3.5. The "in" and "out" lines are decreasing curves. The "both" line decreases more rapidly, then flattens. A legend in the top right corner of this panel identifies the line styles for "in", "out", and "both".
>
> Each panel has a title below it: "(a) Validation", "(b) Test", "(c) Corruption 1, 2, 3", and "(d) Corruption 4, 5".

Figure 7. Results of VGG16 on CIFAR100 with different latent variable structures. Here we use $K=4$ components. We report the mean and standard deviation over 5 runs for each result. Overall, using either only the latent input variables or latent output variables requires higher optimal $\gamma$ than using both. Using only the latent output variables produces better results than the latent input variables on OOD data, despite similar ID performance.

structures in, out, and both.
These figures show that performance across different test sets improves as $\gamma$ increases up until a threshold and then degrades afterward. The optimal $\gamma$ for each set of test images correlates with the severity of the corruptions, where more severe corruptions can be handled by enforcing more diverse set of implicit corruptions during training. However, learning on a more diverse implicit corruptions requires higher capacity, and reduces the learning capacity needed to obtain good performance on the in-distribution (ID) data. The entropy coefficient $\gamma$ thus controls the induced tradeoff between ID performance and out-of-distribution (OOD) robustness.

Fig. 6 shows that for ID data, the optimal performance of the model (at optimal $\gamma$ ) remains similar under different $K$. On OOD data, however, higher $K$ consistently produces better results as $\gamma$ varies. The optimal $\gamma$ is higher for variational distributions with more components. This finding is likely because with more mixture components, the variational posterior can approximate the true posterior more accurately, and thus it can better expand into the high-density region of the true posterior as its entropy increases.

Fig. 7 shows the optimal performance on ID data is quite similar between different latent architectures. On OOD, the optimal performance of using both input and output latent variables is similar to using only output latent variables,

---

#### Page 7

> **Image description.** The image contains a single panel showing a line graph.
>
> - **Overview:** The graph depicts the relationship between "Avg NLL (clean label)" and "γ" (gamma). There are two lines plotted on the graph, one solid blue and one dashed orange, each with its own y-axis scale.
>
> - **Axes:**
>
>   - The x-axis is labeled "γ" and ranges from 0 to 160.
>   - The left y-axis is labeled "Avg NLL (clean label)" and ranges from 0.1 to 0.4.
>   - The right y-axis ranges from 0.2 to 0.6.
>
> - **Data:**
>
>   - The solid blue line starts at approximately (0, 0.1), dips slightly, and then increases sharply as γ increases beyond approximately 80.
>   - The dashed orange line starts at approximately (0, 0.1), increases to a peak around γ = 80, and then decreases.
>
> - **Text:**
>   _ "(a) 20%" appears below the graph.
>   _ The numbers "1" and "2" appear to the right of the graph, presumably indicating which y-axis scale corresponds to which line.
>   (a) $20 \%$

> **Image description.** The image contains a single line graph.
>
> The graph has two y-axes. The left y-axis ranges from 2 to 4, with tick marks at each integer value. The right y-axis is not explicitly labeled with values. The x-axis is labeled "$\gamma$" and ranges from 0 to 160, with tick marks at 80 and 160.
>
> There are two lines plotted on the graph. One line is solid blue, starting at approximately y=2 on the left and increasing to approximately y=4 on the right. The other line is dashed orange, starting at approximately y=2 on the left, increasing to a peak at approximately y=4 in the middle, and then decreasing to approximately y=2 on the right.
>
> Below the graph is the text "(b) $40 \%$".
> (b) $40 \%$

> **Image description.** The image is a line graph showing the relationship between "γ" (gamma) on the x-axis and "Avg NLL (noisy label)" on the y-axis. The x-axis ranges from 0 to 160, and the y-axis ranges from 1 to 4. There are two lines on the graph: a solid blue line and a dashed orange line. Both lines generally increase as "γ" increases. The title below the graph is "(c) 80%".
> (c) $80 \%$

Figure 8. Results of RESNET18 on two subsets of CIFAR10 training samples with clean and noisy labels. Here we use $K=4$ components and only the latent output variables. We denote the percentage of training samples with corrupted labels under each subfigure. We report the mean and standard deviation over 5 runs for each result. As $\gamma$ increases, the NLL of noisy labels increases much faster than that of clean labels even when the majority of labels are wrong (c), indicating that higher $\gamma$ prevents the model from memorizing random labels.

> **Image description.** This image contains three line graphs arranged horizontally, each representing data related to "Test NLL" (Negative Log-Likelihood) versus "γ/K". The graphs share a similar structure but display different data patterns.
>
> - **General Layout:** Each graph is enclosed in a white box with light gray grid lines. The x-axis ranges from 0 to 40, labeled as "γ/K". The y-axis represents "Test NLL" and has varying scales depending on the graph.
>
> - **Graph (a) 20%:** The y-axis ranges from 0.5 to 0.6. Two lines are plotted: a dashed blue line labeled "in" and a solid orange line labeled "out". Both lines initially decrease, reaching a minimum around x=15-20, then increase sharply. The orange line appears to increase more rapidly than the blue line.
>
> - **Graph (b) 40%:** The y-axis ranges from 0.8 to 1.0. Similar to graph (a), there's a dashed blue line ("in") and a solid orange line ("out"). The lines exhibit a similar trend of decreasing to a minimum and then increasing, with the orange line increasing more rapidly.
>
> - **Graph (c) 80%:** The y-axis ranges from 2.0 to 3.0. Again, a dashed blue line ("in") and a solid orange line ("out") are present. The orange line starts at a higher value and decreases sharply before increasing slightly. The blue line decreases to a minimum and then increases slightly. There is a shaded light blue area around the blue line, indicating the standard deviation.
>
> - **Labels:** Below each graph, there's a label indicating a percentage: "(a) 20%", "(b) 40%", and "(c) 80%". These percentages likely represent the percentage of noisy labels.
>
> - **Legend:** A legend is present in the top right corner of the third graph (c), indicating that the dashed blue line represents "in" and the solid orange line represents "out".

Figure 9. Results of RESNET18 on clean CIFAR10 test sets under different percentages of noise in training labels. We report the mean and standard deviation over 5 runs for each result. As high $\gamma$ prevents learning from noisy labels as demonstrated in Fig. 8, it leads to improved performance on clean test sets.

while using only input latent variables produces slightly worse optimal performance. The optimal $\gamma$ is lower when the model uses both types of latent variables ( $\mathbf{z}, \mathbf{s}$ ), because the entropy of the product of two latent variables increases rapidly as we increase the entropy of both latent variables.

We also observe these patterns in other architectures and datasets (see Appendix I). In summary, from our experimental results we find that using only output latent variables with a sufficient number of components (e.g., $K=4$ ) achieves excellent results for node-based BNNs in our benchmark.

### 5.2. Effects of $\gamma$ on robustness against noisy labels

Learning wrong labels amounts to memorizing random patterns, which requires more capacity from the model than learning generalizable patterns (Arpit et al., 2017). We hypothesize that if we corrupt wrongly labelled training samples with sufficiently diverse implicit corruptions, we overwhelm the neural network making it unable to mem-
orize these spurious patterns during training. To test this intuition, we follow the experiment in Jiang et al. (2018), where we take a percentage of training samples in CIFAR10 and corrupt their labels. We thus split the training set into two parts: $\mathcal{D}_{1}$ containing only samples with correct labels, and $\mathcal{D}_{2}$ including those with wrong labels. We then track the final NLL of $\mathcal{D}_{1}$ and $\mathcal{D}_{2}$ under different $\gamma$, and visualize the results in Fig. 8. This figure shows that as $\gamma$ increases, the NLL of $\mathcal{D}_{2}$ (noisy labels) increases much faster than that of $\mathcal{D}_{1}$ (clean labels), indicating that the network fails to learn random patterns under simulated corruptions. As a consequence, the model generalizes better on the test set, as shown in Fig. 9.

### 5.3. Benchmark results

Figs. 10 and 11 present the results of node-based BNNs and baselines on CIFAR10/CIFAR100 and TINYIMAGENET. We choose SWAG (Maddox et al., 2019), cSG-HMC (Zhang et al., 2020) and ASAM (Kwon et al., 2021) as our baselines. These are strong baselines, as both SWAG and cSG-HMC have demonstrated state-of-the-art uncertainty estimation, while ASAM produce better MAP models than stochastic gradient descent by actively seeking wide loss valleys. We repeated each experiment 25 times with different random seeds. For each method, we also consider its ensemble version where we combine 5 models from different runs when making predictions. For the ensemble versions, each experiment is repeated 5 times. We use 30 Monte Carlo samples for node-based BNNs, SWAG, cSG-HMC and their ensemble versions to estimate the posterior predictive distribution. We use standard performance metrics of expected calibration error (ECE) (Naeini et al., 2015), NLL and predictive error. We use RESNET18 for CIFAR10/CIFAR100 and PREACTRESNET18 for TINYIMAGENET. We also include the result of VGG16 on CIFAR10/CIFAR100 in Appendix G. For evaluation, we use the corrupted test images provided by Hendrycks \& Dietterich (2019).

On CIFAR100, node-based BNNs outperform the baselines in NLL and error, however SWAG performs best on CIFAR10. Interestingly, in CIFAR100, node-based BNNs and their ensembles have worse ECE than the baselines on ID data, however as the test images become increasingly corrupted, the ECEs of the baselines degrade rapidly while the ECE of node-based BNNs remains below a threshold. Similar behaviors are observed on TINYIMAGENET, with the node-based BNNs produce the lowest NLL and error while not experiencing ECE degradation under corruptions.

## 6. Related works

Multiplicative latent node variables in BNNs. There have been several earlier works that utilize multiplicative latent node variables, either as a primary source of pre-

---

#### Page 8

> **Image description.** The image consists of six plots arranged in a 2x3 grid. Each plot displays data related to different corruption levels, ranging from 0 to 5, on the x-axis. The plots are organized into two rows, with the top row likely representing results on one dataset (CIFAR10) and the bottom row on another (CIFAR100). Within each row, the plots show different metrics: ECE (Expected Calibration Error), NLL (Negative Log-Likelihood), and Error (%). Each metric is indicated by a downward arrow next to its name, suggesting lower values are better.
>
> Each plot contains data points for several methods, distinguished by shape and color:
>
> - node-BNN (blue circle)
> - ens node-BNN (open blue circle)
> - SWAG (orange square)
> - ens SWAG (open orange square)
> - ASAM (gray diamond)
> - ens ASAM (open gray diamond)
> - cSg-HMC (yellow triangle pointing down)
> - ens cSG-HMC (open yellow triangle pointing down)
>
> The data points are plotted against the corruption level, and some data points have error bars indicating standard deviation. The plots show a general trend of increasing ECE, NLL, and Error as the corruption level increases. The node-based BNNs and their ensembles (blue) appear to perform better across all metrics on the bottom row (likely CIFAR100), while having competitive results on the top row (likely CIFAR10).
>
> The x-axis is labeled "Corruption level" and ranges from 0 to 5. The y-axes vary depending on the metric being plotted. The top row has y-axis ranges of approximately 0 to 0.2 for ECE, 0 to 1.5 for NLL, and 0 to 40 for Error (%). The bottom row has y-axis ranges of approximately 0 to 0.2 for ECE, 0 to 3.0 for NLL, and 20 to 60 for Error (%).

Figure 10. Results of RESNET18 on CIFAR10 (top) and CIFAR100 (bottom). We use $K=4$ and only the latent output variables for node-based BNNs. We plot ECE, NLL and error for different corruption levels, where level 0 indicates no corruption. We report the average performance over 19 corruption types for level 1 to 5 . We denote the ensemble of a method using the shorthand ens in front of the name. Each result is the average over 25 runs for nonens versions and 5 runs for ens versions. The error bars represent the standard deviations across different runs. Node-based BNNs and their ensembles (blue) perform best across all metrics on OOD data of CIFAR100, while having competitive results on CIFAR10. We include a larger version of this plot in Appendix G.

dictive uncertainty such as MC-Dropout (Gal \& Ghahramani, 2016), Variational Dropout (Kingma et al., 2015), Rank-1 BNNs (Dusenberry et al., 2020) and Structured Dropout (Nguyen et al., 2021); or to improve the flexibility of the mean-field Gaussian posterior in variational inference (Louizos \& Welling, 2017). Here we study the contribution of these latent variables to robustness under covariate shift.

BNNs under covariate shift. Previous works have evaluated the predictive uncertainty of BNNs under covariate shift (Ovadia et al., 2019; Izmailov et al., 2021b), with the recent work by Izmailov et al. (2021b) showing that standard BNNs with high-fidelity posteriors perform worse than MAP solutions under covariate shift. Izmailov et al.

> **Image description.** The image consists of three scatter plots, each displaying the performance of different machine learning models under varying levels of data corruption. Each plot shares a common horizontal axis labeled "Corruption level" ranging from 0 to 5. The plots are arranged horizontally.
>
> - **Plot 1 (ECE ↓):** The vertical axis is labeled "ECE ↓" and ranges from 0.0 to 0.3. Data points are scattered across the plot, representing different models: "node-BNN" (blue circles), "ens node-BNN" (hollow blue circles), "SWAG" (orange squares), "ens SWAG" (hollow orange squares), "ASAM" (grey diamonds), "ens ASAM" (hollow grey diamonds), "cSGHMC" (yellow triangles), and "ens cSGHMC" (hollow yellow triangles). Error bars are visible for some data points, particularly for the "SWAG" and "ens SWAG" models. The plot generally shows an increasing trend in ECE with increasing corruption level.
>
> - **Plot 2 (NLL ↓):** The vertical axis is labeled "NLL ↓" and ranges from 2.0 to 6.0. Similar to the first plot, data points represent the same models, and their arrangement shows an increasing trend in NLL with increasing corruption level.
>
> - **Plot 3 (Error (%) ↓):** The vertical axis is labeled "Error (%) ↓" and ranges from 30 to 80. Again, the same models are represented, and the plot shows an increasing trend in error percentage with increasing corruption level.
>
> A legend is provided below the three plots, mapping the colors and shapes to the corresponding model names. The legend entries are:
>
> - "node-BNN" (blue circle)
> - "ens node-BNN" (hollow blue circle)
> - "SWAG" (orange square)
> - "ens SWAG" (hollow orange square)
> - "ASAM" (grey diamond)
> - "ens ASAM" (hollow grey diamond)
> - "cSGHMC" (yellow triangle)
> - "ens cSGHMC" (hollow yellow triangle)

Figure 11. Results of PRACTRESNET18 on TINYIMAGENET. We use $K=4$ and only the latent output variables for node-based BNNs. We plot ECE, NLL and error for different corruption levels, where level 0 indicates no corruption. We report the average performance over 19 corruption types for level 1 to 5 . We denote the ensemble of a method using the shorthand ens in front of the name. Each result is the average over 25 runs for non-ens versions and 5 runs for ens versions. The error bars represent the standard deviations across different runs. Node-based BNNs and their ensembles (blue) perform best accross all metrics on OOD data, while having competitive performance on ID data. We include a larger version of this plot in Appendix G.

(2021a) attributed this phenomenon to the absence of posterior contraction on the null-space of the data manifold. This problem is avoided in node-based BNNs as they still maintain a point-estimate for the weights.

Dropout as data augmentation. Similar to our study, a previous work by Bouthillier et al. (2015) studied Dropout from the data augmentation perspective. Here we study latent variables with more flexible posterior (mixture of Gaussians) and focus on simulating input corruptions for OOD robustness.

Adversarial robustness via feature perturbations. Data-space perturbations have been investigated as a means to defend neural networks against adversarial attacks ( Li et al., 2018; Jeddi et al., 2020; Vadera et al., 2020).

Tempered posteriors. Tempered posteriors have been used in variational inference to obtain better variational posterior approximations (Mandt et al., 2016). A recent study put the focus on the cold posterior effect of weightbased BNNs (Wenzel et al., 2020). We have shown that our approach of regularizing the variational entropy is equivalent to performing variational inference with a hot posterior

---

#### Page 9

as the target distribution. Tempered posteriors have also been studied in Bayesian statistics as a means to defend against model misspecification (Grünwald, 2012; Miller \& Dunson, 2019; Alquier \& Ridgway, 2020; Medina et al., 2021). Covariate shift is a form of model misspecification, as model mismatch arises from using a model trained under different assumptions about the statistics of the data.

## 7. Conclusion

We analyzed node-based BNNs from the perspective of using latent node variables for simulating input corruptions. We showed that by regularizing the entropy of the latent variables, we increase the diversity of the implicit corruptions, and thus improve performance of node-based BNNs under covariate shift. Across CIFAR10, CIFAR100 and TINYIMAGENET, entropy regularized node-based BNNs produce excellent results in uncertainty metrics on OOD data.

In this study, we focused on variational inference, leaving the study of implicit corruptions under other approximate inference methods as future work. Furthermore, our work shows the benefits of hot posteriors and argues for an inherent trade-off between ID and OOD performance in nodebased BNNs. It is an interesting future direction to study these questions in weight-based BNNs. Finally, our work presented entropy as a surprisingly useful summary statistic that can partially explain the complex connection between the variational posterior and corruption robustness. One important research direction is to develop more informative statistics that can better encapsulate this connection.

---

# Tackling covariate shift with node-based Bayesian neural networks - Backmatter

---

## Acknowledgement

This work was supported by the Academy of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI and grants no. 292334, 294238, 319264, 328400) and UKRI Turing AI World-Leading Researcher Fellowship, EP/W002973/1. We acknowledge the computational resources provided by Aalto Science-IT project and CSC-IT Center for Science, Finland.

## References

Alquier, P. and Ridgway, J. Concentration of tempered posteriors and of their variational approximations. The Annals of Statistics, 48(3):1475-1497, 2020.

Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., Maharaj, T., Fischer, A., Courville, A., Bengio, Y., and Lacoste-Julien, S. A closer look at memorization in deep networks. In ICML, pp. 233-242. PMLR, 2017.

Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Varia-
tional inference: A review for statisticians. Journal of the American statistical Association, 112(518):859-877, 2017.

Bouthillier, X., Konda, K., Vincent, P., and Memisevic, R. Dropout as data augmentation. arXiv preprint arXiv:1506.08700, 2015.

Dusenberry, M., Jerfel, G., Wen, Y., Ma, Y., Snoek, J., Heller, K., Lakshminarayanan, B., and Tran, D. Efficient and scalable Bayesian neural nets with rank-1 factors. In ICML, pp. 2782-2792, 2020.

Gal, Y. and Ghahramani, Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In ICML, 2016.

Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

Grünwald, P. The safe bayesian. In International Conference on Algorithmic Learning Theory, pp. 169-183. Springer, 2012.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016a.

He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In European conference on computer vision, pp. 630-645. Springer, 2016b.

Hendrycks, D. and Dietterich, T. Benchmarking neural network robustness to common corruptions and perturbations. Proceedings of the International Conference on Learning Representations, 2019.

Hinton, G. E. and van Camp, D. Keeping the neural networks simple by minimizing the description length of the weights. In COLT, pp. 5-13, 1993.

Izmailov, P., Maddox, W. J., Kirichenko, P., Garipov, T., Vetrov, D., and Wilson, A. G. Subspace inference for Bayesian deep learning. In UAI, pp. 1169-1179, 2020.

Izmailov, P., Nicholson, P., Lotfi, S., and Wilson, A. G. Dangers of bayesian model averaging under covariate shift. arXiv preprint arXiv:2106.11905, 2021a.

Izmailov, P., Vikram, S., Hoffman, M. D., and Wilson, A. G. What are bayesian neural network posteriors really like? arXiv preprint arXiv:2104.14421, 2021b.

Jebara, T. and Kondor, R. Bhattacharyya and expected likelihood kernels. In Learning theory and kernel machines, pp. 57-71. Springer, 2003.

---

#### Page 10

Jebara, T., Kondor, R., and Howard, A. Probability product kernels. The Journal of Machine Learning Research, 5: 819-844, 2004.

Jeddi, A., Shafiee, M. J., Karg, M., Scharfenberger, C., and Wong, A. Learn2perturb: an end-to-end feature perturbation learning to improve adversarial robustness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1241-1250, 2020.

Jiang, L., Zhou, Z., Leung, T., Li, L.-J., and Fei-Fei, L. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018.

Karaletsos, T. and Bui, T. D. Hierarchical gaussian process priors for bayesian neural network weights. arXiv preprint arXiv:2002.04033, 2020.

Karaletsos, T., Dayan, P., and Ghahramani, Z. Probabilistic meta-representations of neural networks. arXiv preprint arXiv:1810.00555, 2018.

Kingma, D. P., Salimans, T., and Welling, M. Variational dropout and the local reparameterization trick. In NIPS, pp. 2575-2583, 2015.

Kolchinsky, A. and Tracey, B. D. Estimating mixture entropy with pairwise distances. Entropy, 19(7), 2017. ISSN 1099-4300. doi: 10.3390/e19070361. URL https: //www.mdpi. com/1099-4300/19/7/361.

Krizhevsky, A., Nair, V., and Hinton, G. Cifar-10 and cifar100 datasets. URI: https://www. cs. toronto. edu/kriz/cifar. html, 6(1):1, 2009.

Kwon, J., Kim, J., Park, H., and Choi, I. K. Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. arXiv preprint arXiv:2102.11600, 2021.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. In NIPS, pp. 6405-6416, 2017.

Le, Y. and Yang, X. S. Tiny imagenet visual recognition challenge. 2015.

Li, B., Chen, C., Wang, W., and Carin, L. Certified adversarial robustness with additive noise. arXiv preprint arXiv:1809.03113, 2018.

Louizos, C. and Welling, M. Multiplicative normalizing flows for variational bayesian neural networks. In International Conference on Machine Learning, pp. 2218-2227. PMLR, 2017.

MacKay, D. J. C. A practical Bayesian framework for backpropagation networks. Neural Computation, 4(3): 448-472, May 1992. ISSN 0899-7667.

MacKay, D. J. C. Probable networks and plausible predictions - a review of practical Bayesian methods for supervised neural networks. Network: Computation in Neural Systems, 6(3):469-505, 1995.

Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., and Wilson, A. G. A simple baseline for bayesian uncertainty in deep learning. In Advances in Neural Information Processing Systems, pp. 13153-13164, 2019.

Mandt, S., McInerney, J., Abrol, F., Ranganath, R., and Blei, D. Variational tempering. In Artificial Intelligence and Statistics, pp. 704-712. PMLR, 2016.

Medina, M. A., Olea, J. L. M., Rush, C., and Velez, A. On the robustness to misspecification of $\alpha$-posteriors and their variational approximations. arXiv preprint arXiv:2104.08324, 2021.

Miller, J. W. and Dunson, D. B. Robust Bayesian Inference via Coarsening. Journal of the American Statistical Association, 114(527):1113-1125, July 2019. ISSN 0162-1459. doi: 10.1080/01621459.2018. 1469995. URL https://doi.org/10.1080/ 01621459.2018.1469995. Publisher: Taylor \& Francis.

Naeini, M. P., Cooper, G. F., and Hauskrecht, M. Obtaining well calibrated probabilities using Bayesian binning. In AAAI, 2015.

Neal, R. M. Bayesian Learning for Neural Networks. Lecture Notes in Statistics. Springer-Verlag, New York, 1996. ISBN 978-0-387-94724-2.

Nguyen, S., Nguyen, D., Nguyen, K., Than, K., Bui, H., and Ho, N. Structured dropout variational inference for bayesian neural networks. In NeurIPS, 2021.

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., and Snoek, J. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. arXiv preprint arXiv:1906.02530, 2019.

Pradier, M. F., Pan, W., Yao, J., Ghosh, S., and Doshi-Velez, F. Projected bnns: Avoiding weight-space pathologies by learning latent representations of neural network weights. arXiv preprint arXiv:1811.07006, 2018.

Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

Springenberg, J. T., Dosovitskiy, A., Brox, T., and Riedmiller, M. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.

---

#### Page 11

Sun, S., Zhang, G., Shi, J., and Grosse, R. Functional variational bayesian neural networks. arXiv preprint arXiv:1903.05779, 2019.

Trinh, T., Kaski, S., and Heinonen, M. Scalable bayesian neural networks by layer-wise input augmentation. arXiv preprint arXiv:2010.13498, 2020.

Vadera, M. P., Shukla, S. N., Jalaian, B., and Marlin, B. M. Assessing the adversarial robustness of monte carlo and distillation methods for deep bayesian neural network classification. arXiv preprint arXiv:2002.02842, 2020.

Wenzel, F., Roth, K., Veeling, B. S., Świątkowski, J., Tran, L., Mandt, S., Snoek, J., Salimans, T., Jenatton, R., and Nowozin, S. How good is the bayes posterior in deep neural networks really? arXiv preprint arXiv:2002.02405, 2020.

Zhang, R., Li, C., Zhang, J., Chen, C., and Wilson, A. G. Cyclical stochastic gradient mcmc for bayesian deep learning. International Conference on Learning Representations, 2020.

---

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