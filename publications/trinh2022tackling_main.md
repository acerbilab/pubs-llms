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
