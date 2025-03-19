```
@inproceedings{trinh2024improving,
    title={Improving robustness to corruptions with multiplicative weight perturbations},
    author={Trung Trinh and Markus Heinonen and Luigi Acerbi and Samuel Kaski},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)},
    year={2024},
    url={https://openreview.net/forum?id=M8dy0ZuSb1}
}
```

#### Page 1

# Improving robustness to corruptions with multiplicative weight perturbations

Trung Trinh ${ }^{1}$ Markus Heinonen ${ }^{1}$ Luigi Acerbi ${ }^{2}$ Samuel Kaski ${ }^{1,3}$<br>${ }^{1}$ Department of Computer Science, Aalto University, Finland<br>${ }^{2}$ Department of Computer Science, University of Helsinki, Finland<br>${ }^{3}$ Department of Computer Science, University of Manchester, United Kingdom<br>\{trung.trinh, markus.o.heinonen, samuel.kaski\}@aalto.fi,<br>luigi.acerbi@helsinki.fi

#### Abstract

Deep neural networks (DNNs) excel on clean images but struggle with corrupted ones. Incorporating specific corruptions into the data augmentation pipeline can improve robustness to those corruptions but may harm performance on clean images and other types of distortion. In this paper, we introduce an alternative approach that improves the robustness of DNNs to a wide range of corruptions without compromising accuracy on clean images. We first demonstrate that input perturbations can be mimicked by multiplicative perturbations in the weight space. Leveraging this, we propose Data Augmentation via Multiplicative Perturbation (DAMP), a training method that optimizes DNNs under random multiplicative weight perturbations. We also examine the recently proposed Adaptive Sharpness-Aware Minimization (ASAM) and show that it optimizes DNNs under adversarial multiplicative weight perturbations. Experiments on image classification datasets (CIFAR-10/100, TinyImageNet and ImageNet) and neural network architectures (ResNet50, ViT-S/16, ViT-B/16) show that DAMP enhances model generalization performance in the presence of corruptions across different settings. Notably, DAMP is able to train a ViT-S/16 on ImageNet from scratch, reaching the top-1 error of $23.7 \%$ which is comparable to ResNet50 without extensive data augmentations. ${ }^{1}$

## 1 Introduction

Deep neural networks (DNNs) demonstrate impressive accuracy in computer vision tasks when evaluated on carefully curated and clean datasets. However, their performance significantly declines when test images are affected by natural distortions such as camera noise, changes in lighting and weather conditions, or image compression algorithms (Hendrycks and Dietterich, 2019). This drop in performance is problematic in production settings, where models inevitably encounter such perturbed inputs. Therefore, it is crucial to develop methods that produce reliable DNNs robust to common image corruptions, particularly for deployment in safety-critical systems (Amodei et al., 2016).

To enhance robustness against a specific corruption, one could simply include it in the data augmentation pipeline during training. However, this approach can diminish performance on clean images and reduce robustness to other types of corruptions (Geirhos et al., 2018). More advanced data augmentation techniques (Cubuk et al., 2018; Hendrycks et al., 2019; Lopes et al., 2019) have been developed which effectively enhance corruption robustness without compromising accuracy on clean images. Nonetheless, a recent study by Mintun et al. (2021) has identified a new set of image corruptions to which models trained with these techniques remain vulnerable. Besides data

[^0]
[^0]: ${ }^{1}$ Our code is available at https://github.com/trungtrinh44/DAMP

---

#### Page 2

> **Image description.** This image consists of three diagrams, labeled (a), (b), and (c), depicting a pre-activation neuron z in different scenarios. The diagrams are arranged horizontally.
>
> Panel (a) shows a neuron with inputs x1, x2, ..., xn, represented as blue circles. Each input has an associated covariate shift ε1, ε2, ..., εn, represented as red squares connected to the inputs by short lines. The shifted inputs (x1 + ε1, x2 + ε2, ..., xn + εn) are represented as circles with a gradient from blue to red. Each shifted input is connected to the neuron z (a white circle) by a line labeled with weights w1, w2, ..., wn. The panel is labeled " (a) z = w^T (x + ε)".
>
> Panel (b) shows a similar neuron structure. Inputs x1, x2, ..., xn (blue circles) are connected to the neuron z (white circle). The connecting lines are labeled with expressions w1(1 + ε1/x1), w2(1 + ε2/x2), ..., wn(1 + εn/xn) in red. The panel is labeled "(b) z = (w o (1 + ε/x))^T x".
>
> Panel (c) shows inputs x1, x2, ..., xn (blue circles) connected to the neuron z (white circle). The connecting lines are labeled with expressions w1ξ1, w2ξ2, ..., wnξn in teal. The panel is labeled "(c) z = (w o ξ)^T x, ξ ~ p(ξ)".
>
> In all three panels, the inputs are arranged vertically on the left, the neuron z is on the right, and the connections are represented by lines with arrowheads pointing towards z. The ellipsis (...) indicates that the pattern continues for n inputs.

Figure 1: Depictions of a pre-activation neuron $z=\mathbf{w}^{\top} \mathbf{x}$ in the presence of (a) covariate shift $\epsilon$, (b) a multiplicative weight perturbation (MWP) equivalent to $\epsilon$, and (c) random MWPs $\xi$, $\circ$ denotes the Hadamard product. Figs. (a) and (b) show that for a covariate shift $\epsilon$, one can always find an equivalent MWP. From this intuition, we propose to inject random MWPs $\xi$ to the forward pass during training as shown in Fig. (c) to robustify a DNN to covariate shift.
augmentation, ensemble methods such as Deep ensembles and Bayesian neural networks have also been shown to improve generalization in the presence of corruptions (Lakshminarayanan et al., 2017; Ovadia et al., 2019; Dusenberry et al., 2020; Trinh et al., 2022). However, the training and inference costs of these methods increase linearly with the number of ensemble members, rendering them less suitable for very large DNNs.

Contributions In this work, we show that simply perturbing weights with multiplicative random variables during training can significantly improve robustness to a wide range of corruptions. Our contributions are as follows:

- We show in Section 2 and Fig. 1 that the effects of input corruptions can be simulated during training via multiplicative weight perturbations.
- From this insight, we propose a new training algorithm called Data Augmentation via Multiplicative Perturbations (DAMP) which perturbs weights using multiplicative Gaussian random variables during training while having the same training cost as standard SGD.
- In Section 3, we show a connection between adversarial multiplicative weight perturbations and Adaptive Sharpness-Aware Minimization (ASAM) (Kwon et al., 2021).
- Through a rigorous empirical study in Section 4, we demonstrate that DAMP consistently improves generalization ability of DNNs under corruptions across different image classification datasets and model architectures.
- Notably, we demonstrate that DAMP can train a Vision Transformer (ViT) (Dosovitskiy et al., 2021) from scratch on ImageNet, achieving similar accuracy to a ResNet50 (He et al., 2016a) in 200 epochs with only basic Inception-style preprocessing (Szegedy et al., 2016). This is significant as ViT typically requires advanced training methods or sophisticated data augmentation to match ResNet50's performance when being trained on ImageNet from scratch (Chen et al., 2022; Beyer et al., 2022). We also show that DAMP can be combined with modern augmentation techniques such as MixUp (Zhang et al., 2018) and RandAugment (Cubuk et al., 2020) to further improve robustness of neural networks.

# 2 Data Augmentation via Multiplicative Perturbations

In this section, we demonstrate the equivalence between input corruptions and multiplicative weight perturbations (MWPs), as shown in Fig. 1, motivating the use of MWPs for data augmentation.

---

#### Page 3

> **Image description.** This is a diagram illustrating how corruption affects the output of a Deep Neural Network (DNN).
>
> The diagram is structured horizontally, showing a sequence of operations. It begins with an input 'x', represented inside a light blue circle. An arrow points from 'x' to a rectangular box labeled 'f^(1)'. The output of this box, 'x^(1)', is again represented inside a light blue circle. This pattern of light blue circles and rectangular boxes repeats several times, with the boxes labeled 'f^(2)', and eventually 'f^(H)'. Between the boxes are ellipses of dots, indicating that the pattern continues. The final light blue circle is labeled 'x^(H)'. An arrow points from 'x^(H)' to a rectangular box labeled 'ℓ(ω, x, y)'.
>
> Below this sequence, a parallel sequence begins with 'x_g', also in a light blue circle. 'x_g' is connected to 'x' by a dashed arrow labeled 'δ_g x^(0)'. The 'x_g' circle also points to the 'f^(1)' box. The pattern continues in parallel to the sequence above, with 'x_g^(1)' connected to 'x^(1)' by a dashed arrow labeled 'δ_g x^(1)', and so on, until 'x_g^(H)' is connected to 'x^(H)' by a dashed arrow labeled 'δ_g x^(H)'. Finally, 'x_g^(H)' points to a rectangular box labeled 'ℓ(ω, x_g, y)'. This box is connected to 'ℓ(ω, x, y)' by a dashed arrow labeled 'δ_g ℓ'.
>
> The diagram illustrates how an input 'x' is transformed through a series of functions 'f^(i)' to produce an output, and how a corrupted input 'x_g' similarly propagates through the network, ultimately affecting the loss function 'ℓ'.

Figure 2: Depiction of how a corruption g affects the output of a DNN. Here $\mathbf{x}_{\mathbf{g}}=\mathbf{g}(\mathbf{x})$. The corruption $\mathbf{g}$ creates a shift $\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(0)}=\mathbf{x}_{\mathbf{g}}-\mathbf{x}$ in the input $\mathbf{x}$, which propagates into shifts $\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}$ in the output of each layer. This will eventually cause a shift in the loss $\boldsymbol{\delta}_{\mathbf{g}} \ell$. This figure explains why the model performance tends to degrade under corruption.

# 2.1 Problem setting

Given a training data set $\mathcal{S}=\left\{\left(\mathbf{x}_{k}, y_{k}\right)\right\}_{k=1}^{N} \subseteq \mathcal{X} \times \mathcal{Y}$ drawn i.i.d. from the data distribution $\mathcal{D}$, we seek to learn a model that generalizes well on both clean and corrupted inputs. We denote $\mathcal{G}$ as a set of functions whose each member $\mathbf{g}: \mathcal{X} \rightarrow \mathcal{X}$ represents an input corruption. That is, for each $\mathbf{x} \in \mathcal{X}, \mathbf{g}(\mathbf{x})$ is a corrupted version of $\mathbf{x}$. ${ }^{2}$ We define $\mathbf{g}(\mathcal{S}):=\left\{\left(\mathbf{g}\left(\mathbf{x}_{k}\right), y_{k}\right)\right\}_{k=1}^{N}$ as the training set corrupted by $\mathbf{g}$. We consider a DNN $\mathbf{f}: \mathcal{X} \rightarrow \mathcal{Y}$ parameterized by $\boldsymbol{\omega} \in \mathcal{W}$. Given a per-sample loss $\ell: \mathcal{W} \times \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}_{+}$, the training loss is defined as the average loss over the samples $\mathcal{L}(\boldsymbol{\omega} ; \mathcal{S}):=\frac{1}{N} \sum_{k=1}^{N} \ell\left(\boldsymbol{\omega}, \mathbf{x}_{k}, y_{k}\right)$. Our goal is to find $\boldsymbol{\omega}$ which minimizes:

$$
\mathcal{L}(\boldsymbol{\omega} ; \mathcal{G}(\mathcal{S})):=\mathbb{E}_{\mathbf{g} \sim \mathcal{G}}[\mathcal{L}(\boldsymbol{\omega} ; \mathbf{g}(\mathcal{S}))]
$$

without knowing exactly the types of corruption contained in $\mathcal{G}$. This problem is crucial for the reliable deployment of DNNs, especially in safety-critical systems, since it is difficult to anticipate all potential types of corruption the model might encounter in production.

### 2.2 Multiplicative weight perturbations simulate input corruptions

To address the problem above, we make two key assumptions about the corruptions in $\mathcal{G}$ :
Assumption 1 (Bounded corruption). For each corruption function $\mathbf{g}: \mathcal{X} \rightarrow \mathcal{X}$ in $\mathcal{G}$, there exists a constant $M>0$ such that $\|\mathbf{g}(\mathbf{x})-\mathbf{x}\|_{2} \leq M$ for all $\mathbf{x} \in \mathcal{X}$.
Assumption 2 (Transferable robustness). A model's robustness to corruptions in $\mathcal{G}$ can be indirectly enhanced by improving its resilience to a more easily simulated set of input perturbations.

Assumption 1 implies that the corrupted versions of an input $\mathbf{x}$ must be constrained within a bounded neighborhood of $\mathbf{x}$ in the input space. Assumption 2 is corroborated by Rusak et al. (2020), who demonstrated that distorting training images with Gaussian noise improves a DNN's performance against various types of corruption. We further validate this observation for corruptions beyond Gaussian noise in Section 4.1. However, Section 4.1 also reveals that using corruptions as data augmentation degrades model performance on clean images. Consequently, we need to identify a method that efficiently simulates diverse input corruptions during training, thereby robustifying a DNN against a wide range of corruptions without compromising its performance on clean inputs.
One such method involves injecting random multiplicative weight perturbations (MWPs) into the forward pass of DNNs during training. The intuition behind this approach is illustrated in Fig. 1. Essentially, for a pre-activated neuron $z=\mathbf{w}^{\top} \mathbf{x}$ in a DNN, given a corruption causing a covariate shift $\boldsymbol{\epsilon}$ in the input $\mathbf{x}$, Figs. 1a and 1b show that one can always find an equivalent MWP $\boldsymbol{\xi}(\boldsymbol{\epsilon}, \mathbf{x})$ :

$$
z=\mathbf{w}^{\top}(\mathbf{x}+\boldsymbol{\epsilon})=(\mathbf{w} \circ \boldsymbol{\xi}(\boldsymbol{\epsilon}, \mathbf{x}))^{\top} \mathbf{x}, \quad \boldsymbol{\xi}(\boldsymbol{\epsilon}, \mathbf{x})=1+\boldsymbol{\epsilon} / \mathbf{x}
$$

where $\circ$ denotes the Hadamard product. This observation suggests that input corruptions can be simulated during training by injecting random MWPs into the forward pass, as depicted in Fig. 1c, resulting in a model more robust to corruption. We thus move the problem of simulating corruptions from the input space to the weight space.
Here we provide theoretical arguments supporting the usage of MWPs to robustify DNNs. To this end, we study how corruption affects training loss. We consider a feedforward neural network $\mathbf{f}(\mathbf{x} ; \boldsymbol{\omega})$

[^0]
[^0]: ${ }^{2}$ For instance, if $\mathbf{x}$ is a clean image then $\mathbf{g}(\mathbf{x})$ could be $\mathbf{x}$ corrupted by Gaussian noise.

---

#### Page 4

of depth $H$ parameterized by $\boldsymbol{\omega}=\left\{\mathbf{W}^{(h)}\right\}_{h=1}^{H} \in \mathcal{W}$, which we define recursively as follows:

$$
\mathbf{f}^{(0)}(\mathbf{x}):=\mathbf{x}, \quad \mathbf{z}^{(h)}(\mathbf{x}):=\mathbf{W}^{(h)} \mathbf{f}^{(h-1)}(\mathbf{x}), \quad \mathbf{f}^{(h)}(\mathbf{x}):=\boldsymbol{\sigma}^{(h)}\left(\mathbf{z}^{(h)}(\mathbf{x})\right), \quad \forall h=1, \ldots, H
$$

where $\mathbf{f}(\mathbf{x} ; \boldsymbol{\omega}):=\mathbf{f}^{(H)}(\mathbf{x})$ and $\boldsymbol{\sigma}^{(h)}$ is the non-linear activation of layer $h$. For brevity, we use $\mathbf{x}^{(h)}$ and $\mathbf{x}_{\mathbf{g}}^{(h)}$ as shorthand notations for $\mathbf{f}^{(h)}(\mathbf{x})$ and $\mathbf{f}^{(h)}(\mathbf{g}(\mathbf{x}))$ respectively. Given a corruption function g, Fig. 2 shows that $\mathbf{g}$ creates a covariate shift $\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(0)}:=\mathbf{x}_{\mathbf{g}}^{(0)}-\mathbf{x}^{(0)}$ in the input $\mathbf{x}$ leading to shifts $\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}:=\mathbf{x}_{\mathbf{g}}^{(h)}-\mathbf{x}^{(h)}$ in the output of each layer. This will eventually cause a shift in the per-sample loss $\boldsymbol{\delta}_{\mathbf{g}} \ell(\boldsymbol{\omega}, \mathbf{x}, y):=\ell\left(\boldsymbol{\omega}, \mathbf{x}_{\mathbf{g}}, y\right)-\ell(\boldsymbol{\omega}, \mathbf{x}, y)$. The following lemma characterizes the connection between $\boldsymbol{\delta}_{\mathbf{g}} \ell(\boldsymbol{\omega}, \mathbf{x}, y)$ and $\boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}$ :
Lemma 1. For all $h=1, \ldots, H$ and for all $\mathbf{x} \in \mathcal{X}$, there exists a scalar $C_{\mathbf{g}}^{(h)}(\mathbf{x})>0$ such that:

$$
\delta_{\mathbf{g}} \ell(\boldsymbol{\omega}, \mathbf{x}, y) \leq\left\langle\nabla_{\mathbf{g}^{(h+1)}} \ell(\boldsymbol{\omega}, \mathbf{x}, y) \otimes \boldsymbol{\delta}_{\mathbf{g}} \mathbf{x}^{(h)}, \mathbf{W}^{(h+1)}\right\rangle_{F}+\frac{C_{\mathbf{g}}^{(h)}(\mathbf{x})}{2}\left\|\mathbf{W}^{(h)}\right\|_{F}^{2}
$$

Here $\otimes$ denotes the outer product of two vectors, $\langle\cdot, \cdot\rangle_{F}$ denotes the Frobenius inner product of two matrices of the same dimension, $\|\cdot\|_{F}$ is the Frobenius norm, and $\nabla_{\mathbf{z}^{(h)}} \ell(\boldsymbol{\omega}, \mathbf{x}, y)$ is the Jacobian of the per-sample loss with respect to the pre-activation output $\mathbf{z}^{(h)}(\mathbf{x})$ at layer $h$. To prove Lemma 1, we use Assumption 1 and the following assumption about the loss function:
Assumption 3 (Lipschitz-continuous objective input gradients). The input gradient of the per-sample loss $\nabla_{\mathbf{x}} \ell(\boldsymbol{\omega}, \mathbf{x}, y)$ is Lipschitz continuous.

Assumption 3 allows us to define a quadratic bound of the loss function using a second-order Taylor expansion. The proof of Lemma 1 is provided in Appendix A. Using Lemma 1, we prove Theorem 1, which bounds the training loss in the presence of corruptions using the training loss under multiplicative perturbations in the weight space:
Theorem 1. For a function $\mathbf{g}: \mathcal{X} \rightarrow \mathcal{X}$ satisfying Assumption 1 and a loss function $\mathcal{L}$ satisfying Assumption 3, there exists $\boldsymbol{\xi}_{\mathbf{g}} \in \mathcal{W}$ and $C_{\mathbf{g}}>0$ such that:

$$
\mathcal{L}(\boldsymbol{\omega} ; \mathbf{g}(\mathcal{S})) \leq \mathcal{L}\left(\boldsymbol{\omega} \circ \boldsymbol{\xi}_{\mathbf{g}} ; \mathcal{S}\right)+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2}
$$

We provide the proof of Theorem 1 in Appendix B. This theorem establishes an upper bound for the target loss in Eq. (1):

$$
\mathcal{L}(\boldsymbol{\omega} ; \mathcal{G}(\mathcal{S})) \leq \mathbb{E}_{\mathbf{g} \sim \mathcal{G}}\left[\mathcal{L}\left(\boldsymbol{\omega} \circ \boldsymbol{\xi}_{\mathbf{g}} ; \mathcal{S}\right)+\frac{C_{\mathbf{g}}}{2}\|\boldsymbol{\omega}\|_{F}^{2}\right]
$$

This bound implies that training a DNN using the following loss function:

$$
\mathcal{L}_{\Xi}(\boldsymbol{\omega} ; \mathcal{S}):=\mathbb{E}_{\boldsymbol{\xi} \sim \Xi}[\mathcal{L}(\boldsymbol{\omega} \circ \boldsymbol{\xi} ; \mathcal{S})]+\frac{\lambda}{2}\|\boldsymbol{\omega}\|_{F}^{2}
$$

where the expected loss is taken with respect to a distribution $\boldsymbol{\Xi}$ of random MWPs $\boldsymbol{\xi}$, will minimize the upper bound of the loss $\mathcal{L}(\boldsymbol{\omega} ; \tilde{\mathcal{G}}(\mathcal{S}))$ of a hypothetical set of corruptions $\tilde{\mathcal{G}}$ simulated by $\boldsymbol{\xi} \sim \boldsymbol{\Xi}$. This approach results in a model robust to these simulated corruptions, which, according to Assumption 2, could indirectly improve robustness to corruptions in $\mathcal{G}$.
We note that the second term in Eq. (7) is the $L_{2}$-regularization commonly used in optimizing DNNs. Based on this proxy loss, we propose Algorithm 1 which minimizes the objective function in Eq. (7) when $\boldsymbol{\Xi}$ is an isotropic Gaussian distribution $\mathcal{N}\left(\mathbf{1}, \sigma^{2} \mathbf{I}\right)$. We call this algorithm Data Augmentation via Multiplicative Perturbations (DAMP), as it uses random MWPs during training to simulate input corruptions, which can be viewed as data augmentations.

Remark The standard method to calculate the expected loss in Eq. (7), which lacks a closed-form solution, is the Monte Carlo (MC) approximation. However, the training cost of this approach scales linearly with the number of MC samples. To match the training cost of standard SGD, Algorithm 1 divides each data batch into $M$ equal-sized sub-batches (Line 6) and calculates the loss on each sub-batch with different multiplicative noises from the noise distribution $\boldsymbol{\Xi}$ (Lines 7-9). The final gradient is obtained by averaging the sub-batch gradients (Line 11). Algorithm 1 is thus suitable for data parallelism in multi-GPU training, where the data batch is evenly distributed across $M>1$ GPUs. Compared to SGD, Algorithm 1 requires only two additional operations: generating Gaussian samples and point-wise multiplication, both of which have negligible computational costs. In our experiments, we found that both SGD and DAMP had similar training times.

---

#### Page 5

```
Input: training data \(\mathcal{S}=\left\{\left(\mathbf{x}_{k}, y_{k}\right)\right\}_{k=1}^{N}\), a neural network \(\mathbf{f}(\cdot ; \boldsymbol{\omega})\) parameterized by \(\boldsymbol{\omega} \in \mathbb{R}^{P}\), number of iterations \(T\), step sizes \(\left\{\eta_{t}\right\}_{t=1}^{T}\), number of sub-batch \(M\), batch size \(B\) divisible by \(M\), a noise distribution \(\boldsymbol{\Xi}=\mathcal{N}\left(\mathbf{1}, \sigma^{2} \mathbf{I}_{P}\right)\), weight decay coefficient \(\lambda\), a loss function \(\mathcal{L}: \mathbb{R}^{P} \rightarrow \mathbb{R}_{+}\).
Output: Optimized parameter \(\boldsymbol{\omega}^{(T)}\).
    Initialize parameter \(\boldsymbol{\omega}^{(0)}\).
    for \(t=1\) to \(T\) do
        Draw a mini-batch \(\mathcal{B}=\left\{\left(\mathbf{x}_{k}, y_{k}\right)\right\}_{k=1}^{B} \sim \mathcal{S}\).
        Divide the mini-batch into \(M\) disjoint sub-batches \(\left\{\mathcal{B}_{m}\right\}_{m=1}^{M}\) of equal size.
        for \(m=1\) to \(M\) in parallel do
            Draw a noise sample \(\boldsymbol{\xi}_{m} \sim \boldsymbol{\Xi}\).
            Compute the gradient \(\mathbf{g}_{m}=\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega} ; \mathcal{B}_{m}\right) \mid_{\boldsymbol{\omega}^{(t)} \in \boldsymbol{\xi}}\).
        end for
        Compute the average gradient: \(\mathbf{g}=\frac{1}{M} \sum_{m=1}^{M} \mathbf{g}_{m}\).
        Update the weights: \(\boldsymbol{\omega}^{(t+1)}=\boldsymbol{\omega}^{(t)}-\eta_{t}\left(\mathbf{g}+\lambda \boldsymbol{\omega}^{(t)}\right)\).
    end for
```

# 3 Adaptive Sharpness-Aware Minimization optimizes DNNs under adversarial multiplicative weight perturbations

In this section, we demonstrate that optimizing DNNs with adversarial MWPs follows a similar update rule to Adaptive Sharpness-Aware Minimization (ASAM) (Kwon et al., 2021). We first provide a brief description of ASAM and its predecessor Sharpness-Aware Minimization (SAM) (Foret et al., 2021):

SAM Motivated by previous findings that wide optima tend to generalize better than sharp ones (Keskar et al., 2017; Jiang et al., 2020), SAM regularizes the sharpness of an optimum by solving the following minimax optimization:

$$
\min _{\boldsymbol{\omega}} \max _{\|\boldsymbol{\xi}\|_{2} \leq \rho} \mathcal{L}(\omega+\boldsymbol{\xi} ; \mathcal{S})+\frac{\lambda}{2}\|\boldsymbol{\omega}\|_{F}^{2}
$$

which can be interpreted as optimizing DNNs under adversarial additive weight perturbations. To efficiently solve this problem, Foret et al. (2021) devise a two-step procedure for each iteration $t$ :

$$
\boldsymbol{\xi}^{(t)}=\rho \frac{\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega}^{(t)} ; \mathcal{S}\right)}{\left\|\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega}^{(t)} ; \mathcal{S}\right)\right\|_{2}}, \quad \boldsymbol{\omega}^{(t+1)}=\boldsymbol{\omega}^{(t)}-\eta_{t}\left(\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega}^{(t)}+\boldsymbol{\xi}^{(t)} ; \mathcal{S}\right)+\lambda \boldsymbol{\omega}^{(t)}\right)
$$

where $\eta_{t}$ is the learning rate. Each iteration of SAM thus takes twice as long to run than SGD.
ASAM Kwon et al. (2021) note that SAM attempts to minimize the maximum loss over a rigid sphere of radius $\rho$ around an optimum, which is not suitable for ReLU networks since their parameters can be freely re-scaled without affecting the outputs. The authors thus propose ASAM as an alternative optimization problem to SAM which regularizes the adaptive sharpness of an optimum:

$$
\min _{\boldsymbol{\omega}} \max _{\left\|T_{\boldsymbol{\omega}}^{\prime}\right\| \boldsymbol{\xi} \|_{2} \leq \rho} \mathcal{L}(\omega+\boldsymbol{\xi} ; \mathcal{S})+\frac{\lambda}{2}\|\boldsymbol{\omega}\|_{F}^{2}
$$

where $T_{\boldsymbol{\omega}}$ is an invertible linear operator used to reshape the perturbation region (so that it is not necessarily a sphere as in SAM). Kwon et al. (2021) found that $T_{\boldsymbol{\omega}}=|\boldsymbol{\omega}|$ produced the best results. Solving Eq. (10) in this case leads to the following two-step procedure for each iteration $t$ :

$$
\tilde{\boldsymbol{\xi}}^{(t)}=\rho \frac{\left(\boldsymbol{\omega}^{(t)}\right)^{2} \circ \nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega}^{(t)} ; \mathcal{S}\right)}{\left\|\boldsymbol{\omega}^{(t)} \circ \nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega}^{(t)} ; \mathcal{S}\right)\right\|_{2}}, \quad \boldsymbol{\omega}^{(t+1)}=\boldsymbol{\omega}^{(t)}-\eta_{t}\left(\nabla_{\boldsymbol{\omega}} \mathcal{L}\left(\boldsymbol{\omega}^{(t)}+\tilde{\boldsymbol{\xi}}^{(t)} ; \mathcal{S}\right)+\lambda \boldsymbol{\omega}^{(t)}\right)
$$

Similar to SAM, each iteration of ASAM also takes twice as long to run than SGD.

---

#### Page 6

ASAM and adversarial multiplicative perturbations Algorithm 1 minimizes the expected loss in Eq. (7). Instead, we could minimize the loss under the adversarial MWP:

$$
\mathcal{L}_{\max }(\omega ; \mathcal{S}):=\max _{\|\boldsymbol{\xi}\|_{2} \leq \rho} \mathcal{L}(\omega+\omega \circ \boldsymbol{\xi} ; \mathcal{S})+\frac{\lambda}{2}\|\omega\|_{F}^{2}
$$

Following Foret et al. (2021), we solve this optimization problem by using a first-order Taylor expansion of $\mathcal{L}(\omega+\omega \circ \boldsymbol{\xi} ; \mathcal{S})$ to find an approximate solution of the inner maximization:

$$
\underset{\|\boldsymbol{\xi}\|_{2} \leq \rho}{\arg \max } \mathcal{L}(\omega+\omega \circ \boldsymbol{\xi} ; \mathcal{S}) \approx \underset{\|\boldsymbol{\xi}\|_{2} \leq \rho}{\arg \max } \mathcal{L}(\omega ; \mathcal{S})+\left\langle\omega \circ \boldsymbol{\xi}, \nabla_{\omega} \mathcal{L}(\omega ; \mathcal{S})\right\rangle
$$

The maximizer of the Taylor expansion is:

$$
\widehat{\boldsymbol{\xi}}(\boldsymbol{\omega})=\rho \frac{\boldsymbol{\omega} \circ \nabla_{\boldsymbol{\omega}} \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})}{\left\|\boldsymbol{\omega} \circ \nabla_{\boldsymbol{\omega}} \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})\right\|_{2}}
$$

Subituting back into Eq. (12) and differentiating, we get:

$$
\begin{aligned}
\nabla_{\boldsymbol{\omega}} \mathcal{L}_{\max }(\omega ; \mathcal{S}) & \approx \nabla_{\boldsymbol{\omega}} \mathcal{L}(\widehat{\omega} ; \mathcal{S})+\lambda \omega=\nabla_{\boldsymbol{\omega}} \widehat{\boldsymbol{\omega}} \cdot \nabla_{\widehat{\boldsymbol{\omega}}} \mathcal{L}(\widehat{\omega} ; \mathcal{S})+\lambda \omega \\
& =\nabla_{\widehat{\boldsymbol{\omega}}} \mathcal{L}(\widehat{\omega} ; \mathcal{S})+\nabla_{\boldsymbol{\omega}}(\omega \circ \widehat{\boldsymbol{\xi}}(\omega)) \cdot \nabla_{\widehat{\boldsymbol{\omega}}} \mathcal{L}(\widehat{\omega} ; \mathcal{S})+\lambda \omega
\end{aligned}
$$

where $\widehat{\boldsymbol{\omega}}$ is the perturbed weight:

$$
\widehat{\boldsymbol{\omega}}=\boldsymbol{\omega}+\boldsymbol{\omega} \circ \widehat{\boldsymbol{\xi}}(\boldsymbol{\omega})=\boldsymbol{\omega}+\rho \frac{\boldsymbol{\omega}^{2} \circ \nabla_{\boldsymbol{\omega}} \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})}{\left\|\boldsymbol{\omega} \circ \nabla_{\boldsymbol{\omega}} \mathcal{L}(\boldsymbol{\omega} ; \mathcal{S})\right\|_{2}}
$$

Similar to Foret et al. (2021), we omit the second summand in Eq. (16) for efficiency, as it requires calculating the Hessian of the loss. We then arrive at the gradient formula in the update rule of ASAM in Eq. (11). We have thus established a connection between ASAM and adversarial MWPs.

# 4 Empirical evaluation

In this section, we assess the corruption robustness of DAMP and ASAM in image classification tasks. We conduct experiments using the CIFAR-10/100 (Krizhevsky, 2009), TinyImageNet (Le and Yang, 2015), and ImageNet (Deng et al., 2009) datasets. For evaluation on corrupted images, we utilize the CIFAR-10/100-C, TinyImageNet-C, and ImageNet-C datasets provided by Hendrycks and Dietterich (2019), as well as ImageNet- $\overline{\text { C }}$ (Mintun et al., 2021), ImageNet-D (Zhang et al., 2024), ImageNetA (Hendrycks et al., 2021), ImageNet-Sketch (Wang et al., 2019), ImageNet-\{Drawing, Cartoon\} (Salvador and Oberman, 2022), and ImageNet-Hard (Taesiri et al., 2023) datasets, which encapsulate a wide range of corruptions. Detail descriptions of these datasets are provided in Appendix E. We further evaluate the models on adversarial examples generated by the Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2014). In terms of architectures, we use ResNet18 (He et al., 2016a) for CIFAR-10/100, PreActResNet18 (He et al., 2016b) for TinyImageNet, ResNet50 (He et al., 2016a), ViT-S/16, and ViT-B/16 (Dosovitskiy et al., 2021) for ImageNet. We ran all experiments on a single machine with 8 Nvidia V100 GPUs. Appendix F includes detailed information for each experiment.

### 4.1 Comparing DAMP to directly using corruptions as augmentations

In this section, we compare the corruption robustness of DNNs trained using DAMP with those trained directly on corrupted images. To train models on corrupted images, we utilize Algorithm 2 described in the Appendix. For a given target corruption g, Algorithm 2 randomly selects half the images in each training batch and applies $\mathbf{g}$ to them. This random selection process enhances the final model's robustness to the target corruption while maintaining its accuracy on clean images. We use the imagecorruptions library (Michaelis et al., 2019) to apply the corruptions during training.

Evaluation metric We use the corruption error $\mathrm{CE}_{c}^{f}$ (Hendrycks and Dietterich, 2019) which measures the predictive error of classifier $f$ in the presence of corruption $c$. Denote $E_{s, c}^{f}$ as the error of classifier $f$ under corruption $c$ with corruption severity $s$, the corruption error $\mathrm{CE}_{c}^{f}$ is defined as:

$$
\mathrm{CE}_{c}^{f}=\left(\sum_{s=1}^{5} E_{s, c}^{f}\right) /\left(\sum_{s=1}^{5} E_{s, c}^{f_{\text {tow }}}\right)
$$

---

#### Page 7

> **Image description.** This image is a heatmap displaying numerical data related to the performance of different training methods under various image corruptions.
>
> - **Structure:** The heatmap is organized as a table with rows and columns. Each cell contains a numerical value and is colored according to a color scale on the right. The table is enclosed by thin, light-colored gridlines.
>
> - **Rows:** The rows represent different training configurations, indicated by text labels on the left. These labels include:
>
>   - (SGD, none)
>   - (DAMP, none)
>   - (SGD, zoom_blur)
>   - (SGD, impulse_noise)
>   - (SGD, shot_noise)
>   - (SGD, gaussian_noise)
>   - (SGD, motion_blur)
>
> - **Columns:** The columns represent different types of image corruptions, with labels along the top, rotated diagonally for readability. These labels include:
>
>   - zoom_blur
>   - impulse_noise
>   - shot_noise
>   - gaussian_noise
>   - motion_blur
>   - pixelate
>   - brightness
>   - snow
>   - frost
>   - contrast
>   - fog
>   - defocus_blur
>   - glass_blur
>   - jpeg_compression
>   - elastic_transform
>   - none
>   - Avg
>
> - **Color Scale:** A vertical color bar is present on the right side of the heatmap, ranging from approximately 0.8 to 1.2. The color transitions from blue at the lower end to red at the higher end, indicating the magnitude of the numerical values in the cells.
>
> - **Numerical Values:** Each cell contains a numerical value, typically formatted to two decimal places (e.g., -1.00, 0.81, 0.54). These values represent the performance metric (CEc^f) for each training configuration under each corruption.
>
> - **Interpretation:** The heatmap visually represents the performance of different training methods (SGD and DAMP) when exposed to various types of image corruptions. The color intensity of each cell indicates the level of error, with blue representing lower error (better performance) and red representing higher error (worse performance). The "Avg" column shows the average performance across all corruptions for each training method.

Figure 3: DAMP improves robustness to all corruptions while preserving accuracy on clean images. Results of ResNet18/CIFAR-100 experiments averaged over 5 seeds. The heatmap shows $\mathrm{CE}_{c}^{f}$ described in Eq. (18) (lower is better), where each row corresponds to a tuple of training (method, corruption), while each column corresponds to the test corruption. The Avg column shows the average of the results of the previous columns. none indicates no corruption. We use the models trained under the SGD/none setting (first row) as baselines to calculate the $\mathrm{CE}_{c}^{f}$. The last five rows are the 5 best training corruptions ranked by the results in the Avg column.

For this metric, lower is better. Here $f_{\text {baseline }}$ is a baseline classifier whose usage is to make the error more comparable between corruptions as some corruptions can be more challenging than others (Hendrycks and Dietterich, 2019). For each experiment setting, we use the model trained by SGD without corruptions as $f_{\text {baseline }}$.

Results We visualize the results for the ResNet18/CIFAR-100 setting in Fig. 3. The results for the ResNet18/CIFAR-10 and PreActResNet18/TinyImageNet settings are presented in Figs. 5 and 6 in the Appendix. Figs. 3, 5 and 6 demonstrate that DAMP improves predictive accuracy over plain SGD across all corruptions without compromising accuracy on clean images. Although Fig. 3 indicates that including zoom_blur as an augmentation when training ResNet18 on CIFAR-100 yields better results than DAMP on average, it also reduces accuracy on clean images and the brightness corruption. Overall, these figures show that incorporating a specific corruption as data augmentation during training enhances robustness to that particular corruption but may reduce performance on clean images and other corruptions. In contrast, DAMP consistently improves robustness across all corruptions. Notably, DAMP even enhances accuracy on clean images in the PreActResNet18/TinyImageNet setting, as shown in Fig. 6.

# 4.2 Comparing DAMP to random additive perturbations

In this section, we investigate whether additive weight perturbations can also enhance corruption robustness. To this end, we compare DAMP with its variant, Data Augmentation via Additive Perturbations (DAAP). Unlike DAMP, DAAP perturbs weights during training with random additive Gaussian noises centered at 0 , as detailed in Algorithm 3 in the Appendix. Fig. 7 in the Appendix presents the results of DAMP and DAAP under different noise standard deviations, alongside standard SGD. Overall, Fig. 7 shows that across different experimental settings, the corruption robustness of DAAP is only slightly better than SGD and is worse than DAMP. Therefore, we conclude that MWPs are better than their additive counterparts at improving robustness to corruptions.

### 4.3 Benchmark results

In this section, we compare DAMP with Dropout (Srivastava et al., 2014), SAM (Foret et al., 2021), and ASAM (Kwon et al., 2021). For SAM and ASAM, we optimize the neighborhood size $\rho$ by using $10 \%$ of the training set as a validation set. Similarly, we adjust the noise standard deviation $\sigma$ for DAMP and the drop rate $p$ for Dropout following the same procedure. For hyperparameters and additional training details, please refer to Appendix F.

CIFAR-10/100 and TinyImageNet. Fig. 4 visualizes the predictive errors of DAMP and the baseline methods on CIFAR-10/100 and TinyImageNet, with all methods trained for the same number of epochs. It demonstrates that DAMP consistently outperforms Dropout across various datasets and corruption severities, despite having the same training cost. Notably, DAMP outperforms SAM under

---

#### Page 8

> **Image description.** The image consists of three horizontal bar charts, each representing the predictive errors of different machine learning methods on corrupted images.
>
> - **Overall Structure:** The charts are arranged side-by-side. Each chart has "Corruption intensity" on the vertical axis and "Error (%)" on the horizontal axis. The error values increase from left to right.
>
> - **Chart Titles:**
>
>   - The leftmost chart is titled "ResNet18 / CIFAR-10".
>   - The middle chart is titled "ResNet18 / CIFAR-100".
>   - The rightmost chart is titled "PreActResNet18 / TinyImageNet".
>
> - **Axes:**
>
>   - The vertical axis (Corruption intensity) is labeled with numbers 0 through 5, representing different levels of corruption.
>   - The horizontal axis (Error (%)) ranges from 0 to 40 in the first chart, 0 to 60 in the second chart, and 0 to 80 in the third chart. Each axis has tick marks and labels at consistent intervals. An arrow pointing downwards is placed next to the label, indicating that lower error is better.
>
> - **Bars:** Each chart contains multiple horizontal bars for each corruption intensity level. Each bar represents the error rate of a different method:
>
>   - Blue bars represent "Dropout".
>   - Orange bars represent "DAMP".
>   - Green bars represent "SAM".
>   - Red bars represent "ASAM".
>     Each bar has a small error bar at its end.
>
> - **Legend:** A legend is placed below the charts, indicating the color-coding for each method: "Dropout" (blue), "DAMP" (orange), "SAM" (green), and "ASAM" (red).

Figure 4: DAMP surpasses SAM on corrupted images in most cases, despite requiring only half the training cost. We report the predictive errors (lower is better) averaged over 5 seeds. A severity level of 0 indicates no corruption. We use the same number of epochs for all methods.

Table 1: DAMP surpasses the baselines on corrupted images in most cases and on average. We report the predictive errors (lower is better) averaged over 3 seeds for the ResNet50 / ImageNet experiments. Subscript numbers represent standard deviations. We evaluate the models on IN- $\{\mathrm{C}, \overline{\mathrm{C}}$, A, D, Sketch, Drawing, Cartoon, Hard $\}$, and adversarial examples generated by FGSM. For FGSM, we use $\epsilon=2 / 224$. For IN- $\{\mathrm{C}, \overline{\mathrm{C}}\}$, we report the results averaged over all corruption types and severity levels. We use 90 epochs and the basic Inception-style preprocessing for all experiments.

| Method  | Clean <br> Error (\%) $\downarrow$ | Corrupted Error (\%) $\downarrow$ |                           |                          |                           |                           |                          |                          |                          |                          |
| :-----: | :--------------------------------: | :-------------------------------: | :-----------------------: | :----------------------: | :-----------------------: | :-----------------------: | :----------------------: | :----------------------: | :----------------------: | :----------------------: | ------------------ |
|         |                                    |               FGSM                |             A             |            C             | $\overline{\text { C }}$  |          Cartoon          |            D             |         Drawing          |          Sketch          |           Hard           | Avg                |
| Dropout |            $23.6_{0.2}$            |           $90.7_{0.2}$            | $\mathbf{9 5 . 7}_{<0.1}$ |       $61.7_{0.2}$       |       $61.6_{<0.1}$       |       $49.6_{0.2}$        |      $88.9_{<0.1}$       |       $77.4_{1.3}$       |       $78.3_{0.3}$       |       $85.8_{0.1}$       | 76.6               |
|  DAMP   |           $23.8_{<0.1}$            |     $\mathbf{8 8 . 3}_{0.1}$      |       $96.2_{<0.1}$       | $\mathbf{5 8 . 6}_{0.1}$ | $\mathbf{5 8 . 7}_{<0.1}$ | $\mathbf{4 4 . 4}_{<0.1}$ |      $88.7_{<0.1}$       | $\mathbf{7 1 . 1}_{0.5}$ | $\mathbf{7 0 . 3}_{0.2}$ |       $85.3_{0.2}$       | $\mathbf{7 4 . 2}$ |
|   SAM   |           $23.2_{<0.1}$            |           $90.4_{0.2}$            |       $96.6_{0.1}$        |       $60.2_{0.2}$       |       $60.7_{0.1}$        |       $47.6_{0.1}$        | $\mathbf{8 8 . 3}_{0.1}$ |      $74.8_{<0.1}$       |       $77.5_{0.1}$       |       $85.8_{0.2}$       | 75.8               |
|  ASAM   |      $\mathbf{2 2 . 8}_{0.1}$      |           $89.7_{0.2}$            |       $96.8_{0.1}$        |       $58.9_{0.1}$       |       $59.2_{0.1}$        |       $45.5_{<0.1}$       |       $88.7_{0.1}$       |       $72.3_{0.1}$       |       $76.4_{0.2}$       | $\mathbf{8 5 . 2}_{0.1}$ | 74.7               |

most corruption scenarios, even though SAM takes twice as long to train and has higher accuracy on clean images. Additionally, DAMP improves accuracy on clean images over Dropout on CIFAR-100 and TinyImageNet. Finally, ASAM consistently surpasses other methods on both clean and corrupted images, as it employs adversarial MWPs (Section 3). However, like SAM, each ASAM experiment takes twice as long as DAMP given the same epoch counts.

ResNet50 / ImageNet Table 1 presents the predictive errors for the ResNet50 / ImageNet setting on a variety of corruption test sets. It shows that DAMP consistently outperforms the baselines in most corruption scenarios and on average, despite having half the training cost of SAM and ASAM.

ViT-S16 / ImageNet / Basic augmentations Table 2 presents the predictive errors for the ViT-S16 / ImageNet setting, using the training setup from Beyer et al. (2022) but with only basic Inception-style preprocessing (Szegedy et al., 2016). Remarkably, DAMP can train ViT-S16 from scratch in 200 epochs to match ResNet50's accuracy without advanced data augmentation. This is significant as ViT typically requires either extensive pretraining (Dosovitskiy et al., 2021), comprehensive data augmentation (Beyer et al., 2022), sophisticated training techniques (Chen et al., 2022), or modifications to the original architecture (Yuan et al., 2021) to perform well on ImageNet. Additionally, DAMP consistently ranks in the top 2 for corruption robustness across various test settings and has the best corruption robustness on average (last column). Comparing Tables 1 and 2 reveals that ViT-S16 is more robust to corruptions than ResNet50 when both have similar performance on clean images.

ViT / ImageNet / Advanced augmentations Table 3 presents the predictive errors of ViT-S16 and ViT-B16 on ImageNet with MixUp (Zhang et al., 2018) and RandAugment (Cubuk et al., 2020). These results indicate that DAMP can be combined with modern augmentation techniques to further improve robustness. Furthermore, using DAMP to train a larger model (ViT-B16) yields better results than using SAM/ASAM to train a smaller model (ViT-S16), given the same amount of training time.

---

#### Page 9

Table 2: ViT-S16 / ImageNet (IN) with basic Inception-style data augmentations. Due to the high training cost, we report the predictive error (lower is better) of a single run. We evaluate corruption robustness of the models using IN- $\{\mathrm{C}, \overline{\mathrm{C}}, \mathrm{A}, \mathrm{D}$, Sketch, Drawing, Cartoon, Hard $\}$, and adversarial examples generated by FGSM. For IN- $\{\mathrm{C}, \overline{\mathrm{C}}\}$, we report the results averaged over all corruption types and severity levels. For FGSM, we use $\epsilon=2 / 224$. We also report the runtime of each experiment, showing that SAM and ASAM take twice as long to run than DAMP and AdamW given the same number of epochs. DAMP produces the most robust model on average.

| Method  | \#Epochs | Runtime | Clean Error (\%) $\downarrow$ | Corrupted Error (\%) $\downarrow$ |       |       |                          |         |       |         |        |       |       |
| :-----: | :------: | :-----: | :---------------------------: | :-------------------------------: | :---: | :---: | :----------------------: | :-----: | :---: | :-----: | :----: | :---: | :---: |
|         |          |         |                               |               FGSM                |   A   |   C   | $\overline{\text { C }}$ | Cartoon |   D   | Drawing | Sketch | Hard  |  Avg  |
| Dropout |   100    |  20.6h  |             28.55             |               93.47               | 93.44 | 65.87 |          64.52           |  50.37  | 91.15 |  79.62  | 88.06  | 87.19 | 79.30 |
|         |   200    |  41.1h  |             28.74             |               90.95               | 93.33 | 66.90 |          64.83           |  51.23  | 92.56 |  81.24  | 87.99  | 87.60 | 79.63 |
|  DAMP   |   100    |  20.7h  |             25.50             |               92.76               | 92.92 | 57.85 |          57.02           |  44.78  | 88.79 |  69.92  | 83.16  | 85.65 | 74.76 |
|         |   200    |  41.1h  |             23.75             |               84.33               | 90.56 | 55.58 |          55.58           |  41.06  | 87.87 |  68.36  | 81.82  | 84.18 | 72.15 |
|   SAM   |   100    |   41h   |             23.91             |               87.61               | 93.96 | 55.56 |          55.93           |  42.53  | 88.23 |  69.53  | 81.86  | 85.54 | 73.42 |
|  ASAM   |   100    |  41.1h  |             24.01             |               85.85               | 92.99 | 55.13 |          55.64           |  40.74  | 89.03 |  67.80  | 81.47  | 84.31 | 72.55 |

Table 3: ViT / ImageNet (IN) with MixUp and RandAugment. We train ViT-S16 and ViT-B16 on ImageNet from scratch with advanced data augmentations (DAs). We evaluate the models on IN- $\{\mathrm{C}, \overline{\mathrm{C}}, \mathrm{A}, \mathrm{D}$, Sketch, Drawing, Cartoon, Hard $\}$, and adversarial examples generated by FGSM. For FGSM, we use $\epsilon=2 / 224$. For IN- $\{\mathrm{C}, \overline{\mathrm{C}}\}$, we report the results averaged over all corruption types and severity levels. These results indicate that: (i) DAMP can be combined with modern DA techniques to further enhance robustness; (ii) DAMP is capable of training large models like ViT-B16; (iii) given the same amount of training time, it is better to train a large model (ViT-B16) using DAMP than to train a smaller model (ViT-S16) using SAM/ASAM.

| Model | Method  | \#Epochs | Runtime | Clean Error (\%) $\downarrow$ | Corrupted Error (\%) $\downarrow$ |                          |       |       |         |       |         |        |       |
| :---: | :-----: | :------: | :-----: | :---------------------------: | :-------------------------------: | :----------------------: | :---: | :---: | :-----: | :---: | :-----: | :----: | :---: | ----- |
|       |         |          |         |                               |               FGSM                | $\overline{\text { C }}$ |   A   |   C   | Cartoon |   D   | Drawing | Sketch | Hard  | Avg   |
|  ViT  | Dropout |   500    |  111h   |             20.25             |               62.45               |          40.85           | 84.29 | 44.72 |  34.35  | 86.59 |  56.31  | 71.03  | 80.87 | 62.38 |
|       |  DAMP   |   500    |  111h   |             20.09             |               59.87               |          39.30           | 83.12 | 43.18 |  34.01  | 84.74 |  54.16  | 68.03  | 80.05 | 60.72 |
|  S16  |   SAM   |   300    |  123h   |             20.17             |               59.92               |          40.05           | 83.91 | 44.04 |  34.34  | 85.99 |  55.63  | 70.85  | 80.18 | 61.66 |
|       |  ASAM   |   300    |  123h   |             20.38             |               59.38               |          39.44           | 83.64 | 43.41 |  33.82  | 85.41 |  54.43  | 69.13  | 80.50 | 61.02 |
|  ViT  | Dropout |   275    |  123h   |             20.41             |               56.43               |          39.14           | 82.85 | 43.82 |  33.13  | 87.72 |  56.15  | 71.36  | 79.13 | 61.08 |
|       |  DAMP   |   275    |  124h   |             19.36             |               55.20               |          37.77           | 80.49 | 41.67 |  31.63  | 87.06 |  52.32  | 67.91  | 78.69 | 59.19 |
|  B16  |   SAM   |   150    |  135h   |             19.84             |               61.85               |          39.09           | 82.69 | 43.53 |  32.95  | 88.38 |  55.33  | 71.22  | 79.48 | 61.61 |
|       |  ASAM   |   150    |  136h   |             19.40             |               58.87               |          37.41           | 82.21 | 41.18 |  30.76  | 88.03 |  51.84  | 69.54  | 78.83 | 59.85 |

# 5 Related works

Dropout Perhaps most relevant to our method is Dropout (Srivastava et al., 2014) and its many variants, such as DropConnect (Wan et al., 2013) and Variational Dropout (Kingma et al., 2015). These methods can be viewed as DAMP where the noise distribution $\Xi$ is a structured multivariate Bernoulli distribution. For instance, Dropout multiplies all the weights connecting to a node with a binary random variable $p \sim \operatorname{Bernoulli}(\rho)$. While the main motivation of these Dropout methods is to prevent co-adaptations of neurons to improve generalization on clean data, the motivation of DAMP is to improve robustness to input corruptions without harming accuracy on clean data. Nonetheless, our experiments show that DAMP can improve generalization on clean data in certain scenarios, such as PreActResNet18/TinyImageNet and ViT-S16/ImageNet.

Ensemble methods Ensemble methods, such as Deep ensembles (Lakshminarayanan et al., 2017) and Bayesian neural networks (BNNs) (Graves, 2011; Blundell et al., 2015; Gal and Ghahramani, 2016; Louizos and Welling, 2017; Izmailov et al., 2021; Trinh et al., 2022), have been explored as effective defenses against corruptions. Ovadia et al. (2019) benchmarked some of these methods, demonstrating that they are more robust to corruptions compared to a single model. However, the training and inference costs of these methods increase linearly with the number of ensemble members, making them inefficient for use with very large DNNs.

Data augmentation Data augmentations aim at enhancing robustness include AugMix (Hendrycks et al., 2019), which combines common image transformations; Patch Gaussian (Lopes et al., 2019), which applies Gaussian noise to square patches; ANT (Rusak et al., 2020), which uses adversarially learned noise distributions for augmentation; and AutoAugment (Cubuk et al., 2018), which learns

---

#### Page 10

augmentation policies directly from the training data. These methods have been demonstrated to improve robustness to the corruptions in ImageNet-C (Hendrycks and Dietterich, 2019). Mintun et al. (2021) attribute the success of these methods to the fact that they generate augmented images perceptually similar to the corruptions in ImageNet-C and propose ImageNet- $\overline{\mathrm{C}}$, a test set of 10 new corruptions that are challenging to models trained by these augmentation methods.

Test-time adaptations via BatchNorm One effective approach to using unlabelled data to improve corruption robustness is to keep BatchNorm (Ioffe and Szegedy, 2015) on at test-time to adapt the batch statistics to the corrupted test data (Li et al., 2016; Nado et al., 2020; Schneider et al., 2020; Benz et al., 2021). A major drawback is that this approach cannot be used with BatchNorm-free architectures, such as Vision Transformer (Dosovitskiy et al., 2021).

# 6 Conclusion

In this work, we demonstrate that MWPs improve robustness of DNNs to a wide range of input corruptions. We introduce DAMP, a simple training algorithm that perturbs weights during training with random multiplicative noise while maintaining the same training cost as standard SGD. We further show that ASAM (Kwon et al., 2021) can be viewed as optimizing DNNs under adversarial MWPs. Our experiments show that both DAMP and ASAM indeed produce models that are robust to corruptions. DAMP is also shown to improve sample efficiency of Vision Transformer, allowing it to achieve comparable performance to ResNet50 on medium size datasets such as ImageNet without extensive data augmentations. Additionally, DAMP can be used in conjunction with modern augmentation techniques such as MixUp and RandAugment to further boost robustness. As DAMP is domain-agnostic, one future direction is to explore its effectiveness in domains other than computer vision, such as natural language processing and reinforcement learning. Another direction is to explore alternative noise distributions to the Gaussian distribution used in our work.

Limitations Here we outline some limitations of this work. First, the proof of Theorem 1 assumes a simple feedforward neural network, thus it does not take into accounts modern DNN's components such as normalization layers and attentions. Second, we only explored random Gaussian multiplicative perturbations, and there are likely more sophisticated noise distributions that could further boost corruption robustness.
