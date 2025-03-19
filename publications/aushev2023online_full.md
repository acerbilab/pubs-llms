```
@article{aushev2023online,
  title={Online Simulator-Based Experimental Design for Cognitive Model Selection},
  author={Alexander Aushev and Aini Putkonen and Gregoire Clarte and Suyog H. Chandramouli and Luigi Acerbi and Samuel Kaski and Andrew Howes},
  year={2023},
  journal={Computational Brain \& Behavior},
  doi={10.1007/s42113-023-00180-7}
}
```

---

#### Page 1

# Online Simulator-Based Experimental Design for Cognitive Model Selection

## Alexander Aushev

Department of Computer Science
Aalto University, Finland
alexander.aushev@aalto.fi

## Gregoire Clarte

Department of Computer Science
University of Helsinki \& FCAI, Finland
gregoire.clarte@helsinki.fi

## Aini Putkonen

Department of Communications and Networking
Aalto University, Finland
aini.putkonen@aalto.fi

## Suyog Chandramouli

Department of Communications and Networking
Aalto University, Finland
suyog.chandramouli@aalto.fi

## Luigi Acerbi

Department of Computer Science
University of Helsinki \& FCAI, Finland
luigi.acerbi@helsinki.fi

## Samuel Kaski

Department of Computer Science
Aalto University, Finland
Department of Computer Science
University of Manchester, UK
samuel.kaski@aalto.fi

## Andrew Howes

School of Computer Science
University of Birmingham, UK
a.howes@bham.ac.uk

March 7, 2023

## ABSTRACT

The problem of model selection with a limited number of experimental trials has received considerable attention in cognitive science, where the role of experiments is to discriminate between theories expressed as computational models. Research on this subject has mostly been restricted to optimal experiment design with analytically tractable models. However, cognitive models of increasing complexity, with intractable likelihoods, are becoming more commonplace. In this paper, we propose BOSMOS: an approach to experimental design that can select between computational models without tractable likelihoods. It does so in a data-efficient manner, by sequentially and adaptively generating informative experiments. In contrast to previous approaches, we introduce a novel simulator-based utility objective for design selection, and a new approximation of the model likelihood for model selection. In simulated experiments, we demonstrate that the proposed BOSMOS technique can accurately select models in up to 2 orders of magnitude less time than existing LFI alternatives for three cognitive science tasks: memory retention, sequential signal detection and risky choice.

## 1 Introduction

The problem of selecting between competing models of cognition is critical to progress in cognitive science. The goal of model selection is to choose the model that most closely represents the cognitive process which generated the observed behavioural data. Typically, model selection involves maximizing the fit of each model's parameters to data and balancing the quality of the model-fit and its complexity. It is crucial that any model selection method used is robust and sample-efficient, and that it correctly measures how well each model approximates the data-generating cognitive process.
It is also crucial that any model selection process is provided with high quality data from well-designed experiments, and that these data are sufficiently informative to support efficient selection. Research on optimal experimental design

---

#### Page 2

(OED) addresses this problem by focusing on how to design experiments that support parameter estimation of single models and, in some cases, maximize information for model selection [Cavagnaro et al., 2010, Moon et al., 2022, Blau et al., 2022].

However, one outstanding difficulty for model selection is that many models do not have tractable likelihoods. The model likelihoods represent the probability of observed data being produced by model parameters and simplify tractable inference [van Opheusden et al., 2020]. In their absence, likelihood-free inference (LFI) methods can be used, which rely on forward simulations (or samples from the model) to replace the likelihood. Another difficulty is that existing methods for OED are slow - very slow - which makes them impractical for many applications. In this paper, we address these problems by investigating a new algorithm that automatically designs experiments for likelihood-free models much more quickly than previous approaches. The new algorithm is called Bayesian optimization for simulator-based model selection (BOSMOS).

In BOSMOS, model selection is conducted in a Bayesian framework. In this setting, inference is carried out using marginal likelihood, which incorporate, by definition, a penalty for model complexity, i.e., Occam's Razor. Additionally, the Bayesian framework allows getting Bayesian posteriors over all possible values, rather than point estimates; this is crucial for quantifying uncertainty, for instance, when multiple models can explain the data similarly well (non-identifiability or poor identifiability; Anderson, 1978, Acerbi et al., 2014), or when some of the models are misspecified (e.g. the behaviour cannot be reproduced by the model due to non-independence of the experimental trials; Lee et al., 2019). These problems are compounded in computational cognitive modeling where non-identifiability also arises due to human strategic flexibility [Howes et al., 2009, Madsen et al., 2019, Kangasrääsiö et al., 2019, Oulasvirta et al., 2022]. For these reasons, there is an interest in Bayesian approaches in computational cognitive science [Madsen et al., 2018].

As we have said, a key problem for model selection is selection of the design variables that define an experiment. When resources are limited, experimental designs can be carefully selected to yield as much information about the models as possible. Adaptive Design Optimization (ADO) [Cavagnaro et al., 2010, 2013b] is one influential approach to selecting experimental designs. ADO proposes designs by maximizing the so-called utility objective, which measures the amount of information about the candidate models and their quality. Unfortunately, common utility objectives, such as mutual information [Shannon, 1948, Cavagnaro et al., 2010] or expected entropy [Yang and Qiu, 2005], cannot be applied when computational models lack a tractable likelihood. In such cases, research suggests adopting LFI methods, in which the computational model generates synthetic observations for inference [Gutmann and Corander, 2016, Sisson et al., 2018, Papamakarios et al., 2019]. This broad family of methods is also known as approximate Bayesian computation (ABC) [Beaumont et al., 2002, Kangasrääsiö et al., 2019] and simulator- or simulation-based inference [Cranmer et al., 2020]. To date, LFI methods for ADO have focused on parameter inference for a single model rather than model selection.

Model selection with limited design iterations requires a choice of design variables that optimize model discrimination, as well as improving parameter estimation. The complexity of this task is compounded in the context of LFI, where expensive samples from the model are required. We aim at reducing the number of model simulations. For this reason, in our approach, called BOSMOS, we use Bayesian Optimization (BO) [Frazier, 2018, Greenhill et al., 2020] for both design selection and model selection. The advantage of BO is that it is highly sample-efficient and therefore has a direct impact on reducing the need for model simulation. BOSMOS combines the ADO approach with LFI techniques in a novel way, resulting in a faster method to carry out optimal design of experiments to discriminate between computational cognitive models, with a minimal number of trials.

The main contributions of the paper are as follows:

- A novel approach to simulator-based model selection that casts LFI for multiple models under the Bayesian framework through the approximation of the model likelihood. As a result, the approach provides a full joint Bayesian posterior for models and their parameters given collected experimental data.
- A novel simulator-based utility objective for choosing experimental designs that maximizes the behavioural variation in current beliefs about model configurations. Along with the sample-efficient LFI procedure, it reduces the time cost from one hour, for competitor methods, to less than a minute in the majority of case studies, bringing the method closer to enabling real-time cognitive model testing with human subjects.
- Close integration of the two above contributions yields the first online, sample-efficient, simulation-based, and fully Bayesian experimental design approach to model selection.
- The new approach was tested on three well-known paradigms in psychology - memory retention, sequential signal detection and risky choice - and, despite not requiring likelihoods, reaches similar accuracy to the existing methods which do require them.

---

#### Page 3

# 2 Background

In this article, we are concerned with situations where the purpose of experiments is to gather data that can discriminate between models. The traditional approach in such a context begins with the collection of large amounts of data from a large number of participants on a design that is fixed based on intuition; this is followed by evaluation of the model fits using a desired model selection criteria such as as AIC, BIC, cross-validation, etc. This is an inefficient approach - the informativeness of the collected data for choosing models is unknown in advance, and collecting large amounts of data may often prove expensive in terms of time and monetary resources (for instance, cases that involve expensive equipment, such as fMRI, or in clinical settings). These issues have been addressed by modern optimal experimental design methods which we consider in this section and summarize in Table 1.

Optimal experimental design. Optimal experiment design (OED) is a classic problem in statistics [Lindley, 1956, Kiefer, 1959], which saw a resurgence in the last decade due to improvements in computational methods and availability of computational resources. Specifically, adaptive design optimization (ADO) Cavagnaro et al. [2010, 2013b] was proposed for cognitive science models, which has been successfully applied in different experimental settings including memory and decision-making. In ADO, the designs are selected according to a global utility objective, which is an average value of the local utility over all possible data (behavioural responses) and model parameters, weighted by the likelihood and priors [Myung et al., 2013]. More general approaches, such as Kim et al. [2014], improve upon ADO by combining it with hierarchical modelling, which allow them to form richer priors over the model parameters. While useful, the main drawback of these methods is that they work only with tractable (or analytical) parametric models, that is models whose likelihood is explicitly available and whose evaluation is feasible.

Model selection for simulator-based models. In the LFI setting, a critical feature of many cognitive models is that they lack a closed-form solution, but allow forward simulations for a given set of model parameters. A few approaches have made advances in tackling the problem of intractability of these models. For instance, Kleinegesse and Gutmann [2020] and Valentin et al. [2021] proposed a method which combines Bayesian optimal experimental design (BOED) and approximate inference of simulator-based models. The Mutual Information Neural Estimation for Bayesian Experimental Design (MINEBED) method performs BOED by maximizing a lower bound on the expected information gain for a particular experimental design, which is estimated by training a neural network on synthetic data generated by the computational model. By estimating mutual information, the trained neural network no longer needs to model the likelihood directly for selecting designs and doing the Bayesian update. Similarly, Mixed Neural Likelihood Estimation (MNLE) by Boelts et al. [2022] trains neural density estimators on model simulations to emulate the simulator. Pudlo et al. [2016] proposed an LFI approach to model selection, which uses random forests to approximate the marginal likelihood of the models. Despite these advances, these methods have not been designed for model selection in an adaptive experimental design setting. Table 1 summarizes the main differences between modern approaches and the method proposed in this paper.
Cognitive models increasingly operate in an agent-based paradigm [Madsen et al., 2019], where the model is a reinforcement learning (RL) policy [Kaelbling et al., 1996, Sutton and Barto, 2018]. The main problem with these agent-based models is that they need retraining if any of their parameters are altered, which introduces a prohibitive computational overhead when doing model selection. Recently, Moon et al. [2022] proposed a generalized model parameterized by cognitive parameters, which can quickly adapt to multiple behaviours, theoretically bypassing the need for model selection altogether and replacing it with parameter inference. Although the cost of evaluating these models is low in general, they lack the interpretability necessary for cognitive theory development. Therefore, training a parameterized policy within a single RL model family may be preferable: this would still require model selection but would avoid the need for retraining when parameters change (see Section 4.4 for a concrete example).

Amortized approaches to OED. Recently proposed amortized approaches to OED [Blau et al., 2022] - i.e., flexible machine learning models trained upfront on a large set of problems, with the goal of making fast design selection at runtime - allow more efficient selection of experimental designs by introducing an RL policy that generates design proposals. This policy provides a better exploration of the design space, does not require access to a differentiable probabilistic model and can handle both continuous and discrete design spaces, unlike previous amortized approaches [Foster et al., 2021, Ivanova et al., 2021]. These amortized methods are yet to be applied to model selection.
Even though OED is a classical problem in statistics, its application has mostly been relegated to discriminating between simple tractable models. Modern methods such as likelihood-free inference and amortized inference can however make it more feasible to develop OED methods that can work with complex simulator models. In the next sections, we elaborate on our LFI-based method BOSMOS, and demonstrate its working using three classical cognitive science tasks: memory retention, sequential signal detection and risky choice.

---

#### Page 4

|           Reference            | Method  |     LFI      |  Model sel.  |  Par. inf.   |  Amortized   |
| :----------------------------: | :-----: | :----------: | :----------: | :----------: | :----------: |
|        Cavagnaro et al.        |   ADO   |   $\times$   | $\checkmark$ | $\checkmark$ |   $\times$   |
| Kleinegesse and Gutmann [2020] | MINEBED | $\checkmark$ |   $\times$   | $\checkmark$ | $\checkmark$ |
|       Blau et al. [2022]       | RL-BOED | $\checkmark$ |   $\times$   | $\checkmark$ | $\checkmark$ |
|          Moon et al.           |  BOLFI  | $\checkmark$ |   $\times$   | $\checkmark$ |   $\times$   |
|          Pudlo et al.          | RF-ABC  | $\checkmark$ | $\checkmark$ |   $\times$   | $\checkmark$ |
|           This work            | BOSMOS  | $\checkmark$ | $\checkmark$ | $\checkmark$ |   $\times$   |

Table 1: Comparison of experimental design approaches to parameter inference (Par. inf.) and model selection (Model sel.) with the references to the selected representative works. Here, we emphasize LFI methods, as they do not need tractable model likelihoods, and amortized methods since they are the fastest to propose designs. The amortized approaches, however, need to be retrained when the population distributions (i.e. priors over models or parameters) change, as in the setting such as ours where beliefs are updated sequentially as new data are collected.

# 3 Methods

Our method carries out optimal experiment design for model selection and parameter estimation involving three main stages as shown in Figure 1: selecting the experimental design $\boldsymbol{d}$, collecting new data $\boldsymbol{x}$ at the design $\boldsymbol{d}$ chosen from a design space, and, finally, updating current beliefs about the models and their parameters. The process continues until the allocated budget for design iterations $T$ is exhausted, and the preferred cognitive model $m_{\text {est }} \in \mathcal{M}$, which explains the subject behaviour the best, and its parameters $\boldsymbol{\theta}_{\text {est }} \in \Theta_{\text {est }}$ are extracted. While the method is rooted in Bayesian inference and thus builds a full joint posterior over models and parameters, we also consider that ultimately the experimenter may want to report the single 'best' model and parameter setting, and we use this decision-making objective to guide the choices of our algorithm. The definition of what 'best' here means depends on a cost function chosen by the user [Robert et al., 2007]. In this paper, for the sake of simplicity, we choose the most common Bayesian estimator, the maximum a posteriori (MAP), of the full posterior computed by the method:

$$
\begin{aligned}
m_{\text {est }} & =\arg \max _{m} p\left(m \mid \mathcal{D}_{1: t}\right) \\
\boldsymbol{\theta}_{\text {est }} & =\arg \max _{\boldsymbol{\theta}_{m}} p\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{1: t}\right)
\end{aligned}
$$

where $m \in \mathcal{M}, \boldsymbol{\theta}_{m} \in \Theta_{m}$ and $\mathcal{D}_{1: t}=\left(\left(\boldsymbol{d}_{1}, \boldsymbol{x}_{1}\right), \ldots\left(\boldsymbol{d}_{t}, \boldsymbol{x}_{t}\right)\right)$ is a sequence of experimental designs $\boldsymbol{d}$ (e.g. shown stimulus) and the corresponding behavioural data $\boldsymbol{x}$ (e.g. the response of the subject to the stimuli) pairs.

In our usage context, it is important to make a few reasonable assumptions. First, we assume that the prior over the models $p(m)$ and their parameters $p\left(\boldsymbol{\theta}_{m} \mid m\right)$, as well as the domain of the design space, have been specified using sufficient prior knowledge; they may be given by expert psychologists or previous empirical work. This guarantees that the space of the problem is well-defined. Notice that this also implies that the set of candidate models $\mathcal{M}=$ $\left(m_{1}, \ldots, m_{k}\right)$ is known, and each model is defined, for any design, by its own parameters. Second, we assume that the computational models that we consider may not necessarily have a closed-form solution, in case their likelihoods $p\left(\boldsymbol{x} \mid \boldsymbol{d}, \boldsymbol{\theta}_{m}, m\right)$ are intractable, but it is possible to sample from the forward model $m$, given parameter setting $\boldsymbol{\theta}_{m}$, and design $\boldsymbol{d}$. In other words, we operate in a simulator-based inference setting. Please note that this likelihood depends only on the current design and parameters, as assumed in our setting. The third assumption is that each subject's dataset is analyzed separately: we consider single subjects with fixed parameters undergoing the whole set of experiments, as opposed to the statistical setting where information about one dataset may impact the whole population such as, for instance, in hierarchical modelling or pooled models.

As evidenced by Equations (1) and (2), the sequential choice of the designs at any point depends on the current posterior over the models and parameters $p\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t}\right)=p\left(\boldsymbol{\theta}_{m} \mid \mathcal{D}_{1: t}, m\right) \cdot p\left(m \mid \mathcal{D}_{1: t}\right)$, which needs to be approximated and updated at each iteration step of the main loop in Figure 1. This problem can be formulated through sequential importance sampling methods, such as Sequential Monte Carlo (SMC; Del Moral et al., 2006). Thus, the resulting parameter posteriors can be approximated, up to resampling, in the form of equally weighted particle sets: $q_{i}\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t}\right)=\sum_{i=1}^{N_{1}} N_{1}^{-1} \delta_{\boldsymbol{\theta}_{m}^{(i)}, m}$, with $\boldsymbol{\theta}_{m}^{(i)}, m^{(i)}$ the parameters and models associated with the particle $i$, as an approximation of $p\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{1: t}\right)$. These particle sets are later sampled to select designs and update parameter posteriors. In the following sections, we take a closer look at the design selection and belief update stages.

---

#### Page 5

> **Image description.** This image is a flow diagram illustrating a model selection approach. It is divided into three primary panels: Input, Main loop, and Output, connected by arrows indicating the flow of information.
>
> - **Input Panel:** Located at the top, this panel is enclosed in a rounded rectangle. The left side contains a box labeled "Design policy". The right side is divided into two columns, each containing two boxes stacked vertically. The top boxes are labeled "Model 1" and "Model N" respectively, and are highlighted in orange. The boxes below are labeled "probability" and "Prior over parameters". A dashed vertical line separates the "Design policy" box from the model boxes. Arrows labeled "Propose designs" and "Sample priors" point downwards from this panel.
>
> - **Main Loop Panel:** Situated in the middle, this panel is also enclosed in a rounded rectangle. The left side lists three steps labeled I, II, and III: "Select experimental design", "Collect new data", and "Update current beliefs (particle set)". A cylinder shape labeled "Data" is connected to steps I and III by arrows. The right side of this panel contains a table labeled "Particle set" with four rows. The first row is labeled "1" and contains "model 1" and "par. values". The second row is labeled "2" and contains "model 2" and "par. values". The third row contains ellipses. The fourth row is labeled "1000" and contains "model 1" and "par. values".
>
> - **Output Panel:** Located at the bottom, this panel is enclosed in a rounded rectangle. It contains three boxes. The first box is labeled "Model X" and is highlighted in orange. The second box is labeled "Parameter values". The third box contains the text "which is most consistent with". To the right of these boxes is a cylinder shape labeled "Data".
>
> An arrow labeled "Apply decision rule (e.g. MAP, BIC)" points downwards from the Main loop panel to the Output panel.

Figure 1: Components of the model selection approach. Main loop continues until the experimental design budget is depleted. Input panel: the experimenter defines a design policy (e.g. random choice of designs), as well as the models and their parameter priors. Middle panel: (i) the next experimental design is selected based on the design policy and current beliefs about models and their parameters (initially sampled from model and parameter priors); (ii) the experiment is carried out using the chosen design, and the observed response-design pair is stored; (iii) current beliefs are updated (e.g. resampled) based on experimental evidence acquired thus far. Output panel: the model and parameters that are most consistent with the collected data are selected by applying one of the well-established decision rules to the final beliefs about models and their parameters.

---

#### Page 6

# 3.1 Selecting experimental designs

Traditionally, in the experimental design literature, the designs are selected at each iteration $t$ by maximizing the reduction of the expected entropy $H(\cdot)$ of the posterior $p\left(m, \boldsymbol{\theta}_{m} \mid \mathcal{D}_{1: t}\right)$. By definition of conditional probability we have:

$$
\begin{aligned}
\boldsymbol{d}_{t}= & \operatorname{argmin}_{\boldsymbol{d}_{t}} \mathbb{E}_{\boldsymbol{x}_{t} \mid \mathcal{D}_{1: t-1}}\left[H\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t-1} \cup\left(\boldsymbol{d}_{t}, \boldsymbol{x}_{t}\right)\right)\right] \\
= & \operatorname{argmin}_{\boldsymbol{d}_{t}} \mathbb{E}_{\boldsymbol{x}_{t} \mid \mathcal{D}_{1: t-1}}\left[\mathbb{E}_{p\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t}\right)}\left[-\log p\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t} \cup\left(\boldsymbol{d}_{t}, \boldsymbol{x}_{t}\right)\right)\right]\right] \\
= & \operatorname{argmin}_{\boldsymbol{d}_{t}} \mathbb{E}_{\boldsymbol{x}_{t} \mid \mathcal{D}_{1: t-1}} \mathbb{E}_{p\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t-1}\right)}\left[-\log \left(p\left(\boldsymbol{x}_{t} \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}, m\right)\right)\right] \\
& +\mathbb{E}_{\boldsymbol{x}_{t} \mid \mathcal{D}_{1: t-1}} \log p\left(\boldsymbol{x}_{t} \mid \boldsymbol{d}_{t}, \mathcal{D}_{1: t-1}\right)
\end{aligned}
$$

where $\boldsymbol{x}_{t}$ is the response predicted by the model. The first equality comes from the definition of entropy and the second from Bayes rule, where we removed the prior, as this term is a constant term in $\boldsymbol{d}_{t}$. Here, lower entropy corresponds to a narrower, more concentrated, posterior - with maximal information about models and parameters.
Since neither $p\left(\boldsymbol{x}_{t} \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}, m\right)$ nor, by extension, Equation (4) are tractable in our setting, we propose a simulatorbased utility objective

$$
\boldsymbol{d}_{t}=\arg \min _{\boldsymbol{d}_{t}} \mathbb{E}_{q_{t}\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t-1}\right)}\left[\hat{H}\left(\boldsymbol{x}_{t}^{\prime} \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}, m\right)\right]-\hat{H}\left(\boldsymbol{x}_{t} \mid D_{1: t-1}, \boldsymbol{d}_{t}\right)
$$

where $q_{t}$ is a particle approximation of the posterior at time $t$, and $\hat{H}$ is a kernel-based Monte Carlo approximation of the entropy $H$.
The intuition behind this utility objective is that we choose such designs $\boldsymbol{d}_{t}$ that would maximize identifiability (minimize the entropy) between $N$ responses $\boldsymbol{x}^{\prime}$ simulated from different computational models $p\left(\cdot \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}, m\right)$. The models $m$ as well as their parameters $\boldsymbol{\theta}_{m}$ are sampled from the current beliefs $q_{t}\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{1: t-1}\right)$. The full asymptotic validity of the Monte Carlo approximation of the decision rule in Equation (5) can be found in Appendix A.
The utility objective in (5) allows us to use Bayesian Optimization (BO) to find the design $\boldsymbol{d}_{t}$ and then run the experiment with the selected design. In the next section, we discuss how to update beliefs about the models $m$ and their parameters $\boldsymbol{\theta}_{m}$ based on the data collected from the experiment.

### 3.2 Likelihood-free posterior updates

The response $\boldsymbol{x}_{t}$ from the experiment with the design $\boldsymbol{d}_{t}$ is used to update approximations of the posterior $q_{t}\left(m \mid \mathcal{D}_{t}\right)$ and $q_{t}\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{t}\right)$, obtained via marginalization and conditioning, respectively, from $q_{t}\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{t}\right)$. We use LFI with synthetic responses $\boldsymbol{x}_{\boldsymbol{\theta}_{m}}$ simulated by the behavioural model $m$ to perform the approximate Bayesian update.

Parameter estimation conditioned on the model. We start with parameter estimation for each of the candidate models using Bayesian Optimization for Likelihood-Free Inference (BOLFI; Gutmann and Corander, 2016). In BOLFI, a Gaussian process (GP) [Rasmussen, 2003] surrogate for the discrepancy function between the observed and simulated data, $\rho\left(\boldsymbol{x}_{\boldsymbol{\theta}_{m}}, \boldsymbol{x}_{t}\right)$ (e.g., Euclidean distance), serves as a base to an unnormalized approximation of the intractable likelihood $p\left(\boldsymbol{x}_{t} \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}, m\right)$. Thus, the posterior can be approximated through the following approximation of the likelihood function $\mathcal{L}_{\epsilon_{m}}(\cdot)$ and the prior over model parameters $p\left(\boldsymbol{\theta}_{m}\right)$ :

$$
\begin{aligned}
p\left(\boldsymbol{\theta}_{m} \mid \boldsymbol{x}_{t}\right) & \propto \mathcal{L}_{\epsilon_{m}}\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m}\right) \cdot p\left(\boldsymbol{\theta}_{m}\right) \\
\mathcal{L}_{\epsilon_{m}}\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m}\right) & \approx \mathbb{E}_{\boldsymbol{x}_{\boldsymbol{\theta}_{m}}}\left[\kappa_{\epsilon_{m}}\left(\rho_{m}\left(\boldsymbol{x}_{\boldsymbol{\theta}_{m}}, \boldsymbol{x}_{t}\right)\right)\right]
\end{aligned}
$$

Here, following Section 6.3 of [Gutmann and Corander, 2016], we choose $\kappa_{\epsilon_{m}}(\cdot)=\mathbf{1}_{\left[0, \epsilon_{m}\right]}(\cdot)$, where the bandwidth $\epsilon_{m}$ takes the role of a acceptance/rejection threshold. Using a Gaussian likelihood for the GP, this leads to: $\mathbb{E}_{\boldsymbol{x}_{\boldsymbol{\theta}_{m}}}\left[\kappa_{\epsilon_{m}}\left(\rho\left(\boldsymbol{x}_{\boldsymbol{\theta}_{m}}, \boldsymbol{x}_{t}\right)\right)\right]=\Phi\left(\left(\epsilon_{m}-\mu\left(\boldsymbol{\theta}_{m}\right)\right) / \sqrt{\nu\left(\boldsymbol{\theta}_{m}\right)+\sigma^{2}}\right)$, where $\Phi(\cdot)$ denotes the standard Gaussian cumulative distribution function (cdf). Note that $\mu\left(\boldsymbol{\theta}_{m}\right)$ and $\nu\left(\boldsymbol{\theta}_{m}\right)+\sigma^{2}$ are the posterior predictive mean and variance of the GP surrogate at $\boldsymbol{\theta}_{m}$.

Model estimation. A principled way of performing model selection is via the marginal likelihood, that is $p\left(\boldsymbol{x}_{t} \mid\right.$ $m)=\int p\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m}, m\right) \cdot p\left(\boldsymbol{\theta}_{m} \mid m\right) \mathrm{d} \boldsymbol{\theta}_{m}$, which is proportional to the posterior over models assuming an equal prior for each model. Unfortunately, a direct computation of the marginal likelihood is not possible with Equation (7), since it only allows us to compute a likelihood approximation up to a scaling factor that implicitly depends on $\epsilon$. For instance, when calculating a Bayes factor (ratio of marginal likelihoods) for models $m_{1}$ and $m_{2}$

$$
\frac{p\left(\boldsymbol{x}_{t} \mid m_{1}\right)}{p\left(\boldsymbol{x}_{t} \mid m_{2}\right)}=\frac{\mathbb{E}_{\boldsymbol{\theta}_{m 1}}\left[p\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m 1}, m_{1}\right)\right]}{\mathbb{E}_{\boldsymbol{\theta}_{m 2}}\left[p\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m 2}, m_{2}\right)\right]} \neq \frac{\mathbb{E}_{\boldsymbol{\theta}_{m 1}}\left[\mathcal{L}_{\epsilon_{m 1}}\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m 1}\right)\right]}{\mathbb{E}_{\boldsymbol{\theta}_{m 2}}\left[\mathcal{L}_{\epsilon_{m 2}}\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m 2}\right)\right]}
$$

---

#### Page 7

their respective $\epsilon_{m 1}$ and $\epsilon_{m 2}$, chosen independently, may potentially bias the marginal likelihood ratio in favour of one of the models, rendering it unsuitable for model selection. Choosing the same $\epsilon$ for each model is not possible either, as it would lead to numerical instability due to the shape of the kernel.
To approximate the marginal likelihood $p\left(\boldsymbol{x}_{t} \mid m\right)$, we adopt a similar approach as in Equation (7), by reframing the marginal likelihood computation as a distinct LFI problem. In ABC for parameter estimation, we would generate pseudo-observations from the prior predictive distribution of each model, and compare the discrepancy with the true observations on a scale common to all models. This comparison involves a kernel that maps the discrepancy into a likelihood approximation. For example, in rejection ABC [Tavaré et al., 1997, Marin et al., 2012] this kernel is uniform. In our case, we will generate samples from the joint prior predictive distribution on both models and parameters, and we use a Gaussian kernel $\kappa_{\eta}(\cdot)=\mathcal{N}\left(\cdot \mid 0, \eta^{2}\right)$, chosen to satisfy all of the requirements from Gutmann and Corander [2016]; in particular, this kernel is non-negative, non-concave and has a maximum at 0 . The parameter $\eta>0$ serves as the kernel bandwidth, similarly to $\epsilon_{m}$ in Equation (7). The value of $\kappa_{\eta}(\cdot)$ monotonically increases as the model $m$ produces smaller discrepancy values. This kernel leads to the following approximation of the marginal likelihood:

$$
\mathcal{L}\left(\boldsymbol{x}_{t} \mid m, \mathcal{D}_{t-1}\right) \propto \mathbb{E}_{\boldsymbol{x}_{\boldsymbol{\theta}} \sim p\left(\cdot \mid \boldsymbol{\theta}_{m}, m\right) \cdot q\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{t-1}\right)} \kappa_{\eta}\left(\hat{\rho}\left(\boldsymbol{x}_{\boldsymbol{\theta}}, \boldsymbol{x}_{t}\right)\right)
$$

where $\kappa_{\eta}(\cdot)=\mathcal{N}\left(\cdot \mid 0, \eta^{2}\right)$, and $\hat{\rho}$ is the GP surrogate for the discrepancy. Eq. 9 is a direct equivalent of Eq. 7, but here we integrate (marginalize) over both $\theta$ and $x_{\theta}$. Here we used the Gaussian kernel, instead of the uniform kernel used in Eq. 7, as it produced better results for model selection in preliminary numerical experiments. Note that in Eq. 9 we have two approximations, the first one from $\kappa_{\eta}$, stating that the likelihood is approximated from the discrepancy, and the second from the use of a GP surrogate for the discrepancy.
The choice of $\eta$ is a complex problem, and in this paper we propose the simple solution of setting $\eta$ as the minimum value of $\mathbb{E}_{\boldsymbol{x}_{\boldsymbol{\theta}} \sim p\left(\cdot \mid \boldsymbol{\theta}_{m}, m\right) \cdot q\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{t-1}\right)} \hat{\rho}\left(\boldsymbol{x}_{\boldsymbol{\theta}}, \boldsymbol{x}_{t}\right)$ across all models $m \in \mathcal{M}$. This value has the advantage of giving non extreme values to the estimations of the marginal likelihood, which should in principle avoid over confidence.

Posterior update. The resulting marginal likelihood approximation in Equation (9) can then be used in posterior updates for new design trials as follows:

$$
\begin{gathered}
q\left(m \mid \mathcal{D}_{t}\right) \propto \mathcal{L}\left(\boldsymbol{x}_{t} \mid m, \mathcal{D}_{t-1}\right) \cdot q\left(m \mid \mathcal{D}_{t-1}\right) \approx \kappa_{\eta}\left(\omega_{m}\right) \cdot q\left(m \mid \mathcal{D}_{t-1}\right) \\
q\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{t}\right) \propto \mathcal{L}_{\epsilon_{m}}\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m}, m\right) \cdot q\left(\boldsymbol{\theta}_{m} \mid \mathcal{D}_{t-1}, m\right)
\end{gathered}
$$

Which is equivalent to:

$$
q\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{t}\right) \propto \mathcal{L}_{\epsilon_{m}}\left(\boldsymbol{x}_{t} \mid \boldsymbol{\theta}_{m}, m\right) \cdot \mathcal{L}\left(\boldsymbol{x}_{t} \mid m, \mathcal{D}_{t-1}\right) \cdot q\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{t-1}\right)
$$

Once we updated the joint posterior of models and parameters, it is straightforward to obtain the model and parameter posterior through marginalization and apply a decision rule (e.g. MAP) to choose the estimate. The entire algorithm for BOSMOS can be found in Appendix B.

# 4 Experiments

In the experiments, our goal was to evaluate how well the proposed method described in Section 3 discriminated between different computational models in a series of cognitive tasks: memory retention, signal detection and risky choice. Specifically, we measured how well the method chooses designs which help the estimated model imitate the behaviour of the target model, discriminate between models, and correctly estimate their ground-truth parameters. In our simulated experimental setup, we created 100 synthetic participants by sampling the ground-truth model and its parameters (not available in the real world) through priors $p(m)$ and $p\left(\boldsymbol{\theta}_{m} \mid m\right)$. Then, we ran the sequential experimental design procedure for a range of methods described in Section 4.1, and recorded four main performance metrics shown in Figure 3 for 20 design trials (results analysed further later in the section): the behavioural fitness error $\eta_{\mathrm{b}}$, defined below, the parameter estimation error $\eta_{\mathrm{p}}$, the accuracy of the model prediction $\eta_{\mathrm{m}}$ and the empirical time cost of running the methods. Furthermore, we evaluated the methods at different stages of design iterations in Figure 3 for the convergence analysis. The complete experiments with additional evaluation points can be found in Appendix C.
We compute $\eta_{\mathrm{b}}, \eta_{\mathrm{p}}$ and $\eta_{\mathrm{m}}$ for a single synthetic participant using the known ground truth model $m_{\text {true }}$ and parameters $\boldsymbol{\theta}_{\text {true }}$. The behavioural fitness error $\eta_{\mathrm{b}}=\left\|\boldsymbol{X}_{\text {true }}-\boldsymbol{X}_{\text {est }}\right\|^{2}$ is calculated as the Euclidean distance between the groundtruth model ( $\boldsymbol{X}_{\text {true }}$ ) and synthetic ( $\boldsymbol{X}_{\text {est }}$ ) behavioural datasets, which consist of means $\mu(\cdot)$ of 100 responses evaluated

---

#### Page 8

> **Image description.** This image contains a set of bar and error bar plots comparing the performance of different methods across four cognitive modeling tasks. The plots are arranged in a 4x4 grid.
>
> Each column represents a different cognitive modeling task: "Demonstrative example", "Memory retention", "Signal detection", and "Risky choice". Each row represents a different performance metric: behavioral fitness error (ηb), parameter estimation error (ηp), model predictive accuracy (ηm), and empirical time cost (tlog).
>
> Within each subplot:
>
> - **Error Bar Plots (Rows 1 & 2):** The first two rows contain error bar plots. The x-axis represents the error value. Different methods are represented by different colored markers with error bars: ADO (blue), MINEBED (green), BOSMOS (red), LBIRD (cyan), and Prior (magenta). The error bars indicate the standard deviation.
> - **Bar Plots (Rows 3 & 4):** The last two rows contain bar plots. The x-axis represents the value of the metric. The y-axis is implicit, with each bar corresponding to a different method (ADO, MINEBED, BOSMOS, LBIRD, Prior). The bars are colored according to the method they represent, using the same color scheme as the error bar plots.
>
> The overall visual impression is a comparison of the performance of different methods across different cognitive tasks and metrics. The plots suggest that BOSMOS (red) generally performs well compared to the other methods.

Figure 2: An overview of the performance of the methods, compared with the prior predictive with random design, (rows) after 20 design trials across four different cognitive modelling tasks (columns): demonstrative example, memory retention, signal detection and risky choice. While requiring 10 times fewer simulations and 60-100 times less time, the proposed BOSMOS method (red) shows consistent improvement over the alternative LFI method, MINEBED (green), in terms of behavioural fitness error $\eta_{b}$, parameter estimation error $\eta_{p}$, model predictive accuracy $\eta_{\text {m }}$ and empirical time cost $t_{\text {log }}$ (here, for 100 designs, in minutes on a log scale). The model accuracy bars indicate the proportion of correct prediction of models across 100 simulated participants. The error bars show the mean (marker) and std. (caps) of the error by the respective methods.
at the same 100 random designs $\mathcal{T}$ generated from a proposal distributions $p(\boldsymbol{d})$, defined for each model:

$$
\begin{aligned}
\mathcal{T} & =\left\{\boldsymbol{d}_{i} \sim p(\boldsymbol{d})\right\}_{i=1}^{100} \\
\boldsymbol{X}_{\text {true }} & =\left\{\mu\left(\left\{\boldsymbol{x}_{s}: \boldsymbol{x} \sim p(\cdot \mid \boldsymbol{d}_{i}, \boldsymbol{\theta}_{\text {true }}, m_{\text {true }}\right)\right\}_{s=1}^{100}\right): \boldsymbol{d}_{i} \in \mathcal{T}\}_{i=1}^{100} \\
\boldsymbol{X}_{\text {est }} & =\left\{\mu\left(\left\{\boldsymbol{x}_{s}: \boldsymbol{x} \sim p(\cdot \mid \boldsymbol{d}_{i}, \boldsymbol{\theta}_{\text {est }}, m_{\text {est }}\right)\right\}_{s=1}^{100}\right): \boldsymbol{d}_{i} \in \mathcal{T}\}_{i=1}^{100}
\end{aligned}
$$

Here, $m_{\text {est }}$ and $\boldsymbol{\theta}_{\text {est }}$ are, respectively, the model and parameter values estimated via the MAP rule (unless specified otherwise). $m_{\text {est }}$ is also used to calculate the predictive model accuracy $\eta_{\mathrm{m}}$ as a proportion of correct model predictions for the total number of synthetic-participants, while $\boldsymbol{\theta}_{\text {est }}$ is used to calculate the averaged Euclidean distance $\| \boldsymbol{\theta}_{\text {true }}-$ $\boldsymbol{\theta}_{\text {est }} \|^{2}$ across all synthetic participants, which constitutes the parameter estimation error $\eta_{\mathrm{p}}$.

# 4.1 Comparison methods

Throughout the experiments, we compare several strategies for experimental design selection and parameter inference, where prior predictive distribution (evaluation of the prior without any collected data) with random design choice from the proposal distribution of each model is used as a baseline (we call this method results Prior in the results). The explanations of these methodologies, as well as the exact setup parameters, are provided below.

### 4.1.1 Likelihood-based inference with random design

"Likelihood-based" inference with random design (LBIRD) applies the ground-truth likelihood, where it is possible, to conduct Bayesian inference and samples the design from the proposal distribution $p(\boldsymbol{d})$ instead of design selection:

---

#### Page 9

> **Image description.** This image contains a set of plots comparing the performance of three different methods (ADO, MINEBED, and BOSMOS) across four cognitive tasks: "Demonstrative example", "Memory retention", "Signal detection", and "Risky choice". The plots are arranged in a 3x4 grid. Each column corresponds to one of the cognitive tasks, and each row corresponds to a different performance measure: $\eta_b$, $\eta_p$, and $\eta_m$.
>
> The first two rows contain line plots with error bars. The x-axis represents the number of design trials (1 tr., 4 tr., 20 tr.). The y-axis represents the performance measure. The three methods are represented by different colored lines: ADO (blue, dashed), MINEBED (green, dashed), and BOSMOS (red, solid). The error bars appear to be in the same color as the corresponding line.
>
> The third row contains bar charts. The x-axis represents the number of design trials (1 tr., 4 tr., 20 tr.). The y-axis represents the performance measure $\eta_m$. The three methods are represented by different colored bars: ADO (blue), MINEBED (green), and BOSMOS (red). The bars for each trial number are grouped together.

Figure 3: Evaluation of three performance measures (rows) after 1, 4 and 20 design trials for BOSMOS (solid red) and two alternative best methods, ADO (blue) and MINEBED (green), in four cognitive tasks (columns). As the number of design trials grows, the methods accumulate more observed data from subjects' behaviour and, hence, should reduce behavioural fitness error $\eta_{\mathrm{b}}$, parameter estimation error $\eta_{\mathrm{p}}$, and increase model predictive accuracy $\eta_{\mathrm{m}}$. Since $\eta_{\mathrm{b}}$ is the performance metric MINBED and BOSMOS optimize, its convergence is the most prominent. The lack of convergence for the other two metrics in the memory retention and signal detection tasks is likely due to the possibility of the same behavioural data being produced by models and parameters that are different from the ground-truth (i.e., non-identifiability of these models).
$\mathcal{D}_{t}=\left(x_{t}, \boldsymbol{d}_{t}\right), x_{t} \sim \pi\left(\cdot \mid \boldsymbol{\theta}, m, \boldsymbol{d}_{t}\right), \boldsymbol{d}_{t} \sim p(\cdot)$. This procedure serves as a baseline by providing unbiased estimates of models and parameters. As other methods in this section, LBIRD uses 5000 particles (empirical samples) to approximate the joint posterior of models and parameters for each model. The Bayesian updates are conducted through importance-weighted sampling with added Gaussian noise applied to the current belief distribution.

# 4.1.2 ADO

ADO requires a tractable likelihood of the models, and hence is used as an upper bound of performance in cases where the likelihood is available. ADO [Cavagnaro et al., 2010] employs BO for the mutual information utility objective:

$$
U(\boldsymbol{d})=\sum_{m=1}^{K} p(m) \sum_{y} p(\boldsymbol{x} \mid m, \boldsymbol{d}) \cdot \log \left(\frac{p(\boldsymbol{x} \mid m, \boldsymbol{d})}{\sum_{m=1}^{K} p(m) p(\boldsymbol{x} \mid m, \boldsymbol{d})}\right)
$$

where we used 500 parameters sampled from the current beliefs to integrate

$$
p(\boldsymbol{x} \mid m, \boldsymbol{d})=\int p\left(\boldsymbol{x} \mid \boldsymbol{\theta}_{m}, m, \boldsymbol{d}\right) \cdot p\left(\boldsymbol{\theta}_{m} \mid m\right) \mathrm{d} \boldsymbol{\theta}
$$

Similarly to other approaches below which also use BO, the BO procedure is initialized with 10 evaluations of the utility objective with $\boldsymbol{d}$ sampled from the design proposal distribution $p(\boldsymbol{d})$, while the next 5 design locations are determined by the Monte-Carlo-based noisy expected improvement objective. The GP surrogate for the utility uses a constant mean function, a Gaussian likelihood and the Matern kernel with zero mean and unit variance. All these components of the design selection procedure were implemented using the BOTorch package [Balandat et al., 2020].

### 4.1.3 MINEBED

MINEBED [Kleinegesse and Gutmann, 2020] focuses on design selection for parameter inference with a single model. Since our setting requires model selections and by extension working with multiple models, we compensate for that by having a separate MINEBED instance for each of the models and then assigning a single model (sampled from the

---

#### Page 10

current beliefs) for design optimization at each trial. The model is assigned by the MAP rule over the current beliefs about models $q\left(m \mid \mathcal{D}_{1: t}\right)$, and the data from conducting the experiment with the selected design are used to update all MINEBED instances. We used the original implementation of the MINEBED method by Kleinegesse and Gutmann [2020], which uses a neural surrogate for mutual information consisting of two fully connected layers with 64 neurons. This configuration was optimized using Adam optimizer [Kingma and Ba, 2014] with initial learning rate of 0.001 , 5000 simulations per training at each new design trial and 5000 epochs.

# 4.1.4 BOSMOS

BOSMOS is the method proposed in this paper and described in Section 3. It uses the simulator-based utility objective from Equation (5) in BO to select the design and BO for LFI, along with the marginal likelihood approximation from Equation (9) to conduct inference. The objective for design selection is calculated with the same 10 models (a higher number increases belief representation at the cost of more computations) sampled from the current belief over models (i.e. particle set $q_{t}\left(m \mid \mathcal{D}_{1: t}\right)$ at each time $t$ ), where each model is simulated 10 times to get one evaluation point of the utility ( 100 simulations per point). In total, in each iteration, we spent 1500 simulations to select the design and additional 100 simulations to conduct parameter inference.
As for parameter inference in BOSMOS, BO was initialized with 50 parameter points randomly sampled from the current beliefs about model parameters (i.e. the particle set $q_{t}\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{1: t}\right)$ ), the other 50 points were selected for simulation in batches of 5 through the Lower Confidence Bound Selection Criteria [Srinivas et al., 2009] acquisition function. Once again, a GP is used as a surrogate, with the constant mean function and the radial basis function [Seeger, 2004] kernel with zero mean and unit variance. Once the simulation budget of 100 is exhausted, the parameter posterior is extracted through an importance-weight sampling procedure, where the GP surrogate with the tolerance threshold set at a minimum of the GP mean function [Gutmann and Corander, 2016] acts as a base for the simulator parameter likelihood.

### 4.2 Demonstrative example

The demonstrative example serves to highlight the significance of design optimization for model selection with a simple toy scenario. We consider two normal distribution models with either positive (PM) or negative (NM) mean. Responses are produced according to the experimental design $d \in[0.001,5]$ which determines the quantity of observational noise variance:

$$
\begin{array}{ll}
(\mathrm{PM}) & x \sim \mathcal{N}\left(\theta_{\mu}, d^{2}\right) \\
(\mathrm{NM}) & x \sim \mathcal{N}\left(-\theta_{\mu}, d^{2}\right)
\end{array}
$$

These two models have the same prior over parameters $\theta_{\mu} \in[0,5]$ and may be clearly distinguished when the optimal design value is $d=0.001$. We choose a uniform prior over models.

### 4.2.1 Results

As shown in the first set of analyses in Figure 2, selecting informative designs can be crucial. When compared to the LBIRD method, which picked designs at random, all the design optimization approaches performed exceedingly well. This highlights the significance of design selection, as random designs produce uninformative results and impede the inference procedure.
Figure 3 illustrates the convergence of the key performance measures, demonstrating that the design optimization methods had nearly perfect estimates of ground-truths after only one design trial. This indicates that the PM and NM models are easily separable, provided informative designs. In terms of the model predictive accuracy, MINEBED outperformed BOSMOS after the first trial, however BOSMOS rapidly caught up as trials proceeded. This is most likely because our technique employs fewer simulations per trial but a more efficient LFI surrogate than MINEBED. As a result, our method has the second-best time cost not only for the demonstrative example but also across all cognitive tasks. The only method that was faster is the LBIRD method, which skips the design optimization procedure entirely and avoids lengthy computations related to LFI by accessing the ground-truth likelihood.

### 4.3 Memory retention

Studies of memory are a fundamental research area in experimental psychology. Memory can be viewed functionally as a capability to encode, store and remember, and neurologically as a collection of neural connections [Amin and Malik, 2013]. Studies of memory retention have a long history in psychological research, in particular in relation to the shape of the retention function [Rubin and Wenzel, 1996]. These studies on functional forms of memory retention

---

#### Page 11

seek to quantitatively answer how long a learned skill or material is available [Rubin et al., 1999], or how quickly it is forgotten. Distinguishing retention functions may be a challenge [Rubin et al., 1999], and Cavagnaro et al. [2010] showed that employing an ADO approach can be advantageous. Specifically, studies of memory retention typically consist of a 'study phase' (for memorizing) followed by a 'test phase' (for recalling), and the time interval between the two is called a 'lag time'. Varying the lag time by means of ADO allowed more efficient differentiation of the candidate models [Cavagnaro et al., 2010]. To demonstrate our approach with the classic memory retention task, we consider the case of distinguishing two functional forms, or models, of memory retention, defined as follows.

Power and exponential models of memory retention. In the classic memory retention task, the subject recalls a stimulus (e.g. a word) at a time $d \in[0,100]$, which is modelled by two Bernoulli models $B(1, p)$ : the power (POW) and exponential (EXP) models. The samples from these models are the responses to the task $x$, which can be interpreted as 'stimulus forgotten' in case $x=0$ and $x=1$ otherwise. We follow the definition of these models by Cavagnaro et al. [2010], where $p=\theta_{a} \cdot(d+1)^{-\theta_{\text {POW }}}$ in POW and $p=\theta_{a} \cdot e^{-\theta_{\text {EXP }}-d}$ in EXP, as well as the same priors:

$$
\begin{aligned}
\theta_{a} & \sim \operatorname{Beta}(2,1) \\
\theta_{\mathrm{POW}} & \sim \operatorname{Beta}(1,4) \\
\theta_{\mathrm{EXP}} & \sim \operatorname{Beta}(1,8)
\end{aligned}
$$

Similarly to the previous demonstrative example and the rest of the experiments, we use equal prior probabilities for the models.

# 4.3.1 Results

Studies on the memory task show that the performance gap between LFI approaches and methods that use groundtruth likelihood grows as the number of design trials increases (Figure 2). This is expected, since doing LFI introduces an approximation error, which becomes more difficult to decrease when the most uncertainty around the models and their parameters has been already removed by previous trials. Unlike in the demonstrative example, where design selection was critical, the ground-truth likelihood appears to have a larger influence than design selection for this task, as evidenced by the similar performance of the LBIRD and ADO approaches.
In regard to LFI techniques, BOSMOS outperforms MINEBED in terms of behavioural fitness and parameter estimation, as shown in Figure 3, but only marginally better for model selection. Moreover, both approaches seem to converge to the wrong solutions (unlike ADO), as evidenced by their lack of convergence in the parameter estimation and model accuracy plots. Interestingly, both techniques continued improving behavioural fitness, implying that behavioural data of the models can be reproduced by several parameters that are different from the ground-truth, and LFI methods fail to distinguish them. A deeper examination of the parameter posterior can reveal this issue, which can be likely alleviated by adding new features for observations and designs that can assist in capturing the intricacies within the behavioural data.

### 4.4 Sequential signal detection

Signal detection theory (SDT) focuses on perceptual uncertainty, presenting a framework for studying decisions under such ambiguity [Tanner and Swets, 1954, Peterson et al., 1954, Swets et al., 1961, Wickens, 2002]. SDT is an influential developing model stemming from mathematical psychology and psychophysics, providing an analytical framework for assessing optimal decision-making in the presence of ambiguous and noisy signals. The origins of SDT can be traced to the 1800s, but its modern form emerged in the latter half of the 20th century, with the realization that sensory noise is consciously accessible [Wixted, 2020]. Example of a signal detection task could be a doctor making a diagnosis: they have to make a decision based on a (noisy) signal of different symptoms [Wickens, 2002]. SDT is largely considered a normative approach, assuming that a decision-maker is bounded rational [Swets et al., 1961]. We will consider a sequential signal detection task and two models, Proximal Policy Optimization (PPO) and Probability Ratio (PR), implemented as follows.

SDT. In the signal detection task, the subject needs to correctly discriminate the presence of the signal $o_{\text {sign }} \in$ $\{$ present, absent $\}$ in a sensory input $o_{\text {in }} \in \mathbb{R}$. The sensory input is corrupted with sensory noise $\sigma_{\text {sens }} \in \mathbb{R}$ :

$$
o_{\text {in }}=1_{\text {present }}\left(o_{\text {sign }}\right) \cdot d_{\text {str }}+\gamma, \quad \gamma \propto \mathcal{N}\left(0, \sigma_{\text {sens }}\right)
$$

Due to the noise in the observations, the task may require several consecutive actions to finish. At every time-step, the subject has three actions $a \in\{$ present, absent, look $\}$ at their disposal: to make a decision that the signal is present or absent, and to take another look at the signal. The role of the experimenter is to adjust the signal strength $d_{\text {str }} \sim$ $\operatorname{Unif}(0,4)$ and discrete number of observations $d_{\text {obs }} \sim \operatorname{Unif}_{\text {discr }}(2,10)$ the subject can make such that the experiment

---

#### Page 12

will reveal characteristics of human behaviour. In particular, our goal is to identify the hit value parameter of the subject, which determines how much reward $r(a, s)$ the subject receives, in case the signal is both present and identified correctly. Hence, we have that

$$
r(a, s)=r_{a}(s)+r_{\text {step }}
$$

$$
\begin{aligned}
& r_{a}(s)=\theta_{\text {hit }} \\
& r_{a}(s)=2 \\
& r_{a}(s)=0 \\
& r_{a}(s)=-1
\end{aligned}
$$

when the signal is present, and the action is present.
when the signal is absent, and the action is absent.
when the action is look.
in other cases.
where $r_{\text {step }}=-0.05$ is the constant cost of every consecutive action.
PPO. We implement the SDT task as an RL model due to the sequential nature of the task. In particular, the look action will postpone the signal detection decision to the next observation. The model assumes that the subject acts according to the current observation $o_{\text {in }}$ and an internal state $\beta: \pi\left(a \mid o_{\text {in }}, \beta\right)$. The internal state $\beta$ is updated over trials by aggregating observations $o_{\text {in }}$ using a Kalman Filter, and after each trial, the agent chooses a new action. As we have briefly discussed in Section 2, the RL policies need to be retrained when their parameters change. To address this issue, the policy was parameterized and trained using a wide range of model parameters as policy inputs. The resulting model was implemented using the PPO algorithm [Schulman et al., 2017].

PR. An alternative to the RL model is a PR model. It also assumes sequential observations: a hypothesis test as to whether the signal is present is performed after every observation, and the sequence of observations is called evidence [Griffith et al., 2021]. A likelihood for the evidence (sequence of observations) is the product of likelihoods of each observation. A likelihood ratio is used as a decision variable (denoted $f_{t}$ here). Specifically, $f_{t}$ is evaluated against a threshold, which determines which action $a_{t}$ to take as follows:

$$
\begin{aligned}
& a_{t}=\text { present, } \\
& \text { if } f_{t} \leq \theta_{\text {low }} \\
& a_{t}=\text { absent, } \\
& \text { if } f_{t} \geq \theta_{\text {low }}+\theta_{\text {len }} \\
& a_{t}=\text { look, } \\
& \text { if } \theta_{\text {low }} \leq f_{t} \leq \theta_{\text {low }}+\theta_{\text {len }}
\end{aligned}
$$

where

$$
\begin{gathered}
f_{t}=\prod_{i=1}^{d_{\text {dir }}} \frac{\omega_{1}}{\omega_{2}}, \quad \omega_{1} \sim \mathcal{N}_{\mathrm{CDF}}\left(\frac{1}{\theta_{\text {hit }}-1} ; d_{\text {sir }}, \theta_{\text {sens }}\right), \\
\omega_{2} \sim \mathcal{N}_{\mathrm{CDF}}\left(\frac{1}{\theta_{\text {hit }}-1} ; 0, \theta_{\text {sens }}\right) .
\end{gathered}
$$

Here, $\mathcal{N}_{\text {CDF }}(: ; \mu, \nu)$ is the Gaussian cumulative distribution function (CDF) with the mean $\mu$ and standard deviation $\nu$. For more information about the PR model, we refer the reader to Griffith et al. [2021].
For both models, we used the following priors for their parameters and design values:

$$
\begin{array}{ll}
\theta_{\text {sens }} \sim \operatorname{Unif}(0.1,1), & \theta_{\text {hit }} \sim \operatorname{Unif}(1,7) \\
\theta_{\text {low }} \sim \operatorname{Unif}(0,5), & \theta_{\text {len }} \sim \operatorname{Unif}(0,5) .
\end{array}
$$

# 4.4.1 Results

BOSMOS and MINEBED are the only methodologies capable of performing model selection in sequential signal detection models, as specified in Section 4.4, due to the intractability of their likelihoods. The experimental conditions are therefore very close to those in which these LFI approaches are usually applied, with the exception that we now know the ground-truth of synthetic participants for performance assessments.
BOSMOS showed a faster convergence of the estimates than MINEBED requiring only 4 design trails to reduce the majority of the uncertainty associated with model prediction accuracy and behaviour fitness error, as demonstrated in Figure 3. In contrast, it took 20 design trials for MINEBED to converge, and extending it beyond 20 trials provided very little benefit. Similarly as in the memory retention task from Section 4.3, error in BOSMOS parameter estimates

---

#### Page 13

did not converge to zero, showing difficulty in predicting model parameters for PPO and PR models. Improving parameter inference may require modifying priors to encourage more diverse behaviours and selecting more descriptive experimental responses. Finally, BOSMOS outperformed MINEBED across all performance metrics after only one design trial, with the model predictive accuracy showing a large difference, establishing BOSMOS as a clear favourite approach for this task.

An example of posterior distributions returned by BOSMOS is demonstrated in Figure 4. Despite overall positive results, there are occasional cases in a population of synthetic participants, where BOSMOS fails to converge to the ground-truth. The same problem can be observed with MINEBED, as demonstrated in Appendix D. These findings may be attributed to poor identifiability of the signal detection models, suggested earlier in the memory task, but also due to the approximation inaccuracies accumulated over numerous trials. Since both methods operate in a LFI setting, some inconsistency between replicating the target behaviour and converging to the ground-truth parameters is to be expected when the models are poorly identifiable.

# 4.5 Risky choice

Risky choice problems are typical tasks used in psychology, cognitive science and economics to study attitudes towards uncertainty. Specifically, risk refers to 'quantifiable' uncertainty, where a decision-maker is aware of probabilities associated with different outcomes [Knight, 1985]. In risky choice problems, individuals are presented with options that are lotteries (i.e., probability distributions of outcomes). For example, a risky choice problem could be a decision between winning 100 euros with a chance of $25 \%$, or getting 25 euros with a chance of $99 \%$. The choice is between two lotteries $(100,0.25 ; 0,0.75)$ and $(25,0.99 ; 0,0.01)$. The goal of the participant is to maximize the subjective reward of their single choice, so they need to assess the risk associated with outcomes in each lottery.
Several models have been proposed to explain tendencies in these tasks, including normative approaches derived from logic to descriptive approaches based on empirical findings [Johnson and Busemeyer, 2010]. In this paper, we will consider four classic models (following Cavagnaro et al., 2013b): expected utility theory (EU) [Von Neumann and Morgenstern, 1990], weighted expected utility theory (WEU) [Hong, 1983], original prospect theory (OPT) [Kahneman and Tversky, 1979] and cumulative prospect theory (CPT) [Tversky and Kahneman, 1992]. The risky choice models we consider consist of a subjective utility objective (characterizing the amount of value an individual attaches to an outcome) and possibly a probability weighting function (reflecting the tendency for non-linear weighting of probabilities). Despite the long history of development, risky choice is still a focus of the ongoing research [Begenau, 2020, Gächter et al., 2022, Frydman and Jin, 2022].
The objective is to maximize reward from risky choices. Risky choice problems consist of 2 or more options, each of which is described by a set of probability and outcome pairs. For each option, the probabilities sum to 1 . Problems may also have an endowment and/or have multiple stages. These variants are not modelled in this version. We will use similar implementations as Cavagnaro et al. [2013b] to test four models $\mathcal{M}$ with our method: EU, WEU, OPT and CPT. Each model has its own corresponding parameters $\boldsymbol{\theta}_{m}$. We consider choice problems where individuals choose between two lotteries $A$ and $B$. The design space for the risky-choice problems is a combination of designs for lottery $A$ and $B$. The design space for lottery $A$ is defined as the probabilities of the high and low outcome ( $d_{\mathrm{phA}}$ and $d_{\mathrm{plA}}$ ) in this lottery. The design space for lottery $B$ is analogous to lottery $A\left(d_{\mathrm{plB}}\right.$ and $\left.d_{\mathrm{plB}}\right)$. We assume that there the decisions contain choice stochasticity, which serves as a likelihood for the ADO and LBIRD methods. The models are implemented as follows.

Choice stochasticity. It is typical to assume that individual choices in risky choice problems are not deterministic (i.e., there is choice stochasticity). We use the following definition for probability of choosing lottery $A$ over $B$ in a choice problem $i$ [Cavagnaro et al., 2013a]:

$$
\phi_{i}\left(A_{i} \mid \boldsymbol{\theta}_{m}, \epsilon\right)= \begin{cases}\epsilon, & \text { if } A_{i} \prec B_{i} \\ \frac{1}{2}, & \text { if } A_{i} \sim B_{i} \\ 1-\epsilon, & \text { if } A_{i} \succ B_{i}\end{cases}
$$

where $\theta_{m}$ denotes the model parameters and $\epsilon$ is a value in range $[0,0.5]$ quantifying stochasticity of the choice (with $\epsilon=0$ corresponding to a deterministic choice). Whether lottery $A$ is preferred is determined using the utilities defined for each model separately.

EU. Following Cavagnaro et al. [2013b], we specify EU using indifference curves on the Marschak-Machina (MM) probability triangle. Lottery $A$ consists of three outcomes $\left(x_{\mathrm{IA}}, x_{\mathrm{mA}}, x_{\mathrm{hA}}\right)$, and associated probabilities $\left(p_{\mathrm{IA}}, p_{\mathrm{mA}}, p_{\mathrm{hA}}\right)$. Lottery $A$ can be represented using a right triangle (MM) with two of the probabilities as the plane ( $p_{\mathrm{IA}}$ and $p_{\mathrm{hA}}$ as

---

#### Page 14

> **Image description.** This image shows a series of 8 heatmaps arranged in a 2x4 grid, repeated twice. Each heatmap represents the posterior distribution of two model parameters (sensor noise and hit value) at different numbers of trials (1, 4, 20, and 100). The rows correspond to different models (POW and PR). A black "X" marks the true parameter values.
>
> Here's a breakdown:
>
> - **Overall Structure:** The image consists of two identical sets of plots stacked vertically. Each set contains two rows of four heatmaps.
> - **Rows:**
>   - The top row in each set is labeled "POW" on the left-hand side.
>   - The bottom row in each set is labeled "PR" on the left-hand side.
> - **Columns:** The columns represent the number of trials: 1, 4, 20, and 100, indicated above each column.
> - **Heatmaps:** Each heatmap displays a posterior distribution. The x-axis represents "θsens" (sensor noise), ranging from 0.0 to 0.9. The y-axis represents "θhit" (hit value), ranging from 0 to 8. The color intensity (red shading) indicates the probability density, with darker red representing higher probability.
> - **Black "X":** A black "X" is superimposed on each heatmap in the POW row, indicating the true parameter values.
> - **Observations:**
>   - In the POW rows, the posterior distribution becomes more concentrated around the true parameter values (black "X") as the number of trials increases.
>   - In the PR rows, the posterior distribution becomes negligible (empty plots) as the number of trials increases, especially after 4 trials.
> - **Text:** The following text is present: "Trials = 1", "Trials = 4", "Trials = 20", "Trials = 100", "POW", "PR", "θhit", "θsens". The numerical values on the x-axis range from 0.0 to 0.9 in increments of 0.3. The numerical values on the y-axis range from 0 to 8 in increments of 2.

Figure 4: An example of evolution of the posterior approximation in each of the models tested resulting from BOSMOS in the signal detection task. The last bottom row panels are empty as in both cases the posterior probability of the PR model becomes negligible, so that the particle approximation of this posterior does not contain any more particle. The true value of the parameters is indicated by the cross and the true model is POW in both cases. BOSMOS successfully identified the ground-truth model in both cases: all posterior density (shaded area) has concentrated there by 20 trials, and no more particle exists in the other model. However, only in the first example (top panel) did the ground-truth parameter values (cross) fall inside the $90 \%$ confidence interval, indicating some inconsistency in terms of the posterior convergence towards the ground-truth. The axes correspond to the model parameters: sensor-noise (x-axis) and hit value (y-axis); $\theta_{\text {low }}$ and $\theta_{\text {ten }}$ of the PR model are omitted to simplify visualization.
$x$ and $y$ axes, respectively). Hence, the design space for lottery $A$ consists of only the high and low probability ( $d_{\mathrm{plA}}$ and $d_{\mathrm{phA}}$ ). Lottery $B$ can be represented on the triangle similarly (using $d_{\mathrm{plB}}$ and $d_{\mathrm{phB}}$ ). Then, indifference curves can be drawn on this triangle, as their slope represents the marginal rate of substitution between the two probabilities. EU is defined using indifference curves that all have the same slope $\theta_{a} \in \theta_{\mathrm{EU}}$. If lottery $B$ is riskier, $A \succ B$, if $\left|d_{\mathrm{phB}}-d_{\mathrm{phA}}\right| /\left|d_{\mathrm{plB}}-d_{\mathrm{plA}}\right|<\theta_{a}$. We ask to turn to Cavagnaro et al. [2013b] for a more comprehensive explanation of this modelling approach.

WEU. WEU is also defined using the MM-triangle, as per Cavagnaro et al. [2013b]. In contrast to EU, the slope of the indifference curves varies across the MM-triangle for WEU. This is achieved by assuming that all the indifference curves intersect at a point $\left(\theta_{x}, \theta_{y}\right)$ outside the MM-triangle, where $\left[\theta_{x}, \theta_{y}\right] \in \theta_{\text {WEU }}$. Then, $A \succ B$, if $\left|d_{\mathrm{phA}}-\theta_{y}\right| / \mid$ $d_{\mathrm{plA}}-\theta_{x}|>\left|d_{\mathrm{phB}}-\theta_{y}\right| /\left|d_{\mathrm{plB}}-\theta_{x}\right|$.

---

#### Page 15

OPT. In contrast to EU and WEU, OPT assumes that both the outcomes $x$ and probabilities $p$ have specific editing functions $v$ and $w$, respectively. Assuming that for lottery $A, v\left(x_{\text {low }}^{\mathrm{A}}\right)=0$ and $v\left(x_{\text {high }}^{\mathrm{A}}\right)=1$, the utility objectives in OPT can be defined using $v\left(x_{\text {middle }}^{\mathrm{A}}\right)$ as a parameter $\theta_{v}$

$$
u(A)= \begin{cases}w\left(d_{\mathrm{phA}}\right) \cdot 1+\theta_{v} \cdot\left(1-w\left(d_{\mathrm{phA}}\right)\right), & \text { if } d_{\mathrm{plA}}=0 \\ w\left(d_{\mathrm{phA}}\right) \cdot 1+w\left(1-d_{\mathrm{phA}}-d_{\mathrm{plA}}\right) \cdot \theta_{v}, & \text { otherwise }\end{cases}
$$

Utility $u(B)$ for lottery $B$ can be calculated analogously, and $A_{i} \succ B_{i}$ if $u(A)>u(B)$. The probability weighting function $w(\cdot)$ used is the original work by Tversky and Kahneman [1992] is

$$
w(p)=\frac{p^{\theta_{r}}}{\left(p^{\theta_{r}}+(1-p)^{\theta_{r}}\right)^{\left(1 / \theta_{r}\right)}}
$$

where $\theta_{r}$ is a parameter describing the shape of the function. Thus, OPT has two parameters $\left[\theta_{v}, \theta_{r}\right] \in \theta_{\text {OPT }}$, describing the subjective utility of the middle outcome and the shape of the probability weighting function, respectively.

CPT. CPT is defined similarly to OPT, however, the subjective utilities $u$ for lottery $A$ are calculated using

$$
u(A)=w\left(d_{\mathrm{phA}}\right) \cdot 1+\left(w\left(1-d_{\mathrm{plA}}\right)-w\left(d_{\mathrm{phA}}\right)\right) \cdot \theta_{v}
$$

Utility $u(B)$ for lottery $B$ is calculated similarly and $\left[\theta_{v}, \theta_{r}\right] \in \theta_{\mathrm{CPT}}$. We use the following priors for the parameters of models

$$
\begin{array}{ll}
\theta_{a} \sim \operatorname{Unif}(0,10), & \theta_{v} \sim \operatorname{Unif}(0,1) \\
\theta_{r} \sim \operatorname{Unif}(0.01,1), & \theta_{x} \sim \operatorname{Unif}(-100,0) \\
\theta_{y} \sim \operatorname{Unif}(-100,0), & \theta_{e} \sim \operatorname{Unif}(0,0.5)
\end{array}
$$

with the design proposal distributions

$$
\begin{array}{ll}
d_{\mathrm{plA}} \sim \operatorname{Unif}(0,1), & d_{\mathrm{phA}} \sim \operatorname{Unif}(0,1) \\
d_{\mathrm{plB}} \sim \operatorname{Unif}(0,1), & d_{\mathrm{phB}} \sim \operatorname{Unif}(0,1)
\end{array}
$$

Please note that $d_{\mathrm{pmA}}$ and $d_{\mathrm{pmB}}$ can be calculated analytically from $d_{\mathrm{pmA}}=2-d_{\mathrm{plA}}-d_{\mathrm{phA}}$, after which the designs for the same lottery $\left(d_{\mathrm{plA}}, d_{\mathrm{pmA}}, d_{\mathrm{phA}}\right)$ are normalized, so they are summed to 1 (and similar for the lottery B).

# 4.5.1 Results

The risky choice task comprises four computational models, which significantly expand the space of models and makes it much more computationally costly than the memory task. Despite the larger model space, BOSMOS maintains its position as a preferred LFI approach to model selection, most notably when compared to the parameter estimation error of MINEBED from Figure 2. With more models, BOSMOS's performance advantage over MINEBED grows, with BOSMOS exhibiting higher scalability for larger model spaces.
It is crucial to note that having several candidate models reduces model prediction accuracy by the LFI approaches, thus we recommend reducing the number of candidate models as low as feasible. In terms of performance, BOSMOS is comparable to ground-truth likelihood approaches during the first four design trials, as shown in Figure 3, since it is significantly easier to minimize uncertainty early in the trials. Similarly to the memory task, the error of LFI approximation becomes more apparent as the number of trials rises, as evidenced by comparing BOSMOS to ADO for the behavioural fitness error and model predictive accuracy. In terms of the parameter estimate error, BOSMOS performs marginally better than ADO.
Finally, BOSMOS has a relatively low runtime cost, especially compared to other methods (about one minute per design trial). This brings adaptive model selection closer to being applicable to real-world experiments in risky choice. The proposed method can be useful in online experiments that include lag times between trials, for instance, in assessing investment decisions (e.g., Camerer, 2004, Gneezy and Potters, 1997) or game-like settings (e.g., Bauckhage et al., 2012, Putkonen et al., 2022, Viljanen et al., 2017) where the participant waits between events.

## 5 Discussion

In this paper, we proposed a simulator-based experimental design method for model selection, BOSMOS, that does design selection for model and parameter inference at a speed orders of magnitude higher than other methods, bringing

---

#### Page 16

the method closer to online design selection. This was made possible with newly proposed approximation of the model likelihood and simulator-based utility objective. Despite needing orders of magnitude fewer simulations, BOSMOS significantly outperformed LFI alternatives in the majority of cases, while being orders of magnitude faster, bringing the method closer to an online inference tool. Crucially, the time between experiment trials was reduced to less than a minute. Whereas in some settings this time between trials may be too long, BOSMOS is a viable tool in experiments where the tasks include a lag time, for instance, in studies of language learning (e.g., Gardner et al., 1997, Nioche et al., 2021) and task interleaving (e.g., Payne et al., 2007, Brumby et al., 2009, Gebhardt et al., 2021, Katidioti et al., 2014). Moreover, our code implementation represents a proof of concept and was not fully optimized for maximal efficiency: in particular, a parallel implementation that exploits multiple cores and batches of simulated experiments would enable additional speedups [Wu and Frazier, 2016]. As an interactive and sample-efficient method, BOSMOS can help reduce the number of required experiments. This can be of interest to both the subject and the experimenter. In human trials it allows for faster interventions (e.g. adjusting the treatment plan) in critical settings such as ICUs or RCTs. However, it can also have detrimental applications, such as targeted advertising and collecting personal data, therefore the principles and practices of responsible AI [Dignum, 2019, Arrieta et al., 2020] also have to be taken into account in applying our methodology.
There are at least two remaining issues left for future work. The first issue we witnessed in our experiments is that the accuracy of behaviour imitation does not necessarily correlate with the convergence to ground-truth models. This usually happens due to poor identifiability in the model-parameter space, which may be quite prevalent in current and future computational cognitive models, since they are all designed to explain the same behaviour. Currently, the only way to address this problem is to use Bayesian approaches, such as BOSMOS, that quantify the uncertainty over the models and their parameters. The second issue is the consistency of the method: in selecting only the most informative designs, the methods may misrepresent the posterior and return an overconfident posterior. This bias may occur, for example, due to a poor choice of priors or summary statistics [Nunes and Balding, 2010, Fearnhead and Prangle, 2012] for the collected data (when the data is high-dimensional). Ultimately, these issues do not hinder the goal of automating experimental designs, but introduce the necessity for a human expert, who would ensure that the uncertainty around estimated models is acceptable, and the design space is sufficiently explored to make final decisions.
Future work for simulator-based model selection in computational cognitive science needs to consider adopting hierarchical models, accounting for the subjects' ability to adapt or change throughout the experiments, and incorporating amortized non-myopic design selection. A first step in this direction would be to study hierarchical models [Kim et al., 2014] which would allow adjusting prior knowledge for populations and expanding the theory development capabilities of model selection methods from a single individual to a group level. We could also remove the assumption on the stationarity of the model by proposing a dynamic model of subjects' responses which adapts to the history of previous responses and previous designs, which is more reasonable in longer settings of several dozens of trials. Lastly, amortized non-myopic design selections [Blau et al., 2022] would even further reduce the wait time between design proposals, as the model can be pre-trained before experiments, and would also improve design exploration by encouraging long-term planning of the experiments. Addressing these three potential directions may have a synergistic effect on each other, thus expanding the application of simulator-based model selection in cognitive science even further.

### Code Availability

All code for replicating the experiments is available at https://github.com/AaltoPML/BOSMOS.

---

# ONLINE SIMULATOR-BASED EXPERIMENTAL DESIGN FOR COGNITIVE MODEL SELECTION - Backmatter

---

## Acknowledgements

This work was supported by the Academy of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI; grants 328400, 345604, 328400, 320181). AP and SC were funded by the Academy of Finland projects BAD (Project ID: 318559) and Human Automata (Project ID: 328813). AP was additionally funded by Aalto University School of Electrical Engineering. SC was also funded by Interactive Artificial Intelligence for Research and Development (AIRD) grant by Future Makers. SK was supported by the Engineering and Physical Sciences Research Council (EPSRC; Project ID: EP-W002973-1). Computational resources were provided by the Aalto Science-IT Project.

---

#### Page 17

## References

Luigi Acerbi, Wei Ji Ma, and Sethu Vijayakumar. A framework for testing identifiability of Bayesian models of perception. Advances in Neural Information Processing Systems, 27, 2014.
Hafeez Ullah Amin and Aamir Saeed Malik. Human memory retention and recall processes. a review of EEG and fMRI studies. Neurosciences, 18 4:330-44, 2013.
John R Anderson. Arguments concerning representations for mental imagery. Psychological Review, 85(4):249, 1978.
Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez, Javier Del Ser, Adrien Bennetot, Siham Tabik, Alberto Barbado, Salvador García, Sergio Gil-López, Daniel Molina, Richard Benjamins, et al. Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. Information Fusion, 58:82-115, 2020.

Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. BoTorch: a framework for efficient Monte-Carlo Bayesian optimization. Advances in Neural Information Processing Systems, 33:21524-21538, 2020.
Christian Bauckhage, Kristian Kersting, Rafet Sifa, Christian Thurau, Anders Drachen, and Alessandro Canossa. How players lose interest in playing a game: An empirical study based on distributions of total playing times. In 2012 IEEE Conference on Computational Intelligence and Games (CIG), pages 139-146. IEEE, 2012.
Mark A Beaumont, Wenyang Zhang, and David J Balding. Approximate Bayesian computation in population genetics. Genetics, 162(4):2025-2035, 2002.
Juliane Begenau. Capital requirements, risk choice, and liquidity provision in a business-cycle model. Journal of Financial Economics, 136(2):355-378, 2020.
Tom Blau, Edwin V Bonilla, Iadine Chades, and Amir Dezfouli. Optimizing sequential experimental design with deep reinforcement learning. In International Conference on Machine Learning, pages 2107-2128. PMLR, 2022.
Jan Boelts, Jan-Matthis Lueckmann, Richard Gao, and Jakob H Macke. Flexible and efficient simulation-based inference for models of decision-making. eLife, 11:e77220, 2022.
Duncan P. Brumby, Dario D. Salvucci, and Andrew Howes. Focus on driving: How cognitive constraints shape the adaptation of strategy when dialing while driving. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, CHI '09, page 1629-1638, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605582467. doi: 10.1145/1518701.1518950. URL https://doi.org/10.1145/1518701.1518950.
Colin F Camerer. Prospect theory in the wild: Evidence from the field. Advances in Behavioral Economics, pages $148-161,2004$.
Daniel R Cavagnaro, Jay I Myung, Mark A Pitt, and Janne V Kujala. Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. Neural Computation, 22(4):887-905, 2010 .
Daniel R Cavagnaro, Richard Gonzalez, Jay I Myung, and Mark A Pitt. Optimal decision stimuli for risky choice experiments: An adaptive approach. Management Science, 59(2):358-375, 2013a.
Daniel R Cavagnaro, Mark A Pitt, Richard Gonzalez, and Jay I Myung. Discriminating among probability weighting functions using adaptive design optimization. Journal of Risk and Uncertainty, 47(3):255-289, 2013b.
Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48):30055-30062, 2020.
Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential Monte-Carlo samplers. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(3):411-436, 2006.
Virginia Dignum. Responsible artificial intelligence: how to develop and use AI in a responsible way. Springer Nature, 2019.

Paul Fearnhead and Dennis Prangle. Constructing summary statistics for approximate Bayesian computation: semiautomatic approximate Bayesian computation. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 74(3):419-474, 2012.
Adam Foster, Desi R Ivanova, Ilyas Malik, and Tom Rainforth. Deep adaptive design: Amortizing sequential Bayesian experimental design. In International Conference on Machine Learning, pages 3384-3395. PMLR, 2021.

---

#### Page 18

Peter I Frazier. Bayesian optimization. In Recent Advances in Optimization and Modeling of Contemporary Problems, pages 255-278. Informs, 2018.
Cary Frydman and Lawrence J Jin. Efficient coding and risky choice. The Quarterly Journal of Economics, 137(1): $161-213,2022$.
Simon Gächter, Eric J Johnson, and Andreas Herrmann. Individual-level loss aversion in riskless and risky choices. Theory and Decision, 92(3):599-624, 2022.
Robert C Gardner, Paul F Tremblay, and Anne-Marie Masgoret. Towards a full model of second language learning: An empirical investigation. The Modern Language Journal, 81(3):344-362, 1997.
Christoph Gebhardt, Antti Oulasvirta, and Otmar Hilliges. Hierarchical reinforcement learning explains task interleaving behavior. Computational Brain \& Behavior, 4(3):284-304, 2021.
Uri Gneezy and Jan Potters. An experiment on risk taking and evaluation periods. The Quarterly Journal of Economics, 112(2):631-645, 1997.
Stewart Greenhill, Santu Rana, Sunil Gupta, Pratibha Vellanki, and Svetha Venkatesh. Bayesian optimization for adaptive experimental design: a review. IEEE Access, 8:13937-13948, 2020.
Thom Griffith, Sophie-Anne Baker, and Nathan F Lepora. The statistics of optimal decision-making: Exploring the relationship between signal detection theory and sequential analysis. Journal of Mathematical Psychology, 103: 102544, 2021.
Michael U Gutmann and Jukka Corander. Bayesian optimization for likelihood-free inference of simulator-based statistical models. Journal of Machine Learning Research, 2016.
Chew Soo Hong. A generalization of the quasilinear mean with applications to the measurement of income inequality and decision theory resolving the Allais paradox. Econometrica, 51(4):1065-1092, 1983. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/1912052.

Andrew Howes, Richard L Lewis, and Alonso Vera. Rational adaptation under task and processing constraints: Implications for testing theories of cognition and action. Psychological Review, 116(4):717, 2009.
Desi R Ivanova, Adam Foster, Steven Kleinegesse, Michael U Gutmann, and Thomas Rainforth. Implicit deep adaptive design: policy-based experimental design without likelihoods. Advances in Neural Information Processing Systems, 34:25785-25798, 2021.
Joseph G. Johnson and Jerome R. Busemeyer. Decision-making under risk and uncertainty. WIREs Cognitive Science, 1(5):736-749, 2010. doi: $10.1002 /$ wcs. 76.
Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of Artificial Intelligence Research, 4:237-285, 1996.
Daniel Kahneman and Amos Tversky. Prospect theory: An analysis of decision under risk. Econometrica, 47(2): 263-292, 1979.
Antti Kangasrääsiö, Jussi PP Jokinen, Antti Oulasvirta, Andrew Howes, and Samuel Kaski. Parameter inference for computational cognitive models with approximate Bayesian computation. Cognitive Science, 43(6):e12738, 2019.
Ioanna Katidioti, Jelmer P Borst, and Niels A Taatgen. What happens when we switch tasks: Pupil dilation in multitasking. Journal of Experimental Psychology: Applied, 20(4):380, 2014.
Jack Kiefer. Optimum experimental designs. Journal of the Royal Statistical Society: Series B (Methodological), 21 (2):272-304, 1959.

Woojae Kim, Mark A Pitt, Zhong-Lin Lu, Mark Steyvers, and Jay I Myung. A hierarchical adaptive approach to optimal experimental design. Neural Computation, 26(11):2465-2492, 2014.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Steven Kleinegesse and Michael U Gutmann. Bayesian experimental design for implicit models by mutual information neural estimation. In International Conference on Machine Learning, pages 5316-5326. PMLR, 2020.
Frank H. Knight. Risk, Uncertainty and Profit. University of Chicago Press, Chicago, repr edition, 1985. ISBN $0-226-44691-3$.
Michael D Lee, Amy H Criss, Berna Devezer, Christopher Donkin, Alexander Etz, Fäbio P Leite, Dora Matzke, Jeffrey N Rouder, Jennifer S Trueblood, Corey N White, et al. Robust modeling in cognitive science. Computational Brain \& Behavior, 2(3):141-153, 2019.

---

#### Page 19

Dennis V Lindley. On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, 27(4):986-1005, 1956.
Jens Koed Madsen, Richard M Bailey, and Toby D Pilditch. Large networks of rational agents form persistent echo chambers. Scientific Reports, 8(1):1-8, 2018.
Jens Koed Madsen, Richard Bailey, Ernesto Carrella, and Philipp Koralus. Analytic versus computational cognitive models: Agent-based modeling as a tool in cognitive sciences. Current Directions in Psychological Science, 28(3): 299-305, 2019.
Jean-Michel Marin, Pierre Pudlo, Christian P Robert, and Robin J Ryder. Approximate bayesian computational methods. Statistics and computing, 22(6):1167-1180, 2012.
Hee-Seung Moon, Seungwon Do, Wonjae Kim, Jiwon Seo, Minsuk Chang, and Byungjoo Lee. Speeding up inference with user simulators through policy modulation. In CHI Conference on Human Factors in Computing Systems, pages 1-21, 2022.
Jay I Myung, Daniel R Cavagnaro, and Mark A Pitt. A tutorial on adaptive design optimization. Journal of Mathematical Psychology, 57(3-4):53-67, 2013.
Aurélien Nioche, Pierre-Alexandre Murena, Carlos de la Torre-Ortiz, and Antti Oulasvirta. Improving artificial teachers by considering how people learn and forget. In 26th International Conference on Intelligent User Interfaces, pages 445-453, 2021.
Matthew A Nunes and David J Balding. On optimal selection of summary statistics for approximate Bayesian computation. Statistical Applications in Genetics and Molecular Biology, 9(1), 2010.
Antti Oulasvirta, Jussi PP Jokinen, and Andrew Howes. Computational rationality as a theory of interaction. In CHI Conference on Human Factors in Computing Systems, pages 1-14. ACM, 2022.
George Papamakarios, David Sterratt, and Iain Murray. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 837-848. PMLR, 2019.
Stephen J Payne, Geoffrey B Duggan, and Hansjörg Neth. Discretionary task interleaving: heuristics for time allocation in cognitive foraging. Journal of Experimental Psychology: General, 136(3):370, 2007.
W. Peterson, T. Birdsall, and W. Fox. The theory of signal detectability. Transactions of the IRE Professional Group on Information Theory, 4(4):171-212, 1954. doi: 10.1109/TIT.1954.1057460.
Pierre Pudlo, Jean-Michel Marin, Arnaud Estoup, Jean-Marie Cornuet, Mathieu Gautier, and Christian P Robert. Reliable ABC model choice via random forests. Bioinformatics, 32(6):859-866, 2016.
Aini Putkonen, Aurélien Nioche, Ville Tanskanen, Arto Klami, and Antti Oulasvirta. How suitable is your naturalistic dataset for theory-based user modeling? In Proceedings of the 30th ACM Conference on User Modeling, Adaptation and Personalization, pages 179-190, 2022.
Carl Edward Rasmussen. Gaussian processes in machine learning. In Summer School on Machine Learning, pages 63-71. Springer, 2003.
Christian P Robert et al. The Bayesian choice: from decision-theoretic foundations to computational implementation, volume 2. Springer, 2007.
David C. Rubin and Amy E. Wenzel. One hundred years of forgetting: a quantitative description of retention. Psychological Review, 103(4):734-760, October 1996. ISSN 1939-1471, 0033-295X. doi: 10.1037/0033-295X.103.4.734. URL http://doi.apa.org/getdoi.cfm?doi=10.1037/0033-295X.103.4.734.
David C. Rubin, Sean Hinton, and Amy Wenzel. The precise time course of retention. Journal of Experimental Psychology: Learning, Memory, and Cognition, 25(5):1161-1176, September 1999. ISSN 1939-1285, 0278-7393. doi: 10.1037/0278-7393.25.5.1161. URL http://doi.apa.org/getdoi.cfm?doi=10.1037/0278-7393.25. 5.1161 .

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.
Gideon Schwarz. Estimating the dimension of a model. The Annals of Statistics, pages 461-464, 1978.
Matthias Seeger. Gaussian processes for machine learning. International Journal of Neural Systems, 14(02):69-106, 2004.

Claude Elwood Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27(3): $379-423,1948$.

---

#### Page 20

Scott A Sisson, Yanan Fan, and Mark Beaumont. Handbook of approximate Bayesian computation. CRC Press, 2018.
Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995, 2009.
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT Press, 2018.
John A. Swets, Wilson P. Tanner, and Theodore G. Birdsall. Decision processes in perception. Psychological Review, 68(5):301-340, September 1961. ISSN 1939-1471, 0033-295X. doi: 10.1037/h0040547. URL http://doi.apa. org/getdoi.cfm?doi=10.1037/h0040547.
Wilson P. Tanner and John A. Swets. A decision-making theory of visual detection. Psychological Review, 61(6): 401-409, 1954. ISSN 1939-1471, 0033-295X. doi: 10.1037/h0058700. URL http://doi.apa.org/getdoi. cfm?doi=10.1037/h0058700.
Simon Tavaré, David J Balding, Robert C Griffiths, and Peter Donnelly. Inferring coalescence times from dna sequence data. Genetics, 145(2):505-518, 1997.
Amos Tversky and Daniel Kahneman. Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and Uncertainty, 5(4):297-323, 1992. doi: 10.1007/978-3-319-20451-2 $\backslash .24$.
Simon Valentin, Steven Kleinegesse, Neil R Bramley, Michael Gutmann, and Chris Lucas. Bayesian experimental design for intractable models of cognition. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 43, 2021.
Bas van Opheusden, Luigi Acerbi, and Wei Ji Ma. Unbiased and efficient log-likelihood estimation with inverse binomial sampling. PLoS Computational Biology, 16(12):e1008483, 2020.
Markus Viljanen, Antti Airola, Jukka Heikkonen, and Tapio Pahikkala. Playtime measurement with survival analysis. IEEE Transactions on Games, 10(2):128-138, 2017.
John Von Neumann and Oskar Morgenstern. 3. the notion of utility. In Theory of Games and Economic Behavior, pages 15-31. Princeton University Press, Princeton, New Jersey, 3. edition, 1990. original-date: 1944.
Scott I Vrieze. Model selection and psychological theory: a discussion of the differences between the Akaike information criterion (AIC) and the Bayesian information criterion (BIC). Psychological Methods, 17(2):228, 2012.
Thomas D. Wickens. Elementary signal detection theory. Oxford University Press, Oxford ; New York, 2002. ISBN 9780195092509.

John T. Wixted. The forgotten history of signal detection theory. Journal of Experimental Psychology: Learning, Memory, and Cognition, 46(2):201-233, February 2020. ISSN 1939-1285, 0278-7393. doi: 10.1037/xlm0000732. URL http://doi.apa.org/getdoi.cfm?doi=10.1037/xlm0000732.
Jian Wu and Peter Frazier. The parallel knowledge gradient method for batch bayesian optimization. Advances in neural information processing systems, 29, 2016.
Jiping Yang and Wanhua Qiu. A measure of risk and a decision-making model based on expected utility and entropy. European Journal of Operational Research, 164(3):792-799, 2005.

---

# Online Simulator-Based Experimental Design for Cognitive Model Selection - Appendix

---

# Supplementary information

The article has the following accompanying supplementary materials:

- Appendix A shows the validity of the approximation of the entropy gain for the design selection rule;
- Appendix B details the algorithm for the proposed BOSMOS method;
- Appendix C contains tables with full experimental results, which shows additional design evaluation points;
- Appendix D showcases a side-by-side comparison of the posterior evolution resulting from BOSMOS and MINEBED for the signal detection task.

---

#### Page 21

# A Approximation of the entropy gain for design selection

Since the expected values in the entropy gain from Equation (3) are not tractable, we rely on a Monte-Carlo estimation of these quantities. We focus on the first term, as the second one only features one of the approximations.
Following the SMC framework [Del Moral et al., 2006], we propose to sequentially update a particle population between trials. This allows the online update of the posterior along the experiment.
At each step we estimate the distribution $p\left(m, \boldsymbol{\theta}_{m} \mid \mathcal{D}_{1: t}\right)$ with a population of $N_{1}$ particles $\left(m^{i}, \boldsymbol{\theta}_{m}^{i}\right)$. Following importance sampling, we know that $\sum_{i=1}^{N_{1}} w_{i} \delta_{\left(m^{i}, \boldsymbol{\theta}_{m}^{i}\right)}$ converges to the distribution associated with density $p\left(m, \boldsymbol{\theta}_{m} \mid\right.$ $\mathcal{D}_{t}$ ). This population of particles then evolves according to the SMC algorithm [Del Moral et al., 2006].
To estimate the expected value in Equation (3), we use a standard Monte-Carlo estimation, making use of the particles at time $t$ as an estimation of the posterior at time $t$. For each particle, we simulate $N_{2}$ vectors $\boldsymbol{x}_{t}^{i}(j) \sim p(\cdot \mid$ $\left.\boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}^{i}, m^{i}\right)$. The convergence to prove is then:

$$
\sum_{i, j}\left(N_{2}\right)^{-1} w_{i} \hat{H}\left(\boldsymbol{x}_{t}^{i_{1}}(j) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right) \rightarrow_{N_{1}, N_{2} \rightarrow \infty} \mathbb{E}_{q\left(m, \boldsymbol{\theta}_{m} \mid \mathcal{D}_{t-1}\right)}\left[H\left(\boldsymbol{x}_{t}^{i} \mid m, \boldsymbol{\theta}_{m}\right)\right]
$$

where $\hat{H}$ is a modified version of the entropy. Note that it is also possible to estimate the gradient of this quantity with respect to the design. If we truly computed $H\left(\boldsymbol{x}_{t}^{i}\right)$ as the entropy with respect to the measure $\sum_{i} \delta_{x_{i}^{i}}$, it would lead to constantly null value. Thus, we decide to use a kernel approximation of the distribution: $\hat{H}\left(\boldsymbol{x}_{t}^{i_{1}}(j) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right)=$ $H\left(\sum_{i} \mathcal{N}\left(\cdot \mid x_{i}^{\prime}(j), \sigma_{N_{2}}\right) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right)$, with $\sigma_{N_{2}} \rightarrow_{N_{2} \rightarrow \infty} 0$.
The convergence of the estimator in Equation 42 to the true entropy requires two results.
First, the convergence in $N_{2}$ :

$$
\sum_{i=1}^{N_{1}} \sum_{j=1}^{N_{2}}\left(N_{2}\right)^{-1} w_{i} \hat{H}\left(\boldsymbol{x}_{t}^{i_{1}}(j) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right) \rightarrow_{N_{2} \rightarrow \infty} \sum_{i=1}^{N_{1}} w_{i} \hat{H}\left(p\left(\cdot \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}^{i}, m^{i}\right) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right)
$$

using the convergence of $\mathcal{N}\left(\cdot \mid x_{i}^{\prime}(j), \sigma_{N_{2}}\right)$ to $\delta_{x_{i}^{\prime}(j)}$ in distribution and the law of large numbers. Second, the convergence for $N_{1} \rightarrow \infty$ comes from the results on SMC [Del Moral et al., 2006].

## B Algorithms

```
Algorithm 1 Bayesian optimization for simulator-based model selection
    Input: prior over models \(p(m)\) and parameters \(p\left(\boldsymbol{\theta}_{m} \mid m\right)_{m} ;\) set of all models \(\mathcal{M}=\left\{p\left(\boldsymbol{x} \mid \boldsymbol{\theta}_{m}, m, d\right)\right\} ;\) design
    budget \(N_{d}\); total number of particles \(N_{q}\);
    Output: selected model \(m^{\prime}\) and its parameters \(\boldsymbol{\theta}_{m}^{\prime}\);
    initialize current beliefs from the priors:
        \(q\left(m, \boldsymbol{\theta}_{m}\right):=\left\{\left(m^{\prime}, \boldsymbol{\theta}_{m}^{i}\right): \boldsymbol{\theta}_{m}^{i} \sim p\left(\boldsymbol{\theta}_{m} \mid m^{\prime}\right), m^{\prime} \sim p(m)\right\}_{i=0}^{N_{d}} ;\)
    initialize an empty set for the collected data: \(\mathcal{D}_{0}=\{ \}\);
    for \(i:=1: N_{d}\) do
        get the design \(\boldsymbol{d}^{\prime}\) with Equation (5);
        collect the data \(\boldsymbol{x}^{\prime}\) at the design location \(\boldsymbol{d}^{\prime}\) and store it in \(\mathcal{D}_{i}\);
        for \(m^{\prime} \in \mathcal{M}\) do
            get the likelihood \(\mathcal{L}_{\epsilon_{m^{\prime}}}\left(\boldsymbol{x}_{i} \mid \boldsymbol{\theta}_{m^{\prime}}\right)\) with Equation (7);
        end for
        get the marginal likelihood \(\mathcal{L}\left(\boldsymbol{x}_{i} \mid m, \mathcal{D}_{i-1}\right)\) with Equation (9);
        update \(q\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{i}\right)\) with Equation (12);
    end for
    apply the decision rule (e.g. MAP):
        \(m^{\prime}=\arg \max _{m} \sum_{\boldsymbol{\theta}_{m}} q\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{N_{d}}\right)\);
        \(\boldsymbol{\theta}_{m}^{\prime}=\arg \max _{\boldsymbol{\theta}_{m}} q\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{N_{d}}\right)\)
```

---

#### Page 22

# C Full experimental results

We provide full experimental results data in Tables 2-4, where evaluations of performance metrics were made after different numbers of design iterations. Moreover, we report time costs of running 100 design trials for each method in Table 5, where our method was 80-100 times faster than the other LFI method, MINEBED. The rest of the section discusses three additional minor points with relation to the performance of ADO for the demonstrative example, the bias in the model space for the memory task, and results of testing two decision rules in the risky choice.
As we have seen in the main text, MINEBED had the fastest behavioural fitness convergence rate in the demonstrative example, while BOSMOS was the close second. Hence, ADO had the slowest convergence rate among design optimization methods for behavioural fitness (Table 2) and also for parameter estimation (Table 3). This result is somewhat counterintuitive, as we expected ADO, with its ground-truth likelihood and design optimization, to be the fastest to converge. Since the only other factor influencing this outcome, Bayesian updates, had access to the ground-truth and avoided LFI approximations, suboptimal designs are likely to blame for the poor performance. This problem is likely to be mitigated by expanding the size of the grid used by ADO to calculate the utility objective. However, expanding it would likely increase ADO's convergence at the expense of more calculations; therefore we aimed to get its running time closer to that of BOSMOS, so both methods could be fairly compared.
In the results of model selection for the memory retention task discussed in the main text, MINEBED showed a marginally better average model accuracy than BOSMOS. However, upon closer inspection (Table 4), this accuracy can be solely attributed to the strong bias towards the POW model; the other approaches show it as well, albeit less dramatically. This suggests that the two models in the memory task are separable, but the EXP model cannot likely replicate parts of the behaviour space that the POW model can, resulting in this skewness towards the more flexible model. This is also more broadly related to non-identifiability: since these cognitive models were designed to explain the same target behaviour, it is inevitable that there will be an overlap in their response (or behavioural data) space, complicating model selection.
Since the risky choice model had four models of varied complexity, we experimented with two distinct decisionmaking rules for estimating the models and parameters: the default MAP and Bayesian information criterion (BIC) [Schwarz, 1978, Vrieze, 2012]. Both decision-making rules include a penalty for the size of the model (artificially for BIC, and by definition for MAP). Interestingly, the results are the same for both decision-making rule, indicating that the EU model cannot be completely replaced by a more flexible model. BIC's slightly superior parameter estimates for the BOSMOS technique is most likely explained by the poorer model prediction accuracy. Nevertheless, the BIC rule remains a viable option for model selection in situations when there is a risk of having a more flexible and complex model alongside few-parameter alternatives, despite being less supported theoretically.

## D Posterior evolution examples for BOSMOS and MINEBED

In Figures 5 and 6, we compare posterior distributions returned by MINEBED and BOSMOS for two synthetic participants in the signal detection task. In both examples, the methods have successfully identified the ground-truth POW model, as the majority of the posterior density (shaded area in the figure) has moved to the correct model. Nevertheless, BOSMOS and MINEBED have quite different posteriors, which emphasizes the influence of the design selection strategy on the resulting convergence, as one of the method performs better than the other in each of the provided examples.

---

#### Page 23

> **Image description.** This image is a figure containing a series of 2D density plots showing the evolution of posterior distributions. The figure is arranged in a 4x4 grid of panels.
>
> - **Overall Structure:** The figure is organized into four rows and four columns. Each column represents a different number of trials (1, 4, 20, and 100, from left to right). The first and third rows are labeled "POW", while the second and fourth rows are labeled "PR".
>
> - **Panel Contents:** Each panel contains a 2D density plot. The x-axis is labeled "$\theta_{sens}$", and the y-axis is labeled "$\theta_{hit}$". The y-axis ranges from 0 to 8. The x-axis ranges from 0.0 to 0.9.
>
> - **Density Plots:** The density plots in the first two rows are green, while the density plots in the third and fourth rows are red. The density plots show the distribution of data points, with darker areas indicating higher density. A black "X" is present in the POW plots (rows 1 and 3).
>
> - **Text Labels:** Each column has a title indicating the number of trials: "Trials = 1", "Trials = 4", "Trials = 20", and "Trials = 100". The rows are labeled "POW" and "PR" on the left side, along with "$\theta_{hit}$" and numbers 8, 6, 4, 2, and 0.
>
> - **Observations:** As the number of trials increases from left to right, the density plots in the first two rows (green) become more concentrated. In the third row (red), the density plots also become more concentrated, but the fourth row (red) becomes empty for the higher trial numbers.

Figure 5: The first example of evolution of the posterior distribution resulting from MINEBED (green; rows 1 and 2) and BOSMOS (red; rows 3 and 4) for the signal detection task. For each method, the first row shows the distributions of parameters of the POW model (ground-truth), followed by the PR model parameter distributions in the second row. The axes correspond to the model parameters: sensor-noise (x-axis) and hit value (y-axis); $\theta_{\text {low }}$ and $\theta_{\text {len }}$ of the PR model are omitted to simplify visualization. The last bottom row panels are empty as in both cases the posterior probability of the PR model becomes negligible, so that the particle approximation of this posterior does not contain any more particle.

---

#### Page 24

> **Image description.** This image contains a set of plots arranged in a 4x4 grid, displaying the evolution of posterior distributions.
>
> Each plot in the grid is a 2D contour plot with the x-axis labeled "$\theta_{sens}$" ranging from 0.0 to 0.9, and the y-axis labeled "$\theta_{hit}$" ranging from 0 to 8. The plots are arranged in rows and columns, with the columns representing different numbers of trials (1, 4, 20, and 100).
>
> The first two rows display plots related to the "MINEBED" method, with the first row showing distributions for the "POW" model and the second row for the "PR" model. These plots are colored in shades of green. A black "X" is present in each of the POW model plots. The last two plots in the second row are empty.
>
> The last two rows display plots related to the "BOSMOS" method, with the third row showing distributions for the "POW" model and the fourth row for the "PR" model. These plots are colored in shades of red. A black "X" is present in each of the POW model plots. The last two plots in the fourth row are empty.

Figure 6: The second example of evolution of the posterior distribution resulting from MINEBED (green; rows 1 and 2) and BOSMOS (red; rows 3 and 4) for the signal detection task. For each method, the first row shows the distributions of parameters of the POW model (ground-truth), followed by the PR model parameter distributions in the second row. The axes correspond to the model parameters: sensor-noise (x-axis) and hit value (y-axis); $\theta_{\text {low }}$ and $\theta_{\text {hit }}$ of the PR model are omitted to simplify visualization. The last bottom row panels are empty as in both cases the posterior probability of the PR model becomes negligible, so that the particle approximation of this posterior does not contain any more particle.

---

#### Page 25

|    Methods    |     Tasks: number of design trials     |                 |                 |                 |                 |
| :-----------: | :------------------------------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|               |         Demonstrative example          |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |            $0.03 \pm 0.03$             | $0.02 \pm 0.02$ | $0.02 \pm 0.02$ | $0.01 \pm 0.01$ |        -        |
|    MINEBED    |            $0.02 \pm 0.07$             | $0.01 \pm 0.00$ | $0.01 \pm 0.00$ | $0.01 \pm 0.00$ |        -        |
|    BOSMOS     |            $0.05 \pm 0.07$             | $0.01 \pm 0.01$ | $0.01 \pm 0.00$ | $0.01 \pm 0.00$ |        -        |
|     LBIRD     |            $0.36 \pm 0.24$             | $0.33 \pm 0.25$ | $0.29 \pm 0.24$ | $0.14 \pm 0.18$ |        -        |
|     Prior     | Baseline for 0 trials: $0.33 \pm 0.3$  |                 |                 |                 |                 |
|               |            Memory retention            |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |            $0.20 \pm 0.16$             | $0.17 \pm 0.14$ | $0.15 \pm 0.10$ | $0.07 \pm 0.06$ | $0.05 \pm 0.03$ |
|    MINEBED    |            $0.27 \pm 0.22$             | $0.24 \pm 0.19$ | $0.24 \pm 0.19$ | $0.23 \pm 0.15$ | $0.23 \pm 0.17$ |
|    BOSMOS     |            $0.24 \pm 0.19$             | $0.19 \pm 0.16$ | $0.17 \pm 0.14$ | $0.15 \pm 0.13$ | $0.13 \pm 0.11$ |
|     LBIRD     |            $0.20 \pm 0.17$             | $0.17 \pm 0.15$ | $0.14 \pm 0.11$ | $0.08 \pm 0.06$ | $0.05 \pm 0.03$ |
|     Prior     | Baseline for 0 trials: $0.33 \pm 0.47$ |                 |                 |                 |                 |
|               |            Signal detection            |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|    MINEBED    |            $0.27 \pm 0.24$             | $0.24 \pm 0.20$ | $0.23 \pm 0.17$ | $0.21 \pm 0.18$ | $0.20 \pm 0.17$ |
|    BOSMOS     |            $0.25 \pm 0.21$             | $0.20 \pm 0.17$ | $0.17 \pm 0.15$ | $0.17 \pm 0.15$ | $0.15 \pm 0.12$ |
|     Prior     | Baseline for 0 trials: $0.40 \pm 0.49$ |                 |                 |                 |                 |
|               |              Risky choice              |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |            $0.32 \pm 0.11$             | $0.30 \pm 0.13$ | $0.27 \pm 0.12$ | $0.14 \pm 0.08$ | $0.07 \pm 0.04$ |
|    MINEBED    |            $0.30 \pm 0.11$             | $0.31 \pm 0.12$ | $0.26 \pm 0.12$ | $0.21 \pm 0.12$ | $0.22 \pm 0.13$ |
|    BOSMOS     |            $0.26 \pm 0.11$             | $0.23 \pm 0.12$ | $0.24 \pm 0.13$ | $0.18 \pm 0.11$ | $0.14 \pm 0.08$ |
| MINEBED (BIC) |            $0.25 \pm 0.11$             | $0.26 \pm 0.13$ | $0.23 \pm 0.11$ | $0.21 \pm 0.12$ | $0.22 \pm 0.13$ |
| BOSMOS (BIC)  |            $0.24 \pm 0.11$             | $0.24 \pm 0.13$ | $0.23 \pm 0.11$ | $0.19 \pm 0.12$ | $0.15 \pm 0.10$ |
|     LBIRD     |            $0.31 \pm 0.12$             | $0.30 \pm 0.13$ | $0.26 \pm 0.13$ | $0.14 \pm 0.07$ | $0.08 \pm 0.03$ |
|     Prior     | Baseline for 0 trials: $0.44 \pm 0.50$ |                 |                 |                 |                 |

Table 2: Convergence of behavioural fitness error $\eta_{\mathrm{b}}$ (mean $\pm$ std. across 100 simulated participants) for comparison methods (rows) with increased number of trials (columns).

---

#### Page 26

|    Methods    | Tasks: number of design trials |                                        |                 |                 |                 |
| :-----------: | :----------------------------: | :------------------------------------: | :-------------: | :-------------: | :-------------: |
|               |                                |         Demonstrative example          |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |        $0.05 \pm 0.06$         |            $0.04 \pm 0.05$             | $0.03 \pm 0.04$ | $0.01 \pm 0.01$ |        -        |
|    MINEBED    |        $0.00 \pm 0.02$         |            $0.00 \pm 0.00$             | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |        -        |
|    BOSMOS     |        $0.05 \pm 0.07$         |            $0.01 \pm 0.02$             | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |        -        |
|     LBIRD     |        $0.34 \pm 0.22$         |            $0.32 \pm 0.24$             | $0.29 \pm 0.25$ | $0.17 \pm 0.18$ |        -        |
|     Prior     |                                | Baseline for 0 trials: $0.33 \pm 0.23$ |                 |                 |                 |
|               |        Memory retention        |                                        |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |        $0.25 \pm 0.21$         |            $0.25 \pm 0.20$             | $0.23 \pm 0.19$ | $0.19 \pm 0.14$ | $0.14 \pm 0.12$ |
|    MINEBED    |        $0.47 \pm 0.38$         |            $0.46 \pm 0.38$             | $0.46 \pm 0.39$ | $0.48 \pm 0.40$ | $0.48 \pm 0.43$ |
|    BOSMOS     |        $0.29 \pm 0.21$         |            $0.27 \pm 0.22$             | $0.28 \pm 0.20$ | $0.27 \pm 0.22$ | $0.29 \pm 0.22$ |
|     LBIRD     |        $0.25 \pm 0.21$         |            $0.25 \pm 0.20$             | $0.22 \pm 0.18$ | $0.22 \pm 0.20$ | $0.15 \pm 0.13$ |
|     Prior     |                                | Baseline for 0 trials: $0.33 \pm 0.20$ |                 |                 |                 |
|               |        Signal detection        |                                        |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|    MINEBED    |        $0.60 \pm 0.24$         |            $0.56 \pm 0.28$             | $0.43 \pm 0.34$ | $0.45 \pm 0.35$ | $0.49 \pm 0.34$ |
|    BOSMOS     |        $0.37 \pm 0.22$         |            $0.35 \pm 0.19$             | $0.36 \pm 0.21$ | $0.35 \pm 0.20$ | $0.35 \pm 0.19$ |
|     Prior     |                                | Baseline for 0 trials: $0.35 \pm 0.20$ |                 |                 |                 |
|               |          Risky choice          |                                        |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |        $0.42 \pm 0.22$         |            $0.44 \pm 0.23$             | $0.43 \pm 0.22$ | $0.33 \pm 0.24$ | $0.26 \pm 0.23$ |
|    MINEBED    |        $0.86 \pm 0.28$         |            $0.87 \pm 0.27$             | $0.81 \pm 0.32$ | $0.76 \pm 0.36$ | $0.86 \pm 0.29$ |
|    BOSMOS     |        $0.41 \pm 0.25$         |            $0.40 \pm 0.21$             | $0.45 \pm 0.26$ | $0.29 \pm 0.25$ | $0.21 \pm 0.23$ |
| MINEBED (BIC) |        $0.85 \pm 0.33$         |            $0.83 \pm 0.31$             | $0.87 \pm 0.30$ | $0.86 \pm 0.31$ | $0.84 \pm 0.35$ |
| BOSMOS (BIC)  |        $0.27 \pm 0.23$         |            $0.26 \pm 0.19$             | $0.40 \pm 0.26$ | $0.23 \pm 0.22$ | $0.21 \pm 0.22$ |
|     LBIRD     |        $0.51 \pm 0.25$         |            $0.48 \pm 0.24$             | $0.40 \pm 0.24$ | $0.35 \pm 0.22$ | $0.24 \pm 0.19$ |
|     Prior     |                                | Baseline for 0 trials: $0.48 \pm 0.25$ |                 |                 |                 |

Table 3: Convergence of parameter estimation error $\eta_{\mathrm{p}}$ (mean $\pm$ std. across 100 simulated participants) when the model is predicted correctly for comparison methods (rows) with increased number of trials (columns).

---

#### Page 27

|    Methods    |        Tasks: number of design trials        |               |               |               |               |
| :-----------: | :------------------------------------------: | :-----------: | :-----------: | :-----------: | :-----------: |
|               |        Demonstrative example (PM, NM)        |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|      ADO      |                 $0.98,0.96$                  |  $0.98,1.00$  |  $0.98,1.00$  |  $1.00,1.00$  |       -       |
|    MINEBED    |                 $0.93,0.98$                  |  $0.98,1.00$  |  $0.98,1.00$  |  $0.98,1.00$  |       -       |
|    BOSMOS     |                 $0.87,0.83$                  |  $1.00,0.98$  |  $1.00,0.98$  |  $1.00,1.00$  |       -       |
|     LBIRD     |                 $0.51,0.48$                  |  $0.56,0.59$  |  $0.60,0.65$  |  $0.82,0.87$  |       -       |
|     Prior     |       Baseline for 0 trials: $0.5,0.5$       |               |               |               |               |
|               |         Memory retention (POW, EXP)          |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|      ADO      |                 $0.43,0.89$                  |  $0.61,0.80$  |  $0.74,0.76$  |  $0.91,0.78$  |  $0.96,0.82$  |
|    MINEBED    |                 $0.53,0.68$                  |  $0.60,0.55$  |  $0.58,0.50$  |  $0.96,0.11$  |  $0.93,0.03$  |
|    BOSMOS     |                 $0.30,0.96$                  |  $0.33,0.96$  |  $0.22,0.96$  |  $0.26,0.96$  |  $0.24,0.96$  |
|     LBIRD     |                 $0.37,0.91$                  |  $0.48,0.84$  |  $0.69,0.87$  |  $0.94,0.70$  |  $0.98,0.82$  |
|     Prior     |       Baseline for 0 trials: $0.5,0.5$       |               |               |               |               |
|               |          Signal detection (PPO, PR)          |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|    MINEBED    |                 $0.02,0.54$                  |  $0.21,0.39$  |  $0.31,0.22$  |  $0.33,0.17$  |  $0.33,0.17$  |
|    BOSMOS     |                 $0.77,0.76$                  |  $0.92,0.52$  |  $0.94,0.46$  |  $0.92,0.43$  |  $0.92,0.43$  |
|     Prior     |       Baseline for 0 trials: $0.5,0.5$       |               |               |               |               |
|               |       Risky choice (EU, WEU; OPT, CPT)       |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|      ADO      |                $0.26,0.39 ;$                 | $0.11,0.39 ;$ | $0.22,0.64 ;$ | $0.07,0.71 ;$ | $0.11,0.86 ;$ |
|               |                 $0.13,0.19$                  |  $0.35,0.24$  |  $0.35,0.38$  |  $0.52,0.57$  |  $0.70,0.81$  |
|    MINEBED    |                $0.38,0.19 ;$                 | $0.50,0.27 ;$ | $0.58,0.23 ;$ | $0.38,0.38 ;$ | $0.38,0.15 ;$ |
|               |                 $0.12,0.29$                  |  $0.29,0.33$  |  $0.35,0.24$  |  $0.12,0.05$  |  $0.18,0.14$  |
|    BOSMOS     |                $0.48,0.36 ;$                 | $0.48,0.36 ;$ | $0.52,0.50 ;$ | $0.56,0.36 ;$ | $0.48,0.32 ;$ |
|               |                 $0.13,0.05$                  |  $0.09,0.14$  |  $0.13,0.29$  |  $0.09,0.33$  |  $0.09,0.33$  |
| MINEBED (BIC) |                $0.55,0.00 ;$                 | $0.00,0.14 ;$ | $0.55,0.00 ;$ | $0.55,0.00 ;$ | $0.55,0.00 ;$ |
|               |                 $0.00,0.00$                  |  $0.23,0.12$  |  $0.00,0.00$  |  $0.00,0.00$  |  $0.00,0.00$  |
| BOSMOS (BIC)  |                $1.00,0.00 ;$                 | $0.33,0.36 ;$ | $0.52,0.43 ;$ | $0.59,0.29 ;$ | $0.63,0.29 ;$ |
|               |                 $0.00,0.00$                  |  $0.04,0.00$  |  $0.00,0.05$  |  $0.22,0.05$  |  $0.22,0.10$  |
|     LBIRD     |                $0.33,0.43 ;$                 | $0.30,0.43 ;$ | $0.33,0.43 ;$ | $0.33,0.39 ;$ | $0.19,0.50 ;$ |
|               |                 $0.17,0.14$                  |  $0.13,0.29$  |  $0.13,0.38$  |  $0.39,0.29$  |  $0.57,0.67$  |
|     Prior     | Baseline for 0 trials: $0.25,0.25,0.25,0.25$ |               |               |               |               |

Table 4: Convergence of model prediction accuracy $\eta_{\mathrm{m}}$ (proportion of correct predictions of models across 100 simulated participants) for comparison methods (rows) with increased number of trials (columns).

|         | Demonstrative example |   Memory retention   |   Signal detection   |     Risky choice     |
| :-----: | :-------------------: | :------------------: | :------------------: | :------------------: |
|   ADO   |   $10.46 \pm 1.06$    |   $75.44 \pm 9.04$   |          -           |  $134.00 \pm 17.08$  |
| MINEBED |  $786.47 \pm 87.63$   | $3614.11 \pm 272.79$ | $6757.20 \pm 399.90$ | $6698.34 \pm 310.16$ |
| BOSMOS  |    $7.65 \pm 0.39$    |   $35.56 \pm 3.54$   |   $73.41 \pm 8.94$   |   $88.32 \pm 5.75$   |
|  LBIRD  |    $3.16 \pm 0.37$    |   $20.07 \pm 1.32$   |          -           |  $35.95 \pm 12.58$   |

Table 5: Empirical time cost (mean $\pm$ std. in minutes across 100 simulated participants) of applying comparison methods (rows) in cognitive tasks (columns) with 100 sequential designs. ADO and LBIRD for the signal detection task need available likelihoods and therefore cannot be used for this task.