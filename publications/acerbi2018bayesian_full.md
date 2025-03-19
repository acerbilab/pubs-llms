```
@article{acerbi2018bayesian,
    author = {Acerbi, Luigi AND Dokka, Kalpana AND Angelaki, Dora E. AND Ma, Wei Ji},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception},
    year = {2018},
    month = {07},
    volume = {14},
    pages = {1-38},
    number = {7},
    doi = {10.1371/journal.pcbi.1006110}
}
```

---

#### Page 1

# Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception

Luigi Acerbi1☯¤\*, Kalpana Dokka2☯, Dora E. Angelaki2, Wei Ji Ma1,3
1 Center for Neural Science, New York University, New York, NY, United States of America, 2 Department of
Neuroscience, Baylor College of Medicine, Houston, TX, United States of America, 3 Department of
Psychology, New York University, New York, NY, United States of America
☯ These authors contributed equally to this work.
¤ Current address: De ́partement des neurosciences fondamentales, Universite ́ de Genève, Genève,
Switzerland

- luigi.acerbi@nyu.edu, luigi.acerbi@gmail.com

## Abstract

The precision of multisensory perception improves when cues arising from the same cause are integrated, such as visual and vestibular heading cues for an observer moving through a stationary environment. In order to determine how the cues should be processed, the brain must infer the causal relationship underlying the multisensory cues. In heading perception, however, it is unclear whether observers follow the Bayesian strategy, a simpler non-Bayesian heuristic, or even perform causal inference at all. We developed an efficient and robust computational framework to perform Bayesian model comparison of causal inference strategies, which incorporates a number of alternative assumptions about the observers. With this framework, we investigated whether human observers' performance in an explicit cause attribution and an implicit heading discrimination task can be modeled as a causal inference process. In the explicit causal inference task, all subjects accounted for cue disparity when reporting judgments of common cause, although not necessarily all in a Bayesian fashion. By contrast, but in agreement with previous findings, data from the heading discrimination task only could not rule out that several of the same observers were adopting a forced-fusion strategy, whereby cues are integrated regardless of disparity. Only when we combined evidence from both tasks we were able to rule out forced-fusion in the heading discrimination task. Crucially, findings were robust across a number of variants of models and analyses. Our results demonstrate that our proposed computational framework allows researchers to ask complex questions within a rigorous Bayesian framework that accounts for parameter and model uncertainty.

## Author summary

As we interact with objects and people in the environment, we are constantly exposed to numerous sensory stimuli. For safe navigation and meaningful interaction with entities in

---

#### Page 2

the environment, our brain must determine if the sensory inputs arose from a common or different causes in order to determine whether they should be integrated into a unified percept. However, how our brain performs such a causal inference process is not well understood, partly due to the lack of computational tools that can address the complex repertoire of assumptions required for modeling human perception. We have developed a set of computational algorithms that characterize the causal inference process within a quantitative model based framework. We have tested the efficacy of our methods in predicting how human observers judge visual-vestibular heading. Specifically, our algorithms perform rigorous comparison of alternative models of causal inference that encompass a wide repertoire of assumptions observers may have about their internal noise or stimulus statistics. Importantly, our tools are widely applicable to modeling other processes that characterize perception.

## Introduction

We constantly interact with people and objects around us. As a consequence, our brain receives information from multiple senses as well as multiple inputs from the same sense. Cues from the same sense (e.g., texture and disparity cues to an object shape) are generally congruent as they usually reflect identical properties of a common external entity. Thus, the brain eventually learns to mandatorily integrate inputs from the same modality as a unified percept, which provides more precise information than either cue alone [1, 2]. Similarly, integration of cues represented in different modalities but associated with a common stimulus also improves perceptual behavior. There is a wealth of evidence that demonstrates increased precision [312], greater accuracy $[13,14]$ and faster speed $[15,16]$ of perceptual performance due to multimodal integration.

However, multimodal cues present a complex problem. Cues from different modalities are not necessarily congruent as different stimuli can simultaneously impinge on our senses, giving rise to coincident yet conflicting information. For example, in a classic ventriloquist illusion, even though the sound originates from the puppeteer's mouth, we perceive that it is the puppet which is talking [17]. Mandatory integration of multimodal cues arising from different stimuli can induce errors in perceptual estimates [6, 14]. Thus, for efficient interaction with the world, the brain must assess whether the multimodal cues originated from the same cause, and should be integrated into a single percept, or instead the cues should be interpreted in isolation as they arose from different causes (segregation). Despite the often overwhelming amount of sensory inputs, we are typically able to integrate relevant cues while ignoring irrelevant sensory input. It is thus plausible that our brain infers the causal relationship between multisensory cues to determine if and how the cues should be integrated.

Bayesian causal inference-inference of the causal relationship between observed cues, based on the inversion of the statistical model of the task-has been proposed as the decision strategy adopted by the brain to address the problem of integration vs. segregation of sensory cues $[18,19]$. Such a decision strategy has described human performance in spatial localization [18-27], orientation judgment [28], oddity detection [29], speech perception [30], time-interval perception [31], simple perceptual organization [32], and heading perception [33, 34]. In recent years, interest in the Bayesian approach to causal inference has further increased as neural imaging has identified a hierarchy of brain areas involved in neural processing while observers implemented a Bayesian strategy to perform a causal inference task [20]. At the same time, Bayesian models have become more complex as they include more precise

---

#### Page 3

descriptions of the sensory noise [22, 33, 34] and alternative Bayesian decision strategies [21, 24]. However, it is still unknown whether observers fully implement Bayesian causal inference, or merely an approximation that does not take into account the full statistical structure of the task. For example, the Bayes-optimal inference strategy ought to incorporate sensory uncertainty into its decision rule. On the other hand, a suboptimal heuristic decision rule may disregard sensory uncertainty [32, 35, 36]. Thus, the growing complexity of models and the need to consider alternative hypotheses require an efficient computational framework to address these open questions while avoiding trappings such as overfitting or lack of model identifiability [37]. For a more detailed overview of open issues in multisensory perception and causal inference at the intersection of behavior, neurophysiology and computational modeling, we refer the reader to $[38-40]$.

# Visuo-vestibular integration in heading perception

Visuo-vestibular integration in heading perception presents an ideal case to characterize the details of the causal inference strategy in multisensory perception. While a wealth of published studies have shown that integration of visual and vestibular self-motion cues increases perceptual precision [9-12, 14, 41-43], and accuracy [14], such an integration only makes sense if the two cues arise from the same cause-that is optic flow and inertial motion signal heading in the same direction. Despite the putative relevance of causal inference in heading perception, the inference strategies that characterize visuo-vestibular integration in the presence of sensory conflict remain poorly understood. For example, a recent study has found that observers predominantly integrated visual and vestibular cues even in the presence of large spatial discrepancies [33]-whereas a subsequent work has presented evidence in favor of causal inference [34]. Furthermore, these studies did not vary cue reliability-a manipulation that is critical to test whether a Bayes-optimal inference strategy or a suboptimal approximation was used [35].

Another aspect that can influence the choice of inference strategy is the type of inference performed by the observer. In particular, de Winkel and colleagues [33, 34] asked subjects to indicate the perceived direction of inertial heading-an 'implicit' causal inference task as subjects implicitly assessed the causal relationship between visual and vestibular cues on their way to indicate the final (integrated or segregated) heading percept. Even in the presence of spatial disparities as high as $90^{\circ}$, one study found that several subjects were best described by a model which fully integrated visual and vestibular cues [33] (possibly influenced by the experimental design; see also [34]). It is plausible that performing an explicit causal inference task, which forces subjects to indicate whether visual and vestibular cues arose from the same or different events, may elicit different inference strategies, as previously reported in category-based induction [44], multi-cue judgment [45], and sensorimotor decision-making [46]. While some studies have tested both explicit and implicit causal inference [18, 21, 47], to our knowledge only one previous study contemplated the possibility of different strategies between implicit and explicit causal inference tasks [21], and a systematic comparison of inference strategies in the two tasks has never been carried out within a larger computational framework.

## Bayesian comparison of causal inference strategies

Thus, the goal of this work is two-fold. First, we introduce a set of techniques to perform robust, efficient Bayesian factorial model comparison of a variety of Bayesian and non-Bayesian models of causal inference in multisensory perception. Factorial comparison is a way to simultaneously test different orthogonal hypotheses about the observers [21, 48-50]. Our approach is fully Bayesian in that we consider both parameter and model uncertainty, improving over previous analyses which used point estimates for the parameters and compared

---

#### Page 4

individual models. A full account of uncertainty in both parameter and model space, by marginalizing over parameters and model components, is particularly prudent when dealing with internal processes, such as decision strategies, which may have different latent explanations. An analysis that disregards such uncertainty might produce unwarranted conclusions about the internal processes that generated the observed behavior [37]. Second, we demonstrate our methods by quantitatively comparing the decision strategies underlying explicit and implicit causal inference in visuo-vestibular heading perception within the framework of Bayesian model comparison. We found that even though the study of explicit and implicit causal inference in isolation might suggest different inference rules, a joint analysis that combines all available evidence points to no difference between tasks, with subjects performing some form of causal inference in both the explicit and implicit tasks that used identical experimental setups.

In sum, we demonstrate how state-of-the-art techniques for model building, fitting, and comparison, combined with advanced analysis tools, allow us to ask nuanced questions about the observer's decision strategies in causal inference. Importantly, these methods come with a number of diagnostics, sanity checks and a rigorous quantification of uncertainty that allow the experimenter to be explicit about the weight of evidence.

# Results

## Computational framework

We compiled a diverse set of computational techniques to perform robust Bayesian comparison of models of causal inference in multisensory perception, which we dub the 'Bayesian cookbook for causal inference in multisensory perception', or herein simply 'the cookbook'. The main goal of the cookbook is to characterize observers' decision strategies underlying causal inference, and possibly other details thereof, within a rigorous Bayesian framework that accounts for both parameter uncertainty and model uncertainty. The cookbook is 'doublyBayesian' in that it affords a fully Bayesian analysis of observers who may or may not be performing Bayesian inference themselves [51]. Fully Bayesian model comparison is computationally intensive, hence the cookbook is concerned with efficient algorithmic solutions.

The cookbook comprises of: (a) a fairly general recipe for building observer models for causal inference in multisensory perception (see Methods and Section 1 of S1 Appendix), which lends itself to a factorial model comparison; (b) techniques for fast evaluation of a large number of causal inference observer models; (c) procedures for model fitting via maximum likelihood, and approximating the Bayesian posterior of the parameters via Markov Chain Monte Carlo (MCMC); (d) state-of-the-art methods to compute model comparison metrics and perform factorial model selection. It is noteworthy that, while the current work focuses on the example of visuo-vestibular heading perception, this cookbook is general and can be applied with minor modifications to multisensory perception across sensory domains. Computational details are described in the Methods section and S1 Appendix. Here we present an application of our framework to causal inference in multisensory heading perception. For ease of reference, we summarize relevant abbreviations used in the paper and their meaning in Table 1.

## Causal inference in heading perception

We demonstrate our framework taking as a case study the comparison of explicit vs. implicit causal inference strategies in heading perception. In this section we briefly summarize our methods. Extended details and description of the cookbook can be found in the Methods and S1 Appendix.

---

#### Page 5

Table 1. Abbreviations and symbols.

|              Abbreviation              |                                   Meaning                                    |              Context               |
| :------------------------------------: | :--------------------------------------------------------------------------: | :--------------------------------: |
|                                        |                                   General                                    |                                    |
|                $\Delta$                |                    Directional disparity between stimuli                     |          Generative model          |
| $s_{\text {vis }}, s_{\text {vest }}$  |                         Visual / vestibular heading                          |          Generative model          |
| $s_{\text {visc }}, s_{\text {vest }}$ |               Noisy measurement of visual / vestibular heading               |          Generative model          |
|                   C                    | Causal scenario ( $\mathrm{C}=1$ for 'same', $\mathrm{C}=2$ for 'different') |          Generative model          |
|           $c_{\text {vis }}$           |                Visual coherence level (low, medium, or high)                 |          Generative model          |
|               $\rho_{c}$               |                 Probability of common cause (Bayesian model)                 |           Observer model           |
|              $\kappa_{c}$              |              Criterion for common cause (fixed-criterion model)              |           Observer model           |
|                                        |                                Model factors                                 |                                    |
|                  Bay                   |                              Bayesian strategy                               |     Causal inference strategy      |
|                  Fix                   |                           Fixed-criterion strategy                           |     Causal inference strategy      |
|                  Fus                   |                               Fusion strategy                                |     Causal inference strategy      |
|                   -C                   |                                Constant noise                                |        Sensory noise shape         |
|                   -X                   |                         Eccentricity-dependent noise                         |        Sensory noise shape         |
|                   -E                   |                               Empirical prior                                |             Prior type             |
|                   -I                   |                              Independent priors                              |             Prior type             |
|                                        |                         Model fitting and comparison                         |                                    |
|                 AIC(c)                 |                  (corrected) Akaike's Information Criterion                  |      Model comparison metric       |
|                  BIC                   |                        Bayesian Information Criterion                        |      Model comparison metric       |
|                  LML                   |                           Log marginal likelihood                            |      Model comparison metric       |
|                  LOO                   |                                Leave-one-out                                 |      Model comparison metric       |
|                  MCMC                  |                           Markov Chain Monte Carlo                           |      Model fitting technique       |
|            $\hat{\varphi}$             |                       Protected exceedance probability                       | Bayesian model selection statistic |
|                  BOR                   |                            Bayesian Omnibus Risk                             | Bayesian model selection statistic |

List of abbreviations and symbols used in the paper, with associated description and usage context.
https://doi.org/10.1371/journal.pcbi. 1006110.1001

Experiments. Human observers were presented with synchronous visual ( $s_{\text {vis }}$ ) and vestibular ( $s_{\text {vest }}$ ) headings in the same direction $(C=1)$ or in different directions $(C=2)$ separated by a directional disparity $\Delta$ (Fig 1A). Mean stimulus direction $\left(-25^{\circ},-20^{\circ},-15^{\circ}, \ldots, 25^{\circ}\right)$, cue disparity $\left(0^{\circ}, \pm 5^{\circ}, \pm 10^{\circ}, \pm 20^{\circ}\right.$, and $\left. \pm 40^{\circ}\right)$, and visual cue reliability $c_{\text {vis }}$ (coherence: high, medium and low) changed randomly on a trial-by-trial basis (Fig 1B). On each trial, non-zero disparity was either positive (vestibular heading to the right of visual heading) or negative. Observers ( $n=11$ ) first performed several sessions of an explicit causal inference task ('unity judgment'), in which they indicated if the visual and vestibular stimuli signaled heading in the same direction ('common cause') or in different directions ('different causes'). The same observers then participated in a number of sessions of the implicit causal inference task ('inertial left/right discrimination') wherein they indicated if their perceived inertial heading (vestibular) was to the left or right of straight forward. Both tasks consisted of a binary classification (same/different or left/right) with identical experimental apparatus and stimuli. No feedback was given to subjects about the correctness of their response. All observers also performed a number of practice trials and an initial session of a 'unisensory left/right discrimination' task in which they reported heading direction (left or right of straight forward) of visual or vestibular stimuli presented in isolation. For each subject we obtained $350-750$ trials of the unisensory discrimination task ( 1 session), 700-1200 trials of the unity judgment task (2-3 sessions), and 21003000 trials of the inertial discrimination task (7-9 sessions).

---

#### Page 6

> **Image description.** This image contains two panels, labeled A and B, which illustrate an experiment layout and stimulus distribution.
>
> Panel A: Depicts two head outlines viewed from above.
>
> - Left side (C=1): Three red arrows emanate from the head, one pointing straight ahead, labeled "s = 0°", and the other two angled to the left (-) and right (+).
> - Right side (C=2): Two arrows emanate from the head, one black (Svis) and one red (Svest), with a small double-headed arrow labeled "Δ" indicating the angular difference between them.
>
> Panel B: Contains a scatter plot and two bar charts.
>
> - Scatter plot: The x-axis is labeled "Svis" and ranges from -45° to 45°. The y-axis is labeled "Svest" and ranges from -45° to 45°. Gray squares are scattered across the plot, forming a correlated pattern. Black squares are positioned along a dashed diagonal line.
> - Top right bar chart: Titled "Causal scenario". The x-axis is labeled "C" and has values 1 and 2. The y-axis is labeled "Probability" and ranges from 0 to 0.8. A black bar is at C=1 with a probability around 0.2, and a gray bar is at C=2 with a probability around 0.8.
> - Bottom right bar chart: Titled "Visual cue reliability". The x-axis is labeled "cvis" and has labels "low", "med", and "high". The y-axis is labeled "Probability" and ranges from 0 to 1/3. Three bars are shown: a yellow bar for "low", a brown bar for "med", and a dark brown bar for "high", all at approximately the same probability level (1/3).

Fig 1. Experiment layout. A: Subjects were presented with visual $\left(x_{\text {vis }}\right)$ and vestibular $\left(x_{\text {vis }}\right)$ headings either in the same direction $(C=1)$ or in different directions $(C=2)$. In different sessions, subjects were asked to judge whether stimuli had the same cause ('unity judgment', explicit causal inference) or whether the vestibular heading was to the left or right of straight forward ('inertial discrimination', implicit causal inference). B: Distribution of stimuli used in the task. Mean stimulus direction was drawn from a discrete uniform distribution $\left(-25^{\circ},-20^{\circ},-15^{\circ}, \ldots, 25^{\circ}\right)$. In $20 \%$ of the trials, $x_{\text {vis }} \equiv x_{\text {vest }}$ ('same' trials, $C=1$ ); in the other $80 \%$ ('different', $C=2$ ), disparity was drawn from a discrete uniform distribution $\left( \pm 5^{\circ}, \pm 10^{\circ}, \pm 20^{\circ}, \pm 40^{\circ}\right)$, which led to a correlated pattern of heading directions $x_{\text {vis }}$ and $x_{\text {vest }}$. Visual cue reliability $c_{\text {vis }}$ was also drawn randomly on each trial (high, medium, and low).
https://doi.org/10.1371/journal.pcbi.1006110.g001

Theory. For each task we built a set of observer models by factorially combining three model components-hence also called model factors-that represent different assumptions about the observers: shape of sensory noise, type of prior over stimuli, and causal inference strategy (Fig 2A).

In each trial of the explicit and implicit causal inference tasks, two stimuli are presented: a visual heading $s_{\text {vis }}$ with known reliability $c_{\text {vis }} \in\{$ high, medium, low $\}$, and a vestibular heading $s_{\text {vest }}$. We assume that stimuli $s_{\text {vis }}, s_{\text {vest }}$ induce noisy measurements $x_{\text {vis }}$ (resp., $x_{\text {vest }}$ ) with conditionally independent distributions $p\left(x_{\text {vis }} \mid s_{\text {vis }}, c_{\text {vis }}\right)$ and $p\left(x_{\text {vest }} \mid s_{\text {vest }}\right)$. For any stimulus $s$ we assume that the noise distribution is a (wrapped) Gaussian centered on $s$ and with variance $\sigma^{2}(s)$. For each observer model we consider a variant in which $\sigma^{2}$ depends only on the stimulus modality and reliability (constant, ' $C$ ') and a variant in which $\sigma^{2}(s)$ also depends on stimulus location, growing with heading eccentricity, that is with the distance from $0^{\circ}$ (eccentricitydependent, ' X '; see Methods). With a few notable exceptions [22, 33, 34], stimulus-dependence in the noise has been generally ignored in previous work [18, 20, 21, 24, 27]. The base noise magnitude is governed by model parameters $\sigma_{0 \text { vest }}$ and $\sigma_{0 \text { vis }}\left(c_{\text {vis }}\right)$, where the latter is one parameter per visual reliability level. The eccentricity-dependent noise model has additional parameters $w_{\text {vest }}$ and $w_{\text {vis }}$ which govern the growth of noise with heading eccentricity (see Methods and S1 Appendix for details). We assume that the noise distribution equally affects both the generative model and the observer's decision model, that is, observers have an approximately correct model of their own sensory noise $[4,6,9]$.

We assume that the observer considers two causal scenarios [18]: either there is a single common heading direction $(C=1)$ or the two stimuli correspond to distinct headings $(C=2)$ [18] (Fig 2B). If $C=1$, the observer believes that the measurements are generated from the same underlying source $s$ with prior distribution $p_{\text {prior }}(s)$. If $C=2$, stimuli are believed to be distinct, but not necessarily statistically independent, with prior distribution $p_{\text {prior }}\left(s_{\text {vis }}, s_{\text {vest }}\right)$. For the type of these priors, we consider an empirical ('E') observer whose priors correspond to an approximation of the discrete, correlated distribution of stimuli in the task (as per

---

#### Page 7

> **Image description.** This image is a 3D diagram illustrating different observer models based on three factors: causal inference strategy, shape of sensory noise, and type of prior over stimuli. The diagram is presented as a cube with each axis representing one of these factors.
>
> Here's a breakdown of the visual elements:
>
> - **Overall Structure:** The diagram is structured as a cube, with nodes representing different combinations of the three model factors. The nodes are connected by lines, forming the edges of the cube.
>
> - **Nodes:** Each node is represented by a colored circle. The color of the circle seems to correspond to the causal inference strategy. The circles contain text indicating the combination of factors they represent.
>
>   - Green circles represent Bayesian (Bay) causal inference strategy.
>   - Red/Pink circles represent Fixed-criterion (Fix) causal inference strategy.
>   - Blue/Purple circles represent Fusion (Fus) causal inference strategy.
>   - The second part of the text in each circle indicates the shape of sensory noise: Constant (C), or Eccentricity dependent (X).
>   - The third part of the text in each circle indicates the type of prior over stimuli: Empirical (E) or Independent (I).
>
> - **Axes Labels:**
>
>   - The vertical axis on the left is labeled "Causal inference strategy" and has three levels: "Bayesian (Bay)", "Fixed-criterion (Fix)", and "Fusion (Fus)".
>   - The horizontal axis at the bottom is labeled "Shape of sensory noise" and has two levels: "Constant (C)" and "Eccentricity dependent (X)".
>   - The diagonal axis on the right is labeled "Type of prior over stimuli" and has two levels: "Empirical (E)" and "Independent (I)".
>
> - **Text:** Various text labels are present to identify the different components of the diagram, as described above. The text is generally black and clearly legible. A large "A" is present in the top left corner of the diagram.

- Bayes (High reliability)
- Bayes (Medium reliability)
- Bayes (Low reliability)
  $\rightarrow$ Fixed criterion

# B

> **Image description.** The image contains two panels, labeled B and C.
>
> Panel B shows two directed acyclic graphs, illustrating observer models. The left graph is labeled "C=1" and the right graph is labeled "C=2". Both graphs have the same general structure. At the top is a box labeled "Causal inference strategy". Below that, "C=1" points to a circle labeled "Type of prior over stimuli". Below that, a circle labeled "Svis vest" is present. To the left of "Svis vest" is "Cvis" in a circle. Below "Svis vest" are two circles, "Xvis" and "Xvest". "Cvis" points to "Xvis", and "Svis vest" points to both "Xvis" and "Xvest". Below the "Svis vest", a box is labeled "Shape of sensory noise". The right graph is similar, but "Svis" and "Svest" are in separate circles that overlap. "Cvis" points to "Xvis", "Svis" points to "Xvis", and "Svest" points to "Xvest". The circles "Cvis", "Xvis", and "Xvest" are shaded gray.
>
> Panel C shows a plot with x and y axes ranging from -45 to 45 degrees. The x-axis is labeled "xvis" and the y-axis is labeled "xvest". A dashed line runs diagonally from the bottom left to the top right. There are three pairs of curved lines, each pair in a different shade of brown/yellow. Each pair of curved lines is roughly parallel to the dashed line. A thicker dashed line also runs diagonally from the bottom left to the top right, but is slightly offset from the thinner dashed line.

Fig 2. Observer models. A: Observer models consist of three model factors: Causal inference strategy, Shape of sensory noise, and Type of prior over stimuli (see text). B: Graphical representation of the observer model. In the left panel $(C=1)$, the visual $\left(x_{\text {vis }}\right)$ and vestibular $\left(x_{\text {vest }}\right)$ heading direction have a single, common cause. In the right panel ( $\mathrm{C}=2$ ), $x_{\text {vis }}$ and $x_{\text {vest }}$ have separate sources, although not necessarily statistically independent. The observer has access to noisy sensory measurements $x_{\text {vis }}, x_{\text {vest }}$, and knows the visual reliability level of the trial $c_{\text {vis }}$. The observer is either asked to infer the causal structure (unity judgment, explicit causal inference), or whether the vestibular stimulus is rightward of straight ahead (inertial discrimination, implicit causal inference). Model factors affect different stages of the observer model: the strategy used to combine the two causal scenarios; the type of prior over stimuli $p_{\text {prior }}\left(x_{\text {vis }}, x_{\text {vest }}\right) C$; and the shape of sensory noise distributions $p\left(x_{\text {vis }}\right) c_{\text {vis }}, c_{\text {vis }}$ ) and $p\left(x_{\text {vest }}\right) c_{\text {vest }}$ ) (which affects equally both how noisy measurements are generated and the observer's beliefs about such noise). C: Example decision boundaries for the Bay-X-E model (for the three reliability levels), and for the Fix model, for a representative observer. The observer reports 'unity' when the noisy measurements $x_{\text {vis }}, x_{\text {vest }}$ fall within the boundaries. Note that the Bayesian decision boundaries expand with larger noise. Nonlinearities are due to the interaction between eccentricitydependence of the noise and the prior (wiggles are due to the discrete empirical prior).
https://doi.org/10.1371/journal.pcbi.1006110.g002

Fig 1B); and an independent ( T ) observer who uses a common and independent uni-dimensional Gaussian prior centered on $0^{\circ}$ for the two stimuli.

Parameter $\sigma_{\text {prior }}$ represents the SD of each independent prior (for ' $T$ ' priors), or of the prior over mean stimulus direction (for ' $E$ ' priors); whereas $\Delta_{\text {prior }}$ governs the SD of the prior over disparity ('E' priors only). See Methods for details.

We assume that observers are Bayesian in dealing with each causal scenario $(C=1$ or $C=2$ ), but may follow different strategies for weighting and combining information from the two causal hypotheses. Specifically, we consider three families of causal inference strategies. The Bayesian ('Bay') strategy computes the posterior probability of each causal scenario $\operatorname{Pr}\left(C \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)$ based on all information available in the trial. The fixed-criterion ('Fix')

---

#### Page 8

strategy decides based on a fixed threshold of disparity between the noisy visual and vestibular measurements, disregarding reliability and other statistics of the stimuli. Finally, the fusion ('Fus') strategy disregards any location information, either always combining cues, or combining them with some probability (depending on whether the task involves implicit or explicit causal inference).

In the explicit causal inference task, the Bayesian ('Bay') observer reports a common cause if its posterior probability is greater than $0.5, \operatorname{Pr}\left(C=1 \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)>0.5$. The prior probability of common cause, $p_{\mathrm{c}} \equiv \operatorname{Pr}(C=1)$, is a free parameter of the model. The fixed-criterion ('Fix') observer reports a common cause whenever the two noisy measurements are closer than a fixed distance $\kappa_{c}$, that is $\left|x_{\text {vis }}-x_{\text {vest }}\right|<\kappa_{c}$, where the criterion $\kappa_{c}$ is a free parameter that does not depend on stimulus reliability [36]. The fixed-criterion decision rule differs fundamentally from the Bayesian one in that it does not take cue reliability and other stimulus statistics into account (although noise will still affect behavior). As an example, Fig 2C shows the decision boundaries for the Bayesian (constant noise, empirical prior) and fixed-criterion rule for a representative observer. Finally, as a variant of the 'fusion' strategy we consider an observer that does not perform causal inference at all, but simply reports unity with probability $\eta\left(c_{\text {vis }}\right)$ regardless of stimulus disparity, where $\eta_{\text {low }}, \eta_{\text {med }}, \eta_{\text {high }}$ are the only parameters of the model (stochastic fusion, 'SFu'). This variant generalizes a trivial 'forced fusion' strategy $(\eta \equiv 1)$ that would always report a common cause in the explicit inference.

For the implicit causal inference task, the observer first computes the posterior probability of rightward vestibular motion, $\operatorname{Pr}\left(s_{\text {vest }}>0^{\prime} \mid x_{\text {vest }}, x_{\text {vis }}, c_{\text {vis }}, C=k\right)$ for the two causal scenarios, $k=1,2$. The Bayesian ('Bay') observer then reports 'right' if the posterior probability of rightward vestibular heading, averaged over the Bayesian posterior over causal structures, is greater than 0.5 . The fixed-criterion ('Fix') observer reports 'right' if $\operatorname{Pr}\left(s_{\text {vest }}>0^{\prime} \mid x_{\text {vest }}, x_{\text {vis }}, c_{\text {vis }}, C=\right.$ $\left.k_{\text {fix }}\right)>0.5$, where $k_{\text {fix }}=1$ if $\left|x_{\text {vis }}-x_{\text {vest }}\right|<\kappa_{c}$, and $k_{\text {fix }}=2$ otherwise. Finally, for the Fusion strategy we consider here the forced fusion ('FFu') observer, for which $C \equiv 1$. The forced fusion observer is equivalent to a Bayesian observer with $p_{\mathrm{c}} \equiv 1$, and to a fixed-criterion observer for $\kappa_{c} \rightarrow \infty$.

Observers also performed a unisensory left/right heading discrimination task, in which either a visual or vestibular heading was presented on each trial. In this case observers were modeled as standard Bayesian observers that respond 'right' if $\operatorname{Pr}\left(s_{\text {vis }}>0^{\prime} \mid x_{\text {vis }}, c_{\text {vis }}\right)>0.5$ for visual trials, and if $\operatorname{Pr}\left(s_{\text {vest }}>0^{\prime} \mid x_{\text {vest }}\right)>0.5$ for vestibular trials. These data were used to constrain the joint model fits (see below).

For all observer models and tasks (except stochastic fusion in the explicit task), we considered a lapse probability $0 \leq \lambda \leq 1$ of the observer giving a random response. Finally, we note that the Bayesian observer models considered in our main analysis perform Bayesian model averaging (the proper Bayesian strategy). At the end of the Results section we will also consider a 'probability matching' suboptimal Bayesian observer [24].

Analysis strategy. Our analysis strategy consisted of first examining subjects' behavior separately in the explicit and implicit tasks via model fitting and comparison. We then compared the model fits across tasks to ensure that model parameters were broadly compatible, allowing us to aggregate data from different tasks without changing the structure of the models. Finally, we re-analyzed observers' performance by jointly fitting data from all three tasks (explicit causal inference, implicit causal inference, and unisensory heading discrimination), thereby combining all available evidence to characterize subjects' decision making processes.

Given the large number of models and distinct datasets involved, we coded each model using efficient computational techniques at each step (see Methods for details).

We fitted our models to the data first via maximum-likelihood estimation, and then via Bayesian estimation of the posterior over parameters using Markov Chain Monte Carlo

---

#### Page 9

(MCMC). Posteriors are an improvement over point estimates in that they allow us to incorporate uncertainty over individual subjects' model parameters in our analysis, and afford computation of more accurate comparison metrics (see below).

We computed for each task, subject, and model the leave-one-out cross-validation score (LOO) directly estimated from the MCMC output [52] (reported in S1 Appendix). LOO has several advantages over other model selection metrics in that it takes parameter uncertainty into account and provides a more accurate measure of predictive performance [53] (see Discussion). We combined model evidence (LOO scores) from different subjects and models using a hierarchical Bayesian approach for group studies [54]. For each model component within the model factors of interest (noise, prior, and causal inference strategy), we reported as the main summary statistic of the analysis the protected exceedence probability $\hat{\varphi}$, that is the (posterior) probability of a model component being the most likely component, above and beyond chance [55]. As a test of robustness, we also computed additional model comparison metrics: the corrected Akaike's information criterion (AICc), the Bayesian information criterion (BIC), and an estimate of the log marginal likelihood (LML). While we prefer LOO as the main metric (see Discussion), we verified that the results of the model comparison were largely invariant of the choice of comparison metric.

Finally, for each model we estimated the absolute goodness of fit as the fraction of information gain above chance (where $0 \%$ is chance and $100 \%$ is the estimated intrinsic variability of the data, that is the entropy [56]).

# Explicit causal inference task

We examined how subjects perceived the causal relationship of synchronous visual and vestibular headings as a function of disparity ( $s_{\text {vest }}-s_{\text {vis }}$, nine levels) and visual reliability level (high, medium, low; Fig 3A). Common cause reports were more frequent near zero disparities than for well-separated stimuli (Repeated-measures ANOVA with Greenhouse-Geisser correction; $\left.F_{(1.82,18.17)}=76.0, \epsilon=0.23, p<10^{-4}, \eta_{p}^{2}=0.88\right)$. This means that observers neither performed complete integration (always reporting a common cause) nor complete segregation (never reporting a common cause). Common-cause reports were not affected by visual cue reliability alone $\left(F_{(1.23,12.33)}=1.84, \epsilon=0.62, p=.2, \eta_{p}^{2}=0.16\right)$, but were modulated by an interaction of visual reliability and disparity $\left(F_{(7.44,74.44)}=7.38, \epsilon=0.47, p<10^{-4}, \eta_{p}^{2}=0.42\right)$. Thus, observers' performance was affected by both cue disparity as well as visual cue reliability when explicitly reporting about the causal relationship between visual and vestibular cues. However, this does not necessarily mean that the subjects' causal inference strategy took visual cue reliability into account. Changes in sensory noise may affect measured behavior even if the observer's decision rule ignores such changes [35]; a quantitative model comparison is needed to probe this question.

We compared a subset of models from the full factorial comparison (Fig 2A), since some models are equivalent when restricted to the explicit causal inference task. In particular, here fixed-criterion models are not influenced by the 'prior' factor, and the (stochastic) fusion model is not affected by sensory noise or prior, thus reducing the list of models to seven: Bay-C-E, Bay-C-I, Bay-X-E, Bay-X-I, Fix-C, Fix-X, SFu.

To assess the evidence for distinct determinants of subjects' behavior, we combined LOO scores from individual subjects and models with a hierarchical Bayesian approach [54] (Fig 3B). Since we are investigating model factors that comprise of an unequal number of models, we reweighted the prior over models such that distinct components within each model factor had equal prior probability (Fix models had $2 \times$ weight, and SFu $4 \times$ ). In Fig 3B we report the protected exceedance probabilities $\hat{\varphi}$ and, for reference, the posterior model frequencies they

---

#### Page 10

> **Image description.** This image presents results from an explicit causal inference task, displayed across three panels (A, B, and C).
>
> **Panel A:** This panel shows a scatter plot with error bars. The x-axis is labeled "Stimulus disparity" with values ranging from -40° to 40° in increments of 20°. The y-axis is labeled "Fraction responses 'unity'" and ranges from 0 to 1. Data is presented for three conditions: "High reliability" (dark brown), "Medium reliability" (medium brown), and "Low reliability" (yellow). Each condition shows the fraction of 'unity' responses at different stimulus disparities. The error bars represent $\pm 1$ SEM across subjects. A dashed horizontal line is present at y=0.5.
>
> **Panel B:** This panel contains three bar charts. Each chart displays "Probability" on the y-axis, ranging from 0 to 1. The first chart is titled "Causal inference strategy" and presents data for "Bayes", "Fixed", and "Fusion". The bars represent the "Protected exceedance probability $\hat{\varphi}$" (black), and the gray dots represent "Posterior frequency". The Bayesian omnibus risk (BOR) is given as BOR = 0.12. The second chart is titled "Sensory noise" and presents data for "Constant" and "Eccentric". The BOR is given as BOR = 0.96. The third chart is titled "Prior" and presents data for "Empirical" and "Independent". The BOR is given as BOR = 0.78.
>
> **Panel C:** This panel contains four scatter plots with error bars. The x-axis is labeled "Stimulus disparity" with values ranging from -40° to 40° in increments of 20°. The y-axis is labeled "Fraction responses 'unity'" and ranges from 0 to 1. Each plot represents a different model: "Bay-X-E" (0.76), "Bay-C-I" (0.75), "Fix-C" (0.73), and "SFu" (0.17). The numbers in parentheses indicate the absolute goodness of fit. Each model shows the fraction of 'unity' responses at different stimulus disparities, with data for "High reliability" (dark brown), "Medium reliability" (medium brown), and "Low reliability" (yellow). Shaded areas represent $\pm 1$ SEM of model predictions across subjects. A dashed horizontal line is present at y=0.5 in each plot.

Fig 3. Explicit causal inference. Results of the explicit causal inference (unity judgment) task. A: Proportion of 'unity' responses, as a function of stimulus disparity (difference between vestibular and visual heading direction), and for different levels of visual cue reliability. Bars are $\pm 1$ SEM across subjects. Unity judgments are modulated by stimulus disparity and visual cue reliability. B: Protected exceedance probability $\hat{\varphi}$ and estimated posterior frequency (mean $\pm$ SD) of distinct model components for each model factor. Each factor also displays the Bayesian omnibus risk (BOR). C: Model fits of several models of interest (see text for details). Shaded areas are $\pm 1$ SEM of model predictions across subjects. Numbers on top right of each panel report the absolute goodness of fit.
https://doi.org/10.1371/journal.pcbi. 1006110 . g003
are based on, and the Bayesian omnibus risk (BOR), which is the estimated probability that the observed differences in factor frequencies may be due to chance [55]. We found that the most likely factor of causal inference was the Bayesian model $(\hat{\varphi}=0.78)$, followed by fixed-criterion $(\hat{\varphi}=0.18)$ and probabilistic fusion $(\hat{\varphi}=0.04)$. That is, fusion was $\sim 24$ times less likely to be the most representative model than any form of causal inference combined, which is strong evidence against fusion, and in agreement with our model-free analysis. The Bayesian strategy was $\sim 3.5$ times more likely than the others, which is positive but not strong evidence [57]. Conversely, the explicit causal inference data do not allow us to draw conclusions about noise models (constant vs. eccentric) or priors (empirical vs. independent), as we found that all factor components are about equally likely ( $\hat{\varphi} \sim 0.5$ ).

At the level of specific models-as opposed to aggregate model factors -, we found that the probability of being the most likely model was almost equally divided between fixed-criterion (C-I) and Bayesian (either X-E or C-I). All these models yielded reasonable fits (Fig 3C), which captured a large fraction of the noise in the data (absolute goodness of fit $\approx 76 \% \pm 3 \%$; see Methods); a large improvement over a constant-probability model, which had a goodness of fit of $14 \pm 5 \%$. For comparison, we also show in Fig 3C the stochastic fusion model, which had a goodness of fit of $17 \% \pm 5 \%$. Visually, the Fix model in Fig 3C seems to fit better the group data, but we found that this is an artifact of projecting the data on the disparity axis. Disparity is the only relevant dimension for the Fix model; whereas Bay models fits the data along all dimensions. The visual superiority of the Fix model wanes when the data are visualized in their entirety (see S1 Fig).

---

#### Page 11

We verified robustness of our findings by performing the same hierarchical analysis with different model comparison metrics. All metrics were in agreement with respect to the Bayesian causal inference strategy as the most likely, and the same three models being most probable (although possibly with different ranking). BIC and marginal likelihood differed from LOO and AICc mainly in that they reported a larger probability for the constant vs. eccentricitydependent noise (probability ratio $\sim 4.6$, which is positive but not strong evidence).

These results combined provide strong evidence that subjects in the explicit causal inference task took into account some elements of the statistical structure of the trial (disparity, and possibly cue reliability) to report unity judgments, consistent with causal inference, potentially in a Bayesian manner. From these data, it is unclear whether observers took into account the empirical distribution of stimuli, and whether their behavior was affected by eccentricitydependence in the sensory noise.

# Implicit causal inference task

We examined the bias in the reported direction of inertial heading computed as (minus) the point of subjective equality for left/rightward heading choices (L/R PSE), for each visual heading and visual cue reliability (Fig 4A). Specifically, for a given value of visual heading $s_{\text {vis }}$ (or small range thereof), we constructed a psychometric function as a function of $s_{\text {vest }}$ (see Methods for details). If subjects were influenced by $s_{\text {vis }}$ and took visual heading into account while

> **Image description.** This image presents results from an implicit causal inference task, displayed across three panels (A, B, and C).
>
> Panel A:
>
> - This panel shows a scatter plot of vestibular bias as a function of visual heading direction ($s_{vis}$).
> - The x-axis represents $s_{vis}$ ranging from -40 to 40 degrees.
> - The y-axis represents vestibular bias, ranging from -15 to 15 degrees. A dashed horizontal line marks 0 degrees.
> - Data points are shown for three levels of visual reliability: high (dark brown), medium (brown), and low (yellow). Error bars represent ±1 SEM across subjects.
> - The inset shows a psychometric function, plotting Pr(right) against $s_{vest}$. Three sigmoid curves are displayed, presumably corresponding to different visual heading directions. A vertical dashed line indicates $s_{vis} = -25^\circ$. The point of subjective equality (L/R PSE) is marked on one of the curves. The text "Bias = -PSE" and "$s_{vis} = -25^\circ$" are also present.
>
> Panel B:
>
> - This panel contains three bar charts showing probabilities for different models.
> - The first chart, labeled "Causal inference strategy," shows probabilities for "Bayes," "Fixed," and "Fusion" models. The Bayes model has a probability around 0.25, Fixed is slightly lower, and Fusion is the highest at around 0.5. "BOR = 0.72" is written above the chart.
> - The second chart, labeled "Sensory noise," shows probabilities for "Constant" and "Eccentric" noise models. Eccentric noise has a much higher probability than Constant. "BOR = 0.29" is written above the chart.
> - The third chart, labeled "Prior," shows probabilities for "Empirical" and "Independent" priors. The Empirical prior has a higher probability than the Independent prior. "BOR = 0.65" is written above the chart.
> - Each bar chart has a y-axis representing probability, ranging from 0 to 1. Error bars are shown on each bar.
> - The legend indicates that black bars represent "Protected exceedance probability $\tilde{\varphi}$" and gray dots represent "Posterior frequency".
>
> Panel C:
>
> - This panel shows four scatter plots of vestibular bias as a function of visual heading direction ($s_{vis}$), similar to Panel A.
> - Each plot corresponds to a different model: "FFu-X-E," "Bay-X-E," "Fix-C-E," and "Bay-C-E." The value "0.97" is written above each plot.
> - The x-axis represents $s_{vis}$ ranging from -40 to 40 degrees.
> - The y-axis represents vestibular bias, ranging from -15 to 15 degrees. A dashed horizontal line marks 0 degrees.
> - Data points are shown for three levels of visual reliability: high (dark brown), medium (brown), and low (yellow). Error bars represent ±1 SEM across subjects. Shaded areas around the data points indicate confidence intervals.

Fig 4. Implicit causal inference. Results of the implicit causal inference (left/right inertial discrimination) task. A: Vestibular bias as a function of copresented visual heading direction $s_{\text {vis }}$, at different levels of visual reliability. Bars are $\pm 1$ SEM across subjects. The inset shows a cartoon of how the vestibular bias is computed as minus the point of subjective equality of the psychometric curves of left/right responses (L/R PSE) for vestibular stimuli $s_{\text {vest }}$, for a representative subject and for a fixed value of $s_{\text {vis }}$. The vestibular bias is strongly modulated by $s_{\text {vis }}$ and its reliability. B: Protected exceedance probability $\psi$ and estimated posterior frequency (mean $\pm$ SD) of distinct model components for each model factor. Each factor also displays the Bayesian omnibus risk (BOR). C: Model fits of several models of interests (see text for details). Shaded areas are $\pm 1$ SEM of model predictions across subjects. Numbers on top right of each panel report the absolute goodness of fit.

---

#### Page 12

computing inertial heading, this would manifest as bias in the psychometric function (that is, a shifted point of subjective equality). If subjects were able instead to discount the distracting influence of $s_{\mathrm{vis}}$, there should be negligible bias. As per causal inference, we qualitatively expected that there would be bias for smaller $\left|s_{\text {vis }}\right|$, but the bias would either decrease or saturate as $\left|s_{\text {vis }}\right|$ increases. However, note that a nonlinear pattern of bias may also emerge due to eccentricity-dependence of the noise, even in the absence of causal inference.

The bias was significantly affected by visual heading (Repeated-measures ANOVA; $\left.F_{(0.71,7.08)}=19.67, \epsilon=0.07, p=.004, \eta_{p}^{2}=0.66\right)$. We found no main effect of visual cue reliability alone $\left(F_{(0.85,8.54)}=0.51, \epsilon=0.43, p=.47, \eta_{p}^{2}=0.05\right)$, but there was a significant interaction of visual cue reliability and heading $\left(F_{(2.93,29.26)}=7.36, \epsilon=0.15, p<10^{-3}, \eta_{p}^{2}=0.42\right)$. These data suggest that subjects' perception of vestibular headings was modulated by visual cue reliability and visual stimulus, in agreement with previous work in visual-auditory localization [21]. However, quantitative model comparison is required to understand the mechanism in detail since distinct processes, such as different causal inference strategies and noise models, could lead to similar patterns of observed behavior.

We performed a factorial comparison with all models in Fig 2A. In this case, factorial model comparison via LOO was unable to uniquely identify the causal inference strategy adopted by observers (Fig 4B). Forced fusion was slightly favored $(\bar{\varphi} \sim 0.48)$, followed by Bayes $(\bar{\varphi} \sim 0.27)$ and fixed-criterion $(\bar{\varphi} \sim 0.25)$, suggesting that all strategies were similar to forced fusion. Conversely, eccentricity-dependent noise was found to be more likely than constant noise (ratio $\sim 5.7$ ), which is positive but not strong evidence, and empirical priors were marginally more likely than independent priors ( $\sim 2.1$ ). The estimated Bayesian omnibus risk was high ( $\mathrm{BOR} \geq 0.29$ ), hinting at a large degree of similarity within all model factors such that observed differences could have arisen by chance.

All metrics generally agreed on the lack of evidence in favor of any specific inference strategy (with AICc and BIC tending to marginally favor fixed-criterion instead of fusion), and on empirical priors being more likely. As a notable difference, marginal likelihood and BIC reversed the result about noise models, favoring constant noise models over eccentricitydependent ones.

In terms of individual models, the most likely models according to LOO were, in order, forced fusion (X-E), Bayesian (X-E), and fixed-criterion (C-E). However, other metrics also favored other models; for example, Bayesian (C-E) was most likely according to the marginal likelihood. All these models obtained similarly good fits to individual data (Fig 4C; absolute goodness of fit $\approx 97 \%$ ). For reference, a model that responds 'rightward motion' with constant probability performed about at chance (goodness of fit $\approx 0.3 \pm 0.1 \%$ ).

In sum, our analysis shows that the implicit causal inference data alone are largely inconclusive, possibly because almost all models behave similarly to forced fusion. To further explore our results, we examined the posterior distribution of the prior probability of common cause parameter $p_{\mathrm{c}}$ across Bayesian models, and of the criterion $\kappa_{\mathrm{c}}$ for fixed-criterion models (Fig 5, bottom left panels). In both cases we found a broad distribution of parameters, with only a mild accumulation towards 'forced fusion' values ( $p_{\mathrm{c}}=1$ or $\kappa_{\mathrm{c}} \geq 90^{\circ}$ ), suggesting that subjects were not completely performing forced fusion. Thus, it is possible that by constraining the inference with additional data we would be able to draw more defined conclusions.

# Joint model fits

Data from the explicit and implicit causal inference tasks, when analyzed separately, afforded only weak conclusions about subjects' behavior. The natural next step is to combine datasets

---

#### Page 13

> **Image description.** The image consists of eight small plots arranged in a 2x4 grid, displaying posterior distributions over model parameters. Each plot represents a different parameter, and shows the marginal posterior distributions for each subject and task.
>
> Each plot has the following general structure:
>
> - **Y-axis:** Labeled "Tasks", indicating the different experimental conditions.
> - **X-axis:** Represents the parameter value, with different scales and ranges depending on the parameter. Numerical values are marked on the x-axis.
> - **Data representation:** Each subject's posterior distribution is shown as a horizontal line. A thicker line represents the interquartile range, and a thinner, lighter line represents the 95% credible interval. Different colors are used to distinguish the tasks: green for unisensory discrimination, blue for bisensory implicit causal inference, and light blue for bisensory explicit causal inference. Black lines indicate joint datasets.
> - **Title:** Each plot has a title indicating the parameter being analyzed (e.g., "σ0vest", "wvest", "pc"). A value labeled "Cp =" is displayed at the top of each plot, representing the across-tasks compatibility probability.
>
> The parameters represented in the plots are (from left to right, top to bottom):
>
> 1.  σ0vest (with x-axis ranging from 1° to 10°, Cp = 0.07)
> 2.  σ0vis (chigh) (with x-axis ranging from 1° to 30°, Cp = 0.14)
> 3.  σ0vis (cmed) (with x-axis ranging from 1° to 30°, Cp = 0.15)
> 4.  σ0vis (clow) (with x-axis ranging from 1° to 30°, Cp = 0.32)
> 5.  wvest (with x-axis ranging from 0 to 0.8, Cp = 0.75)
> 6.  wvis (with x-axis ranging from 0 to 0.8, Cp = 0.07)
> 7.  λ (with x-axis ranging from 0 to 0.2, Cp = 0.23)
> 8.  pc (with x-axis ranging from 0 to 1, Cp = 0.23)
> 9.  κc (with x-axis ranging from 0.3° to 100°, Cp = 0.20)
> 10. σprior (with x-axis ranging from 1° to 100°, Cp = 0.29)
> 11. Δprior (with x-axis ranging from 1° to 100°, Cp = 0.33)
>
> A legend on the right side of the image clarifies the color coding for each task and the representation of joint datasets.

Fig 5. Posteriors over model parameters. Each panel shows the marginal posterior distributions over a single parameter for each subject and task. Each line is an individual subject's posterior (thick line: interquartile range; light line: $95 \%$ credible interval); different colors correspond to different tasks. For each subject and task, posteriors are marginalized over models according to their posterior probability (see Methods). For each parameter we report the across-tasks compatibility probability $C_{p}$, that is the (posterior) probability that subjects were best described by the assumption that parameter values were the same across separate tasks, above and beyond chance. The first two rows of parameters compute compatibility across all three tasks, whereas in the last row compatibility only includes the bisensory tasks (bisensory inertial discrimination and unity judgment), as these parameters are irrelevant for the unisensory task.
https://doi.org/10.1371/journal.pcbi. 1006110 . g005
from the two tasks along with the data from the unisensory heading discrimination task in order to better constrain the model fits.

Before performing such joint fit, we verified whether there was evidence that model parameters changed substantially across tasks, in which case we might have had to change the structure of the models (e.g., by introducing a subset of distinct parameters for different tasks [49]). For each model parameter, we computed the across-tasks compatibility probability $C_{p}$ (Fig 5), which is the (posterior) probability that subjects were most likely to have the same parameter values across tasks, as opposed to different parameters, above and beyond chance (see Methods for details). We found at most mild evidence towards difference of parameters across the three tasks, but no strong evidence (all $C_{p}>.05$ ). Therefore, we proceeded in jointly fitting the data with the default assumption that parameters were shared across tasks.

For the joint fits there are nine possible models for the causal inference strategy (three explicit causal inference $\times$ three implicit causal inference strategies). However, we considered only a subset of plausible combinations, to avoid 'model overfitting' (see Discussion). First, we disregarded the stochastic fusion strategy for the explicit task, since this strategy was strongly rejected by the explicit task data alone. Second, if subjects performed some form of causal inference (Bayesian or fixed-criterion) in both tasks, we forced it to be the same. This reduces the model space for the causal inference strategy to four components: Bay/Bay, Fix/Fix, Bay/ $\mathrm{FFu}, \mathrm{Fix} / \mathrm{FFu}$ (explicit/implicit task). Combined with the prior and sensory noise factors as per Fig 2A, this leads to sixteen models.

---

#### Page 14

> **Image description.** This image presents a figure with three panels (A, B, and C) displaying results of joint fits across tasks related to causal inference.
>
> **Panel A:** This panel contains three bar charts, each representing a different model factor: "Causal inference strategy," "Noise," and "Prior." Each bar chart has the y-axis labeled "Probability" ranging from 0 to 1.
> _ The "Causal inference strategy" chart shows bars for "Bayes," "Fixed," "Bayes/Fusion," and "Fixed/Fusion." The "Fixed" bar is the tallest. The Bayesian omnibus risk (BOR) is displayed as "BOR = 0.14."
> _ The "Noise" chart shows bars for "Constant" and "Eccentric." The "Eccentric" bar is much taller. The BOR is displayed as "BOR = 0.01."
> _ The "Prior" chart shows bars for "Empirical" and "Independent." The "Empirical" bar is taller. The BOR is displayed as "BOR = 0.45."
> _ A legend indicates that the black bars represent "Protected exceedance probability $\hat{\varphi}$" and gray circles represent "Posterior frequency."
>
> **Panel B:** This panel shows four line graphs representing joint model fits of the explicit causal inference (unity judgment) task. Each graph plots "Fraction responses 'unity'" (y-axis, ranging from 0 to 1) as a function of "Stimulus disparity" (x-axis, ranging from -40° to 40°).
> _ Each graph corresponds to a different model: "Fix-X-E," "Bay-X-E," "Bay/FFu-X-E," and "Bay/FFu-X-I." The absolute goodness of fit is indicated at the top right of each panel (0.91, 0.90, 0.91, and 0.90 respectively).
> _ Each graph displays three lines representing different levels of visual reliability: "High reliability" (dark brown), "Medium reliability" (brown), and "Low reliability" (yellow). Shaded areas around the lines represent ±1 SEM of model predictions across subjects. Error bars are shown at various points along each line. A dashed horizontal line is drawn at y = 0.5.
>
> **Panel C:** This panel shows four line graphs representing joint model fits of the implicit causal inference task, corresponding to the same models as in Panel B. Each graph plots "Vestibular bias" (y-axis, ranging from -15° to 15°) as a function of "s_vis" (x-axis, ranging from -40° to 40°). \* The graphs display the same three lines representing different levels of visual reliability (dark brown, brown, and yellow) with shaded areas representing ±1 SEM of model predictions across subjects. Error bars are shown at various points along each line. A dashed horizontal line is drawn at y = 0.

Fig 6. Joint fits. Results of the joint fits across tasks. A: Protected exceedance probability $\hat{\varphi}$ and estimated posterior frequency (mean $\pm \mathrm{SD}$ ) of distinct model components for each model factor. Each factor also displays the Bayesian omnibus risk (BOR). B: Joint model fits of the explicit causal inference (unity judgment) task, for different models of interest. Each panel shows the proportion of 'unity' responses, as a function of stimulus disparity and for different levels of visual reliability. Bars are $\pm 1$ SEM of data across subjects. Shaded areas are $\pm 1$ SEM of model predictions across subjects. Numbers on top right of each panel report the absolute goodness of fit across all tasks. C: Joint model fits of the implicit causal inference task, for the same models of panel B. Panels show vestibular bias as a function of co-presented visual heading direction $s_{\text {vis }}$, and for different levels of visual reliability. Bars are $\pm 1$ SEM of data across subjects. Shaded areas are $\pm 1$ SEM of model predictions across subjects.
https://doi.org/10.1371/journal.pcbi. 1006110 . g006

Factorial model comparison via LOO found that the most likely causal inference strategy was fixed-criterion $(\hat{\varphi}=0.79)$, followed by Bayesian $(\hat{\varphi}=0.13)$, and then by forced fusion in the implicit task ( $\hat{\varphi}=0.05$ paired with Bayesian explicit causal inference, $\hat{\varphi}=0.03$ paired with fixed-criterion explicit causal inference; Fig 6A). This is positive evidence that subjects were performing some form of causal inference also in the implicit task, as opposed to mere forced fusion (ratio $\sim 11.4$ ). Moreover, we found strong evidence for eccentricity-dependent over constant noise $(\hat{\varphi}>0.99$, ratio $\sim 132.7$ ). Instead, the joint data were still inconclusive about the prior adopted by the subjects, with only marginal evidence for the empirical prior over the independent prior $(\sim 2.9)$.

In terms of specific models, the most likely model was fixed-criterion (X-E), followed by Bayesian (X-E), and explicit Bayesian / implicit forced fusion (both X-I and X-E). The best models gave a good description of the individual joint data, with an absolute goodness of fit of $\approx 91 \% \pm 1 \%$ (Fig 6B).

Examination of the subjects' posteriors over parameters for the joint fits (Table 2 and Fig 5, black lines) showed reasonable results. The base visual noise parameters were generally monotonically increasing with decreasing visual cue reliability; the vestibular base noise was roughly of the same magnitude as the medium visual cue noise (as per experiment design); both visual and vestibular noise increased mildly with the distance from straight ahead; subjects had a small lapse probability. For Bayesian models, $p_{\mathrm{c}}$ was substantially larger than the true value,

---

#### Page 15

Table 2. Joint fit parameters.

|                  Parameter                  |             Description              |          Posterior mean           |                  Allowed range                   |
| :-----------------------------------------: | :----------------------------------: | :-------------------------------: | :----------------------------------------------: |
|                  All tasks                  |                                      |                                   |                                                  |
|          $\sigma_{\text {ftext }}$          |        Vestibular base noise         |  $6.49^{\circ} \pm 0.90^{\circ}$  |  $\left[0.5^{\circ}, 80^{\circ}\right] \dagger$  |
| $\sigma_{\text {fres }}(c_{\text {high }})$ |  Visual base noise (high coherence)  |  $4.08^{\circ} \pm 0.54^{\circ}$  |  $\left[0.5^{\circ}, 80^{\circ}\right] \dagger$  |
| $\sigma_{\text {fres }}(c_{\text {med }}$ ) | Visual base noise (medium coherence) |  $6.32^{\circ} \pm 1.00^{\circ}$  |  $\left[0.5^{\circ}, 80^{\circ}\right] \dagger$  |
| $\sigma_{\text {fres }}(c_{\text {low }}$ ) |  Visual base noise (low coherence)   | $11.57^{\circ} \pm 2.67^{\circ}$  |  $\left[0.5^{\circ}, 80^{\circ}\right] \dagger$  |
|             $w_{\text {rest }}$             |    Vestibular noise eccentricity     |          $0.04 \pm 0.01$          |                     $[0,1]$                      |
|             $w_{\text {vis }}$              |      Visual noise eccentricity       |          $0.07 \pm 0.02$          |                     $[0,1]$                      |
|                  $\lambda$                  |              Lapse rate              |          $0.01 \pm 0.01$          |                     $[0,1]$                      |
|               Bisensory only                |                                      |                                   |                                                  |
|                   $p_{c}$                   |  Prior of common cause (Bay models)  |          $0.56 \pm 0.05$          |                     $[0,1]$                      |
|                $\kappa_{c}$                 |     Fixed criterion (Fix models)     | $26.50^{\circ} \pm 3.52^{\circ}$  | $\left[0.25^{\circ}, 180^{\circ}\right] \dagger$ |
|          $\sigma_{\text {prior }}$          |         Central prior width          | $49.77^{\circ} \pm 12.08^{\circ}$ |  $\left[1^{\circ}, 120^{\circ}\right] \dagger$   |
|          $\Delta_{\text {prior }}$          |        Disparity prior width         | $23.51^{\circ} \pm 6.39^{\circ}$  |  $\left[1^{\circ}, 120^{\circ}\right] \dagger$   |

Posterior means of parameters in the joint fit, marginalized over models according to each subject's posterior model probability, and averaged across subjects ( $\pm$ SEM). For reference, we also report the parameter range used for the optimization and MCMC sampling.
${ }^{\dagger}$ These parameters were transformed and fitted in log space.
https://doi.org/10.1371/journal.pcbi. 1006110.1002
$0.20(t$-test $t_{(10)}=10.8, p<10^{-4}, d=3.25)$, suggesting that observers generally thought that heading directions had a higher a priori chance to be the same. Nonetheless, for all but one subject $p_{c}$ was far from 1 , suggesting that subjects were not performing forced fusion either. An analogous result holds for the fixed criterion $\kappa_{c}$, which was smaller than the largest disparity between heading directions. We found that prior parameters $\sigma_{\text {prior }}$ and $\Delta_{\text {prior }}$ had a lesser impact on the models, and their exact values were less crucial, with generally wide posteriors.

Finally, we verified that our results did not depend on the chosen comparison metric. Remarkably, the findings regarding causal inference factors were quantitatively the same for all metrics, demonstrating robustness of our main result. Marginal likelihood and BIC differed from LOO and AICc in that they only marginally favored eccentricity-dependent noise models, showing that conclusions over the noise model may depend on the specific choice of metric. All metrics agreed in marginally preferring the empirical prior over the independent prior.

In conclusion, when combining evidence from all available data, our model comparison shows that subjects were most likely performing some form of causal inference instead of forced fusion, for both the explicit and the implicit causal inference tasks. In particular, we find that a fixed-criterion, non-probabilistic decision rule (i.e., one that does not take uncertainty into account) describes the joint data better than the Bayesian strategy, although with some caveats (see Discussion).

# Sensitivity analysis and model validation

Performing a factorial comparison, like any other statistical analysis, requires a number of somewhat arbitrary choices, loosely motivated by previous studies, theoretical considerations, or a preliminary investigation of the data (being aware of the 'garden of forking paths' [58]). As good practice, we want to check that our main findings are robust to changes in the setup of the analysis, or be able to report discrepancies.

We take as our main result the protected exceedance probabilties $\hat{\varphi}$ of the model factors in the joint analysis (Fig 6A, reproduced in Fig 7, top row). In the following, we examine whether this finding holds up to several manipulations of the analysis framework.

---

#### Page 16

> **Image description.** The image is a figure containing five rows of bar charts, each row representing a different variant of a factorial comparison in a sensitivity analysis. The figure is organized into three columns, each representing a different model factor: "Causal inference strategy," "Sensory noise," and "Prior." Each bar chart within a row shows the protected exceedance probability of distinct model components for the corresponding model factor.
>
> Each row is labeled on the left with a description of the variant:
>
> - Row 1: "Main"
> - Row 2: "Marginal likelihood"
> - Row 3: "Hyperprior α₀ = 1"
> - Row 4: "Bayesian probability matching (replaced)"
> - Row 5: "Bayesian probability matching (subfactor)"
>
> Each column is labeled at the top with the model factor it represents:
>
> - Column 1: "Causal inference strategy"
> - Column 2: "Sensory noise"
> - Column 3: "Prior"
>
> Each bar chart has a vertical axis labeled "Probability" ranging from 0 to 1 in increments of 0.25. The horizontal axis of each bar chart represents different model components within the factor.
>
> - For "Causal inference strategy," the components are "Bayes," "Fixed," "Bayes/Fusion," and "Fixed/Fusion."
> - For "Sensory noise," the components are "Constant" and "Eccentric."
> - For "Prior," the components are "Empirical" and "Independent."
>
> Each bar chart displays black bars representing the protected exceedance probability for each model component. A gray dot with error bars is superimposed on each bar, representing the estimated posterior frequency (mean ± SD). The Bayesian omnibus risk (BOR) value is displayed at the top right of each bar chart.
>
> A legend at the top left indicates that the black bars represent "Protected exceedance probability φ̂" and the gray dots represent "Posterior frequency."

Fig 7. Sensitivity analysis of factorial model comparison. Protected exceedance probability $\hat{\varphi}$ of distinct model components for each model factor in the joint fits. Each panel also shows the estimated posterior frequency (mean $\pm \mathrm{SD}$ ) of distinct model components, and the Bayesian omnibus risk (BOR). Each row represents a variant of the factorial comparison. 1st row: Main analysis (as per Fig 6A). 2nd row: Uses marginal likelihood as model comparison metric. 3rd row: Uses hyperprior $\alpha_{0}=1$ for the frequencies over models in the population (instead of a flat prior over model factors). 4th row: Uses 'probability matching' strategy for the Bayesian causal inference model (replacing model averaging). 5th row: Includes probability matching as a sub-factor of the Bayesian causal inference family (in addition to model averaging).
https://doi.org/10.1371/journal.pcbi.1006110.g007

A first check consists of testing different model comparison metrics. In the previous sections, we have reported results for different metrics, finding in general only minor differences from our results obtained with LOO. As an example, we show here the model comparison using as metric an estimate of the marginal likelihood-the probability of the data under the model (Fig 7, 2nd row). We see that the marginal likelihood results agree with our results with LOO except for the sensory noise factor (see Discussion). Therefore, our conclusions about the causal inference strategy are not affected.

Second, the hierarchical Bayesian Model Selection method requires to specify a prior over frequencies of models in the population [54]. This (hyper)prior is specified via the concentration parameter vector $\boldsymbol{\alpha}_{0}$ of a Dirichlet distribution over model frequencies. For our analysis, since we focused on the factorial aspect, we chose an approximately 'flat' prior across model factors (see Methods for details), instead of the default flat prior over individual models ( $\alpha_{0}=$ 1). We found that performing the group analysis with $\alpha_{0}=1$ did not change our results (Fig 7, 3rd row).

Another potential source of variation is specific model choices, or inclusion of model factors. For example, a common successful variant of the Bayesian causal inference strategy is 'probability matching', according to which the observer chooses the causal scenario $(C=1$ or $C=2$ ) randomly, proportionally to its posterior probability [24]. As a first check, we performed the model comparison again using a 'probability matching' Bayesian observer instead of our main 'model averaging' observer (Fig 7, 4th row). Results are similar to the main analysis. If anything, the fixed-criterion causal inference strategy gains additional evidence here,

---

#### Page 17

suggesting that probability matching is a worse description of the data than our original Bayesian causal inference model (as confirmed by looking at differences in LOO scores of individual subjects, e.g. for the Bay-X-E model; mean $\pm$ SEM: $\Delta \mathrm{LOO}=-17.3 \pm 5.7$ ). A recent study in audio-visual causal inference perception has similarly found that probability matching provided a poor explanation of the data [21].

In the factorial framework we could also have performed the previous analysis in a different way, by considering 'probability matching' as a sub-factor of the Bayesian strategy, together with 'model averaging'. As we have done before for the explicit causal inference task, we reassign prior probabilities to the models so that they are constant for each factor (in this case, the two Bayesian strategies get a $\times \frac{1}{2}$ multiplier). Results of this alternative approach show an increase of evidence for the Bayesian causal inference family (Fig 7, bottom row). The values of $\hat{\varphi}$ for the fusion models are also slightly higher, which is due to an increase of the Bayesian omnibus risk (the probability that the observed differences in factor frequencies are due to chance, a warning sign that there are too many models for the available data). This result and other lines of reasoning suggest caution when model factors contain an uneven number of models (see Discussion). Nonetheless, the main conclusion does not qualitatively change, in that observers performed some form of causal inference as opposed to forced fusion.

Finally, we performed several sanity checks, including a model recovery analysis to ensure the integrity of our analysis pipeline and that models of interest were meaningfully distinguishable (see Methods and S1 Appendix for details).

In conclusion, we have shown how the computational framework of Bayesian factorial model comparison, which is made possible by a combination of methods described in the cookbook, allows to explore multiple questions about aspects of subjects' behavior in multisensory perception, and to account for uncertainty at different levels of the analysis in a principled, robust manner.

# Discussion

We presented a 'cookbook' of algorithmic recipes for robust Bayesian evaluation of observer models of causal inference that have widespread applications to multisensory perception and modeling perceptual behavior in general. We applied these techniques to investigate the decision strategies that characterize explicit and implicit causal inference in multisensory heading perception. Examination of observers' behavior in the explicit and implicit causal inference tasks provided evidence that observers did not simply fuse visual and vestibular cues. Instead, observers integrated the multisensory cues based on their relative disparity, a signature of causal inference. Importantly, our framework affords investigation of whether humans adopt a statistically optimal Bayesian strategy or instead implement a heuristic decision rule which does not fully consider the uncertainty associated with the stimuli.

## Causal inference in multisensory heading perception

Our findings in the explicit causal inference task demonstrate that subjects used information about the discrepancy between the visual and vestibular cues to infer the causal relationship between them. Results in the implicit causal inference task alone were mixed, in that we could not clearly distinguish between alternative strategies, including forced fusion-in agreement with a previous finding [33]. However, when we combined evidence from all tasks, we found that some form of causal inference was more likely than mere forced fusion, in agreement with a more recent study [34]. Our findings suggest that multiple sources of evidence (e.g., different tasks) can help disambiguate causal inference strategies which might otherwise produce similar patterns of behavioral responses.

---

#### Page 18

Our Bayesian analysis allowed us to examine the distribution of model parameters, in particular the causal inference parameters $p_{\mathrm{c}}$ and $\kappa_{\mathrm{c}}$, which govern the tendency to bind or separate cues for, respectively, a Bayesian and a heuristic fixed-criterion strategy. Evidence from all tasks strongly constrained these parameters for each subject. Interestingly, for the Bayesian models we found an average $p_{\mathrm{c}}$ much higher than the true experimental value (inferred $p_{\mathrm{c}} \sim 0.5$ vs. experimental $p_{\mathrm{c}}=0.2$ ). This suggests that subjects had a tendency to integrate sensory cues substantially more than what the statistics of the task would require. Note that, instead, a Bayesian observer would be able to learn the correct value of $p_{\mathrm{c}}$ from noisy observations, provided some knowledge of the structure of the task. Our finding is in agreement with previous studies which demonstrated an increased tendency to combine discrepant visual and vestibular cues [10,33, $43,59,60]$ and also a large inter-subject variability in $p_{\mathrm{c}}$, and not obviously related to the statistics of the task [23]. We note that, in all studies so far, the 'binding tendency' $\left(p_{\mathrm{c}}\right.$ or $\left.\kappa_{\mathrm{c}}\right)$ is a descriptive parameter of causal inference models that lacks an independent empirical correlate (as opposed to, for example, noise parameters, which can be independently measured). Understanding the origin of the binding tendency, and which experimental manipulations it is sensitive to, is venue for future work [23,61]. For example, de Winkel and colleagues found that the binding tendency depends on the duration of the motion stimuli; decreasing for motions of longer duration [34].

Previous work has performed a factorial comparison of only causal inference strategies [21]. Our analysis extends that work by including as latent factors the shape of sensory noise (and, thus, likelihoods) and type of priors [48, 49]. Models in our set include a full computation of the observers' posterior beliefs based on eccentricity-dependent likelihoods, which was only approximated in previous studies that considered eccentricity-dependence [22,33,34]. Indeed, in agreement with a recent finding, we found an important role of eccentricity-dependent noise [22]. Conversely, our analysis of priors was inconclusive, as our datasets were unable to tell whether people learnt the empirical (correlated) prior, or made an assumption of independence.

Our main finding, relative to the causal inference strategy, is that subjects performed causal inference both in the explicit and implicit tasks. Interestingly, from our analyses the most likely causal inference strategy is a fixed-criterion strategy, which crucially differs from the Bayesian strategy in that it does not take cue reliability into account-let alone optimally. This finding is seemingly at odds with a long list of results in multisensory perception, in which people are shown to take cue uncertainty into account [9, 10, 42, 62]. We note that this is not necessarily in contrast with existing literature, for several reasons. First, this result pertains specifically to the causal inference part of the observer model, and not how cues are combined once a common cause has been inferred [21]. To our knowledge, no study of multisensory perception has tested Bayesian models of causal inference against heuristic models that take into account disparity but not reliability, as it has been done for example in visual search [56,63] and visual categorization [36,64]. A quantitative modeling approach is needed-qualitatively analyzing the differences in behavior at different levels of reliability is not sufficient to establish that observers take uncertainty into account; patterns of observed differences may be due to a change in sensory noise even if the observer's decision rule disregards cue reliability. Second, our results are not definitive-the evidence for fixed-criterion vs. Bayesian is positive but not decisive. Our interpretation of this result is that subjects are following some suboptimal decision rule which happens to be closer to fixed-criterion than to the Bayesian strategy for the presented stimuli and range of tested reliability levels. It is possible that with a wider range of stimuli and reliabilities, and possibly with different ways of reporting (e.g., estimation instead of discrimination), we would be able to distinguish the Bayesian strategy from a fixed-criterion heuristic.

---

#### Page 19

Finally, we note that model predictions of our Bayesian models are good but still show systematic discrepancies from the data for the explicit causal inference task (Figs 3C and 6B). Previous work has found similar discrepancies in model fits of unity judgments data across multiple sensory reliabilities (e.g., see Fig 2A in [21]). This suggests that there is some element of model mismatch in current Bayesian causal inference models, possibly due to difference in noise models or to other processes that affect causal inference across cue reliabilities, which deserves further investigation.

# Bayesian factorial comparison

We performed our analysis within a factorial model comparison framework [50]. Even though we were mainly interested in a single factor (causal inference strategy), previous work has shown that the inferred observer's decision strategy might depend on other aspects of the observer model, such as sensory noise or prior, due to nontrivial interactions of all these model components [37]. Our method, therefore, consisted of performing inference across a family of observer models that explicitly instantiated plausible model variants. We then marginalized over details of specific observer models, looking at posterior probabilities of model factors, according to a hierarchical Bayesian Model Selection approach [54, 55]. We applied a few tweaks to the Bayesian Model Selection method to account for our focus on factors as opposed to individual models (see Methods).

Our approach was fully Bayesian in that we took into account parameter uncertainty (by computing a metric, LOO, based on the full posterior distribution) and model uncertainty (by marginalizing over model components). A fully Bayesian approach has the advantages of explicitly representing uncertainty in the results (e.g., credible intervals over parameters), and of reducing the risk of overfitting, although it is not immune to it [65].

In our case, we marginalized over models to reduce the risk of model overfitting, which is a complementary problem to parameter overfitting. Model overfitting is likely to happen when model selection is performed within a large number of discrete models. In fact, some authors recommend to skip discrete model selection altogether, preferring instead inference and Bayesian parameter estimation in a single overarching or 'complete' model [66]. We additionally tried to reduce the risk of model overfitting by balancing prior probabilities across factors, although we noted that this may not be enough to counterbalance the additional flexibility that a model factor gains by having more sub-models than a competitor. Our practical recommendation, until more sophisticated comparison methods are available, is to ensure that all model components within a factor have the same number of models, and to limit the overall number of models.

Our approach was also factorial in the treatment of different tasks, in that first we analyzed each bisensory task in isolation, and then combined trials from all data in a joint fit. The fully Bayesian approach allowed us to compute posterior distributions for the parameters, marginalized over models (see Fig 5), which in turn made it possible to test whether model parameters were compatibile across tasks, via the 'compatibility probability' metric. The compatibility probability is an approximation of a full model comparison to test whether a given parameter is the same or should differ across different datasets (in this case, tasks), where we consider 'sameness' to be the default (simplyfing) hypothesis. We note that if the identity or not of a parameter across datasets is a main question of the study, its resolution should be addressed via a proper model comparison.

With the joint fits, we found that almost all parameters were well constrained by the data (except possibly for the parameters governing the observers' priors, $\sigma_{\text {prior }}$ and $\Delta_{\text {prior }}$ ). An alternative option to better constrain the inference for scarce data or poorly identified parameters

---

#### Page 20

is to use informative priors (as opposed to non-informative priors), or a hierarchical approach that assumes a common (hyper)prior to model parameters across subjects [67].

# Model comparison metrics

The general goal of a model comparison metric is to score a model for goodness of fit and somehow penalize for model flexibility. In our analysis we have used Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO [53]) as the main metric to compare models (simply called LOO in the other sections for simplicity). In fact, there is a large number of commonly used metrics, such as (corrected) Akaike's information criterion (AIC(c)) [68], Bayesian information criterion (BIC) [68], deviance information criterion (DIC) [69], widely applicable information criterion (WAIC) [70], and marginal likelihood [71]. The literature on model comparison is vast and with different schools of thought-by necessity here we only summarize some remarks. The first broad distinction between these metrics is between predictive metrics (AIC(c), DIC, WAIC, and PSIS-LOO) [72], that try to approximate out-of-sample predictive error (that is, model performance on unseen data), and BIC and marginal likelihood, which try to establish the true model generating the data [71]. Another orthogonal distinction is between metrics based on point estimates (AIC(c) and BIC) vs. metrics that use partial to full information about the model's uncertainty landscape (DIC, WAIC, PSIS-LOO, based on the posterior, and the marginal likelihood, based on the likelihood integrated over the prior).

First, when computationally feasible we prefer uncertainty-based metrics to point estimates, since the latter are only crude asymptotic approximations that do not take the model and the data into account, besides simple summary statistics (number of free parameters and possibly number of data points). Due to their lack of knowledge of the actual structure of the model, AIC(c) and BIC can grossly misestimate model complexity [72].

Second, we have an ordered preference among predictive metrics, that is PSIS-LOO > WAIC $>$ DIC $>$ AIC(c) [72]. The reason is that all of these metrics more or less asymptotically approximate full leave-one-out cross validation, with increasing degree of accuracy from right to left [53, 72]. As mentioned before, AIC(c) works only in the regime of a large amount of data. DIC, albeit commonly used, has several issues and requires the posterior to be multivariate normal, or at least symmetric and unimodal—gross failures can happen when this is not the case, since DIC bases its estimate of model complexity on the mean (or some other measure of central tendency) of the posterior [72]. WAIC is a great improvement over DIC and does not require normality of the posterior, but its approximation is generally superseded by PSIS-LOO [53]. Moreover, PSIS-LOO has a natural diagnostic, the exponents of the tails of the fitted Pareto distribution, which allows the user to know when the method may be in trouble [53]. Full leave-one-out cross validation is extremely expensive, but PSIS-LOO only requires the user to compute the posterior via MCMC sampling, with no additional cost with respect to DIC or WAIC. Similarly to WAIC, PSIS-LOO requires the user to store for each posterior sample the log likelihood per trial, which with modern computers represent a negligible storage cost.

The marginal likelihood, or Bayes factor (of which BIC is a poor approximation), is an alternative approach to quantify model evidence, related to computing the posterior probability of the models [71]. While this is a principled approach, it entails several practical and theoretical issues. First, the marginal likelihood is generally hard to compute, since it usually involves a complicated, high-dimensional integral of the likelihood over the prior (although this computation can be simplified for nested models [73]). Here, we have applied a novel approximation method for the marginal likelihood following ideas delineated in [74, 75], obtaining generally

---

#### Page 21

sensible values. However, more work is needed to establish the precision and applicability of such technique. Besides practical computational issues, the marginal likelihood, unlike other metrics, is sensitive to the choice of prior over parameters, in particular its range [66]. Crucially, and against common intuition, this sensitivity does not reduce with increasing amounts of data. A badly chosen (e.g., excessively wide) prior for a non-shared parameter might change the marginal likelihood of a model by several points, thus affecting model ranking. The open issue of prior sensitivity has led some authors to largely discard model selection based on the marginal likelihood [66].

For these reasons, we chose (PSIS-)LOO as the main model comparison metric. As a test of robustness, we also computed other metrics and verified that our results were largely independent of the chosen metric, or investigated the reasons when it was not the case.

As a specific example, in our analysis we found that LOO and marginal likelihood (or BIC) generally agreed on all comparisons, except for the sensory noise factor. Unlike LOO, the marginal likelihood tended to prefer constant noise models as opposed to eccentricity-dependent models. Our explanation of this discrepancy is that for our tasks eccentricity-dependence provides a consistent but small improvement to the goodness of fit of the models, which can be overrided by a large penalty due to model complexity (BIC), or to the chosen prior over the eccentricity-dependent parameters ( $w_{\text {vis }}, w_{\text {vest }}$ ), whose range was possibly wider than needed (see Fig 5). The issue of prior sensitivity (specifically, dependence of results on an arbitrarily chosen range) can be attenuated by adopting a Bayesian hierarchical approach over parameters (or a more computationally feasibile approximation, known as empirical Bayes), which is venue for future work.

# Computational framework

Model evaluation, especially from a Bayesian perspective, is a time-consuming business. For this reason, we have compiled several state-of-the-art methods for model building, fitting and comparison, and made our code available.

The main issue of many common observer models in perception is that the expression for the (log) likelihood is not analytical, requiring numerical integration or simulation. To date, this limits the applicability of modern model specification and analysis tools, such as probabilistic programming languages, that exploit auto-differentiation and gradient-based sampling methods (e.g., Stan [76] or PyMC3 [77]). The goal of such computational frameworks is to remove the burden and technical details of evaluating the models from the shoulders of the modeler, who only needs to provide a model specification.

In our case, we strive towards a more modest goal of providing black-box algorithms for optimization and MCMC sampling that exhibit a larger degree of robustness than standard methods. In particular, for optimization (maximum likelihood estimation) we recommend Bayesian adaptive direct search (BADS [78]), a technique based on Bayesian optimization [79, 80], which exhibits robustness to noise and jagged likelihood landscapes, unlike common optimization methods such as fminsearch (Nelder-Mead) and fmincon in MATLAB. Similarly, for MCMC sampling we propose a sampling method that combines the robustness and self-adaptation of slice sampling [81] and ensemble-based methods [82]. Crucially, our proposed method almost completely removes the need of expensive trial-and-error tuning on the part of the modeler, possibly one of the main reasons why MCMC methods and full evaluation of the posterior are relatively uncommon in the field (to our knowledge, this is the first study of causal inference in multisensory perception to adopt a fully Bayesian approach).

Our framework is similar to the concept behind the VBA toolbox, a MATLAB toolbox for probabilistic treatment of nonlinear models for neurobiological and behavioral data [83].

---

#### Page 22

The VBA toolbox tackles the problem of model fitting via a variational approximation that assumes factorized, Gaussian posteriors over the parameters (mean field/Laplace approximation), and provides the variational free energy as an approximation (lower bound) of the marginal likelihood. Our approach, instead, does not make any strong assumption, using MCMC to recover the full shape of the posterior, and state-of-the-art techniques to assess model performance.

Detailed, rigorous modeling of behavior is a necessary step to constrain the search for neural mechanisms implementing decision strategies [84] We have provided a set of computational tools and demonstrated how they can be applied to answer specific questions about internal representation and decision strategies of the observer in multisensory perception, with the goal of increasing the set of models that can be investigated, and the robustness of such analyses. Thus, our tools can be of profound use not only to the field of multisensory perception, but to biological modeling in general.

# Methods

## Ethics statement

The Institutional Review Board at the Baylor College of Medicine approved the experimental procedures (protocol number H-29411, "Psychophysics of spatial orientation and vestibular influences on spatial constancy and movement planning") and all subjects gave written informed consent.

## Human psychophysics

Subjects. Eleven healthy adults ( 4 female; age $26.4 \pm 4.6$ years, mean $\pm \mathrm{SD}$ ) participated in the full study. Subjects had no previous history of neurological disorders and had normal or corrected-to-normal vision. Four other subjects completed only a partial version of the experiment, and their data were not analyzed here.

Apparatus. Details of the experimental apparatus have been previously published and are only described here briefly [9, 14, 85, 86]. Subjects were seated comfortably in a cockpit-style chair and were protectively restrained with a 5-point racing safety harness. Each subject wore a custom-made thermoplastic mesh mask that was attached to the back of the chair for head stabilization. The chair, a three-chip DLP projector (Galaxy 6; Barco) and a large projection screen $(149 \times 127 \mathrm{~cm})$ were all mounted on a motion platform (6DOF2000E; Moog, Inc.). The projection screen was located $\sim 65 \mathrm{~cm}$ in front of the eyes, subtending a visual angle of $\sim 94^{\circ} \times 84^{\circ}$. Subjects wore LCD-based active 3D stereo shutter glasses (Crystal Eyes 4, RealD, Beverly Hills) to provide stereoscopic depth cues and headphones for providing trial timingrelated feedback (a tone to indicate when a trial was about the begin and another when a button press was registered). This apparatus was capable of providing three self-motion conditions: vestibular (inertial motion through the movement of the platform), visual (optic flow simulating movement of the observer in a 3D virtual cloud of stars, platform stationary) and combined visual-vestibular heading (temporally-synchronized optic flow and platform motion) at various spatial discrepancies.

Stimuli. We modified a previous multisensory heading discrimination task [9]. Here subjects experienced combined visual and vestibular translation in the horizontal plane (Fig 1A). The visual scene and platform movement followed a Gaussian velocity profile (displacement $=13 \mathrm{~cm}$, peak Gaussian velocity $=26 \mathrm{~cm} / \mathrm{s}$ and peak acceleration $=0.9 \mathrm{~m} / \mathrm{s}^{2}$, duration $=$ 1 s ). Visual and vestibular headings were either in the same direction or their movement trajectories were separated by a directional disparity, $\Delta$, expressed in degrees (Fig 1A). The

---

#### Page 23

directional disparity $\Delta$ and visual cue reliability were varied on a trial-by-trial basis. $\Delta$ took one of five values, selected with equal probability: $0^{+}$(no conflict), $5^{+}, 10^{+}, 20^{+}$and $40^{+}$. Thus, visual and vestibular stimuli were in conflict in $80 \%$ of the trials. In each trial, $\Delta$ was randomly assigned to be positive (Fig 1A right, vestibular heading to the right of visual heading) or negative. Once a disparity value, $\Delta$, was chosen, the mean heading angle $(\bar{s})$ which represents the average of vestibular and visual headings, was uniformly randomly drawn from the discrete set $\left\{-25^{+},-20^{+}, \ldots, 25^{+}\right\}$. Vestibular heading ( $s_{\text {vest }}$, red trace in Fig 1) and visual heading ( $s_{\text {vis }}$, black trace in Fig 1A) were generated by displacing the platform motion and optic flow on either side of the mean heading by $\Delta / 2$. The vestibular and visual headings experienced by subjects were defined as $s_{\text {vest }}=\bar{s}+\Delta / 2$ and $s_{\text {vis }}=\bar{s}-\Delta / 2$, respectively. This procedure entailed that visual and vestibular heading directions presented in experiment were correlated (Fig 1B). Three levels of visual cue reliability (high, medium, and low) were tested. Visual reliability was manipulated by varying the percentage of stars in the optic flow that coherently moved in the specified heading direction. For all subjects, visual motion coherence at high reliability was set at $100 \%$. Coherence at medium reliability was selected for each subject during a preliminary session via a manual staircasing procedure such that their visual and vestibular thresholds were approximately matched. Coherence at low reliability was also selected for each subject separately and this was a value that was chosen to be lower than the medium reliability. Thus, the optic flow coherences for medium and low reliabilities were different across subjects with ranges of $40-70 \%$ and $25-50 \%$, respectively. Overall, there were 297 stimulus conditions ( 9 directional disparities $\times 11$ mean heading directions $\times 3$ visual cue reliabilities) which were randomly interleaved.

Tasks. First, subjects $(n=11)$ performed in a session of a unisensory heading discrimination task (left/right of straight ahead), in which visual or vestibular stimuli were presented in isolation. Vestibular stimuli had one fixed reliability level, whereas visual stimuli were tested on three different reliability levels, randomly interleaved, resulting in a total of 350-750 trials.

Then, subjects performed two-three sessions of the explicit causal inference task (unity judgment). Here, subjects indicated if the visual and vestibular cues indicated heading in the same direction ("common" cause, $C=1$ ) or in different directions ("different" causes, $C=2$ ). Each combination of disparity and reliability was presented at least 20 times. Since each disparity was randomly assigned to be positive or negative on each trial, $0^{+}$disparity was presented at least 40 times at each visual cue reliability resulting in a total of 700-1200 trials. Subjects did not receive feedback about the correctness of their responses.

Finally, the same subjects also participated in the implicit causal inference task—bisensory (inertial) discrimination. Here, subjects indicated the perceived direction of their inertial selfmotion (left or right of straight ahead). Note that although both visual and vestibular stimuli were presented in each trial, subjects were asked to only indicate their perceived direction of inertial heading, similar to the bisensory auditory localization procedure in [21]. Each combination of disparity and visual cue reliability was presented at least 70 times. Since each disparity was randomly assigned to be positive or negative on each trial, $0^{+}$disparity was presented at least 140 times resulting in a total of 2100-3000 trials divided across 7-9 sessions. No feedback was given about the correctness of subjects' responses.

For all tasks, sessions were about one hour long and subjects were required to take multiple breaks within each session.

Data analysis. For the explicit causal inference task, we computed the proportion of trials in which subjects perceived a common cause at each disparity and visual cue reliability. For the implicit causal inference task, we calculated the shift in perceived inertial heading as a function of $s_{\text {vis }}$, that is the influence that $s_{\text {vis }}$ had on $s_{\text {vest }}$, and we called this model-free

---

#### Page 24

summary statistic 'bias'. In order to build psychometric functions with enough trials, we binned values of $s_{\text {vis }}$ in the following intervals: $\left\{\left[-45^{\circ},-30^{\circ}\right],\left[-27.5^{\circ},-22.5^{\circ}\right],\left[-20^{\circ},-15^{\circ}\right]\right.$, $\left.\left[-12.5^{\circ},-7.5^{\circ}\right],\left[-5^{\circ},-2.5^{\circ}\right], 0^{\circ},\left[2.5^{\circ}, 5^{\circ}\right],\left[7.5^{\circ}, 12.5^{\circ}\right],\left[15^{\circ}, 20^{\circ}\right],\left[22.5^{\circ}, 27.5^{\circ}\right],\left[30^{\circ}, 45^{\circ}\right]\right\}$. Bin ranges were chosen to yield a comparable number of trials per bin, according to the nonuniform distribution of $s_{\text {vis }}$ in the experiment (see Fig 1B). For each visual bin and level of visual cue reliability, we constructed psychometric functions by fitting the proportion of rightward responses as a function of $s_{\text {vest }}$ with cumulative Gaussian functions (inset in Fig 3A). Thus, we defined the bias in the perceived inertial heading as minus the point of subjective equality (L/R PSE). A bias close to zero indicates that subjects accurately perceived their inertial (vestibular) heading. Large shifts of the PSE away from zero, that is substantial biases, suggest that misleading visual cues exerted a significant influence on the accuracy of inertial heading discrimination. Note that we do not expect the psychometric curves to be exact cumulative Gaussian functions, because of nonlinearities due to eccentricity-dependence of the noise and effects of causal inference. Nonetheless, the bias as we defined it is useful as a simple model-free statistic. Repeated-measures ANOVA with disparity or visual bin and visual cue reliability as within-subjects factors were performed separately on the proportion of common cause reports and bias in perceived inertial heading. We applied Greenhouse-Geisser correction of the degrees of freedom in order to account for deviations from sphericity [87], and report effect sizes as partial eta squared, denoted with $\eta_{e}^{2}$. For all analyses the criterion for statistical significance was $p<.05$, and we report uncorrected $p$ values. Unless specified otherwise, summary statistics are reported in the text as mean $\pm \mathrm{SE}$ between subjects. Finally, we remark that the summary statistics described above were used only for visualization and to perform simple descriptive statistics; we fit all models to raw trial data.

# Causal inference models

We build upon standard causal inference models of multisensory perception [18]. For concreteness, in the following description of causal inference models we refer to the visuo-vestibular example with binary responses ('left/right' for discrimination, and 'yes/no' for unity judgements). The basic component of any observer model is the trial response probability, that is the probability of observing a given response for a given trial condition (e.g., stimulus pair, uncertainty level, task). In the following we briefly review how these probabilities are computed.

All analysis code was written in MATLAB (Mathworks, Inc.), with core computations in C for increased performance (via mex files in MATLAB). Code is available at https://github.com/ lacerbi/visvest-causinf.

Unisensory heading discrimination. We used subjects' binary ('left or right of straight forward') heading choices, measured in the presence of visual-only and vestibular-only stimuli, to estimate subjects' measurement noise in the respective sensory signals. Let us consider a trial with a vestibular-only stimulus (the computation for a visual-only stimulus is analogous). Subjects are asked whether the perceived direction of motion $s_{\text {vest }}$ is to the left or to the right of straight forward $\left(0^{\circ}\right)$. We assume that the observer has access to a noisy measurement $x_{\text {vest }}$ of stimulus $s_{\text {vest }}$ (direction of motion), with probability density

$$
p\left(x_{\text {vest }} \mid s_{\text {vest }}\right)=\mathcal{N}\left(x_{\text {vest }} \mid s_{\text {vest }}, \sigma^{2}\left(s_{\text {vest }}\right)\right)
$$

where $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ is a normal probability density with mean $\mu$ and variance $\sigma^{2}$. Since stimulus directions are defined over the circle, we also considered a wrapped normal or, similarly, a von

---

#### Page 25

Mises distribution instead of Eq 1. Because of the relatively small range of stimuli used in the experiment, we found no difference between the distributions defined over the full circle and the simple normal distribution in Eq 1 (see S1 Appendix). Incidentally, in an additional investigation we also found no empirical difference between a wrapped normal and a von Mises, so either noise distribution could be used in the presence of fully circular stimuli (see S1 Appendix).

Depending on the sensory noise model, the variance in Eq 1 is either constant $\left(\sigma^{2}\left(s_{\text {vest }}\right) \equiv \sigma_{0 \text { vest }}^{2}\right)$ or eccentricity-dependent with base magnitude $\sigma_{0 \text { vest }}^{2}$ and noise that increases with eccentricity (distance from $0^{\prime}$ ) approximately quadratically, at least for small headings, according to a parameter $w_{\text {vest }} \geq 0$ (see S1 Appendix for details). For $w_{\text {vest }}=0$, the eccentricitydependent model reduces to the constant model. The observer's posterior probability density over the vestibular stimulus is $p\left(s_{\text {vest }} \mid x_{\text {vest }}\right) \propto p\left(x_{\text {vest }} \mid s_{\text {vest }}\right) p_{\text {prior }}\left(s_{\text {vest }}\right)$, and we will see that under some assumptions the prior over heading directions is irrelevant for subsequent computations in the left/right unisensory task (see S1 Appendix).

We assume that observers compute the posterior probability that the stimulus is right of straight forward as $\operatorname{Pr}\left(s_{\text {vest }}>0 \mid x_{\text {vest }}\right)=\int_{0}^{|b|} p\left(s_{\text {vest }} \mid x_{\text {vest }}\right) d s_{\text {vest }}$, and respond 'right' if $\operatorname{Pr}\left(s_{\text {vest }}>\right.$ $0 \mid x_{\text {vest }}$ ) $>0.5$; 'left' otherwise (see S1 Appendix for details). Observers may also lapse and give a completely random response with probability $\lambda$ (lapse rate). This yields

$$
\operatorname{Pr}\left(\text { choose right } \mid x_{\text {vest }}\right)=\frac{\lambda}{2}+(1-\lambda) \llbracket \operatorname{Pr}\left(s_{\text {vest }}>0 \mid x_{\text {vest }}\right)>0.5 \rrbracket
$$

where $\llbracket \cdot \rrbracket$ is Iverson bracket, which is 1 if the argument is true, and 0 otherwise [88].
An analogous derivation is applied to each unisensory visual stimulus condition for respectively low, medium, and high visual reliability. We assume a distinct $\sigma_{0 \text { vis }}$ for each visual reliability condition, and, for the eccentricity-dependent models, a common $w_{\text {vis }}$ for all visual reliability conditions, so as to reduce model complexity.

Unity judgment (explicit causal inference). In a unity judgment trial, the observer explicitly evaluates whether there is a single cause $(C=1)$ underlying the noisy measurements $x_{\text {vis }}, x_{\text {vest }}$, or two separate causes ( $C=2$; see Fig 2B). All following probability densities are conditioned on $c_{\text {vis }}$, the level of visual cue reliability in the trial, which is assumed to be known to the observer; we omit this dependence to reduce clutter. We consider three families of explicit causal inference strategies.

The Bayesian causal inference strategy computes the posterior probability of common cause

$$
\operatorname{Pr}\left(C=1 \mid x_{\text {vis }}, x_{\text {vest }}\right)=\frac{p\left(x_{\text {vis }}, x_{\text {vest }} \mid C=1\right) p_{c}}{p\left(x_{\text {vis }}, x_{\text {vest }} \mid C=1\right) p_{c}+p\left(x_{\text {vis }}, x_{\text {vest }} \mid C=2\right)\left(1-p_{c}\right)}
$$

where $0 \leq p_{c} \equiv \operatorname{Pr}(C=1) \leq 1$, the prior probability of a common cause, is a free parameter of the model. The derivation of $p\left(x_{\text {vis }}, x_{\text {vest }}\right \mid C=k$ ), for $k=1,2$, is available in S1 Appendix. The observer reports unity if the posterior probability of common cause is greater than 0.5 , with the added possibility of random lapse,

$$
\operatorname{Pr}\left(\text { choose unity } \mid x_{\text {vis }}, x_{\text {vest }}\right)=\frac{\lambda}{2}+(1-\lambda) \llbracket \operatorname{Pr}\left(C=1 \mid x_{\text {vis }}, x_{\text {vest }}\right)>0.5 \rrbracket
$$

For a separate analysis we also considered a 'probability matching' variant that reports unity with probability equal to $\operatorname{Pr}\left(C=1 \mid x_{\text {vis }}, x_{\text {vest }}\right)$ (plus lapses).

---

#### Page 26

As a non-Bayesian causal inference heuristic model, we consider a fixed criterion observer, who reports a common cause whenever the two noisy measurements are within a distance $\kappa_{c} \geq 0$ from each other,

$$
\operatorname{Pr}\left(\text { choose unity } \mid x_{\text {vis }}, x_{\text {vest }}\right)=\frac{\lambda}{2}+(1-\lambda)\left[\left\|x_{\text {vis }}-x_{\text {vest }}\right\|<\kappa_{c}\right]
$$

Crucially, the fixed criterion observer does not take into account stimulus reliability or other statistical information when inferring the causal structure.

Finally, we consider a fusion observer that eschews causal inference altogether. A classical 'forced fusion' observer would always report 'unity' in the explicit causal inference task, which is easily rejected by the data. Instead, we consider a stochastic fusion observer that reports 'unity' with probability $\eta_{\text {low }}, \eta_{\text {med }}$, or $\eta_{\text {high }}$, depending only on the reliability of the visual cue, and discards any other information.

Bisensory inertial discrimination (implicit causal inference). In bisensory inertial discrimination trials, the observer reports whether the perceived inertial heading $s_{\text {vest }}$ is to the left or right of straight forward $\left(0^{+}\right)$. In this experiment, we do not ask subjects to report $s_{\text {vis }}$, but the inference would be analogous. The inertial discrimination task requires an implicit evaluation of whether there is a single cause to the noisy measurements $x_{\text {vis }}, x_{\text {vest }}(C=1)$, or two separate causes $(C=2)$, for a known level of visual coherence $c_{\text {vis }}$ (omitted from the notation for clarity).

If the observer knew that $C=k$, for $k=1,2$, the posterior probability density over the vestibular stimulus would be (see S1 Appendix)

$$
p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, C=k\right) \propto \int_{-90^{\circ}}^{90^{\circ}} p\left(x_{\text {vest }} \mid s_{\text {vest }}\right) p\left(x_{\text {vis }} \mid s_{\text {vis }}, c_{\text {vis }}\right) p\left(s_{\text {vis }}, s_{\text {vest }} \mid C=k\right) d s_{\text {vis }}
$$

where the likelihoods are defined as per the uni-sensory task, Eq 1, and for the prior over heading directions, $p\left(s_{\text {vis }}, s_{\text {vest }} \mid C\right)$, see 'Observers' priors' below.

The posterior probability of rightward motion is computed for $k=1,2$ as

$$
\operatorname{Pr}\left(s_{\text {vest }}>0 \mid x_{\text {vest }}, x_{\text {vis }}, C=k\right) \propto \int_{0^{\circ}}^{90^{\circ}} p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, C=k\right) d s_{\text {vest }}
$$

and an analogous equation holds for the posterior probability of leftward motion.
In general, the causal structure is implicitly inferred by the observer. We assume that observers combine cues according to

$$
\begin{aligned}
p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}\right)= & v_{1}\left(x_{\text {vis }}, x_{\text {vest }}\right) \cdot p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, C=1\right)+ \\
& {\left[1-v_{1}\left(x_{\text {vis }}, x_{\text {vest }}\right)\right] \cdot p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, C=2\right) }
\end{aligned}
$$

where $0 \leq v_{1}\left(x_{\text {vis }}, x_{\text {vest }}\right) \leq 1$ is the implicit causal weight associated by the observer to the hypothesis of a single cause, $C=1$. The form of the causal weight depends on the observer's implicit causal inference strategy.

We consider three families of implicit causal inference. For the Bayesian causal inference observer, the causal weight is equal to the posterior probability, $v_{1}\left(x_{\text {vis }}, x_{\text {vest }}\right)=\operatorname{Pr}\left(C=1 \mid x_{\text {vis }}\right.$, $x_{\text {vest }}$ ), so that Eq 6 becomes the expression for Bayesian model averaging [18] (see Eq 3 and S1 Appendix). As a variant of the Bayesian observer we consider a probability matching Bayesian strategy for which $v_{1}=1$ with probability $\operatorname{Pr}\left(C=1 \mid x_{\text {vis }}, x_{\text {vest }}\right)$, and $v_{1}=0$ otherwise. For the fixed-criterion observer, $v_{1}=\left\|\left\|x_{\text {vis }}-x_{\text {vest }}\right\|<\kappa_{c}\right\|$, with $\kappa_{c} \geq 0$ as per Eq 5. Finally, for the forced fusion observer $v_{1} \equiv 1$.

---

#### Page 27

The posterior probability of rightward motion is then
$\operatorname{Pr}\left(s_{\text {vest }}>0 \mid x_{\text {vest }}, x_{\text {vis }}\right)=\int_{0^{\circ}}^{90^{\circ}} p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}\right) d s_{\text {vest }}$, and an analogous equation holds for the posterior probability of leftward motion. We assume the observer reports the direction with highest posterior probability, with occasional lapses (see also Eq 2),

$$
\operatorname{Pr}\left(\text { choose right } \mid x_{\text {vis }}, x_{\text {vest }}\right)=\frac{\lambda}{2}+(1-\lambda)\left\|\operatorname{Pr}\left(s_{\text {vest }}>0 \mid x_{\text {vis }}, x_{\text {vest }}\right)>0.5\right\|
$$

where $\lambda \geq 0$ is the lapse rate.
Observers' prior. We assume subjects develop a symmetric, unimodal prior over heading directions for unisensory trials. Due to the form of the decision rule (Eq 2), a symmetric prior has no effect on the unisensory trials, so we only focus on the bisensory case.

For the bisensory prior over heading directions, $p\left(s_{\text {vis }}, s_{\text {vest }}\right \mid C$ ) we consider two families of priors. The empirical prior approximately follows the correlated structure of the discrete distribution of vestibular and visual headings presented in the experiment (Fig 1B). The independent prior assumes that observers learn a generic uncorrelated Gaussian prior over heading directions, as per [18]. See S1 Appendix for details.

We note that previous work in heading perception has found a 'repulsive' bias away from straight ahead [89, 90], which is seemingly at odds with the central prior assumed here. However, the repulsion bias previously reported can be explained by the current Bayesian framework by means of a stimulus-dependent likelihood [91, 92]. According to the Bayesian theory, such a stimulus-dependent likelihood may induce a bias away from regions of higher sensory precision. Whether the net bias is going to be attractive or repulsive depends on the relative contribution of prior and likelihood [93]. Thus, our models that combine a central prior and stimulus-dependent likelihood are not incompatible with previous findings of repulsive biases. See also S1 Appendix.

Trial response probabilities. Eqs 2, 4, 5 and 7 represent the probability that an observer chooses a specific response $r$ ('rightward' or 'leftward' for discrimination trials, 'same' or 'different' for unity judgment trials), for given noisy measurements $x_{\text {vis }}$ and $x_{\text {vest }}$ (or only one of the two for the unisensory task), and known visual reliability $c_{\text {vis }}$. Since as experimenters we do not have access to subjects' internal measurements, to compute the trial response probabilities we integrate ('marginalize') over the unseen noisy measurements for given heading directions $s_{\text {vis }}$ and $s_{\text {vest }}$ presented in the trial.

For the unisensory case, considering as example the vestibular case, we get

$$
\operatorname{Pr}\left(\text { observed } r \mid s_{\text {vest }}\right)=\int_{-90^{\circ}}^{90^{\circ}} \operatorname{Pr}\left(\text { choose } r \mid x_{\text {vest }} \mid p\left(x_{\text {vest }} \mid s_{\text {vest }}\right) d x_{\text {vest }}\right.
$$

For the bisensory case, either unity judgment or inertial discrimination, we have

$$
\begin{aligned}
\operatorname{Pr}(\text { observed } r \mid s_{\text {vis }}, s_{\text {vest }}, c_{\text {vis }})= & \int_{-90^{\circ}}^{90^{\circ}} \int_{-90^{\circ}}^{90^{\circ}} \operatorname{Pr}\left(\text { choose } r \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right) \\
& \times p\left(x_{\text {vest }} \mid s_{\text {vest }}\right) p\left(x_{\text {vis }} \mid s_{\text {vis }}, c_{\text {vis }}\right) d x_{\text {vest }} d x_{\text {vis }}
\end{aligned}
$$

It is customary in the causal inference literature to approximate these integrals via Monte Carlo sampling, by drawing a large number of noisy measurements from the noise distributions (e.g., $[18,20,24,33]$ ). Instead, we computed the integrals via numerical integration, which is more efficient than Monte Carlo techniques for low dimensional problems [94]. We used the same numerical approach to evaluate Eqs 2, 4, 5 and 7, including an adaptive method

---

#### Page 28

for choice of integration grid. All numerical integrals were then coded in C (mex files in MATLAB) for additional speed. See S1 Appendix for computational details.

# Model fitting

For a given model, we denote its set of parameters by a vector $\boldsymbol{\theta}$. For a given model and dataset, we define the parameter log likelihood function as

$$
\begin{aligned}
\mathrm{LL}(\boldsymbol{\theta}, \text { model }) & =\log p(\text { data } \mid \boldsymbol{\theta}, \text { model }) \\
& =\log \prod_{i=1}^{N_{\text {read }}} p\left(r^{(i)} \mid \hat{x}_{\mathrm{cu}}^{(i)}, \hat{x}_{\mathrm{ver}}^{(i)}, \hat{c}_{\mathrm{cu}}^{(i)}, \boldsymbol{\theta}, \text { model }\right) \\
& =\sum_{i=1}^{N_{\text {read }}} \log p\left(r^{(i)} \mid \hat{x}_{\mathrm{cu}}^{(i)}, \hat{x}_{\mathrm{ver}}^{(i)}, \hat{c}_{\mathrm{cu}}^{(i)}, \boldsymbol{\theta}, \text { model }\right)
\end{aligned}
$$

where we assumed conditional independence between trials; $r^{(i)}$ denotes the subject's response ('right' or 'left' for the discrimination trials; 'common' or 'separate' causes in unity judgment trials); $\hat{x}_{\mathrm{cu}}^{(i)}$ and $\hat{x}_{\mathrm{ver}}^{(i)}$ are, respectively, the direction of motion of the visual (resp. vestibular) stimulus (if present), and $c_{\mathrm{cu}}^{(i)}$ is the visual coherence level (that is, reliability: low, medium, or high), in the $i$-th trial.

Maximum likelihood estimation. First, we fitted our models to the data via maximum likelihood estimation, by finding the parameter vector $\boldsymbol{\theta}^{*}$ that maximizes the log likelihood in Eq 10. For optimization of the log likelihood, we used Bayesian Adaptive Direct Search (BADS; https://github.com/lacerbi/bads). BADS is a black-box optimization algorithm that combines a mesh-adaptive direct search strategy [95] with a local Bayesian optimization search step based on Gaussian process surrogates (see [80, 96] for an introduction to Bayesian optimization). Bayesian optimization is particularly useful when the target function is costly to evaluate or the likelihood landscape is rough, as it is less likely to get stuck in local optima than other algorithms, and may reduce the number of function evaluations to find the (possibly global) optimum. In our case, evaluation of the log likelihood function for a single parameter vector $\boldsymbol{\theta}$ could take up to $\sim 2-3 \mathrm{~s}$ for bisensory datasets, which makes it a good target for Bayesian optimization. We demonstrated in a separate benchmark that BADS is more effective than a large number of other MATLAB optimizers for our problem ('causal inference' problem set in [78]). See S1 Appendix for more details about the algorithm and the optimization procedure.

For each subject we first fitted separately the datasets corresponding to three tasks (unisensory and bisensory heading discrimination, unity judgment), and then performed joint fits by combining datasets from all tasks (summing the respective log likelihoods).

Posterior sampling. As a complementary approach to ML parameter estimation, for each dataset and model we calculated the posterior distribution of the parameters,

$$
p(\boldsymbol{\theta} \mid \text { data }, \text { model }) \propto p(\text { data } \mid \boldsymbol{\theta}, \text { model }) p(\boldsymbol{\theta} \mid \text { model })
$$

where $p$ (data $\mid \boldsymbol{\theta}$, model) is the likelihood (see Eq 10) and $p(\boldsymbol{\theta} \mid$ model) is the prior over parameters. We assumed a factorized prior $p(\boldsymbol{\theta} \mid$ model $)=\prod_{i=1}^{k} p\left(\theta_{i}\right)$ and a non-informative uniform prior over a bounded interval for each model parameter (uniform in log space for scale parameters such as all noise base magnitudes, fixed criterion $\kappa_{c}$, and prior parameters $\sigma_{\text {prior }}$ and $\Delta_{\text {prior }}$ ); see Table 2.

We approximated Eq 11 via Markov Chain Monte Carlo (MCMC) sampling. We used a custom-written sampling algorithm that combines slice sampling [81] with adaptive direction

---

#### Page 29

sampling [82] and a number of other tricks (https://github.com/lacerbi/eissample). Slice sampling is a flexible MCMC method that, in contrast with the common Metropolis-Hastings transition operator, requires very little tuning in the choice of length scale. Adaptive direction sampling is an ensemble MCMC method that shares information between several dependent chains (also called 'walkers' [97]) in order to speed up mixing and exploration of the state space. For details about the MCMC algorithm and the sampling procedure, see S1 Appendix.

# Factorial model comparison

We built different observer models by factorially combining three factors: causal inference strategy (Bayesian, fixed-criterion, or fusion); shape of sensory noise (constant or eccentricitydependent); and type of prior over heading directions (empirical or independent); see Fig 2A and 'Causal inference models' section of the Methods for a description of the different factors.

For each subject, we fitted the different observer models, first separately to different tasks (unity judgment and bisensory inertial discrimination), and then performed a joint fit by combining datasets from all tasks (including the unisensory discrimination task). We evaluated the fits with a number of model comparison metrics and via an objective goodness of fit metric. Finally, we combined evidence for different model factors across subjects with a hierarchical Bayesian approach.

We verified our ability to distinguish different models with a model recovery analysis, described in S1 Appendix.

Model comparison metrics. For each dataset and model we computed a number of different model comparison metrics, all of which take into account quality of fit and penalize model flexibility, but with different underlying assumptions.

Based on the maximum likelihood solution, we computed Akaike information criterion with a correction for sample size (AICc) and Schwarz's 'Bayesian' Information criterion (BIC),

$$
\begin{aligned}
\mathrm{AICc} & =-2 L L\left(\boldsymbol{\theta}^{\prime}\right)+2 k+\frac{2 k(k+1)}{N_{\text {trials }}-k-1} \\
\mathrm{BIC} & =-2 L L\left(\boldsymbol{\theta}^{\prime}\right)+k \log N_{\text {trials }}
\end{aligned}
$$

where $N_{\text {trials }}$ is the number of trials in the dataset and $k$ is the number of parameters of the model. The factor of -2 that appears in both definitions is due to historical reasons, so that both metrics have the same scale of the deviance.

To assess model performance on unseen data, we performed Bayesian leave-one-out (LOO) cross-validation. Bayesian LOO cross-validation computes the posterior of the parameters given $N_{\text {trials }} \sim 1$ trials (training), and evaluates the (log) expected likelihood of the left-out trial (test); the procedure is repeated for each trial, yielding the leave-one-out score

$$
\mathrm{LOO}=\sum_{i=1}^{N_{\text {trials }}} \log \int p\left(r_{i} \mid \boldsymbol{\theta}\right) p\left(\boldsymbol{\theta} \mid \mathcal{D}_{-i}\right) d \boldsymbol{\theta}
$$

where $p\left(r_{i} \mid \boldsymbol{\theta}\right)$ is the likelihood associated to the $i$-th trial alone, and $p\left(\boldsymbol{\theta} \mid \mathcal{D}_{-i}\right)$ is the posterior over $\boldsymbol{\theta}$ given all trials except the $i$-th one. Eq 13 can be estimated at prohibitive computational cost by separately sampling from the leave-one-out posteriors via $N_{\text {trials }}$ distinct MCMC runs. A more feasible approach comes from noting that all posteriors differ from the full posterior by only one data point. Therefore, the leave-one-out posteriors can be approximated via importance sampling, reweighting the full posterior obtained via MCMC. However, a direct approach of importance sampling can be unstable, since the full posterior is typically narrower than the leave-one-out posteriors. Pareto-smoothed importance sampling (PSIS) is a recent

---

#### Page 30

technique to stabilize the importance weights [52], implemented in the psisloo package (https://github.com/avehtari/PSIS). Thus, Eq 13 is approximated as

$$
\mathrm{LOO} \approx \sum_{i=1}^{N_{s}} \log \frac{\sum_{s=1}^{k} w_{i}^{(s)} p\left(r_{i} \mid \boldsymbol{\theta}^{(s)}\right)}{\sum_{s=1}^{k} w_{s}^{(s)}}
$$

where $\boldsymbol{\theta}^{(s)}$ is the $s$-th parameter sample from the posterior, and $w_{i}^{(s)}$ are the Pareto-smoothed importance weights associated to the $i$-th trial and $s$-th sample (out of $S$ ); see [53] for details. PSIS also returns for each trial the exponent $k_{i}$ of the fitted Pareto distribution; if $k_{i}$ is greater than 1 the moments of the importance ratios distribution do not exist and the variance of the PSIS estimate is finite but may be large; this provides a natural diagnostic for the method [53] (see S1 Appendix). LOO is our comparison metric of choice (see Discussion). LOO scores for all models and subjects are reported in S1 Appendix.

Finally, we approximated the marginal likelihood of the model,

$$
p(\text { data } \mid \text { model })=\int p(\text { data } \mid \boldsymbol{\theta}, \text { model }) p(\boldsymbol{\theta} \mid \text { model }) d \boldsymbol{\theta}
$$

The marginal likelihood is a common metric of model evidence that naturally incorporates a penalty for model complexity due to Bayesian Occam razor [71]. However, the integral in Eq 15 is notoriously hard to evaluate. Here we computed an approximation of the log marginal likelihood (LML) based on MCMC samples from the posterior, by using a weighted harmonic mean estimator [74]. The formula for the approximation is

$$
\mathrm{LML}=-\log \left(\frac{1}{S} \sum_{s=1}^{k} \frac{\varphi\left(\boldsymbol{\theta}^{(s)}\right)}{p\left(\boldsymbol{\theta}^{(s)}\right) L\left(\boldsymbol{\theta}^{(s)}\right)}\right)
$$

where the sum is over $S$ samples from the posterior, $\boldsymbol{\theta}^{(s)}$ is the $s$-th sample, $p(\boldsymbol{\theta})$ the prior, $L(\boldsymbol{\theta})$ the likelihood, and $\varphi(\boldsymbol{\theta})$ is an arbitrary weight probability density. The behavior of the approximation depends crucially on the choice of $\varphi$; it is important that $\varphi$ has thinner tails than the posterior, lest the variance of the estimator grows unboundedly. We followed the suggestion of [74] and adopted a finite support distribution over a high posterior density region. We fitted a variational Gaussian mixture model to the posterior samples [98] (https://github.com/lacerbi/ vbgmm ), and then we replaced each Gaussian component with a uniform distribution over an ellipsoid region proportional to the covariance matrix of the component. The proportionality constant, common to all components, was picked by minimizing the empirical variance of the sum in Eq 16 [75].

Hierarchical Bayesian model selection. We performed Bayesian model selection at the group level via a hierarchical approach that treats subjects and models as random variables [54]. Group Bayesian Model Selection infers the posterior over model frequencies in the population, expressed as Dirichlet distributions parametrized by the concentration parameter vector $\boldsymbol{\alpha}$. As a summary statistic we consider the protected exceedance probability $\tilde{\varphi}$, that is the probabilty that a given model or model factor is the most likely model or model factor, above and beyond chance [55]. For the $i$-th model or model factor,

$$
\tilde{\varphi}_{i}=(1-\mathrm{BOR}) \varphi_{i}+\frac{1}{K} \mathrm{BOR}
$$

where $K$ is the number of models (or model factors), $\varphi_{i}$ is the unprotected exceedance probability for the $i$-th model or model factor [54], and BOR is the Bayesian omnibus risk-the posterior probability that the data may be explained by the null hypothesis according to which all

---

#### Page 31

models (or model factors) have equal probability [55]. For completeness, we report posterior model frequencies and BOR in the figures, but we do not focus on model frequencies per se since our sample size does not afford a more detailed population analysis.

To compute the posterior over model factors in the population we exploit the agglomerative property of the Dirichlet distribution, and sum the concentration parameters of models that belong to the same factor component [54]. While the agglomerative property allows to easily compute the posterior frequencies and the unprotected exceedance probabilities for each model factor, calculation of the protected exceedance probabilities required us to compute the BOR for the model factor setup (the probability that the observed differences in factor frequencies may have arisen due to chance).

Additionally, the group Bayesian Model Selection method requires to specify a Dirichlet prior over model frequencies, represented by a concentration parameter vector $\alpha_{0} \cdot \boldsymbol{w}$, with $w_{k}=1$ for any model $k$ and $\alpha_{0}>0$. The common choice is $\alpha_{0}=1$ (flat prior over model frequencies), but given the nature of our factorial analysis we prefer a flat prior over model factors ( $\alpha_{0}=$ average number factors / number of models), where the average number of factors is $\approx 2.33$ for the bisensory tasks and $\approx 2.67$ for the joint fits. This choice entails that the concentration parameter of the agglomerate Dirichlet distributions, obtained by grouping models that belong to the same factor component, is of order $\sim 1$ (it cannot be exactly one since different factors have different number of components). When factor components within the same factor had unequal numbers of models, we modified the prior weight vector $\boldsymbol{w}$ such that every component had equal prior weight. We verified that our main results did not depend on the specific choice of Dirichlet prior (Fig 7, third row).

Parameter compatibility metric. Before performing the joint fits, we tested whether model parameters differed across the three tasks (unisensory and bisensory discrimination, unity judgment). On one end of the spectrum, the fully Bayesian approach would consist of comparing all combinations of models in which parameters are shared vs. distinct across tasks, and check which combination best explains the data. However, this approach is intractable in practice due to the combinatorial explosion of models, and undesirable in theory due to the risk model overfitting. On the simplest end of the spectrum, we could look at the credible intervals of the parameter posteriors for each subject and visually check whether they are mostly overlapping for different tasks.

As a middle ground, we computed separately for each parameter what we defined as the compatibility probability $C_{p}$, that is the probability that for most subjects the parameter is exactly the same across tasks $\left(H_{0}\right)$, as opposed to being different $\left(H_{1}\right)$, above and beyond chance.

For a given subject, let $y_{1}, y_{2}$, and $y_{3}$ be the datasets of the three tasks. For a given parameter $\theta$ (e.g., lapse rate), we computed the compatibility likelihoods

$$
\begin{aligned}
& p\left(y_{1}, y_{2}, y_{3} \mid H_{0}\right)=\int\left[\prod_{i=1}^{n} g_{i}\left(\theta \mid y_{i}\right)\right] f(\theta) d \theta \\
& p\left(y_{1}, y_{2}, y_{3} \mid H_{1}\right)=\prod_{i=1}^{n}\left[\int g_{i}\left(\theta \mid y_{i}\right) f(\theta) d \theta\right]
\end{aligned}
$$

where $g_{i}\left(\theta \mid y_{i}\right)$ is the marginal posterior over $\theta$ for the dataset $y_{i}$, and $f(\theta)$ is the prior over $\theta$. Having computed the compatibility likelihoods for all subjects, we defined $C_{p}$ as the protected exceedance probability of model $H_{0}$ vs. model $H_{1}$ for the entire group.

For each subject and task, the marginal posteriors $g_{i}\left(\theta \mid y_{i}\right)$ were obtained as a weighted average over models, with weight equal to each model's posterior probability for that subject

---

#### Page 32

according to the group Bayesian Model Selection method via LOO, and considering only the subset of models that include the parameter of interest (see Fig 5).

For the prior $f(\theta)$ over a given parameter $\theta$, for the purposes of this analysis only, we followed an empirical Bayes approach informed by the data and use a truncated Cauchy prior fitted to the average marginal posterior of $\theta$ across subjects, defined over the range of the MCMC samples for $\theta$.

Absolute goodness of fit. Model comparison yields only a relative measure of goodness of fit, but does not convey any information of whether a model is a good description of the data in an absolute sense. A standard metric such as the coefficient of variation $R^{2}$ is not appropriate for binary data. Instead, we extended the approach of [56] and defined absolute goodness of fit as

$$
g(\text { model }) \equiv 1-\frac{\hat{H}_{G}(\text { data })+\operatorname{LOO}(\text { model })}{\hat{H}_{G}(\text { data })-\hat{N}_{\text {trials }} \log 2}
$$

where $\hat{H}_{G}($ data $)$ is an estimate of the entropy of the data obtained via Grassberger's estimator [99] and LOO(model) is the LOO score of the model of interest.

The numerator in Eq 18 represents the Kullback-Leibler (KL) divergence between the distribution of the data and the distribution predicted by the model (that is, how well the model captures the data), which is compared as a reference to the KL divergence between the data and a chance model (at the denominator). See S1 Appendix for a derivation of Eq 18, and code is available at https://github.com/lacerbi/gofit.

# The cookbook

The Bayesian cookbook for causal inference in multisensory perception, or simply 'the cookbook', consists of a recipe to build causal inference observer models for multisensory perception, and a number of algorithms and computational techniques to perform efficient and robust Bayesian comparison of such models. We applied and demonstrated these methods at different points in the main text; further details can be found here in the Methods and S1 Appendix. For reference, we summarize the main techniques of interest in Table 3.

Table 3. List of algorithms and computational procedures.

| Description                                  | Code                                       |       References       |
| :------------------------------------------- | :----------------------------------------- | :--------------------: |
| Model fitting                                |                                            |                        |
| Efficient computation of log likelihood      | https://github.com/lacerbi/visvest-causinf |       This work        |
| Maximum-likelihood estimation (optimization) | https://github.com/lacerbi/bads            |         $[78]$         |
| Posterior estimation (MCMC sampling)         | https://github.com/lacerbi/eissample       |     In preparation     |
| Model evaluation and comparison              |                                            |       $[52,53]$        |
| Leave-one-out cross validation (LOO)         | https://github.com/avehtari/PSIS           | $[74]$, in preparation |
| Estimate of the marginal likelihood          | https://github.com/lacerbi/marglike        |   $[54]$, this work    |
| Parameter compatibility test                 | https://github.com/lacerbi/comprob         |   $[56]$, this work    |
| Objective goodness of fit                    | https://github.com/lacerbi/gofit           |       $[54,55]$        |
| Group Bayesian Model Selection               |                                            |                        |

List of useful algorithms and computational procedures.
https://doi.org/10.1371/journal.pcbi.1006110.t003

---

# Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception - Backmatter

---

#### Page 33

## Colophon

Citation: Acerbi L, Dokka K, Angelaki DE, Ma WJ (2018) Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS Comput Biol 14(7): e1006110. https://doi.org/10.1371/journal.pcbi.1006110

Editor: Samuel J. Gershman, Harvard University, UNITED STATES

Received: July 26, 2017
Accepted: March 28, 2018
Published: July 27, 2018
Copyright: © 2018 Acerbi et al. This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

Data Availability Statement: All relevant data and code are available from the following GitHub repository: https://github.com/lacerbi/visvestcausinf.

Funding: This work was supported by award number R01EY020958 from the National Eye Institute, and award number W911NF-12-1-0262 from the Army Research Office to WJM. KD was supported by National Institute of Deafness and Communications Disorders Grant R03 DC013987. DEA was supported by National Institute of Health Grant R01 DC007620. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

Competing interests: The authors have declared that no competing interests exist.

# Supporting information

S1 Fig. Explicit causal inference; model fits of full data. Results of the explicit causal inference (unity judgment) task, for two models of interest. Proportion of 'unity' responses for a given $\left(s_{\text {vis }}, s_{\text {vest }}\right)$ heading direction pair (indexed from 1 to 99 ), and for different levels of visual cue reliability. Points are data, lines are model fits (average fit across subjects). Error bars are omitted for clarity. A: Best Bayesian model (Bay-X-E). B: Best fixed-criterion model (Fix-C). Neither model appears clearly superior across all noise levels (see main text).
(TIF)
S1 Appendix. Supplemental methods. Cookbook for causal inference observers. Observer model factors. Comparison between wrapped normal and von Mises noise. Computational details. Absolute goodness of fit. LOO scores for all subjects and models.
(PDF)

## Acknowledgments

We thank Bas van Opheusden and Shan Shen for useful discussions about absolute goodness of fit. This work has utilized the NYU IT High Performance Computing resources and services.

## Author Contributions

Conceptualization: Luigi Acerbi, Kalpana Dokka, Dora E. Angelaki, Wei Ji Ma.
Data curation: Luigi Acerbi, Kalpana Dokka.
Formal analysis: Luigi Acerbi.
Funding acquisition: Kalpana Dokka, Dora E. Angelaki, Wei Ji Ma.
Investigation: Kalpana Dokka.
Methodology: Luigi Acerbi, Kalpana Dokka.
Software: Luigi Acerbi.
Supervision: Dora E. Angelaki, Wei Ji Ma.
Validation: Luigi Acerbi, Kalpana Dokka.
Visualization: Luigi Acerbi.
Writing - original draft: Luigi Acerbi, Kalpana Dokka.
Writing - review \& editing: Luigi Acerbi, Kalpana Dokka, Dora E. Angelaki, Wei Ji Ma.

## References

1. Hillis JM, Ernst MO, Banks MS, Landy MS. Combining sensory information: Mandatory fusion within, but not between, senses. Science. 2002; 298(5598):1627-1630. https://doi.org/10.1126/science. 1075396 PMID: 12446912
2. Nardini M, Bedford R, Mareschal D. Fusion of visual cues is not mandatory in children. Proceedings of the National Academy of Sciences. 2010; 107(39):17041-17046. https://doi.org/10.1073/pnas. 1001699107
3. Jacobs RA. Optimal integration of texture and motion cues to depth. Vision Research. 1999; 39 (21):3621-3629. https://doi.org/10.1016/S0042-6989(99)00088-7 PMID: 10746132
4. Ernst MO, Banks MS. Humans integrate visual and haptic information in a statistically optimal fashion. Nature. 2002; 415(6870):429-433. https://doi.org/10.1038/415429a PMID: 11807554

---

#### Page 34

5. Knill DC, Saunders JA. Do humans optimally integrate stereo and texture information for judgments of surface slant? Vision Research. 2003; 43(24):2539-2558. https://doi.org/10.1016/S0042-6989(03) 00458-9 PMID: 13129541
6. Alais D, Burr D. The ventriloquist effect results from near-optimal bimodal integration. Current Biology. 2004; 14(3):257-262. https://doi.org/10.1016/j.cub.2004.01.029 PMID: 14761661
7. Hillis JM, Watt SJ, Landy MS, Banks MS. Slant from texture and disparity cues: Optimal cue combination. Journal of Vision. 2004; 4(12):967-992. https://doi.org/10.1167/4.12.1 PMID: 15669906
8. Helbig HB, Ernst MO. Optimal integration of shape information from vision and touch. Experimental Brain Research. 2007; 179(4):595-606. https://doi.org/10.1007/s00221-006-0814-y PMID: 17225091
9. Fetsch CR, Turner AH, DeAngelis GC, Angelaki DE. Dynamic reweighting of visual and vestibular cues during self-motion perception. The Journal of Neuroscience. 2009; 29(49):15601-15612. https://doi. org/10.1523/JNEUROSCI.2574-09.2009 PMID: 20007484
10. Butler JS, Smith ST, Campos JL, Bülthoff HH. Bayesian integration of visual and vestibular signals for heading. Journal of Vision. 2010; 10(11):1-23. https://doi.org/10.1167/10.11.23
11. de Winkel KN, Weesie J, Werkhoven PJ, Groen EL. Integration of visual and inertial cues in perceived heading of self-motion. Journal of Vision. 2010; 10(12):1-10. https://doi.org/10.1167/10.12.1 PMID: 21047733
12. Butler JS, Campos JL, Bülthoff HH, Smith ST. The role of stereo vision in visual-vestibular integration. Seeing and perceiving. 2011; 24(5):453-470. https://doi.org/10.1163/187847511X588070 PMID: 21888763
13. Dokka K, MacNeilage PR, DeAngelis GC, Angelaki DE. Multisensory self-motion compensation during object trajectory judgments. Cerebral Cortex. 2015; 25(3):619-630. https://doi.org/10.1093/cercor/ bht247 PMID: 24062317
14. Dokka K, DeAngelis GC, Angelaki DE. Multisensory Integration of Visual and Vestibular Signals Improves Heading Discrimination in the Presence of a Moving Object. The Journal of Neuroscience. 2015; 35(40):13599-13607. https://doi.org/10.1523/JNEUROSCI.2267-15.2015 PMID: 26446214
15. Brandwein AB, Foxe JJ, Butler JS, Russo NN, Altschuler TS, Gomes H, et al. The development of multisensory integration in high-functioning autism: High-density electrical mapping and psychophysical measures reveal impairments in the processing of audiovisual inputs. Cerebral Cortex. 2012; 23 (6):1329-1341. https://doi.org/10.1093/cercor/bhs109 PMID: 22628458
16. Stewart CR, Sanchez SS, Grenesko EL, Brown CM, Chen CP, Keehn B, et al. Sensory symptoms and processing of nonverbal auditory and visual stimuli in children with autism spectrum disorder. Journal of autism and developmental disorders. 2016; 46(5):1590-1601. https://doi.org/10.1007/s10803-015-2367-z PMID: 25652601
17. Calvert GA, Brammer MJ, Iversen SD. Crossmodal identification. Trends in cognitive sciences. 1998; 2(7):247-253. https://doi.org/10.1016/S1364-6613(98)01189-9 PMID: 21244923
18. Körding KP, Beierholm U, Ma WJ, Quartz S, Tenenbaum JB, Shams L. Causal inference in multisensory perception. PLoS ONE. 2007; 2(9):e943. https://doi.org/10.1371/journal.pone.0000943 PMID: 17895984
19. Sato Y, Toyoizumi T, Aihara K. Bayesian inference explains perception of unity and ventriloquism aftereffect: Identification of common sources of audiovisual stimuli. Neural Computation. 2007; 19 (12):3335-3355. https://doi.org/10.1162/neco.2007.19.12.3335 PMID: 17970656
20. Rohe T, Noppeney U. Cortical hierarchies perform Bayesian causal inference in multisensory perception. PLoS Biol. 2015; 13(2):e1002073. https://doi.org/10.1371/journal.pbio. 1002073 PMID: 25710328
21. Rohe T, Noppeney U. Sensory reliability shapes perceptual inference via two mechanisms. Journal of Vision. 2015; 15(5):1-22. https://doi.org/10.1167/15.5.22
22. Odegaard B, Wozny DR, Shams L. Biases in visual, auditory, and audiovisual perception of space. PLoS Comput Biol. 2015; 11(12):e1004649. https://doi.org/10.1371/journal.pcbi. 1004649 PMID: 26646312
23. Odegaard B, Shams L. The Brain's Tendency to Bind Audiovisual Signals Is Stable but Not General. Psychological Science. 2016; 27(4):583-591. https://doi.org/10.1177/0956797616626860 PMID: 26944861
24. Wozny DR, Beierholm UR, Shams L. Probability matching as a computational strategy used in perception. PLoS Computational Biology. 2010; 6(8):e1000871. https://doi.org/10.1371/journal.pcbi. 1000871 PMID: 20700493
25. Wozny DR, Shams L. Computational characterization of visually induced auditory spatial adaptation. Frontiers in Integrative Neuroscience. 2011; 5:75. https://doi.org/10.3389/fnint.2011.00075 PMID: 22069383

---

#### Page 35

26. Bejjanki VR, Knill DC, Aslin RN. Learning and inference using complex generative models in a spatial localization task. Journal of Vision. 2016; 16(5):1-9. https://doi.org/10.1167/16.5.9
27. Beierholm UR, Quartz SR, Shams L. Bayesian priors are encoded independently from likelihoods in human multisensory perception. Journal of Vision. 2009; 9(5):1-9. https://doi.org/10.1167/9.5.23 PMID: 19757901
28. van den Berg R, Vogel M, Josić K, Ma WJ. Optimal inference of sameness. Proceedings of the National Academy of Sciences. 2012; 109(8):3178-3183. https://doi.org/10.1073/pnas. 1108790109
29. Hospedales T, Vijayakumar S. Multisensory oddity detection as Bayesian inference. PLoS ONE. 2009; 4(1):e4205. https://doi.org/10.1371/journal.pone.0004205 PMID: 19145254
30. Magnotti JF, Ma WJ, Beauchamp MS. Causal inference of asynchronous audiovisual speech. Frontiers in Psychology. 2013; 4:798. https://doi.org/10.3389/fpsyg.2013.00798 PMID: 24294207
31. Sawai Ki, Sato Y, Aihara K. Auditory time-interval perception as causal inference on sound sources. Frontiers in Psychology. 2012; 3.
32. Zhou Y, Acerbi L, Ma WJ. The Role of Sensory Uncertainty in Simple Perceptual Organization. bioRxiv. 2018; p. 350082.
33. de Winkel KN, Katliar M, Bülthoff HH. Forced fusion in multisensory heading estimation. PLoS ONE. 2015; 10(5):e0127104. https://doi.org/10.1371/journal.pone.0127104 PMID: 25938235
34. de Winkel KN, Katliar M, Bülthoff HH. Causal Inference in Multisensory Heading Estimation. PLoS ONE. 2017; 12(1):e0169676. https://doi.org/10.1371/journal.pone.0169676 PMID: 28060957
35. Ma WJ. Organizing probabilistic models of perception. Trends in Cognitive Sciences. 2012; 16 (10):511-518. https://doi.org/10.1016/j.tics.2012.08.010 PMID: 22981359
36. Qamar AT, Cotton RJ, George RG, Beck JM, Prezhdo E, Laudano A, et al. Trial-to-trial, uncertaintybased adjustment of decision boundaries in visual categorization. Proceedings of the National Academy of Sciences. 2013; 110(50):20332-20337. https://doi.org/10.1073/pnas. 1219756110
37. Acerbi L, Ma WJ, Vijayakumar S. A Framework for Testing Identifiability of Bayesian Models of Perception. In: Advances in Neural Information Processing Systems 27. Curran Associates, Inc.; 2014. p. $1026-1034$.
38. Angelaki DE, Gu Y, DeAngelis GC. Multisensory integration: Psychophysics, neurophysiology, and computation. Current opinion in neurobiology. 2009; 19(4):452-458. https://doi.org/10.1016/j.conb. 2009.06.008 PMID: 19616425
39. Shams L, Beierholm UR. Causal inference in perception. Trends in cognitive sciences. 2010; 14 (9):425-432. https://doi.org/10.1016/j.tics.2010.07.001 PMID: 20705502
40. Colonius H, Diederich A. Formal models and quantitative measures of multisensory integration: a selective overview. European Journal of Neuroscience. 2018;. https://doi.org/10.1111/ejn. 13813
41. de Winkel KN, Soyka F, Barnett-Cowan M, Bülthoff HH, Groen E, Werkhoven P. Integration of visual and inertial cues in the perception of angular self-motion. Experimental Brain Research. 2013; 231 (2):209-218. https://doi.org/10.1007/s00221-013-3683-1 PMID: 24013788
42. Gu Y, Angelaki DE, DeAngelis GC. Neural correlates of multisensory cue integration in macaque MSTd. Nature Neuroscience. 2008; 11(10):1201-1210. https://doi.org/10.1038/nn. 2191 PMID: 18776893
43. Prsa M, Gale S, Blanke O. Self-motion leads to mandatory cue fusion across sensory modalities. Journal of Neurophysiology. 2012; 108(8):2282-2291. https://doi.org/10.1152/jn.00439.2012 PMID: 22832567
44. Chen SY, Ross BH, Murphy GL. Implicit and explicit processes in category-based induction: Is induction best when we don't think? Journal of Experimental Psychology: General. 2014; 143(1):227. https://doi. org/10.1037/a0032064
45. Evans JSB. Dual-processing accounts of reasoning, judgment, and social cognition. Annu Rev Psychol. 2008; 59:255-278. https://doi.org/10.1146/annurev.psych.59.103006.093629 PMID: 18154502
46. Trommershäuser J, Maloney LT, Landy MS. The expected utility of movement. In: Glimcher PW, Fehr E, Camerer CF, Poldrack RA, editors. Neuroeconomics: Decision making and the brain. Academic Press; 2008. p. 95-111.
47. Wallace MT, Roberson G, Hairston WD, Stein BE, Vaughan JW, Schirillo JA. Unifying multisensory signals across time and space. Experimental Brain Research. 2004; 158(2):252-258. https://doi.org/10. 1007/s00221-004-1899-9 PMID: 15112119
48. Acerbi L, Wolpert DM, Vijayakumar S. Internal representations of temporal statistics and feedback calibrate motor-sensory interval timing. PLoS Computational Biology. 2012; 8(11):e1002771. https://doi. org/10.1371/journal.pcbi.1002771 PMID: 23209386

---

#### Page 36

49. Acerbi L, Vijayakumar S, Wolpert DM. On the Origins of Suboptimality in Human Probabilistic Inference. PLoS Computational Biology. 2014; 10(6):e1003661. https://doi.org/10.1371/journal.pcbi. 1003661 PMID: 24945142
50. van den Berg R, Awh E, Ma WJ. Factorial comparison of working memory models. Psychological Review. 2014; 121(1):124-149. https://doi.org/10.1037/a0035234 PMID: 24490791
51. Huszár F, Noppeney U, Lengyel M. Mind reading by machine learning: A doubly Bayesian method for inferring mental representations. In: Proceedings of the Thirty-Second Annual Conference of the Cognitive Science Society; 2010. p. 2810-2815.
52. Vehtari A, Gelman A, Gabry J. Pareto smoothed importance sampling. arXiv preprint arXiv:150702646. 2015;.
53. Vehtari A, Gelman A, Gabry J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing. 2016; p. 1-20.
54. Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ. Bayesian model selection for group studies. Neuroimage. 2009; 46(4):1004-1017. https://doi.org/10.1016/j.neuroimage.2009.03.025 PMID: 19306932
55. Rigoux L, Stephan KE, Friston KJ, Daunizeau J. Bayesian model selection for group studies-revisited. Neuroimage. 2014; 84:971-985. https://doi.org/10.1016/j.neuroimage.2013.08.065 PMID: 24018303
56. Shen S, Ma WJ. A detailed comparison of optimality and simplicity in perceptual decision making. Psychological Review. 2016; 123(4):452-480. https://doi.org/10.1037/rev0000028 PMID: 27177259
57. Kass RE, Raftery AE. Bayes factors. Journal of the American Statistical Association. 1995; 90 (430):773-795. https://doi.org/10.1080/01621459.1995.10476572
58. Gelman A, Loken E. The garden of forking paths: Why multiple comparisons can be a problem, even when there is no "fishing expedition" or "p-hacking" and the research hypothesis was posited ahead of time; 2013.
59. Kaliuzhna M, Prsa M, Gale S, Lee SJ, Blanke O. Learning to integrate contradictory multisensory selfmotion cue pairings. Journal of Vision. 2015; 15(1):1-10. https://doi.org/10.1167/15.1.10 PMID: 25589294
60. Campos JL, Siegle JH, Mohler BJ, Bülthoff HH, Loomis JM. Imagined self-motion differs from perceived self-motion: evidence from a novel continuous pointing method. PLoS ONE. 2009; 4(11):e7793. https:// doi.org/10.1371/journal.pone.0007793 PMID: 19907655
61. Odegaard B, Wozny DR, Shams L. A simple and efficient method to enhance audiovisual binding tendencies. PeerJ. 2017; 5:e3143. https://doi.org/10.7717/peerj.3143 PMID: 28462016
62. Ernst MO, Bülthoff HH. Merging the senses into a robust percept. Trends in Cognitive Sciences. 2004; 8(4):162-169. https://doi.org/10.1016/j.tics.2004.02.002 PMID: 15050512
63. Ma WJ, Navalpakkam V, Beck JM, Van Den Berg R, Pouget A. Behavior and neural basis of near-optimal visual search. Nature Neuroscience. 2011; 14(6):783-790. https://doi.org/10.1038/nn.2814 PMID: 21552276
64. Adler WT, Ma WJ. Comparing Bayesian and non-Bayesian accounts of human confidence reports. bioRxiv. 2016.
65. Piironen J, Vehtari A. Comparison of Bayesian predictive methods for model selection. Statistics and Computing. 2016; p. 1-25.
66. Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB. Bayesian data analysis ( 3rd edition). CRC Press; 2013.
67. Friston KJ, Litvak V, Oswal A, Razi A, Stephan KE, van Wijk BC, et al. Bayesian model reduction and empirical Bayes for group (DCM) studies. Neuroimage. 2016; 128:413-431. https://doi.org/10.1016/j. neuroimage.2015.11.015 PMID: 26569570
68. Burnham KP, Anderson DR. Model selection and multimodel inference: A practical information-theoretic approach. Springer Science \& Business Media; 2003.
69. Spiegelhalter DJ, Best NG, Carlin BP, Van Der Linde A. Bayesian measures of model complexity and fit. Journal of the Royal Statistical Society: Series B (Statistical Methodology). 2002; 64(4):583-639. https://doi.org/10.1111/1467-9868.00353
70. Watanabe S. Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. Journal of Machine Learning Research. 2010; 11(Dec):3571-3594.
71. MacKay DJ. Information theory, inference and learning algorithms. Cambridge university press; 2003.
72. Gelman A, Hwang J, Vehtari A. Understanding predictive information criteria for Bayesian models. Statistics and Computing. 2014; 24(6):997-1016. https://doi.org/10.1007/s11222-013-9416-2

---

#### Page 37

73. Verdinelli I, Wasserman L. Computing Bayes factors using a generalization of the Savage-Dickey density ratio. Journal of the American Statistical Association. 1995; 90(430):614-618. https://doi.org/10. 1080/01621459.1995.10476554
74. Robert CP, Wraith D, Goggans PM, Chan CY. Computational methods for Bayesian model choice. In: AIP Conference Proceedings. vol. 1193. AIP; 2009. p. 251-262.
75. Caldwell A, Liu C. Target density normalization for Markov Chain Monte Carlo algorithms. arXiv preprint arXiv:14107149. 2014;.
76. Carpenter B, Gelman A, Hoffman M, Lee D, Goodrich B, Betancourt M, et al. Stan: A probabilistic programming language. Journal of Statistical Software. 2016; 20.
77. Salvatier J, Wiecki TV, Fonnesbeck C. Probabilistic programming in Python using PyMC3. PeerJ Computer Science. 2016; 2:e55. https://doi.org/10.7717/peerj-cs. 55
78. Acerbi L, Ma WJ. Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In: Advances in Neural Information Processing Systems 30; 2017. p. 1836-1846.
79. Jones DR, Schonlau M, Welch WJ. Efficient global optimization of expensive black-box functions. Journal of Global optimization. 1998; 13(4):455-492. https://doi.org/10.1023/A:1008306431147
80. Shahriari B, Swersky K, Wang Z, Adams RP, de Freitas N. Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE. 2016; 104(1):148-175. https://doi.org/10.1109/ JPROC.2015.2494218
81. Neal RM. Slice sampling. Annals of Statistics. 2003; 31(3):705-741. https://doi.org/10.1214/aos/ 1056562461
82. Gilks WR, Roberts GO, George EI. Adaptive direction sampling. The Statistician. 1994; 43(1):179-189. https://doi.org/10.2307/2348942
83. Daunizeau J, Adam V, Rigoux L. VBA: A probabilistic treatment of nonlinear models for neurobiological and behavioural data. PLoS Comput Biol. 2014; 10(1):e1003441. https://doi.org/10.1371/journal.pcbi. 1003441 PMID: 24465198
84. Krakauer JW, Ghazanfar AA, Gomez-Marin A, MacIver MA, Poeppel D. Neuroscience needs behavior: Correcting a reductionist Bias. Neuron. 2017; 93(3):480-490. https://doi.org/10.1016/j.neuron.2016.12. 041 PMID: 28182904
85. Dokka K, MacNeilage PR, DeAngelis GC, Angelaki DE. Estimating distance during self-motion: A role for visual-vestibular interactions. Journal of Vision. 2011; 11(13):1-16. https://doi.org/10.1167/ 11.13.2
86. MacNeilage PR, Zhang Z, DeAngelis GC, Angelaki DE. Vestibular facilitation of optic flow parsing. PLoS ONE. 2012; 7(7):e40264. https://doi.org/10.1371/journal.pone.0040264 PMID: 22768345
87. Greenhouse SW, Geisser S. On methods in the analysis of profile data. Psychometrika. 1959; 24 (2):95-112. https://doi.org/10.1007/BF02289823
88. Knuth DE. Two notes on notation. The American Mathematical Monthly. 1992; 99(5):403-422. https:// doi.org/10.2307/2325085
89. Gu Y, Fetsch CR, Adeyemo B, DeAngelis GC, Angelaki DE. Decoding of MSTd population activity accounts for variations in the precision of heading perception. Neuron. 2010; 66(4):596-609. https://doi. org/10.1016/j.neuron.2010.04.026 PMID: 20510863
90. Cuturi LF, MacNeilage PR. Systematic biases in human heading estimation. PLoS ONE. 2013; 8(2): e56862. https://doi.org/10.1371/journal.pone.0056862 PMID: 23457631
91. Stocker AA, Simoncelli EP. Noise characteristics and prior expectations in human visual speed perception. Nature Neuroscience. 2006; 9(4):578-585. https://doi.org/10.1038/nn1669 PMID: 16547513
92. Girshick AR, Landy MS, Simoncelli EP. Cardinal rules: Visual orientation perception reflects knowledge of environmental statistics. Nature Neuroscience. 2011; 14(7):926-932. https://doi.org/10.1038/nn. 2831 PMID: 21642976
93. Wei XX, Stocker AA. A Bayesian observer model constrained by efficient coding can explain 'antiBayesian' percepts. Nature neuroscience. 2015; 18(10):1509. https://doi.org/10.1038/nn. 4105 PMID: 26343249
94. Press WH, Flannery BP, Teukolsky SA, Vetterling WT. Numerical recipes 3rd edition: The art of scientific computing. Cambridge University Press; 2007.
95. Audet C, Dennis JE Jr. Mesh adaptive direct search algorithms for constrained optimization. SIAM Journal on Optimization. 2006; 17(1):188-217.
96. Brochu E, Cora VM, De Freitas N. A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. arXiv preprint arXiv:10122599. 2010;.

---

#### Page 38

97. Foreman-Mackey D, Hogg DW, Lang D, Goodman J. emcee: The MCMC hammer. Publications of the Astronomical Society of the Pacific. 2013; 125(925):306. https://doi.org/10.1086/670067
98. Bishop CM. Pattern recognition and machine learning. Springer; 2006.
99. Grassberger P. Entropy estimates from insufficient samplings. arXiv preprint physics/0307138. 2003;.

---

# Bayesian Comparison of Explicit and Implicit Causal Inference Strategies in Multisensory Heading Perception - Appendix

---

# Figure S1

> **Image description.** This is a set of line graphs showing results from an explicit causal inference experiment comparing two models: Bay-X-E (panel A) and Fix-C (panel B).
>
> The figure contains six separate graphs arranged in a 3×2 grid. Each graph plots "Fraction response unity" (y-axis, ranging from 0 to 1) against "Stimuli pairs" (x-axis, numbered from 1 to 99). The data is separated by reliability level: high reliability (top row, brown dots), medium reliability (middle row, golden-brown dots), and low reliability (bottom row, yellow dots).
>
> In each graph, data points (colored dots) appear in a oscillating pattern along the x-axis, and these experimental measurements are connected by light beige lines representing model fits. A horizontal dashed line at the 0.5 mark is present in each graph. The patterns show similar oscillatory behavior across all reliability conditions and between both models (Bay-X-E and Fix-C), with data points generally distributed between 0 and 1 on the y-axis.
>
> The data points and fitted lines appear nearly identical between the two model types (panel A vs. panel B), suggesting that neither model demonstrates clear superiority in fitting the experimental data across the different reliability conditions, as mentioned in the context information.

Figure S1. Explicit causal inference; model fits of full data. Results of the explicit causal inference (unity judgment) task, for two models of interest. Proportion of ‘unity’ responses for a given (svis, svest) heading direction pair (indexed from 1 to 99), and for different levels of visual cue reliability. Points are data, lines are model fits (average fit across subjects). Error bars are omitted for clarity. A: Best Bayesian model (Bay-X-E). B: Best fixed-criterion model (Fix-C). Neither model appears clearly superior across all noise levels (see main text).

---

#### Page 1

## Appendix S1 - Supplemental methods

Luigi Acerbi*, Kalpana Dokka*, Dora E. Angelaki, Wei Ji Ma

- These authors contributed equally to this work.

Contact: luigi.acerbi@\{nyu.edu, gmail.com\}

## Contents

1 Cookbook for causal inference observers ..... 2
1.1 Pick a sensory noise model ..... 3
1.1.1 Measurement distribution and likelihoods ..... 3
1.2 Pick a prior over stimuli ..... 4
1.3 Pick a causal inference strategy ..... 4
1.3.1 Bayesian causal inference strategies ..... 4
1.3.2 Non-Bayesian causal inference strategies ..... 5
1.3.3 Non-causal inference strategies ..... 5
1.4 Pick other sources of suboptimality ..... 5
2 Observer model factors ..... 6
2.1 Sensory noise ..... 6
2.2 Prior ..... 6
2.3 Causal inference strategy ..... 7
2.4 Suboptimalities ..... 10
2.5 Model parameters ..... 10
3 Comparison between wrapped normal and von Mises noise ..... 10
3.1 Theoretical comparison ..... 10
3.2 Empirical comparison ..... 11
4 Computational details ..... 11
4.1 Integrals ..... 11
4.2 Optimization ..... 12
4.3 Markov Chain Monte Carlo (MCMC) sampling ..... 13
4.4 Pareto smoothed importance sampling diagnostics ..... 13
4.5 Visualization of model fits ..... 14
4.6 Model validation and recovery ..... 14

---

#### Page 2

5 Absolute goodness of fit ..... 16
5.1 Computing the absolute goodness of fit ..... 16
5.2 Entropy of the data ..... 17
5.3 Cross-entropy ..... 17
6 LOO scores for all models ..... 18
6.1 Unity judgment task ..... 18
6.2 Bimodal inertial discrimination task ..... 19
6.3 Joint fits ..... 19
Supplemental References ..... 19

# 1 Cookbook for causal inference observers

We describe here a fairly general recipe for building an observer model for causal inference in multisensory perception. We consider the most common case of two sensory modalities (see [1] for work on three modalities). Stimuli take value on some one-dimensional physical continuum, such as location or heading direction. ${ }^{1}$ The observer model is designed to apply to three types of tasks:

- Unisensory estimation/discrimination: The observer is presented with one stimulus from either modality, and is asked to report the value of the stimulus (or how the stimulus compares to a given reference).
- Bisensory estimation/discrimination: The observer is presented with two stimuli from different modalities, and is asked to report the value of either one, or of both (or how one of the stimuli, or both, compare to a given reference). Also referred to as implicit (causal) inference.
- (Bisensory) unity judgement: The observer is presented with two stimuli from different modalities, and is asked whether they were perceived as having the same value/source. Also referred to as explicit (causal) inference.

Depending on the experimental setup, the bisensory estimation/discrimination and unity judgment tasks might be performed in the same trial (a 'dual task' setup; see for example [2,3]).

Our construction makes the following assumptions:

- When two stimuli are presented in the same trial, the observer follows a 'causal inference strategy' to decide whether the stimuli belong to a common cause $(C=1)$ or not $(C=2)$.
- Conditioned on a given causal scenario $(C=1$ or $C=2)$, or in the unisensory task, the observer performs the estimation/discrimination task according to Bayesian inference.
- When responding, the observer might exhibit additional suboptimalities, such as lapsing and cue switching.

A specific observer model is built by picking four model components (also called model factors): (1) a sensory noise model; (2) a prior over stimuli; (3) a causal inference strategy; and (4) additional sources of suboptimality.

[^0]
[^0]: ${ }^{1}$ With the risk of stating the obvious, we remark that stimulus value here is the quantity associated with the stimulus continuum and has nothing to do with 'value' in value-based decision making.

---

#### Page 3

# 1.1 Pick a sensory noise model

For each modality 'mod', pick a sensory noise model for the observer. The common assumption is a Gaussian measurement noise distribution of the form

$$
p\left(x_{\bmod } \mid s_{\bmod }\right)=\mathcal{N}\left(x_{\bmod } \mid s_{\bmod }, \sigma^{2}\left(s_{\bmod }\right)\right)
$$

where $x_{\text {mod }}$ is the noisy measurement, $s_{\text {mod }}$ the stimulus value, $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ is a normal distribution with mean $\mu$ and variance $\sigma^{2}$, and the function $\sigma^{2}\left(s_{\bmod }\right)>0$ encodes how the variance of measurement noise depends on the stimulus value, which is a feature of the sensory domain. Common shapes could be a constant noise, or alternatively noise that grows proportionally to $\left|s_{\bmod }\right|$ ( $\sim$ Weber's law). The are no constraints on the shape of $\sigma^{2}\left(s_{\bmod }\right)$ besides positivity and, arguably, continuity.

Eq. S1 is suitable for unbounded stimulus domains, or circular domains (such as orientation, or heading direction) with small angles, which effectively behave as unbounded domains. For an actually circular stimulus domain, we replace Eq. S1 with a wrapped normal distribution

$$
p\left(x_{\bmod } \mid s_{\bmod }\right)=\sum_{k=-\infty}^{\infty} \mathcal{N}\left(x_{\bmod } \mid s_{\bmod }+360^{\circ} k, \sigma^{2}\left(s_{\bmod }+360^{\circ} k\right)\right), \quad s_{\bmod } \in\left[-180^{\circ}, 180^{\circ}\right)
$$

which, for $\sigma\left(s_{\bmod }\right)<360^{\circ}$, is very well approximated by only three components $k=-1,0,1$. An alternative to Eq. S2 is to use a von Mises (i.e., circular normal) distribution; we show in Section 3 that the two choices are essentially equivalent.

### 1.1.1 Measurement distribution and likelihoods

We use Eqs. S1 (or S2) both for the sensory measurement noise distribution, that is the generative process of measurement $x_{\text {mod }}$ for a given stimulus $s_{\text {mod }}$ in the experiment, and for the observer's sensory likelihood used in the inference process of the posterior probability over $s_{\text {mod }}$ for a given measurement $x_{\text {mod }}$. According to Bayes' rule, for the example of a unisensory stimulus, the latter takes the form

$$
p\left(s_{\bmod } \mid x_{\bmod }\right)=\frac{p\left(x_{\bmod } \mid s_{\bmod }\right) p_{\text {prior }}\left(s_{\bmod }\right)}{\int p\left(x_{\bmod } \mid s_{\bmod }^{\prime}\right) p_{\text {prior }}\left(s_{\bmod }^{\prime}\right) d s_{\bmod }^{\prime}}
$$

where $p_{\text {prior }}\left(s_{\bmod }\right)$ is the prior over unisensory stimuli (see Section 1.2), and here $p\left(x_{\bmod } \mid s_{\bmod }\right)$ is the likelihood.

Previous computational work has modified the equation of the measurement distribution by including terms, such as a scaling factor in front of $x_{\text {mod }}$, not reflected in the likelihood. This form of model mismatch has the effect of introducing explicit biases in subjects' percepts. ${ }^{2}$ The rationale for this adhoc modification of the measurement distribution is that such biases are observed experimentally, see for example $[4,5]$ in the case of heading estimation. In our construction, instead, we follow the common practice in Bayesian psychophysics of assuming that biases in the observers' performance emerge implicitly and normatively from the interaction between statistics of the stimuli (i.e., priors) and precision of the sensory apparatuses (i.e., likelihoods) $[6,7]$. Recent theoretical work has shown that this might agree with encoding of stimuli in neural populations $[8,9]$. In particular, as demonstrated in these studies, priors will generally induce 'attractive' biases, whereas stimulus-dependent noise distributions (and, thus, likelihoods) can induce both 'attractive' and 'repulsive' biases. For this reason, we do not force biases by hand in the formulation of the sensory noise distribution, but this should not be mistaken for a lack of biases in the performance of our observer models.

[^0]
[^0]: ${ }^{2}$ Note that if the same modifications were included in both measurement distribution and likelihood, they would 'cancel out' in the inference.

---

#### Page 4

The fact that we use the same expressions (and parameters) for both the sensory measurement distribution and the likelihood is equivalent to saying that observers implicitly know their own noise model (that is, how noise changes as function of other parameters of the task, such as reliability and stimulus eccentricity). This modeling choice is motivated both by experimental work that shows trial-to-trial reweighing of multisensory cues [10-12] and by theoretical reasons, in that models in which noise (e.g., variance of measurement distributions) and beliefs about noise (e.g., 'variance' in the likelihoods) are decoupled may suffer from a lack of identifiability, unless the experiment is designed to avoid such issues [13].

# 1.2 Pick a prior over stimuli

The observer will have a prior over stimuli in the unisensory and bisensory conditions. A common choice for the prior is an independent, identical Gaussian prior across modalities for stimuli $s_{1}$ and $s_{2}$,

$$
p_{\text {prior }}\left(s_{1}, s_{2} \mid C=2\right)=\mathcal{N}\left(s_{1} \mid \mu_{\text {prior }}, \sigma_{\text {prior }}^{2}\right) \mathcal{N}\left(s_{2} \mid \mu_{\text {prior }}, \sigma_{\text {prior }}^{2}\right)
$$

where $\mu_{\text {prior }}$ is the mean of the prior (which might represent a global bias, often assumed to be zero), and $\sigma_{\text {prior }}^{2}$ represents the width of the prior (the wider the prior, the lesser its influence on behavior). The same prior is then applied to the common cause scenario and unisensory cases,

$$
p_{\text {prior }}\left(s \mid C=1\right)=\mathcal{N}\left(s \mid \mu_{\text {prior }}, \sigma_{\text {prior }}^{2}\right)
$$

This simple prior induces a 'compression' or 'regression to the mean' bias as observed in many psychophysical experiments [14].

Another possibility is that the observer develops a prior (approximately) based on the empirical distribution of stimuli presented in the experiment, which may differ from Eqs. S4 and S5.

### 1.3 Pick a causal inference strategy

The causal inference strategy defines how the observer decides on the hypotheses $C=1$ and $C=2$ when presented with two stimuli. In general, the causal inference strategy may or may not be Bayesian, can be deterministic or stochastic, and might dictate to combine the two causal scenarios (e.g., by performing a weighted average of $C=1$ and $C=2$ ). This strategy also determines what the observer would report in an explicit, unity-judgment task.

### 1.3.1 Bayesian causal inference strategies

A Bayesian strategy will compute the posterior probabilities of the two causal scenarios, given the two noisy measurements $x_{1}, x_{2}$, as follows,

$$
\begin{aligned}
p\left(C \mid x_{1}, x_{2}\right) & \propto p\left(x_{1}, x_{2} \mid C\right) p(C) \\
& =p(C) \int p\left(x_{1}, x_{2}, s_{1}, s_{2} \mid C\right) d s_{1} d s_{2} \\
& =p(C) \int p\left(x_{1} \mid s_{1}\right) p\left(x_{2} \mid s_{2}\right) p_{\text {prior }}\left(s_{1}, s_{2} \mid C\right) d s_{1} d s_{2}
\end{aligned}
$$

where $p(C)$ represents the prior belief of a common or separate cause, with $p(C=1)=1-p(C=2) \equiv p_{\mathrm{c}}$. While $p_{\mathrm{c}}$ should typically stem from the statistics of the task, it is general practice to keep it as a free parameter of any Bayesian model, since subjects tend to exhibit a wide spectrum of beliefs about the probability of a common cause (see Fig 2 in [15]).

Different variants of Bayesian observers will use the posterior over causal scenarios differently to respond to estimation/discrimination task. Typical models are Bayesian model averaging (average the

---

#### Page 5

estimates of $C=1$ and $C=2$, weighted by their posterior probability), Bayesian model selection (pick the estimate of either $C=1$ or $C=2$, based on which one has the larger posterior probability), or Bayesian probability matching (pick either scenario stochastically, with probability equal to their posterior probability).

For the unity judgment task, the standard Bayesian strategy is to respond with the scenario $(C=1$ or $C=2$ ) with highest posterior probability. Another possibility is posterior probability matching, that is pick either scenario stochastically, with probability equal to their posterior.

# 1.3.2 Non-Bayesian causal inference strategies

The main feature of a non-Bayesian strategy is that it does not compute a posterior distribution over causal scenarios, but uses instead (usually simpler) heuristics as a decision rule to whether $C=1$ or $C=2$.

A typical heuristic of this kind stipulates that $C=1$ whenever the two noisy measurements $x_{1}, x_{2}$ are closer in value than some criterion $\kappa$, that is $\left|x_{1}-x_{2}\right|<\kappa$. If $\kappa$ is fixed for all experimental conditions, we call this a fixed-criterion causal inference strategy [16]. If $\kappa$ is allowed to change for different experimental conditions, and in particular as a function of stimulus reliability, then the decision rule becomes 'probabilistic', that is uncertainty-dependent [17].

A fixed-criterion strategy that discards reliability information might seem to clash with the assumption that observers know the stimulus reliability when combining cues. However, there is neural evidence that sensory integration (that is forced fusion, with reliability-dependent weighing) and causal inference happen in different brain areas [18]. For this reason, it is not obvious that reliability information would be automatically available to higher areas, or that it would be used in the correct way. Fixed-criterion models represent a valid 'null' alternative for a class of models in which reliability information is unavailable (or corrupted) at the causal inference stage.

### 1.3.3 Non-causal inference strategies

Extreme cases of causal inference strategies are observers that do not quite perform causal inference at all.

In this case, an observer might use a forced fusion strategy that always combines cues $(C \equiv 1)$, or, alternatively, a forced segregation strategy that always segregates them $(C \equiv 2)$. Mathematically, these strategies can be considered as limiting cases of previously presented causal inference strategies. For example, forced fusion is equivalent to a Bayesian causal inference strategy with $p_{\mathrm{c}} \rightarrow 1$, or a fixedcriterion strategy with $\kappa \rightarrow \infty$. Analogously, forced segregation is equivalent to a Bayesian strategy with $p_{\mathrm{c}} \rightarrow 0$, or a fixed-criterion strategy with $\kappa \rightarrow 0$.

As a generalization of forced fusion/segregation, we can consider a stochastic fusion strategy that on each trial has probability $\eta$ of deciding $C=1$, and $C=2$ otherwise, where $\eta$ might depend on the experimental condition.

### 1.4 Pick other sources of suboptimality

Experimental subjects will often exhibit additional sources of variability, which might be included explicitly in the model. Here we consider lapses and cue switching.

A common feature of many psychophysical models is a lapse rate, that is the probability $\lambda$ that the observer gives a completely random response (typically, uniform over the range of possible responses) [19].

Another form of error for multisensory perception experiments is that the observer switches modality, that is in a bisensory estimation/discrimination task they respond about the wrong modality with switching rates $\rho_{1 \rightarrow 2}$ and $\rho_{2 \rightarrow 1}$, respectively for responding with the second modality when asked about the first, and vice versa. Note that the switching rate can be used to implement suboptimal strategies

---

#### Page 6

such as cue capture, whereby all responses are absorbed by a single modality: pick the 'forced segregation' causal inference strategy, then set, say, $\rho_{2 \rightarrow 1}=1$ and $\rho_{1 \rightarrow 2}=0$, if responses are supposed to be captured by the first modality. Similarly, by picking 'forced segregation' with nonzero $\rho_{1 \rightarrow 2}$ and $\rho_{2 \rightarrow 1}$, one can implement a switching strategy observer [20].

# 2 Observer model factors

In this section we describe details of the factors used to build the observer models in the paper.

### 2.1 Sensory noise

For a given modality $\bmod \in\{$ vis, vest $\}$, the measurement noise distribution follows Eq. S1. Note that for a visual stimulus the measurement distribution and the variance in Eq. S1 also depend on the visual coherence level $c_{\text {vis }}$ in the trial, such that $\sigma^{2}\left(s_{\text {vis }}\right) \equiv \sigma^{2}\left(s_{\text {vis }}, c_{\text {vis }}\right)$, but in the following we omit this dependence to simplify the notation.

For the variance we consider two possible models,

$$
\sigma^{2}\left(s_{\bmod }\right)=\left\{\begin{array}{ll}
\sigma_{0 \bmod }^{2} & \text { (constant) } \\
\sigma_{0 \bmod }^{2}\left\{1+2 w_{\bmod }^{2}\left(\frac{90^{\circ}}{\pi}\right)^{2}\left[1-\cos \left(\frac{s_{\bmod }}{90^{\circ}} \pi\right)\right]\right\} & \text { (eccentricity-dependent) }
\end{array}\right.
$$

where $\sigma_{0 \text { modality }}^{2}$ is the base variance and $w_{\text {mod }}$ is related to the Weber fraction near $0^{\circ}$. In fact, for small values of $s_{\text {mod }}$, Eq. S7 reduces to $\sigma^{2}\left(s_{\bmod }\right) \approx \sigma_{0 \bmod }^{2}\left(1+w_{\bmod }^{2} s_{\bmod }^{2}\right)$, which is a generalized Weber's law. ${ }^{3}$

The broad shape of the chosen periodic formula for the eccentricity-dependent noise model, which peaks at $\pm 90^{\circ}$, derives from empirical results in a visuo-vestibular task with the same apparatus with human and monkey subjects (see Fig 2 in [4]; see also [22]). We note that our noise shape differs from that adopted in other works (with different setups), which used a sinusoidal with twice the frequency that peaks at $\pm 45^{\circ}, \pm 135^{\circ}[23,24]$. Since in our setup the heading directions were restricted to the $\pm 45^{\circ}$ range (with most directions in the $\pm 25^{\circ}$ range), the exact shape of periodicity is largely irrelevant, but understanding differences in noise models may be important for experiments with wider heading direction ranges.

For the paper, we implemented the measurement distribution (and, thus, the stimulus likelihood in the inference process) as a mixture of three wrapped Gaussians (Eq. S2). However, we found that, due to the limited range of directions in our experiment, a single Gaussian was sufficient. Note that our choice of using Gaussians rather than von Mises (circular normal) distributions yields no loss of generality in practice, as we demonstrate in Section 3.

All constant noise models have four parameters ( $\sigma_{0 \text { vest }}$, and a separate $\sigma_{0 \text { vis }}$ for each visual coherence level, low, medium and high). Eccentricity-dependent models have two additional parameters, $w_{\text {vest }}$ and $w_{\text {vis }}$ (the latter is common to all visual stimuli, to prevent overfitting).

### 2.2 Prior

For unisensory trials, we assume that observers have a unimodal symmetric prior over heading directions, peaked at $0^{\circ}$ (the exact shape is irrelevant). Due to the form of the decision rule for the left/right discrimination task, such prior has no influence over the observer's response, which only depends on whether the noisy measurement falls to the left or to the right of straight ahead.

[^0]
[^0]: ${ }^{3}$ Here by Weber's law we simply denote the fact that noise scales proportionally to stimulus magnitude, that is $\sigma(s) \propto|s|$. Technically, Weber's law is defined only for quantity-related continua, whereas heading is a quality-related continuum [21].

---

#### Page 7

For bisensory trials (both unity judgment and inertial discrimination tasks), we consider two alternative models for priors. The empirical prior consists of an approximation of the actual prior used in the experiment, that is

$$
\begin{aligned}
& p\left(s_{\text {vis }}, s_{\text {vest }} \mid C=1\right) \propto \sum_{(s, s) \in \mathcal{S}} \mathcal{N}\left(s \mid 0, \sigma_{\text {prior }}^{2}\right) \delta\left(s_{\text {vis }}-s\right) \delta\left(s_{\text {vest }}-s\right) \\
& p\left(s_{\text {vis }}, s_{\text {vest }} \mid C=2\right) \propto \sum_{\substack{\left(s_{i}, s_{j}\right) \in \mathcal{S} \\
s_{i} \neq s_{j}}} \mathcal{N}\left(\frac{s_{\text {vis }}+s_{\text {vest }}}{2} \mid 0, \sigma_{\text {prior }}^{2}\right) \mathcal{N}\left(s_{\text {vest }}-s_{\text {vis }} \mid 0, \Delta_{\text {prior }}^{2}\right) \delta\left(s_{\text {vis }}-s_{i}\right) \delta\left(s_{\text {vest }}-s_{j}\right)
\end{aligned}
$$

where $\mathcal{S}$ is the discrete set of pairs of visual and vestibular headings in the experiment. The two equations consider respectively only diagonal elements (equal heading directions, $C=1$ ) or off-diagonal elements (different directions, $C=2$ ) of Fig 1B in the main text. The approximation here is given by the two Gaussian distributions (defined on the discrete set), which impose additional shrinkage for the mean of the stimuli (governed by $\sigma_{\text {prior }}^{2}$ ) and for the disparity (governed by $\Delta_{\text {prior }}^{2}$ ). For $\sigma_{\text {prior }}^{2}, \Delta_{\text {prior }}^{2} \rightarrow \infty$, Eq. S8 converges to the distributions of directions used in the experiment for $C=1$ and $C=2$.

Alternatively, we consider an independent prior, that is

$$
\begin{aligned}
& p\left(s_{\text {vis }}, s_{\text {vest }} \mid C=1\right)=\int \mathcal{N}\left(s \mid 0, \sigma_{\text {prior }}^{2}\right) \delta\left(s_{\text {vis }}-s\right) \delta\left(s_{\text {vest }}-s\right) d s \\
& p\left(s_{\text {vis }}, s_{\text {vest }} \mid C=2\right)=\mathcal{N}\left(s_{\text {vis }} \mid 0, \sigma_{\text {prior }}^{2}\right) \mathcal{N}\left(s_{\text {vest }} \mid 0, \sigma_{\text {prior }}^{2}\right)
\end{aligned}
$$

which assumes observers build a single prior over heading directions which is applied independently to both modalities [25]. The first integral is a formal way to impose $s \equiv s_{\text {vis }}=s_{\text {vest }}$.

We note that a continuous approximation of Eq. S8 may seem more realistic than the adopted discrete distribution of directions. However, an observer model with a correlated, continuous prior is computationally intractable since evaluation of the log likelihood involves a non-analytical four-dimensional integral, which increases the computational burden by an order of magnitude. As a sanity check, we implemented observers that use a continuous approximation of Eq. S8 and verified on a subset of observers and models that results of model fits and model predictions were indeed nearly identical to the discrete case.

Independent prior models have one parameter $\sigma_{\text {prior }}$ for the width of the prior over headings. Empirical prior models have an additional parameter $\Delta_{\text {prior }}$ for the width of the prior over disparities.

# 2.3 Causal inference strategy

The basic causal inference strategies: Bayesian, fixed-criterion and fusion are described in the main text. We report here some additional definitions and derivations.

All integrals in this section are in the $\left[-90^{\circ}, 90^{\circ}\right]$ range, unless noted otherwise. The rationale of such integration range for our experiment is that subjects were informed that the movement was forward (either left or right of straight-forward). Moreover, due to the relatively narrow range of stimuli used in our experiment, we found with preliminary analyses that beliefs more than $90^{\circ}$ away from straight-ahead had negligible influence on left/right decisions. In the more general case of stimuli distributed along the full circle, the integration range should go to $\pm 180^{\circ}$. For a non-circular dimension, appropriate empirical bounds should be chosen (e.g., the width of the projection screen for a localization task).

## Posterior probability of causal structure

For a Bayesian observer, the posterior probability of common cause is

$$
\operatorname{Pr}\left(C=1 \mid x_{\text {vest }}, x_{\text {vis }}, c_{\text {vis }}\right) \propto p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }} \mid C=1\right) \operatorname{Pr}(C=1)
$$

---

#### Page 8

where $\operatorname{Pr}(C=1) \equiv p_{c}$, the prior probability of a common cause, is a free parameter of the model. Then

$$
\begin{aligned}
& p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis} \mid} C=1\right)= \\
& \quad=\operatorname{Pr}\left(c_{\mathrm{vis}}\right) \iint p\left(x_{\mathrm{vis}} \mid s_{\mathrm{vis}}, c_{\mathrm{vis}}\right) p\left(x_{\mathrm{vest}} \mid s_{\mathrm{vest}}\right) p\left(s_{\mathrm{vis}}, s_{\mathrm{vest}} \mid C=1\right) d s_{\mathrm{vis}} d s_{\mathrm{vest}}
\end{aligned}
$$

where the likelihoods are defined by Eq. S1, the prior is defined by Eqs. S8 and S9, and $\operatorname{Pr}\left(c_{\text {vis }}\right)=\frac{1}{3}$. For the independent prior case we can further simplify

$$
p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}} \mid C=1\right) \propto \int p\left(x_{\mathrm{vis}} \mid s_{\mathrm{vis}}=s_{\mathrm{vest}}, c_{\mathrm{vis}}\right) p\left(x_{\mathrm{vest}} \mid s_{\mathrm{vest}}\right) \mathcal{N}\left(s_{\mathrm{vest}} \mid 0, \sigma_{\mathrm{prior}}^{2}\right) d s_{\mathrm{vest}}
$$

whereas the solution for the empirical prior is similar, but with a sum over the discrete stimuli such that $s_{\text {vis }}=s_{\text {vest }}$.

Conversely, the posterior probability of separate causes is

$$
\operatorname{Pr}\left(C=2 \mid x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}}\right) \propto p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}} \mid C=2\right)\left(1-p_{c}\right)
$$

where

$$
p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}} \mid C=2\right)=\operatorname{Pr}\left(c_{\mathrm{vis}}\right) \iint p\left(x_{\mathrm{vis}} \mid s_{\mathrm{vis}}, c_{\mathrm{vis}}\right) p\left(x_{\mathrm{vest}} \mid s_{\mathrm{vest}}\right) p\left(s_{\mathrm{vis}}, s_{\mathrm{vest}} \mid C=2\right) d s_{\mathrm{vis}} d s_{\mathrm{vest}}
$$

which for the independent prior becomes

$$
p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}} \mid C=2\right) \propto\left(\int p\left(x_{\mathrm{vis}} \mid s_{\mathrm{vis}}, c_{\mathrm{vis}}\right) p_{\mathrm{prior}}\left(s_{\mathrm{vis}}\right) d s_{\mathrm{vis}}\right) \cdot\left(\int p\left(x_{\mathrm{vest}} \mid s_{\mathrm{vest}}\right) p_{\mathrm{prior}}\left(s_{\mathrm{vest}}\right) d s_{\mathrm{vest}}\right)
$$

that is the product of two one-dimensional integrals. For the empirical prior Eq. S11 does not simplify, but becomes a discrete sum over $\mathcal{S}$ (see Eq. S8).

# Posterior probability of left/right discrimination $(C=1)$

In bisensory inertial discrimination trials the observer may implicitly contemplate two scenarios: that there is only one common cause $(C=1)$, or that there are two distinct causes $(C=2)$. We consider inference in the two separate scenarios, and then see how the observer can combine them.

For $C=1$, the observer's posterior probability density over over the inertial heading direction is

$$
\begin{aligned}
p( & \left.s_{\mathrm{vest}} \mid x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}}, C=1\right)= \\
& =\int \frac{p\left(s_{\mathrm{vis}}, s_{\mathrm{vest}}, x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}}, C=1\right)}{p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}}, C=1\right)} d s_{\mathrm{vis}} \\
& =\int \frac{p\left(s_{\mathrm{vis}}, s_{\mathrm{vest}}, x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}} \mid C=1\right) \operatorname{Pr}(C=1)}{p\left(x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}} \mid C=1\right) \operatorname{Pr}(C=1)} d s_{\mathrm{vis}} \\
& \propto \int p\left(x_{\mathrm{vest}} \mid s_{\mathrm{vest}}\right) p\left(x_{\mathrm{vis}} \mid s_{\mathrm{vis}}, c_{\mathrm{vis}}\right) p\left(s_{\mathrm{vis}}, s_{\mathrm{vest}} \mid C=1\right) d s_{\mathrm{vis}}
\end{aligned}
$$

which for the independent prior becomes

$$
p\left(s_{\mathrm{vest}} \mid x_{\mathrm{vis}}, x_{\mathrm{vest}}, c_{\mathrm{vis}}, C=1\right) \propto p\left(x_{\mathrm{vest}} \mid s_{\mathrm{vest}}\right) p\left(x_{\mathrm{vis}} \mid s_{\mathrm{vis}}=s_{\mathrm{vest}}, c_{\mathrm{vis}}\right) \mathcal{N}\left(s_{\mathrm{vest}} \mid 0, \sigma_{\mathrm{prior}}^{2}\right)
$$

and the solution is similar for the empirical prior, constraining $s_{\text {vest }}$ to take only the discrete values used in the experiment for $C=1$.

---

#### Page 9

# Posterior probability of left/right discrimination $(C=2)$

For $C=2$, the observer's posterior over inertial heading is

$$
\begin{aligned}
& p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right)= \\
& \quad=\int \frac{p\left(s_{\text {vis }}, s_{\text {vest }}, x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right)}{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right)} d s_{\text {vis }} \\
& \quad \propto \int p\left(x_{\text {vest }} \mid s_{\text {vest }}\right) p\left(x_{\text {vis }} \mid s_{\text {vis }}\right) p\left(s_{\text {vis }}, s_{\text {vest }} \mid C=2\right) d s_{\text {vis }}
\end{aligned}
$$

which for the independent prior can be further simplified as

$$
p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right) \propto p\left(x_{\text {vest }} \mid s_{\text {vest }}\right) \mathcal{N}\left(s_{\text {vest }} \mid 0, \sigma_{\text {prior }}^{2}\right)
$$

whereas for the empirical prior the integral in Eq. S13 becomes a sum over discrete pairs of heading directions used in the experiment.

## Posterior probability of left/right discrimination ( $C$ unknown)

If the causal structure is unknown, a Bayesian observer that follows a 'model averaging' strategy marginalizes over possible causal structures (here, $C=1$ and $C=2$ ) [25]. The observer's posterior probability density over the inertial heading direction is

$$
\begin{aligned}
& p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)= \\
& =\sum_{C=1,2} \int \frac{p\left(s_{\text {vis }}, s_{\text {vest }}, x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C\right)}{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)} d s_{\text {vis }} \\
& =\frac{1}{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)}\left[\int p\left(s_{\text {vis }}, s_{\text {vest }}, x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=1\right) d s_{\text {vis }}+\right. \\
& \left.\int p\left(s_{\text {vis }}, s_{\text {vest }}, x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right) d s_{\text {vis }}\right] \\
& =\frac{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=1\right)}{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)} p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=1\right)+ \\
& \frac{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right)}{p\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)} p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right) \\
& =\operatorname{Pr}\left(C=1 \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right) \cdot p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=1\right)+ \\
& \operatorname{Pr}\left(C=2 \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right) \cdot p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right)
\end{aligned}
$$

where $p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C\right)$ has been defined in the previous subsections and $\operatorname{Pr}\left(C \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)$ is the posterior over causal structures.

We generalize Eq. S14 as

$$
\begin{gathered}
p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)=v_{1}\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right) \cdot p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=1\right)+ \\
v_{2}\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right) \cdot p\left(s_{\text {vest }} \mid x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}, C=2\right)
\end{gathered}
$$

where $v_{k}\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)$, for $k=1,2$, are the posterior causal weights assigned by the observer to the two causal structures, with $v_{2}\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right)=1-v_{1}\left(x_{\text {vis }}, x_{\text {vest }}\right)$ and $0 \leq v_{1}\left(x_{\text {vis }}, x_{\text {vest }}, c_{\text {vis }}\right) \leq 1$. For a Bayesian observer, the causal weights are equal to the posterior probabilities (Eq. S14); in the main text we describe other models.

---

#### Page 10

# 2.4 Suboptimalities

For all our observer models, we considered a lapse rate $\lambda$. Due to the format of our bisensory discrimination data (i.e., only inertial left/right responses), which limits the identifiability of switching models, we did not consider a switching rate, leaving that to future work.

### 2.5 Model parameters

All models except stochastic fusion have five parameters $\boldsymbol{\theta}_{\text {default }}$ by default: three visual base noise parameters $\sigma_{0 \text { vis }}\left(c_{\text {high }}\right), \sigma_{0 \text { vis }}\left(c_{\text {med }}\right)$, and $\sigma_{0 \text { vis }}\left(c_{\text {low }}\right)$; a vestibular base noise parameter $\sigma_{0 \text { vest }} ;$ and a lapse rate $\lambda$.

| Observer model                          | Parameters                                                                        |  \# |
| :-------------------------------------- | :-------------------------------------------------------------------------------- | --: |
| Bayesian (unisensory only)              | $\boldsymbol{\theta}_{\text {default }}$                                          |   5 |
| Bayesian causal inference               | $\boldsymbol{\theta}_{\text {default }}, \sigma_{\text {prior }}, p_{\mathrm{c}}$ |   7 |
| Fixed-criterion causal inference        | $\boldsymbol{\theta}_{\text {default }}, \kappa_{\mathrm{c}}$                     |   6 |
| Fusion causal inference                 | $\boldsymbol{\theta}_{\text {default }}$                                          |   5 |
| Stochastic fusion (unity judgment only) | $\eta_{\text {high }}, \eta_{\text {med }}, \eta_{\text {low }}$                  |   3 |
| Add-ons                                 | +                                                                                 |   + |
| with eccentricity-dependent noise       | +                                                                                 |   + |
| with empirical priors (Bayesian)        | +                                                                                 |   + |
| with empirical priors (non-Bayesian)    | +                                                                                 |   + |

## 3 Comparison between wrapped normal and von Mises noise

In the presentation of our general causal inference observer model, and in the manuscript, we assumed that measurement noise distributions took the shape of (wrapped) normals (see Eqs. S1 and S2). Moreover, for wrapped normals, we advocated that three mixture components $(k=0, \pm 1)$ are sufficient. Our modeling proposal differs from the typical choice of using von Mises (circular normal) distributions for circular variables (see for example [23,24]). Here we test whether our choice is sensible and generally applicable, by asking whether there is a practical difference between using von Mises and wrapped normals, for experiments with stimuli over the entire circular domain.

First, we note that, qualitatively, the von Mises and wrapped normals have very similar properties. They are both bell-shaped distributions over the circle, and they are both related to the normal distribution. von Mises distributions are the maximum-entropy distributions over the circle, so theoretically more appealing, but on the other hand wrapped normals, especially as a mixture of three Gaussians (one at the mean, the other two at $\pm 360^{\circ}$ from the mean), have computational advantages. It remains to be established whether these distributions differ quantitatively in an empirically meaningful way. In the following analyses, we always consider wrapped normals approximated with three mixture components.

### 3.1 Theoretical comparison

To answer this question theoretically, we assess the difference between the two noise distributions by computing the Kullback-Leibler (KL) divergence between a von Mises distribution with a given concentration parameter $\kappa$ and the best approximating wrapped normal (this construction assumes that the true underlying distribution is a von Mises, but the results are similar after inverting the role). The KL-divergence represents the expected difference in log likelihood between the two noise models per trial (assuming the data were generated from a von Mises). Thus, the inverse of the KL-divergence can be taken as a ballpark of the minimum number of samples required to empirically see a difference between the two models (that is, one point of log likelihood of difference summed over trials). We call this quantity the identifiability threshold.

---

#### Page 11

As expected, for large values of $\kappa$ (when the von Mises converges to a normal distribution) and for small values of $\kappa$ (when the von Mises converges to a uniform distribution over the circle), the identifiability threshold between wrapped normal and von Mises is way over $10^{3}$, and even $10^{4}$, meaning that several thousand trials would be needed to distinguish the two models (assuming no other confounding elements). However, there is a range of values of $\kappa$, around $\approx 50-60$ (that is, a circular SD of $\approx 7^{\circ}$ ), in which the identifiability threshold drops to $\approx 60-100$. This analysis tells that, at least in some cases, the models could be distinguished within a large but feasible amount of trials. Whether the two noise models can be distinguished in practice is an empirical question, since in real data differences in the noise models will be obfuscated by other details. Moreover, it is possible that neither model is the true one (but they could be both equally good at approximating the true model). Finally, subjects' typical parameters might reside in ranges in which the two distributions are not empirically distinguishable.

# 3.2 Empirical comparison

To answer this question empirically, we took the data from a recent paper on causal inference in multisensory heading estimation [24]. For all subjects ( 17 datasets between Experiment 1 and 2, 400-600 trials per dataset), we fit the unisensory data (four conditions: one visual and three inertial) using the basic modeling framework described in the section "Analyses of Unisensory Data" of [24]. One minor difference with their analysis is that, as a principled way of dealing with outliers, we added for each subject a lapse rate parameter, shared across conditions (instead of discarding data points more than three standard deviations away from the mean). The lapse rate represents the probability of a completely random response (e.g., due to a missed stimulus, or a mistake in the response).

Crucially, we considered two models, one in which the noise model is a von Mises (as per [24]), and another one in which the noise model is a wrapped Gaussian (implemented as a mixture of Gaussians with three components). We fitted each dataset to both models via maximum-likelihood estimation. For the optimization, we used MATLAB's fmincon function with 100 random restarts, plus one starting point represented by the maximum-likelihood solution reported in [24, S2 Table]. Since both models have the same number of parameters (and, moreover, all parameters have the same meaning), we can directly compare differences in log likelihood without the need to account for model complexity. Across subjects, we found a difference of log likelihood of $0.13 \pm 0.18$ (mean $\pm$ S.E.M.), which is negligible evidence in favor of the von Mises distribution. In fact, most of the evidence comes from a single subject; otherwise, eight subjects slightly favor the wrapped normal, and other eight slightly favor the von Mises. These results show that the two models are practically indistinguishable in real continuous estimation data. Note that this would be even more so with our data, since we have only discrete (binary) responses.

In conclusion, these analyses support our choice of using (wrapped) normals as an equivalent alternative to von Mises distributions, and suggest that wrapped normals, approximated via three mixture components, could be used more generally as a valid computational alternative to von Mises distributions.

## 4 Computational details

We describe in this section a number of computational and algorithmic details.

### 4.1 Integrals

Due to lack of analytical solutions, we computed all one-dimensional and two-dimensional integrals numerically, via either Simpson's or trapezoidal rule with a equi-spaced grid on the integration domain [26]. We had two types of integrals: integrals over $x_{\text {vis }}, x_{\text {vest }}$ for marginalization over the noisy stimuli, and integrals over $s_{\text {vis }}$ and/or $s_{\text {vest }}$ for computation of the observer's decision rule (Eqs. S10, S11, S12 and S13).

---

#### Page 12

For marginalization over noisy measurement $x_{\text {vis }}$ and $x_{\text {vest }}$, we used a regular $401 \times 401$ grid for which we adjusted the range of integration in each modality to up to 5 SD from the mean of the noisy measurement distribution (or $\pm 180^{\circ}$, whichever was smaller). For large noise, we used wrapped normal distributions, which turned out to have little effect due to our setup.

For computation of the decision rule, we assumed that observers believed, due to the experimental setup and task instructions, that the movement direction would be forward, so limited to the $\pm 90^{\circ}$ range. We adjusted the integration grid spacing $\Delta s$ (hence the number of grid points) adaptively for each parameter vector $\boldsymbol{\theta}$, defining

$$
\begin{aligned}
\sigma_{\min }\left(\boldsymbol{\theta}, c_{\mathrm{vis}}\right) & =\min \left\{\sigma_{0 \mathrm{vis}}\left(c_{\mathrm{vis}}\right), \sigma_{0 \mathrm{vest}}, \sigma_{\text {prior }}\right\} \\
\Delta s & \equiv \frac{\sigma_{\min }\left(\boldsymbol{\theta}, c_{\mathrm{vis}}\right)}{4} \quad \text { with } \quad \frac{1}{8} \leq \Delta s \leq 1
\end{aligned}
$$

and we rounded $\Delta s$ to the lowest exact fraction of the form $\frac{1}{m}$, with $m \in \mathbb{N}$ and $1 \leq m \leq 8$. The above heuristic afforded fast and accurate evaluation of the integrals, since the grid spacing was calibrated to be smaller than the length scale of the involved distributions (measurement noise and prior).

Finally, we note that we tried other standard numerical integration methods which were ineffective. Gauss-Hermite quadrature [26] led to large numerical errors because the integrand is discontinuous and bounded, a very bad fit for a polynomial. Global adaptive quadrature methods (such as quad in MATLAB, and other custom-made implementations) were simply too slow, even when reducing the requested precision. We coded all two-dimensional numerical integrals in C (via mex files in MATLAB) for maximal performance.

# 4.2 Optimization

For optimization of the log likelihood (maximum-likelihood estimation), we used Bayesian Adaptive Direct Search (BADS [27]; https://github.com/lacerbi/bads). BADS follows a mesh adaptive direct search (MADS) procedure that alternates POLL steps and SEARCH steps. In the POLL step, points are evaluated on a (random) mesh by taking one step in one coordinate direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise. In the SEARCH step, a Gaussian process is fit to a (local) subset of the points evaluated so far. Points to evaluate during the search are iteratively chosen by maximizing the predicted improvement (with respect to the current optimum) over a set of candidate points. Adherence to the MADS framework guarrantees convergence to a (local) stationary point of a noiseless function under general conditions [28]. The basic scheme is enhanced with heuristics to accelerate the POLL step, to update the Gaussian process hyperparameters, to generate a good set of candidate points in the search step, and to deal robustly with noisy functions. See [27] for details.

For each optimization run, we initialized our algorithm by randomly choosing a point inside a hypercube of plausible parameter values in parameter space. We refined the output of each BADS run with a run of patternsearch (MATLAB). To avoid local optima, for each optimization problem we performed 150 independent restarts of the whole procedure and picked the highest log likelihood value.

As a heuristic diagnostic of global convergence, we computed by bootstrap the value of the global optimum we would have found had we only used $n_{\mathrm{r}}$ restarts, with $1 \leq n_{\mathrm{r}} \leq 150$. We define the 'estimated regret' as the difference between the actual best value of the log likelihood found and the bootstrapped optimum. For each optimization problem, we computed the minimum value $n_{\mathrm{r}}^{*}$ for which the probability of having an estimated regret less than 1 was $99 \%\left(n_{\mathrm{r}}^{*} \equiv \infty\right.$ if such $n_{\mathrm{r}}$ does not exist). The rationale is that if the optimization landscape presents a large number of local optima, and new substantially improved optima keep being found with increasing $n_{\mathrm{r}}$, the bootstrapped estimated regret would keep changing with $n_{\mathrm{r}}$, and $n_{\mathrm{r}}^{*}$ would be 150 or $\infty$. For almost all optimization problems, we found $n_{\mathrm{r}}^{*} \ll 150$. This suggests that the number of restarts was large enough; although no optimization procedure in a non-convex setting can guarantee convergence to a global optimum in a finite time without further assumptions.

---

#### Page 13

# 4.3 Markov Chain Monte Carlo (MCMC) sampling

As a complementary approach to maximum-likelihood model fitting, for each dataset and model we calculated the posterior distribution of the parameters via MCMC (see main text).

We used a custom-written sampling algorithm that combines slice sampling [29] with adaptive direction sampling [30]. ${ }^{4}$ Slice sampling is a flexible MCMC method that, in contrast with the common Metropolis-Hastings transition operator, requires very little tuning in the choice of length scale. Adaptive direction sampling is an ensemble MCMC method that shares information between several dependent chains (also called 'walkers' [31]) in order to speed up mixing and exploration of the state space. For each ensemble we used $2(p+1)$ walkers, where $p$ is the number of parameters of the model. Walkers were initialized to a neighborhood of the best local optima found by the optimization algorithm. Each ensemble was run for $10^{4}$ to $2.5 \cdot 10^{4}$ burn-in steps that were discarded, after which we collected $5 \cdot 10^{3}$ to $10^{4}$ samples per ensemble.

At each step, our method iteratively selects one walker in the ensemble and first attempts an independent Metropolis update. The proposal distribution for the independent Metropolis is a variational mixture of Gaussians [32] fitted to a fraction of the samples obtained during burn-in via the vbgmm toolbox for MATLAB. ${ }^{5}$ Note that the proposal distribution is fixed at the end of burn-in and does not change thereafter, ensuring that the Markov property is not affected (although non-Markovian adaptive MCMC methods could be applied; see [33]). After the Metropolis step, the method randomly applies with probability $1 / 3$ one of three Markov transition operators to the active walker: coordinate-wise slice sampling [29], parallel-direction slice sampling [34], and adaptive-direction slice sampling [29, 30]. We also fit a variational Gaussian mixture model to the last third of the samples at the end of the burnin period, and we used the variational mixture as a proposal distribution for an independent Metropolis step which was attempted at every step.

For each dataset and model, we ran three independent ensembles. We visually checked for convergence the marginal pdfs and distribution of log likelihoods of the three sampled chains. For all parameters, we computed Gelman and Rubin's potential scale reduction statistic $R$ and effective sample size $n_{\text {eff }}[35]$ using Simo Särkkä and Aki Vehtari's psrf function for MATLAB. ${ }^{6}$ For each dataset and model, we looked at the largest $R\left(R_{\max }\right)$ and smallest $n_{\text {eff }}\left(n_{\text {effmin }}\right)$ across parameters. Large values of $R$ indicate convergence problems whereas values close to 1 suggest convergence. $n_{\text {eff }}$ is an estimate of the actual number of independent samples in the chains; a few hundred independent samples are sufficient for a coarse approximation of the posterior [35]. Longer chains were run when suspicion of a convergence problem arose from any of these methods. Samples from independent ensembles were then combined (thinned, if necessary), yielding $1.5 \cdot 10^{4}$ posterior samples per dataset and model. In the end, average $R_{\max }$ (across datasets and models) was $\sim 1.002$ (range: $[1.000-1.035]$ ), suggesting good convergence. Average $n_{\text {effmin }}$ was $\sim 8881$ (range: $[483-15059]$ ), suggesting that we had obtained a reasonable approximation of the posteriors.

### 4.4 Pareto smoothed importance sampling diagnostics

As our main metric of model comparison we computed the Bayesian leave-one-out cross-validation score (LOO) via Pareto-smoothed importance sampling (PSIS; [36,37]); see Methods in the main text.

For a given trial $1 \leq i \leq N_{\text {trials }}$, with $N_{\text {trials }}$ the total number of trials, the PSIS approximation may fail if the leave-one-out posterior differs too much from the full posterior. As a natural diagnostic, PSIS also returns for each trial the exponent $k_{i}$ of the fitted Pareto distribution. If $k_{i}>0.5$ the variance of the raw importance ratios distribution does not exist, and for $k_{i}>1$ also the mean does not exist. In

[^0]
[^0]: ${ }^{4}$ URL: https://github.com/lacerbi/eissample.
${ }^{5}$ URL: https://github.com/lacerbi/vbgmm.
${ }^{6}$ URL: http://becs.aalto.fi/en/research/bayes/mcmcdiag/.

---

#### Page 14

the latter case, the variance of the PSIS estimate is still finite but may be large. In practice, Vehtari et al. suggest to double-check trials with $k_{i}>0.7$ [37].

Across all our models and datasets, we found 2382 trials out of 1137100 with $k_{i}>0.7(0.21 \%)$. We examined the problematic trials, finding that the issue was in almost all cases the discontinuity of the observer's decision rule. For all problematic trials the $\mathrm{LOO}_{i}$ scores were compatible with the values found for non-problematic trials, suggesting that the variance of the PSIS estimate was still within an acceptable range. We verified on a subset of subjects that the introduction a softmax with small spatial constant on the decision rule would remove the discontinuity and the problems with Pareto fitting, without significantly affecting the $\mathrm{LOO}_{i}$ itself.

# 4.5 Visualization of model fits

Let $\mathcal{O}(\mathcal{D})$ be a summary statistic of interest, that is an arbitrary function of a dataset $\mathcal{D}$ (e.g., the vestibular bias for a given bin of $s_{\text {vis }}$ and visual reliability level, as per Fig 4 in the main paper). For a given model, we generated the posterior predictive distribution of the group mean of $\mathcal{O}$ by following this bootstrap procedure:

- For $m=1, \ldots, M=100$ iterations:
- Generate a synthetic group of $n=11$ subjects by taking $n$ samples from the individual posterior distributions of the model parameters.
- For each synthetic subject, generate a dataset $\mathcal{D}_{i}$ of simulated responses to the same trials experienced by the subject.
- Compute the group mean of the summary statistic across synthetic subjects, $o_{m}=\frac{1}{n} \sum_{i=1}^{n} \mathcal{O}\left(\mathcal{D}_{i}\right)$.
- Compute mean and standard deviation of $o_{m}$, which correspond to group mean and SEM of the summary statistic.

The shaded areas shown in the model fits figures in the main text are the posterior predictive distributions (mean $\pm$ SEM) of the summary statistics of interest.

### 4.6 Model validation and recovery

We performed sanity checks and unit tests to verify the integrity of our code.
To test the implementation of our models, for a given observer (given model and parameter vector $\boldsymbol{\theta}$ ) we tested the data simulation code (functions that simulate responses; used e.g. to generate figures) against the log likelihood code (functions that compute the log likelihood of the data). For a number of subjects and models we verified that, at the maximum-likelihood solution, the log likelihood of the data approximated via simulation (by computing the probability of the responses via simple Monte Carlo) was $\sim$ equal to the log likelihood of the data computed numerically. This ensured that our simulation code matched the log likelihood code, being a sanity check for both.

We performed a model recovery analysis to validate the correctness of our analysis pipeline, and assess our ability to distinguish models of interest using all tasks ('joint fits'); see e.g. [13, 38]. For computational tractability, we restricted our analysis to six observer models: the most likely four models for each different causal inference strategy (to verify our ability to distinguish between strategies), and, for the most likely model, its variants along the prior and noise factors (to verify whether we can distinguish models along those axes). Thus, we consider the following models: Fix-X-E, Bay-X-E, Bay/FFu-X-I, Fix/FFu-C-I, Fix-X-I, Fix-C-E (see main text for a description). We generated synthetic datasets from each of these six models, for all three tasks jointly, using the same sets of stimuli that were originally displayed to the 11 subjects. For each subject, we took four randomly chosen posterior parameter vectors obtained via MCMC sampling (as described in Section 4.3), so as to ensure that the statistics of the

---

#### Page 15

simulated responses were similar to those of the subjects. Following this procedure, we generated 264 datasets in total ( 6 generating models $\times 11$ subjects $\times 4$ posterior samples). We then fit all 6 models to each synthetic dataset, yielding 1584 fitting problems. For computational tractability, we only performed maximum likelihood estimation (see Section 4.2, with 50 restarts), as opposed to MCMC sampling, whose cost would be prohibitive for this number of fits. The analysis was otherwise exactly the same as that used for fitting the subject data. We then computed the fraction of times that a model was the 'best fitting' model for a given generating model, according to AICc (considering that AICc approximates LOO in the limit of large data).

# Fraction recovered

> **Image description.** This image is a heatmap representing a model recovery analysis.
>
> The heatmap is a square grid, with rows representing "Generating models" and columns representing "Fitted models". The y-axis label "Generating models" is rotated 90 degrees counter-clockwise. The x-axis label "Fitted models" is rotated slightly clockwise.
>
> The rows are labeled with the following model names:
>
> - Fix-X-E
> - Bay-X-E
> - Bay/FFu-X-I
> - Fix/FFu-C-I
> - Fix-X-I
> - Fix-C-E
>
> The columns are labeled with the same model names, in the same order.
>
> Each cell in the grid represents the fraction of datasets for which a particular fitted model was the best fit for a given generating model, according to the AICc score. The cells are shaded in grayscale, with brighter shades of gray corresponding to larger fractions.
>
> The values in each cell are explicitly written in white or gray text. The diagonal elements of the matrix are generally brighter than the off-diagonal elements. The values on the diagonal are 0.77, 1.00, 1.00, 0.75, 1.00, and 0.84. Other notable values include 0.02, 0.09, 0.14, 0.07, 0.09 and 0.23.

Fitted models
Model recovery analysis. Each square represents the fraction of datasets that were 'best' fitted from a model (columns), for a given generating model (rows), according to the AICc score. Bright shades of gray correspond to larger fractions. The bright diagonal indicates that the true generating model was, on average, the best-fitting model in all cases, leading to a successful model recovery.

We found that the true generating model was recovered correctly in $89.4 \%$ of the datasets on average (see above). This finding means that our models are distinguishable in a realistic setting, and at the same time validates the model fitting pipeline (as it would be unlikely to obtain a successful recovery in the presence of a substantial coding error). Since our model recovery method differs from the procedure used on subject data in the comparison metric (AICc via maximum-likelihood estimation, rather than LOO via MCMC), we verified on subject data that AICc and LOO scores were highly correlated across subjects [39]. The Spearman's rank correlation coefficient between the two metrics was larger than 0.99 for each of the sixteen models in the joint fits, providing strong evidence that results of our model recovery analysis would also transfer to the framework used for the subject data.

---

#### Page 16

# 5 Absolute goodness of fit

In this section we describe a general method to compute absolute goodness of fit, largely based on the approach of $[40]^{7}$

### 5.1 Computing the absolute goodness of fit

Let $\boldsymbol{X}$ be a dataset of discrete categorical data grouped in $M$ independent batches with $K$ classes each, such that $X_{j k}$ is the number of observations for the $j$-th batch and the $k$-th class. We define $N_{j}=\sum_{k} X_{j k}$ the number of observations for the $j$-th batch.

We assume that observations are ordered and independent, such that the distribution of observations in each batch $j$ is the product of $N_{j}$ categorical distributions with parameters $\boldsymbol{p}_{j}=\left(p_{j 1}, \ldots, p_{j K}\right)$ (frequencies), such that the probability of the data is

$$
p(\boldsymbol{X})=\prod_{j=1}^{M} \prod_{k=1}^{K} p_{j k}^{X_{j k}}
$$

with unknown vectors of frequencies $\boldsymbol{p}_{j}$.
We assume that we have a model of interest $q$ that predicts frequencies $q_{j k}$ for the observations, with $\sum_{k} q_{j k}=1$ for $1 \leq j \leq M$. As a reference, we consider the chance model $q^{0}$ with frequencies $q_{j k}^{0}=1 / K$.

We define the absolute goodness of fit of $q$ as

$$
g(q)=1-\frac{\operatorname{KL}(p \| q)}{\operatorname{KL}(p \mid q^{0})}
$$

where $\operatorname{KL}(p \| q)$ is the Kullback-Leibler divergence (also known as relative entropy) between a 'true' distribution $p$ and an 'approximating' distribution $q$.

Importantly, $g(q)=0$ when a model performs at chance, and $g(q) \leq 1$, with $g(q)=1$ only when the model matches the true distribution of the data. In other words, $g(q)$ represents the fractional information gain over chance. Note that $g(q)$ can be negative, in the unfortunate case that a model performs worse than chance.

As another important reference, we recommend to also compute the absolute goodness of fit $g(\bar{q})$ of the histogram model $\bar{q}$, with frequencies defined from the empirical frequencies across batches as $\bar{q}_{j k}=\sum_{l=1}^{M} X_{l k} / N$, for $1 \leq j \leq M$ and $N=\sum_{j} N_{j}$. A comparison between $g(q)$ and $g(\bar{q})$ is informative of how better the current model is than a simple histogram of categorical observations collapsed across batches. In some circumstances, the chance model can be a straw model, whereas the histogram model may represent a more sensible reference point.

In order to estimate Eq. S15, we need to compute the relative entropy $\operatorname{KL}(p \| q)$ between the data and a given distribution $q$,

$$
\begin{aligned}
\mathrm{KL}(p \| q) & =\mathbb{E}_{p}[\log p]-\mathbb{E}_{p}[\log q] \\
& =-H(p)+H(p, q)
\end{aligned}
$$

where the first term is the (negative) entropy of the data, and the second term is called the cross-entropy between $p$ and $q$. We will show in the following sections that the negative cross-entropy is approximated by the cross-validated log likelihood of the data, $\mathrm{LL}_{\mathrm{CV}}(q)$.

Combining Eq. S15 with our estimates of Eq. S16, we obtain

$$
g(q) \equiv 1-\frac{H(p)+\mathrm{LL}_{\mathrm{CV}}(q)}{H(p)+\mathrm{LL}_{\text {chance }}(q)}
$$

[^0]
[^0]: ${ }^{7}$ URL: https://github.com/lacerbi/gefit.

---

#### Page 17

We show next how to estimate the entropy of the data, and prove that the negative cross-entropy between $p$ and $q$ is approximated by the cross-validated log likelihood.

# 5.2 Entropy of the data

As noted in [40], the naïve plug-in estimator of the entropy of the data leads to a biased estimate of the entropy, and this bias can be substantial when the data are sparse (a few observations per batch). Instead, we use the Grassberger estimator of the entropy [41],

$$
H(p)=\sum_{j=1}^{M} H\left(\boldsymbol{p}_{j}\right) \approx \sum_{j=1}^{M} N_{j} \hat{H}_{G}\left(\boldsymbol{X}_{j}\right)
$$

where the Grassberger estimator of the entropy per trial is defined as

$$
\hat{H}_{G}\left(\boldsymbol{X}_{j}\right)=G\left(N_{j}\right)-\frac{1}{N_{j}} \sum_{k=1}^{K} X_{j k} G\left(X_{j k}\right)
$$

and $G(h)$ for $h \in \mathbb{N}$ are Grassberger's numbers defined as

$$
G(0)=0, \quad G(h)=\psi(h)+\frac{1}{2}(-1)^{h}\left[\psi\left(\frac{h+1}{2}\right)-\psi\left(\frac{h}{2}\right)\right] \quad \text { for } h>0
$$

where $\psi$ is the digamma function.
That is, our estimate of the negative entropy is

$$
-H(p) \approx-\sum_{j=1}^{M} N_{j}\left[G\left(N_{j}\right)-\frac{1}{N_{j}} \sum_{k=1}^{K} X_{j k} G\left(X_{j k}\right)\right]
$$

which is the same as Eq. 21 in [40], when restricted to the binomial case $(K=2)$, and after correcting for a typo ( $N$ in the denominator of their equation should read as $N_{j}$ ).

### 5.3 Cross-entropy

The estimated cross-entropy is

$$
\widehat{H}(p, q)=-\mathbb{E}_{p}[\log q]=-\sum_{j=1}^{M} N_{j} \mathbb{E}_{\boldsymbol{p}_{j}}\left[\log \boldsymbol{q}_{j}\right]=-\sum_{j=1}^{M} N_{j} \mathbb{E}_{\widehat{\boldsymbol{p}}_{j}}\left[\log \boldsymbol{q}_{j}\right]
$$

where in a slight abuse of notation we denoted with $\boldsymbol{p}_{j}$ (resp., $\boldsymbol{q}_{j}$ ) the categorical distributions associated to the data (resp., model) for the $j$-th batch. Crucially, since the expectations only involve $q, \widehat{p}_{j k} \equiv X_{j k} / N_{j}$ is an unbiased estimator of $p_{j k}$.

---

#### Page 18

Eq. S22 becomes

$$
\begin{aligned}
-\sum_{j=1}^{M} N_{j} \mathbb{E}_{\widehat{\boldsymbol{p}}_{j}}\left[\log \boldsymbol{q}_{j}\right] & =-\sum_{j=1}^{M} N_{j} \mathbb{E}_{\widehat{\boldsymbol{p}}_{j}}\left[\log q_{j 1}^{x_{1}} \cdots q_{j K}^{x_{K}}\right] \\
& =-\sum_{j=1}^{M} N_{j} \sum_{k=1}^{K} \mathbb{E}_{\widehat{\boldsymbol{p}}_{j}}\left[x_{k}\right] \log q_{j k} \\
& =-\sum_{j=1}^{M} \sum_{k=1}^{K} N_{j} \widehat{p}_{j k} \log q_{j k} \\
& =-\sum_{j=1}^{M} \sum_{k=1}^{K} X_{j k} \log q_{j k}
\end{aligned}
$$

which is the negative log likelihood of the model, $-\mathrm{LL}(q)$.
Note that typically we also need to estimate the model parameters, and computing Eq. S23 on the same dataset used to estimate parameters will yield a biased estimate of the log likelihood (see e.g., [42]). Shen and Ma suggest to obtain an independent estimate of the log likelihood of the model via crossvalidation, $\mathrm{LL}_{\mathrm{CV}}$ [40]. According to their method, model parameters are estimated on half of the data, and the log likelihood of the model (and also the entropy of the data) is evaluated with the other half of the data. As an improvement over their method, we advocate to estimate the expected log likelihood via leave-one-out (LOO) cross-validation score obtained via MCMC [37]. This will produce an unbiased estimator of the expected log likelihood, and allows to use all the available data to obtain a more robust estimate of the relative entropy.

In conclusion, our estimate for the cross-entropy is

$$
\widehat{H}(p, q)=-\mathrm{LL}_{\mathrm{CV}}(q)
$$

with $\mathrm{LL}_{\mathrm{CV}}(q)$ computed as the LOO score of the model, and it corresponds to Eq. 19 in [40].

# 6 LOO scores for all models

In this section we report tables of LOO scores for all models and subjects, which were used to perform group Bayesian Model Selection, the model comparison technique adopted in the main text. For each subject, LOO scores are shown relative to the LOO of the model with highest mean LOO across subject, which is printed in boldface. Models are ranked according to average LOO.

Summing (equivalently, averaging) LOO scores across subjects is a simple 'fixed-effect' model comparison analysis, in which all subjects are believed to belong to the same model. Results of the fixed-effect analysis differ in details from the group Bayesian Model Selection, but the overall qualitative findings are analogous.

### 6.1 Unity judgment task

|  Model  |    S1    |    S2    |    S3    |    S4    |    S5    |    S6    |    S7    |    S8    |    S9    |   S10    |   S11    |   Mean $\pm$ SE   |
| :-----: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :---------------: |
| Bay-X-I |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   $0.0 \pm 0.0$   |
| Bay-X-E | $-22.1$  |   4.5    | $-12.7$  | $-29.7$  |   22.2   | $-24.7$  |  $-1.8$  |  $-2.6$  |   1.7    |   35.9   |  $-0.4$  |  $-2.7 \pm 5.9$   |
|   Fix   | $-31.6$  |   12.5   | $-12.9$  |   0.7    | $-12.4$  | $-18.8$  |   1.8    |   12.3   |  $-2.8$  |   10.2   |  $-4.8$  |  $-4.2 \pm 4.2$   |
| Bay-C-I |  $-0.3$  |   4.6    |   0.4    | $-11.7$  | $-11.9$  |   2.2    |  $-0.4$  |  $-0.8$  |  $-2.8$  | $-25.6$  |  $-1.8$  |  $-4.4 \pm 2.6$   |
|  Fix-C  | $-30.6$  |   13.2   | $-10.5$  |   2.3    | $-21.1$  | $-18.0$  |   1.1    |   14.4   |  $-2.6$  | $-29.0$  |  $-7.6$  |  $-8.0 \pm 4.7$   |
| Bay-C-E | $-26.4$  | $-18.7$  | $-14.2$  | $-29.8$  |   16.0   | $-41.9$  |  $-1.6$  | $-17.0$  |  $-1.9$  |   12.5   |  $-2.9$  |  $-11.4 \pm 5.4$  |
|   SFu   | $-272.4$ | $-119.9$ | $-245.8$ | $-122.5$ | $-112.1$ | $-154.5$ | $-272.4$ | $-120.9$ | $-250.2$ | $-122.0$ | $-117.5$ | $-173.7 \pm 21.1$ |

---

#### Page 19

# 6.2 Bimodal inertial discrimination task

|  Model  |   S1    |   S2    |   S3    |   S4    |   S5    |   S6    |   S7    |   S8   |   S9   |   S10   |  S11   |  Mean $\pm$ SE  |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :----: | :----: | :-----: | :----: | :-------------: |
| Bay-X-E |   0.0   |   0.0   |   0.0   |   0.0   |   0.0   |   0.0   |   0.0   |  0.0   |  0.0   |   0.0   |  0.0   |  $0.0 \pm 0.0$  |
| Fix-X-E | $-0.9$  |   0.5   | $-2.2$  | $-13.0$ | $-0.8$  |   0.8   | $-0.6$  |  0.4   |  0.3   |   3.4   |  1.5   | $-1.0 \pm 1.3$  |
| FFu-X-I | $-0.7$  |   0.9   | $-3.5$  | $-11.3$ |   0.4   |   1.3   |   1.2   |  1.6   |  1.5   | $-12.3$ |  1.1   | $-1.8 \pm 1.6$  |
| Fix-X-I | $-0.9$  |   2.0   | $-3.2$  | $-11.5$ |   0.6   |   1.3   |   0.6   |  0.0   |  0.7   | $-12.5$ |  1.1   | $-2.0 \pm 1.6$  |
| FFu-X-E | $-0.2$  |   0.9   | $-3.6$  | $-10.2$ |   0.6   |   1.5   |   0.9   |  1.5   |  1.4   | $-18.8$ |  1.2   | $-2.3 \pm 2.0$  |
| Fix-C-E | $-9.8$  | $-3.7$  |   0.1   | $-18.7$ | $-0.9$  | $-2.5$  | $-7.1$  |  1.1   | $-2.3$ |   3.9   |  0.4   | $-3.6 \pm 1.9$  |
| Bay-C-E | $-10.5$ |   0.3   | $-0.6$  | $-5.7$  |   0.6   | $-1.9$  | $-11.8$ |  0.1   | $-1.8$ | $-5.4$  | $-3.2$ | $-3.6 \pm 1.3$  |
| Bay-X-I | $-3.1$  | $-2.6$  | $-5.7$  | $-13.0$ | $-1.6$  | $-1.5$  |   0.2   | $-0.8$ |  0.2   | $-15.5$ |  1.4   | $-3.8 \pm 1.7$  |
| FFu-C-E | $-20.1$ | $-22.1$ | $-9.9$  | $-34.7$ | $-14.8$ | $-21.9$ | $-31.9$ | $-6.0$ | $-2.4$ | $-57.7$ | $-2.1$ | $-20.3 \pm 5.0$ |
| FFu-C-I | $-20.2$ | $-22.1$ | $-9.9$  | $-34.8$ | $-14.8$ | $-21.8$ | $-31.9$ | $-6.0$ | $-2.6$ | $-57.7$ | $-2.2$ | $-20.3 \pm 5.0$ |
| Fix-C-I | $-20.2$ | $-22.1$ | $-9.9$  | $-34.8$ | $-14.8$ | $-21.9$ | $-30.6$ | $-6.8$ | $-3.1$ | $-57.8$ | $-2.3$ | $-20.4 \pm 4.9$ |
| Bay-C-I | $-19.6$ | $-21.6$ | $-10.4$ | $-34.7$ | $-15.9$ | $-22.7$ | $-32.3$ | $-6.2$ | $-2.8$ | $-58.2$ | $-2.8$ | $-20.6 \pm 5.0$ |

### 6.3 Joint fits

|    Model    |    S1    |    S2    |    S3    |    S4    |    S5    |    S6    |    S7    |    S8    |    S9    |   S10    |   S11   |   Mean $\pm$ SE   |
| :---------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :-----: | :---------------: |
|   Fix-X-E   |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0    |   0.0   |   $0.0 \pm 0.0$   |
|   Fix-X-I   |   7.3    |  $-7.2$  |  $-4.4$  |   31.9   | $-16.1$  |   23.7   | $-40.6$  | $-26.6$  | $-38.6$  | $-20.4$  |   2.4   |  $-8.0 \pm 7.1$   |
| Fix/FFu-X-E | $-14.8$  | $-19.7$  | $-15.4$  |   4.8    |  $-3.4$  |   0.5    | $-42.6$  |  $-4.3$  | $-14.8$  |  $-8.9$  |   2.4   |  $-10.6 \pm 4.0$  |
|   Fix-C-E   |   0.4    |  $-9.2$  |  $-1.9$  | $-14.3$  | $-41.9$  |  $-7.9$  |  $-3.5$  |   0.9    |  $-6.5$  | $-46.7$  | $-10.7$ |  $-12.8 \pm 4.9$  |
| Fix/FFu-X-I | $-26.6$  | $-19.7$  | $-22.5$  |   4.8    |  $-2.4$  |   0.5    | $-52.4$  |  $-4.2$  | $-59.7$  |  $-2.9$  |   2.3   |  $-16.6 \pm 6.7$  |
|   Bay-X-E   |   17.1   | $-34.2$  |  $-6.2$  | $-31.3$  | $-25.8$  | $-20.6$  |  $-9.5$  | $-128.6$ |   12.7   |   12.3   | $-6.2$  | $-20.0 \pm 12.1$  |
| Bay/FFu-X-E | $-20.8$  | $-39.0$  | $-25.3$  |   0.5    |   10.8   | $-14.9$  | $-42.9$  | $-127.0$ | $-40.8$  |   3.7    |   1.2   | $-26.8 \pm 11.6$  |
| Fix/FFu-C-E | $-14.2$  | $-21.0$  | $-17.9$  | $-20.5$  | $-47.6$  |  $-6.0$  | $-44.5$  |  $-3.3$  | $-19.0$  | $-103.6$ | $-8.5$  |  $-27.8 \pm 8.7$  |
|   Fix-C-I   |  $-3.6$  | $-32.0$  | $-15.8$  | $-14.1$  | $-59.5$  |   2.1    | $-85.5$  | $-25.2$  | $-59.0$  | $-94.9$  | $-9.4$  | $-36.1 \pm 10.1$  |
| Fix/FFu-C-I | $-25.6$  | $-21.1$  | $-22.0$  | $-20.6$  | $-47.7$  |  $-6.1$  | $-86.0$  |  $-3.4$  | $-59.1$  | $-103.7$ | $-8.6$  | $-36.7 \pm 10.1$  |
|   Bay-C-E   |   2.7    | $-73.1$  | $-29.5$  | $-44.2$  | $-33.6$  | $-74.9$  | $-16.1$  | $-191.4$ |  $-6.7$  | $-12.3$  | $-26.4$ | $-45.9 \pm 16.4$  |
| Bay/FFu-C-E | $-36.4$  | $-77.3$  | $-47.1$  | $-31.0$  | $-15.4$  | $-45.3$  | $-90.7$  | $-206.7$ | $-72.9$  | $-74.3$  | $-11.8$ | $-64.4 \pm 16.2$  |
|   Bay-X-I   | $-356.3$ | $-128.2$ | $-193.6$ | $-204.0$ | $-91.3$  | $-35.6$  | $-177.3$ | $-235.6$ | $-208.7$ | $-105.2$ | $-6.3$  | $-166.6 \pm 32.2$ |
|   Bay-C-I   | $-462.0$ | $-222.2$ | $-318.1$ | $-231.3$ | $-158.8$ | $-77.6$  | $-319.6$ | $-338.0$ | $-488.1$ | $-259.2$ | $-51.2$ | $-266.0 \pm 42.1$ |
| Bay/FFu-X-I | $-872.8$ | $-416.4$ | $-544.9$ | $-589.5$ | $-304.5$ | $-424.8$ | $-555.4$ | $-397.0$ | $-593.0$ | $-272.0$ | $-53.7$ | $-456.7 \pm 64.2$ |
| Bay/FFu-C-I | $-888.7$ | $-445.3$ | $-556.3$ | $-611.2$ | $-340.1$ | $-441.8$ | $-551.3$ | $-396.2$ | $-625.7$ | $-351.2$ | $-69.8$ | $-479.8 \pm 62.6$ |

## Supplemental References

1. Wozny DR, Beierholm UR, Shams L. Human trimodal perception follows optimal statistical inference. Journal of vision. 2008;8(3):1-24.
2. Wallace MT, Roberson G, Hairston WD, Stein BE, Vaughan JW, Schirillo JA. Unifying multisensory signals across time and space. Experimental Brain Research. 2004;158(2):252-258.
3. Rohe T, Noppeney U. Sensory reliability shapes perceptual inference via two mechanisms. Journal of Vision. 2015;15(5):1-22.
4. Gu Y, Fetsch CR, Adeyemo B, DeAngelis GC, Angelaki DE. Decoding of MSTd population activity accounts for variations in the precision of heading perception. Neuron. 2010;66(4):596-609.
5. Cuturi LF, MacNeilage PR. Systematic biases in human heading estimation. PLoS ONE. 2013;8(2):v56862.
6. Stocker AA, Simoncelli EP. Noise characteristics and prior expectations in human visual speed perception. Nature Neuroscience. 2006;9(4):578-585.

---

#### Page 20

7. Girshick AR, Landy MS, Simoncelli EP. Cardinal rules: Visual orientation perception reflects knowledge of environmental statistics. Nature Neuroscience. 2011;14(7):926-932.
8. Ganguli D, Simoncelli EP. Efficient sensory encoding and Bayesian inference with heterogeneous neural populations. Neural computation. 2014;26(10):2103-2134.
9. Wei XX, Stocker AA. A Bayesian observer model constrained by efficient coding can explain 'anti-Bayesian' percepts. Nature neuroscience. 2015;18(10):1509.
10. Ernst MO, Banks MS. Humans integrate visual and haptic information in a statistically optimal fashion. Nature. 2002;415(6870):429-433.
11. Alais D, Burr D. The ventriloquist effect results from near-optimal bimodal integration. Current Biology. 2004;14(3):257-262.
12. Fetsch CR, Turner AH, DeAngelis GC, Angelaki DE. Dynamic reweighting of visual and vestibular cues during self-motion perception. The Journal of Neuroscience. 2009;29(49):15601-15612.
13. Acerbi L, Ma WJ, Vijayakumar S. A Framework for Testing Identifiability of Bayesian Models of Perception. In: Advances in Neural Information Processing Systems 27. Curran Associates, Inc.; 2014. p. $1026-1034$.
14. Petzschner FH, Glasauer S. Iterative Bayesian estimation as an explanation for range and regression effects: A study on human path integration. The Journal of Neuroscience. 2011;31(47):17220-17229.
15. Odegaard B, Shams L. The Brain's Tendency to Bind Audiovisual Signals Is Stable but Not General. Psychological Science. 2016;27(4):583-591. doi:10.1177/0956797616628860.
16. Qamar AT, Cotton RJ, George RG, Beck JM, Prezhdo E, Laudano A, et al. Trial-to-trial, uncertainty-based adjustment of decision boundaries in visual categorization. Proceedings of the National Academy of Sciences. 2013;110(50):20332-20337.
17. Ma WJ. Organizing probabilistic models of perception. Trends in Cognitive Sciences. $2012 ; 16(10): 511-518$.
18. Rohe T, Noppeney U. Cortical hierarchies perform Bayesian causal inference in multisensory perception. PLoS Biol. 2015;13(2):e1002073.
19. Wichmann FA, Hill NJ. The psychometric function: I. Fitting, sampling, and goodness of fit. Percept Psychophys. 2001;63(8):1293-1313.
20. de Winkel KN, Katliar M, Diers D, Buelthoff HH. What's Up: an assessment of Causal Inference in the Perception of Verticality. bioRxiv. 2017; p. 189985.
21. Stevens SS. On the psychophysical law. Psychological review. 1957;64(3):153.
22. Crane BT. Direction specific biases in human visual and vestibular heading perception. PLoS ONE. 2012;7(12):e51383.
23. de Winkel KN, Katliar M, Bülthoff HH. Forced fusion in multisensory heading estimation. PLoS ONE. 2015;10(5):e0127104.
24. de Winkel KN, Katliar M, Bülthoff HH. Causal Inference in Multisensory Heading Estimation. PLoS ONE. 2017;12(1):e0169676.

---

#### Page 21

25. Körding KP, Beierholm U, Ma WJ, Quartz S, Tenenbaum JB, Shams L. Causal inference in multisensory perception. PLoS ONE. 2007;2(9):e943.
26. Press WH, Flannery BP, Teukolsky SA, Vetterling WT. Numerical recipes 3rd edition: The art of scientific computing. Cambridge University Press; 2007.
27. Acerbi L, Ma WJ. Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. In: Advances in Neural Information Processing Systems 30; 2017. p. 1836-1846.
28. Audet C, Dennis Jr JE. Mesh adaptive direct search algorithms for constrained optimization. SIAM Journal on Optimization. 2006;17(1):188-217.
29. Neal RM. Slice sampling. Annals of Statistics. 2003;31(3):705-741.
30. Gilks WR, Roberts GO, George EI. Adaptive direction sampling. The Statistician. 1994;43(1):179189 .
31. Foreman-Mackey D, Hogg DW, Lang D, Goodman J. emcee: The MCMC hammer. Publications of the Astronomical Society of the Pacific. 2013;125(925):306.
32. Bishop CM. Pattern recognition and machine learning. Springer; 2006.
33. Andrieu C, Thoms J. A tutorial on adaptive MCMC. Statistics and Computing. 2008;18(4):343373 .
34. MacKay DJ. Information theory, inference and learning algorithms. Cambridge university press; 2003 .
35. Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB. Bayesian data analysis (3rd edition). CRC Press; 2013.
36. Vehtari A, Gelman A, Gabry J. Pareto smoothed importance sampling. arXiv preprint arXiv:150702646. 2015;.
37. Vehtari A, Gelman A, Gabry J. Practical Bayesian model evaluation using leave-one-out crossvalidation and WAIC. Statistics and Computing. 2016; p. 1-20.
38. van den Berg R, Awh E, Ma WJ. Factorial comparison of working memory models. Psychological Review. 2014;121(1):124-149.
39. Adler WT, Ma WJ. Comparing Bayesian and non-Bayesian accounts of human confidence reports. bioRxiv. 2016;doi:10.1101/093203.
40. Shen S, Ma WJ. A detailed comparison of optimality and simplicity in perceptual decision making. Psychological Review. 2016;123(4):452-480.
41. Grassberger P. Entropy estimates from insufficient samplings. arXiv preprint physics/0307138. 2003;.
42. Burnham KP, Anderson DR. Model selection and multimodel inference: A practical informationtheoretic approach. Springer Science \& Business Media; 2003.