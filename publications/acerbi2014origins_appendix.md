# On the Origins of Suboptimality in Human Probabilistic Inference - Appendix

---

## Supporting Information

Dataset S1 Subject's datasets. Subjects' datasets for the main experiment $(n=24$, training and test sessions) and for the sensorimotor estimation experiment $(n=10)$, with relevant metadata, in a single MATLAB data file.
(ZIP)

Text S1 Additional analyses and observer models. This supporting text includes sections on: Translational invariance of subjects' behavior; Success probability; Inverted Gaussian loss function; Model comparison with DIC; Model comparison for different shared parameters between sessions; Nonstationary analysis.
(PDF)

Text S2 Noisy probabilistic inference. Description of the models of stochastic probabilistic inference ('noisy posterior' and 'sample-based posterior') and discussion about unstructured noise in the prior.
(PDF)

Text S3 Sensorimotor estimation experiment. Methods and results of the additional experiment to estimate the range of subjects' sensorimotor parameters.
(PDF)

---

#### Page 1

# Supporting Text S1 - Additional analyses and observer models

# Contents

1 Translational invariance of subjects' targeting behavior ..... 2
2 Success probability ..... 3
3 Inverted Gaussian loss function ..... 4
3.1 Observer model with variable loss width $\sigma_{\ell}$ ..... 4
4 Model comparison with DIC ..... 5
4.1 Basic model comparison ..... 5
4.2 Comparison of alternative models of decision making ..... 5
4.3 Comparison of distinct model components ..... 8
5 Model comparison for different shared parameters between sessions ..... 9
6 Nonstationary analysis ..... 11
6.1 Iterative non-Bayesian observer model ..... 11

---

#### Page 2

# 1 Translational invariance of subjects' targeting behavior

In this section we show that subject's behavior depends only on the relative position of the cue with respect to the prior. This result allows us to express all positions in a 'prior-centric' coordinate system $\left(\mu_{\text {prior }}=0\right)$ without loss of generality.

In the paper we assumed that all variables (e.g. cue position $x_{\text {cue }}$, subjects' response $r$, target position $x$ ) can be expressed relative to the current location of the prior ( $\mu_{\text {prior }}$ ); a shift of $\mu_{\text {prior }}$ simply produces an equal shift in all other position variables. That is, subjects' behavior is independent of screen coordinates (translational invariance). The alternative hypothesis is that subjects' responses instead show some form of bias that is screen-coordinate dependent, for example a central tendency towards the middle of the screen.

In order to test whether subjects' relative responses depend on the absolute location of the prior, for each subject we fit a linear regression line to the relation between the relative response $\tilde{r}=r-\mu_{\text {prior }}$ and the prior mean $\mu_{\text {prior }}$ across all trials. Given the generative model of our task, we expected the average relative response to be zero irrespective of prior mean, $\langle\tilde{r}\rangle=0$ and therefore tested whether the slope or intercept are different than zero.

For almost all subjects, the slope and intercept were not significantly different than zero $(p>0.05)$. For two subjects we found that slope or intercept may be significantly different from zero ( $p=0.002$ and $p=0.04$ ). However, even in these cases a correction for multiple comparisons $(n=24)$ suggests that the these differences are not statistically significant or at most marginally so. This analysis confirms that subjects' responses in general do not show statistically significant departures from the assumption of translational invariance.

---

#### Page 3

# 2 Success probability

Figure 1 shows the success probability (see Methods in the paper) averaged across subjects, divided by sessions.

> **Image description.** The image shows a set of four bar charts arranged in a 2x2 grid. Each chart displays the "Success probability" on the y-axis, ranging from 0 to 1, and "Prior distribution" on the x-axis, labeled with numbers 1 through 8. Each chart is further divided into two sections, representing "Low-noise cue" (shown in shades of red) and "High-noise cue" (shown in shades of blue).
>
> Here's a breakdown of each chart:
>
> - **Top Left:** Titled "Gaussian training (n = 24)". It shows eight red bars (low-noise cue) gradually decreasing in height from left to right, followed by eight blue bars (high-noise cue), also decreasing in height. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> - **Top Right:** Titled "Gaussian test (n = 8)". The structure is similar to the top-left chart, with eight red bars (low-noise cue) and eight blue bars (high-noise cue), both sets decreasing in height. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> - **Bottom Left:** Titled "Unimodal test (n = 8)". Similar structure with eight red bars (low-noise cue) and eight blue bars (high-noise cue). The bars are more uniform in height compared to the top charts. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> - **Bottom Right:** Titled "Bimodal test (n = 8)". Similar structure with eight red bars (low-noise cue) and eight blue bars (high-noise cue). The bars are more uniform in height compared to the top charts. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> A legend at the top of the image reads: "Best stochastic model (BDT-P-L)" (represented by a solid dark line) and "Maximal success probability" (represented by a dashed line).

Figure 1. Group mean success probability for all sessions. Each bar represents the groupaveraged success probability for a specific session, for each prior (indexed from 1 to 8 , see also Figure 2 in the paper) and cue type, low-noise cues (red bars) or high-noise cues (blue bars). Error bars are SE across subjects. Priors are arranged in the order of differential entropy (i.e. increasing variance for Gaussian priors), except for 'unimodal test' priors which are listed in order of increasing width of the main peak in the prior (see main paper). The dashed line represents the maximal success probability for an ideal observer. The continuous line represents the 'postdiction' of the best Bayesian model, BDT-P-L (see 'Analysis of best observer model' in the paper). Also, compare this figure with Figure 5 in the paper, which shows the optimality index.

---

#### Page 4

# 3 Inverted Gaussian loss function

In this section we show that the inverted Gaussian loss function described by Eq. 16 in the paper is a very good approximation of the true loss model of the task, the square well loss (Eq. 15 in the paper), meaning that in our analysis we can adopt the inverted Gaussian without loss of generality.

In order to compare the Gaussian loss with the square well loss, we first compute the theoretical distribution of observed cues, given each combination of prior and cue (low-noise and high-noise). The distribution of cues is a convolution between the prior and the cue variability, $p\left(x_{\text {cue }} \mid p_{\text {prior }}, d_{\text {cue }}\right)=$ $\int p_{\text {prior }}(x) \mathcal{N}\left(x_{\text {cue }} \mid x, \sigma_{x}^{2}\left(d_{\text {cue }}\right)\right) d x$. For each combination we calculate the RMSE between the 'optimal target' predicted by the two loss functions for a certain cue position, weighted by cue probability:

$$
R M S E\left(p_{\text {prior }}, d_{\text {cue }}\right)=\left\{\int_{\mathcal{D}} p\left(x_{\text {cue }} \mid p_{\text {prior }}, d_{\text {cue }}\right)\left[x_{\text {Gauss }}^{*}\left(x_{\text {cue }}\right)-x_{\text {well }}^{*}\left(x_{\text {cue }}\right)\right]^{2} d x_{\text {cue }}\right\}^{\frac{1}{2}}
$$

where $\mathcal{D}$ is the range of the discrete representation of $p_{\text {prior }}(x)$. We exclude from the analysis singleGaussian priors, as in that case the predicted optimal target is identical for both loss models. We repeat the calculation for a range of values of the scale of the inverted Gaussian, $\sigma_{\ell}$, while we keep the window size of the square well loss fixed to the 'true' value ( $\ell^{*}=0.083$ screen units, the cursor diameter).

This procedure allow us to find the value of $\sigma_{\ell}$ for which the inverted Gaussian loss best approximates the true loss function of the task in terms of observable behavior, by minimizing the average RMSE across all our experimental conditions. We find an optimal value of $\sigma_{\ell}^{*} \approx 0.027$ screen units, close to the SD of a uniform distribution of range $\ell^{*}$, which is 0.024 screen units (the square well loss can be thought of as an 'inverted uniform distribution'). For $\sigma_{\ell}^{*}$, the total RMSE is $1.2 \cdot 10^{-4} \pm 1.5 \cdot 10^{-4}$ screen units (mean $\pm$ SD across different conditions), which is on average less than a tenth of a mm. In terms of performance, the optimality index of an ideal Bayesian observer that uses the inverted Gaussian loss in place of the square-well loss is $0.9999 \pm 0.0001$ (mean $\pm$ SE across conditions) which is empirically indistinguishable from 1. This analysis shows that the inverted Gaussian loss approximates the behavior of the square well loss far below empirical error for our set of distributions. Hence we can use the inverted Gaussian loss function for our Bayesian observer models without loss of generality.

The inverted Gaussian loss has several advantages over the square well loss. Primarily for us, it allows us to derive an analytic expression of the expected loss that involves only sums of Gaussian distributions (see Eq. 4 in the paper). In general, the inverted Gaussian loss is also a very flexible model, as the scale parameter $\sigma_{\ell}$ allows to interpolate between two very well-known models of loss, a delta function (for $\sigma_{\ell} \rightarrow 0$, which leads to a MAP solution) and a quadratic loss (for $\sigma_{\ell} \rightarrow \infty$, corresponding to the mean of the posterior). In addition to theoretical appeal, experimentally the inverted Gaussian loss has been proven to account very well for people's behavior in a spatial targeting task [1].

### 3.1 Observer model with variable loss width $\sigma_{\ell}$

In the paper we either fixed $\sigma_{\ell}$ to the value that best approximates the square well loss or we considered models that explictly or implicitly assume a quadratic loss $\left(\sigma_{\ell} \rightarrow \infty\right)$. Here we examine the performance of an extended BDT-P-L model (the best model that follows BDT) in which the loss width $\sigma_{\ell}$ is allowed to vary freely. Since the parameter $\sigma_{\ell}$ is irrelevant for Gaussian posteriors, we perform this analysis only for non-Gaussian posteriors (see Methods in the paper). Given the typical scale of the posteriors in the task, a value of $\sigma_{\ell} \gtrsim 0.2$ screen units should be considered near-quadratic for all practical purposes.

We find that subjects fall in two classes with respect to the posterior distribution of parameter $\sigma_{\ell}$. For the majority of subjects ( 10 out of 16), mostly in the bimodal session, the posterior is peaked around $\sigma_{\ell}=0.11 \pm 0.02$ screen units (mean $\pm$ SE across subjects), which is significantly higher than the 'true' value ( $\sigma_{\ell}^{*}=0.027$ screen units; signed rank test, $p<0.01$ ) but still qualitatively different from a nearquadratic loss. For the other six subjects the posterior is much broader and flat in the range of $\sigma_{\ell}$

---

#### Page 5

from 0.2 to 1 screen units, compatibly with a near-quadratic loss. In fact, according to the comparison between alternative models of decision making, these subjects show some preference for a quadratic loss or, similarly, a low-order approximation of the posterior (see Figure 9a in the paper and Figure 3a here, subjects 10-14 and 18). However, note that most of these subjects belong to the unimodal group, where posteriors are still very close to Gaussians and therefore the exact value of the loss width may not be necessarily meaningful. The reason why we find a relatively large loss width in the case of a BDT observer is that it needs to account for large, posterior-dependent targeting errors that are explained instead by stochasticity in decision making by the SPK observer (in neither case posterior-dependent errors can be adequately explained by constant motor noise $\sigma_{\text {motor }}$ ).

Performance of model BDT-P-L with variable loss is better than its corresponding version with fixed $\sigma_{\ell}(\Delta \mathrm{DIC}=-11.5 \pm 4.0, p<0.05)$, but still slightly worse than a model with variability in decision making with the same number of parameters, SPK-L $(\Delta \mathrm{DIC}=22.5 \pm 8.9, p<0.05)$. In conclusion, allowing a degree of freedom to the loss function at most slightly improves model performance for BDT but does not seem to provide a better explanation for the data than models with variability in decision making.

# 4 Model comparison with DIC

We report in this section the DIC scores of invidual models for all subjects, and results of the group DIC (GDIC) model comparison. DIC scores are used in the paper to approximate the marginal likelihood of each dataset and model within a hierarchical Bayesian model selection (BMS) framework [2]. Here we also use DIC scores to compute the average impact of each model factor.

### 4.1 Basic model comparison

Figure 2a shows the model evidence for each individual model and subject. We calculated model evidence as the difference in DIC between a given model and the subject's best model (lower values are better). A difference of more than 10 in this scale should be considered strong evidence for the model with lower DIC. Individual results show that model SPK-P-L performed better than other models for almost all datasets, with the exception of a minority that favored model SPK-P instead. Unlike our BMS analysis, here we see a considerable similarity of performance between model SPK-P-L and SPK-S-P-L, although the latter performs slightly worse than the former in almost all cases. Figure 2b shows the group average DIC (GDIC), relative to the model with lowest average DIC (lower scores are better). SPK-P-L is confirmed as the best model. Model SPK-S-K-L comes second in terms of average score, but note that the difference with SPK-P-L is very significant (pairwise signed-rank test with Bonferroni correction for multiple comparisons, $p<0.001$ ). This suggests that the extra model factor S is not improving model performance, and therefore that SPK-S-P-L is not a 'good' model, in agreement with the small support it obtained in the BMS analysis (see main paper).

### 4.2 Comparison of alternative models of decision making

We consider first the model evidence for each individual model and subject (Figure 3a). Results differ depending on the session (unimodal or bimodal). In both sessions model SPK-L performs consistently well, closely followed by model SPK. However, in the unimodal session there are quite a few subjects whose behavior is well described by several other models. These results are summarized in Figure 3b, which shows the group DIC relative to the model with lowest average DIC (lower scores are better). Due to the difference between sessions we separately computed the group averages for the unimodal and bimodal group. GDIC analysis in the unimodal session alone fails to find significant differences between

---

#### Page 6

> **Image description.** The image presents a model comparison between individual models, displayed in two panels labeled 'a' and 'b'.
>
> **Panel a:**
>
> - This panel is a heatmap-like representation of DIC scores for different models across subjects and test groups.
> - The y-axis is labeled "Models" and lists various models such as "SPK", "BDT", and "PPM", each potentially with suffixes like "-S", "-P", or "-L". These suffixes are also listed as column headers: "Decision making", "Cue noise", "Prior noise", and "Lapse".
> - The x-axis is labeled "Subject number" and ranges from 1 to 24.
> - The subjects are divided into three groups: "Gaussian group," "Unimodal group," and "Bimodal group," separated by vertical dashed lines.
> - Each cell in the grid is colored according to the model's evidence, represented as the DIC difference (ΔDIC) with the best model for that subject. A color bar to the right indicates the ΔDIC range, from 0 (dark red) to >50 (dark blue).
> - Numbers within some cells indicate the ranking of the most supported models with comparable evidence (ΔDIC less than 10).
> - The grid has a dark blue background, with cells varying in color from dark blue to red, indicating the ΔDIC values.
>
> **Panel b:**
>
> - This panel is a horizontal bar chart showing the group average ΔDIC score for each model, relative to the best model.
> - The y-axis lists the same models as in panel a.
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 600.
> - Each bar represents the mean ΔDIC score for a model, with error bars indicating the standard error (SE).
> - Asterisks above the bars denote significant differences in DIC between a given model and the best model, after correction for multiple comparisons: (\*) p<0.05, (\*\*\*) p<0.001.
> - The bars are primarily blue, with the bar for "SPK-P-L" being red.

Figure 2. Model comparison between individual models (DIC scores). a: Each column represents a subject, divided by test group (all datasets include a Gaussian training session), each row an observer model identified by a model string (see Table 2 in the paper). Cell color indicates model's evidence, here displayed as the DIC difference ( $\Delta \mathrm{DIC}$ ) with the best model for that subject (a higher value means a worse performance of a model with respect to the best model). Models are sorted by their group average DIC score (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence ( $\Delta$ DIC less than 10). b: Group average $\Delta$ DIC score, relative to the best model (mean $\pm$ SE). Higher scores indicate worse performance. Asterisks denote significant difference in DIC between a given model and the best model, after correction for multiple comparisons: (\*) $p<0.05$, (\*\*\*) $p<0.001$.

SPK-L and several other observer models. Conversely, GDIC shows significant results in the bimodal session, finding that all models but SPK perform worse than SPK-L.

These results agree with the BMS analysis in the paper in indicating SPK-L as the best model, but otherwise present quite a different pattern. Discrepancies between the two model comparison methods emerge for the following reasons. Firstly, as mentioned in the paper, BMS is not affected by outliers and by construction takes into account group heterogeneity, contrarily to DIC. Secondly, posteriors in the unimodal session may still be very close to Gaussian and therefore distinct models share very similar predictions, which DIC scores alone cannot disambiguate. The hierarchical probabilistic structure of BMS, instead, allows information to flow between global model evidence and individual model evidence for each subject (respectively $\alpha$ and $u_{n k}$ in [2]), at each iteration of the model comparison algorithm. This propagation of belief led BMS to discard less likely models in the paper.

---

#### Page 7

> **Image description.** The image presents a comparison of decision-making models using DIC scores, displayed in two panels labeled 'a' and 'b'.
>
> Panel a: This panel is a heatmap. The y-axis is labeled "Models" and lists various models (SPK, BDT, PSA, PPM) combined with parameters such as 'MV', 'LA', and 'L'. The x-axis is labeled "Subject number" and ranges from 9 to 24. The subjects are divided into two groups: "Unimodal group" (subjects 9-16) and "Bimodal group" (subjects 17-24), separated by a vertical dashed line. Above the subject groups are labels "Decision making", "Gaussian approx.", and "Lapse". Each cell in the heatmap represents a subject and a model. Cell color indicates the model's evidence, displayed as the DIC difference (ΔDIC) with the best model for that subject. A colorbar to the right of the heatmap shows the color scale, ranging from dark blue (0) to red (>50), representing the ΔDIC value. Numbers within the cells (1 to 7) indicate the ranking of models with comparable evidence.
>
> Panel b: This panel is a horizontal bar chart. The y-axis lists the same models as in panel a. The x-axis is labeled "ΔDIC" and ranges from 0 to 500. Each model has two horizontal bars representing the group average ΔDIC for the "Unimodal group" and "Bimodal group". Error bars are present on each bar, representing the standard error. The "Unimodal group" is represented by white diamonds, and the "Bimodal group" is represented by cyan circles. Asterisks above the bars indicate significant differences in DIC between a given model and the best model, with one asterisk (\*) indicating p < 0.05, two asterisks (**) indicating p < 0.01, and three asterisks (\***) indicating p < 0.001. A legend in the top right corner clarifies which color represents each group.

Figure 3. Model comparison between alternative models of decision making (DIC scores). We tested a class of alternative models of decision making which differ with respect to predictions for non-Gaussian trials only. a: Each column represents a subject, divided by group (either unimodal or bimodal test session), each row an observer model identified by a model string (see Table 2 in the paper). Cell color indicates model's evidence, here displayed as the DIC difference ( $\Delta \mathrm{DIC}$ ) with the best model for that subject (a higher value means a worse performance of a model with respect to the best model). Models are sorted by their group average DIC score across both sessions (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence ( $\Delta \mathrm{DIC}$ less than 10). b: Group average $\Delta$ DIC, divided by test group (unimodal or bimodal session), relative to the best model (mean $\pm$ SE). Higher scores indicate worse performance. Asterisks denote significant difference in DIC between a given model and the best model, after correction for multiple comparisons: $\left(^{*}\right) p<0.05,\left({ }^{* *}\right) p<0.01$, $\left({ }^{* * *}\right) p<0.001$.

---

#### Page 8

# 4.3 Comparison of distinct model components

We assess the relevance of each model level within a factor by measuring the average contribution to DIC of each level across all tested observer models, relative to the best level (Figure 4). This is the GDIC counterpart of the BMS computation of the posterior likelihood of each model component (Figures 8c and 9 c in the paper). Results of the GDIC analysis are qualitatively similar to BMS for all factors, with the sole exception of factor S (sensory noise in estimation of the cue position). BMS rejects factor S, whereas from GDIC we can see that, on average, it seems that not having factor S decreases model performance ( $\Delta$ DIC: $33.0 \pm 5.6$, mean $\pm$ SE across subjects). This is not a contradiction: for many simple observer models the addition of any reasonable form of noise, including cue-estimation noise, will improve model performance. However, model factor S becomes redundant when other more fitting forms of noise are present. Since GDIC weights equally all model contributions, model S appears to have a useful influence on model performance due to the average contribution of 'simpler' models. On the contrary, BMS weights evidence differentially and component S appears to be irrelevant for the most likely models (see paper).

> **Image description.** This image contains two horizontal bar charts, labeled "a" and "b".
>
> **Panel a:**
>
> - The title of the chart is "All test trials".
> - The y-axis is labeled "Model Factors" and lists the following factors from top to bottom: SPK, BDT, PPM, S, ¬S, P, ¬P, L, ¬L.
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 500.
> - There are horizontal blue bars corresponding to each model factor, with error bars at the end of each bar.
> - Asterisks are used to denote significance levels. "BDT", "PPM", "¬S", "¬P", and "¬L" have three asterisks each, indicating high significance.
>
> **Panel b:**
>
> - The title of the chart is "Non-Gaussian test trials".
> - The y-axis is labeled "Model Factors" and lists the following factors from top to bottom: SPK, PPM, BDT, ¬GA, MV, LA, SPK, PSA, L, ¬L.
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 200.
> - There are horizontal blue bars corresponding to each model factor, with error bars at the end of each bar.
> - Asterisks are used to denote significance levels. "PPM", "BDT", and "¬L" have three asterisks each. "MV" has one asterisk, and "PSA" has two asterisks.

Figure 4. Influence of different model factors on DIC. Difference in DIC between different levels within factors, relative to the best level (lowest DIC); highest scores denote worse performance. Each group of bars represent a factor, each bar a level within the factor, identified by a model label (see Table 2 in the paper). Error bars are SE across subjects. Asterisks denote significant difference in DIC between a given level and the best level, after correction for multiple comparisons. a: Factors in the basic model comparison. b: Factors in the comparison of alternative models of decision making. Label ' $\neg$ GA' stands for no Gaussian approximation (full posterior).

---

#### Page 9

# 5 Model comparison for different shared parameters between sessions

In the paper we assumed that each subject shared two parameters between the training session and the test session (the motor noise $\sigma_{\text {motor }}$ and the ratio between the cue noise, $\tilde{\sigma}_{\text {high }} / \tilde{\sigma}_{\text {low }}$ ), whereas all the other parameters were specified separately for the two sessions (see 'Model comparison' section in the paper). Here we motivate our modelling choice by showing that it is optimal, at least on a subset of observer models. By 'optimal' we mean that models that share more parameters between sessions perform substantially worse, whereas models that share less parameters (and therefore have more free parameters to specify) do not provide a significant advantage.

For the current analysis we consider a set of variants of observer model SPK (stochastic posterior). We focus on this model since it is the simplest model with the 'best' decision-making component, as found in the paper. These variants differ from the standard SPK model only with respect to the number of parameters shared between training and test sessions. For a single session, model SPK can be characterized by four parameters ( $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa$; see 'Suboptimal Bayesian observer models' section in the paper). Table 1 lists the considered variants, labelled by number of parameters shared across sessions (model SPK\#2 corresponds to the variant adopted in the paper). ${ }^{1}$

| Model  | Total number of parameters |                                 Free parameters $\left(\boldsymbol{\theta}_{M}\right)$                                 |
| :----- | :------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| SPK\#4 |             4              |            $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa$            |
| SPK\#3 |             5              |       $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa \times 2$        |
| SPK\#2 |             6              |       $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }},(\tilde{\sigma}_{\text {high }}, \kappa) \times 2$       |
| SPK\#1 |             7              |       $\sigma_{\text {motor }},(\tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa) \times 2$       |
| SPK\#0 |             8              | $\left(\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa\right) \times 2$ |

Table 1. Observer model SPK with different shared parameters. Table of observer models based on SPK (stochastic posterior) but with different number of shared parameters (model SPK\#2 corresponds to the version in the paper). The number after the '\#' symbol represents the number of parameters the model shares between training and test session. For each model it is also specified the total number of free parameters used to characterize both sessions. A ' $\times 2$ ' means that a parameter is specified independently for training and test sessions; otherwise parameters are shared across sessions. See main text and Methods in the paper for the meaning of the various parameters.

Here we use GDIC instead of BMS since we want to find the modelling choice that works best on average for all subjects. Figure 5 shows the relative DIC scores of the model for different number of shared parameters. Unsurprisingly, the model with lowest group DIC is the model with the highest number of parameters (SPK\#0). However, models SPK\#1 and SPK\#2 closely match the performance of model SPK\#0. In particular, the difference between SPK\#2 and SPK\#0 is nonsignificant ( $\Delta \mathrm{DIC}=3.5 \pm 2.1$; $p=0.55$ ). Conversely, observer models with 3 or more shared parameters perform significantly worse (e.g., for SPK\#3: $\Delta \mathrm{DIC}=32.4 \pm 7.3 ; p<0.001$ ).

These results show that a model that shares the motor noise parameter and the ratio between the estimated cues' SDs between sessions achieves the optimal balance between model fit and simplicity, supporting our choice in the paper.

[^0]
[^0]: ${ }^{1}$ Although there are in total $2^{4}$ variants of model SPK that share different combinations of parameters between sessions, the five models in Table 1 represent the most natural combinations, in order of increasing model complexity.

---

#### Page 10

> **Image description.** This image consists of two panels, labeled 'a' and 'b', presenting a comparison of statistical models.
>
> Panel a: This panel is a heatmap-like representation.
>
> - The y-axis is labeled "Models" and lists five models: "SPK #0", "SPK #1", "SPK #2", "SPK #3", and "SPK #4".
> - The x-axis is labeled "Subject number" and ranges from 1 to 24. The subjects are divided into three groups: "Gaussian group", "Unimodal group", and "Bimodal group", separated by dashed vertical lines.
> - Above the subject groups are the labels "Decision making" and "# Shared parameters", oriented diagonally.
> - The heatmap itself consists of colored cells, each containing a number from 1 to 5. The color of each cell corresponds to a value on the colorbar to the right, ranging from 0 (red) to >50 (dark blue). The colorbar is labeled "ΔDIC".
>
> Panel b: This panel is a horizontal bar graph.
>
> - The y-axis lists the same five models as in panel a: "SPK #0", "SPK #1", "SPK #2", "SPK #3", and "SPK #4".
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 100.
> - Each model has a corresponding horizontal bar, with error bars extending from the end of the bar. The bars are colored to match the colors in the heatmap of panel a.
> - Asterisks are present above the bars for "SPK #3" and "SPK #4", indicating statistical significance.

Figure 5. Comparison of models with different number of shared parameters. Model comparison between observer models based on model SPK but with different number of shared parameters between sessions. a: Each column represents a subject, divided by test group (all datasets include a Gaussian training session), each row an observer model identified by a model string (see Table 1). Cell color indicates model's evidence, here displayed as the DIC difference ( $\Delta \mathrm{DIC}$ ) with the best model for that subject (a higher value means a worse performance of a model with respect to the best model). Models are sorted by their group average DIC score (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence ( $\Delta$ DIC less than 10). b: Group average $\Delta$ DIC score, relative to the best model (mean $\pm$ SE). Higher scores indicate worse performance. Asterisks denote significant difference in DIC between a given model and the best model, after correction for multiple comparisons: $\left({ }^{(* * *}\right) p<0.001$.

---

#### Page 11

# 6 Nonstationary analysis

In our analysis of the data in the paper we have assumed stationarity of participants' behavior: in first approximation, trials are statistically independent and observers' parameters do not change during the course of a session. Stationarity is a common simplifying assumption in the analysis of psychophysical data, although deviations from stationarity can lead to misestimation of the participants' parameters [3]. A typical source of nonstationarity is 'memory', the influence of recent trials on the current response [4]. This is of prime interest to our study, as it could be the case that the variability that we observe in decision making is not random but due to recency effects.

As a simple, model-free test for recency effects, we look at correlations between trial variables at trial $i$ and trial $i+1$. In particular, we define the error at trial $i, \operatorname{Error}(i)$, as the difference between the subjects's response $r$ and the true target position $x$. The shift at trial $i, \operatorname{Shift}(i)$, is the difference between subject's response $r$ and the cue position $x_{\text {cue }}$. In formulas:

$$
\operatorname{Error}(i)=r^{(i)}-x^{(i)}, \quad \operatorname{Shift}(i)=r^{(i)}-x_{\text {cue }}^{(i)}
$$

Note that subjects explicitly knew the error only during the training session, in which they received full performance feedback. During the test trials they only received a qualitative feedback on whether they succeded or missed in the trial.

For each subject we analyze separately training and test sessions, computing the correlations between Error and/or Shift between trial $i$ and trial $i+1$ for each dataset ( $n=24$ training sessions and $n=24$ test sessions, for four possible combinations of variable interaction). Figure 6 shows the trial to trial correlations for individual subjects and their mean. In all cases we find a small but statistically significant anticorrelation between trial variables in the training sessions (t-test, $p<0.05$ ) and no significant correlation in the test sessions. The anticorrelation in the training sessions could easily emerge from a strategy that produces small adjustments in the opposite direction of the experienced error vector. Since the test sessions did not provide full feedback, we do not see any significant effect. These small and null effects suggest that the major variability in the subjects' responses, observed in both training and test sessions, was not due to some trivial trial-to-trial correlation.

### 6.1 Iterative non-Bayesian observer model

Although the correlations seen in Figure 6 are modest, it may be that an iterative (trial-to-trial) model that captures longer-term correlations may fare better at explaining the data. Iterative Bayesian models have been successful at explaining subjects' perfomance in different domains, such as target estimation [5], distance perception [6] and motor adaptation [7]. Simple heuristics may reproduce a behavior that is very close to the Bayesian prediction $[4,8]$. We consider here an iterative, non-Bayesian linear observer model with lapse (IT-L) that implements a simple trial-to-trial heuristic.

In a trial without lapse, the non-Bayesian iterative observer chooses the target $x$ according to a linear mapping $f$ of the current cue position $x_{\text {cue }}$ (in prior-centric coordinates), depending on the current cue type $d_{\text {cue }}$ and prior $p_{\text {prior }}:$

$$
x=f\left(x_{\text {cue }} ; d_{\text {cue }}, p_{\text {prior }}\right)=W\left(d_{\text {cue }}, p_{\text {prior }}\right) \cdot x_{\text {cue }}
$$

where $W$ is a table of linear weights with one entry for each combination of prior and cue type. The table $W$ is updated on a trial by trial basis according to the feedback received each trial (see below for implementation details). To account for mistakes and other sources of variability, we include a probability of lapse, according to model factor L (see paper) . The final response is as usual obtained by adding motor noise with SD $\sigma_{\text {motor }}$. Although conceptually simple, the model has a total of eight free parameters, most of which are involved in the update rule in order to allow for maximum flexibility (see below).

---

#### Page 12

> **Image description.** The image is a scatter plot showing trial-to-trial correlations between different variables related to error and shift in a learning experiment.
>
> The plot has the following key features:
>
> - **Axes:**
>
>   - The vertical axis is labeled "Trial to trial correlation" and ranges from -0.25 to 0.25 with increments of 0.05.
>   - The horizontal axis is labeled "Trial variables".
>
> - **Data Points:** The data is presented as scatter plots. Each plot consists of:
>
>   - Yellow diamonds: Represent "Training sessions (subjects)".
>   - Blue diamonds: Represent "Test sessions (subjects)".
>   - Yellow circles with error bars: Represent "Training sessions (mean)". The error bars appear to be confidence intervals.
>   - Blue circles with error bars: Represent "Test sessions (mean)". The error bars appear to be confidence intervals.
>
> - **Groups of Data:** The data points are grouped into four categories along the horizontal axis, each representing a different correlation:
>
>   - "Error(i) vs Error(i + 1)"
>   - "Error(i) vs Shift(i + 1)"
>   - "Shift(i) vs Error(i + 1)"
>   - "Shift(i) vs Shift(i + 1)"
>
> - **Legend:** A legend is present in the upper right corner explaining the meaning of the different data point markers.
>
> - **Horizontal Line:** A horizontal line is drawn at y = 0.

Figure 6. Trial to trial correlations between Error and Shift. Correlations between trial variables at trial $i$ and trial $i+1$ for four possible combinations of relevant variables Error (difference between response and target position) and Shift (difference between response and cue position). Each data point is an individual session (training sessions in green, test sessions in blue). Mean correlations, averaged across subjects, are plotted as circles. Error bars are $95 \%$ confidence intervals, computed via bootstrap.

Since in our previous analysis only training sessions showed significant trial-to-trial correlations and, moreover, our update rule assumes that full feedback is available to the subjects, we test the model on the training sessions only. We compare the non-Bayesian iterative model against model SPK (stochastic posterior), the simplest Bayesian observer that includes variability in decision making. GDIC analysis shows that model SPK significantly outperforms model IT-L (20 subjects out of 24 ; paired signed-rank test $p<0.01$ ). We, therefore, reject the hypothesis that our data can be explained by this simple iterative non-Bayesian model (see Discussion in the paper).

# Implementation of the model

For model IT-L, we substitute Eq. 11 of the paper with a trial-dependent equation,

$$
p_{\text {target }}^{(l a p s e)}\left(x^{(i)} \mid x_{c u e}^{(i)}, d_{c u e}^{(i)}, p_{p r i o r}^{(i)}\right)=(1-\lambda) \cdot \delta\left[x^{(i)}-f\left(x_{c u e}^{(i)}, d_{c u e}^{(i)}, p_{p r i o r}^{(i)}\right)\right]+\lambda \cdot p_{p r i o r}^{(i)}\left(x^{(i)}\right)
$$

where all variables now show a dependence on the trial number $i$, but otherwise all symbols have a comparable role as in Eq. 11 in the paper. Here, $f\left(x_{c u e}, d_{c u e}, p_{p r i o r}\right)$ is assumed to be a linear mapping from the position of the cue to the chosen target (see Eq. S3), whose linear weights are stored in table $W^{(i)}$, which is updated each trial. The table contains a separate entry for each combination of prior type ( $p_{\text {prior }}$, or equivalently $\sigma_{\text {prior }}$ for Gaussian priors) and cue type ( $d_{\text {cue }}$, either 'short' or 'long'). We assume for simplicity that the table is initialized with two weight values, respectively one for all short, low-noise

---

#### Page 13

cues, $w_{\text {short }}^{(0)}$, and another one for long, high-noise cues, $w_{\text {long }}^{(0)}$, irrespective of prior type.
In a noise-free scenario, in each trial the error term between current weight and 'correct' weight (according to feedback) can be computed as:

$$
\delta^{(i)}=\frac{r^{(i)}}{x_{c u e}^{(i)}}-\frac{x^{(i)}}{x_{c u e}^{(i)}}=\left(r^{(i)}-x^{(i)}\right) \frac{1}{x_{c u e}^{(i)}}
$$

where $x^{(i)}$ is the actual target position (all positions are measured in coordinates relative to the mean of the prior). However, due to noise, Eq. S5 can take arbitrarily large values because of $x_{c u e}^{(i)}$ at the denominator. We therefore apply a regularization factor to the error, so that

$$
\delta^{(i)}=\left(r^{(i)}-x^{(i)}\right) \cdot \frac{\operatorname{sgn}\left(x_{c u e}^{(i)}\right)}{\left|x_{c u e}^{(i)}\right|+\omega}
$$

with $\omega>0$. For the update rule we take a delta-rule [8]:

$$
W^{(i+1)}\left(\sigma_{\text {prior }}, d_{c u e}\right)=W^{(i)}\left(\sigma_{\text {prior }}, d_{c u e}\right)-\eta \cdot \delta^{(i)} \cdot g\left(\sigma_{\text {prior }}, d_{c u e}, \sigma_{\text {prior }}^{(i)}, d_{c u e}^{(i)}\right)
$$

where $\eta>0$ is a learning factor and $g\left(\sigma_{\text {prior }}, d_{\text {cue }}, \sigma_{\text {prior }}^{\prime}, d_{\text {cue }}^{\prime}\right)$ a transfer function assessing how the learning about a specific combination of prior and cue generalizes to another combination. We assume a simple local learning of the form

$$
g\left(\sigma_{\text {prior }}, d_{\text {cue }}, \sigma_{\text {prior }}^{\prime}, d_{\text {cue }}^{\prime}\right)=e^{-\frac{\left(\sigma_{\text {prior }}-\sigma_{\text {prior }}^{\prime}\right)^{2}}{2 \Delta_{\sigma}^{2}}} \cdot e^{-\frac{\left(d_{\text {cue }}-d_{\text {cue }}^{\prime}\right)^{2}}{2 \Delta_{\text {cue }}^{2}}}
$$

where $\Delta_{\sigma}$ and $\Delta_{\text {cue }}$ are two parameters measuring the generalization length respectively in prior and cue space. Overall, the model has eight parameters: the motor variability $\sigma_{\text {motor }}$ and the lapse rate $\lambda$, the initial weights $w_{\text {short }}^{(0)}$ and $w_{\text {long }}^{(0)}$, the learning factor $\eta$, the regularization parameter $\omega$ and the generalization lengths $\Delta_{\sigma}$ and $\Delta_{\text {cue }}$.

This wide array of parameters allows the model to capture different possible classes of non-Bayesian strategies and update rules. Given the number of parameters and complexity of the log likelihood space of the model, when computing the posterior distribution of the parameters we ran much longer chains in order to improve convergence of the sampling algorithm $\left(3 \cdot 10^{5}\right.$ burn-in samples, $3 \cdot 10^{5}$ saved samples per chain).

# References

1. Körding KP, Wolpert DM (2004) The loss function of sensorimotor learning. Proc Natl Acad Sci U S A 101: 9839-9842.
2. Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009) Bayesian model selection for group studies. Neuroimage 46: 1004-1017.
3. Fründ I, Haenel NV, Wichmann FA (2011) Inference for psychometric functions in the presence of nonstationary behavior. J Vis 11: 1-19.
4. Raviv O, Ahissar M, Loewenstein Y (2012) How recent history affects perception: the normative approach and its heuristic approximation. PLoS Comput Biol 8: e1002731.
5. Berniker M, Voss M, Kording K (2010) Learning priors for bayesian computations in the nervous system. PLoS One 5: e12686.

---

#### Page 14

6. Petzschner F, Glasauer S (2011) Iterative bayesian estimation as an explanation for range and regression effects: a study on human path integration. J Neurosci 31: 17220-17229.
7. Verstynen T, Sabes PN (2011) How each movement changes the next: an experimental and theoretical study of fast adaptive priors in reaching. J Neurosci 31: 10050-10059.
8. Nassar MR, Wilson RC, Heasly B, Gold JI (2010) An approximately bayesian delta-rule model explains the dynamics of belief updating in a changing environment. J Neurosci 30: 12366-12378.

---

#### Page 1

## Supporting Text S2 - Noisy probabilistic inference

We introduce two alternative models of stochastic computations in Bayesian inference ('Stochastic posterior models'). The first one (noisy posterior) comprises a representation of the posterior corrupted by noise; in the second one (sample-based posterior), a discrete, approximate representation of the posterior distribution is built out of a number of samples drawn from the posterior.

We show that, for the loss function of our task, for both models the predicted distribution of chosen targets is quantitatively very close to a power function of the posterior distribution in the trial ('Results'). The generality of this result motivates the power function approximation used for decision-making model level SPK (stochastic posterior), Eq. 7 in the paper.

Lastly, we show that, under specific assumptions, the stochasticity in the posterior can also represent a certain type of noise in the prior ('Stochastic posterior from unstructured noise in the prior').

## Stochastic posterior models

According to Bayesian Decision Theory (BDT), the computation of the optimal target $x^{*}$ for a given loss function $\mathcal{L}$ requires three steps:

1. Computation of the posterior probability $p_{\text {post }}(x)$.
2. Computation of the expected loss, $\mathcal{E}(x)=\int p_{\text {post }}(x) \mathcal{L}\left(x, x^{\prime}\right) d x^{\prime}$.
3. Computation of the target $x^{*}$ that minimizes the expected loss, $x^{*}=\arg \min _{x} \mathcal{E}(x)$.

Step 1 corresponds to the inference step and is described by Eq. 2 in the paper. Steps 2 and 3 correspond to action selection (Eq. 4 in the paper).

In principle, noise in decision making could be added to any of the above steps. For parsimony, here we consider models that adds stochasticity to the computation (or representation) of the posterior distribution (step 1), and we analyze how this noise propagates to the inferred optimal target $x^{*}$. However, our results are compatible also with noise injected at later stages (e.g. in action selection).

## Noisy posterior

For ease of calculation, we convert the continuous posterior distribution $p_{\text {post }}(x)$ to a discrete probability distribution $p_{i}=p_{\text {post }}\left(x_{i}\right)$ for a discrete set of target values $\left\{x_{i}\right\}_{1 \leq i \leq N}$ where we assume that the $x_{i}$ cover uniformly the target space with dense spacing $\Delta x .^{1}$

[^0]
[^0]: ${ }^{1}$ The discretization step could be skipped by modelling continuous noise with a Gaussian process [1]. However, the discrete representation makes the model simpler and easier to interpret. The lattice spacing $\Delta x$ is related to the correlation length of a Gaussian process and affects the amount of noise and discretization error.

---

#### Page 2

We model the computation of a 'noisy posterior' (step 1) by adding normally distributed noise to the posterior (see Figure 7 b in the paper):

$$
\widetilde{p}_{\text {post }}(x)=\sum_{i=1}^{N} y_{i} \delta\left(x-x_{i}\right) \quad \text { with } \quad y_{i}=p_{i}+\sigma\left(p_{i}\right) \eta_{i}
$$

where the $\eta_{i}$ are i.i.d. normal random variables and $\sigma\left(p_{i}\right)$ is the SD of the 'decision noise', that in general depends on the value $p_{i}{ }^{2}$ For simplicity, the $\eta_{i}$ are assumed to be statistically independent but it is easy to extend the model to take into account correlations in the noise.

For the form of $\sigma(p)$ we consider two common alternative rules:

- A Poisson-like law: $\sigma_{\text {Poisson }}(p)=\sqrt{p / g}$, where we have defined $g>0$ as a 'neuronal gain' parameter; higher gain corresponds to less noise. The rationale for this rule is that the $y_{i}$ can be thought of as a population of $N$ independent units or channels ('neurons'), each one noisily encoding the posterior probability at a given target value $x_{i}$ (see Figure 7b in the paper). The activation of each unit ('firing rate'), with a global rescaling factor $g$, takes the form $y_{i}=g p_{i}+\sqrt{g p_{i}} \eta_{i}$ which approximates the response of a Poisson neuron with mean activation $g p_{i}$.
- Weber's law (multiplicative noise), in which the noise is proportional to the probability itself, a form of variability which is typical to many sensory magnitudes: $\sigma_{W e b e r}(p)=w \cdot p$, with $w>0$ the Weber's fraction.

For a fixed lattice spacing $\Delta x$, this model of noise in decision making has only one free parameter, $g$ (or $w$ ), that sets the amount of variability in the inference. Note that the 'neural population' description allows for an intuitive understanding of Eq. S1, but the noisy posterior model does not require to commit to this intepretation.

# Sample-based posterior

This model assumes that a discrete, approximate representation of the posterior is constructed by drawing $K$ samples from the posterior [2-4] (see Figure 7c in the paper):

$$
\widetilde{p}_{\text {post }}(x)=\frac{1}{K} \sum_{i=1}^{K} \delta\left(x-x^{(i)}\right) \quad \text { with } \quad x^{(i)} \sim p_{\text {post }}
$$

where the $x^{(i)}$ are i.i.d. samples from the posterior. The parameter $K$ is inversely proportional to the noise in the representation.

## Target choice distribution

For a given posterior distribution $p_{\text {post }}(x)$, Eqs. S1 and S2 allow us to compute several instances of a stochastic posterior $\widetilde{p}_{\text {post }}(x)$ which, after minimization of the expected loss, entail different chosen targets $x^{*}$. By repeating this procedure and binning the results, we can obtain the shape of the distribution of target choices $p_{\text {target }}(x)$ for a given model of stochasticity (see Figure 7e \& 7 f in the paper). However, this method is computationally very expensive.

A simple expression for $p_{\text {target }}(x)$ is needed in order to make efficient use of a stochastic posterior model in data analysis, e.g. to compute the marginal likelihood of a dataset. Our goal is to show that the

[^0]
[^0]: ${ }^{2}$ Formally, $\widetilde{p}_{\text {post }}(x)$ as defined in Eq. S1 is not a probability distribution since, aside of normalization, it is not always non-negative (the $p_{i}$ 's may take negative values for large amounts of noise in the inference). In this case the 'noisy posterior' should be more correctly interpreted simply as an intermediate step in a noisy computation of the expected loss.

---

#### Page 3

target choice probability of these noisy decision-making models is well approximated by a power function of the posterior distribution:

$$
p_{\text {target }}(x) \sim\left[p_{\text {post }}(x)\right]^{\kappa}
$$

where $\kappa \geq 0$ is an appropriate exponent that is the direct equivalent of the noise parameter $g, w$ or $K$; higher values of $\kappa$ correspond to less decision noise. In general, we would like the exponent in Eq. S3 to be a function of the noise parameter, that is for example $\kappa=\kappa(g)$, where the mapping does not depend on the posterior distribution itself but only on the decision noise level (note that the mapping will depend on other fixed details of the model such as the loss function, and the chosen discretization spacing $\Delta x$ for the 'noisy posterior' model).

# Results

We computed the target choice probability predicted by the stochastic posterior models in our task (noisy posterior with either Poisson-like or Weber's law noise, and sample-based posterior). We chose as loss function the inverted Gaussian approximation used by the observer models in the paper (see Methods in the paper; results did not qualitatively change with the square well loss). We took as posterior distributions a representative set of all posterior distributions of the task, built out of several combinations of prior, cue position and cue type (low-noise and high-noise cues), for a total of about 1000 posterior distributions. We took several levels of decision noise (values of $g, w$ or $K$, depending on the model), ranging from an approximately correct inference to an extremely noisy inference. For each posterior distribution and decision noise level we calculated the shape of the target choice distribution via Monte Carlo sampling ( $10^{5}$ samples per distribution).

Figure 1 shows the target choice distributions and related posterior-power fit distributions (Eq. S3) for three illustrative posteriors and five levels of decision noise for the noisy posterior model with Poisson-like noise. For high levels of decision noise, the target choice distribution resembles the posterior distribution (i.e. a posterior-matching strategy), whereas for low levels of decision noise it becomes a narrow distribution peaked on the mode of the posterior (the model tends to a MAP strategy for $g \rightarrow \infty$ ). This may intuitively explain why a power function of the posterior would be a good approximation of the target choice distribution.

We quantified how well a power function of the posterior can approximate the target choice distributions in terms of Kullback-Leibler (KL) divergence. For each noise level, we computed the exponent $\kappa$ that minimizes the KL divergence between posterior-power distributions and target choice distributions in the set (crucially, the same exponent $\kappa$ fit simultaneously all $\sim 1000$ distributions). To assess the goodness of fit in our experiment, we computed mean and SD of the KL divergence according to a log-normal approximation of the posterior distribution of values of $\kappa$ found in the test sessions for our subjects (see paper, section 'Analysis of best observer model').

In general, we found that the posterior-power fit approximates quite well the target choice distribution of all stochastic posterior models. The KL divergence between true distribution and its approximation was $\sim 0.02 \pm 0.02$ nats (mean $\pm$ s.d. across the distribution of values of $\kappa$ ) for all distinct models of noisy inference. These values are equivalent to the KL divergence between two Gaussian distributions with same SD and whose means differ by about one-fourth of their SD.

This analysis shows that a power function of the posterior represents a good approximation of the distribution of target choices of a Bayesian oberver that takes action according to a noisy or sample-based representation of the posterior. This result provides a sound basis for the analytical form chosen for model level SPK (stochastic posterior), Eq. 7 in the paper.

---

#### Page 4

> **Image description.** The image is a figure composed of multiple plots arranged in a grid, comparing target choice distributions with posterior-power approximations under varying noise levels and posterior distributions.
>
> The figure consists of three columns, labeled "Gaussian posterior," "Unimodal posterior," and "Bimodal posterior" at the top. Each column represents a different type of posterior distribution.
>
> Each column contains six plots stacked vertically. The y-axis of each plot is labeled "Probability." The x-axis of the bottom row of plots is labeled "Optimal target (screen units)." The y-axis ranges from 0 to 0.05 in the first three rows and from 0 to 0.1 in the fourth row, and from 0 to 0.2 in the fifth and sixth rows. The x-axis ranges from 0 to 0.8 in all plots.
>
> The first plot in each column shows the posterior distribution as a filled gray curve. The subsequent five plots in each column show a blue line representing the "Decision distribution" and a dashed red line representing the "Posterior-power fit." The level of noise decreases from top to bottom, as indicated by an arrow labeled "Decreasing noise" on the right side of the figure.
>
> At the bottom of the figure, there is a key indicating the representation of each type of data: a filled gray curve for "Posterior," a blue line for "Decision distribution," and a dashed red line for "Posterior-power fit."

Figure 1. Posterior-power approximation of the noisy posterior model. Comparison between the target choice distributions computed according to the true noisy posterior model (with Poisson-like noise) and their posterior-power approximations. The various panels show the target choice distributions $p_{\text {target }}(x)$ (blue lines) and the associated posterior-power fits (red dashed lines) for different posterior distribution and noise level $g$ in the computation. Each column corresponds to a different illustrative posterior distribution, shown on top, divided by class (Gaussian, unimodal and bimodal). Each row, excluding the first, corresponds to a different level of decision noise, with noise decreasing from top to bottom. Analogous fits were found for the sample-based approximation of the posterior.

---

#### Page 5

# Stochastic posterior from unstructured noise in the prior

We show here that the posterior noise model SPK may also subsume the unstructured components of noise in the prior.

If we assume that the internal measurement of the prior is corrupted by multiplicative sensory noise (according to the approximate Weber's law for density or numerosity esimation [5]) and that it changes smoothly in the target position, the estimated prior can be written as:

$$
\tilde{p}_{\text {prior }}(x)=p_{\text {prior }}(x) \cdot(1+\epsilon(x))
$$

where $\epsilon(x)$ is a Gaussian process with zero mean and some appropriately chosen SD and covariance function (see [1]). Crucially, if the observer uses Eq. S4 to build a posterior distribution, we obtain:

$$
\tilde{p}_{\text {post }}(x)=p_{\text {post }}(x)(1+\epsilon(x))
$$

where $p_{\text {post }}(x)$ is the usual, non-noisy posterior (Eq. 2 in the paper). Eq. S5, once appropriately discretized, is formally equivalent to the equation we used to describe a noisy posterior with multiplicative noise (Eq. S1; see also Figure 7b in the paper). Therefore, under these assumptions, the random, unstructured components of noise in the prior can be absorbed within the noisy posterior model.

Note that the estimation noise on the prior that we considered in the paper, model factor P , is a structured form of noise that varies along task-relevant dimensions (such as the width of the prior or the relative weights of bimodal priors). Whereas structured noise can be identified at least in principle, teasing out which stage or component unstructured noise belongs to represents a greater challenge. For example, an experiment that involves a variable number of inference step may be able to distinguish whether noise stems from the computation of the posterior, which is repeated at every step, or from noise in the encoding of the original prior, which happens only once. A paradigm of this kind has been recently used to explore similar issues in a perceptual categorization task [6]. However, this method is still unable to distinguish whether noise appears in the first step (in the encoding or recall of the prior) or at the very last stage, during action selection. Another way to identify noise in the prior could consist in imposing a strong hyperprior on the subjects via extensive training. The level of attraction to such hyperpriors, once learned, may be indicative of the amount of uncertainty in the subjects' measurement of the prior.

## References

1. Rasmussen C, Williams CKI (2006) Gaussian Processes for Machine Learning. The MIT Press.
2. Sundareswara R, Schrater PR (2008) Perceptual multistability predicted by search model for bayesian decisions. J Vis 8: 1-19.
3. Vul E, Goodman ND, Griffiths TL, Tenenbaum JB (2009) One and done? optimal decisions from very few samples. In: Proceedings of the 31st annual conference of the cognitive science society. volume 1, pp. 66-72.
4. Fiser J, Berkes P, Orbán G, Lengyel M (2010) Statistically optimal perception and learning: from behavior to neural representations. Trends Cogn Sci 14: 119-130.
5. Ross J (2003) Visual discrimination of number without counting. Perception 32: 867-870.
6. Drugowitsch J, Wyarta V, Koechlin E (2014). The origin and structure of behavioral variability in perceptual decision-making. Cosyne Abstracts 2014, Salt Lake City USA.

---

#### Page 1

## Supporting Text S3 - Sensorimotor estimation experiment

We performed a sensorimotor estimation experiment to obtain an independent measure of subjects' sensorimotor variability (see 'Methods'). The sensorimotor variability includes subjects' noise in determining the location of the cue and projecting it back onto the target line as well as any motor noise in indicating that location. We found that in general the sensorimotor variability was small and had a negligible impact on performance ('Results'). The estimated parameters were used to construct informative priors for the model comparison in the paper ('Informative priors for the model comparison').

## Methods

Ten subjects ( 3 male and 7 female; age range $21-33$ years) that had taken part in the main experiment also participated in the control experiment.

The experimental setup had the same layout as the main experiment (see Methods and Figure 1 in the paper), with the following differences: (a) no discrete distribution of targets was shown on screen, only a horizontal target line; (b) in all trials the target was drawn randomly from a uniform distribution whose range covered the width of the active screen window; (c) as usual, half of the trials featured short-distance cues and the other half long-distance cues, but both types of cues had no added noise. In each trial the target was always perfectly above the shown cue, with $x \equiv x_{\text {cue }}$.

Subjects performed a short practice session ( 64 trials) followed by a test session (288 trials). Full performance feedback was provided during both practice and test. Feedback consisted in a visual display of the true position of the target and an integer-valued score that was maximal ( 10 points) for a perfect 'hit' and decreased rapidly away from the target, according to the following equation:

$$
\operatorname{Score}(r, x)=\left\lfloor 10 \cdot e^{-\frac{(r-x)^{2}}{2 \sigma_{\text {score }}^{2}}}+0.5\right\rfloor
$$

where $r$ is the response in the trial, $x$ is the target position, $\sigma_{\text {score }}$ is one-tenth of the cursor diameter $\left(8.3 \cdot 10^{-3}\right.$ screen units or 2.5 mm$)$ and $\lfloor x\rfloor$ denotes the floor function.

All subjects' datasets for the sensorimotor estimation session are available online in Dataset S1.

## Results

Results of the sensorimotor estimation session for all subjects are plotted in Figure 1. The root-meansquared error (RMSE) of the response with respect to the true target position was on average ( $9.3 \pm$ $0.8) \cdot 10^{-3}$ screen units for long-distance cues and $(5.2 \pm 0.3) \cdot 10^{-3}$ screen units for short-distance cues (mean $\pm \mathrm{SE}$ across subjects). In general, the RMSE can be divided in a constant bias term and a variance term, but the bias term was overall small, on average $(0.6 \pm 0.5) \cdot 10^{-3}$ screen units, and not significantly different than zero $(p=0.26)$, which means that the error arose almost entirely from the subject's response variability.

---

#### Page 2

> **Image description.** This image contains two scatter plots comparing targeting error for short-distance and long-distance cues. The plots are labeled 'a' and 'b'.
>
> **Panel a:**
>
> - The plot shows individual RMSE (Root Mean Squared Error) on the y-axis, labeled "RMSE (screen units)", ranging from 0 to 0.02. The x-axis represents "Subjects", numbered from 1 to 10.
> - Two distinct sets of data points are plotted: red dots represent "Short-distance cues," and blue dots represent "Long-distance cues."
> - Each data point has a vertical error bar representing the 95% confidence interval.
> - The red dots, representing short-distance cues, are generally clustered lower on the y-axis than the blue dots, indicating lower RMSE values. The blue dots tend to decrease from left to right.
>
> **Panel b:**
>
> - This plot shows the mean RMSE, averaged across all subjects.
> - The y-axis is the same as in panel a, "RMSE (screen units)", ranging from 0 to 0.02. The x-axis is labeled "Mean."
> - Similar to panel a, there are two data points: a red dot for short-distance cues and a blue dot for long-distance cues, each with a vertical error bar representing the 95% confidence interval.
> - The blue dot (long-distance cues) is higher on the y-axis than the red dot (short-distance cues), indicating a higher mean RMSE for long-distance cues.

Figure 1. Targeting error for short-distance and long-distance cues. RMSE of the responses, with respect to true target position, for different distance of the cues from the target line, either 'short' (brown dots) or 'long' (blue dots); 0.01 screen units correspond to 3 mm . a: Individual RMSE. For visualization, subjects are ordered by average precision. b: Mean RMSE, averaged across subjects. In both graphs error bars are $95 \%$ confidence intervals computed via bootstrap.

Since subjects knew that the cues were fully informative about the target position, all variability in their responses originated from two sources: sensory noise (error in projecting the cue position on the target line) and motor noise. We assumed that sensory and motor noise were independent and normally distributed, and that sensory variability was proportional to the distance of the cue from the target line (Weber's law). Under these assumptions, variance of subjects' responses was described by the following formula:

$$
\sigma_{\text {response }}^{2}=\sigma_{\text {motor }}^{2}+w_{\text {sensory }}^{2} d_{\text {cue }}^{2}
$$

where $w_{\text {sensory }}$ is Weber's fraction and $d_{\text {cue }}$ is the distance of the cue from the target line. Using Eq. S1 we were able to estimate participants' sensorimotor parameters; results are reported in Table 1.

|         Parameter         | Description                | Mean $\pm$ Std <br> (screen units) | Mean $\pm$ Std <br> $(\mathrm{mm})$ |
| :-----------------------: | :------------------------- | :--------------------------------: | :---------------------------------: |
| $\sigma_{\text {motor }}$ | Motor noise                |   $(3.6 \pm 1.1) \cdot 10^{-3}$    |            $1.1 \pm 0.3$            |
|  $\Sigma_{\text {low }}$  | Sensory noise (short cues) |   $(3.5 \pm 1.1) \cdot 10^{-3}$    |            $1.1 \pm 0.3$            |
| $\Sigma_{\text {high }}$  | Sensory noise (long cues)  |   $(8.1 \pm 2.6) \cdot 10^{-3}$    |            $2.4 \pm 0.8$            |

Table 1. Average estimated sensorimotor parameters. Group-average estimated motor and sensory noise parameters. Estimates were obtained from the data through Eq. S1.

The estimated parameters in Table 1 allowed us to assess the typical impact of realistic values of sensorimotor noise on subjects' performance. First, we computed the performance of the optimal ideal observer model with added realistic noise. In order to do so, we generated 1000 subjects by sampling from the distribution of estimated sensorimotor parameters and we then simulated their behavior on our subjects' datasets according to the optimal observer model. We found an average optimality index of $0.997 \pm 0.001$ which is empirically indistinguishable from one. The difference in performance induced by the sensorimotor noise was analogously negligible for the simulations of other ideal observer models, such

---

#### Page 3

as the 'prior-only' or 'cue-only' models (see Figure 5 in the paper). These results show that motor and sensory noise had a very limited impact on subjects' performance.

# Informative priors for the model comparison

The pooled estimated parameters summarized in Table 1 were used to construct informative priors for the motor and sensory parameters that were applied in our model comparison (see paper and Text S1). Bootstrapped parameters were fit with log-normal distributions with log-scale $\mu$ and shape parameter $\sigma$ (which correspond to mean and SD in log space; see Figure 2). The resulting parameters of the priors were $\mu=\log 3.4 \cdot 10^{-3}$ screen units, $\sigma=0.38$ for $\sigma_{\text {motor }}$; and $\mu=\log 7.7 \cdot 10^{-3}$ screen units, $\sigma=0.32$ for $\Sigma_{\text {high }}$. The prior on $\sigma_{\text {motor }}$ was used in all observer models, whereas the prior on $\Sigma_{\text {high }}$ was used only in the observer models with sensory noise (model factor S).

Using an independent experiment to construct informative priors can be thought of as a 'soft' generalization of the typical procedure that consists in directly applying independently estimated parameters to an observer model [1]. In that case, the constructed priors are delta functions on point estimates of the subjects' parameters. Here, instead, pooled measured parameters were used to compute distributions that represent realistic values for the model parameters in our task (that is, informative priors).

> **Image description.** This image is a plot showing probability density curves for motor and sensory noise parameters, along with experimental data points.
>
> - **Axes:** The horizontal axis is labeled "Parameter value (screen units)" and ranges from 0 to 0.02. The vertical axis is labeled "Probability density" and ranges from 0 to 1.2.
>
> - **Curves:** There are two probability density curves:
>
>   - A brown/olive-green curve representing "Motor noise parameter σmotor". This curve has a peak around 0.003-0.004.
>   - A blue curve representing "Sensory noise parameter Σhigh". This curve has a peak around 0.007.
>
> - **Data Points:** There are two sets of data points plotted on the graph:
>
>   - Brown/olive-green dots, clustered around the left side of the graph, corresponding to motor noise.
>   - Blue dots, clustered more towards the center of the graph, corresponding to sensory noise.
>   - Each data point has a horizontal error bar associated with it.
>
> - **Legend:** A legend in the upper right corner identifies the curves:
>   - A brown line is labeled "Motor noise parameter σmotor".
>   - A blue line is labeled "Sensory noise parameter Σhigh".

Figure 2. Priors over sensorimotor parameters. The experimental estimates of individual parameters for motor noise (brown dots) and sensory noise (purple dots) are used to construct informative log-normal priors for $\sigma_{\text {motor }}$ (brown line) and $\Sigma_{\text {high }}$ (purple line) in the main experiment. Error bars are $95 \%$ confidence intervals, computed via bootstrap.

## References

1. Tassinari H, Hudson T, Landy M (2006) Combining priors and noisy visual cues in a rapid pointing task. J Neurosci 26: 10154-10163.
