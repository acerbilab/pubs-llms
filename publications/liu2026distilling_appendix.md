# Distilling noise characteristics and prior expectations in multisensory causal inference - Appendix

---

## Section A: Lapse distribution selection

One observation in all our model fit visualizations over unisensory data is a systematic overestimation of the SD in participant responses. This is demonstrated in Subpanel c) of every relevant figure starting from Fig 2 in the main text. This SD overestimation is a consequence of our simplistic lapse distribution assumption, which gives nonzero probabilities of responding uniformly within the response range $[-45, 45]$, yielding possibly unrealistically large responses.

For demonstration purposes, we have refrained from plotting lapses in Fig A. Similar to Fig 8 in the main text, this figure portrays the Exp-GaussianLaplace model fitted to unisensory data, but now without showing lapse responses in the model predictive samples. A comparison between Fig 8C in the main text and Fig A Panel C here demonstrates how the uniform lapse leads to SD overestimations.

In light of this SD overestimation effect, we have also considered a truncated-Gaussian lapse distribution with the response range $[-45, 45]$ to mitigate the problem. In contrast to the uniform lapse distribution $U[-45, 45]$ used in all previous models, the truncated-Gaussian lapse distribution requires the fitting of not only a lapse rate $\lambda$, but also the lapse distribution’s standard deviation $\sigma_{lapse}$.

We have fitted an Exp-GaussianLaplace model assuming Gaussian-lapse to the unisensory data. Unfortunately, its visualization in Fig B reveals that using a truncated-Gaussian lapse distribution does not seem to resolve the SD overestimation problem. Furthermore, BIC model comparison results cannot distinguish the truncated-Gaussian lapse model from its uniform lapse counterpart featuring one less parameter. Specifically, the Sum and $[2.5\%, 97.5\%]$ bootstrapping intervals for the BIC difference are $-41.8$ and $[-114, 18.6]$ respectively, where the difference is computed with respect to the Uniform lapse model. For this reason, we used the standard uniform lapse model in the main text, leaving further investigations for future work.

> **Image description.** A 4-row × 3-column grid of plots comparing model predictions (shaded bands) to human localization data (points with error bars), for three visual-reliability conditions and one auditory condition. Column (a) shows response-distribution histograms, column (b) shows response bias, and column (c) shows response variability (SD).
>
> **Column (a), response distributions.** In each row (High-, Medium-, Low-Reliability visual, and Auditory), seven colored curves (red, blue, green, purple, orange, brown, pink, ordered from the most negative to the most positive stimulus location) plot proportion of responses (y-axis, 0 to about 0.6–0.7) against reported stimulus location in degrees (x-axis, roughly -30° to 30°). Each curve combines a shaded continuous band (model-predicted distribution) with discrete data points and small vertical error bars (empirical data). A legend at top right of the first row lists the visual stimulus-location bins as contiguous ranges (e.g. [-20,-14.3]°, [-14.3,-8.6]°, ... up to [14.3,20]°); a legend in the bottom row lists the auditory stimulus locations as seven discrete values from -15° to 15° in 5° steps. In every row the curves are ordered left-to-right matching their stimulus location, and the central (purple, near 0°) curve is consistently the tallest and narrowest (peak proportion roughly 0.6–0.7), while peripheral curves are shorter and wider. Curves broaden and increasingly overlap from High- to Medium- to Low-Reliability, reflecting greater response noise; the Auditory row's curves are about as broad as the Low-Reliability ones.
>
> **Column (b), bias.** Black data points with vertical error bars plot response bias in degrees (y-axis, -20° to 20°) against the same seven stimulus locations (x-axis, roughly -17° to 17°), with a dashed horizontal reference line at 0° and a thin gray shaded band tracing the model fit through the points. High-Reliability: a gentle decline, from about +2° at the leftmost location to about -1° at the rightmost. Medium-Reliability: a similar but slightly steeper decline, from about +3° to -2°. Low-Reliability: a pronounced decline, from about +7° at the leftmost location, crossing zero near the center, down to about -6° at the rightmost — i.e., responses are pulled toward the central location. Auditory: nearly flat and close to 0° across the whole range.
>
> **Column (c), variability.** Black data points with error bars plot the SD of the localization response in degrees (y-axis, 0° to 6°) against stimulus location, overlaid with a gray shaded model-fit band. All four rows show a V-shaped (or shallow U-shaped) pattern, with SD lowest near the central location and rising toward both periphery. High-Reliability dips to about 1.7°–2° near center and rises to about 2.5°–2.7° at the extremes; the dip deepens and the range widens through Medium-Reliability (about 2°–3.5°) and Low-Reliability (about 2.5°–4.8°); the Auditory row's V sits highest (data about 2.9°–5°, with the band reaching about 5.5° at the edges, slightly above the data).

Fig A. The Exp-GaussianLaplace parametric model fitted jointly on UV and UA data for all participants, visualized without lapses. This figure is identical to Fig 8 in the main text, but without lapses in the model predictive samples. The layout and color schemes are identical to Fig 8 in the main text.

> **Image description.** A 4-row × 3-column grid of plots comparing model predictions (shaded bands) to human localization data (points with error bars), for three visual-reliability conditions and one auditory condition. Column (a) shows response-distribution histograms, column (b) shows response bias, and column (c) shows response variability (SD).
>
> **Column (a), response distributions.** In each row (High-, Medium-, Low-Reliability visual, and Auditory), seven colored curves (red, blue, green, purple, orange, brown, pink, ordered from the most negative to the most positive stimulus location) plot proportion of responses (y-axis, 0 to about 0.6–0.7) against reported stimulus location in degrees (x-axis, roughly -30° to 30°). Each curve combines a shaded continuous band (model-predicted distribution) with discrete data points and small vertical error bars (empirical data). A legend at top right of the first row lists the visual stimulus-location bins as contiguous ranges (e.g. [-20,-14.3]°, [-14.3,-8.6]°, ... up to [14.3,20]°); a legend in the bottom row lists the auditory stimulus locations as seven discrete values from -15° to 15° in 5° steps. In every row the curves are ordered left-to-right matching their stimulus location, and the central (purple, near 0°) curve is consistently the tallest and narrowest (peak proportion roughly 0.6–0.7), while peripheral curves are shorter and wider. Curves broaden and increasingly overlap from High- to Medium- to Low-Reliability, reflecting greater response noise; the Auditory row's curves are about as broad as the Low-Reliability ones.
>
> **Column (b), bias.** Black data points with vertical error bars plot response bias in degrees (y-axis, -20° to 20°) against the same seven stimulus locations (x-axis, roughly -17° to 17°), with a dashed horizontal reference line at 0° and a thin gray shaded band tracing the model fit through the points. High-Reliability: a gentle decline, from about +2° at the leftmost location to about -1° at the rightmost. Medium-Reliability: a similar but slightly steeper decline, from about +3° to -2°. Low-Reliability: a pronounced decline, from about +7° at the leftmost location, crossing zero near the center, down to about -6° at the rightmost — i.e., responses are pulled toward the central location. Auditory: nearly flat and close to 0° across the whole range.
>
> **Column (c), variability.** Black data points with error bars plot the SD of the localization response in degrees (y-axis, 0° to 6°) against stimulus location, overlaid with a gray shaded model-fit band. In all four rows the data trace a V-shaped (or shallow U-shaped) pattern, with SD lowest near the central location and rising toward both extremes: about 1.7°–2.6° in the High-Reliability row, 1.9°–3.5° in the Medium-Reliability row, 2.7°–4.8° in the Low-Reliability row and 2.9°–5° in the Auditory row. The model band follows the V but lies above most of the peripheral data points in every row, overestimating the SD there: at the outermost locations it reaches about 3.1° (High), 3.6° (Medium), 5° (Low) and 5.8° (Auditory). Near the center the band comes down to within about 0.5° of the lowest data point.

Fig B. The Exp-GaussianLaplace parametric model fitted jointly on UV and UA data for all participants, in which the auditory range recalibration factor is a free parameter, and a Gaussian lapse distribution is used. The layout and color schemes are identical to Fig A.

## Section B: Model comparison result tables

### Section B.1: Unisensory data fits

#### Section B.1.1: Const-SingleGaussian (vanilla) models

> **Image description.** Three stacked horizontal bar-chart panels, one above another, each comparing the same three models by a different fit-quality metric (ΔNLL, ΔAIC, ΔBIC from top to bottom, each named beneath its own panel). Only the bottom panel carries numeric x-axis tick labels, running from 0 to 3000 in steps of 500, suggesting a common horizontal scale shared by all three panels.
>
> Each panel has the same three horizontal bars, labeled (top to bottom within the panel) Const-SingleGaussian, Const-SingleGaussian_4/3, and Const-SingleGaussian_1. Const-SingleGaussian is the reference model and appears only as a black dot at 0 with no bar. The other two rows are solid gray rectangles extending from 0 to a black dot marking the point estimate, with a horizontal line-and-cap error bar (95% interval) drawn through the dot.
>
> In the ΔNLL panel, read on the shared axis, the Const-SingleGaussian_4/3 bar reaches roughly 480 (error bar spanning about 225–765), and Const-SingleGaussian_1 reaches roughly 2050 (error bar about 1305–2850); these are twice the NLL differences listed in Table A, and the Const-SingleGaussian_1 bar here is slightly longer than in the other two panels. In the ΔAIC panel, Const-SingleGaussian_4/3 reaches roughly 450 (about 195–735) and Const-SingleGaussian_1 roughly 2020 (about 1275–2820). In the ΔBIC panel, Const-SingleGaussian_4/3 reaches roughly 380 (about 120–650) and Const-SingleGaussian_1 roughly 1950 (about 1200–2715).
>
> Across all three panels the pattern is consistent: Const-SingleGaussian_1 has the longest bar and widest error bar, Const-SingleGaussian_4/3 a much shorter bar, and Const-SingleGaussian sits at zero.

Fig C. NLL, AIC, and BIC model comparison results for Const-SingleGaussian (vanilla) parametric models fitted on UV+UA data. Quantitative values are in the tables below.

Table A. NLL differences of Const-SingleGaussian (vanilla) models fitted on unisensory data, with sum-across-participants values and 95% bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Const-SingleGaussian | 0 | (0, 0) |
| Const-SingleGaussian_4/3 | 240.90 | (112.56, 382.04) |
| Const-SingleGaussian_1 | 1026.00 | (653.40, 1424.77) |

Table B. AIC differences of Const-SingleGaussian (vanilla) models fitted on unisensory data, with sum-across-participants values and 95% bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Const-SingleGaussian | 0 | (0, 0) |
| Const-SingleGaussian_4/3 | 451.80 | (195.11, 734.08) |
| Const-SingleGaussian_1 | 2022.00 | (1276.81, 2819.54) |

Table C. BIC differences of Const-SingleGaussian (vanilla) models fitted on unisensory data, with sum-across-participants values and 95% bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Const-SingleGaussian | 0 | (0, 0) |
| Const-SingleGaussian_4/3 | 378.42 | (121.79, 648.75) |
| Const-SingleGaussian_1 | 1948.62 | (1203.43, 2714.57) |

#### Section B.1.2: All models

> **Image description.** A single horizontal bar chart with ten bars, one per model, plotting ΔBIC (x-axis, 0 to 6000) for models fitted jointly on visual and auditory unisensory localization data. Each non-reference bar is a solid gray rectangle running from 0 to a black dot (point estimate), with a horizontal line-and-cap error bar (95% interval) through the dot; the reference model shows only a dot at 0, no bar.
>
> From top to bottom: Exp-GaussianLaplace is the reference (dot at 0). Exp-GaussianLaplace_4/3 has a short bar (roughly 370, interval about 100–680). Exp-GaussianLaplace_1 is markedly longer (roughly 1490, interval about 975–1990). Exp-SingleGaussian is longer still (roughly 2025, interval about 1270–2790). Const-GaussianLaplace has a short bar (roughly 250, interval about 10–500). Exp-TwoGaussians has the shortest non-reference bar (roughly 190, interval about 70–330). Const-SingleGaussian is long (roughly 2440, interval about 1490–3380). Const-SingleGaussian_4/3 is longer still (roughly 2820, interval about 1790–3830). Const-SingleGaussian_1 has the longest bar of all, extending past the panel's midpoint (roughly 4390, interval about 3210–5625). Semiparametric, at the bottom, has a mid-length bar (roughly 2275) with a noticeably narrower error bar (interval about 2180–2360) than any other model, indicating a much more precise estimate.
>
> Overall, the chart shows a mix of short bars close to the reference (Exp-GaussianLaplace_4/3, Exp-TwoGaussians and Const-GaussianLaplace), an intermediate Exp-GaussianLaplace_1, and long bars far from it (the Const-SingleGaussian family and Exp-SingleGaussian), with error-bar width scaling roughly with bar length except for the visibly tighter Semiparametric interval.

Fig D. BIC model comparison results for models fitted on UV+UA data. Quantitative values are in the tables below.

Table D. NLL differences of all models fitted on unisensory data, with sum-across-participants values and bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Semiparametric | 0 | (0, 0) |
| Exp-GaussianLaplace | 207.35 | (163.56, 252.06) |
| Exp-GaussianLaplace_4/3 | 444.93 | (293.59, 629.90) |
| Exp-GaussianLaplace_1 | 1004.52 | (751.52, 1264.55) |
| Exp-SingleGaussian | 1324.08 | (916.25, 1750.00) |
| Const-GaussianLaplace | 540.74 | (383.44, 705.77) |
| Exp-TwoGaussians | 301.86 | (213.96, 396.62) |
| Const-SingleGaussian | 1737.16 | (1235.21, 2257.05) |
| Const-SingleGaussian_4/3 | 1978.06 | (1432.95, 2534.14) |
| Const-SingleGaussian_1 | 2763.16 | (2150.18, 3426.67) |

Table E. AIC differences of models fitted on unisensory data, with sum-across-participants values and bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Semiparametric | 365.31 | (275.87, 452.88) |
| Exp-GaussianLaplace | 0 | (0, 0) |
| Exp-GaussianLaplace_4/3 | 445.18 | (176.60, 763.97) |
| Exp-GaussianLaplace_1 | 1564.36 | (1049.02, 2088.37) |
| Exp-SingleGaussian | 2173.48 | (1416.30, 2971.41) |
| Const-GaussianLaplace | 546.79 | (304.72, 805.99) |
| Exp-TwoGaussians | 189.03 | (69.30, 336.71) |
| Const-SingleGaussian | 2879.62 | (1934.28, 3862.44) |
| Const-SingleGaussian_4/3 | 3331.42 | (2304.28, 4382.25) |
| Const-SingleGaussian_1 | 4901.63 | (3723.83, 6196.64) |

Table F. BIC differences of models fitted on unisensory data, with sum-across-participants values and bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Semiparametric | 2273.21 | (2180.21, 2360.43) |
| Exp-GaussianLaplace | 0 | (0, 0) |
| Exp-GaussianLaplace_4/3 | 371.80 | (103.23, 677.21) |
| Exp-GaussianLaplace_1 | 1490.98 | (975.64, 1992.09) |
| Exp-SingleGaussian | 2026.72 | (1269.31, 2791.96) |
| Const-GaussianLaplace | 253.27 | (10.68, 502.29) |
| Exp-TwoGaussians | 189.03 | (69.30, 329.82) |
| Const-SingleGaussian | 2439.34 | (1492.82, 3383.42) |
| Const-SingleGaussian_4/3 | 2817.76 | (1789.63, 3827.16) |
| Const-SingleGaussian_1 | 4387.96 | (3209.29, 5626.36) |

### Section B.2: All-tasks fits

> **Image description.** A single horizontal bar chart with 21 bars — seven model families, each split into three rows for the causal-inference strategies MS, MA, and PM — plotting ΔBIC (x-axis, -2000 to 8000) for parametric models fitted on all tasks combined. A vertical line at x = 0 runs through the whole plot. Each bar is a solid gray rectangle from 0 to a black dot (point estimate), with a horizontal line-and-cap error bar (95% interval) through the dot; three rows additionally show a second, white/unfilled (outline-only) bar.
>
> Exp-GaussianLaplace-PM is the reference (a dot at 0, no visible bar). Exp-GaussianLaplace-MS has a short bar (roughly 700) and -MA a very short bar (roughly 270), whose error bar dips slightly below zero. Exp-SingleGaussian's three bars (MS, MA, PM) are all long and similar in length (roughly 2970–3640), among the longest in the chart. Const-GaussianLaplace's three bars are medium length (roughly 1250–1660). Exp-TwoGaussians' three bars are short-to-medium (roughly 730–1120). Const-SingleGaussian's three bars are the longest in the chart (roughly 4190–4940), extending furthest to the right. ParamBest's three bars are short: MS (roughly 550) and MA (roughly 220) are positive with wide error bars crossing zero, while PM is a small bar extending slightly left of zero (roughly -85).
>
> The three LiftedSemiparam rows (MS, MA, PM) are unique in showing two bars each: a short gray filled bar sitting at or left of zero (roughly -220 to -680, growing more negative from MS to PM) paired with a longer white/outline bar extending from 0 well to the right (roughly 3500–3975, each with its own black-dot marker and error bar). The PM row's white bar is the shortest of the three (roughly 3520) while its gray bar is the most negative of the three (roughly -680).
>
> Error bars are generally wide relative to bar length for the shorter bars (e.g., Exp-GaussianLaplace-MA, ParamBest-MA/PM) and narrower, relatively speaking, for the longest bars (e.g., the Const-SingleGaussian and Exp-SingleGaussian rows).

Fig E. BIC model comparison results for parametric models fitted on all tasks. For the LiftedSemiparam models, the white bar denotes BIC scores accounting for parameters fitted in the earlier semiparametric fits that LiftedSemiparam model fits were based on. The gray bar denotes BIC scores that exclude these parameters. Quantitative values are in the tables below.

Table G. NLL differences of models fitted on all tasks, with sum-across-participants values and bootstrapping intervals.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Exp-GaussianLaplace-MS | 357.80 | (72.31, 674.7) |
| Exp-GaussianLaplace-MA | 136.10 | (-209.48, 507.23) |
| Exp-GaussianLaplace-PM | 0 | (0, 0) |
| Exp-SingleGaussian-MS | 1941.64 | (1374.28, 2522.51) |
| Exp-SingleGaussian-MA | 1605.36 | (1123.01, 2109.31) |
| Exp-SingleGaussian-PM | 1677.29 | (1133.31, 2230.46) |
| Const-GaussianLaplace-MS | 1070.72 | (773.87, 1381.42) |
| Const-GaussianLaplace-MA | 956.96 | (552.64, 1419.1) |
| Const-GaussianLaplace-PM | 865.79 | (666.54, 1059.77) |
| Exp-TwoGaussians-MS | 559.20 | (176.21, 996.49) |
| Exp-TwoGaussians-MA | 366.71 | (-58.13, 836.76) |
| Exp-TwoGaussians-PM | 462.45 | (117.84, 896.98) |
| Const-SingleGaussian-MS | 2829.18 | (2107.87, 3548.45) |
| Const-SingleGaussian-MA | 2454.61 | (1837.34, 3077.97) |
| Const-SingleGaussian-PM | 2555.92 | (1889.36, 3217.92) |
| ParametricBest-MS | 273.13 | (-17.98, 593.01) |
| ParametricBest-MA | 116.55 | (-233.08, 493.42) |
| ParametricBest-PM | -43.32 | (-97.43, -7.55) |
| LiftedSemiparametric-MS | 368.95 | (151.54, 610.39) |
| LiftedSemiparametric-MA | 344.07 | (61.09, 657.78) |
| LiftedSemiparametric-PM | 140.17 | (-35.01, 317.72) |

Table H. AIC differences of models fitted on all tasks, with sum-across-participants values and bootstrapping intervals. * denotes AIC computed for LiftedSemiparam models, including parameters fitted during the earlier Semiparam model fits.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Exp-GaussianLaplace-MS | 715.59 | (144.62, 1349.4) |
| Exp-GaussianLaplace-MA | 272.19 | (-418.95, 1014.46) |
| Exp-GaussianLaplace-PM | 0 | (0, 0) |
| Exp-SingleGaussian-MS | 3823.28 | (2688.56, 4985.02) |
| Exp-SingleGaussian-MA | 3150.72 | (2186.02, 4158.63) |
| Exp-SingleGaussian-PM | 3294.57 | (2206.63, 4400.92) |
| Const-GaussianLaplace-MS | 2021.44 | (1427.74, 2642.83) |
| Const-GaussianLaplace-MA | 1793.93 | (985.29, 2718.2) |
| Const-GaussianLaplace-PM | 1611.58 | (1213.09, 1999.54) |
| Exp-TwoGaussians-MS | 1118.40 | (352.41, 1992.97) |
| Exp-TwoGaussians-MA | 733.41 | (-116.26, 1673.52) |
| Exp-TwoGaussians-PM | 924.91 | (235.68, 1793.97) |
| Const-SingleGaussian-MS | 5478.35 | (4035.73, 6916.9) |
| Const-SingleGaussian-MA | 4729.21 | (3494.69, 5975.95) |
| Const-SingleGaussian-PM | 4931.83 | (3598.72, 6255.85) |
| ParametricBest-MS | 546.25 | (-35.95, 1186.02) |
| ParametricBest-MA | 229.10 | (-469.63, 982.62) |
| ParametricBest-PM | -86.64 | (-194.86, -15.1) |
| LiftedSemiparametric-MS | 497.90 | (63.08, 980.78) |
| LiftedSemiparametric-MA | 448.14 | (-117.81, 1075.57) |
| LiftedSemiparametric-PM | 40.34 | (-310.02, 395.43) |
| LiftedSemiparametric-MS* | 1547.90 | (1113.08, 2030.78) |
| LiftedSemiparametric-MA* | 1498.14 | (932.19, 2125.57) |
| LiftedSemiparametric-PM* | 1090.34 | (739.98, 1445.43) |

Table I. BIC differences of models fitted on all tasks, with sum-across-participants values and bootstrapping intervals. * denotes BIC computed for LiftedSemiparam models, including parameters fitted during the earlier Semiparam model fits.

| | Sum | CI (2.5%, 97.5%) |
| --- | --- | --- |
| Exp-GaussianLaplace-MS | 715.59 | (144.62, 1321.81) |
| Exp-GaussianLaplace-MA | 272.19 | (-418.95, 984.14) |
| Exp-GaussianLaplace-PM | 0 | (0, 0) |
| Exp-SingleGaussian-MS | 3643.42 | (2508.39, 4758.8) |
| Exp-SingleGaussian-MA | 2970.86 | (2005.75, 3937.04) |
| Exp-SingleGaussian-PM | 3114.71 | (2026.6, 4178.73) |
| Const-GaussianLaplace-MS | 1661.72 | (1067.5, 2257.67) |
| Const-GaussianLaplace-MA | 1434.20 | (624.75, 2320.52) |
| Const-GaussianLaplace-PM | 1251.86 | (852.92, 1625.8) |
| Exp-TwoGaussians-MS | 1118.40 | (352.41, 1955.4) |
| Exp-TwoGaussians-MA | 733.41 | (-116.26, 1631.71) |
| Exp-TwoGaussians-PM | 924.91 | (235.68, 1753.44) |
| Const-SingleGaussian-MS | 4938.77 | (3495.05, 6323.04) |
| Const-SingleGaussian-MA | 4189.63 | (2953.93, 5389.78) |
| Const-SingleGaussian-PM | 4392.25 | (3058.31, 5666.54) |
| ParametricBest-MS | 546.25 | (-35.95, 1159.76) |
| ParametricBest-MA | 217.08 | (-481.21, 940.81) |
| ParametricBest-PM | -86.64 | (-194.86, -17.57) |
| LiftedSemiparametric-MS | -221.55 | (-657.97, 240.25) |
| LiftedSemiparametric-MA | -271.30 | (-837.84, 328.91) |
| LiftedSemiparametric-PM | -679.10 | (-1030.17, -338.32) |
| LiftedSemiparametric-MS* | 3976.02 | (3545.92, 4427.55) |
| LiftedSemiparametric-MA* | 3926.26 | (3363.12, 4523.31) |
| LiftedSemiparametric-PM* | 3518.46 | (3170.21, 3856.51) |

## Section C: Model response distributions

### Section C.1: Unisensory parametric model response distributions

> **Image description.** A 4-row × 3-column grid of plots comparing model predictions (shaded ribbons) to human localization data (points with vertical error bars), for three visual-reliability conditions and one auditory condition. A square bracket to the left of the first three rows groups them under the label "Visual" (High Reliability, Med. Reliability, Low Reliability, top to bottom); "Auditory" labels the fourth row separately. Column (a) shows response-distribution curves, column (b) shows response bias, and column (c) shows the SD of the localization response.
>
> **Column (a), response distributions.** In each row, seven colored curves (red, blue, green, purple, orange, brown, pink, ordered from the most negative to the most positive stimulus location) plot proportion of responses (y-axis ticks 0 to 0.6, with the tallest data points above the top tick) against reported stimulus location in degrees (x-axis, roughly -30° to 30°). Each curve overlays a shaded continuous band (the model's predicted distribution) with discrete data points and small vertical error bars (the empirical data). A legend at top right of the first row lists the seven visual stimulus-location bins as contiguous ranges (e.g. [-20,-14.3]°, [-14.3,-8.6]°, ... up to [14.3,20]°); a legend in the Auditory row lists seven discrete auditory locations from -15° to 15° in 5° steps. In every row the curves are ordered left-to-right matching their stimulus location, and the central (purple, near 0°) data are by far the tallest. In every row this central data point (roughly 0.57–0.7 in proportion) sits far above the top of its model ribbon, which peaks much lower: about 0.47 in the High-Reliability row, 0.4 in the Medium-Reliability row and about 0.3 in the Low-Reliability and Auditory rows. The ribbon is much wider and shorter than the sharp empirical spike at the center. In the High-Reliability row the green and orange data also peak nearer the center than their ribbons (data about 0.52–0.53 at ±3°), while the more peripheral ribbons track their own data points closely. Curves broaden and increasingly overlap from High- to Medium- to Low-Reliability; in the Low-Reliability row the seven curves pile up so heavily near the center that individual ribbons and points are hard to disentangle, and the Auditory row's curves are about as broad as the Low-Reliability ones.
>
> **Column (b), bias.** Black data points with vertical error bars plot response bias in degrees (y-axis, -20° to 20°) against the same seven stimulus locations (x-axis, roughly -17° to 17°), with a dashed horizontal reference line at 0° and a thin gray shaded model-fit band running through the points. High-Reliability: a gentle decline from about +2° at the leftmost location to about -1° at the rightmost. Medium-Reliability: a similar, slightly steeper decline, from about +3° to -2°. Low-Reliability: a pronounced decline, from about +7° at the leftmost location, crossing zero near the center, down to about -6° at the rightmost. Auditory: nearly flat, close to 0° across the whole range. In all four rows the gray band sits directly on the data points, tracking them closely with no visible mismatch.
>
> **Column (c), variability (SD).** Black data points with error bars plot the SD of the localization response in degrees (y-axis, 0° to 6°) against stimulus location, overlaid with a gray shaded model-fit band. In every row the data trace a V-shaped (or shallow U-shaped) pattern, lowest near the central location and rising toward both extremes. The model band, however, is nearly flat across the whole stimulus range in all four rows and does not reproduce this dip. High-Reliability: the band sits roughly level around 2.5°–2.8°, while the data dip to about 1.7°–2° near the center. Medium-Reliability: the band is roughly level around 3°–3.5°, while data dip to about 2° at the center. Low-Reliability: the band sits around 4°–4.3°, essentially level, while data dip to about 2.7° at the center and rise to about 4.4°–4.8° at the extremes. Auditory: the band sits around 4.7°–5°, flat, while the data dip to about 2.9° at the center. In every row the flat band passes noticeably above the central data points and, in the Medium- and Low-Reliability rows, below the rightmost data point, so the band cuts through the middle of the V rather than following its shape — the clearest visible mismatch in the figure.

Fig F. The Const-SingleGaussian (vanilla) model fitted jointly on UV and UA data for all participants, with a free auditory range recalibration parameter. Subplot notations are identical to Fig A.

> **Image description.** A 4-row × 3-column grid of plots comparing model predictions (shaded ribbons) to human localization data (points with vertical error bars), for three visual-reliability conditions and one auditory condition. A square bracket to the left of the first three rows groups them under the label "Visual" (High Reliability, Med. Reliability, Low Reliability, top to bottom); "Auditory" labels the fourth row separately. Column (a) shows response-distribution curves, column (b) shows response bias, and column (c) shows the SD of the localization response.
>
> **Column (a), response distributions.** In each row, seven colored curves (red, blue, green, purple, orange, brown, pink, ordered from the most negative to the most positive stimulus location) plot proportion of responses (y-axis ticks 0 to 0.6, with the tallest data points above the top tick) against reported stimulus location in degrees (x-axis, roughly -30° to 30°). Each curve overlays a shaded continuous band (the model's predicted distribution) with discrete data points and small vertical error bars (the empirical data). A legend at top right of the first row lists the seven visual stimulus-location bins as contiguous ranges (e.g. [-20,-14.3]°, [-14.3,-8.6]°, ... up to [14.3,20]°); a legend in the Auditory row lists seven discrete auditory locations from -15° to 15° in 5° steps. In every row the curves are ordered left-to-right matching their stimulus location, and the central (purple, near 0°) data are by far the tallest. Its data point (roughly 0.57–0.7 in proportion) sits visibly above the top of its model ribbon, which peaks lower, around 0.42–0.5, in every row: the ribbon is wider and shorter than the sharp empirical spike at the center. The gap is about 0.2 in the three visual rows and smallest in the Auditory row (about 0.57 against 0.48). In the High-Reliability row the green and orange data also peak nearer the center than their ribbons (data about 0.52–0.53 at ±3°), while the more peripheral ribbons track their own data points closely. Curves broaden and increasingly overlap from High- to Medium- to Low-Reliability; in the Low-Reliability row the seven curves pile up so heavily near the center that individual ribbons and points are hard to disentangle, and the Auditory row's curves are about as broad as the Low-Reliability ones.
>
> **Column (b), bias.** Black data points with vertical error bars plot response bias in degrees (y-axis, -20° to 20°) against the same seven stimulus locations (x-axis, roughly -17° to 17°), with a dashed horizontal reference line at 0° and a thin gray shaded model-fit band running through the points. High-Reliability: a gentle decline from about +2° at the leftmost location to about -1° at the rightmost. Medium-Reliability: a similar, slightly steeper decline, from about +3° to -2°. Low-Reliability: a pronounced decline, from about +7° at the leftmost location, crossing zero near the center, down to about -6° at the rightmost. Auditory: nearly flat, close to 0° across the whole range. In all four rows the gray band sits directly on the data points, tracking them closely with no visible mismatch.
>
> **Column (c), variability (SD).** Black data points with error bars plot the SD of the localization response in degrees (y-axis, 0° to 6°) against stimulus location, overlaid with a gray shaded model-fit band. In every row the data trace a V-shaped (or shallow U-shaped) pattern, lowest near the central location and rising toward both extremes. The model band dips at the center too, but in the visual rows much less deeply than the data, so it sits above the central data points. High-Reliability: the band runs from about 2.8°–2.9° at the edges down to only about 2.5° at the center, above all data points, which dip to about 1.7°. Medium-Reliability: the band dips from about 3.3°–3.4° at the edges to about 3° at the center, while the data dip to about 1.9°. Low-Reliability: the band dips from about 4.5°–4.7° at the edges to about 3.5° at the center, above the three central data points (about 2.7°–3.3°) while matching the outermost points (about 4.4°–4.8°). Auditory: the band stays high and nearly level (about 5°) at every location except 0°, where it drops sharply to about 3°, close to the central data point (about 2.9°) but well above the data at ±5° (about 3.8° and 4.2°).

Fig G. The Exp-SingleGaussian parametric model fitted jointly on UV and UA data for all participants, in which the auditory range recalibration factor is a free parameter. The layout and color schemes are identical to Fig A.

> **Image description.** A 4-row × 3-column grid of plots comparing model predictions (shaded ribbons) to human localization data (points with vertical error bars), for three visual-reliability conditions and one auditory condition. A square bracket to the left of the first three rows groups them under the label "Visual" (High Reliability, Med. Reliability, Low Reliability, top to bottom); "Auditory" labels the fourth row separately. Column (a) shows response-distribution curves, column (b) shows response bias, and column (c) shows the SD of the localization response.
>
> **Column (a), response distributions.** In each row, seven colored curves (red, blue, green, purple, orange, brown, pink, ordered from the most negative to the most positive stimulus location) plot proportion of responses (y-axis ticks 0 to 0.6, with the tallest data points above the top tick) against reported stimulus location in degrees (x-axis, roughly -30° to 30°). Each curve overlays a shaded continuous band (the model's predicted distribution) with discrete data points and small vertical error bars (the empirical data). A legend at top right of the first row lists the seven visual stimulus-location bins as contiguous ranges (e.g. [-20,-14.3]°, [-14.3,-8.6]°, ... up to [14.3,20]°); a legend in the Auditory row lists seven discrete auditory locations from -15° to 15° in 5° steps. In every row the curves are ordered left-to-right matching their stimulus location, and the central (purple, near 0°) curve is by far the tallest and narrowest. Its data point (roughly 0.57–0.7 in proportion) sits slightly above the top of its model ribbon in every row, though the gap between the tall narrow empirical spike and the ribbon's peak (about 0.68 in the High-Reliability row, falling to about 0.52 in the Auditory row) is modest here, at most about 0.06. In the High-Reliability row the green and orange data peak nearer the center than their ribbons (data about 0.52–0.53 at ±3°, well above the ribbons there), while the more peripheral ribbons track their own data points closely. Curves broaden and increasingly overlap from High- to Medium- to Low-Reliability; in the Low-Reliability row the seven curves pile up so heavily near the center that individual ribbons and points are hard to disentangle, and the Auditory row's curves are about as broad as the Low-Reliability ones.
>
> **Column (b), bias.** Black data points with vertical error bars plot response bias in degrees (y-axis, -20° to 20°) against the same seven stimulus locations (x-axis, roughly -17° to 17°), with a dashed horizontal reference line at 0° and a thin gray shaded model-fit band running through the points. High-Reliability: a gentle decline from about +2° at the leftmost location to about -1° at the rightmost. Medium-Reliability: a similar, slightly steeper decline, from about +3° to -2°. Low-Reliability: a pronounced decline, from about +7° at the leftmost location, crossing zero near the center, down to about -6° at the rightmost. Auditory: nearly flat, close to 0° across the whole range. In all four rows the gray band sits directly on the data points, tracking them closely with no visible mismatch.
>
> **Column (c), variability (SD).** Black data points with error bars plot the SD of the localization response in degrees (y-axis, 0° to 6°) against stimulus location, overlaid with a gray shaded model-fit band. In every row the data trace a V-shaped (or shallow U-shaped) pattern, lowest near the central location and rising toward both extremes, and the model band dips and rises along with it rather than staying level. High-Reliability: the band descends from about 2.9° at the extremes to about 2.1° at the center, above most data points (which dip to about 1.7°). Medium-Reliability: the band dips to about 2.4° at the center from about 3.3°–3.4° at the edges, above the central data point (about 1.9°). Low-Reliability: the band dips to about 2.7° at the center from about 4.8°–5° at the edges, closely following the data. Auditory: the band dips to about 3.2° at the center from about 5.5° at the edges, lying slightly above the data both at the center (about 2.9°) and at the outermost locations (about 4.8°–5°).

Fig H. The Const-GaussianLaplace parametric model fitted jointly on UV and UA data for all participants, in which the auditory range recalibration factor is a free parameter. The layout and color schemes are identical to Fig A.

> **Image description.** A 4-row × 3-column grid of plots comparing model predictions (shaded ribbons) to human localization data (points with vertical error bars), for three visual-reliability conditions and one auditory condition. A square bracket to the left of the first three rows groups them under the label "Visual" (High Reliability, Med. Reliability, Low Reliability, top to bottom); "Auditory" labels the fourth row separately. Column (a) shows response-distribution curves, column (b) shows response bias, and column (c) shows the SD of the localization response.
>
> **Column (a), response distributions.** In each row, seven colored curves (red, blue, green, purple, orange, brown, pink, ordered from the most negative to the most positive stimulus location) plot proportion of responses (y-axis ticks 0 to 0.6, with the tallest data points above the top tick) against reported stimulus location in degrees (x-axis, roughly -30° to 30°). Each curve overlays a shaded continuous band (the model's predicted distribution) with discrete data points and small vertical error bars (the empirical data). A legend at top right of the first row lists the seven visual stimulus-location bins as contiguous ranges (e.g. [-20,-14.3]°, [-14.3,-8.6]°, ... up to [14.3,20]°); a legend in the Auditory row lists seven discrete auditory locations from -15° to 15° in 5° steps. In every row the curves are ordered left-to-right matching their stimulus location, and the central (purple, near 0°) curve is by far the tallest and narrowest. In the three visual rows its data point (roughly 0.62–0.7 in proportion) sits slightly above the top of its model ribbon (peaking around 0.6–0.63); in the Auditory row the ribbon peaks higher (about 0.65) than the data point (about 0.57). In the High-Reliability row the green and orange data peak nearer the center than their ribbons (data about 0.52–0.53 at ±3°, well above the ribbons there), while the more peripheral ribbons track their own data points closely. Curves broaden and increasingly overlap from High- to Medium- to Low-Reliability; in the Low-Reliability row the seven curves pile up so heavily near the center that individual ribbons and points are hard to disentangle, and the Auditory row's curves are about as broad as the Low-Reliability ones.
>
> **Column (b), bias.** Black data points with vertical error bars plot response bias in degrees (y-axis, -20° to 20°) against the same seven stimulus locations (x-axis, roughly -17° to 17°), with a dashed horizontal reference line at 0° and a thin gray shaded model-fit band running through the points. High-Reliability: a gentle decline from about +2° at the leftmost location to about -1° at the rightmost. Medium-Reliability: a similar, slightly steeper decline, from about +3° to -2°. Low-Reliability: a pronounced decline, from about +7° at the leftmost location, crossing zero near the center, down to about -6° at the rightmost. Auditory: nearly flat, close to 0° across the whole range. In all four rows the gray band sits directly on the data points, tracking them closely with no visible mismatch.
>
> **Column (c), variability (SD).** Black data points with error bars plot the SD of the localization response in degrees (y-axis, 0° to 6°) against stimulus location, overlaid with a gray shaded model-fit band. In every row the data trace a V-shaped (or shallow U-shaped) pattern, lowest near the central location and rising toward both extremes, and the model band dips and rises along with it rather than staying level. High-Reliability: the band descends from about 2.8°–2.9° at the extremes to about 2.2° at the center, above most data points (which dip to about 1.7°). Medium-Reliability: the band dips to about 2.4° at the center from about 3.3°–3.5° at the edges, above the central data point (about 1.9°). Low-Reliability: the band dips to about 2.7° at the center, matching the central data point, from about 4.8° at the edges, slightly above the outermost points (about 4.4°–4.8°); this row shows the closest fit. Auditory: the band dips sharply to about 2.6° at the center, reaching the lowest data point (about 2.9°), but elsewhere lies above the data, at about 5.3° at the edges (data about 4.9°) and about 4.5° at -5° (data about 3.8°).

Fig I. The Exp-TwoGaussians parametric model fitted jointly on UV and UA data for all participants, in which the auditory range recalibration factor is a free parameter. The layout and color schemes are identical to Fig A.

### Section C.2: Lifted-semiparametric model response distributions

> **Image description.** A large four-block panel array showing lifted-semiparametric model fits (assuming the MS causal-inference strategy) against human data for every task, organized as: a 4×3 grid of unisensory (UV/UA) response-distribution, bias, and SD plots (top left); a 3×2 grid of bisensory "same"-judgment (BC) curves (top right); a 3×3 grid of visual-bias (BV) plots (bottom left); and a matching 3×3 grid of auditory-bias (BA) plots (bottom right). Throughout, points with vertical error bars are empirical data (colored in the response-distribution column, black elsewhere) and colored or gray shaded bands are the model's predictive envelope.
>
> **Top-left block (unisensory tasks, columns a/b/c).** Rows are labeled, top to bottom, "High Reliability", "Med. Reliability", "Low Reliability" (all Visual, bracketed together on the far left under "Visual") and "Auditory". Column (a) plots "Proportion" (0–0.6) against "Reported stimulus location (°)" (–30 to 30): each row overlays seven bell-shaped response distributions, one per true-stimulus bin, using a fixed 7-color legend ("Stimulus loc.": red, blue, green, purple, orange, brown, pink; at the top right of the High Reliability and Auditory panels) given as ranges (e.g. [-20, -14.3]°, ..., [14.3,20]°) for the visual rows and as discrete values (-15° to 15° in 5° steps) for the Auditory row. Curves are narrowest and tallest (peak ≈0.6–0.7) near midline and broaden/shorten toward the periphery (peak ≈0.35–0.4 in High Reliability); overall width increases from High to Low Reliability (peripheral peaks fall to ≈0.2 with growing overlap between adjacent bins' curves in the Low Reliability row), and the Auditory distributions are about as broad as the Low Reliability ones (peripheral peaks ≈0.2, central peak ≈0.57). Column (b) plots "Bias (°)" (–20 to 20) against "Stimulus location (°)", with a dashed zero line: bias is nearly flat in High Reliability (within about ±2°) and becomes a clear negative slope in Low Reliability (roughly +7° at the leftmost location down to about –6° at the rightmost) — a central/regression bias that strengthens as reliability drops — while the Auditory row stays essentially flat at zero. Column (c) plots "SD of loc. response (°)" (0–6): the data dip to a minimum at the central location and rise toward the periphery in every row, the whole curve shifting upward as reliability drops (minimum ≈1.7°, ≈1.9° and ≈2.6°, periphery ≈2.5°, ≈3.5° and ≈4.7° for High, Med. and Low Reliability); the Auditory data run from ≈2.9° at center to ≈5° at the periphery. The gray model band lies above the data in the High and Med. Reliability rows (≈2.6–3.5° against ≈1.7–2.6° in High Reliability) and above the peripheral Auditory points (≈6° against ≈5°).
>
> **Top-right block (BC task).** Three rows (High/Med/Low Visual Reliability) × two columns ("$|s_\mathrm{A}+s_\mathrm{V}|$ in center", "$|s_\mathrm{A}+s_\mathrm{V}|$ in periphery"); y-axis "Proportion responding 'same'" (0–1); x-axis "Stimulus location disparity, $s_\mathrm{A}-s_\mathrm{V}$ (°)" (–30 to 30). Each panel shows nine data points forming a symmetric, peaked curve, maximal (≈0.9–0.95) at zero disparity and falling to ≈0.4–0.57 at ±16° and ≈0.14–0.3 at the largest disparities (±27°); the curves look much the same across the three reliability rows and the two columns. The gray band follows the data near the peak but lies above the outermost points in several panels (e.g. ≈0.25–0.35 against ≈0.18 in the High Visual Reliability periphery panel, ≈0.3–0.43 against ≈0.23 in the Low Visual Reliability center panel).
>
> **Bottom-left block (BV task, visual bias).** Three rows (High/Med/Low Visual Reliability) × three columns ("$(s_\mathrm{A}+s_\mathrm{V})$ on left", "in center", "on right"); y-axis "Visual bias (°)" (–15 to 15); x-axis disparity $s_\mathrm{A}-s_\mathrm{V}$ (–30 to 30; the data span about −13° to 20° in the left column, −30° to 28° in the center column, and −18° to 13° in the right column). Visual bias rises with disparity in every panel; the slope is shallow in High Visual Reliability (bias within about −2° to +2°) and becomes steep in Low Visual Reliability (spanning roughly –8° to +10°), showing visual estimates being pulled toward the auditory location increasingly as vision becomes less reliable. The bands follow the data closely.
>
> **Bottom-right block (BA task, auditory bias).** Same 3×3 layout; y-axis "Auditory bias (°)" (–15 to 15). Auditory bias falls with disparity (opposite sign from BV), steepest in High Visual Reliability (from about +10° to +13° at the most negative disparities to about −9° at the most positive) and shallower (roughly +5° to –4°) in Low Visual Reliability. In the High Visual Reliability row the bands fall short of the most extreme data points (e.g. ≈+4° to +5° against ≈+13° at the most negative disparity in the right-hand panel), and in the center column the band narrows to a thin line through zero at zero disparity.

Fig J. The lifted-semiparametric fits on all tasks, assuming the MS causal inference strategy. (A) UV+UA response distributions from data and model predictive samples. (B) BC response distributions from data and model predictive samples. Rows correspond to different visual reliability levels, and Cols are stratified based on whether the sum of the two true stimuli locations is below or above its median value across all BC trials. (C) BV response distributions from data and model predictive samples. Rows correspond to different visual reliability levels, and Cols are stratified based on where the sum of the two true stimuli locations lies with respect to its 100/3th and 200/3 percentiles across all BV trials. (D) BA response distributions from data and model predictive samples. Rows correspond to different visual reliability levels, and Cols are stratified based on where the sum of the two true stimuli locations lies with respect to its 100/3th and 200/3 percentiles across all BA trials.

> **Image description.** Same four-block layout as Fig J (4×3 unisensory response-distribution/bias/SD grid top left; 3×2 BC "same"-proportion grid top right; 3×3 BV visual-bias grid bottom left; 3×3 BA auditory-bias grid bottom right), here showing the lifted-semiparametric fit assuming the MA causal-inference strategy instead of MS.
>
> All axis ranges, row/column labels, the two "Stimulus loc." legends, and the data points with error bars are identical to Fig J, and the model bands are nearly indistinguishable from Fig J's in every block. As there, the gray SD band in the High and Med. Reliability rows of column (c) lies above the data (≈2.6–3.6° against ≈1.7–2.6° in High Reliability); the BC "same"-judgment curves peak at ≈0.9–0.95 at zero disparity and fall to ≈0.14–0.3 at ±27°, with the band above the outermost points in several panels; BV visual bias rises with disparity, shallow in High (within about ±2°) and steep in Low Visual Reliability (roughly −8° to +10°); and BA auditory bias falls with disparity, steepest for High and shallowest for Low Visual Reliability, with the High Visual Reliability bands falling short of the most extreme data points (≈+5° to +6° against ≈+10° to +13° at the most negative disparities).

Fig K. The lifted-semiparametric fits on all tasks, assuming the MA causal inference strategy. Subplot legends are identical to Fig J.

> **Image description.** Same four-block layout as Figs J/K (4×3 unisensory response-distribution/bias/SD grid top left; 3×2 BC "same"-proportion grid top right; 3×3 BV visual-bias grid bottom left; 3×3 BA auditory-bias grid bottom right), here showing the lifted-semiparametric fit assuming the PM causal-inference strategy.
>
> Axis ranges, row/column labels, the two "Stimulus loc." legends, and the data points with error bars are identical to Fig J. The unisensory block (columns a/b/c: response-distribution, bias, and SD panels) closely resembles Figs J/K, except that the SD band in the High and Med. Reliability rows of column (c) sits lower and closer to the data (High Reliability: ≈2.2° at center to ≈3.1° at the periphery, against data of ≈1.7–2.6°). In the BC block (top right, "Proportion responding 'same'" vs. stimulus disparity), the gray band's peak at zero disparity (≈0.88–0.9) is slightly below the data (≈0.93) in all six panels, and away from the peak the band flattens, so that at the largest disparities (±27°) it lies well above the data in most panels: ≈0.35–0.45 against ≈0.18 in the High Visual Reliability periphery panel, ≈0.42–0.5 against ≈0.23 in the Low Visual Reliability center panel, and ≈0.25–0.42 against ≈0.14–0.26 in both Med. Visual Reliability panels; only in the High Visual Reliability center and Low Visual Reliability periphery panels does it stay close to the outermost points. The BV (visual bias) and BA (auditory bias) blocks show the same trends as Fig J — visual bias rising with disparity, shallow in High and steep in Low Visual Reliability; auditory bias falling with disparity, steepest for High and shallowest for Low Visual Reliability, with the High Visual Reliability bands falling short of the most extreme auditory biases.

Fig L. The lifted-semiparametric fits on all tasks, assuming the PM causal inference strategy. Subplot notations are identical to Fig J.

### Section C.3: All-tasks parametric model response distributions

We will only visualize the two best models fitted on all tasks.

> **Image description.** A dense four-block panel array showing the Exp-GaussianLaplace-MA parametric model (one of the two best-fitting models) fitted jointly to all tasks: four stacked row-panels for the unisensory tasks, each with three columns of subplots (top left); a 3×2 grid of bisensory "same"-judgment curves (top right); and two 3×3 grids of bisensory visual-bias and auditory-bias curves (bottom, left and right respectively). Throughout, points with vertical error bars are human data, and colored or gray shaded bands (sometimes with a thin fit line running through the band) are the model's predictive envelope.
>
> **Top-left block (unisensory tasks).** Four row-panels, top to bottom: "High Reliability", "Med. Reliability", "Low Reliability" (all bracketed under "Visual"), and "Auditory". Column (a) plots "Proportion" (0–0.6) against "Reported stimulus location (°)" (–30 to 30): each row overlays seven bell-shaped curves, one per stimulus-location bin, using a 7-color legend ("Stimulus loc.": red, blue, green, purple, orange, brown, pink; at the top right of the High Reliability and Auditory panels), labeled as ranges (e.g. [-20,-14.3]°… [14.3,20]°) for the visual rows and as discrete values (-15° to 15° in 5° steps) for Auditory. The central (purple) curve is tallest (peak ≈0.65–0.7) and narrowest; peripheral curves (red, pink) are lower (peak ≈0.35–0.4 in High Reliability) and broader. Curves broaden and overlap increasingly from High to Low Reliability, so that in Low Reliability neighboring bins' curves overlap substantially, though the central peak stays comparatively tall (≈0.6). Column (b) plots "Bias (°)" (–20 to 20, dashed zero line) against "Stimulus location (°)": bias is nearly flat in High Reliability and develops a clear negative slope in Low Reliability (roughly +7° at the leftmost location to about –6° at the rightmost) — a central/regression bias that strengthens as reliability drops — while the Auditory row stays essentially flat near zero throughout. Column (c) plots "SD of loc. response (°)" (0–6): the data dip to a minimum at the central location (≈1.7° in High, ≈2.6° in Low Reliability) and rise toward the periphery in every visual row, the whole curve shifting upward as reliability drops (periphery ≈2.5° in High, ≈4.4–4.8° in Low Reliability); the Auditory data run from ≈2.9° at center to ≈5° at the periphery. Model bands follow the data in columns (a) and (b), though in column (a) they peak below the tallest points (e.g. ≈0.6 against ≈0.7 for the central bin in High Reliability); in column (c) the band lies above the data in the High and Med. Reliability rows (≈2.4–3.3° against ≈1.7–2.6° in High Reliability) and above the peripheral Auditory points (≈6° against ≈5°).
>
> **Top-right block (BC task).** Three rows (High/Med/Low Visual Reliability) × two columns ("$|s_\mathrm{A}+s_\mathrm{V}|$ in center", "$|s_\mathrm{A}+s_\mathrm{V}|$ in periphery"); y-axis "Proportion responding 'same'" (0–1); x-axis "Stimulus location disparity, $s_\mathrm{A}-s_\mathrm{V}$ (°)" (–30 to 30). Each panel shows nine data points forming a symmetric, peaked curve, maximal (≈0.9–0.95) at zero disparity and falling to ≈0.4–0.57 at ±16° and ≈0.14–0.3 at ±27°; the curves are similar across the three reliability rows and the two columns. The narrow gray model band tracks the data closely, deviating by up to ≈0.1–0.15 at the outermost points (e.g. ≈0.27–0.4 against ≈0.23 in the Low Visual Reliability center panel).
>
> **Bottom-left block (BV task, visual bias).** Three rows (High/Med/Low Visual Reliability) × three columns ("$(s_\mathrm{A}+s_\mathrm{V})$ on left", "in center", "on right"); y-axis "Visual bias (°)" (–15 to 15); x-axis disparity $s_\mathrm{A}-s_\mathrm{V}$ (–30 to 30). Visual bias increases with disparity in every panel; the slope is shallow in High Visual Reliability (bias within about −2° to +2°) and becomes steep in Low Visual Reliability (spanning roughly –8° to +10°) — visual estimates are pulled toward the auditory location increasingly as vision becomes less reliable. The three columns (left/center/right placement of the combined stimulus) show similar slopes with modest vertical offsets.
>
> **Bottom-right block (BA task, auditory bias).** Same 3×3 layout; y-axis "Auditory bias (°)" (–15 to 15). Auditory bias falls with disparity (opposite sign from BV), steepest in High Visual Reliability (from about +10° to +13° at the most negative disparities to about −9° at the most positive) and shallower (roughly +5° to –4°) in Low Visual Reliability. In the High Visual Reliability row the bands fall short of the most extreme data points (≈+5° against ≈+10° to +13°, and ≈−5° against ≈−9°); in the "in center" panels the band is nearly flat at the largest disparities on either side and narrows to a thin line through zero at zero disparity.

Fig M. The Exp-GaussianLaplace-MA parametric model fitted on all tasks. Subplot notations are identical to Fig J.

> **Image description.** The same four-block panel array and layout as Fig M, here showing the Exp-GaussianLaplace-PM parametric model fitted jointly to all tasks: four stacked row-panels for the unisensory tasks with three columns each (top left); a 3×2 grid of bisensory "same"-judgment curves (top right); and two 3×3 grids of bisensory visual-bias and auditory-bias curves (bottom, left and right). Points with vertical error bars are human data; colored or gray shaded bands (sometimes with a thin fit line through the band) are the model's predictive envelope. The data points are identical to those of Fig M; only the model bands differ.
>
> **Top-left block (unisensory tasks).** Four row-panels, top to bottom: "High Reliability", "Med. Reliability", "Low Reliability" (bracketed under "Visual"), and "Auditory". Column (a) plots "Proportion" (0–0.6) against "Reported stimulus location (°)" (–30 to 30): seven overlaid bell-shaped curves per row, one per stimulus-location bin, colored per a 7-color legend ("Stimulus loc.": red, blue, green, purple, orange, brown, pink; at the top right of the High Reliability and Auditory panels), given as ranges for the visual rows and discrete values (-15° to 15° in 5° steps) for Auditory. The central curve peaks highest (≈0.65–0.7) and narrowest; peripheral curves are lower (≈0.35–0.4 in High Reliability) and broader, with increasing overlap between neighboring bins from High to Low Reliability, though the central peak remains tall (≈0.6) even in Low Reliability. Column (b) plots "Bias (°)" (–20 to 20, dashed zero line): nearly flat in High Reliability, developing a clear negative slope in Low Reliability (roughly +7° to –6° across the range) — a central/regression bias that strengthens as reliability drops — while Auditory bias stays near zero throughout. Column (c) plots "SD of loc. response (°)" (0–6): the data dip to a minimum at the central location (≈1.7° in High, ≈2.6° in Low Reliability) and rise toward the periphery, shifting upward overall as reliability drops (periphery ≈2.5° in High, ≈4.4–4.8° in Low Reliability); the Auditory data run from ≈2.9° at center to ≈5° at the periphery. The SD band lies somewhat above the data in the High Reliability row (≈2.2–3.1° against ≈1.7–2.6°) and above the peripheral Auditory points (≈6° against ≈5°); elsewhere the bands track the data closely.
>
> **Top-right block (BC task).** Three rows (High/Med/Low Visual Reliability) × two columns ("$|s_\mathrm{A}+s_\mathrm{V}|$ in center", "$|s_\mathrm{A}+s_\mathrm{V}|$ in periphery"); y-axis "Proportion responding 'same'" (0–1); x-axis "Stimulus location disparity, $s_\mathrm{A}-s_\mathrm{V}$ (°)" (–30 to 30). Nine data points per panel form symmetric peaked curves, maximal (≈0.9–0.95) at zero disparity, falling to ≈0.4–0.57 at ±16° and ≈0.14–0.3 at ±27°, similar across the three reliability rows and the two columns. The gray band peaks slightly below the data (≈0.9) and at the largest disparities lies above the data in several panels (≈0.27–0.37 against ≈0.18 in the High Visual Reliability periphery panel, ≈0.33–0.45 against ≈0.23 in the Low Visual Reliability center panel).
>
> **Bottom-left block (BV task, visual bias).** Three rows (High/Med/Low Visual Reliability) × three columns ("$(s_\mathrm{A}+s_\mathrm{V})$ on left", "in center", "on right"); y-axis "Visual bias (°)" (–15 to 15); x-axis disparity (–30 to 30). Visual bias rises with disparity in every panel, shallow in High Visual Reliability (within about −2° to +2°) and steep in Low Visual Reliability (roughly –8° to +10°) — increasing capture of the visual estimate toward the auditory location as vision becomes less reliable — with similar slopes and modest vertical offsets across the three placement columns.
>
> **Bottom-right block (BA task, auditory bias).** Same 3×3 layout; y-axis "Auditory bias (°)" (–15 to 15). Auditory bias falls with disparity (opposite sign from BV), steepest in High Visual Reliability (from about +10° to +13° at the most negative disparities to about −9° at the most positive) and shallower (roughly +5° to –4°) in Low Visual Reliability. As in Fig M, the High Visual Reliability bands fall short of the most extreme data points (≈+5° to +6° against ≈+10° to +13°), and in the "in center" panels the band is nearly flat at the largest disparities on either side and narrows to a thin line through zero at zero disparity.

Fig N. The Exp-GaussianLaplace-PM parametric model fitted on all tasks. Subplot notations are identical to Fig J.

## Section D: Model response distributions, individual participants

We provide here individual-level model visualizations for the best parametric models fitted on either the unisensory data (Exp-GaussianLaplace) or all the data (Exp-GaussianLaplace-PM).

### Section D.1: Unisensory data fit

> **Image description.** A 4×4 grid of small line-and-scatter plots, one per participant ("Subject 1" through "Subject 15", filling the grid left-to-right/top-to-bottom), with the final cell replaced by a color legend. Every panel shares the same axes — "Stimulus location (°)" on x (–20 to 20) and "Bias (°)" on y (–20 to 20, with a dashed horizontal line at zero) — and the same four-series encoding: dark blue = Visual (high reliability), medium blue = Visual (med. reliability), light blue = Visual (low reliability), green = Auditory. Filled dots are binned human data and lines of matching color are the corresponding Exp-GaussianLaplace model's predictions (fit jointly on UV and UA data).
>
> **Shared pattern.** In most participants the three visual bias curves slope downward through zero (positive bias at the leftmost stimulus locations, negative at the rightmost), i.e. a central/regression bias, and this slope is shallowest for high reliability (dark blue) and steepest for low reliability (light blue). The auditory bias (green) differs in direction across participants: it slopes downward like the visual curves in Subjects 1, 8, 9 and 12, upward in Subjects 2, 3, 6, 7, 10 and 14, and stays close to zero in Subjects 4, 5, 11, 13 and 15.
>
> **Between-participant differences.** The upward auditory slope is most pronounced in Subject 3, whose auditory data rise from about −7° to −9° on the left to about +1° to +3.5° on the right (model line from about −6° to +6°) while the visual curves still slope gently downward, so the green line crosses the blue ones. Subject 4 is the flattest participant overall, with all four curves staying close to zero across the full stimulus range. Subjects 1, 5 and 8 show the steepest visual bias, with low-reliability data running from about +10° to +14° at the leftmost location to about −8° to −13° at the rightmost. Model lines generally follow the data points closely; the largest deviations are in the auditory data of Subjects 3 and 10 (in Subject 10 the auditory points stay positive, up to ≈+5.5°, while the line runs from about −2.5° to +2.5°).

Fig O. Bias of UV and UA responses of each human participant and their Exp-GaussianLaplace parametric models fitted jointly on UV and UA data. Each participant is plotted in one subfigure. Different colors correspond to different visual reliability levels and sensory modalities. The trial-binning process is identical to Fig 2B in the main text, but with points denoting human data and lines denoting model predictions.

> **Image description.** A 4×4 grid of small line-and-scatter plots, one per participant ("Subject 1" through "Subject 15"), with the final cell replaced by a color legend. Every panel shares the same axes — "Stimulus location (°)" on x (–20 to 20) and "SD of location response (°)" on y (0–8) — and the same four-series encoding as Fig O: dark blue = Visual (high reliability), medium blue = Visual (med. reliability), light blue = Visual (low reliability), green = Auditory. Filled dots are binned human data and lines of matching color are the corresponding Exp-GaussianLaplace model's predictions (fit jointly on UV and UA data).
>
> **Shared pattern.** Nearly every panel shows a V-shaped (or shallow-U) noise function: SD is lowest near the central stimulus location (roughly 1–3°) and rises toward the periphery (roughly 3–6°); Subject 3, whose curves are nearly flat, is the main exception. The three visual curves are consistently ordered by reliability across most participants and locations, with high reliability (dark blue) lowest and low reliability (light blue) highest. At the periphery the green (auditory) curve lies above the high- and medium-reliability visual curves in every participant; relative to the low-reliability curve it is clearly higher in Subjects 2, 3, 7, 8, 10 and 12, similar in Subjects 1, 5, 6, 11, 13 and 15, and lower in Subjects 4, 9 and 14.
>
> **Between-participant differences.** The auditory-visual separation varies widely. In Subjects 2, 3, and 10, auditory SD is dramatically higher than any visual condition (reaching roughly 7–8.5° versus roughly 2–6° for vision); in Subject 3 the auditory line stays at ≈7.4–8.1° across the full range, and the visual lines are also comparatively flat. Subject 10 shows the most extreme peripheral auditory noise, exceeding 8° in the periphery while its central dip only reaches about 4.5°. By contrast, in Subjects 11 and 15 the auditory curve sits much closer to the visual curves, only modestly higher. The low-reliability visual line reaches its highest values (≈6–6.3°) in Subjects 3, 9 and 13. Model lines generally track the data, but in several participants (e.g. Subjects 9, 12, 13 and 15) the high-reliability visual points lie about 1–2° below the model line, and a few isolated points fall far from their line (e.g. a high-reliability point at ≈5.7° at the central location in Subject 12).

Fig P. SD of UV and UA responses of each human participant and their Exp-GaussianLaplace parametric models fitted jointly on UV and UA data. Each participant is plotted in one subfigure. Different colors correspond to different sensory modalities; different transparencies correspond to visual reliability levels. The trial-binning process is identical to Fig 2C in the main text, but with points denoting human data and lines denoting model predictions.

### Section D.2: All-tasks fit

> **Image description.** A 4×4 grid of small line-and-scatter plots, one per participant ("Subject 1" through "Subject 15"), with a legend in the sixteenth (bottom-right) cell. Every subplot shares the same x-axis, "Stimulus location (°)" (−20 to 20, ticks at −20, −10, 0, 10, 20), and y-axis, "Bias (°)" (−20 to 20), with a dashed horizontal reference line at 0.
>
> **Shared encoding.** Each subplot overlays four conditions: three shades of blue for visual stimuli at high, medium, and low reliability (darkest to lightest), and green for auditory stimuli. Filled circles are the participant's mean response bias per stimulus-location bin; solid lines of matching color are the fitted Exp-GaussianLaplace-PM model (fit jointly on all tasks) evaluated at the same locations.
>
> **Patterns and between-participant differences.** In most panels the visual lines slope downward across the range — positive bias (~2–14°) at the leftmost (negative) locations falling to negative bias (roughly −3 to −13°) at the rightmost (positive) locations — consistent with a central-tendency (regression-to-the-mean) effect. This downward slope is generally steepest for low-reliability visual (lightest blue) and shallowest for high-reliability visual (darkest blue), e.g. in Subjects 1, 5, 8, 10, and 15, where the three blue lines fan out with low-reliability reaching the largest bias magnitude at the extremes. The auditory (green) line differs in direction across participants: it slopes downward like the visual lines in Subjects 1, 8, 9 and 12, upward in Subjects 2, 3, 6, 7, 10 and 14, and stays close to zero in Subjects 4, 5, 11, 13 and 15. The upward slope is most pronounced in Subject 3 (line from about −6.5° to +6°), the reverse of the visual lines in the same panel. Subject 4 stands out for having all four lines nearly flat and clustered near zero, and Subject 11 is almost as flat (all lines within about ±3°). The fitted lines track the visual data closely; the largest misfits are in the auditory data of Subjects 3 and 10 (in Subject 3 the points at −10° and −5° lie near −8° against a line at about −3° to −4.5°; in Subject 10 the auditory points stay positive, about +1.5° to +5.5°, while the line runs from about −2.5° to +2.5°).

Fig Q. Mean UV and UA responses of each human participant and their Exp-GaussianLaplace-PM parametric models fitted on all tasks. The visualization process is identical to Fig O.

> **Image description.** A 4×4 grid of small line-and-scatter plots, one per participant ("Subject 1" through "Subject 15"), with a legend in the sixteenth cell. All subplots share the x-axis "Stimulus location (°)" (−20 to 20) and the y-axis "SD of location response (°)" (0 to 8, occasionally exceeded slightly by data points).
>
> **Shared encoding.** As in Fig Q, three shades of blue encode visual stimuli at high, medium, and low reliability (darkest to lightest) and green encodes auditory stimuli. Circles are each participant's response SD per stimulus-location bin; solid lines are the fitted Exp-GaussianLaplace-PM model (all-tasks fit).
>
> **Patterns and between-participant differences.** Most curves are V-shaped (cup-shaped): SD is lowest near the central location (0°) and rises toward both peripheral ends, though in Subjects 2, 3 and 12 the curves are comparatively flat. The three visual lines are usually ordered by reliability — high-reliability visual (darkest blue) lowest (SD roughly 1–4°), low-reliability visual (lightest blue) highest (up to 5–7° at the periphery), with medium reliability in between — though in Subjects 1 and 8 the three visual lines nearly coincide and cross. The auditory (green) line's position relative to the visual lines varies across participants: it lies clearly above all three in Subjects 2, 3, 7, 8, 10 and 12, slightly above them in Subject 1, close to the low-reliability visual line in Subjects 4, 5, 6, 11, 13, 14 and 15, and below it in Subject 9; in several participants (e.g. Subjects 7, 8, 14, 15) it dips at the central location to about 1.6–2°, as low as or below the visual lines there. Subjects 2 and 3 have the highest auditory lines, which stay high across the whole range (≈6.3–7.9° in Subject 2, flat at ≈7.5° in Subject 3); in Subject 2 the line misses the data's central dip (≈2.3° at 0°). Subject 10 shows the largest peripheral auditory variability, its line reaching ≈8.3–8.5° at the periphery with a dip to about 4.3° at center. The fitted lines generally reproduce the V-shape and the reliability ordering, but in several participants (e.g. Subjects 2, 9, 12, 13 and 15) the high-reliability visual points lie about 1–2° below the model line.

Fig R. SD of UV and UA responses of each human participant and their Exp-GaussianLaplace-PM parametric models fitted on all tasks. The visualization process is identical to Fig P.

> **Image description.** A 4×4 grid of plots, one per participant ("Subject 1" through "Subject 15"), with a legend in the sixteenth cell. Each participant's cell is itself split by a vertical divider line into two side-by-side sub-panels, Center on the left (purple) and Periphery on the right (orange), identified by color through the legend. All sub-panels share the y-axis "Proportion responding 'same'" (0 to 1) and the x-axis "Stimulus location disparity, $s_\mathrm{A}-s_\mathrm{V}$ (°)" (roughly −27 to 27, ticks at −20, 0, 20).
>
> **Shared encoding.** Six conditions are color-coded: three purple shades for the Center sub-panel at high, medium, and low visual reliability (darkest to lightest), and three orange shades for the Periphery sub-panel at the same three reliability levels. Circles are each participant's proportion of "same"-source responses per disparity bin; solid lines are the fitted Exp-GaussianLaplace-PM model (all-tasks fit).
>
> **Patterns and between-participant differences.** Every sub-panel in every participant shows a single peaked, roughly bell-shaped (tent-like) curve: the proportion of "same" responses is highest near zero disparity and falls as the disparity magnitude grows, in both Center and Periphery. How far it falls varies across participants. Subjects 4 and 11 show unusually sharp, narrow tuning: the proportion drops close to 0 outside a fairly small disparity window in both sub-panels. In Subjects 5, 12 and 13, by contrast, it only falls to about 0.4–0.7 at the largest disparities. Subject 8 is an outlier with much broader, flatter model curves, which stay between ≈0.55 and ≈0.86 across the whole disparity range in both Center and Periphery, while its data at the largest disparities scatter widely (from ≈0 to ≈0.85). Subject 15's curves peak lower than most others, topping out around 0.75–0.8 rather than near 1, in both sub-panels. For most participants the Center and Periphery curves are similar in shape and peak height. The three visual-reliability lines within each sub-panel nearly coincide near zero disparity, but in the Center sub-panel the low-reliability line often stays higher at the largest disparities (e.g. Subjects 1, 5, 6 and 9). The fitted lines follow the data near the peak; at the largest disparities the points scatter more, and in some participants (e.g. Subject 3) many of them lie well below the lines.

Fig S. BC responses of each human participant and their Exp-GaussianLaplace-PM parametric models fitted jointly on all tasks. Each participant is plotted in one subfigure. Different colors correspond to whether trial-specific $|s_\mathrm{A} + s_\mathrm{V}|$ is below (center) or above (periphery) its median value across all trials. The center-periphery-stratification and trial-binning processes are identical to Fig 6A in the main text, but now with points denoting human data and lines denoting model predictions.

> **Image description.** A 4×4 grid of plots, one per participant ("Subject 1" through "Subject 15"), with a legend in the sixteenth cell. Each participant's cell is split by two vertical divider lines into three side-by-side sub-panels, Left (orange), Center (purple), and Right (brown), identified by color through the legend (the three groups formed by stratifying trials on $s_\mathrm{A}+s_\mathrm{V}$). All sub-panels share the y-axis "Visual bias (°)" (−15 to 15) and the x-axis "Stimulus location disparity, $s_\mathrm{A}-s_\mathrm{V}$ (°)" (ticks at −20, 0, 20 in each sub-panel; the data span roughly −15° to 20° in Left, −28° to 28° in Center, and −20° to 15° in Right).
>
> **Shared encoding.** Nine conditions are color-coded: three orange shades for Left at high/medium/low visual reliability (darkest to lightest), three purple shades for Center, and three brown/tan shades for Right. Circles are each participant's mean visual localization bias per disparity bin; solid lines are the fitted Exp-GaussianLaplace-PM model (all-tasks fit).
>
> **Patterns and between-participant differences.** In almost every sub-panel and participant, bias increases (slopes upward) with disparity — visual location reports are pulled toward the paired auditory location, becoming more positive as $s_\mathrm{A}-s_\mathrm{V}$ grows. The steepness of this slope depends clearly on visual reliability: the lightest shade in each color family (low reliability) shows the steepest slope, in several participants spanning nearly the full −15° to +15° range in the Center sub-panel (e.g. Subjects 1, 5, 6, 8), while the most saturated shade (high reliability) is comparatively flat, staying close to 0° in most participants (e.g. Subjects 2, 3, 6, 7, 9, 11, 13, 14). Subject 4 is an outlier with all nine lines nearly flat and near zero. Subjects 1, 8 and 12 show the steepest high-reliability slopes, most visibly in their Center sub-panels, where the high-reliability lines run from about −4° to −5° up to about +4° to +5°. The Left, Center, and Right sub-panels show broadly similar slope patterns within a given participant, though the Center sub-panel has the widest disparity range and, in several participants, the steepest low-reliability slope; in some participants the Left points lie mostly above zero and the Right points mostly below (e.g. Subjects 1, 8, 12). The low-reliability points scatter most around the fitted lines.

Fig T. BV responses of each human participant and their Exp-GaussianLaplace-PM parametric models fitted jointly on all tasks. Each participant is plotted in one subfigure. Different colors correspond to whether trial-specific $(s_\mathrm{A} + s_\mathrm{V})$ is stratified into the left, center, or right group. The trial-stratification and trial-binning processes are identical to Fig 6B in the main text, but now with points denoting human data and lines denoting model predictions.

> **Image description.** A 4×4 grid of plots, one per participant ("Subject 1" through "Subject 15"), with a legend in the sixteenth cell. Each participant's cell is split by two vertical divider lines into three side-by-side sub-panels, Left (orange), Center (purple), and Right (brown), identified by color through the legend (the three groups formed by stratifying trials on $s_\mathrm{A}+s_\mathrm{V}$). All sub-panels share the y-axis "Auditory bias (°)" (−15 to 15) and the x-axis "Stimulus location disparity, $s_\mathrm{A}-s_\mathrm{V}$ (°)" (ticks at −20, 0, 20 in each sub-panel; the data span roughly −15° to 20° in Left, −28° to 28° in Center, and −20° to 15° in Right). This figure mirrors Fig T but for the auditory estimate in the BA task.
>
> **Shared encoding.** Nine conditions are color-coded: three orange shades for Left at high/medium/low visual reliability (darkest to lightest), three purple shades for Center, and three brown/tan shades for Right. Circles are each participant's mean auditory localization bias per disparity bin; solid lines are the fitted Exp-GaussianLaplace-PM model (all-tasks fit).
>
> **Patterns and between-participant differences.** In most sub-panels and participants, bias decreases (slopes downward) with disparity — the opposite direction from the visual bias in Fig T — consistent with the auditory estimate being pulled toward the paired visual location as $s_\mathrm{A}-s_\mathrm{V}$ grows; the Center sub-panels of Subjects 2, 4, 6 and 11 are exceptions, with lines that zigzag or stay nearly flat. Unlike Fig T, the three visual-reliability lines within a spatial group are often closely clustered and overlapping rather than clearly fanned out, so the slope here depends less consistently on visual reliability across participants. Data points are also noticeably noisier relative to the fitted lines than in Fig T, with substantial scatter in several participants (e.g. Subjects 2, 4, 6, and 10), and in a few sub-panels (e.g. Subject 2's Left and Center) the points show little discernible trend at all. Subject 8 stands out with an unusually steep decline in the Center sub-panel, its high-reliability line running from the top of the axis (≈+15°) to about −15°, steeper than any other condition in the figure. Subjects 5, 8, 12 and 13 show comparatively clean, consistent negative slopes across all three spatial groups. Auditory biases reach ±10–15° in many participants (e.g. Subjects 3, 5, 8, 12 and 13). The left, center and right groups behave similarly to one another within a given participant.

Fig U. BA responses of each human participant and their Exp-GaussianLaplace-PM parametric models fitted jointly on all tasks. Each participant is plotted in one subfigure. Different colors correspond to whether trial-specific $(s_\mathrm{A} + s_\mathrm{V})$ is stratified into the left, center, or right group. The trial-stratification and trial-binning processes are identical to Fig 6B in the main text, but now with points denoting human data and lines denoting model predictions.

### Section D.3: Individual fitted parameters

For each participant and the best-fitting Exp-GaussianLaplace-PM model (fitted over unisensory and bisensory data), we report their individual fitted parameters below. For other models, each participant’s fitted parameters are available in the Github repository.

Table J. Individual fitted parameters for the Exp-GaussianLaplace-PM model.

| Sub. | $\sigma_{0,\mathrm{V}}$ | $k_{1,\mathrm{V}}$ | $k_{2,\mathrm{V}}$ | $\alpha_\mathrm{med}$ | $\alpha_\mathrm{low}$ | $\sigma_s$ | $\lambda$ | $b$ | $w$ | $\sigma_\mathrm{motor}$ | $\sigma_{0,\mathrm{A}}$ | $k_{1,\mathrm{A}}$ | $k_{2,\mathrm{A}}$ | $\rho_\mathrm{A}$ | $p_\mathrm{same}$ | $\beta_\mathrm{V}$ | $\beta_\mathrm{A}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ***1*** | 0.37 | 1.41 | 2.15 | 1.69 | 2.65 | 4.19 | 0.00 | 1.38 | 0.93 | 0.25 | 2.37 | 5.68 | 0.03 | 1.35 | 0.77 | 1.33 | 1.52 |
| ***2*** | 0.86 | 8.45 | 0.01 | 1.63 | 3.65 | 6.64 | 0.01 | 0.12 | 0.42 | 0.82 | 5.37 | 0.00 | 9.11 | 2.08 | 0.91 | 1.10 | 1.16 |
| ***3*** | 0.97 | 1.41 | 0.18 | 1.89 | 3.02 | 11.63 | 0.01 | 14.75 | 0.00 | 1.51 | 5.09 | 0.08 | 9.08 | 1.68 | 0.94 | 1.00 | 1.86 |
| ***4*** | 0.76 | 1.14 | 0.07 | 1.77 | 3.07 | 20.53 | 0.00 | 1.68 | 0.63 | 0.25 | 1.82 | 5.42 | 0.03 | 1.07 | 0.89 | 1.21 | 1.25 |
| ***5*** | 1.42 | 2.40 | 0.06 | 1.92 | 3.95 | 5.14 | 0.00 | 4.59 | 0.60 | 0.25 | 3.52 | 6.52 | 0.02 | 1.25 | 0.98 | 1.03 | 1.55 |
| ***6*** | 0.49 | 1.03 | 0.40 | 1.93 | 4.95 | 10.42 | 0.00 | 0.71 | 0.45 | 0.25 | 3.35 | 4.70 | 0.02 | 1.50 | 0.97 | 1.06 | 1.29 |
| ***7*** | 0.98 | 19.25 | 0.00 | 1.59 | 2.80 | 7.09 | 0.00 | 1.49 | 0.97 | 0.25 | 2.35 | 1.74 | 0.24 | 1.64 | 0.94 | 1.17 | 1.44 |
| ***8*** | 1.12 | 2.64 | 0.06 | 1.57 | 3.10 | 4.56 | 0.00 | 1.03 | 1.00 | 0.25 | 2.96 | 2.66 | 0.27 | 1.94 | 0.79 | 1.17 | 1.90 |
| ***9*** | 0.75 | 2.11 | 0.09 | 1.54 | 3.08 | 17.4 | 0.01 | 2.29 | 0.68 | 0.25 | 3.28 | 0.65 | 0.37 | 0.96 | 0.87 | 1.18 | 1.27 |
| ***10*** | 0.38 | 1.98 | 0.67 | 1.41 | 2.56 | 9.64 | 0.01 | 2.92 | 0.97 | 0.25 | 4.15 | 4.80 | 0.04 | 1.82 | 0.86 | 1.07 | 1.46 |
| ***11*** | 0.14 | 1.80 | 0.46 | 1.28 | 1.83 | 6.67 | 0.00 | 3.74 | 0.99 | 0.25 | 2.90 | 0.61 | 0.06 | 1.24 | 0.81 | 1.13 | 1.30 |
| ***12*** | 1.07 | 1.65 | 10.00 | 1.18 | 1.98 | 6.00 | 0.01 | 5.09 | 0.21 | 0.29 | 4.15 | 6.22 | 0.02 | 1.25 | 0.94 | 1.25 | 1.37 |
| ***13*** | 0.89 | 0.68 | 0.48 | 1.71 | 3.49 | 16.02 | 0.00 | 3.61 | 0.84 | 0.25 | 4.05 | 0.08 | 7.95 | 1.20 | 0.92 | 1.11 | 2.05 |
| ***14*** | 0.48 | 1.28 | 1.74 | 1.68 | 3.44 | 0.10 | 0.00 | 5.65 | 0.42 | 0.25 | 2.32 | 6.42 | 0.02 | 1.32 | 0.84 | 0.98 | 1.73 |
| ***15*** | 2.58 | 0.48 | 0.02 | 1.08 | 1.87 | 7.13 | 0.00 | 0.20 | 0.24 | 0.25 | 1.32 | 2.28 | 0.48 | 1.31 | 0.71 | 1.08 | 2.36 |

---

*Transcribed from the PDF text layer and corrected with LLMs; text, equations, tables, and figure descriptions may contain mistakes.*
