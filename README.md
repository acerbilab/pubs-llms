# 📚 pubs-llms: Our Publications for Humans and Machines

A repository of academic publications from the [Machine and Human Intelligence Group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) converted to LLM-friendly text-only Markdown format.

## Overview

This repository contains research papers converted to plain text with AI-generated descriptions of figures, making them easily accessible for large language model (LLM) analysis and interactions, for both humans and machines.

The full list of papers is available [below](#Publications).

### Content

For practical usage, each paper is available in full as well as split into three parts:

| **Part**       | **Description**                                                                | **Example**                                                                                                  |
| -------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **Main Text**  | The core content of the paper.                                                 | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_main.md)             |
| **Backmatter** | References, acknowledgments, and other auxiliary content rarely fed to an LLM. | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md) |
| **Appendix**   | Supplementary materials, when available.                                       | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md)     |
| **Full Text**  | Combined version with all parts in a single document.                          | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_full.md)             |

### Usage Guide

- **Quick usage:** Navigate to the paper of interest, click "Copy raw file" on GitHub, paste the full content or individual parts and excerpts into your LLM chat to ask questions about the paper.
- **Luigi's usage:** Include relevant papers in project repositories for use with advanced LLM assistants. Luigi uses Athanor (an in-house LLM research and coding assistant), but other options include [Aider](https://aider.chat/), [Cline](https://cline.bot/), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), and keep growing.

### Technical Details

The paper-to-Markdown conversion process uses [paper2llm](https://lacerbi.github.io/paper2llm/), with [Mistral OCR](https://mistral.ai/news/mistral-ocr) for text and table extraction and [Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/flash/) for image-to-text descriptions.

### Disclaimer

<details>
<summary>Important notes about conversion accuracy.</summary>

- Papers have been converted automatically with minimal human intervention.
- OCR models have now become extremely robust, and vision models show practical utility in image understanding, but occasional inaccuracies may occur.
- **Errors** may take the form of missing sentences near non-standard page formatting, typos in equations or tables, or image descriptions missing or misrepresenting parts of the figure.
- Please **report such mistakes** by raising a [GitHub issue](https://github.com/acerbilab/pubs-llms/issues).

For non-critical applications, we consider that the benefit of having LLM-friendly access to research papers outweigh the potential inaccuracies, which generally do not affect the gist of the paper. As usual, double-check key assumptions and results.

</details>

---


## Publications

### 2025

- **Amortized Probabilistic Conditioning for Optimization, Simulation and Inference**<br>
  Chang PE, Loka N, Huang D, Remes U, Kaski S & Acerbi L<br>
  `AISTATS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_full.md)

- **Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations**<br>
  Li C, Huggins B, Mikkola P & Acerbi L<br>
  `AABI` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/li2025normalizing_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/li2025normalizing_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/li2025normalizing_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/li2025normalizing_full.md)

### 2024

- **Improving robustness to corruptions with multiplicative weight perturbations**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_full.md)

- **Amortized Bayesian Experimental Design for Decision-Making**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_full.md)

- **Preferential Normalizing Flows**<br>
  Mikkola P, Acerbi L & Klami A<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_full.md)

- **Amortized Bayesian Workflow (Extended Abstract)**<br>
  Schmitt M, Li C, Vehtari A, Acerbi L, Burkner P & Radev ST<br>
  `NeurIPS Workshop` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_full.md)

- **Amortized Decision-Aware Bayesian Experimental Design**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  `NeurIPS Workshop` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_full.md)

- **Input-gradient space particle inference for neural network ensembles**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  `ICLR` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_full.md)

- **PyBADS: Fast and robust black-box optimization in Python**<br>
  Singh G & Acerbi L<br>
  `JOSS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_backmatter.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_full.md)

### 2023

- **Practical Equivariances via Relational Conditional Neural Processes**<br>
  Huang D, Hausmann M, Remes U, Clarté G, Luck KS, Kaski S & Acerbi L<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_full.md)

- **Learning Robust Statistics for Simulation-based Inference under Model Misspecification**<br>
  Huang D, Bharti A, Souza A, Acerbi L & Kaski S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_full.md)

- **Online Simulator-Based Experimental Design for Cognitive Model Selection**<br>
  Aushev A, Putkonen A, Clarte G, Chandramouli SH, Acerbi L, Kaski S & Howes A<br>
  `Comput Brain Behav` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_full.md)

- **PyVBMC: Efficient Bayesian inference in Python**<br>
  Huggins B, Li C, Tobaben M, Aarnos MJ & Acerbi L<br>
  `JOSS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huggins2023pyvbmc_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huggins2023pyvbmc_backmatter.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/huggins2023pyvbmc_full.md)

### 2022

- **Parallel MCMC Without Embarrassing Failures**<br>
  de Souza DARMA, Mesquita D, Kaski S & Acerbi L<br>
  `AISTATS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/desouza2022parallel_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/desouza2022parallel_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/desouza2022parallel_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/desouza2022parallel_full.md)

- **Tackling covariate shift with node-based Bayesian neural networks**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  `ICML` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2022tackling_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2022tackling_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2022tackling_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2022tackling_full.md)

### 2021

- **Uncertainty is maintained and used in working memory**<br>
  Yoo AH, Acerbi L & Ma WJ<br>
  `JOV` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/yoo2021uncertainty_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/yoo2021uncertainty_backmatter.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/yoo2021uncertainty_full.md)

### 2020

- **Variational Bayesian Monte Carlo with Noisy Likelihoods**<br>
  Acerbi L<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2020variational_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2020variational_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2020variational_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2020variational_full.md)

- **Dynamic allocation of limited memory resources in reinforcement learning**<br>
  Patel N, Acerbi L & Pouget A<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/patel2020dynamic_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/patel2020dynamic_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/patel2020dynamic_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/patel2020dynamic_full.md)

- **Unbiased and Efficient Log-Likelihood Estimation with Inverse Binomial Sampling**<br>
  van Opheusden B, Acerbi L & Ma WJ<br>
  `PLoS Comput Biol` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/vanopheusden2020unbiased_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/vanopheusden2020unbiased_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/vanopheusden2020unbiased_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/vanopheusden2020unbiased_full.md)

- **The role of sensory uncertainty in simple contour integration**<br>
  Zhou Y, Acerbi L & Ma WJ<br>
  `PLoS Comput Biol` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/zhou2020role_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/zhou2020role_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/zhou2020role_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/zhou2020role_full.md)

### 2019

- **An Exploration of Acquisition and Mean Functions in Variational Bayesian Monte Carlo**<br>
  Acerbi L<br>
  `AABI` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2019exploration_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2019exploration_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2019exploration_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2019exploration_full.md)

- **Human online adaptation to changes in prior probability**<br>
  Norton EH, Acerbi L, Ma WJ & Landy MS<br>
  `PLoS Comput Biol` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/norton2019human_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/norton2019human_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/norton2019human_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/norton2019human_full.md)

### 2018

- **Variational Bayesian Monte Carlo**<br>
  Acerbi L<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018variational_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018variational_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018variational_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018variational_full.md)

- **Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception**<br>
  Acerbi L, Dokka K, Angelaki DE & Ma WJ<br>
  `PLoS Comput Biol` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018bayesian_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018bayesian_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018bayesian_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2018bayesian_full.md)

### 2017

- **Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search**<br>
  Acerbi L & Ma WJ<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017practical_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017practical_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017practical_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017practical_full.md)

- **Target Uncertainty Mediates Sensorimotor Error Correction**<br>
  Acerbi L, Sethu V & Wolpert DM<br>
  `PLoS ONE` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017target_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017target_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017target_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2017target_full.md)

### 2014

- **A Framework for Testing Identifiability of Bayesian Models of Perception**<br>
  Acerbi L, Ma WJ & Vijayakumar S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014framework_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014framework_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014framework_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014framework_full.md)

- **On the Origins of Suboptimality in Human Probabilistic Inference**<br>
  Acerbi L, Vijayakumar S & Wolpert DM<br>
  `PLoS Comput Biol` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014origins_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014origins_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014origins_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2014origins_full.md)

### 2012

- **Internal Representations of Temporal Statistics and Feedback Calibrate Motor-Sensory Interval Timing**<br>
  Acerbi L, Wolpert DM & Vijayakumar S<br>
  `PLoS Comput Biol` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2012internal_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2012internal_backmatter.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2012internal_appendix.md) | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/acerbi2012internal_full.md)

