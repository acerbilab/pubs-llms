# pubs-llms 📄: Our Publications for Humans and Machines

A repository of academic publications from the [Machine and Human Intelligence Group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) converted to LLM-friendly text-only Markdown format.

## Overview

This repository contains research papers converted to plain text with AI-generated descriptions of figures, making them easily accessible for large language model (LLM) analysis and interactions, for both humans and machines.

The full list of papers is available [below](#Publications).

### Content

For practical usage, each paper is split into three separate files:

| **Part**    | **Description** | **Example** |
|------------------|---------------|-----------------|
| **Main Text**    | The core content of the paper. | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_main.md) |
| **Appendix**     | Supplementary materials, when available. | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md) |
| **Backmatter**   | References, acknowledgments, and other auxiliary content rarely fed to an LLM. | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md) |

### Usage Guide

- **Quick usage:** Navigate to the paper of interest, click "Copy raw file" on GitHub, paste the content (or excerpts) into your LLM chat to ask questions about the paper.
- **Luigi's usage:** Include relevant papers in project repositories for use with advanced LLM assistants. Luigi uses Athanor (an in-house LLM research and coding assistant), but other options include [Aider](https://aider.chat/), [Cline](https://cline.bot/), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), and keep growing.

### Technical Details

The paper-to-Markdown conversion process uses [paper2llm](https://lacerbi.github.io/paper2llm/), with [Mistral OCR](https://mistral.ai/news/mistral-ocr) for text and table extraction and [Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/flash/) for image-to-text descriptions.

#### Note: This repository is a work in progress.

---


## Publications

### 2025

- **Amortized Probabilistic Conditioning for Optimization, Simulation and Inference**<br>
  Chang PE, Loka N, Huang D, Remes U, Kaski S & Acerbi L<br>
  `AISTATS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md)

### 2024

- **Improving robustness to corruptions with multiplicative weight perturbations**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_backmatter.md)

- **Amortized Bayesian Experimental Design for Decision-Making**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_backmatter.md)

- **Preferential Normalizing Flows**<br>
  Mikkola P, Acerbi L & Klami A<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_backmatter.md)

- **Amortized Bayesian Workflow (Extended Abstract)**<br>
  Schmitt M, Li C, Vehtari A, Acerbi L, Burkner P & Radev ST<br>
  `NeurIPS Workshop` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_backmatter.md)

- **Amortized Decision-Aware Bayesian Experimental Design**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  `NeurIPS Workshop` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_backmatter.md)

- **Input-gradient space particle inference for neural network ensembles**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  `ICLR` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_backmatter.md)

- **PyBADS: Fast and robust black-box optimization in Python**<br>
  Singh G & Acerbi L<br>
  `JOSS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_backmatter.md)

### 2023

- **Practical Equivariances via Relational Conditional Neural Processes**<br>
  Huang D, Hausmann M, Remes U, Clarté G, Luck KS, Kaski S & Acerbi L<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023practical_backmatter.md)

- **Learning Robust Statistics for Simulation-based Inference under Model Misspecification**<br>
  Huang D, Bharti A, Souza A, Acerbi L & Kaski S<br>
  `NeurIPS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2023learning_backmatter.md)

- **Online Simulator-Based Experimental Design for Cognitive Model Selection**<br>
  Alex, Aushev e, Putkonen A, Clarte G, Ch SH, ramouli, Acerbi L, Kaski S & Howes A<br>
  `Comput Brain Behav` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/aushev2023online_backmatter.md)

- **PyVBMC: Efficient Bayesian inference in Python**<br>
  Huggins B, Li C, Tobaben M, Aarnos MJ & Acerbi L<br>
  `JOSS` | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huggins2023pyvbmc_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huggins2023pyvbmc_backmatter.md)

