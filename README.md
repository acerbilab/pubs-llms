# pubs-llms 📄: Our Publications for Humans and Machines

A repository of academic publications from the [Machine and Human Intelligence Group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) converted to LLM-friendly text-only Markdown format.

## Overview

This repository contains research papers converted to plain text with AI-generated descriptions of figures, making them easily accessible for large language model (LLM) analysis and interactions, for both humans and machines.

The full list of papers is available [below](#Publications).

### Content

For practical usage, each paper is split into three separate files:

| **Document Part**    | **Description** | **Example Link** |
|------------------|---------------|-----------------|
| **Main Text**    | The core content of the paper. | [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_main.md) |
| **Appendix**     | Supplementary materials, when available. | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md) |
| **Backmatter**   | References, acknowledgments, checklists, and other auxiliary content rarely fed to an LLM. | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md) |

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
  *28th Int. Conf. on Artificial Intelligence & Statistics (AISTATS 2025)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md)

### 2024

- **Improving robustness to corruptions with multiplicative weight perturbations**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_backmatter.md)

- **Amortized Bayesian Experimental Design for Decision-Making**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_backmatter.md)

- **Preferential Normalizing Flows**<br>
  Mikkola P, Acerbi L & Klami A<br>
  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_backmatter.md)

- **Amortized Bayesian Workflow (Extended Abstract)**<br>
  Schmitt M, Li C, Vehtari A, Acerbi L, Burkner P & Radev ST<br>
  *NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/schmitt2024amortized_backmatter.md)

- **Amortized Decision-Aware Bayesian Experimental Design**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  *NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024bamortized_backmatter.md)

- **Input-gradient space particle inference for neural network ensembles**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  *International Conference on Learning Representations (ICLR 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_main.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024input_backmatter.md)

- **PyBADS: Fast and robust black-box optimization in Python**<br>
  Singh G & Acerbi L<br>
  *Journal of Open Source Software*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_main.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/singh2024pybads_backmatter.md)

