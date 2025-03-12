# pubs-llms 📄: Our Publications for Humans and Machines

A repository of academic publications from the [Machine and Human Intelligence Group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) converted to LLM-friendly text-only Markdown format.

## Overview

This repository contains research papers converted to plain text with AI-generated descriptions of figures, making them easily accessible for large language model (LLM) analysis and interactions, for both humans and machines.

The full list of papers is available [below](#Publications).

### Content

For practical usage, each paper is split into three separate files:

1. **Main Text** - The core content of the paper ([example](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized.md))
2. **Appendix** - Supplementary materials, when available ([example](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md))
3. **Backmatter** - References, acknowledgments, checklists, and other auxiliary content rarely fed to an LLM ([example](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md))

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
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md)

### 2024

- **Improving robustness to corruptions with multiplicative weight perturbations**<br>
  Trinh T, Heinonen M, Acerbi L & Kaski S<br>
  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/trinh2024improving_backmatter.md)

- **Preferential Normalizing Flows**<br>
  Mikkola P, Acerbi L & Klami A<br>
  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/mikkola2024preferential_backmatter.md)

- **Amortized Bayesian Experimental Design for Decision-Making**<br>
  Huang D, Guo Y, Acerbi L & Kaski S<br>
  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*<br>
  [main](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized.md) | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_appendix.md) | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/huang2024amortized_backmatter.md)

