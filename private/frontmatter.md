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
| **Appendix**   | Supplementary materials, when available.                                       | [appendix](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_appendix.md)     |
| **Backmatter** | References, acknowledgments, and other auxiliary content rarely fed to an LLM. | [backmatter](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_backmatter.md) |
| **Full Text**  | Combined version with all parts in a single document.                          | [full](https://github.com/acerbilab/pubs-llms/blob/main/publications/chang2025amortized_full.md)             |

### Usage Guide

- **Quick usage:** Navigate to the paper of interest, click "Copy raw file" on GitHub, paste the content (or excerpts) into your LLM chat to ask questions about the paper. Use the full version when you want the complete paper in a single file.
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
