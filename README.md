# Tree-Enhanced CodeBERTa

This repository contains the implementation of **Tree-Enhanced CodeBERTa**, an improved Transformer-based model for source code representation that integrates **hierarchical positional embeddings** derived from **Abstract Syntax Trees (ASTs)**.

This is the official repository for the paper:
**[Seamlessly Integrating Tree-Based Positional Embeddings into Transformer Models for Source Code Representation](https://aclanthology.org/2025.xllm-1.10/)**
*Patryk Bartkowiak and Filip Graliński*
Published at the **1st Joint Workshop on Large Language Models and Structure Modeling (XLLM 2025)**.

---

## 📜 Overview

Transformer-based models like CodeBERTa excel at capturing semantic relationships in source code but struggle to represent its **hierarchical structure**. This repository introduces **Tree-Based Positional Embeddings**, which encode **depth and sibling index information** from ASTs into the Transformer architecture.

We evaluate Tree-Enhanced CodeBERTa on:
- **Masked Language Modeling (MLM)**
- **Clone Detection**

Our results show that integrating **hierarchical embeddings** improves structural understanding and model performance while maintaining efficiency.

---

## 📚 Citation

```bibtex
@inproceedings{bartkowiak-gralinski-2025-seamlessly,
    title = "Seamlessly Integrating Tree-Based Positional Embeddings into Transformer Models for Source Code Representation",
    author = "Bartkowiak, Patryk  and Grali{\'n}ski, Filip",
    booktitle = "Proceedings of the 1st Joint Workshop on Large Language Models and Structure Modeling (XLLM 2025)",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.xllm-1.10/",
    doi = "10.18653/v1/2025.xllm-1.10",
    pages = "91--98",
    ISBN = "979-8-89176-286-2",
    abstract = "Transformer-based models have demonstrated significant success in various source code representation tasks. Nonetheless, traditional positional embeddings employed by these models inadequately capture the hierarchical structure intrinsic to source code, typically represented as Abstract Syntax Trees (ASTs). To address this, we propose a novel tree-based positional embedding approach that explicitly encodes hierarchical relationships derived from ASTs, including node depth and sibling indices. These hierarchical embeddings are integrated into the transformer architecture, specifically enhancing the CodeBERTa model. We thoroughly evaluate our proposed model through masked language modeling (MLM) pretraining and clone detection fine-tuning tasks. Experimental results indicate that our Tree-Enhanced CodeBERTa consistently surpasses the baseline model in terms of loss, accuracy, F1 score, precision, and recall, emphasizing the importance of incorporating explicit structural information into transformer-based representations of source code."
}
```
