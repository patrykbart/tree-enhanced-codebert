# Tree-Enhanced CodeBERTa

This repository contains the implementation of **Tree-Enhanced CodeBERTa**, an improved Transformer-based model for source code representation that integrates **hierarchical positional embeddings** derived from **Abstract Syntax Trees (ASTs)**. This work is part of an anonymous ACL submission.

---

## 📜 Overview

Transformer-based models like CodeBERTa excel at capturing semantic relationships in source code but struggle to represent its **hierarchical structure**. This repository introduces **Tree-Based Positional Embeddings**, which encode **depth and sibling index information** from ASTs into the Transformer architecture.

We evaluate Tree-Enhanced CodeBERTa on:
- **Masked Language Modeling (MLM)**
- **Clone Detection**

Our results show that integrating **hierarchical embeddings** improves structural understanding and model performance while maintaining efficiency.
