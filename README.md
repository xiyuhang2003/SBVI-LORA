# Code for ICML 2026 Paper: "Spectral Bridge Variational Inference"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc)

This repository contains the **Minimal Reproducible Example (MRE)** for our ICML 2026 paper: 
**"Spectral Bridge Variational Inference: Dynamic LoRA via Bures-Wasserstein Gradient Flows"**.

## Abstract
Parameter-Efficient Fine-Tuning (PEFT) is essential for adapting Large Language Models, yet existing methods struggle to balance capacity with computational efficiency. Standard approaches enforce rigid low-rank constraints, while dynamic alternatives incur significant memory overheads. To resolve this, we propose **Spectral Bridge Variational Inference (SBVI)**, a geometric framework reformulating LoRA as a continuous Wasserstein gradient flow on the manifold of Gaussian measures. Instead of fixing ranks at initialization, SBVI governs singular value evolution via a stochastic differential equation driven by thermodynamic competition between task gradients and adaptive entropic friction. This induces a spectral bifurcation that automatically prunes noise modes while amplifying signal-rich components, discovering an optimal layer-wise rank distribution. 

## About This Repository (MRE)
This repository is explicitly designed to be a lightweight, highly readable validation of the mathematical algorithms detailed in Algorithm 1 of our paper. It eschews heavy boilerplate and complex engineering wrappers, allowing reviewers and researchers to immediately audit the core logic of the factorized structure and the thermodynamic EMA friction penalty.

> **Disclaimer**: This repository serves as the minimal reproducible version for the ICML 2026 camera-ready submission. A comprehensive, highly-optimized codebase—featuring advanced training efficiency, extended task support, and broader framework integrations—will be open-sourced concurrently with our upcoming journal extension.

## Repository Structure
- `sbvi.py`: The core `SBVILinear` PyTorch module implementing the $M = B A^\top$ factorization, EMA spectral friction, and dynamic regularization, along with `merge_weights` for zero-latency inference.
- `train.py`: A Hugging Face `Trainer` implementation demonstrating how to inject SBVI into LLaMA-2-7B, accumulate the thermodynamic loss, and track the `Effective Rank` dropping in real-time.
- `requirements.txt`: Minimal dependencies.


