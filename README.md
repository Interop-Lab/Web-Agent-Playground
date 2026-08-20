# Web-Agent-Playground

> Adaptive Optimization of Safe Access Strategies for Web Agents in Black-Box Protection Environments (SHSOC)

This repository implements the Safety-Constrained Hybrid Soft Optimal Control (**SHSOC**) method, along with the baseline methods and experimental environments used for comparison.

## Background & Problem

When web agents perform automated tasks on the web, they frequently run into "black-box" protection mechanisms deployed by target websites. The detection rules, thresholds, and internal state of such protection systems are completely invisible to the agent, which can only infer risk indirectly through a binary signal: whether an action succeeds. Once flagged as anomalous, the current session is terminated immediately and irreversibly, and the risk of being blocked is accumulated over the entire history of actions. Existing countermeasures mostly rely on hand-crafted heuristic rules (fixed delays, simulated human reading speed, etc.); they do not generalize across websites and cannot adaptively balance access efficiency against safety.

This project formulates the problem as a partially observable constrained Markov decision process and addresses its three key structural properties — **irreversible termination on failure**, **hybrid action space** (discrete operations + continuous waiting time), and **partial observability** of the environment — with a unified reinforcement learning solution.

## The SHSOC Method

SHSOC integrates three core mechanisms:

- **Risk-Backtracking Reward**: Ties the cost of being blocked to the cumulative gain accumulated before the block. Once a ban is triggered, all previously earned rewards are retroactively cancelled, so that aggressive strategies can no longer boost expected return by raising pre-ban gains. This turns a trajectory-level safety constraint into an unconstrained return-maximization problem.
- **Dynamic Target Entropy**: Introduces a risk beacon that smoothly estimates the recent ban rate and accordingly adjusts the target entropy along both the discrete-operation and continuous-time dimensions. When risk rises, exploration increases to escape high-risk regions; when risk falls, exploration tightens to improve efficiency.
- **History Behavior Encoder**: An LSTM-based encoder-decoder that compresses variable-length action histories into a fixed-dimensional latent state, enabling the agent to infer the current ban risk even when the protection system's internal state is unobservable. The encoder is pre-trained offline and frozen, making it independent of any specific protection mechanism and reusable across environments.

Experiments show that, without any manual parameter tuning, SHSOC improves the success rate by 8%–9% and reduces task completion time by 9%–10% over the best hand-tuned baseline.