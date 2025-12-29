# SAPIENT: Learning Negotiation Through Strategic Ambiguity and Preference Calibration

**Technical Specification v1.0**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
   - 2.1 [Motivation](#motivation)
   - 2.2 [Core Contributions](#core-contributions)
   - 2.3 [Related Work](#related-work)
3. [Problem Formulation](#problem-formulation)
   - 3.1 [Negotiation Setting](#negotiation-setting)
   - 3.2 [Information Structure](#information-structure)
   - 3.3 [Strategic Ambiguity Space](#strategic-ambiguity-space)
   - 3.4 [Preference Calibration Objective](#preference-calibration-objective)
4. [Theoretical Framework](#theoretical-framework)
   - 4.1 [Game-Theoretic Analysis](#game-theoretic-analysis)
   - 4.2 [Information Revelation Mechanisms](#information-revelation-mechanisms)
   - 4.3 [Strategic Ambiguity Optimization](#strategic-ambiguity-optimization)
   - 4.4 [Preference State Formalization](#preference-state-formalization)
5. [Architecture Design](#architecture-design)
   - 5.1 [Multi-Agent System Architecture](#multi-agent-system-architecture)
   - 5.2 [State Space Formalization](#state-space-formalization)
   - 5.3 [Action Space Specification](#action-space-specification)
   - 5.4 [Reward Function Formulation](#reward-function-formulation)
   - 5.5 [Communication Protocol](#communication-protocol)
6. [Algorithms](#algorithms)
   - 6.1 [Main Training Loop](#main-training-loop)
   - 6.2 [Information Revelation Policy Learning](#information-revelation-policy-learning)
   - 6.3 [Preference Calibration Mechanism](#preference-calibration-mechanism)
   - 6.4 [Multi-Agent Coordination Algorithm](#multi-agent-coordination-algorithm)
   - 6.5 [Convergence Analysis](#convergence-analysis)
   - 6.6 [Complexity Analysis](#complexity-analysis)
7. [Mathematical Foundations](#mathematical-foundations)
   - 7.1 [Information-Theoretic Metrics](#information-theoretic-metrics)
   - 7.2 [Utility Functions and Ambiguity Constraints](#utility-functions-and-ambiguity-constraints)
   - 7.3 [Fairness and Efficiency Measures](#fairness-and-efficiency-measures)
   - 7.4 [Formal Theorems](#formal-theorems)
8. [Experimental Setup](#experimental-setup)
   - 8.1 [Benchmark Domains](#benchmark-domains)
   - 8.2 [Baseline Methods](#baseline-methods)
   - 8.3 [Evaluation Metrics](#evaluation-metrics)
   - 8.4 [Hyperparameters and Training Details](#hyperparameters-and-training-details)
   - 8.5 [Dataset Specifications](#dataset-specifications)
9. [Implementation Details](#implementation-details)
   - 9.1 [Framework Recommendations](#framework-recommendations)
   - 9.2 [Network Architectures](#network-architectures)
   - 9.3 [Training Procedures](#training-procedures)
   - 9.4 [Logging and Monitoring](#logging-and-monitoring)
10. [Limitations and Future Work](#limitations-and-future-work)
    - 10.1 [Known Limitations](#known-limitations)
    - 10.2 [Scalability Considerations](#scalability-considerations)
    - 10.3 [Extensions and Open Problems](#extensions-and-open-problems)
11. [Implementation Notes](#implementation-notes)
12. [References](#references)

---

## 1. Executive Summary

SAPIENT (Strategic Ambiguity and Preference Inference through Efficient Negotiation Training) is a novel multi-agent reinforcement learning framework designed to learn effective negotiation strategies through the strategic management of information revelation and preference calibration. Unlike traditional negotiation approaches that rely on complete information or purely adversarial dynamics, SAPIENT learns to strategically control the ambiguity of communicated information to facilitate efficient preference discovery while maintaining strategic advantage.

The framework addresses a fundamental challenge in automated negotiation: how to balance the revelation of private information (necessary for discovering mutually beneficial agreements) with the maintenance of strategic ambiguity (necessary for avoiding exploitation). SAPIENT achieves this through:

1. **Formal modeling** of strategic ambiguity as a continuous control variable in negotiation
2. **Multi-agent reinforcement learning** algorithms that learn optimal information revelation policies
3. **Preference calibration mechanisms** that infer opponent preferences from ambiguous signals
4. **Game-theoretic guarantees** on convergence and incentive compatibility

This specification provides complete mathematical formulations, algorithmic details, and implementation guidelines for researchers and practitioners seeking to build or extend SAPIENT-based negotiation agents.

---

## 2. Introduction

### 2.1 Motivation

Negotiation is a fundamental mechanism for multi-agent coordination, resource allocation, and conflict resolution. However, effective negotiation requires agents to navigate a delicate balance:

- **Information revelation**: Sharing information about preferences enables discovery of Pareto-optimal agreements
- **Strategic concealment**: Revealing too much information allows opponents to exploit strategic advantage

Human negotiators naturally manage this trade-off through strategic ambiguity—the deliberate use of imprecise, vague, or incomplete communication that reveals some information while maintaining flexibility. Current automated negotiation systems typically fall into two extremes:

1. **Complete information approaches**: Assume full transparency, vulnerable to exploitation
2. **Zero-sum approaches**: Assume adversarial dynamics, missing cooperative opportunities

SAPIENT introduces a principled framework for learning the optimal level of strategic ambiguity across different negotiation contexts, enabling agents to discover efficient agreements without unnecessary information loss.

### 2.2 Core Contributions

1. **Formal framework for strategic ambiguity**: Mathematical formalization of ambiguity as a controllable parameter in negotiation communication
2. **Multi-agent RL algorithms**: Novel training procedures that learn both strategic ambiguity policies and preference calibration mechanisms
3. **Theoretical guarantees**: Convergence analysis and game-theoretic properties of learned policies
4. **Empirical validation**: Comprehensive evaluation across multiple negotiation domains
5. **Human alignment**: Demonstrations that SAPIENT agents negotiate in ways aligned with human preferences and norms

### 2.3 Related Work

**Automated Negotiation**: Classical approaches include the Monotonic Concession Protocol, heuristic-based agents (e.g., AgentK, Caduceus), and optimization-based methods. These typically require hand-crafted strategies and domain-specific knowledge.

**Multi-Agent Reinforcement Learning**: Recent work on QMIX, MAPPO, and other MARL algorithms provides foundations for learning coordinated behaviors. However, these methods are rarely applied to negotiation with strategic information control.

**Game Theory**: Concepts from Bayesian games, signaling games, and mechanism design inform our theoretical framework, particularly the work on information revelation in strategic settings.

**Preference Elicitation**: Machine learning approaches to preference learning typically assume non-strategic settings. SAPIENT extends these to strategic environments where agents have incentives to misrepresent preferences.

---

## 3. Problem Formulation

### 3.1 Negotiation Setting

We model negotiation as a sequential game between $N$ agents, where $N \geq 2$. The negotiation proceeds over a finite time horizon $T$ and concerns the allocation of resources or agreement on outcomes from a feasible set.

**Definition 3.1 (Negotiation Game)**: A negotiation game is a tuple $\mathcal{G} = \langle \mathcal{N}, \mathcal{X}, \mathcal{U}, \Theta, T \rangle$ where:

- $\mathcal{N} = \{1, 2, \ldots, N\}$ is the set of agents
- $\mathcal{X}$ is the space of possible agreements (outcomes)
- $\mathcal{U} = \{u_i: \mathcal{X} \times \Theta_i \to \mathbb{R}\}_{i \in \mathcal{N}}$ is the set of utility functions
- $\Theta = \prod_{i=1}^N \Theta_i$ is the space of preference types, where $\Theta_i$ represents agent $i$'s private type
- $T \in \mathbb{N}$ is the negotiation deadline

Each agent $i$ has a private type $\theta_i \in \Theta_i$ drawn from a prior distribution $P(\theta_i)$. The type $\theta_i$ parameterizes agent $i$'s utility function $u_i(x, \theta_i)$ over outcomes $x \in \mathcal{X}$.

**Agreement Space**: For concreteness, consider multi-issue negotiation where:

$$\mathcal{X} = \prod_{j=1}^M \mathcal{X}_j$$

where $\mathcal{X}_j$ represents the space of values for issue $j$. For example, in a bilateral negotiation over price and delivery time, $\mathcal{X} = \mathcal{X}_{\text{price}} \times \mathcal{X}_{\text{delivery}}$.

**Utility Structure**: We assume quasi-linear utility functions:

$$u_i(x, \theta_i) = \sum_{j=1}^M w_{ij}(\theta_i) \cdot v_{ij}(x_j, \theta_i)$$

where $w_{ij}(\theta_i) \geq 0$ is the weight (importance) of issue $j$ to agent $i$, and $v_{ij}(x_j, \theta_i)$ is the value derived from outcome $x_j$ on issue $j$.

**Negotiation Protocol**: At each time step $t \in \{1, 2, \ldots, T\}$:

1. A designated agent (determined by protocol) proposes an outcome $x_t \in \mathcal{X}$
2. Other agents respond with accept/reject decisions
3. If all agents accept, negotiation concludes with agreement $x_t$
4. If any agent rejects, negotiation continues to $t+1$
5. If $t = T$ is reached without agreement, all agents receive disagreement utility $u_i^{\text{dis}}$

### 3.2 Information Structure

The key innovation in SAPIENT is the explicit modeling of information revelation through strategic ambiguity.

**Definition 3.2 (Information State)**: At time $t$, each agent $i$ maintains:

- **Private information**: True type $\theta_i \in \Theta_i$
- **Public history**: Sequence of all proposals and responses $h_t = (x_1, r_1, \ldots, x_{t-1}, r_{t-1})$
- **Belief state**: Probability distribution $b_i^t(\theta_{-i})$ over opponent types $\theta_{-i} = (\theta_j)_{j \neq i}$

**Information Asymmetry**: The fundamental challenge is that agent $i$ does not observe $\theta_j$ for $j \neq i$. Instead, $i$ must infer opponent preferences from:

1. **Proposals**: The outcomes $x_t$ proposed by opponents
2. **Responses**: Accept/reject decisions
3. **Ambiguous signals**: Explicit communication with controlled precision (introduced by SAPIENT)

**Definition 3.3 (Ambiguous Signal)**: An ambiguous signal is a random variable $s_i^t \in \mathcal{S}$ that provides partial information about $\theta_i$. Formally, $s_i^t$ is generated by a signaling function:

$$s_i^t = \sigma_i(\theta_i, \alpha_i^t, \epsilon_t)$$

where:
- $\alpha_i^t \in [0, 1]$ is the **ambiguity level** chosen by agent $i$
- $\epsilon_t$ is random noise
- $\sigma_i: \Theta_i \times [0, 1] \times \mathcal{E} \to \mathcal{S}$ is the signaling function

**Ambiguity Semantics**:
- $\alpha_i^t = 0$: Maximum precision (full revelation of $\theta_i$)
- $\alpha_i^t = 1$: Maximum ambiguity (signal is independent of $\theta_i$)
- $\alpha_i^t \in (0, 1)$: Partial revelation with controlled noise

**Signal Space**: The signal space $\mathcal{S}$ depends on the negotiation domain. For multi-issue negotiation:

$$\mathcal{S} = \prod_{j=1}^M \mathcal{S}_j$$

where $\mathcal{S}_j$ represents signals about preferences over issue $j$. Common signal types include:

1. **Preference intervals**: $s_j \in \{[\ell_j, u_j] \subseteq \mathcal{X}_j\}$ (range of acceptable values)
2. **Importance rankings**: $s_j \in \{\text{high}, \text{medium}, \text{low}\}$ (qualitative importance)
3. **Utility estimates**: $s_j \in \mathbb{R}$ (noisy utility signal)

### 3.3 Strategic Ambiguity Space

We formalize strategic ambiguity as a continuous control problem.

**Definition 3.4 (Ambiguity Policy)**: An ambiguity policy for agent $i$ is a mapping:

$$\pi_i^{\alpha}: \mathcal{H}_t \times \Theta_i \to \Delta([0, 1])$$

where $\mathcal{H}_t$ is the space of histories up to time $t$, and $\Delta([0, 1])$ is the space of probability distributions over ambiguity levels.

**Signaling Function**: We employ a parametric signaling function that interpolates between full revelation and complete ambiguity:

$$\sigma_i(\theta_i, \alpha_i^t, \epsilon_t) = (1 - \alpha_i^t) \cdot f(\theta_i) + \alpha_i^t \cdot \epsilon_t$$

where:
- $f: \Theta_i \to \mathcal{S}$ is a deterministic encoding of type to signal
- $\epsilon_t \sim P_{\epsilon}$ is random noise (e.g., uniform over $\mathcal{S}$)

**Mutual Information**: The information content of signal $s_i^t$ about type $\theta_i$ is quantified by mutual information:

$$I(\theta_i; s_i^t \mid \alpha_i^t) = H(\theta_i) - H(\theta_i \mid s_i^t, \alpha_i^t)$$

where $H(\cdot)$ is entropy and $H(\cdot \mid \cdot)$ is conditional entropy.

**Proposition 3.1**: For the linear signaling function above, mutual information is monotonically decreasing in ambiguity level $\alpha_i^t$:

$$\frac{\partial I(\theta_i; s_i^t \mid \alpha_i^t)}{\partial \alpha_i^t} \leq 0$$

*Proof sketch*: As $\alpha_i^t$ increases, the signal $s_i^t$ depends less on $\theta_i$ and more on random noise $\epsilon_t$, reducing the correlation and thus the mutual information. $\square$

### 3.4 Preference Calibration Objective

The goal of each agent is to maximize expected utility while managing information revelation.

**Definition 3.5 (Agent Objective)**: Agent $i$'s objective is to maximize:

$$J_i(\pi_i) = \mathbb{E}_{\theta, h \sim \pi}\left[\sum_{t=1}^T \gamma^{t-1} r_i^t(h_t, \theta_i, \alpha_i^t)\right]$$

where:
- $\pi = (\pi_1, \ldots, \pi_N)$ is the joint policy of all agents
- $\gamma \in (0, 1)$ is a discount factor
- $r_i^t$ is the reward at time $t$

**Reward Structure**: The reward at time $t$ consists of multiple components:

$$r_i^t = \mathbb{1}_{[\text{agreement at } t]} \cdot u_i(x_t, \theta_i) + (1 - \mathbb{1}_{[\text{agreement at } t]}) \cdot c_i(\alpha_i^t)$$

where:
- $u_i(x_t, \theta_i)$ is the utility from agreement
- $c_i(\alpha_i^t)$ is a cost/benefit of ambiguity level
- $\mathbb{1}_{[\text{agreement at } t]}$ is an indicator for whether agreement is reached at time $t$

**Ambiguity Cost Function**: We model the trade-off of ambiguity through:

$$c_i(\alpha_i^t) = -\lambda_{\text{reveal}} \cdot (1 - \alpha_i^t) + \lambda_{\text{explore}} \cdot I_t$$

where:
- $\lambda_{\text{reveal}} > 0$ penalizes information revelation
- $\lambda_{\text{explore}} > 0$ rewards exploration leading to better agreements
- $I_t$ is an estimate of progress toward efficient agreement

**Preference Calibration**: Agent $i$ maintains a belief state $b_i^t(\theta_{-i})$ over opponent types and updates it using Bayesian inference:

$$b_i^{t+1}(\theta_{-i}) \propto P(o^t \mid \theta_{-i}, h_t) \cdot b_i^t(\theta_{-i})$$

where $o^t$ is the observation at time $t$ (proposals, responses, signals).

**Social Welfare Objective**: For the system as a whole, we aim to maximize social welfare:

$$W(\pi) = \mathbb{E}_{\theta, h \sim \pi}\left[\sum_{i=1}^N u_i(x^*, \theta_i) - u_i^{\text{dis}}\right]$$

where $x^*$ is the agreed outcome (if any).

---

## 4. Theoretical Framework

### 4.1 Game-Theoretic Analysis

We analyze SAPIENT negotiations through the lens of Bayesian games with communication.

**Definition 4.1 (Extended Negotiation Game)**: The SAPIENT framework induces an extended game $\tilde{\mathcal{G}} = \langle \mathcal{N}, \mathcal{X}, \mathcal{U}, \Theta, \mathcal{S}, T \rangle$ where agents can send signals from $\mathcal{S}$ in addition to making proposals.

**Strategy Space**: A complete strategy for agent $i$ consists of:

1. **Signaling strategy**: $\pi_i^{\alpha}: \mathcal{H}_t \times \Theta_i \to \Delta([0, 1])$
2. **Proposal strategy**: $\pi_i^x: \mathcal{H}_t \times \Theta_i \to \Delta(\mathcal{X})$
3. **Response strategy**: $\pi_i^r: \mathcal{H}_t \times \mathcal{X} \times \Theta_i \to \Delta(\{\text{accept}, \text{reject}\})$

**Definition 4.2 (Bayesian Nash Equilibrium)**: A strategy profile $\pi^* = (\pi_1^*, \ldots, \pi_N^*)$ is a Bayesian Nash Equilibrium (BNE) if for all $i \in \mathcal{N}$:

$$\mathbb{E}_{\theta_i \sim P(\theta_i)}[J_i(\pi_i^*, \pi_{-i}^* \mid \theta_i)] \geq \mathbb{E}_{\theta_i \sim P(\theta_i)}[J_i(\pi_i, \pi_{-i}^* \mid \theta_i)]$$

for all alternative strategies $\pi_i$.

**Theorem 4.1 (Existence of Equilibrium)**: Under standard regularity conditions (compact strategy spaces, continuous payoffs), the extended negotiation game $\tilde{\mathcal{G}}$ admits at least one Bayesian Nash Equilibrium.

*Proof*: Follows from standard fixed-point arguments (Glicksberg's theorem) for Bayesian games with finite horizons. $\square$

**Equilibrium Properties**: We characterize properties of equilibria in SAPIENT negotiations:

**Proposition 4.1 (Information Monotonicity)**: In equilibrium, higher ambiguity levels lead to less efficient agreements on average:

$$\mathbb{E}[W(\pi) \mid \alpha] \text{ is decreasing in } \bar{\alpha} = \frac{1}{NT}\sum_{i,t} \alpha_i^t$$

*Intuition*: More ambiguity means less information revelation, reducing the ability of agents to discover Pareto-optimal agreements.

**Proposition 4.2 (Strategic Ambiguity)**: In equilibrium with heterogeneous agents, there exist preference types $\theta_i \in \Theta_i$ for which optimal ambiguity level $\alpha_i^* > 0$ (i.e., full revelation is suboptimal).

*Intuition*: Agents with preferences that could be exploited optimally maintain some ambiguity to avoid unfavorable agreements.

### 4.2 Information Revelation Mechanisms

We analyze the dynamics of information revelation in SAPIENT negotiations.

**Definition 4.3 (Revelation Mechanism)**: A revelation mechanism is a mapping $\mathcal{M}: \mathcal{S}^N \to \Delta(\mathcal{X})$ that takes signals from all agents and produces a distribution over outcomes.

**Incentive Compatibility**: A key question is whether agents have incentives to truthfully reveal preferences.

**Definition 4.4 (Bayesian Incentive Compatibility)**: A mechanism $\mathcal{M}$ is Bayesian incentive compatible (BIC) if for all agents $i$ and all types $\theta_i, \theta_i' \in \Theta_i$:

$$\mathbb{E}_{\theta_{-i}, x \sim \mathcal{M}(f(\theta_i), s_{-i})}[u_i(x, \theta_i)] \geq \mathbb{E}_{\theta_{-i}, x \sim \mathcal{M}(f(\theta_i'), s_{-i})}[u_i(x, \theta_i)]$$

where $s_{-i}$ are truthful signals from opponents.

**Proposition 4.3 (Impossibility of Full Revelation)**: For generic negotiation games with conflicting interests, no mechanism is simultaneously BIC and guarantees Pareto-optimal outcomes with full revelation ($\alpha_i^t = 0$ for all $i, t$).

*Proof sketch*: This follows from the revelation principle and standard impossibility results in mechanism design. With transferable utility, agents can be incentivized to reveal truthfully only through appropriate payments, which are not available in standard negotiation settings. $\square$

**Controlled Revelation**: SAPIENT circumvents this impossibility by allowing *controlled* revelation through ambiguity:

**Theorem 4.2 (Ambiguity-Incentive Trade-off)**: There exists a non-trivial ambiguity level $\alpha^* \in (0, 1)$ such that:

1. Agents are incentivized to follow the signaling protocol (no strategic misreporting)
2. Sufficient information is revealed to improve upon the no-communication equilibrium

*Proof sketch*: At $\alpha = 1$ (no information), agents cannot misreport but gain no benefit from communication. At $\alpha = 0$ (full revelation), agents can benefit from misreporting. By continuity, there exists an intermediate $\alpha^*$ balancing these forces. Formal proof requires specifying the utility structure and prior distributions. $\square$

### 4.3 Strategic Ambiguity Optimization

We formulate the problem of finding optimal ambiguity levels.

**Optimization Problem**: For a fixed proposal and response strategy, the optimal ambiguity policy solves:

$$\max_{\pi^{\alpha}} \mathbb{E}_{\theta, h \sim \pi}[J_i(\pi_i^{\alpha}, \pi_{-i})]$$

subject to:
$$0 \leq \alpha_i^t \leq 1, \quad \forall t$$

**Value Function**: Define the value function for agent $i$:

$$V_i^t(h_t, \theta_i, b_i^t) = \max_{\pi_i} \mathbb{E}\left[\sum_{\tau=t}^T \gamma^{\tau-t} r_i^{\tau} \mid h_t, \theta_i, b_i^t\right]$$

**Bellman Equation**: The value function satisfies:

$$V_i^t(h_t, \theta_i, b_i^t) = \max_{\alpha_i^t, x_i^t, r_i^t} \left\{ r_i^t + \gamma \mathbb{E}[V_i^{t+1}(h_{t+1}, \theta_i, b_i^{t+1})] \right\}$$

**Optimal Ambiguity**: The optimal ambiguity level at time $t$ is characterized by:

$$\alpha_i^{t,*} = \arg\max_{\alpha \in [0,1]} \left\{ -\lambda_{\text{reveal}} \cdot (1 - \alpha) + \gamma \mathbb{E}[V_i^{t+1} \mid \alpha] \right\}$$

**Proposition 4.4 (Ambiguity Dynamics)**: Under mild conditions, optimal ambiguity follows a decreasing trend: $\mathbb{E}[\alpha_i^t] > \mathbb{E}[\alpha_i^{t+1}]$ for $t < T$.

*Intuition*: As the deadline approaches, the benefit of reaching an agreement outweighs the cost of information revelation, leading agents to gradually reduce ambiguity.

### 4.4 Preference State Formalization

We provide a formal treatment of preference states and their evolution.

**Definition 4.5 (Preference State)**: The preference state at time $t$ is the tuple:

$$\Psi^t = (h_t, \{b_i^t\}_{i=1}^N, \{\theta_i\}_{i=1}^N)$$

consisting of the public history, all belief states, and all true types.

**State Transitions**: The preference state evolves according to:

$$\Psi^{t+1} = \tau(\Psi^t, \{a_i^t\}_{i=1}^N, \{\epsilon_i^t\}_{i=1}^N)$$

where $a_i^t = (\alpha_i^t, x_i^t, r_i^t)$ is agent $i$'s action and $\epsilon_i^t$ is random noise.

**Belief Update**: The belief update for agent $i$ given observation $o^t$ is:

$$b_i^{t+1}(\theta_{-i}) = \frac{P(o^t \mid \theta_{-i}, h_t, \alpha_{-i}^t) \cdot b_i^t(\theta_{-i})}{\int_{\Theta_{-i}} P(o^t \mid \theta_{-i}', h_t, \alpha_{-i}^t) \cdot b_i^t(\theta_{-i}') \, d\theta_{-i}'}$$

**Observation Model**: For ambiguous signals, the observation model is:

$$P(s_j^t \mid \theta_j, \alpha_j^t) = (1 - \alpha_j^t) \cdot \delta_{f(\theta_j)}(s_j^t) + \alpha_j^t \cdot P_{\epsilon}(s_j^t)$$

where $\delta_{f(\theta_j)}$ is a point mass at the true signal and $P_{\epsilon}$ is the noise distribution.

**Convergence of Beliefs**: As more information is revealed, beliefs converge to truth.

**Theorem 4.3 (Belief Convergence)**: If agents follow consistent signaling and proposal strategies with $\alpha_i^t \to 0$ as $t \to T$, then beliefs converge to true types:

$$\lim_{t \to T} b_i^t(\theta_{-i}) = \delta_{\theta_{-i}^*}(\theta_{-i})$$

where $\theta_{-i}^*$ is the true opponent type vector.

*Proof*: Follows from standard results on Bayesian learning with consistent signals. As ambiguity decreases, signals become increasingly informative, driving belief concentration around the truth. $\square$

---

## 5. Architecture Design

### 5.1 Multi-Agent System Architecture

SAPIENT employs a decentralized multi-agent architecture where each agent consists of modular components:

**System Overview**:

```
┌─────────────────────────────────────────────────────────┐
│                    SAPIENT System                        │
│                                                          │
│  ┌────────────┐  ┌────────────┐      ┌────────────┐   │
│  │  Agent 1   │  │  Agent 2   │ ...  │  Agent N   │   │
│  │            │  │            │      │            │   │
│  │ ┌────────┐ │  │ ┌────────┐ │      │ ┌────────┐ │   │
│  │ │Encoder │ │  │ │Encoder │ │      │ │Encoder │ │   │
│  │ └────┬───┘ │  │ └────┬───┘ │      │ └────┬───┘ │   │
│  │      │     │  │      │     │      │      │     │   │
│  │ ┌────▼───┐ │  │ ┌────▼───┐ │      │ ┌────▼───┐ │   │
│  │ │Belief  │ │  │ │Belief  │ │      │ │Belief  │ │   │
│  │ │Module  │ │  │ │Module  │ │      │ │Module  │ │   │
│  │ └────┬───┘ │  │ └────┬───┘ │      │ └────┬───┘ │   │
│  │      │     │  │      │     │      │      │     │   │
│  │ ┌────▼───────────────┐ │  │ │ ┌────▼──────────┐   │
│  │ │  Policy Networks   │ │  │ │  Policy Networks│   │
│  │ │  - Ambiguity π^α   │ │  │ │  - Ambiguity π^α│   │
│  │ │  - Proposal π^x    │ │  │ │  - Proposal π^x │   │
│  │ │  - Response π^r    │ │  │ │  - Response π^r │   │
│  │ └────┬───────────────┘ │  │ └────┬────────────┘   │
│  │      │     │  │      │     │      │      │     │   │
│  │ ┌────▼───┐ │  │ ┌────▼───┐ │      │ ┌────▼───┐ │   │
│  │ │Value   │ │  │ │Value   │ │      │ │Value   │ │   │
│  │ │Network │ │  │ │Network │ │      │ │Network │ │   │
│  │ └────────┘ │  │ └────────┘ │      │ └────────┘ │   │
│  └────────────┘  └────────────┘      └────────────┘   │
│         │               │                    │          │
│         └───────────────┼────────────────────┘          │
│                         │                               │
│                  ┌──────▼──────┐                       │
│                  │ Communication│                       │
│                  │   Protocol   │                       │
│                  └──────┬──────┘                       │
│                         │                               │
│                  ┌──────▼──────┐                       │
│                  │ Environment  │                       │
│                  │  (Nego. Game)│                       │
│                  └──────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

**Component Descriptions**:

1. **Encoder**: Processes raw observations (history, signals, proposals) into fixed-size state representations
2. **Belief Module**: Maintains and updates probability distributions over opponent types
3. **Policy Networks**: 
   - Ambiguity policy $\pi^{\alpha}$: outputs ambiguity level $\alpha_i^t \in [0,1]$
   - Proposal policy $\pi^x$: outputs proposal distribution over $\mathcal{X}$
   - Response policy $\pi^r$: outputs accept/reject decision
4. **Value Network**: Estimates value function $V(s_t)$ for training
5. **Communication Protocol**: Manages message passing and signal transmission between agents
6. **Environment**: Simulates the negotiation game, tracks state, enforces protocol

### 5.2 State Space Formalization

Each agent $i$ observes a partial state consisting of:

**Local Observation**: 
$$o_i^t = (h_t^{\text{pub}}, \theta_i, b_i^t, t)$$

where:
- $h_t^{\text{pub}}$ is the public history of proposals, responses, and signals
- $\theta_i$ is agent $i$'s private type
- $b_i^t$ is agent $i$'s belief distribution over $\theta_{-i}$
- $t$ is the current time step

**State Representation**: The encoder transforms $o_i^t$ into a fixed-size vector $s_i^t \in \mathbb{R}^d$:

$$s_i^t = \phi(o_i^t; \psi_i)$$

where $\phi$ is a neural network parameterized by $\psi_i$.

**History Encoding**: The public history is encoded using a recurrent or transformer architecture:

$$h_{\text{enc}}^t = \text{RNN}(h_{\text{enc}}^{t-1}, (x^{t-1}, r^{t-1}, s^{t-1}))$$

or

$$h_{\text{enc}}^t = \text{Transformer}((x^1, r^1, s^1, \ldots, x^{t-1}, r^{t-1}, s^{t-1}))$$

**Type Encoding**: The private type $\theta_i$ is encoded as a vector:

$$\theta_{\text{enc}}^i = \text{MLP}(\theta_i; \psi_{\theta})$$

For multi-issue negotiation with weights and values:

$$\theta_i = (w_{i1}, \ldots, w_{iM}, v_{i1}, \ldots, v_{iM})$$

**Belief Encoding**: The belief distribution $b_i^t(\theta_{-i})$ is represented by:

1. **Parametric**: Mean and covariance of a Gaussian approximation
2. **Particle-based**: Set of sampled types $\{\hat{\theta}_{-i}^{(k)}\}_{k=1}^K$
3. **Neural**: Implicit representation via neural network

For simplicity, we use a parametric approach:

$$b_{\text{enc}}^{i,t} = (\mu_{b}^{i,t}, \Sigma_{b}^{i,t})$$

where $\mu_{b}^{i,t} \in \mathbb{R}^{|\Theta_{-i}|}$ and $\Sigma_{b}^{i,t} \in \mathbb{R}^{|\Theta_{-i}| \times |\Theta_{-i}|}$.

**Full State Encoding**: The complete state representation is:

$$s_i^t = [h_{\text{enc}}^t \,\|\, \theta_{\text{enc}}^i \,\|\, b_{\text{enc}}^{i,t} \,\|\, \text{onehot}(t)]$$

where $\|$ denotes concatenation.

### 5.3 Action Space Specification

Each agent's action space consists of three components:

**Ambiguity Action**: $\alpha_i^t \in [0, 1]$ (continuous)

The ambiguity policy outputs parameters of a Beta distribution:

$$\pi_i^{\alpha}(\cdot \mid s_i^t; \theta_{\pi}^{\alpha}) = \text{Beta}(\alpha; a^t, b^t)$$

where $a^t, b^t = f_{\alpha}(s_i^t; \theta_{\pi}^{\alpha})$ are computed by a neural network.

**Proposal Action**: $x_i^t \in \mathcal{X}$ (continuous or discrete depending on domain)

For multi-issue negotiation with continuous issues:

$$\pi_i^x(\cdot \mid s_i^t; \theta_{\pi}^x) = \prod_{j=1}^M \mathcal{N}(x_j; \mu_j^t, \sigma_j^t)$$

where $\mu_j^t, \sigma_j^t = f_x^j(s_i^t; \theta_{\pi}^x)$.

For discrete issues:

$$\pi_i^x(x \mid s_i^t; \theta_{\pi}^x) = \text{Categorical}(f_x(s_i^t; \theta_{\pi}^x))$$

**Response Action**: $r_i^t \in \{\text{accept}, \text{reject}\}$ (binary)

$$\pi_i^r(\text{accept} \mid s_i^t, x^t; \theta_{\pi}^r) = \sigma(f_r(s_i^t, x^t; \theta_{\pi}^r))$$

where $\sigma$ is the sigmoid function.

**Action Masking**: Not all actions are available at all times:

- Only the designated proposer can make a proposal at each time step
- Response actions are only available when a proposal has been made
- Ambiguity actions are available at all times (can be part of any communication)

### 5.4 Reward Function Formulation

The reward function balances multiple objectives:

**Terminal Reward**: At time $T$ (or when agreement is reached):

$$r_i^T = \begin{cases}
u_i(x^*, \theta_i) & \text{if agreement } x^* \text{ reached} \\
u_i^{\text{dis}} & \text{otherwise}
\end{cases}$$

**Step Rewards**: At each time step $t < T$:

$$r_i^t = r_i^{\text{time}} + r_i^{\text{info}} + r_i^{\text{progress}}$$

**Time Cost**: Penalizes delays in reaching agreement:

$$r_i^{\text{time}} = -c_{\text{time}}$$

where $c_{\text{time}} > 0$ is a small constant.

**Information Cost/Benefit**:

$$r_i^{\text{info}} = -\lambda_{\text{reveal}} \cdot (1 - \alpha_i^t) + \lambda_{\text{learn}} \cdot \Delta H_i^t$$

where:
- $\lambda_{\text{reveal}} > 0$ penalizes information revelation
- $\lambda_{\text{learn}} > 0$ rewards learning about opponents
- $\Delta H_i^t = H(b_i^{t-1}) - H(b_i^t)$ is the reduction in belief entropy (information gain)

**Progress Reward**: Encourages movement toward efficient agreements:

$$r_i^{\text{progress}} = \lambda_{\text{progress}} \cdot \left(\mathbb{E}_{\theta_{-i} \sim b_i^t}[u_i(x_{\text{best}}^t, \theta_i)] - \mathbb{E}_{\theta_{-i} \sim b_i^{t-1}}[u_i(x_{\text{best}}^{t-1}, \theta_i)]\right)$$

where $x_{\text{best}}^t$ is the estimated best agreement given current beliefs.

**Hyperparameters**: Typical values:
- $c_{\text{time}} = 0.01$
- $\lambda_{\text{reveal}} = 0.1$
- $\lambda_{\text{learn}} = 0.05$
- $\lambda_{\text{progress}} = 0.2$

### 5.5 Communication Protocol

SAPIENT uses a structured communication protocol:

**Message Structure**: At each time step, agents can send messages of the form:

$$m_i^t = (\text{type}, \text{content}, \alpha_i^t)$$

where:
- $\text{type} \in \{\text{signal}, \text{proposal}, \text{response}, \text{query}\}$
- $\text{content}$ depends on message type
- $\alpha_i^t$ is the ambiguity level (applies to signals)

**Message Types**:

1. **Signal**: $(s_i^t, \alpha_i^t)$ - ambiguous preference signal
2. **Proposal**: $(x_i^t, \emptyset)$ - concrete outcome proposal
3. **Response**: $(r_i^t, \emptyset)$ - accept/reject decision
4. **Query**: $(q_i^t, \emptyset)$ - question about opponent preferences (optional)

**Protocol Rules**:

1. **Turn-taking**: Agents alternate as proposers (or simultaneous in some variants)
2. **Signal broadcasting**: Signals are broadcast to all other agents
3. **Proposal response**: All non-proposing agents must respond to proposals
4. **Termination**: Negotiation ends when all agents accept or deadline $T$ is reached

**Signal Transmission**: For signal $s_i^t$ with ambiguity $\alpha_i^t$:

1. Agent $i$ computes true signal $s_{\text{true}}^i = f(\theta_i)$
2. Samples noise $\epsilon_t \sim P_{\epsilon}$
3. Transmits $s_i^t = (1 - \alpha_i^t) \cdot s_{\text{true}}^i + \alpha_i^t \cdot \epsilon_t$
4. Receivers observe $s_i^t$ and update beliefs

**Synchronization**: In the multi-agent setting, we use:

- **Synchronous**: All agents act simultaneously, environment steps forward
- **Asynchronous**: Agents act in sequence according to protocol

For simplicity, we primarily consider synchronous communication.

---

## 6. Algorithms

### 6.1 Main Training Loop

The main training procedure uses multi-agent reinforcement learning with experience replay and periodic policy updates.

**Algorithm 6.1: SAPIENT Training Loop**

```python
# Initialization
Initialize agent networks {θ_π^α_i, θ_π^x_i, θ_π^r_i, θ_V_i} for i = 1, ..., N
Initialize target networks {θ_π^α_i', θ_π^x_i', θ_π^r_i', θ_V_i'} ← {θ_π^α_i, θ_π^x_i, θ_π^r_i, θ_V_i}
Initialize replay buffer D ← ∅
Initialize belief networks {θ_b_i} for i = 1, ..., N

# Training loop
for episode = 1 to MAX_EPISODES:
    # Sample types for all agents
    θ_i ~ P(θ_i) for i = 1, ..., N
    
    # Initialize episode
    h_0 ← ∅
    b_i^0 ← P(θ_{-i}) for i = 1, ..., N  # Prior beliefs
    s_i^0 ← encode_state(h_0, θ_i, b_i^0, t=0) for i = 1, ..., N
    
    # Episode rollout
    for t = 0 to T-1:
        # Action selection for each agent
        for i = 1, ..., N:
            # Sample ambiguity level
            α_i^t ~ π_i^α(· | s_i^t; θ_π^α_i)
            
            # Generate ambiguous signal
            s_i^t ← generate_signal(θ_i, α_i^t)
            
            # If agent i is proposer at time t:
            if is_proposer(i, t):
                x_i^t ~ π_i^x(· | s_i^t; θ_π^x_i)
            
            # If proposal exists from another agent:
            if exists_proposal(t):
                r_i^t ~ π_i^r(· | s_i^t, x^t; θ_π^r_i)
        
        # Environment step
        (h_{t+1}, {r_i^t}, done) ← env.step({α_i^t, x_i^t, r_i^t})
        
        # Belief updates for each agent
        for i = 1, ..., N:
            # Update beliefs based on observed signals and actions
            b_i^{t+1} ← update_beliefs(b_i^t, observations^t, α_{-i}^t; θ_b_i)
            
            # Encode next state
            s_i^{t+1} ← encode_state(h_{t+1}, θ_i, b_i^{t+1}, t+1)
        
        # Store transition in replay buffer
        D ← D ∪ {(s^t, {α_i^t, x_i^t, r_i^t}, {r_i^t}, s^{t+1}, done)}
        
        # Check termination
        if done:
            break
    
    # Policy updates
    if episode % UPDATE_FREQ == 0 and |D| ≥ BATCH_SIZE:
        for update_step = 1 to NUM_UPDATES:
            # Sample mini-batch
            B ← sample_batch(D, BATCH_SIZE)
            
            # Update policies and value functions
            for i = 1, ..., N:
                update_agent_i(B, θ_π^α_i, θ_π^x_i, θ_π^r_i, θ_V_i, θ_π^α_i', θ_π^x_i', θ_π^r_i', θ_V_i')
            
            # Update belief networks
            update_belief_networks({θ_b_i}, B)
        
        # Target network updates (soft update)
        for i = 1, ..., N:
            θ_π^α_i' ← τ · θ_π^α_i + (1 - τ) · θ_π^α_i'
            θ_π^x_i' ← τ · θ_π^x_i + (1 - τ) · θ_π^x_i'
            θ_π^r_i' ← τ · θ_π^r_i + (1 - τ) · θ_π^r_i'
            θ_V_i' ← τ · θ_V_i + (1 - τ) · θ_V_i'
    
    # Logging
    if episode % LOG_FREQ == 0:
        log_metrics(episode, {r_i^t}, avg_ambiguity, agreement_rate, etc.)

return {θ_π^α_i, θ_π^x_i, θ_π^r_i, θ_V_i, θ_b_i} for i = 1, ..., N
```

### 6.2 Information Revelation Policy Learning

The information revelation policy (ambiguity policy) is trained using policy gradient methods.

**Algorithm 6.2: Ambiguity Policy Update**

```python
def update_ambiguity_policy(agent_i, batch, θ_π^α_i, θ_V_i, θ_V_i'):
    """
    Update ambiguity policy using PPO (Proximal Policy Optimization)
    
    Args:
        agent_i: Agent index
        batch: Mini-batch of transitions
        θ_π^α_i: Current ambiguity policy parameters
        θ_V_i: Current value network parameters
        θ_V_i': Target value network parameters
    """
    
    # Extract batch data for agent i
    states = [transition.s_i^t for transition in batch]
    actions_α = [transition.α_i^t for transition in batch]
    rewards = [transition.r_i^t for transition in batch]
    next_states = [transition.s_i^{t+1} for transition in batch]
    dones = [transition.done for transition in batch]
    
    # Compute returns using GAE (Generalized Advantage Estimation)
    returns = []
    advantages = []
    
    for idx in range(len(batch)):
        # Compute TD error
        V_current = V(states[idx]; θ_V_i)
        V_next = V(next_states[idx]; θ_V_i') if not dones[idx] else 0
        td_error = rewards[idx] + γ * V_next - V_current
        
        # Compute GAE advantage
        advantage = compute_gae(td_error, γ, λ_gae)
        advantages.append(advantage)
        
        # Compute return
        ret = advantage + V_current
        returns.append(ret)
    
    # Normalize advantages
    advantages = (advantages - mean(advantages)) / (std(advantages) + ε)
    
    # Store old policy probabilities
    old_log_probs = []
    for idx in range(len(batch)):
        old_dist = π^α(· | states[idx]; θ_π^α_i)
        old_log_prob = old_dist.log_prob(actions_α[idx])
        old_log_probs.append(old_log_prob)
    
    # PPO update loop
    for ppo_epoch in range(PPO_EPOCHS):
        # Compute new policy probabilities
        new_log_probs = []
        entropies = []
        
        for idx in range(len(batch)):
            new_dist = π^α(· | states[idx]; θ_π^α_i)
            new_log_prob = new_dist.log_prob(actions_α[idx])
            new_log_probs.append(new_log_prob)
            entropies.append(new_dist.entropy())
        
        # Compute probability ratios
        ratios = exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = clip(ratios, 1 - ε_clip, 1 + ε_clip) * advantages
        policy_loss = -mean(min(surr1, surr2))
        
        # Entropy bonus for exploration
        entropy_loss = -c_entropy * mean(entropies)
        
        # Value function loss
        value_loss = mean((V(states; θ_V_i) - returns)^2)
        
        # Total loss
        total_loss = policy_loss + c_value * value_loss + entropy_loss
        
        # Gradient descent step
        θ_π^α_i ← θ_π^α_i - α_lr * ∇_{θ_π^α_i} total_loss
        θ_V_i ← θ_V_i - α_lr * ∇_{θ_V_i} value_loss
    
    return θ_π^α_i, θ_V_i
```

**Key Components**:

1. **Advantage Estimation**: Uses GAE to reduce variance in policy gradients
2. **PPO Clipping**: Prevents destructively large policy updates
3. **Entropy Regularization**: Encourages exploration of ambiguity levels
4. **Value Function Learning**: Estimates expected returns for variance reduction

### 6.3 Preference Calibration Mechanism

The preference calibration mechanism maintains and updates belief distributions over opponent types.

**Algorithm 6.3: Belief Update**

```python
def update_beliefs(agent_i, b_i^t, observations^t, α_{-i}^t, θ_b_i):
    """
    Update belief distribution over opponent types using Bayesian inference
    
    Args:
        agent_i: Agent index
        b_i^t: Current belief distribution (mean μ, covariance Σ)
        observations^t: Observations at time t (signals, proposals, responses)
        α_{-i}^t: Ambiguity levels of opponents
        θ_b_i: Parameters of belief network
    
    Returns:
        b_i^{t+1}: Updated belief distribution
    """
    
    # Extract current belief parameters
    μ_t, Σ_t = b_i^t
    
    # Process observations for each opponent j ≠ i
    likelihood_params = []
    
    for j in agents_except(i):
        # Extract opponent j's signal
        s_j^t = observations^t.signals[j]
        α_j^t = α_{-i}^t[j]
        
        # Compute likelihood P(s_j^t | θ_j, α_j^t)
        # Using the signaling model: s_j^t = (1 - α_j^t) * f(θ_j) + α_j^t * ε
        
        # For Gaussian approximation:
        # If α_j^t = 0: signal is deterministic f(θ_j)
        # If α_j^t = 1: signal is pure noise
        # Intermediate: weighted combination
        
        # Compute expected signal given θ_j
        def expected_signal(θ_j):
            return (1 - α_j^t) * f(θ_j) + α_j^t * μ_ε
        
        # Compute signal variance
        signal_variance = (α_j^t)^2 * σ_ε^2 + σ_obs^2
        
        # Likelihood is Gaussian: N(s_j^t; expected_signal(θ_j), signal_variance)
        likelihood_params.append({
            'observed': s_j^t,
            'variance': signal_variance,
            'signal_fn': expected_signal
        })
    
    # Bayesian update (Extended Kalman Filter for Gaussian beliefs)
    μ_{t+1}, Σ_{t+1} = extended_kalman_update(μ_t, Σ_t, likelihood_params)
    
    # Alternative: Use neural network to directly predict updated beliefs
    # μ_{t+1}, Σ_{t+1} = belief_network(μ_t, Σ_t, observations^t, α_{-i}^t; θ_b_i)
    
    # Optional: Incorporate proposal information
    if observations^t.proposal_exists:
        x^t = observations^t.proposal
        proposer = observations^t.proposer
        
        # Proposal provides information about proposer's preferences
        # Update belief about proposer using rationality assumption
        # (proposer likely proposes outcomes favorable to them)
        
        μ_{t+1}[proposer], Σ_{t+1}[proposer] = update_from_proposal(
            μ_{t+1}[proposer], Σ_{t+1}[proposer], x^t
        )
    
    # Optional: Incorporate response information
    if observations^t.responses_exist:
        for j, response in observations^t.responses.items():
            # Response provides information about preferences
            # Accept → proposal likely gives positive utility
            # Reject → proposal likely gives negative utility
            
            μ_{t+1}[j], Σ_{t+1}[j] = update_from_response(
                μ_{t+1}[j], Σ_{t+1}[j], x^t, response
            )
    
    b_i^{t+1} = (μ_{t+1}, Σ_{t+1})
    return b_i^{t+1}


def extended_kalman_update(μ_t, Σ_t, likelihood_params):
    """
    Extended Kalman Filter update for non-linear observation model
    """
    # Prediction step (no dynamics, beliefs don't change without observations)
    μ_pred = μ_t
    Σ_pred = Σ_t
    
    # Update step for each observation
    for params in likelihood_params:
        s_obs = params['observed']
        R = params['variance']  # Observation noise covariance
        h = params['signal_fn']  # Observation function
        
        # Linearize observation function around current mean
        H = jacobian(h, μ_pred)  # Jacobian matrix
        
        # Compute innovation
        innovation = s_obs - h(μ_pred)
        
        # Innovation covariance
        S = H @ Σ_pred @ H.T + R
        
        # Kalman gain
        K = Σ_pred @ H.T @ inv(S)
        
        # Update mean
        μ_pred = μ_pred + K @ innovation
        
        # Update covariance
        Σ_pred = (I - K @ H) @ Σ_pred
    
    return μ_pred, Σ_pred
```

**Belief Update Properties**:

1. **Ambiguity-Aware**: Update magnitude depends on opponent's ambiguity level
2. **Multi-Modal**: Can be extended to particle filters for multi-modal beliefs
3. **Rational Inference**: Incorporates proposals and responses as preference signals
4. **Learned**: Can replace analytical update with learned belief network

### 6.4 Multi-Agent Coordination Algorithm

For multi-agent coordination, we adapt QMIX to learn centralized value functions while maintaining decentralized execution.

**Algorithm 6.4: QMIX-SAPIENT Update**

```python
def update_qmix_sapient(batch, {θ_Q_i}, θ_mix, {θ_Q_i'}, θ_mix'):
    """
    Update QMIX value networks for multi-agent coordination
    
    Args:
        batch: Mini-batch of joint transitions
        {θ_Q_i}: Individual agent Q-network parameters
        θ_mix: Mixing network parameters
        {θ_Q_i'}, θ_mix': Target network parameters
    
    Returns:
        Updated parameters
    """
    
    # Extract batch data
    joint_states = [transition.s^t for transition in batch]
    joint_actions = [transition.a^t for transition in batch]
    rewards = [transition.r^t for transition in batch]  # Joint reward
    next_joint_states = [transition.s^{t+1} for transition in batch]
    dones = [transition.done for transition in batch]
    
    # Compute individual Q-values for current actions
    Q_values = []
    for idx in range(len(batch)):
        Q_i_values = []
        for i in range(N):
            s_i = joint_states[idx][i]
            a_i = joint_actions[idx][i]
            Q_i = Q_network(s_i, a_i; θ_Q_i)
            Q_i_values.append(Q_i)
        Q_values.append(Q_i_values)
    
    # Mix individual Q-values using mixing network
    Q_tot = []
    for idx in range(len(batch)):
        # Mixing network takes individual Q-values and global state
        # Produces monotonic combination (ensures consistency with decentralized policies)
        global_state = extract_global_state(joint_states[idx])
        Q_total = mixing_network(Q_values[idx], global_state; θ_mix)
        Q_tot.append(Q_total)
    
    # Compute target Q-values using target networks
    Q_targets = []
    for idx in range(len(batch)):
        if dones[idx]:
            target = rewards[idx]
        else:
            # Compute max Q-value for next state
            next_Q_i_values = []
            for i in range(N):
                s_i_next = next_joint_states[idx][i]
                # Get best action according to current policy
                a_i_next = argmax_{a'} Q_network(s_i_next, a'; θ_Q_i')
                Q_i_next = Q_network(s_i_next, a_i_next; θ_Q_i')
                next_Q_i_values.append(Q_i_next)
            
            # Mix next Q-values
            global_state_next = extract_global_state(next_joint_states[idx])
            Q_total_next = mixing_network(next_Q_i_values, global_state_next; θ_mix')
            
            target = rewards[idx] + γ * Q_total_next
        
        Q_targets.append(target)
    
    # Compute TD loss
    td_errors = Q_tot - Q_targets
    loss = mean(td_errors^2)
    
    # Gradient descent update
    # Update individual Q-networks
    for i in range(N):
        θ_Q_i ← θ_Q_i - α_lr * ∇_{θ_Q_i} loss
    
    # Update mixing network
    θ_mix ← θ_mix - α_lr * ∇_{θ_mix} loss
    
    return {θ_Q_i}, θ_mix


def mixing_network(Q_values, global_state, θ_mix):
    """
    Monotonic mixing network that combines individual Q-values
    
    Ensures: ∂Q_tot/∂Q_i ≥ 0 for all i (monotonicity constraint)
    """
    # Hypernetwork generates weights based on global state
    # Weights are constrained to be non-negative (via abs or softplus)
    
    # First layer
    W1 = abs(hypernetwork_W1(global_state; θ_mix))
    b1 = hypernetwork_b1(global_state; θ_mix)
    
    hidden = relu(W1 @ Q_values + b1)
    
    # Second layer
    W2 = abs(hypernetwork_W2(global_state; θ_mix))
    b2 = hypernetwork_b2(global_state; θ_mix)
    
    Q_tot = W2 @ hidden + b2
    
    return Q_tot
```

**QMIX-SAPIENT Properties**:

1. **Centralized Training**: Uses global information to learn better value functions
2. **Decentralized Execution**: Individual agents act based on local observations
3. **Monotonicity**: Ensures consistency between individual and joint Q-values
4. **Scalability**: More efficient than full joint action-space methods

### 6.5 Convergence Analysis

We provide conditions under which SAPIENT training converges.

**Theorem 6.1 (Convergence of Policy Gradients)**: Under the following conditions:

1. Bounded rewards: $|r_i^t| \leq R_{\max}$ for all $i, t$
2. Lipschitz continuous value functions: $|V_i(s) - V_i(s')| \leq L \|s - s'\|$
3. Decreasing learning rates: $\sum_{k=1}^{\infty} \alpha_k = \infty$ and $\sum_{k=1}^{\infty} \alpha_k^2 < \infty$
4. Sufficient exploration: $\epsilon_t \to 0$ slowly enough

The policy gradient updates in Algorithm 6.2 converge to a local optimum of the expected return $J_i(\pi_i)$.

*Proof sketch*: Follows from standard policy gradient convergence results. The use of PPO with clipping ensures bounded policy updates, preventing divergence. GAE provides unbiased gradient estimates with reduced variance. $\square$

**Theorem 6.2 (Belief Convergence)**: If ambiguity levels decrease over time ($\alpha_i^t \to 0$ as $t \to T$) and agents follow consistent policies, then beliefs converge to true types:

$$\lim_{t \to T} \|μ_i^t - \theta_{-i}^*\| = 0 \quad \text{a.s.}$$

where $\theta_{-i}^*$ is the true opponent type vector.

*Proof*: As ambiguity decreases, signals become increasingly informative. The Extended Kalman Filter update ensures that beliefs concentrate around the maximum likelihood type given observations. With enough informative signals, the maximum likelihood estimate converges to the true type. $\square$

**Corollary 6.1 (Near-Optimal Agreements)**: Under the conditions of Theorems 6.1 and 6.2, SAPIENT agents reach agreements that approach Pareto-optimality:

$$\lim_{t \to T} \mathbb{E}[W(\pi)] \to W^*$$

where $W^*$ is the maximum social welfare achievable with complete information.

### 6.6 Complexity Analysis

**Time Complexity (Training)**:

- **Per episode**: $O(T \cdot N \cdot (d^2 + |\mathcal{X}|))$ where:
  - $T$ is negotiation horizon
  - $N$ is number of agents
  - $d$ is state representation dimension
  - $|\mathcal{X}|$ is size of outcome space (for discrete domains)
  
- **Per update**: $O(B \cdot N \cdot (d^3 + |\mathcal{X}|^2))$ where:
  - $B$ is batch size
  - Dominated by network forward/backward passes

- **Total training**: $O(E \cdot T \cdot N \cdot d^2)$ for $E$ episodes

**Space Complexity**:

- **Replay buffer**: $O(C \cdot T \cdot N \cdot d)$ for capacity $C$
- **Networks**: $O(N \cdot (L \cdot d^2))$ for $L$ layers and dimension $d$
- **Beliefs**: $O(N \cdot |\Theta|^2)$ for Gaussian beliefs (covariance matrices)

**Inference Complexity**:

- **Per time step**: $O(N \cdot d^2)$ (forward passes through policy networks)
- **Per negotiation**: $O(T \cdot N \cdot d^2)$

**Scalability**:

- **Linear in $N$**: Architecture scales linearly with number of agents
- **Linear in $T$**: Recurrent encoding handles variable horizons efficiently
- **Quadratic in $d$**: Network complexity depends on state dimension
- **Exponential in $M$**: Outcome space grows exponentially with number of issues (can be mitigated with factored representations)

---

## 7. Mathematical Foundations

### 7.1 Information-Theoretic Metrics

Information theory provides rigorous quantification of knowledge and learning.

**Definition 7.1 (Entropy)**: The entropy of a random variable $X$ with distribution $P$ is:

$$H(X) = -\sum_{x} P(x) \log P(x)$$

or for continuous variables:

$$H(X) = -\int P(x) \log P(x) \, dx$$

**Interpretation**: Entropy measures uncertainty or information content.

**Definition 7.2 (Conditional Entropy)**: The conditional entropy of $X$ given $Y$ is:

$$H(X \mid Y) = -\sum_{y} P(y) \sum_{x} P(x \mid y) \log P(x \mid y)$$

**Definition 7.3 (Mutual Information)**: The mutual information between $X$ and $Y$ is:

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

**Interpretation**: Mutual information quantifies how much learning $Y$ reduces uncertainty about $X$.

**Application to SAPIENT**:

1. **Belief Uncertainty**: $H(b_i^t)$ measures agent $i$'s uncertainty about opponent types
2. **Information Revelation**: $I(\theta_i; s_i^t \mid \alpha_i^t)$ measures information content of signal
3. **Learning Progress**: $\Delta H_i^t = H(b_i^{t-1}) - H(b_i^t)$ measures information gained

**Proposition 7.1 (Information Gain Bound)**: The information gain from signal $s_j^t$ is bounded by:

$$I(\theta_j; s_j^t \mid \alpha_j^t) \leq (1 - \alpha_j^t) \cdot H(\theta_j)$$

*Proof*: At $\alpha_j^t = 1$, signal is independent of type, so $I(\theta_j; s_j^t \mid \alpha_j^t = 1) = 0$. At $\alpha_j^t = 0$, signal fully reveals type, so $I(\theta_j; s_j^t \mid \alpha_j^t = 0) = H(\theta_j)$. By monotonicity (Proposition 3.1), information is linear in $(1 - \alpha_j^t)$. $\square$

**Differential Entropy**: For continuous types with Gaussian beliefs $\mathcal{N}(\mu, \Sigma)$:

$$H(b_i^t) = \frac{1}{2} \log \det(2\pi e \Sigma_t)$$

**KL Divergence**: Measures distance between belief distributions:

$$D_{KL}(b_i^t \| b_i^{t-1}) = \int b_i^t(\theta) \log \frac{b_i^t(\theta)}{b_i^{t-1}(\theta)} \, d\theta$$

### 7.2 Utility Functions and Ambiguity Constraints

**Definition 7.4 (Expected Utility)**: Agent $i$'s expected utility given belief $b_i^t$ is:

$$EU_i(x, \theta_i, b_i^t) = u_i(x, \theta_i) + \mathbb{E}_{\theta_{-i} \sim b_i^t}[\phi_i(x, \theta_i, \theta_{-i})]$$

where $\phi_i$ captures strategic considerations (e.g., exploitation risk).

**Risk Aversion**: Agents may be risk-averse regarding information revelation:

$$U_i^{\text{risk}}(\alpha_i^t) = EU_i(\alpha_i^t) - \beta_i \cdot \text{Var}[U_i \mid \alpha_i^t]$$

where $\beta_i \geq 0$ is risk aversion parameter.

**Ambiguity Constraints**: We can impose constraints on ambiguity:

1. **Minimum revelation**: $\alpha_i^t \leq \alpha_{\max} < 1$ (ensure some information is shared)
2. **Temporal decrease**: $\alpha_i^{t+1} \leq \alpha_i^t$ (commitment to increasing transparency)
3. **Reciprocity**: $|\alpha_i^t - \alpha_j^t| \leq \delta$ (matching opponent's ambiguity level)

**Definition 7.5 (Incentive-Compatible Revelation)**: An ambiguity level $\alpha^*$ is incentive-compatible if:

$$EU_i(\alpha^*, \theta_i^{\text{true}}) \geq EU_i(\alpha', \theta_i^{\text{false}})$$

for all alternative ambiguity levels $\alpha'$ and misrepresented types $\theta_i^{\text{false}}$.

**Proposition 7.2 (Optimal Ambiguity)**: The optimal ambiguity level satisfies the first-order condition:

$$\frac{\partial}{\partial \alpha_i^t} \left[ \mathbb{E}[u_i(x^*, \theta_i) \mid \alpha_i^t] - \lambda_{\text{reveal}} \cdot (1 - \alpha_i^t) \right] = 0$$

**Trade-off Analysis**: This yields:

$$\alpha_i^{t,*} = \begin{cases}
0 & \text{if } \frac{\partial \mathbb{E}[u_i]}{\partial \alpha_i^t} < -\lambda_{\text{reveal}} \\
1 & \text{if } \frac{\partial \mathbb{E}[u_i]}{\partial \alpha_i^t} > \lambda_{\text{reveal}} \\
\text{interior solution} & \text{otherwise}
\end{cases}$$

### 7.3 Fairness and Efficiency Measures

**Definition 7.6 (Pareto Efficiency)**: An outcome $x^*$ is Pareto efficient if there exists no other outcome $x'$ such that:

$$u_i(x', \theta_i) \geq u_i(x^*, \theta_i) \quad \forall i$$

with strict inequality for at least one $i$.

**Definition 7.7 (Social Welfare)**: The social welfare of outcome $x$ is:

$$W(x, \theta) = \sum_{i=1}^N u_i(x, \theta_i)$$

**Nash Bargaining Solution**: The Nash bargaining solution maximizes:

$$\text{NBS}(\theta) = \arg\max_{x \in \mathcal{X}} \prod_{i=1}^N (u_i(x, \theta_i) - u_i^{\text{dis}})$$

**Kalai-Smorodinsky Solution**: Focuses on proportional gains:

$$\text{KS}(\theta) = \arg\max_{x \in \mathcal{X}} \min_{i} \frac{u_i(x, \theta_i) - u_i^{\text{dis}}}{\max_{x'} u_i(x', \theta_i) - u_i^{\text{dis}}}$$

**Fairness Metrics**:

1. **Envy-freeness**: $u_i(x_i, \theta_i) \geq u_i(x_j, \theta_i)$ for all $i, j$
2. **Proportionality**: $u_i(x_i, \theta_i) \geq \frac{1}{N} \sum_j u_i(x_j, \theta_i)$
3. **Egalitarian**: Maximize $\min_i u_i(x, \theta_i)$ (maximin)

**Definition 7.8 (Price of Ambiguity)**: The efficiency loss due to ambiguity is:

$$\text{PoA}(\alpha) = \frac{W^*(\theta)}{W(\alpha, \theta)}$$

where $W^*(\theta) = \max_x W(x, \theta)$ is optimal welfare with full information.

**Theorem 7.1 (Bounded Price of Ambiguity)**: Under strategic revelation, the price of ambiguity is bounded:

$$\text{PoA}(\alpha) \leq 1 + C \cdot \bar{\alpha}$$

for some constant $C > 0$ depending on the utility structure.

*Proof sketch*: With full revelation ($\bar{\alpha} = 0$), agents can discover Pareto-optimal agreements, so $\text{PoA}(0) = 1$. As ambiguity increases, information loss prevents optimal coordination, but the marginal loss is bounded by the informativeness of signals. $\square$

### 7.4 Formal Theorems

**Theorem 7.2 (Individual Rationality)**: Under SAPIENT, all agents achieve at least their disagreement utility in expectation:

$$\mathbb{E}[u_i(x^*, \theta_i) \mid \pi^*] \geq u_i^{\text{dis}}$$

where $x^*$ is the agreed outcome and $\pi^*$ is the equilibrium policy.

*Proof*: Agents have the option to reject all proposals and receive disagreement utility. In equilibrium, rational agents only accept proposals that provide at least disagreement utility. $\square$

**Theorem 7.3 (Convergence to Efficiency)**: If ambiguity decreases to zero over time and beliefs converge to truth, then agreements converge to Pareto-optimal outcomes:

$$\lim_{t \to T} P(x^t \text{ is Pareto-optimal} \mid \alpha^t \to 0, b^t \to \theta^*) = 1$$

*Proof*: With true beliefs and full information revelation, agents can identify the Pareto frontier. Rational agents with accurate beliefs will only agree to Pareto-optimal outcomes. $\square$

**Theorem 7.4 (No Exploitation)**: Under strategic ambiguity with incentive-compatible revelation, agents cannot systematically exploit each other:

$$\mathbb{E}[u_i(x^*, \theta_i) \mid \pi^*] \geq \mathbb{E}[u_i(x^{\text{naive}}, \theta_i)]$$

where $x^{\text{naive}}$ is the outcome with truthful full revelation.

*Proof*: Strategic ambiguity prevents full revelation of exploitable information. Agents maintain sufficient uncertainty about opponents to prevent systematic exploitation. $\square$

---

## 8. Experimental Setup

### 8.1 Benchmark Domains

We evaluate SAPIENT on three primary negotiation domains:

**Domain 1: Bilateral Bargaining**

- **Description**: Two agents negotiate over the division of resources or terms of trade
- **Outcome space**: $\mathcal{X} = [0, 1]^M$ for $M$ issues (e.g., price, quantity, delivery time)
- **Utility structure**: Linear additive with private weights
  $$u_i(x, \theta_i) = \sum_{j=1}^M w_{ij} \cdot x_j$$
  where $w_{ij} \sim \text{Uniform}(0, 1)$ normalized to sum to 1
- **Complexity**: Simple, interpretable, allows detailed analysis
- **Instances**: 
  - Single-issue: $M = 1$ (pure bargaining)
  - Multi-issue: $M = 3$ (price, quality, delivery)
  - High-dimensional: $M = 10$

**Domain 2: Coalition Formation**

- **Description**: $N = 3-5$ agents negotiate coalition structures and payoff divisions
- **Outcome space**: Coalition structures $\mathcal{C}$ and payoff allocations
- **Utility structure**: Coalition value function $v: 2^{\mathcal{N}} \to \mathbb{R}$ (private knowledge)
  $$u_i(C, y) = y_i \quad \text{subject to} \quad \sum_{i \in S} y_i = v(S)$$
  where $C$ is coalition structure and $y$ is payoff allocation
- **Complexity**: Exponential coalition space, strategic interdependence
- **Instances**:
  - 3-player with synergies
  - 5-player with subadditivity
  - Asymmetric power (weighted voting)

**Domain 3: Multi-Issue Negotiation (Diplomatic/Business)**

- **Description**: Complex negotiation with multiple issues, constraints, and package deals
- **Outcome space**: Discrete values for each issue
  $$\mathcal{X} = \prod_{j=1}^M \{v_j^1, v_j^2, \ldots, v_j^{k_j}\}$$
- **Utility structure**: Non-linear with interdependencies
  $$u_i(x, \theta_i) = \sum_{j=1}^M w_{ij} \cdot f_{ij}(x_j) + \sum_{j < k} \phi_{ijk}(x_j, x_k)$$
  where $\phi$ captures issue interdependencies
- **Complexity**: Large discrete space, non-linear utilities, synergies
- **Instances**:
  - Business deal: price, contract length, terms, exclusivity
  - Diplomatic: territory, resources, treaties, alliances
  - Supply chain: quantity, timing, quality, logistics

### 8.2 Baseline Methods

We compare SAPIENT against several baselines:

**1. Complete Information (Upper Bound)**
- Agents have full knowledge of all types
- Use game-theoretic solution concepts (Nash bargaining, Shapley value)
- Represents best-case scenario

**2. Zero-Sum Adversarial**
- Agents assume pure conflict and zero-sum utilities
- Use minimax strategies
- Represents worst-case scenario

**3. Fixed Revelation Policies**
- **No Communication**: No signals exchanged ($\alpha = 1$ always)
- **Full Revelation**: Complete honesty ($\alpha = 0$ always)
- **Random Ambiguity**: Uniform random $\alpha \sim \text{Uniform}(0, 1)$

**4. Heuristic Negotiation Agents**
- **Tit-for-Tat**: Match opponent's concession behavior
- **Boulware**: Concede slowly until deadline approaches
- **Conceder**: Concede quickly to reach agreement
- **Time-Dependent**: Concession rate depends on remaining time

**5. Learning-Based Baselines**
- **MARL without Ambiguity**: Standard QMIX or MAPPO without strategic ambiguity
- **Separate Learning**: Agents learn independently (no coordination)
- **Centralized Controller**: Single policy controls all agents (ignores decentralization)

**6. Existing Negotiation Frameworks**
- **ANL Agents**: Agents from Automated Negotiating Agents Competition
- **Negotiation with Opponent Modeling**: Learn opponent preferences without explicit signaling

### 8.3 Evaluation Metrics

**Primary Metrics**:

1. **Social Welfare**: $W(x^*) = \sum_i u_i(x^*, \theta_i)$
   - Normalized by complete information optimal: $W_{\text{norm}} = W(x^*) / W^{\text{opt}}$

2. **Agreement Rate**: Percentage of negotiations reaching agreement before deadline

3. **Pareto Efficiency**: Distance to Pareto frontier
   $$d_{\text{Pareto}} = \max_{x \in \text{Pareto}} \min_i (u_i(x, \theta_i) - u_i(x^*, \theta_i))$$

4. **Fairness**: Variance in individual utilities
   $$\text{Fairness} = 1 - \frac{\text{Var}(u_1, \ldots, u_N)}{\text{Var}_{\max}}$$

**Information Metrics**:

5. **Information Revealed**: Average information content of signals
   $$\bar{I} = \frac{1}{NT} \sum_{i,t} I(\theta_i; s_i^t \mid \alpha_i^t)$$

6. **Belief Accuracy**: KL divergence between final beliefs and true types
   $$\text{Belief Error} = \frac{1}{N} \sum_i D_{KL}(\delta_{\theta_{-i}^*} \| b_i^T)$$

7. **Ambiguity Dynamics**: Evolution of average ambiguity over time
   $$\bar{\alpha}(t) = \frac{1}{N} \sum_i \alpha_i^t$$

**Strategic Metrics**:

8. **Exploitation Resistance**: Utility loss when facing adversarial opponents
   $$\text{Exploit}_i = u_i^{\text{normal}} - u_i^{\text{vs. adversary}}$$

9. **Robustness**: Performance across different opponent types and strategies

10. **Sample Efficiency**: Number of training episodes to reach performance threshold

**Human Alignment Metrics** (for human studies):

11. **Human Preference**: Win rate in human preference comparisons

12. **Naturalness**: Human ratings of negotiation naturalness (1-5 scale)

13. **Trustworthiness**: Human ratings of agent trustworthiness (1-5 scale)

### 8.4 Hyperparameters and Training Details

**Network Architecture**:

```
State Encoder:
  - Input: History (variable length) + Type (fixed) + Belief (fixed)
  - History Encoder: LSTM(hidden=256) or Transformer(heads=4, layers=2)
  - Type Encoder: MLP(128-64)
  - Belief Encoder: MLP(128-64)
  - Concatenation + MLP(512-256-128) → state embedding

Ambiguity Policy:
  - Input: state embedding (128)
  - Hidden: MLP(128-64-32)
  - Output: Beta parameters (2) → Beta distribution over [0, 1]

Proposal Policy:
  - Input: state embedding (128)
  - Hidden: MLP(128-64)
  - Output: Proposal parameters (2M for Gaussian per issue)

Response Policy:
  - Input: state embedding (128) + proposal encoding (M)
  - Hidden: MLP(128+M-64-32)
  - Output: Accept probability (1) via sigmoid

Value Network:
  - Input: state embedding (128)
  - Hidden: MLP(128-64-32)
  - Output: Value estimate (1)

Belief Network (optional learned update):
  - Input: Previous belief (2|Θ|) + Observations (variable)
  - Hidden: MLP(256-128-64)
  - Output: Updated belief parameters (2|Θ|)
```

**Training Hyperparameters**:

```
# Environment
num_agents: 2-5 (depending on domain)
time_horizon: T = 20 (bilateral), T = 30 (coalition), T = 50 (multi-issue)
discount_factor: γ = 0.99

# Optimization
optimizer: Adam
learning_rate: 3e-4 (policies), 1e-3 (value functions)
learning_rate_schedule: Linear decay to 1e-5
batch_size: 256
replay_buffer_size: 100000
update_frequency: every 10 episodes
num_updates_per_step: 5
target_network_update_frequency: 100 episodes (hard) or τ = 0.005 (soft)

# PPO (for policy updates)
ppo_epochs: 10
ppo_clip_epsilon: 0.2
gae_lambda: 0.95
value_loss_coef: 0.5
entropy_coef: 0.01 (annealed to 0.001)
max_grad_norm: 0.5

# Training
max_episodes: 50000 (bilateral), 100000 (complex domains)
eval_frequency: every 1000 episodes
eval_episodes: 100
early_stopping: yes (patience = 5000 episodes)

# Reward weights
lambda_reveal: 0.1
lambda_learn: 0.05
lambda_progress: 0.2
c_time: 0.01

# Signal generation
noise_distribution: Uniform (for discrete), Gaussian (for continuous)
signal_resolution: 10 levels (for discretization)
```

**Computational Resources**:

- **Hardware**: 8 NVIDIA V100 GPUs (32GB each)
- **Training time**: 
  - Bilateral: ~6 hours
  - Coalition: ~12 hours
  - Multi-issue: ~24 hours
- **Evaluation**: 1 GPU, ~1 hour for comprehensive evaluation

### 8.5 Dataset Specifications

**Synthetic Datasets**:

1. **Uniform Types**: Types sampled uniformly from $\Theta$
   - Used for general training and evaluation
   - Ensures coverage of diverse preferences

2. **Structured Types**: Types sampled from structured distributions
   - **Complementary**: Agent preferences are negatively correlated (conflict)
   - **Synergistic**: Agent preferences are positively correlated (cooperation)
   - **Mixed**: Random mixture of complementary and synergistic

3. **Adversarial Types**: Types designed to test robustness
   - **Strategic**: Agents with highly strategic preferences
   - **Extreme**: Agents with extreme preferences (e.g., all weight on one issue)
   - **Deceptive**: Agents incentivized to misrepresent

**Human Demonstration Datasets**:

1. **Expert Negotiations**: Collected from professional negotiators
   - **Source**: Crowdsourcing platform with verified negotiators
   - **Size**: 1000 negotiation episodes (bilateral), 500 (multi-party)
   - **Annotation**: Preference types, outcomes, satisfaction ratings
   - **Use**: Supervised pretraining, human alignment evaluation

2. **Naive Negotiations**: Collected from general population
   - **Source**: Crowdsourcing platform (broader population)
   - **Size**: 5000 negotiation episodes
   - **Use**: Understanding typical human behavior, baseline comparison

**Domain-Specific Datasets**:

1. **Business Negotiations**: Based on real business deals
   - **Source**: Anonymized contract databases, case studies
   - **Size**: 200 realistic scenarios
   - **Features**: Complex utilities, constraints, legal considerations

2. **Diplomatic Negotiations**: Based on historical international negotiations
   - **Source**: Diplomatic archives, historical records
   - **Size**: 100 scenarios (territorial, trade, alliance)
   - **Features**: Multi-party, long-term consequences, reputational effects

**Data Split**:
- Training: 70%
- Validation: 15%
- Test: 15% (held out, not used during development)

**Data Augmentation**:
- **Permutation**: Swap agent roles and issue ordering
- **Scaling**: Scale utilities while preserving relative preferences
- **Noise**: Add small perturbations to test robustness

---

## 9. Implementation Details

### 9.1 Framework Recommendations

**Primary Framework: PyTorch**

Recommended for its flexibility, dynamic computation graphs, and extensive ecosystem.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta, Normal, Categorical

# Key libraries
import numpy as np
import gym  # For environment interface
import wandb  # For logging and monitoring
import ray  # For parallel training (optional)
```

**Alternative: TensorFlow/JAX**

TensorFlow 2.x with Keras API or JAX for hardware acceleration and automatic differentiation.

**Environment: OpenAI Gym Interface**

```python
class SAPIENTNegotiationEnv(gym.Env):
    """
    Custom Gym environment for SAPIENT negotiations
    """
    def __init__(self, num_agents, domain_config):
        self.num_agents = num_agents
        self.config = domain_config
        
        self.action_space = gym.spaces.Dict({
            'ambiguity': gym.spaces.Box(0, 1, shape=(1,)),
            'proposal': gym.spaces.Box(0, 1, shape=(domain_config['num_issues'],)),
            'response': gym.spaces.Discrete(2)
        })
        
        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Sequence(gym.spaces.Box(...)),
            'type': gym.spaces.Box(0, 1, shape=(type_dim,)),
            'belief': gym.spaces.Box(0, 1, shape=(belief_dim,))
        })
    
    def reset(self):
        # Sample new types
        # Initialize history
        # Reset beliefs to prior
        return observations
    
    def step(self, actions):
        # Process actions from all agents
        # Update history
        # Check for agreement
        # Compute rewards
        return observations, rewards, done, info
```

### 9.2 Network Architectures

**State Encoder Implementation**:

```python
class StateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # History encoder (LSTM)
        self.history_lstm = nn.LSTM(
            input_size=config['history_input_dim'],
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # Type encoder
        self.type_encoder = nn.Sequential(
            nn.Linear(config['type_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Belief encoder
        self.belief_encoder = nn.Sequential(
            nn.Linear(config['belief_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 + 64 + config['time_embed_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, history, type_vec, belief_vec, time_step):
        # Encode history
        h_enc, (h_n, c_n) = self.history_lstm(history)
        h_final = h_n[-1]  # Take last layer's hidden state
        
        # Encode type
        type_enc = self.type_encoder(type_vec)
        
        # Encode belief
        belief_enc = self.belief_encoder(belief_vec)
        
        # Time encoding (sinusoidal)
        time_enc = self.sinusoidal_encoding(time_step)
        
        # Concatenate and fuse
        combined = torch.cat([h_final, type_enc, belief_enc, time_enc], dim=-1)
        state_embedding = self.fusion(combined)
        
        return state_embedding
    
    def sinusoidal_encoding(self, time_step, d=32):
        # Positional encoding for time
        position = time_step.unsqueeze(-1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * 
                             -(np.log(10000.0) / d))
        encoding = torch.zeros(time_step.shape[0], d)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding
```

**Ambiguity Policy Network**:

```python
class AmbiguityPolicy(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output Beta distribution parameters
        self.alpha_head = nn.Linear(32, 1)
        self.beta_head = nn.Linear(32, 1)
    
    def forward(self, state):
        features = self.network(state)
        
        # Beta distribution parameters (must be positive)
        alpha = F.softplus(self.alpha_head(features)) + 1.0
        beta = F.softplus(self.beta_head(features)) + 1.0
        
        return Beta(alpha, beta)
```

**Proposal Policy Network**:

```python
class ProposalPolicy(nn.Module):
    def __init__(self, state_dim, num_issues):
        super().__init__()
        
        self.num_issues = num_issues
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Separate heads for each issue (Gaussian parameters)
        self.mean_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_issues)
        ])
        self.logstd_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_issues)
        ])
    
    def forward(self, state):
        features = self.network(state)
        
        means = []
        stds = []
        
        for i in range(self.num_issues):
            mean = torch.sigmoid(self.mean_heads[i](features))  # [0, 1]
            logstd = self.logstd_heads[i](features)
            std = torch.exp(logstd).clamp(min=0.01, max=0.5)
            
            means.append(mean)
            stds.append(std)
        
        means = torch.cat(means, dim=-1)
        stds = torch.cat(stds, dim=-1)
        
        return Normal(means, stds)
```

**Value Network**:

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state):
        return self.network(state)
```

### 9.3 Training Procedures

**Full Training Pipeline**:

```python
class SAPIENTTrainer:
    def __init__(self, config):
        self.config = config
        self.env = SAPIENTNegotiationEnv(config)
        
        # Initialize agents
        self.agents = [Agent(i, config) for i in range(config['num_agents'])]
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # Logging
        wandb.init(project="sapient", config=config)
    
    def train(self):
        for episode in range(self.config['max_episodes']):
            # Collect episode
            transitions = self.collect_episode()
            
            # Store in buffer
            self.replay_buffer.add(transitions)
            
            # Update policies
            if episode % self.config['update_freq'] == 0:
                for _ in range(self.config['num_updates']):
                    batch = self.replay_buffer.sample(self.config['batch_size'])
                    losses = self.update_agents(batch)
                    
                    # Log losses
                    wandb.log(losses, step=episode)
            
            # Evaluate
            if episode % self.config['eval_freq'] == 0:
                eval_metrics = self.evaluate()
                wandb.log(eval_metrics, step=episode)
                
                # Save checkpoint
                self.save_checkpoint(episode)
        
        wandb.finish()
    
    def collect_episode(self):
        obs = self.env.reset()
        transitions = []
        
        done = False
        while not done:
            # Get actions from all agents
            actions = {}
            for i, agent in enumerate(self.agents):
                actions[i] = agent.select_action(obs[i])
            
            # Environment step
            next_obs, rewards, done, info = self.env.step(actions)
            
            # Store transition
            transitions.append({
                'obs': obs,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_obs,
                'done': done
            })
            
            obs = next_obs
        
        return transitions
    
    def update_agents(self, batch):
        losses = {}
        
        for i, agent in enumerate(self.agents):
            agent_losses = agent.update(batch)
            losses[f'agent_{i}'] = agent_losses
        
        return losses
    
    def evaluate(self):
        # Run evaluation episodes
        eval_rewards = []
        agreement_rates = []
        
        for _ in range(self.config['eval_episodes']):
            obs = self.env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                actions = {}
                for i, agent in enumerate(self.agents):
                    actions[i] = agent.select_action(obs[i], deterministic=True)
                
                next_obs, rewards, done, info = self.env.step(actions)
                episode_reward += sum(rewards.values())
                obs = next_obs
            
            eval_rewards.append(episode_reward)
            agreement_rates.append(info['agreement'])
        
        return {
            'eval/mean_reward': np.mean(eval_rewards),
            'eval/agreement_rate': np.mean(agreement_rates)
        }
```

**Agent Update Procedure**:

```python
class Agent:
    def __init__(self, agent_id, config):
        self.id = agent_id
        
        # Networks
        self.encoder = StateEncoder(config)
        self.ambiguity_policy = AmbiguityPolicy(config['state_dim'])
        self.proposal_policy = ProposalPolicy(config['state_dim'], config['num_issues'])
        self.response_policy = ResponsePolicy(config['state_dim'])
        self.value_network = ValueNetwork(config['state_dim'])
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.ambiguity_policy.parameters()) +
            list(self.proposal_policy.parameters()) +
            list(self.response_policy.parameters()) +
            list(self.value_network.parameters()),
            lr=config['lr']
        )
    
    def update(self, batch):
        # Extract data
        states = batch['states']
        actions_alpha = batch['actions_alpha']
        actions_proposal = batch['actions_proposal']
        actions_response = batch['actions_response']
        returns = batch['returns']
        advantages = batch['advantages']
        old_log_probs = batch['old_log_probs']
        
        # PPO update
        for _ in range(self.config['ppo_epochs']):
            # Forward pass
            state_embeddings = self.encoder(states)
            
            # Ambiguity policy
            alpha_dist = self.ambiguity_policy(state_embeddings)
            alpha_log_probs = alpha_dist.log_prob(actions_alpha)
            alpha_entropy = alpha_dist.entropy()
            
            # Proposal policy (similar)
            # Response policy (similar)
            
            # Compute losses
            ratio = torch.exp(alpha_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['ppo_clip'], 
                                1 + self.config['ppo_clip']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value_network(state_embeddings)
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -alpha_entropy.mean()
            
            # Total loss
            loss = (policy_loss + 
                    self.config['value_coef'] * value_loss + 
                    self.config['entropy_coef'] * entropy_loss)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 
                                           self.config['max_grad_norm'])
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': alpha_entropy.mean().item()
        }
```

### 9.4 Logging and Monitoring

**Comprehensive Logging**:

```python
# Key metrics to log
metrics = {
    # Training metrics
    'train/policy_loss': policy_loss,
    'train/value_loss': value_loss,
    'train/entropy': entropy,
    'train/episode_reward': episode_reward,
    'train/episode_length': episode_length,
    
    # Ambiguity metrics
    'ambiguity/mean': mean_ambiguity,
    'ambiguity/std': std_ambiguity,
    'ambiguity/min': min_ambiguity,
    'ambiguity/max': max_ambiguity,
    'ambiguity/over_time': ambiguity_trajectory,
    
    # Negotiation metrics
    'negotiation/agreement_rate': agreement_rate,
    'negotiation/social_welfare': social_welfare,
    'negotiation/pareto_efficiency': pareto_efficiency,
    'negotiation/fairness': fairness,
    
    # Information metrics
    'info/mutual_information': mutual_information,
    'info/belief_entropy': belief_entropy,
    'info/belief_accuracy': belief_accuracy,
    
    # Performance metrics
    'perf/training_time': training_time,
    'perf/gpu_memory': gpu_memory,
    'perf/throughput': samples_per_second
}

# Log to Weights & Biases
wandb.log(metrics, step=episode)

# Save visualizations
fig = plot_ambiguity_trajectory(ambiguity_over_time)
wandb.log({"ambiguity_trajectory": wandb.Image(fig)})

fig = plot_preference_calibration(beliefs_over_time, true_types)
wandb.log({"preference_calibration": wandb.Image(fig)})
```

**Visualization Tools**:

```python
def plot_ambiguity_trajectory(ambiguity_data):
    """Plot how ambiguity evolves over negotiation"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for agent_id, trajectory in ambiguity_data.items():
        ax.plot(trajectory, label=f'Agent {agent_id}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Ambiguity Level')
    ax.set_title('Ambiguity Evolution During Negotiation')
    ax.legend()
    ax.grid(True)
    return fig

def plot_preference_calibration(beliefs, true_types):
    """Visualize belief convergence to true types"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(beliefs), figsize=(15, 5))
    for i, (agent_beliefs, true_type) in enumerate(zip(beliefs, true_types)):
        ax = axes[i]
        # Plot belief distribution over time
        for t, belief in enumerate(agent_beliefs):
            ax.plot(belief, alpha=0.3, color='blue')
        # Plot true type
        ax.axvline(true_type, color='red', linestyle='--', label='True Type')
        ax.set_title(f'Agent {i} Belief Evolution')
        ax.set_xlabel('Type Space')
        ax.set_ylabel('Belief Density')
        ax.legend()
    return fig
```

---

## 10. Limitations and Future Work

### 10.1 Known Limitations

1. **Computational Complexity**
   - Training requires significant computational resources (GPU hours)
   - Scales quadratically with number of agents due to belief updates
   - Large outcome spaces become intractable without approximations

2. **Assumption of Rationality**
   - Framework assumes agents are rational utility maximizers
   - May not capture human irrationality, emotions, or bounded rationality
   - Real humans may not follow game-theoretic optimal strategies

3. **Information Structure**
   - Assumes specific signaling model (linear interpolation with noise)
   - Real communication is richer and more nuanced
   - Natural language integration is challenging

4. **Convergence Guarantees**
   - Theoretical convergence guarantees require strong assumptions
   - In practice, may converge to local optima
   - Multi-agent learning is non-stationary and unstable

5. **Generalization**
   - Agents trained on specific domains may not generalize to novel negotiation settings
   - Transfer learning across different negotiation types is limited
   - Overfitting to training distribution of types

6. **Evaluation Challenges**
   - Difficult to evaluate without ground truth for optimal ambiguity
   - Human evaluation is expensive and subjective
   - Metrics may not capture all aspects of negotiation quality

### 10.2 Scalability Considerations

**Scaling to More Agents**:

- **Challenge**: $O(N^2)$ complexity in belief updates and communication
- **Solutions**:
  - Hierarchical negotiation structures (subgroups)
  - Approximate belief updates (sampling, mean-field approximations)
  - Attention mechanisms for selective belief updates

**Scaling to Complex Domains**:

- **Challenge**: Exponential growth of outcome space with issues
- **Solutions**:
  - Factored representations (assume independence across issues)
  - Neural function approximators for outcomes
  - Constraint-based search in outcome space

**Scaling to Longer Horizons**:

- **Challenge**: Credit assignment over long negotiation sequences
- **Solutions**:
  - Hierarchical policies (high-level strategy, low-level tactics)
  - Temporal abstraction (options framework)
  - Improved temporal credit assignment (transformers, attention)

### 10.3 Extensions and Open Problems

**Theoretical Extensions**:

1. **Incomplete Information**: Extend to settings where agents don't know structure of opponent utilities
2. **Dynamic Types**: Allow types to change over time (preferences evolve during negotiation)
3. **Repeated Negotiations**: Analyze repeated game dynamics with reputation effects
4. **Mechanism Design**: Design protocols that incentivize efficient ambiguity revelation

**Algorithmic Extensions**:

1. **Meta-Learning**: Learn to quickly adapt to new opponents and domains
2. **Transfer Learning**: Transfer negotiation skills across domains
3. **Hierarchical Negotiation**: Decompose complex negotiations into sub-negotiations
4. **Multi-Modal Communication**: Integrate natural language, gestures, and structured signals

**Application Extensions**:

1. **Human-AI Negotiation**: Optimize for negotiation with humans (human-in-the-loop training)
2. **Multi-Modal Domains**: Extend to negotiations with vision, language, and structured data
3. **Real-World Deployment**: Apply to e-commerce, supply chain, diplomatic scenarios
4. **Fairness and Ethics**: Ensure negotiation outcomes satisfy fairness constraints

**Open Research Questions**:

1. **Optimal Ambiguity**: What is the theoretically optimal level of ambiguity for different negotiation contexts?
2. **Learning from Humans**: How can we effectively learn strategic ambiguity from human demonstrations?
3. **Robustness**: How robust are learned policies to adversarial opponents and distribution shift?
4. **Interpretability**: Can we interpret and explain why agents choose specific ambiguity levels?
5. **Cultural Differences**: How do cultural norms affect optimal ambiguity and communication strategies?

---

## 11. Implementation Notes

**Code Organization**:

```
sapient/
├── agents/
│   ├── agent.py              # Base agent class
│   ├── policies.py           # Policy networks (ambiguity, proposal, response)
│   ├── value_networks.py     # Value function networks
│   └── beliefs.py            # Belief update mechanisms
├── environments/
│   ├── base_env.py           # Base negotiation environment
│   ├── bilateral.py          # Bilateral bargaining
│   ├── coalition.py          # Coalition formation
│   └── multi_issue.py        # Multi-issue negotiation
├── algorithms/
│   ├── ppo.py                # PPO implementation
│   ├── qmix.py               # QMIX implementation
│   └── belief_update.py      # Belief update algorithms
├── utils/
│   ├── replay_buffer.py      # Experience replay buffer
│   ├── logging.py            # Logging and visualization
│   ├── metrics.py            # Evaluation metrics
│   └── signal_generation.py  # Ambiguous signal generation
├── configs/
│   ├── bilateral_config.yaml # Bilateral bargaining config
│   ├── coalition_config.yaml # Coalition formation config
│   └── training_config.yaml  # Training hyperparameters
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── visualize.py          # Visualization tools
└── tests/
    ├── test_agents.py        # Unit tests for agents
    ├── test_environments.py  # Unit tests for environments
    └── test_algorithms.py    # Unit tests for algorithms
```

**Installation**:

```bash
# Clone repository
git clone https://github.com/username/sapient.git
cd sapient

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Dependencies** (requirements.txt):

```
torch>=1.12.0
numpy>=1.21.0
gym>=0.21.0
wandb>=0.12.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
tqdm>=4.62.0
scipy>=1.7.0
pandas>=1.3.0
```

**Quick Start**:

```bash
# Train bilateral bargaining agents
python scripts/train.py --config configs/bilateral_config.yaml

# Evaluate trained agents
python scripts/evaluate.py --checkpoint checkpoints/bilateral_best.pt

# Visualize results
python scripts/visualize.py --results results/bilateral_evaluation.json
```

**Configuration Example** (bilateral_config.yaml):

```yaml
environment:
  type: bilateral_bargaining
  num_agents: 2
  num_issues: 3
  time_horizon: 20
  
agent:
  state_dim: 128
  hidden_dims: [256, 128, 64]
  
training:
  max_episodes: 50000
  batch_size: 256
  learning_rate: 0.0003
  gamma: 0.99
  ppo_epochs: 10
  ppo_clip: 0.2
  
reward:
  lambda_reveal: 0.1
  lambda_learn: 0.05
  lambda_progress: 0.2
  c_time: 0.01
  
logging:
  log_frequency: 100
  eval_frequency: 1000
  save_frequency: 5000
```

**Testing**:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_agents.py::test_ambiguity_policy

# Run with coverage
pytest --cov=sapient tests/
```

---

## 12. References

**Automated Negotiation**:

1. Jennings, N. R., Faratin, P., Lomuscio, A. R., Parsons, S., Wooldridge, M. J., & Sierra, C. (2001). Automated negotiation: prospects, methods and challenges. *Group Decision and Negotiation*, 10(2), 199-215.

2. Baarslag, T., Hendrikx, M. J., Hindriks, K. V., & Jonker, C. M. (2016). Learning about the opponent in automated bilateral negotiation: a comprehensive survey of opponent modeling techniques. *Autonomous Agents and Multi-Agent Systems*, 30(5), 849-898.

3. Fatima, S., Kraus, S., & Wooldridge, M. (2014). *Principles of automated negotiation*. Cambridge University Press.

**Multi-Agent Reinforcement Learning**:

4. Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for decentralised multi-agent reinforcement learning. *ICML*.

5. Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2021). The surprising effectiveness of PPO in cooperative multi-agent games. *NeurIPS*.

6. Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S. (2018). Counterfactual multi-agent policy gradients. *AAAI*.

**Game Theory and Mechanism Design**:

7. Myerson, R. B. (1991). *Game theory: analysis of conflict*. Harvard University Press.

8. Osborne, M. J., & Rubinstein, A. (1994). *A course in game theory*. MIT Press.

9. Nisan, N., Roughgarden, T., Tardos, E., & Vazirani, V. V. (2007). *Algorithmic game theory*. Cambridge University Press.

**Information Theory**:

10. Cover, T. M., & Thomas, J. A. (2006). *Elements of information theory*. John Wiley & Sons.

**Strategic Communication**:

11. Crawford, V. P., & Sobel, J. (1982). Strategic information transmission. *Econometrica*, 1431-1451.

12. Lipman, B. L., & Seppi, D. J. (1995). Robust inference in communication games with partial provability. *Journal of Economic Theory*, 66(2), 370-405.

**Preference Learning**:

13. Fürnkranz, J., & Hüllermeier, E. (Eds.). (2010). *Preference learning*. Springer.

14. Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS*.

**Reinforcement Learning**:

15. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT Press.

16. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

17. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*.

**Applications**:

18. Williams, C. R., Robu, V., Gerding, E. H., & Jennings, N. R. (2014). Negotiating concurrently with unknown opponents in complex, real-time domains. *ECAI*.

19. An, B., Bazzan, A., Leite, J., Villata, S., & Wooldridge, M. (2017). Agent-based negotiation in multi-agent systems. *AAMAS*.

**Ethics and Fairness**:

20. Mökander, J., Schuett, J., Kirk, H. R., & Floridi, L. (2023). Auditing large language models: a three-layered approach. *AI and Ethics*, 1-31.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Authors**: SAPIENT Research Team  
**Contact**: sapient-research@example.com  
**License**: MIT

---

*This technical specification is intended for research and educational purposes. Implementation details may require adjustment based on specific application requirements and constraints.*
