Executive Summary

In the ever-evolving landscape of artificial intelligence, the quest for developing advanced agents capable of conquering complex games is an ongoing challenge. Chess has long been a benchmark for artificial intelligence, and this project aims to leverage advanced reinforcement learning techniques to develop an agent capable of competing at a high level against human and computer opponents. The objective of this project is to design and implement a highly sophisticated chess-playing agent using Proximal Policy Optimization (PPO). The agent will learn to play chess through self-play and interactions with various opponents, adapting and improving its strategies over time.

Introduction

To build a chess agent, we need to define the problem statement as Markov Decision Process (MDP).  The main components of chess learning project are the environment and the agent. The environment contains state space, action space, rewards, transition probabilities, etc. The agent contains policy, RL algorithms, replay buffer etc. In this project, we used Proximal Policy Optimization (PPO) algorithm, which is extension to Trusted Region Policy optimization (TRPO), to train the agent. This is a policy gradient method. The proximal policy optimization (PPO) algorithm separate policy (actor) and value (critic) networks to optimize decision-making. The actor network outputs probability distributions over actions given the current state, whereas the critic network estimates state values, representing expected cumulative future rewards. PPO improves upon policy-based methods by clipping the objective function to limit the size of policy updates, promoting stable learning. PPO enforces a trust region to control the size of policy updates, preventing large deviations. It utilizes multiple optimization epochs and batch sampling for efficient learning. The elaborate detailing of all the above-mentioned components will be given in the later sections of this report.

Approach

The project will explore two different methods for applying the PPO algorithm to the chess environment.
The first method will involve training two different agents with separate neural networks. These two agents will compete against each other, with each agent learning from the other's moves. This approach is known as self-play and has been shown to be effective in training game-playing agents.
The second method will involve training a single agent with a single neural network. This agent will learn to play both sides of the chessboard, meaning it will learn to play as both white and black pieces. This approach is known as joint training and has the advantage of being more computationally efficient since it only requires one agent to be trained.
