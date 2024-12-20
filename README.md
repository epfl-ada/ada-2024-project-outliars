# Wikispeedia: The Grandmaster’s Path

[Click here to read the full datastory.](https://kvrancic.notion.site/wikispeedia-the-grandmasters-path?pvs=4)

## Overview

This repository contains the work conducted for our project *"Wikispeedia: The Grandmaster’s Path"* - a journey to bring a "chessified" competitive and analytical ecosystem to the Wikispeedia game.

Our goal has been to create a framework that infuses the game with robust metrics of difficulty, player skill assessment, dynamic feedback, and an Elo-like rating system, ultimately rendering Wikispeedia as strategic and community-driven as classical chess.

**Check out our full data story here:**  
# [**Wikispeedia: The Grandmaster’s Path (Data Story)**](https://kvrancic.notion.site/wikispeedia-the-grandmasters-path?pvs=4)

In the data story, we walk through:

- **Cleaning and Preprocessing**: Ensuring the dataset (originally from West & Leskovec’s work) is consistent, handling temporal mismatches, invalid targets, isolated islands, and shared IP anomalies.
- **Identifying Players**: Differentiating individual players behind shared IPs and ensuring that player performance metrics truly reflect single-user skill progression.
- **Quantifying Inherent Difficulty**:  
  We explored a wide array of features—shortest path metrics, node centrality, semantic embeddings, and LLM created metrics to estimate how challenging a given mission (source-target pair) is, independent of who attempts it.
- **Predicting Outcomes and Live Feedback**:  
  By analyzing partial trajectories and introducing novel features, we predicted the likelihood of success after each click. This dynamic “evaluation bar” approach provides instant, in-game feedback.
- **Developing an Elo-Like Rating System**:  
  We adapted the Elo framework, treating missions as “opponents” with their own difficulty ratings. Over time, player ratings stabilize to reflect genuine skill, guiding matchups and promoting a competitive, growth-oriented environment.
- **Human-Like Strategy and LLM Benchmarks**:  
  We also examined how the framework can be used to test large language models in Wikispeedia, providing a benchmark for evaluating their reasoning and navigation skills, and comparing their strategies to human players.

All of these efforts together build a data-backed ecosystem that encourages improvement, makes the game more engaging, and may set the stage for a future Wikispeedia community with player profiles, rankings, and replay analysis.

## Team Contributions

- **Davor Dobrota**:  
  Researched graph metrics, path statistics, and shortest-path computations. Integrated C++ routines to improve performance and led the inherent difficulty estimation efforts. 

- **Božo Đerek**:  
  Focused on semantic embeddings and collaborated with Davor during prototyping of the difficulty estimation pipeline. Served as an important middle men between different parts of the project.

- **Benedicte Gabelica**:  
  Led the data cleaning and player identification efforts. Developed the Elo system and in-game analysis. The most instrumental member to the pure Data Analysis part of the project.

- **Seif Labib**:  
  Was developing the Jekyll page. Collaborated in discussions and brainstorming sessions, supported other members. 

- **Karlo Vrancic**:  
  Ideated and crafted the project narative and the data story. Engineered features such as fame metric and link probability using LLMs, and did model benchmarking. 

The team members collaborated well and everybody was involved in all parts of the project. Everybody was ready to step in and help when needed.

---

For the full narrative, key insights, and visualizations, don’t forget to read our [Data Story](https://kvrancic.notion.site/wikispeedia-the-grandmasters-path?pvs=4). Feel free to contact us with questions or collaborate on extending these ideas further!