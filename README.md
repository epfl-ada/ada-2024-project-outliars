# Grandmastering Wikispeedia: Decoding Difficulty and Streamlining Strategies

---

## Abstract

Inspired by the enduring appeal of chess, a centuries-old game that continues to attract new players, we aim to achieve a similar legacy for a modern favorite - Wikispeedia. While we cannot produce a Netflix limited series to popularize the game, we aim to make it more accessible by uncovering what makes the best players excel and why do less skilled players get stuck, consequently paving the way for new players. 

Our goals are the following: 
- Understand the inherent difficulty of missions using a player-independent metric.
- Uncover the strategies of successful players to provide actionable feedback to new players.
- Develop an interpretable model that predicts outcomes based on mission difficulty, player rating, and early-game behavior.

This would enable the implementation of an Elo-like rating system, real-time feedback during gameplay, and the formalization of strategies. As a result, the game becomes more compelling and focused on thoughtful decision-making.

## Research Questions

1. What makes a game inherently difficult?
   - Which properties of the Wikipedia graph the players navigate are important for a good, player-independent metric of difficulty (from now on referred to as estimated difficulty)?
   - Can we use data from the games of players identified as newbies (those with few completed games) to extract information that captures *inherent* difficulty for our estimate?
2. If we can sufficiently reliably identify a player, what metric should be used for rating their performance?
   - Should the metric be as simple as combining the player's win rate and the average estimated difficulty of the games the player tried?
   - Can this score be easily updated based on the current player score and the game he just completed, similarly to the Elo rating system applied to chess?
3. We consider game duration, deviation from the shortest path length, and high accuracy (low backtracking rate) as important indicators of how well the player performed - can this notion be quantified?
   - If this importance can be learned in some way, how can we use it for player rating updates?
4. Given the estimated difficulty, player rating, and information available from the first $k$ clicks, can we reliably predict whether a player will successfully finish the game?
   - If so, how many early clicks are necessary, how confident are we, and what information is the most valuable?
   - If not, why is this not possible and what does it reveal about the game's dynamics?
5. How does the score of a particular game change as the game progresses?
   - Is this a monotonically decreasing function for good players?
   - Are there any identifiable "strategies" that good players use?
6. (Optional) How does a chatbot compare with a human player - what score would it have in this setting?
   - Are there quantitative differences between the way a chatbot plays on different "temperature" settings?
   - Are there qualitative differences between how humans and chatbots play the game?

## Methods

1. Since the provided dataset only contains shortest path information between articles, and we believe there is more relevant data, 
   we carry out a more advanced analysis
   - obtaining all distinct shortest paths between all unique pairs, as well as paths that are $1$ and $2$ steps longer than the shortest paths to judge sensitivity and connectedness
   - since this is very compute intensive, the per pair BFS search is executed in C++

2. Perform semantic analysis of Wikispeedia graph using article embedding
   - attempt to capture features which are not connected to the graph structure
   - use one of the existing tools for this task (e.g. RoBERTa)

3. To address inherent difficulty, we would use logistic regression to fit success against the parameters which we obtained in the previous two steps:
   - shortest path between the source and destination
   - the number of different shortest paths
   - the degree of nodes in those paths (explore whether taking an average or maximum has a meaningful impact)
   - number of paths $1$ step, or possibly $2$ steps, longer than the shortest path (sensitivity analysis)
   - distance between source and destination articles in the embedding space

4. Attempt to estimate how well a player did in a particular game
   - first using unsupervised learning (e.g. clustering) on games
   - if this fails, a simple idea is to look at how much the estimated difficulty fluctuates during the game

5. Gather quantitative information from the first $k$ clicks
   - how much closer is the player to the final destination
   - did the player choose intermediate articles such that reaching the destination from them is more difficult than from the starting point
   - the number of times the player backtracks
   - quality of the nodes along the path thus far (measured by e.g. outdegree)

6. Build two interpretable models for outcome prediction and perform feature importance analysis, perhaps revealing "strategies"
   - a logistic regression model
   - a decision tree

7. (Optional) Use one of the popular chatbots to see quantify how it compares to human players - give it a score


## Proposed Timeline and Organization

| Week      | Task                                                                 | Deliverables                                         |
|-----------|----------------------------------------------------------------------|-----------------------------------------------------|
| Week 1    | Extract data, calculate inherent difficulties using links and graph, set up logistic regression pipeline, determine a player rating system | Preprocessed data, logistic regression setup, initial player metrics |
| Week 2    | Determine player skill level, assess game performance, create embeddings of articles | Player skill metrics, game evaluation framework, article embeddings |
| Week 3    | Use embeddings for semantic analysis to predict game success         | Prediction model leveraging semantic insights       |
| Week 4    | Draw initial conclusions, analyze results, start building a page, optionally test chatbot behavior | Preliminary analysis, basic webpage framework       |
| Week 5    | Final touches                                                        | Final data story, polished results       |


---