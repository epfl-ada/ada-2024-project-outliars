# Mastering Wikispeedia: Decoding Difficulty and Winning Strategies

---

## Abstract

Wikispeedia offers a unique blend of learning and fun, and we see an opportunity to enhance the experience further by introducing a scoring system.
Our primary goal is to discover and understand the strategies that lead to success in Wikispeedia, while also formulating a player-independent metric that characterizes the difficulty of a given mission.
We seek to formalize our intuitive sense of what makes a mission well-executed: the path taken is close to the shortest possible, with quick and accurate navigation. In doing so, we hope to uncover what makes the best players excel and attempt to set new players on this path by providing feedback — encouraging good moves and offering hints when they get stuck.
This decision is the centerpiece of our project and relies on building an interpretable model that predicts the outcome of games based on the inherent difficulty of the source-destination pair, the player’s rating, and their first $k$ clicks.

## Research Questions

1. What makes a game inherently difficult?
   - Which properties of the Wikipedia graph the players navigate are important for a good player-independent metric of difficulty (from now on referred to as estimated difficulty)?
   - Can we use data from the games of players identified as newbies (those with few completed games) to extract information that captures *inherent* difficulty for our estimate?
2. If we can sufficiently reliably identify a player, what metric should be used for rating their performance?
   - Should the metric be as simple as combining the player's win rate and the average estimated difficulty of the missions the player tried?
   - Can this score be easily updated based on the current player score and the mission he just completed, similarly to the Elo rating system applied to chess?
3. We consider mission duration, deviation from the shortest path length, and high accuracy (low backtracking rate) as important indicators of how well the player performed - can this notion be quantified?
   - If this importance can be learned in some way, how can we use it for player rating updates?
4. Given the estimated difficulty, player rating, and information available from the first $k$ clicks, can we reliably predict whether a player will successfully finish the game?
   - If so, how many early clicks are necessary, how confident are we, and what information is the most valuable?
   - If not, why is this not possible and what does it reveal about the game's dynamics?
5. TODO nesto sa strategijama i onim grafom koji plota tezinu kroz vrijeme
6. (Optional) How does a chatbot compare with a human player - what score would it have in this setting?
   - Are there quantitative differences between the way a chatbot plays on different "temperature" settings?
   - Are there qualitative differences between how humans and chatbots play the game?

## Methods

1. TODO Extracting data from the Wikipedia graph

TODO, pojasnimo svrhu ovoga, opcenito -> koja metoda je za sta?

2. To address inherent difficulty, we would use logistic regression to fit success against the following parameters:
   - shortest path between the source and destination
   - the number of different shortest paths
   - the degree of nodes in those paths (explore whether taking an average or maximum has a meaningful impact)
   - number of paths $1$ step longer than the shortest path (sensitivity analysis)

3. Attempt to estimate how well a player did in a particular game with an unsupervised algorithm
   - clustering games in a higher dimensional space
   - properties such as TODO

4. Gather quantitative information from the first $k$ clicks
   - how much closer is the player to the final destination
   - did the player choose intermediate articles such that reaching the destination from them is more difficult than from the starting point
   - the number of times the player backtracks
   - quality of the nodes along the path thus far (measured by e.g. outdegree)

5. Build two interpretable models for outcome prediction and perform feature importance analysis
   - a logistic regression model
   - a decision tree

## Proposed timeline and organisation

TODO (moze i samo Week 1: nesto, Week 2: nesto drugo)
| Milestone               | Description                                         | Estimated Completion Date  |
|-------------------------|-----------------------------------------------------|----------------------------|
| X                       |                                                     |                            |
| Y                       |                                                     |                            |
| Z                       |                                                     |                            |

---

## Questions for TAs (Optional)

---
