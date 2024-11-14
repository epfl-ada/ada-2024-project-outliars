# The Grandmaster's Path - Difficulty Metrics for Navigating the Wikispeedia Game

---

## Abstract

Since we love the concept behind Wikispeedia, we would like to make it even more enticing and to do that we need a 
scoring system. 
Our goal is to further gamify Wikispeedia by formulating a user-independent metric which characterizes the difficulty 
of a given mission, giving feedback during the mission, and updating the player's rating upon termination. 
We seek to formalize our intuitive understanding of what makes a mission well executed:
missions with close-to-shortest paths, with quick and accurate navigation.
Finally, we explore what makes the best players great and attempt to set new players on this path by giving them 
feedback, encouraging when they make good moves and giving hints whtn they get stuck. 
This decision is the centerpiece of our project and is based an algorithm for predicting whether a player will finish 
a particular game instance based on source-destination difficultly, player rating, and the first $k$ clicks. 

## Research Questions

1. What data about the Wikipedia graph the users navigate should be used for the user-independent metric of difficulty
   (from now on referred to as estimated difficulty)?
   - What are the relative weights assigned to this data and are they correlated?
   - Is it possible to identify users which played the game only a few times and thus have little experience, allowing
     us to exploit their lack of knowledge to extract these weights, which should be as user-independent as possible?
2. If we can sufficiently reliably identify a player, what metric should be used for ranking their performance?
   - Should the metric be as simple as the win rate and the average estimated difficulty of the missions the player
     completed (successfully and unsuccessfully)?
   - Can this score be easily updated based on the current player score and the mission he just completed, similarly to 
     chess?
3. We consider mission duration, deviation from the shortest path length, and high accuracy (low backtracking rate) as
   important indicators of how well the player performed - can this notion be quantified?
   - If this importance can be learned in some way, is it possible to use it for ranking updates
4. Given the estimated difficulty, player rating, and information available from the first $k$ clicks, can we reliably 
   predict whether a player will successfully finish the game?
   - If yes, how many clicks are necessary, how confident are we, and what information is the most valuable?
   - If no, why is this not possible and what does it reveal about the game's dynamics?
5. (Optional) How does a chatbot compare with a human player - what score would it have in this setting?
   - Are there quantitative differences between the way a chatbot plays on different "temperature" settings?
   - Are there qualitative differences between how humans and chatbots play the game?

   

## Proposed Additional Datasets


1. We use logistic regression to fit  with input parameters:
   - shortest path between the source and destination
   - the number of different shortest paths
   - what is the degree of nodes in those paths (explore whether taking an average or maximum has a meaningful impact)
   - how many paths are there which are 1 step longer than the shortest one (sensitivity analysis)

2. Attempt to estimate how well a player did in a particular game with an unsupervised algorithm
   - clustering games in a higher dimensional space
   - properties such as 

3. Track information gathered in the first $k$ clicks
   - how much closer is the player to the final destination
   - did the player choose to go to intermediate steps which are such that getting to the destination from them is more
     difficult than the initial source
   - the number of times the player backtrack
   - quality of the nodes along the path thus far 
     - experiment with various information available at each node, e.g. degree, and how much each contributes

4. 

## Methods

---

## Proposed Timeline

| Milestone               | Description                                         | Estimated Completion Date |
|-------------------------|-----------------------------------------------------|----------------------------|
| Project Milestone 1     |                                                     |                            |
| Project Milestone 2     |                                                     |                            |
| Project Milestone P3    |                                                     |                            |

---

## Organization Within the Team

---

## Questions for TAs (Optional)

---
