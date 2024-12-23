# ai-safety-camp-unsearch
My contribution to the AI Safety Camp project UnSearch.
See presentation at: https://docs.google.com/presentation/d/17oTChrZIr7VfLTU8BinJQBfgXH0S9IeyG0UXR_7kZsY/edit?usp=sharing

# My project
I participated in AI Safety Camp in the first half of 2024 as part of the UnSearch team (https://unsearch.org/). The UnSearch team's goal is to understand how transformer based systems may perform search using a toy model of maze solving transformers. 

My specific project was to try doing activation steering on these maze solving transformers, and then analyse the steering vectors in the hopes of gaining some understanding on the internal structure of the models.

I have attempted to create steering vectors for two different tasks:
 - turn to a specific direction
 - terminate path at current position

# Method
I have attempted to create steering vectors for these two tasks by creating specific datasets for both where the transformer does the desired operation. For example, to steer the transformer to turn right in the maze at a specific step, I have created a dataset where at that specific step the transformer always turns right. Then averaging the activations for these mazes when the transformer solves them results in the steering vector. I did this for a specific activation layer. I have also tried using centering, according to https://arxiv.org/abs/2312.03813.

In addition, to get a final "turn right" steering vector, I have subtracted a "turn left" vector from the "turn right" vector, and so on for other directions.

# Results
I have managed to steer the transformer to turn to a specific direction at the first step of the maze to be solved. 

See presentation of project and results: https://docs.google.com/presentation/d/17oTChrZIr7VfLTU8BinJQBfgXH0S9IeyG0UXR_7kZsY/edit?usp=sharing

