
# Soft Actor Critic - Reinforcement Learning With Trainable Alpha

![SAC-FinalResultVideo](https://github.com/user-attachments/assets/e45c6118-0414-4e29-9ede-58a7388f535a)


A soft actor critic implementation, based on the Haarnoja etc al. paper (https://arxiv.org/pdf/1801.01290), with a learnable alpha value instead of the regular set value.

Utilizes priority replay with a sum-tree approach for faster retrieval. Trialled with normalization pre-processing functions specific for the intended environment.

Designed to work with OpenAI Gym - supports multiple environments to parallelize training and tested on OpenAI Gym Humanoid V4.


