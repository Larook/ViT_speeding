# ViT_speeding
Project in AI - visuomotor behavior cloning using Vision Transformers


- Idea
The idea behind this project is to develop and compare the driving capabilities of automated car driving neural network models.
Model was trained and evaluated on a data collected from a built simulated enviroment.


- <img src="https://github.com/Larook/ViT_speeding/blob/master/figures/vit_architecture.png" width="300" height="300">

- <img src="https://github.com/Larook/ViT_speeding/blob/master/figures/simple_idea.png" width="300" height="300">


- Simulation engine

The data was acquired from human expert (you just play a simple game), which records input-action pairs (front camera images with the steering angle of the car's steerer and the velocity of a car imitating the angle of the accelerator pedal).
For the simulation environment python based api to Bullet physics engine was used (pybullet https://pybullet.org/wordpress/).

- <img src="https://github.com/Larook/ViT_speeding/blob/master/figures/overleaf/simulation_environment.png" width="400" height="400">

- Network model training

Training of the vision model was made on the google cloud on the Google collab.
The network models were written in python using pytorch library.

- WandB

For the supervised machine learning pipeline, and thus hyperparameter search, the weights and biases platform was used (https://wandb.ai/site)
The final comparison was testing the ResNet (convolutial network with residual skipping connections) with the recently named SOTA method --- Visual Transformers, which base on the Attention mechanism, which was widely used in the NLP community).

- <img src="https://github.com/Larook/ViT_speeding/blob/master/figures/sweeps/sweep_0/sweeps_0_best_ViT.png" width="500" height="500">


