# ViT_speeding
Project in AI - visuomotor behavior cloning using Vision Transformers

The idea behind this project is to develop and compare the driving capabilities of automated car driving neural network models.
Model was trained and evaluated on a data collected from a built simulated enviroment.

The data was acquired from human expert (you just play a simple game), which records input-action pairs (front camera images with the steering angle of the car's steerer and the velocity of a car imitating the angle of the accelerator pedal).
For the simulation environment python based api to Bullet physics engine was used (pybullet https://pybullet.org/wordpress/).

Training of the vision model was made on the google cloud on the Google collab.
The network models were written in python using pytorch library.

For the supervised machine learning pipeline, and thus hyperparameter search, the weights and biases platform was used (https://wandb.ai/site)
The final comparison was testing the ResNet (convolutial network with residual skipping connections) with the recently named SOTA method --- Visual Transformers, which base on the Attention mechanism, which was widely used in the NLP community).


