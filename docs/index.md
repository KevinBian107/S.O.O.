# Supervised & Sub-optimality Forward Model
I am currently researching into the theoretical aspects of <strong>Continual Learning</strong> and <strong>Transfer Learning</strong> with advising from professor <a href="https://talmopereira.com/" target="_blank">Talmo Pereira</a> in the Salk Institute. I try to frame the general problem of Continual Learning from the perspective of Reinforcement Learning & Cognitive Neuroscience, hoping to develop algorithms that utilize the same strategies of "how we learn" onto an artificial agent. Moreover, I hope that such development can serve more than an engineering improvmenet, but rather contributing back to the neuroscience community to understand more about ourselves.

I am developing a conceptual framework that may be used for families of algorithms and I am trying to build internal representations for artificial agents through designing a <strong>Forward Model</strong>, one similar to
what we think the Cerabellum is doing in human brain, so their learned skills in one task can be modularized and transferable to other control tasks. The belows are some training results on classical control problems using 
our <strong>SoFM-PPO</strong> algorithm (an adaptation using Supervised Forward Model and Proximal Policy Optimization).

## Biological Context:
The cerebellum have been long theorized to play an crucial rule in motor control and learning (Forward modeling). Corollary discharge encodes a efferent copy of the motor command to be processed to predict the consequences of actions before sensory feedback is available. Such process would help us predicts how the sensory state of our body will change and how should these actions be performed, achieving better performances in control.

Using examples from (Albert and Shadmehr, 2018), with the starting and ending positions in hand, the parietal regions of your cerebral cortex compute the path of the arm that connects these positions in space the trajectory of the movement. After the trajectory is determined, your primary motor cortex and other associated pre-motor areas then carefully transform this sensory signal into a motor plan, namely the patterns of muscle contraction that will move your arm along the desired path towards the coffee.

## SFM & SoFM System Overview
This is an overview of our model:

<div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
  <img src="website/dynamics_model.png" alt="schematics" style="width: 100%; height: auto;">
  <blockquote>Schematics for Forward Models</blockquote>
</div>

## Analogy

### Mountains After Mountains
Every agent that learns in different world serves as a constraint for each other. They explore the world from their own perspective and “pull” each other on the way to prevent any one of them from falling into the “local optimality illusion” that one sees in one moment.

### Wold Model
We believe that policy is to a task specific, but if we build an reward model or an world model, such model may be agonist of the environment or the task, achieveing continual learning purpose.

### Constraint Solving
Such a forward model can be interpreted in various ways from a constraint solving perspective:

1. We can consider such model as a constrained process, forcing the agent to learn while incorporating its previous experiences.
2. Similarly, the forward model's facilitation does not come directly from action facilitation but rather from providing a better representation of the feature space \(\vec \phi(\vec x)\) that the agent has built up.

It's about operating in this <strong>imaginary state</strong>. It is never about <strong>explicitly</strong> facilitating actions but rather <strong>implicitly</strong> making the model understand the dynamics and interactions in this space more comprehensively.