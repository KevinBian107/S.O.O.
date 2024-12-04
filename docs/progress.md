## Question In Interest

### Question 1:
Does establishing a Forward Model, similar to the Cerebellum's function, facilitate motor action execution by providing a motor plan derived from previous motor control experiences for additional guidance (compare to pure sensory feedback like in model-free RL)? Moreover, can this new motor learning process be incorporated into the GDP for future motor controls?

- Objective 1: See if such biologically inspired strategy (for example, maybe using mechanistic insight, maybe using neuronal representation as inductive biases) improves performance;
- Objective 2: See if the Forward Model would resemble functionality and behavior of the cerebellum (for example, showing gradual learning of new motor skills).
  - Idealy using a more biological realistic model with more biological realistic task such as the rodent model in VNL.

### Question 2:
With just change of the understanding for the rules of the world (intention changes), can I still find a sub-optimal point in this
training world such that it works still fine or even better than solely one-world-model trained agent in the other world?

## SFM-PPO & SoFM-PPO Control Examples
SFM-PPO is a vriant of PPO where an Supervised Forward Model (SFM) is added to understand the dynamics of teh environment amd SoFM-PPO is a variant of PPO (Sub-Optimal Forward Model) as well where instead of just the SFM, the model is also trained to find "local minimum" or sub-optimal minimum in one task in order for completions in more than just one task. The below are a few Deep-RL Half Cheetah agent demos:

<div style="width: 100%; padding: 5px; display: flex; justify-content: center; gap: 20px;">
          <div style="width: 30%; display: flex; flex-direction: column; align-items: center;">
            <video controls autoplay style="width: 100%; height: auto;" muted>
              <source src="../website/ppo_jump_weird.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            <blockquote>Trained using PPO on "Jump Task" and stuck on weird local minimum</blockquote>
          </div>
          <div style="width: 30%; display: flex; flex-direction: column; align-items: center;">
            <video controls autoplay style="width: 100%; height: auto;" muted>
              <source src="../website/sfmppo_converge_712.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            <blockquote>Trained using SFM-PPO on "Normal Running Task"</blockquote>
          </div>
        <div style="width: 30%; display: flex; flex-direction: column; align-items: center;">
            <video controls autoplay style="width: 100%; height: auto;" muted>
              <source src="../website/sofppo_demo1.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            <blockquote>Trained using SoFM-PPO on "Normal Running Task" but with world model intention bounding</blockquote>
        </div>
</div>

## Latent Created Example
Demonstartion of latent representation of agent during task

<div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
  <video controls autoplay style="width: 100%; height: auto;" muted>
    <source src="../website/latent_demo.mp4" type="video/mp4">
      Your browser does not support the video tag.
  </video>
  <blockquote>Deep-RL SoFM-PPO Half Cheetah agent trained and visualize with PCA</blockquote>
</div>

## Performance in Sensory Delayed Environment
Simple testing in mimicing delayed sensory environment like in real life:

<div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
  <img src="../website/delay_sensory_eval.png" alt="Delay sensory evals" style="width: 100%; height: auto;">
  <blockquote>Deep-RL SoFM-PPO Half Cheetah agent evaluated on sensory delayed environment</blockquote>
</div>