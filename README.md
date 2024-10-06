Does UNP Belief network serve as a forward model, similar to the Cerebellum function, to facilitate motor action execution by providing a motor plan derived from previous motor control experiences for additional guidance (than just sensory feedback)? Moreover, can this new motor learning process be incorporated into the GDP for future motor controls?

## FMPPO Control Examples

### Retrained PPO & Fm-PPO Agents on Half-Cheetah Task
![Alt text](demos/vectorized_half_cheetah/fmppo_results.png)

![Alt text](demos/vectorized_half_cheetah/ppo_results_twice_trained.png)

<div style="width: 100%; padding: 5px; display: flex; justify-content: center; gap: 20px;">
          <div style="width: 50%; display: flex; flex-direction: column; align-items: center;">
            <video controls autoplay style="width: 100%; height: auto;" muted>
              <source src="../assets/fmppo_demo1.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            <blockquote>Deep-RL Inverted Pendulum agent trained using Fm-PPO</blockquote>
          </div>
          <div style="width: 50%; display: flex; flex-direction: column; align-items: center;">
            <video controls autoplay style="width: 100%; height: auto;" muted>
              <source src="../assets/fmppo_demo2.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            <blockquote>Deep-RL Half Cheetah agent trained using Fm-PPO</blockquote>
          </div>
        </div>
