Does UNP Belief network serve as a forward model, similar to the Cerebellum function, to facilitate motor action execution by providing a motor plan derived from previous motor control experiences for additional guidance (than just sensory feedback)? Moreover, can this new motor learning process be incorporated into the GDP for future motor controls?

![Alt text](demos/dynamics_model.png)

## FMPPO Control Examples

### Retrained PPO & Fm-PPO Agents on Half-Cheetah Task
All Fm-PPO should start tarining from scratch, with the goal to try to observe how the Fm-Core help the agent to get higher rewards.

PPO trained on 2e5 global steps:

![Alt text](demos/vectorized_half_cheetah/original/ppo_trained.png)

Fm-PPO trained on 2e5 global steps with transfered Fm-core and imitation data:

Transfer Twice Core:

![Alt text](demos/vectorized_half_cheetah/original/fmppo_transfer_2.png)

Transfer Three Times Core:

![Alt text](demos/vectorized_half_cheetah/original/fmppo_transfer_3.png)

PPO and Fm-PPO Agent post-training evaluation on new random environment for same running task as training:

![Alt text](demos/vectorized_half_cheetah/eval/eval_vel1.png)

### Transfer Learning

PPO trained on 2e5 global steps:

![Alt text](demos/vectorized_half_cheetah/jump/ppo_direct.png)

Fm-PPO trained on 2e5 global steps with transfered Fm-core and imitation data:

![Alt text](demos/vectorized_half_cheetah/jump/fmppo_transfer.png)

PPO and Fm-PPO Agent post-training evaluation on new random environment for jump task:

![Alt text](demos/vectorized_half_cheetah/eval/eval_jump.png)

PPO and Fm-PPO Agent post-training evaluation on new random environment for 2x running task:

![Alt text](demos/vectorized_half_cheetah/eval/eval_vel2.png)

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