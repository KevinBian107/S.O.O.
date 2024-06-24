import numpy as np
import gym
from sklearn.kernel_ridge import KernelRidge

class PendulumControl:
    def __init__(self, agop_dict, num_components):
        self.env = gym.make('Pendulum-v0')
        self.agop_dict = agop_dict
        self.num_components = num_components
        self.principal_vectors = self.extract_principal_components()
        self.model = KernelRidge(alpha=1.0, kernel='rbf')

    def extract_principal_components(self):
        # Assuming the AGOP is stored for the first layer's weights
        agop_matrix = np.array(self.agop_dict['layer1.weight'])

        print(f'agpo matrix shape: {agop_matrix.shape}')

        eigenvalues, eigenvectors = np.linalg.eigh(agop_matrix)
        idx = eigenvalues.argsort()[::-1]
        principal_vectors = eigenvectors[:, idx[:self.num_components]]

        print(f'PCA matrix shape: {principal_vectors.shape}')
        return principal_vectors

    def transform_state(self, state):
        return np.dot(state, self.principal_vectors)

    def fit_model(self, states, actions):
        transformed_states = np.array([self.transform_state(s) for s in states])
        self.model.fit(transformed_states, actions)

    def predict_action(self, state):
        transformed_state = self.transform_state(state)
        predicted_action = self.model.predict([transformed_state])[0]
        return predicted_action

    def control_pendulum(self):
        state = self.env.reset()
        total_reward = 0
        for _ in range(1000):
            action = self.predict_action(state)
            state, reward, done, _ = self.env.step([action])
            total_reward += reward
            if done:
                break
        return total_reward

    def train_and_evaluate(self, train_states, train_actions):
        self.fit_model(train_states, train_actions)
        return self.control_pendulum()