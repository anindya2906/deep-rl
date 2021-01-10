import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNetwork
from replay_buffer import ReplayBuffer

# Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay Buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount Factor
TAU = 1e-3              # For soft update of target parameters
LERANING_RATE = 5e-4    # Learning rate
UPDATE_EVERY = 4        # How often to update the parameters


class Agent():
    """Interacts with the environment and learns from it"""

    def __init__(self, state_size, action_size, seed, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Q Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=LERANING_RATE)

        # Replay Buffer
        self.replay_memory = ReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize timestep for updating every UPDATE_EVERY steps
        self.time_step = 0

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get the max predicted Q values from the target network from next states
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute the loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, traget_model, tau):
        """Soft update model parameters
        theta_target = tau * theta_local + (1 - tau) * theta_target
        """

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def step(self, states, actions, rewards, next_states, dones):

        # Store exprience in replay memory
        self.replay_memory.add(states, actions, rewards, next_states, dones)

        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0:
            if len(self.replay_memory) > BATCH_SIZE:
                experiences = self.replay_memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, epsilon=0.0):
        """Return action for given state as puer current policy"""

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon Greedy
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
