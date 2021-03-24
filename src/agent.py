import numpy as np
import torch
import random
from numpy.core.arrayprint import IntegerFormat

from .hyperparameters import HPHolder, HyperParameters
from .io_data import Reward, Context
from .model import QNetwork, TransformedDiscreteState
from .replay_memory import ReplayBuffer
from .utils import ActionMapping, get_cell_state_v2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DuelingDQNAgent(HPHolder):
    """Interacts with and learns form environment."""

    def __init__(self, hp: HyperParameters, context: Context, action_size, state_shape: TransformedDiscreteState.Shape,
                 mem_size, batch_size, drone_id,
                 replace=1000, chkpt_dir='tmp/dqn'):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(hp)
        self.action_size = action_size
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.drone_id = drone_id
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(action_size)]
        self.learn_step_counter = 0
        self.context = context
        self.last_action = np.random.randint(9)

        self.memory = ReplayBuffer(hp, mem_size, state_shape, action_size)

        self.q_eval = QNetwork(hp,
                               state_shape=self.state_shape,
                               action_size=self.action_size,
                               name=f'q_eval_{drone_id}',
                               chkpt_dir=self.chkpt_dir)

        self.q_next = QNetwork(hp,
                               state_shape=self.state_shape,
                               action_size=self.action_size,
                               name=f'q_next_{drone_id}',
                               chkpt_dir=self.chkpt_dir)

        self.action = None

    def store_transition(self, state: TransformedDiscreteState, action, reward, new_state: TransformedDiscreteState,
                         ksis):
        self.memory.store_transition(state, action, reward, new_state, ksis)

    def to_correct_tensor(self, x, dtype = torch.float):
        return torch.tensor(x, dtype=dtype).to(self.q_eval.device)

    def sample_memory(self):
        state, action, reward, new_state, xi = self.memory.sample_buffer(self.batch_size)

        info_maps = self.to_correct_tensor(state.information_map)
        gamma_coords = self.to_correct_tensor(state.gamma_coords)
        rewards = self.to_correct_tensor(reward)
        actions = self.to_correct_tensor(action, dtype=torch.int64)
        new_info_maps = self.to_correct_tensor(new_state.information_map)
        new_gamma_coords = self.to_correct_tensor(new_state.gamma_coords)
        xis = self.to_correct_tensor(xi)

        return TransformedDiscreteState(info_maps, gamma_coords), actions, rewards, TransformedDiscreteState(
            new_info_maps, new_gamma_coords), xis

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, state: TransformedDiscreteState):

        cell_states = get_cell_state_v2(state.information_map, self.context.grid.zone, self.hp.k, self.hp.l,
                                     state.gamma_coords[self.drone_id], self.action_size)
        # only contains reachable cells ( remove boundaries )
        possible_actions = [a for a in range(self.action_size) if cell_states[a] is not None]
        if len(possible_actions)>1 and 0 in possible_actions:
            possible_actions.remove(0)

        best_actions = [a for a in possible_actions if cell_states[a] == 0]
        eps = 0.1
        if np.random.rand() < eps : # with proba eps
            self.action = random.choice(possible_actions)
        elif self.last_action in possible_actions and np.random.rand() < eps: # with proba eps
            self.action = self.last_action
        elif best_actions: # with proba 1-2*eps
            info_map = self.to_correct_tensor(state.information_map)
            gamma_coords = self.to_correct_tensor(state.gamma_coords)
            _, advantage = self.q_eval.forward(QNetwork.prepare_inputs(info_map, gamma_coords))
            self.action = torch.argmax(advantage).item()
            if self.action not in possible_actions and self.last_action in possible_actions:
                self.action = self.last_action
            elif self.action not in possible_actions :
                self.action = random.choice(best_actions)
        self.last_action = self.action
        return ActionMapping.to_coords(state.gamma_coords[self.drone_id], self.action)

    def compute_ksi(self, state: TransformedDiscreteState, new_state: TransformedDiscreteState, reward: Reward):
        V_s, A_s = self.q_eval.forward(QNetwork.prepare_inputs(self.to_correct_tensor(state.information_map),
                                                               self.to_correct_tensor(state.gamma_coords)))
        V_s_, A_s_ = self.q_next.forward(QNetwork.prepare_inputs(self.to_correct_tensor(new_state.information_map),
                                                                 self.to_correct_tensor(new_state.gamma_coords)))

        q_pred = torch.add(V_s, (A_s - A_s.mean()))[self.action].item()
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean())).max().item()

        return q_pred + self.hp.alpha * (reward.reward + self.hp.lambda_ * q_next - q_pred)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, ksis = self.sample_memory()
        V_s, A_s = self.q_eval.forward(QNetwork.prepare_batched_inputs(states.information_map, states.gamma_coords))
        V_s_, A_s_ = self.q_next.forward(QNetwork.prepare_batched_inputs(states_.information_map, states_.gamma_coords))

        indices = np.arange(self.batch_size)
        q_pred = torch.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

        ksis = ksis.numpy()
        q_target = self.hp.w*ksis[:, self.drone_id]

        for i in range(self.hp.N):
            q_target += (1-self.hp.w) * np.sum(ksis, axis = 1)
        loss = self.q_eval.loss(self.to_correct_tensor(q_target), q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        #self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
