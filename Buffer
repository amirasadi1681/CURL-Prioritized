import numpy as np
import torch
import random
from skimage.util.shape import view_as_windows

# --- load SumTree for PER ---
try:
    from memory.tree import SumTree
except Exception as e:
    raise ImportError("SumTree not found at memory/tree.py") from e


# --- Prioritized Replay Buffer for CURL ---
class PrioritizedReplayBufferCURL:
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,
                 image_size=84, transform=None, eps=1e-2, alpha=0.6, beta=0.4):
        # core params
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform

        # buffer memory
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # PER setup
        self.tree = SumTree(size=capacity)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        self.idx = 0
        self.full = False

    # --- store transition ---
    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], 1.0 - float(done))
        self.tree.add(self.max_priority, self.idx)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    # --- internal sampling (PER) ---
    def _sample_indices(self, batch_size):
        current_size = self.capacity if self.full else self.idx
        total = self.tree.total
        segment = total / batch_size
        sample_idxs, tree_idxs = [], []
        priorities = np.empty((batch_size, 1), dtype=np.float32)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            t_idx, prio, data_idx = self.tree.get(s)
            priorities[i, 0] = prio
            tree_idxs.append(t_idx)
            sample_idxs.append(data_idx)

        probs = priorities / total
        weights = (current_size * probs) ** (-self.beta)
        weights = weights / weights.max()
        return np.array(sample_idxs, np.int64), np.array(tree_idxs, np.int64), weights.astype(np.float32)

    # --- RL sample ---
    def sample_proprio(self):
        idxs, _, _ = self._sample_indices(self.batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    # --- CPC sample ---
    def sample_cpc(self):
        idxs, _, _ = self._sample_indices(self.batch_size)
        obses_u8 = self.obses[idxs]
        next_u8 = self.next_obses[idxs]
        pos_u8 = obses_u8.copy()

        obses = random_crop(obses_u8, self.image_size)
        next_obs = random_crop(next_u8, self.image_size)
        pos = random_crop(pos_u8, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obs = torch.as_tensor(next_obs, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos)
        return obses, actions, rewards, next_obs, not_dones, cpc_kwargs

    # --- RL sample (PER) ---
    def sample_proprio_per(self):
        idxs, tree_idxs, weights = self._sample_indices(self.batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        weights = torch.as_tensor(weights, device=self.device)
        return obses, actions, rewards, next_obses, not_dones, weights, tree_idxs

    # --- CPC sample (PER) ---
    def sample_cpc_per(self):
        idxs, tree_idxs, weights = self._sample_indices(self.batch_size)
        obses_u8 = self.obses[idxs]
        next_u8 = self.next_obses[idxs]
        pos_u8 = obses_u8.copy()

        obses = random_crop(obses_u8, self.image_size)
        next_obs = random_crop(next_u8, self.image_size)
        pos = random_crop(pos_u8, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obs = torch.as_tensor(next_obs, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        pos = torch.as_tensor(pos, device=self.device).float()
        weights = torch.as_tensor(weights, device=self.device)
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos)
        return obses, actions, rewards, next_obs, not_dones, cpc_kwargs, weights, tree_idxs

    # --- update TD-error priorities ---
    def update_priorities(self, tree_idxs, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.abs(td_errors).reshape(-1)
        for t_idx, prio in zip(tree_idxs, td_errors):
            priority = (prio + self.eps) ** self.alpha
            self.tree.update(int(t_idx), priority)
            self.max_priority = max(self.max_priority, priority)


# --- standard buffer (random sampling) ---
class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], 1.0 - float(done))
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        hi = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, hi, size=self.batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        next_obs = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obs, not_dones

    def sample_cpc(self):
        hi = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, hi, size=self.batch_size)
        obses_u8 = self.obses[idxs]
        next_u8 = self.next_obses[idxs]
        pos_u8 = obses_u8.copy()

        obses = random_crop(obses_u8, self.image_size)
        next_obs = random_crop(next_u8, self.image_size)
        pos = random_crop(pos_u8, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obs = torch.as_tensor(next_obs, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos)
        return obses, actions, rewards, next_obs, not_dones, cpc_kwargs


# --- random crop augmentation ---
def random_crop(imgs, output_size):
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    cropped = windows[np.arange(n), w1, h1]
    return cropped
