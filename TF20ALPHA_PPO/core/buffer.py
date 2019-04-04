import numpy as np

def statistics_scalar(x):
    """
    Get mean/std  of scalar x.
    Args: x: An array containing samples of the scalar to produce statistics for.
    """
    mean = np.mean(x)       
    std = np.std(x)
    return mean, std

class Buffer_PPO:

    def __init__(self, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.empty((size,4), dtype=np.float32)
        self.act_buf = np.empty((size,), dtype=np.int32)
        self.adv_buf = np.empty((size,), dtype=np.float32)
        self.rew_buf = np.empty((size,), dtype=np.float32)
        self.ret_buf = np.empty((size,), dtype=np.float32)
        self.val_buf = np.empty((size,), dtype=np.float32)
        self.logp_buf = np.empty((size,), dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0,0,size


    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    
    def finish_path(self, last_val=0):
        # slices the path which to bootstrap
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # GAE Algortihm
        # td = r(t) + gamma * V(St+1) - V(St) --> A(t) = Summe[(gamma*lambda)**l x td(t+1)]
        td_delta = np.empty_like(rews[:-1])
        for idx in range(len(rews)-1):
            td_delta[idx] = rews[idx] + self.gamma * vals[idx+1] - vals[idx]
        self.adv_buf[path_slice] = self.discount_cum_sum_my_way(td_delta, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cum_sum_my_way(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    
    def get(self):
        # buffer has to be full before you can get and reset
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]


    def discount_cum_sum_my_way(self, vec, discount):
        n = len(vec)
        dcs = np.empty_like(vec, dtype= 'float32')
        for i in reversed(range(n)):
            dcs[i] = vec[i] + (discount * dcs[i+1] if i+1 < n else 0)
        return dcs