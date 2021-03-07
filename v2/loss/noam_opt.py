class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, **kwargs):
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.optimizer = optimizer

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size**(-0.5) *
                              min(step**(-0.5), step * self.warmup**(-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def lr(self):
        return self._rate

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "rate": self._rate
        }

    def load_state_dict(self, state):
        self._rate = state["rate"]
        self._step = state["step"]
        self.optimizer.load_state_dict(state["optimizer"])
