import torch
from nanochat.gpt import GPTConfig, GPT
from itertools import combinations
from math import comb

def initialise_model(config: GPTConfig, device):
    # Build the model, move to device, init the weights
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
    model.init_weights() # 3) All tensors get initialized
    return model

class Population:
    def __init__(self, num_strongest, gptconfig: GPTConfig, device) -> None:
        self.num_strongest = num_strongest
        self.pop_cap = num_strongest + comb(self.num_strongest, 2)
        self.population: list[GPT] = []
        self.gptconfig = gptconfig
        self.device = device
        for _ in range(self.num_strongest):
            model = initialise_model(self.gptconfig, self.device)
            self.population.append(model)

    def fill_with_random(self):
        assert len(self.population) == self.num_strongest
        for _ in range(self.pop_cap - self.num_strongest):
            model = initialise_model(self.gptconfig, self.device)
            self.population.append(model)

    def compile(self):
        for model in self.population:
            model = torch.compile(model, dynamic=False)

    def breed(self):
        for i, p in enumerate(combinations(range(self.num_strongest), 2)):
            model = self.population[i+self.num_strongest]
            model.cross_over(self.population[p[0]], self.population[p[1]])
        for model in self.population[:self.num_strongest]:
            model.mutate()

    def train(self):
        for model in self.population:
            model.train()

    def eval(self):
        for model in self.population:
            model.eval()

    def loss(self, x, y) -> list[float]:
        losses = []
        for model in self.population:
            loss = model(x, y)
            loss = loss.detach()
            losses.append(loss)
        return losses

    def sort_to_fittest(self, losses: list[float]):
        modeltoloss = dict(zip(self.population, losses))
        self.population.sort(key=lambda m: modeltoloss[m])

    def setup_optimisers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        self.optimisers = {}
        for model in self.population:
            self.optimisers[model] = model.setup_optimizer(
                        unembedding_lr, embedding_lr, matrix_lr, weight_decay, scalar_lr
                    )

    def zero_grad(self, set_to_none: bool):
        for model in self.population:
            model.zero_grad(set_to_none)
