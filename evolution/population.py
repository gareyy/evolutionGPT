import torch
from nanochat.gpt import GPTConfig, GPT

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
        self.pop_cap = self.num_strongest**2
        self.population = []
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
        #self.population[2].cross_over(self.population[0], self.population[1])
        print(hash(str(self.population[0])))
        self.population[0].mutate()
        print(hash(str(self.population[0])))
