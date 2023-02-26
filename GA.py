import torch
import torch.optim as optim
import keras
import random
from model import DRL4TSP
import trainer
import argparse
from tasks import tsp

import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


class GAOptimizer(optim.Optimizer):

    def __init__(self, generations, mutation_rate, population_size):
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.population = None
        

    def create_individual(self, statistic_size, dynamic_size, update_fn, args):
        return DRL4TSP(statistic_size, dynamic_size, 
                              args.hidden_size, update_fn, 
                              tsp.update_mask, 
                              args.num_layers, 
                              args.dropout).to(device)
    
    def create_population(self, statistic_size, dynamic_size, update_fn, args):
        return [self.create_individual(statistic_size, dynamic_size, update_fn, args) for indiv in range(self.population_size)]
        
    def fitness(self, reward, fitness):
        fitness.append(reward)
        
    def parents_selection(self, fitness):
        parents = []
        for i in range(self.population_size//4):
            parent_index = fitness.index(max(fitness))
            parents.append(self.population[parent_index])
            fitness.pop(parent_index)
            self.population.pop(parent_index)
        return parents
    
    def crossover(self, parents, statistic_size, dynamic_size, update_fn, args):
        offsprings = []
        for i in range(len(parents)//2):
            parent1_weights = parents[i].get_weights()
            parent2_weights = parents[len(parents)-i-1].get_weights()
            child1 = self.create_individual(statistic_size, dynamic_size, update_fn, args)
            child2 = self.create_individual(statistic_size, dynamic_size, update_fn, args)
            crossover_point = random.randint(1, len(parent1_weights)-1)
            child1_weights = parent1_weights[:crossover_point] + parent2_weights[crossover_point:]
            child1.set_weights(child1_weights)
            child2_weights = parent2_weights[:crossover_point] + parent1_weights[crossover_point:]
            child2.set_weights(child1_weights)
            offsprings.append(child1,child2)
        return offsprings
    
    def mutation(self):
        mutated_population = []
        for individual in self.population:
            mutated_individual = individual
            for weight in range(len(mutated_individual.get_weights())):
                if random.rand() < self.mutation_rate:
                    mutated_individual.get_weights()[weight] += torch.normal(0, 0.1, size=weight.shape)
            mutated_population.append(mutated_individual)
        return mutated_population
    
'''''''''
    def best_model(self, data_loader, reward_fn, statistic_size, dynamic_size, update_fn, args):
        self.population = self.create_population(statistic_size, dynamic_size, update_fn, args)
        for generation in range(self.generations):
            fitness = self.fitness(data_loader, reward_fn)
            parents = self.parents_selection(fitness)
            offsprings = self.crossover(parents)
            self.population = parents + offsprings
            self.population = self.mutation()
        best_model_index = fitness.index(max(fitness))
        return self.population[best_model_index]
'''''''''

''''''''''
parser = argparse.ArgumentParser(description='Combinatorial Optimization')
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--task', default='vrp')
parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
parser.add_argument('--actor_lr', default=5e-4, type=float)
parser.add_argument('--critic_lr', default=5e-4, type=float)
parser.add_argument('--max_grad_norm', default=2., type=float)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--layers', dest='num_layers', default=1, type=int)
parser.add_argument('--train-size',default=1000000, type=int)
parser.add_argument('--valid-size', default=1000, type=int)

from tasks import vrp
from tasks.vrp import VehicleRoutingDataset
from torch.utils.data import DataLoader

args = parser.parse_args()
LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
MAX_DEMAND = 9
max_load = LOAD_DICT[args.num_nodes]

train_data = VehicleRoutingDataset(args.train_size,
                                    args.num_nodes,
                                    max_load,
                                    MAX_DEMAND,
                                    args.seed)

test_data = VehicleRoutingDataset(args.valid_size,
                                  args.num_nodes,
                                  max_load,
                                  MAX_DEMAND,
                                  args.seed + 2)

test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

update_fn = None

GAoptim = GAOptimizer(generations=50, mutation_rate=0.1, population_size=100)
actor = GAoptim.best_model(test_loader, vrp.reward, statistic_size = 2, dynamic_size = 1, update_fn=update_fn, args=args)

print("actor = ", actor.get_weights())

'''''''''''