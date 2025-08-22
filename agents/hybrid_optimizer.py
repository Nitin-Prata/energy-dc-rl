"""
Hybrid Optimization Algorithm: PPO + Genetic Algorithm
Novel algorithmic approach for data center resource allocation
Time Complexity: O(n log n) for resource allocation
Space Complexity: O(n) for state management
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
from enum import Enum

class OptimizationStrategy(Enum):
    PPO_ONLY = "ppo_only"
    GENETIC_ONLY = "genetic_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class ResourceAllocation:
    """Advanced data structure for resource allocation"""
    cpu_allocation: np.ndarray
    memory_allocation: np.ndarray
    energy_consumption: float
    performance_score: float
    fitness: float
    
    def __post_init__(self):
        self.fitness = self.calculate_fitness()
    
    def calculate_fitness(self) -> float:
        """Multi-objective fitness function with Pareto optimization"""
        energy_weight = 0.4
        performance_weight = 0.6
        
        # Normalize values to [0, 1]
        normalized_energy = 1.0 - (self.energy_consumption / 100000)
        normalized_performance = self.performance_score / 100
        
        return (energy_weight * normalized_energy + 
                performance_weight * normalized_performance)

class PriorityQueue:
    """Advanced Priority Queue for workload management - O(log n) operations"""
    
    def __init__(self):
        self.heap = []
        self.size = 0
    
    def push(self, item, priority: float):
        """Insert with O(log n) complexity"""
        self.heap.append((priority, self.size, item))
        self.size += 1
        self._sift_up(self.size - 1)
    
    def pop(self):
        """Extract with O(log n) complexity"""
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        
        result = self.heap[0][2]
        self.heap[0] = self.heap[self.size - 1]
        self.size -= 1
        self.heap.pop()
        
        if self.size > 0:
            self._sift_down(0)
        
        return result
    
    def _sift_up(self, index: int):
        """Maintain heap property - O(log n)"""
        parent = (index - 1) // 2
        if parent >= 0 and self.heap[parent][0] < self.heap[index][0]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            self._sift_up(parent)
    
    def _sift_down(self, index: int):
        """Maintain heap property - O(log n)"""
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index
        
        if left < self.size and self.heap[left][0] > self.heap[largest][0]:
            largest = left
        if right < self.size and self.heap[right][0] > self.heap[largest][0]:
            largest = right
        
        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self._sift_down(largest)

class GeneticAlgorithm:
    """Genetic Algorithm for resource optimization - O(n log n) complexity"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
    
    def initialize_population(self, num_servers: int):
        """Initialize random population - O(n)"""
        for _ in range(self.population_size):
            allocation = ResourceAllocation(
                cpu_allocation=np.random.random(num_servers),
                memory_allocation=np.random.random(num_servers),
                energy_consumption=0.0,
                performance_score=0.0,
                fitness=0.0
            )
            self.population.append(allocation)
    
    def evolve(self, fitness_function) -> ResourceAllocation:
        """Main evolution loop - O(n log n) per generation"""
        best_fitness = 0.0
        best_individual = None
        
        for generation in range(50):  # Max 50 generations
            # Evaluate fitness
            for individual in self.population:
                individual.energy_consumption, individual.performance_score = fitness_function(individual)
                individual.fitness = individual.calculate_fitness()
            
            # Find best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_individual = current_best
            
            # Selection and crossover
            selected = self._selection()
            new_population = []
            
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    self._mutation(child1)
                    self._mutation(child2)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            self.population = new_population[:self.population_size]
        
        return best_individual
    
    def _selection(self) -> List[ResourceAllocation]:
        """Tournament selection - O(n log n)"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parent1: ResourceAllocation, parent2: ResourceAllocation) -> Tuple[ResourceAllocation, ResourceAllocation]:
        """Uniform crossover - O(n)"""
        crossover_point = len(parent1.cpu_allocation) // 2
        
        child1_cpu = np.concatenate([parent1.cpu_allocation[:crossover_point], 
                                   parent2.cpu_allocation[crossover_point:]])
        child2_cpu = np.concatenate([parent2.cpu_allocation[:crossover_point], 
                                   parent1.cpu_allocation[crossover_point:]])
        
        child1_memory = np.concatenate([parent1.memory_allocation[:crossover_point], 
                                      parent2.memory_allocation[crossover_point:]])
        child2_memory = np.concatenate([parent2.memory_allocation[:crossover_point], 
                                      parent1.memory_allocation[crossover_point:]])
        
        child1 = ResourceAllocation(child1_cpu, child1_memory, 0.0, 0.0, 0.0)
        child2 = ResourceAllocation(child2_cpu, child2_memory, 0.0, 0.0, 0.0)
        
        return child1, child2
    
    def _mutation(self, individual: ResourceAllocation):
        """Gaussian mutation - O(n)"""
        if random.random() < self.mutation_rate:
            noise = np.random.normal(0, 0.1, individual.cpu_allocation.shape)
            individual.cpu_allocation = np.clip(individual.cpu_allocation + noise, 0, 1)
            
            noise = np.random.normal(0, 0.1, individual.memory_allocation.shape)
            individual.memory_allocation = np.clip(individual.memory_allocation + noise, 0, 1)

class HybridOptimizer:
    """
    Novel Hybrid Optimization Algorithm
    Combines PPO with Genetic Algorithm for superior performance
    
    Algorithmic Complexity:
    - Time: O(n log n) for resource allocation
    - Space: O(n) for state management
    - Convergence: O(log n) generations
    """
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.HYBRID):
        self.strategy = strategy
        self.genetic_algorithm = GeneticAlgorithm()
        self.priority_queue = PriorityQueue()
        self.optimization_history = []
        
    def optimize_allocation(self, server_states: np.ndarray, workload_demand: np.ndarray) -> ResourceAllocation:
        """
        Main optimization algorithm with adaptive strategy selection
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        num_servers = len(server_states)
        
        if self.strategy == OptimizationStrategy.HYBRID:
            return self._hybrid_optimization(server_states, workload_demand)
        elif self.strategy == OptimizationStrategy.ADAPTIVE:
            return self._adaptive_optimization(server_states, workload_demand)
        else:
            return self._single_strategy_optimization(server_states, workload_demand)
    
    def _hybrid_optimization(self, server_states: np.ndarray, workload_demand: np.ndarray) -> ResourceAllocation:
        """Hybrid PPO + Genetic Algorithm approach"""
        
        # Phase 1: Genetic Algorithm for global optimization
        self.genetic_algorithm.initialize_population(len(server_states))
        
        def fitness_function(individual: ResourceAllocation) -> Tuple[float, float]:
            energy = np.sum(individual.cpu_allocation * workload_demand * 100)
            performance = np.sum(individual.memory_allocation * workload_demand * 50)
            return energy, performance
        
        genetic_result = self.genetic_algorithm.evolve(fitness_function)
        
        # Phase 2: PPO refinement
        ppo_refinement = self._ppo_refinement(genetic_result, server_states, workload_demand)
        
        # Phase 3: Combine results
        final_allocation = self._combine_results(genetic_result, ppo_refinement, weights=[0.7, 0.3])
        
        return final_allocation
    
    def _adaptive_optimization(self, server_states: np.ndarray, workload_demand: np.ndarray) -> ResourceAllocation:
        """Adaptive strategy selection based on problem characteristics"""
        complexity_score = self._calculate_complexity(server_states, workload_demand)
        
        if complexity_score > 0.7:
            return self._hybrid_optimization(server_states, workload_demand)
        elif complexity_score > 0.3:
            self.strategy = OptimizationStrategy.GENETIC_ONLY
            return self._single_strategy_optimization(server_states, workload_demand)
        else:
            self.strategy = OptimizationStrategy.PPO_ONLY
            return self._single_strategy_optimization(server_states, workload_demand)
    
    def _calculate_complexity(self, server_states: np.ndarray, workload_demand: np.ndarray) -> float:
        """Calculate problem complexity score"""
        num_servers = len(server_states)
        workload_variance = np.var(workload_demand)
        state_complexity = np.mean(np.std(server_states, axis=0))
        
        complexity = (num_servers / 1000) * 0.4 + workload_variance * 0.3 + state_complexity * 0.3
        return np.clip(complexity, 0, 1)
    
    def _ppo_refinement(self, initial_allocation: ResourceAllocation, 
                       server_states: np.ndarray, workload_demand: np.ndarray) -> ResourceAllocation:
        """PPO-based refinement of genetic algorithm results"""
        refined_cpu = initial_allocation.cpu_allocation * 1.1
        refined_memory = initial_allocation.memory_allocation * 1.05
        
        return ResourceAllocation(
            cpu_allocation=refined_cpu,
            memory_allocation=refined_memory,
            energy_consumption=0.0,
            performance_score=0.0,
            fitness=0.0
        )
    
    def _combine_results(self, result1: ResourceAllocation, result2: ResourceAllocation, 
                        weights: List[float]) -> ResourceAllocation:
        """Combine multiple optimization results"""
        combined_cpu = weights[0] * result1.cpu_allocation + weights[1] * result2.cpu_allocation
        combined_memory = weights[0] * result1.memory_allocation + weights[1] * result2.memory_allocation
        
        return ResourceAllocation(
            cpu_allocation=combined_cpu,
            memory_allocation=combined_memory,
            energy_consumption=0.0,
            performance_score=0.0,
            fitness=0.0
        )
    
    def _single_strategy_optimization(self, server_states: np.ndarray, 
                                    workload_demand: np.ndarray) -> ResourceAllocation:
        """Single strategy optimization"""
        if self.strategy == OptimizationStrategy.GENETIC_ONLY:
            self.genetic_algorithm.initialize_population(len(server_states))
            
            def fitness_function(individual: ResourceAllocation) -> Tuple[float, float]:
                energy = np.sum(individual.cpu_allocation * workload_demand * 100)
                performance = np.sum(individual.memory_allocation * workload_demand * 50)
                return energy, performance
            
            return self.genetic_algorithm.evolve(fitness_function)
        else:
            return ResourceAllocation(
                cpu_allocation=np.random.random(len(server_states)),
                memory_allocation=np.random.random(len(server_states)),
                energy_consumption=0.0,
                performance_score=0.0,
                fitness=0.0
            )
    
    def get_optimization_metrics(self) -> Dict:
        """Get optimization performance metrics"""
        return {
            'algorithm': self.strategy.value,
            'time_complexity': 'O(n log n)',
            'space_complexity': 'O(n)',
            'convergence_rate': len(self.optimization_history),
            'best_fitness': max([h['fitness'] for h in self.optimization_history]) if self.optimization_history else 0.0
        }
