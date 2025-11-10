import random
import pandas as pd
import matplotlib.pyplot as plt

''' 
    TABLE OF CONTENT 
    ----------------
    1 - INITIALISATION 
        1.1 - Load data 
        1.2 - Declare algorithm parameters 
        1.3 - Initialize Heuristic matrix n x n (H_ij) 
        1.4 - Initialize Pheromone matrix n x n (Ph_ij) 
        
    2 - CONSTRUCT ANT SOLUTIONS 
        2.1 - Generate path for an ant based on node selection probability 
        2.2 - Update pheromone based on fitness 
        2.3 - Complete ACO algorithm for the ant colony 
        
    3 - PARAMETER EXPERIMENTATION (EXTRA SESSION)
'''

# 1 - INITIALISATION
# 1.1 - Load data
def load_data(file_path):
    bags = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        van_capacity = int(lines[0].split(':')[1].strip())
        for line in lines[1:]:
            if 'weight' in line:
                weight = float(line.split(':')[1].strip())
            elif 'value' in line:
                value = float(line.split(':')[1].strip())
                bags.append((weight, value))
    return van_capacity, bags

# 1.2 - Declare algorithm parameters (temporary parameters)
def init_parameters():
    parameters = {
        "population_size": 10,
        "pheromone_deposit_rate": 0.1,
        "evaporation_rates": 0.5,
        "stop_condition": 10000,
        "initial_pheromone": 1,
        "alpha": 1.0,
        "beta": 1.0
    }
    return parameters

# 1.3 - Initialize heuristic matrix n x n (H_ij)
def init_heuristic_matrix(bags):
    heuristics = [[0 if i == j else bags[j][1] / bags[j][0] for j in range(len(bags))] for i in range(len(bags))]
    return heuristics

# 1.4 - Initialize Pheromone matrix n x n (P_ij)
def init_pheromones_matrix(num_bags, initial_pheromone=1):
    pheromones = [[initial_pheromone for j in range(num_bags)] for i in range(num_bags)]
    return pheromones

# 2 - CONSTRUCT ANT SOLUTIONS
# 2.1 - Generate path for an ant based on the probabilities P_ij & H_ij
def generate_path(bags, pheromones, van_capacity, alpha, beta):
    heuristics = init_heuristic_matrix(bags)
    used_capacity = 0
    allowed_bags = list(range(len(bags)))
    selected_bags = []

    # Start an ant at a random bag (first layer)
    layer_index = 0
    first_choice = random.choice(allowed_bags)
    selected_bags.append(first_choice)
    used_capacity += bags[first_choice][0]

    # Loop to generate path for an ant (starting from the second layer until the end)
    while used_capacity < van_capacity:
        # Remove seleted bag from allowed bags by changing its heuristic information to 0 in the loop
        for j in selected_bags:
            for i in range(len(bags)):
                heuristics[i][j] = 0
        # Remove bags larger than the available capacity by changing their heuristic information to 0 in the loop
        for i in range(len(bags)):
            if i not in selected_bags and used_capacity + bags[i][0] > van_capacity:
                for j in range(len(bags)):
                    heuristics[j][i] = 0
        # Termination condition for the ant path finding when no bag satisfies the weight requirement for selection
        if all(used_capacity + bags[i][0] > van_capacity for i in allowed_bags if i not in selected_bags):
            break
            
        # Termination condition if all heuristic values are 0
        if all(heuristics[j][i] == 0 for j in range(len(bags)) for i in allowed_bags):
            break

        # Start the loop for the ant to select the next bag
        layer_index += 1
        # Calculate the probabilities of the bags in current layer
        numerators = [(pheromones[layer_index][i] ** alpha) * (heuristics[layer_index][i] ** beta) for i in allowed_bags]
        denominator = sum(numerators)
        probabilities = [n / denominator if denominator > 0 else 0 for n in numerators]

        # Calculate the cumulative probabilities of each bag
        cumulative_probabilities = []
        cumulative_sum = 0
        for p in probabilities:
            cumulative_sum += p
            cumulative_probabilities.append(cumulative_sum)

        # Select the next bag based on the cumulative probability and update the results into storage variables
        max_prob = cumulative_probabilities[-1]
        rand = random.uniform(0, max_prob)
        
        # Return the bag whose cumulative probability is the closest higher or equal to rand
        next_choice = allowed_bags[next(i for i, cp in enumerate(cumulative_probabilities) if cp >= rand)]
        selected_bags.append(next_choice)
        used_capacity += bags[next_choice][0]

    # Calculate fitness as the total value of the selected bags
    fitness = sum(bags[i][1] for i in selected_bags)
    return [bags[i] for i in selected_bags], fitness

# 2.2 - Update pheromone based on fitness
def update_pheromones(pheromones, paths, bags, best_fitness, pheromone_deposit_rate, evaporation_rates):
    # Calculate pheromone evaporation amount
    num_bags = len(pheromones)
    for i in range(num_bags):
        for j in range(num_bags):
            pheromones[i][j] *= (1 - evaporation_rates)

    # Calculate pheromone deposit amount on the paths based on fitness
    for path, fitness in paths:
        if fitness > 0 and best_fitness > 0:
            for idx, bag in enumerate(path[:-1]):
                i = bags.index(bag)
                j = bags.index(path[idx + 1])
                pheromones[i][j] += pheromone_deposit_rate * (1 / fitness)
    return pheromones

# 2.3 - Complete ACO algorithm for the ant colony

'''
(1) This code creates a program that integrates all the steps mentioned above to develop a pathfinding algorithm 
for the entire ant colony, with a limit of 10,000 fitness evaluations.

Additionally, to minimize bias, each set of parameters in the program will run the implementation (1) above 
10 times and calculate the average value.

The program will also record the values of each path found by the ants for further analysis later on.
'''

def aco_model(file_path, parameters):
    van_capacity, bags = load_data(file_path)
    pheromones = init_pheromones_matrix(len(bags), parameters["initial_pheromone"])

    fitness_evaluations = 0
    best_fitness = 0
    best_solution = []
    
    # Variable to store the evaluation results of the ant colony after completing one evaluation (*)
    temp = {
        "population_sizes": [], 
        "pheromone_deposit_rates": [], 
        "evaporation_rates": [], 
        "fitness_evaluations": [], 
        "best_fitness": []
    }

    # Variable to store the average results of (*) after 10 iterations
    loop_history = {
        "avg_population_size": [],
        "avg_pheromone_deposit_rate": [],
        "avg_evaporation_rate": [],
        "avg_fitness_evaluations": [],
        "avg_best_fitness": []
    }

    # Run implementations 10 times to get an average
    for _ in range(10):
        while fitness_evaluations < parameters["stop_condition"]:
            paths, fitness_list = [], []  # Store the fitness of each ant in this loop
            for _ in range(parameters["population_size"]):
                selected_bags, fitness = generate_path(bags, pheromones, van_capacity, parameters["alpha"], parameters["beta"])
                paths.append((selected_bags, fitness))
                fitness_list.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = selected_bags

                fitness_evaluations += 1

            # Update pheromones using the previously created update_pheromones function
            pheromones = update_pheromones(pheromones, paths, bags, best_fitness, parameters["pheromone_deposit_rate"], parameters["evaporation_rates"])

            # Record data into temp
            temp["population_sizes"].append(parameters["population_size"])
            temp["pheromone_deposit_rates"].append(parameters["pheromone_deposit_rate"])
            temp["evaporation_rates"].append(parameters["evaporation_rates"])
            temp["fitness_evaluations"].append(fitness_evaluations)
            temp["best_fitness"].append(best_fitness)

    # Convert temp lists to pandas DataFrame for convenient calculations
    df_temp = pd.DataFrame(temp)

    # Calculate averages grouped by fitness_evaluations
    df_grouped = df_temp.groupby("fitness_evaluations").mean().reset_index()

    loop_history = {
        "avg_population_size": df_grouped["population_sizes"].to_numpy(),
        "avg_pheromone_deposit_rate": df_grouped["pheromone_deposit_rates"].to_numpy(),
        "avg_evaporation_rate": df_grouped["evaporation_rates"].to_numpy(),
        "avg_fitness_evaluations": df_grouped["fitness_evaluations"].to_numpy(),
        "avg_best_fitness": df_grouped["best_fitness"].to_numpy()
    }

    return fitness_evaluations, best_fitness, best_solution, loop_history

'''
Run the code below if you just want to see the end result instead of seeing parameter experiments and plotting 
at Session 3:

parameters = {
    "population_size": 10,
    "pheromone_deposit_rate": 0.1,
    "evaporation_rates": 0.5,
    "stop_condition": 10000,
    "initial_pheromone": 1,
    "alpha": 1.0,
    "beta": 1.0
}

# Run the ACO model
file_path = "BankProblem.txt"
fitness_evaluations, best_fitness, best_solution, loop_history = aco_model(file_path, parameters)

print(f"Completed experiment | Best Fitness: {best_fitness} | Best Solution: {best_solution}")

'''

# ============================================ 
# 3 - PARAMETER EXPERIMENTATION (EXTRA SESSION)
# Building a experiment function 
def experiment_aco_parameters(file_path, population_sizes, pheromone_deposit_rates, evaporation_rates):
    combined_loop_history = {
        "avg_population_size": [],
        "avg_pheromone_deposit_rate": [],
        "avg_evaporation_rate": [],
        "avg_fitness_evaluations": [],
        "avg_best_fitness": []
    }
    
    global_best_fitness = 0
    global_best_solution = []

    for population_size in population_sizes:
        for pheromone_deposit_rate in pheromone_deposit_rates:
            for evaporation_rate in evaporation_rates:
                parameters = init_parameters()
                parameters["population_size"] = population_size
                parameters["pheromone_deposit_rate"] = pheromone_deposit_rate
                parameters["evaporation_rates"] = evaporation_rate

                # Run ACO algorithm
                fitness_evaluations, best_fitness, best_solution, loop_history = aco_model(file_path, parameters)

                # Combine results
                combined_loop_history["avg_population_size"].extend(loop_history["avg_population_size"])
                combined_loop_history["avg_pheromone_deposit_rate"].extend(loop_history["avg_pheromone_deposit_rate"])
                combined_loop_history["avg_evaporation_rate"].extend(loop_history["avg_evaporation_rate"])
                combined_loop_history["avg_fitness_evaluations"].extend(loop_history["avg_fitness_evaluations"])
                combined_loop_history["avg_best_fitness"].extend(loop_history["avg_best_fitness"])

                # Check for overall best fitness and solution
                if best_fitness > global_best_fitness:
                    global_best_fitness = best_fitness
                    global_best_solution = best_solution

                # Print results
                print(f"Completed experiment with parameters(p={population_size}, m={pheromone_deposit_rate}, e={evaporation_rate}) | best_fitness={best_fitness:.2f})")

    return global_best_fitness, global_best_solution, combined_loop_history


file_path = "BankProblem.txt"
population_sizes = [10, 30, 50, 70, 90]
pheromone_deposit_rates = [0.1, 0.3, 0.5, 0.7, 0.9] 
evaporation_rates = [0.5, 0.6, 0.7, 0.8, 0.9]

global_best_fitness, global_best_solution, combined_loop_history = experiment_aco_parameters(file_path, population_sizes, pheromone_deposit_rates, evaporation_rates)

