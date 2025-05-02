from random import randint, random, sample
from statistics import mean, stdev
from Problem_Domain.environment import Environment
from Robots.lookup_table_robot import LookupTableRobot

LOOKUP_TABLE_SIZE = 243
NUMBER_OF_TRIALS_PER_EVALUATION = 10
NUMBER_OF_ACTIONS_PER_TRIAL = 200
POPULATION_SIZE = 200
NUMBER_OF_GENERATIONS = 1000
TOURNAMENT_SIZE = 4
EXPECTED_NO_OF_MUTATIONS_PER_CHILD = 2
MUTATION_RATE = EXPECTED_NO_OF_MUTATIONS_PER_CHILD/LOOKUP_TABLE_SIZE


def random_action():
    return randint(0, 6)


def select_parent(fitnesses):
    candidates = sample(range(len(fitnesses)), TOURNAMENT_SIZE)
    candidate_fitnesses = [fitnesses[i] for i in candidates]
    selected_parent = candidates[candidate_fitnesses.index(
        max(candidate_fitnesses))]
    return selected_parent


def crossover(parent1, parent2):
    crossover_point = randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point]
    child1.extend(parent2[crossover_point:])
    child2 = parent2[:crossover_point]
    child2.extend(parent1[crossover_point:])
    return child1, child2


def mutate(lookup_table):
    for i in range(len(lookup_table)):
        if random() < MUTATION_RATE:
            lookup_table[i] = random_action()


def random_individual():
    individual = [random_action() for _ in range(LOOKUP_TABLE_SIZE)]
    return individual


def evaluate(lookup_table, seeds):
    robot = LookupTableRobot(action_lookup_table=lookup_table)
    total_score = 0
    for i in range(NUMBER_OF_TRIALS_PER_EVALUATION):
        environment = Environment(random_seed=seeds[i])
        robot.set_environment(environment)
        environment.set_robot(robot)
        for _ in range(NUMBER_OF_ACTIONS_PER_TRIAL):
            action = robot.choose_action()
            environment.perform_action(action)
        total_score += robot.score
    return total_score / NUMBER_OF_TRIALS_PER_EVALUATION


def get_next_generation(current_generation, fitnesses):
    next_generation = []
    while len(next_generation) < POPULATION_SIZE:
        parent1 = current_generation[select_parent(fitnesses)]
        parent2 = current_generation[select_parent(fitnesses)]
        child1, child2 = crossover(parent1, parent2)
        mutate(child1)
        mutate(child2)
        next_generation.append(child1)
        next_generation.append(child2)
    return next_generation


# Generate the initial random population
population = [random_individual() for _ in range(POPULATION_SIZE)]

# Create a set of random seeds. This ensures that all individuals will be evaluated on the
# same set of randomly generated environments.
random_seeds = [randint(-2147483648, 2147483647)
                for _ in range(NUMBER_OF_TRIALS_PER_EVALUATION)]

# Calculate the fitness scores for each member of the population.
fitness_scores = [evaluate(lookup_table, random_seeds)
                  for lookup_table in population]

# Note the best individual seen so far.
best_fitness = max(fitness_scores)
best_lookup_table = population[fitness_scores.index(best_fitness)]

# Successively breed new generations of robot controllers.
for generation in range(1, NUMBER_OF_GENERATIONS + 1):
    population = get_next_generation(population, fitness_scores)

    # Here we apply a kind of elitist strategy by adding the best known individual from the previous population.
    # In this way we avoid losing the best controller even if it is not selected as a candidate parent.
    population.append(best_lookup_table)

    # Create our random seeds for this round of evaluation.
    random_seeds = [randint(-2147483648, 2147483647)
                    for _ in range(NUMBER_OF_TRIALS_PER_EVALUATION)]

    # The fitness score for the same individual could change because the evaluation is being carried out
    # on a different set of randomly generated environments.
    fitness_scores = [evaluate(lookup_table, random_seeds)
                      for lookup_table in population]

    best_fitness = max(fitness_scores)
    best_lookup_table = population[fitness_scores.index(best_fitness)]

    print(f'Generation: {generation}, '
          f'Best: {max(fitness_scores):.2f}, '
          f'Mean: {mean(fitness_scores):.2f}, '
          f'Worst: {min(fitness_scores):.2f}, '
          f'StDev: {stdev(fitness_scores):.2f}')
    print(best_lookup_table)
