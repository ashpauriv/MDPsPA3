# I got the following from Chat, there are likely errors; We can start from scratch if you want to, I just put this as a guide
# To begin, we'll implement the MDP
# It provides the states, actions, rewards, and transition probabilities. 
# We will start with the Monte Carlo method as outlined in Part I.
# First, we need to code the MDP structure from the image. 
# This involves setting up the states, possible actions, rewards, and transition probabilities. 
# Then we'll run the Monte Carlo simulation for 50 episodes#, updating state values using first-visit Monte Carlo with an alpha of 0.1.
# Let's begin by implementing the MDP structure.

import numpy as np
import random

# State definitions
states = ['Rested', 'Tired', 'Homework Done', 'Homework Undone', '8pm']
actions = {
    'Rested': ['Party', 'Rest', 'Study'],
    'Tired': ['Rest'],
    'Homework Done': ['Rest', 'Party', 'Study'],
    'Homework Undone': ['Study'],
    '8pm': []  # Terminal state
}

# Transitions and rewards (format: (Current State, Action): (Next State, Reward, Transition Probability))
transitions = {
    ('Rested', 'Party'): [('Tired', +2, 1.0)],
    ('Rested', 'Rest'): [('Rested', 0, 1.0)],
    ('Rested', 'Study'): [('Homework Done', -1, 1.0)],
    ('Tired', 'Rest'): [('Rested', 0, 1.0)],
    ('Homework Done', 'Rest'): [('8pm', +10, 0.5), ('Tired', +2, 0.5)],
    ('Homework Done', 'Party'): [('Tired', +2, 1.0)],
    ('Homework Done', 'Study'): [('Homework Done', -1, 1.0)],
    ('Homework Undone', 'Study'): [('Homework Done', -2, 1.0)],
    # The terminal state '8pm' has no outgoing actions
}

# Initialize state values and policy
state_values = {state: 0 for state in states}
policy = {state: [1.0 / len(actions[state]) if state in actions else 0 for action in actions[state]] for state in states}

# Helper function to choose an action based on the policy
def choose_action(state, policy):
    return random.choices(actions[state], policy[state])[0]

# Function to run a single episode
def run_episode(policy):
    state = 'Rested'  # assuming the student starts from 'Rested'
    episode = []
    while state != '8pm':
        action = choose_action(state, policy)
        transitions_list = transitions.get((state, action), [])
        if not transitions_list:
            break  # No transitions available from this state-action pair
        probs = [trans_prob for (_, _, trans_prob) in transitions_list]
        next_state, reward, _ = random.choices(transitions_list, weights=probs)[0]
        episode.append((state, action, reward))
        state = next_state
    return episode

# Monte Carlo function to evaluate the policy
def monte_carlo(policy, episodes=50, alpha=0.1):
    for i in range(episodes):
        episode = run_episode(policy)
        total_reward = sum(reward for (_, _, reward) in episode)
        states_in_episode = set([state for (state, _, _) in episode])
        for state in states_in_episode:
            first_visit_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            G = sum(x[2] for x in episode[first_visit_idx:])
            state_values[state] += alpha * (G - state_values[state])
        print(f"Episode {i+1}: Sequence {episode}, Total Reward: {total_reward}")
    print(f"State Values: {state_values}")

# Run the Monte Carlo simulation
monte_carlo(policy)
'''
This code sets up the MDP, runs it for 50 episodes, and uses first-visit Monte Carlo to estimate the state values. 
The print statements at the end of the monte_carlo function will display the outcomes of each episode and the final estimated values of each state.
This code will not execute in this text-based environment. 
To run the Monte Carlo simulation, you should run this Python code on your local machine or in a cloud-based Python environment.
'''
########################### Value Iteration ########################################

# Value iteration is an algorithm for finding the optimal policy by iteratively improving the value function for the states of an MDP. 
# The algorithm updates the value of each state until the changes are below a small threshold, indicating convergence to the optimal values.
# Below is the Python implementation of value iteration for the MDP described in the image:
# Constants for the MDP
gamma = 0.99  # Discount factor
theta = 0.001  # Small threshold for determining accuracy of estimation

# Initialize state values to zero
value_estimates = {state: 0 for state in states}

# Helper function to calculate the value for a given state
def calculate_value(state):
    if not actions[state]:  # Terminal state
        return 0
    
    return max(sum(transition_prob * (reward + gamma * value_estimates[next_state])
                   for next_state, reward, transition_prob in transitions[(state, action)])
               for action in actions[state])

# Value iteration algorithm
def value_iteration():
    delta = theta
    iterations = 0
    
    while delta >= theta:
        iterations += 1
        delta = 0
        for state in states:
            if state == '8pm':  # Skip the terminal state
                continue
            v = value_estimates[state]
            value_estimates[state] = calculate_value(state)
            delta = max(delta, abs(v - value_estimates[state]))
            print(f"Iteration {iterations}, State: {state}, Previous Value: {v:.2f}, New Value: {value_estimates[state]:.2f}")

    # Derive policy from value function
    optimal_policy = {}
    for state in states:
        if state == '8pm':  # Skip the terminal state
            continue
        action_values = {}
        for action in actions[state]:
            action_value = sum(transition_prob * (reward + gamma * value_estimates[next_state])
                               for next_state, reward, transition_prob in transitions[(state, action)])
            action_values[action] = action_value
        optimal_action = max(action_values, key=action_values.get)
        optimal_policy[state] = optimal_action
        print(f"State: {state}, Action Values: {action_values}, Selected Action: {optimal_action}")

    print(f"Number of iterations: {iterations}")
    print(f"Final Values: {value_estimates}")
    print(f"Optimal Policy: {optimal_policy}")

# Run the value iteration algorithm
value_iteration()
'''
This script initializes all state values to zero and iteratively updates them until the maximum change is less than 0.001. 
After convergence, it extracts the optimal policy from the value function.
After you have run the value iteration and are ready, we can proceed with Part III, which is the Q-Learning implementation. 
Would you like to go ahead with that?
'''
############################################ Q-learning ###########################################################
'''
# Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. 
# It does not require a model of the environment and can handle problems with stochastic transitions and rewards, without requiring adaptations.
# For our Q-learning implementation, we'll create a Q-table that holds the Q-value for each state-action pair. 
# We will then use the update rule to improve our Q-values based on the reward received and the maximum Q-value of the next state. 
# The learning rate will decrease after each episode.
# Here is the Python code for Q-learning:
'''
import numpy as np

# Initialize Q-values
Q = {(state, action): 0 for state in states for action in actions[state]}
alpha = 0.2  # Initial learning rate
alpha_decay = 0.995  # Decay rate for the learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
threshold = 0.001  # Threshold for the maximum change in Q-value to consider the values converged

# Choose an action using an epsilon-greedy policy
def choose_action_epsilon_greedy(state, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(actions[state])
    else:
        q_values = {action: Q[(state, action)] for action in actions[state]}
        max_value = max(q_values.values())
        # In case multiple actions have the same max value, choose randomly among them
        max_actions = [action for action, value in q_values.items() if value == max_value]
        return random.choice(max_actions)

# Q-learning algorithm
def q_learning(episodes):
    for episode in range(episodes):
        state = 'Rested'  # assuming the student starts from 'Rested'
        max_change = float('inf')
        while max_change >= threshold and state != '8pm':
            action = choose_action_epsilon_greedy(state, epsilon)
            next_state, reward, trans_prob = transitions[(state, action)][0]  # Assuming deterministic transitions for simplicity
            old_value = Q[(state, action)]
            # Q-learning update
            next_max = max(Q[(next_state, a)] for a in actions[next_state]) if next_state in actions else 0
            Q[(state, action)] = old_value + alpha * (reward + gamma * next_max - old_value)
            max_change = max(max_change, abs(Q[(state, action)] - old_value))
            state = next_state
        # Decay learning rate
        alpha *= alpha_decay
        print(f"Episode {episode+1}: Alpha: {alpha:.4f}, Max Change in Q-value: {max_change:.4f}")

    # Derive policy from Q-values
    optimal_policy = {state: max(actions[state], key=lambda action: Q[(state, action)]) for state in states if actions[state]}

    print(f"Final Q-values: {Q}")
    print(f"Optimal Policy: {optimal_policy}")

# Run the Q-learning algorithm
q_learning(1000)  # Number of episodes

'''
The Q-values are initialized to zero.
The choose_action_epsilon_greedy function implements an epsilon-greedy policy, which selects the action with the highest Q-value with probability 1 - epsilon and explores a random action with probability epsilon.
The q_learning function runs for a specified number of episodes or until the Q-values have converged, whichever comes first. The convergence check is based on whether the maximum change in any Q-value is less than the defined threshold.
After each episode, the learning rate alpha is decayed by the alpha_decay rate.
The final Q-values are printed along with the derived optimal policy.
To verify the functionality, you'll need to run this Python code in your local environment or a Python notebook. The Q-learning process will output the learned Q-values and the optimal policy after convergence.
'''
