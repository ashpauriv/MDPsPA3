# PA3: Solving MDP's
# Monte Carlo
# Value Iteration
# Q-Learning
# Authors: Ashley Rivas, Jesus Oropeza
# Artificial Intelligence : CS 4320

import numpy as np
import random
import matplotlib.pyplot as plt



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
state_values_history = {state: [] for state in states}  # For recording values
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

state_values_history = {state: [] for state in states}

# monte Carlo function to evaluate the policy
def monte_carlo(policy, episodes=50, alpha=0.1):
    for i in range(episodes):
        episode = run_episode(policy)
        total_reward = sum(reward for (_, _, reward) in episode)
        states_in_episode = set([state for (state, _, _) in episode])
        for state in states_in_episode:
            first_visit_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            G = sum(x[2] for x in episode[first_visit_idx:])
            state_values[state] += alpha * (G - state_values[state])
            state_values_history[state].append(state_values[state])  # Correctly append the updated state value
        print(f"Episode {i+1}: Sequence {episode}, Total Reward: {total_reward}")

    # Plotting after all episodes are completed
    #for state in states:
       # plt.plot(state_values_history[state], label=state)
   # plt.xlabel('Episode')
   # plt.ylabel('State Value')
    #plt.title('State Value Progression - Monte Carlo')
    #plt.legend()
   # plt.show()

# Run the Monte Carlo simulation
monte_carlo(policy)

########################### Value Iteration ########################################

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

############################################ Q-learning ###########################################################

# Initialize Q-values
Q = {(state, action): 0 for state in states for action in actions.get(state, [])}
alpha = 0.2  # Initial learning rate
alpha_decay = 0.995  # Decay rate for the learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
threshold = 0.001  # Threshold for the maximum change in Q-value to consider the values converged

# Choose an action using an epsilon-greedy policy
def choose_action_epsilon_greedy(state, epsilon):
    if not actions[state]:  # Check if there are no actions available (terminal state)
        return None  # Return None or a suitable indicator for no action
    if np.random.rand() < epsilon:
        return random.choice(actions[state])
    else:
        q_values = {action: Q[(state, action)] for action in actions[state]}
        max_value = max(q_values.values())
        max_actions = [action for action, value in q_values.items() if value == max_value]
        return random.choice(max_actions)

# Q-learning algorithm
def q_learning(episodes):
    global alpha  # Ensure that we are modifying the global alpha variable
    for episode in range(episodes):
        state = 'Rested'  # assuming the student starts from 'Rested'
        while state != '8pm':  # Assuming '8pm' is your terminal state
            action = choose_action_epsilon_greedy(state, epsilon)
            if action is None:  # If no action is possible, break the loop
                break
            next_state, reward, _ = transitions.get((state, action), [(state, 0, 0)])[0]  # Access the first tuple from the list
            old_value = Q[(state, action)]

            # Safely get the next Q-values, ensuring only valid actions are considered
            next_max = max((Q.get((next_state, a), 0) for a in actions.get(next_state, [])), default=0)
            Q[(state, action)] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state

        alpha *= alpha_decay  # Decay learning rate
        if episode % 10 == 0:  # Print every 10 episodes, for example
            print(f"Episode {episode+1}: Alpha: {alpha:.4f}")

    # Derive policy from Q-values safely by checking if actions are available
    optimal_policy = {
        state: max(actions.get(state, []), key=lambda action: Q.get((state, action), 0), default=None)
        for state in states if state != '8pm'
    }
    
    print(f"Final Q-values: {Q}")
    print(f"Optimal Policy: {optimal_policy}")

# Run the Q-learning algorithm
q_learning(1000)  # Number of episodes


