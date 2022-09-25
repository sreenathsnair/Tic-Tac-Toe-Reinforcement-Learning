from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""

        winning_states = []
        #Code to extract state data row wise
        for i in range(3):
            start = i*3;
            winning_states.append(curr_state[start:start+3])

        #Code to extract state data column wise
        for i in range(3):
            winning_states.append(curr_state[i::3])

        #Code to extract state data diagnally 
        winning_states.append(curr_state[0::4])
        winning_states.append(curr_state[2::2][:3])

        #Check any of the data is a winning pattern or not
        for state_data in winning_states:
            if not np.isnan(state_data[0]) and not np.isnan(state_data[1]) and not np.isnan(state_data[2]):
                state_sum = sum(state_data)
                if (state_sum == 15):
                    return True
        return False

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        # curr_action = [position, value] we need to update the state by placing given value at identified position
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state

    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        #get the new state after agents move
        state_after_move = self.state_transition(curr_state, curr_action)
        reward = 0
        terminated = False
        
        # Check whether it resulted in terminal state if so update reward. 
        # For a win reward is 10 and for tie its 0
        # If its not terminated then from the new state environment has to perform next move
        terminated, status = self.is_terminal(state_after_move)
        if terminated == True:
            if status == 'Win':
                reward = 10
            else:
                reward = 0
        else:
            # Agents move didnt result in a terminal state. Its now env turn to play.
            # Get environments actions for present state. Choose one random action and 
            # evaluate the result of the action
            action_tuple = self.action_space(state_after_move)
            env_action_list = [x for x in action_tuple[1]]
            env_action = random.choice(env_action_list)
            state_after_move = self.state_transition(state_after_move, env_action)
            terminated, status = self.is_terminal(state_after_move)
            if terminated == True:
                if status == 'Win':
                    reward = -10
                else:
                    reward = 0
            else:
                reward = -1
        return (state_after_move, reward, terminated)
            
        

    def reset(self):
        self.state = [np.nan for _ in range(9)]
        return self.state
