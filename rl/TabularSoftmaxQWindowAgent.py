import numpy as np
import os


class TabularSoftmaxQWindowAgent(object):
    def __init__(self):
            self.rand_generator = None
            self.step_size = None
            self.prev_rew = None
            self.prev_true_obs = None
            self.still = None
            self.num_actions = None
            self.num_states = None
            self.q_path = ""
            self.actions = [0, 1, 2]
            self.debug = False
            
            
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states
            num_actions (int): The number of actions
            step_size (float): The step-size
            still (int): number representing the no-op action
            seed (int): optional seed for random number generation
            path (string): optional, path to saved model
            debug (bool): optional, flag for debugging prints
        }
        
        """
        # Store the parameters provided in agent_init_info.   
        
        if "debug" in agent_info:
            self.debug = agent_info["debug"]
        else:
            self.debug = False
            
        if "learning" in agent_info:
            self.learning = agent_info["learning"]
        else:
            self.learning = True
        self.num_actions = agent_info["num_actions"]
        self.num_states = agent_info["num_states"]
        self.step_size = agent_info["step_size"]
        self.still = agent_info["still"]
        if "seed" in agent_info:
            self.rand_generator = np.random.RandomState(agent_info["seed"])
        else:
            self.rand_generator = np.random.RandomState()

        # Create an array for action-value estimates and initialize it to zero.
        if "path" in agent_info:
            self.q_path = agent_info["path"]
        if os.path.isfile(self.q_path):
            self.q = np.load(self.q_path)
        else:
            # The array of action-value estimates.
            self.q = np.zeros((self.num_states, len(self.actions), self.num_states, len(self.actions), self.num_states, len(self.actions)), dtype='float32')

                        
        
        
    def agent_policy(self, state_action):
        """ policy of the agent
        Args:
            state_action (Numpy array): active tiles returned by tile coder

        Returns:
            The action selected according to the policy
        """
        
        # compute softmax probability
        c = np.max(state_action)
            
        # Compute the numerator by subtracting c from state-action preferences and exponentiating it
        numerator = np.exp(state_action - c)
        
        # Next compute the denominator by summing the values in the numerator (use np.sum)
        denominator = numerator.sum()
             
        # Create a probability array by dividing each element in numerator array by denominator
        softmax_prob = np.array([numerator[a]/denominator for a in range(len(numerator))])
        
        
        # Sample action from the softmax probability array
        chosen_action = self.rand_generator.choice(len(self.actions), p=softmax_prob)

        return chosen_action
    
        
    def process_observation(self, observation):
        """Helper function called for translate the observation received into a smaller set of
        values used for state representation and reward
        Args:
            observation (int): the state observation from the environment (range [-255, 0]).
        Returns:
            reward (int): step reward (range [-3, 0]).
            adj_obs (int): tiled observation space (range [0, 3]). Used to index the elements of the Q_tensor
        """

        """ If this function is called in agent_start there is not yet a previous observation
        so here we set it as current one"""
        #print("process_observation self.prev_true_obs: " + str(self.prev_true_obs))
        if self.prev_true_obs == None:
            self.prev_true_obs = observation
            
        #print("process_observation prev_obs: {}".format(self.prev_true_obs))
            
        """- If the observation received is >=0 it means the agent is in the correct position, thus 0 reward
        - If the observation received is > than the previous it means the agent moved towards the 
        correct position, we want to assign a negative reward to minimize the # of step taken
        - If the observation received is = to the previous it means the agent made a useless  move, 
        we want to discourage it
        - If the observation received is < than the previous it means the agent moved away from the direction
        leading to the correct position, worste value assigned""" 
        #print(observation, self.prev_true_obs)
        if observation > 999:
            reward = 2  
            adj_obs = 3
        elif observation > self.prev_true_obs:
            reward = 1
            adj_obs = 2
        elif observation == self.prev_true_obs:
            reward = 0
            adj_obs = 1
        else:        
            reward = -2
            adj_obs = 0
     
        self.prev_true_obs = observation
        return reward, adj_obs 

    obs_strings={0: "Bad", 1: "Same", 2: "Good", 3: "Best"}       
            
        
    def agent_start(self, observation):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            obs (int): the state observation from the environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        self.actions = [0, 1, 2]
        reward, obs = self.process_observation(observation)
        if self.debug:
            print("start Reward: {}; Obs: {}, Observation: {}".format(reward, obs_strings[obs], observation))

        #print(self.q.shape)
        """Select first action from fake history of [obs, action_still, obs :] 
        the agent "think" that he didn't move and received the same reward"""
        current_q = self.q[obs, self.still, obs, self.still, obs]
        
        # action selection via Softmax
        action = self.agent_policy(current_q)

        self.prev_state = [obs, self.still, obs, self.still, obs, self.still]
        
        self.prev_rew = reward
        self.prev_true_obs = observation
        
        if self.debug:
            print("Previous reward: {}".format(self.prev_rew))        
        
        return action      
        
  
    def agent_step(self, observation):
        """A step taken by the agent.
        Args:
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        reward, obs = self.process_observation(observation)    
        
        if self.debug:
            print("Step Reward: {}; Obs: {}, Observation: {}".format(reward, obs_strings[obs], observation))
        

        t_obs, t_act, s_obs, s_act, f_obs, f_act = self.prev_state
        prev = self.q[t_obs, t_act, s_obs, s_act, f_obs]      
            
        current_q = self.q[s_obs, s_act, f_obs, f_act, obs]
        
        if self.debug:
            print("prev_state indexes: {}" + str([obs_strings[t_obs], t_act, obs_strings[s_obs], 
                                                  s_act, obs_strings[f_obs], f_act]))        
            print("prev_state: {}".format(prev))
            print("current_state indexes: {}" + str([obs_strings[s_obs], s_act, obs_strings[f_obs], 
                  f_act, obs]))                
            print("current_q: {}".format(current_q))
        
        
        # Perform an update
        if self.learning:
            delta = self.step_size * reward
            if self.debug:
                print("Delta: {}".format(delta))
            # greedy RL update
            prev[f_act] += delta
            
            
        # action selection via Softmax
        action = self.agent_policy(current_q)
        
        # change set of actions
        if action == self.still:
            if self.debug:
                print("still -> change set of actions")
            if self.actions[-1]<self.num_actions-1:
                if self.debug:
                    print("increase actions by 2")
                    print("Pre: " + str(self.actions))
                self.actions[1:] = [x+2 for x in self.actions[1:]]
                if self.debug:
                    print("Post: " + str(self.actions))
            else:
                self.actions[1:] = [1, 2]
                if self.debug:
                    print("reset actions to 1, 2")            
                    print("Post: " + str(self.actions))

            current_q = self.q[1, self.still, 1, self.still, obs]
                 
            s_obs, s_act, f_obs, f_act = 1, self.still, 1, self.still

        if self.debug:
            print("Updated previous Q: {}".format(prev))
        # --------------------------

        self.prev_state =  np.append([s_obs, s_act, f_obs, f_act, obs], action)
        self.prev_rew = reward
        self.prev_true_obs = observation  
        return self.actions[action]
        
        
    def agent_save(self, path):
        """Run when the agent terminates.
        Args:
            observation (int): observation (int): the state observation from 
            the environment's step when the agent enter the terminal state.
        """ 
        
        # save agent model
        if len(path)>0:
            np.save(path, self.q)