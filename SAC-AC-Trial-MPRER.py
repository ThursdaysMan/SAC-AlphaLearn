import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import numpy as np
import tensorflow as tf
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pdb

import matplotlib.pyplot as plt

import tensorflow_probability as tfp

# OPTIONS SECTION
# ===================================================================
#Modes
visualMode = False #False
recordMode = False #False

# Rendering and Recording
render = False #False  # True if you want to render the environment
recordStep = 1 #100  # Record at steps. If 0, then do not record


# Choose your environment
envList = {"Cart":"CartPole-v1","Human Walk":"Humanoid-v4","Human Stand":"HumanoidStandup-v4"}
environment = envList["Human Walk"]
nEnvs = 6 #6

# Hyperparameters of the SAC algorithm
hidden_size = 512 #512 #2048 works well
gamma = 0.99
epochs = 400000
learning_rate = 3e-4 
q_learning_rate = 3e-4 #1e-3
batch_size = 256
alpha = 0.2  # Entropy coefficient
bufferSize = 1000000 # * round(nEnvs / 3)
lStart = 100 #4000

actorLogLow = -20 #-1
actorLogHigh = 2 #1

rewardScale = 20 #1 Regular Scale

# Checkpoints
saveInterval = 50 #5000
# Define file paths for the weight files
critic1WeightPath = "critic1_1200_0" #"critic1_3352_3486" #"critic1_499999"
critic2WeightPath = "critic2_1200_0" #"critic2_3352_3486" #"critic2_499999"
valueWeightPath = "value_1200_0" #"value_3352_3486" #"value_499999"
actorWeightPath = "actor_1200_0" #"actor_3352_3486" #"actor_499999"

loadCheckpoints = False #False

#Other Variables
filename = 'WALKEN-SAC5-Buffer100000-Epoch500000.png'
fName = 'WALKEN-SAC-Trial-6-MPRER-Test-Video.mp4'

# Utility Functions
# ===================================================================

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


# MODEL LOADING SECTION
# ===================================================================

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree structure
        self.data = [None] * capacity  # Stores experiences
        self.write = 0  # Position to write new data
        self.size = 0  # Current size of buffer

    def total_priority(self):
        return self.tree[0]  # Root node stores total priority

    def add(self, priority, data):
        idx = self.write + self.capacity - 1  # Leaf index in tree
        self.data[self.write] = data  # Store experience
        self.update(idx, priority)  # Update tree with new priority

        self.write = (self.write + 1) % self.capacity  # Circular buffer
        self.size = min(self.size + 1, self.capacity)  # Track size

    def update(self, idx, priority):
        delta = priority - self.tree[idx]  # Change in priority
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def get_leaf(self, value):
        idx = 0  # Start from root
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

class ReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_increment=1e-6, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Start with max priority

    def add(self, state, next_state, action, reward, done, info):
        experience = (state, next_state, action, reward, done, info)
        self.tree.add(self.max_priority, experience)  # Assign max priority to new samples

    def sample(self):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / self.batch_size  # Divide tree range

        for i in range(self.batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))  # Randomly pick value in segment
            idx, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Convert priorities to probabilities
        priorities = np.array(priorities) / self.tree.total_priority()
        weights = (len(self.tree.data) * priorities) ** (-self.beta)
        weights /= weights.max()  # Normalize

        self.beta = min(1.0, self.beta + self.beta_increment)  # Increment beta

        # Unpack batch
        states, next_states, actions, rewards, dones, infos = zip(*batch)
        return np.array(states), np.array(next_states), np.array(actions), np.array(rewards), np.array(dones), idxs, weights

    def update_priorities(self, idxs, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha  # Compute new priorities
        self.max_priority = max(self.max_priority, max(priorities))  # Update max priority
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)

class ValueNetwork(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = layers.Dense(hidden_dim, activation='relu', input_shape=(None, state_dim))
        self.linear2 = layers.Dense(hidden_dim, activation='relu')
        self.linear3 = layers.Dense(hidden_dim, activation='relu')       
        self.linear4 = layers.Dense(1)
        
    def call(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x

class SoftCritic(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(SoftCritic, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_size, input_shape=(None, num_inputs + num_actions), activation='relu')
        self.linear2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.linear3 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.linear4 = tf.keras.layers.Dense(1)
        
    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x

class Actor(tf.keras.Model):
    def __init__(self, state_dim, num_actions, hidden_size, action_high=-1, action_low=1):
        super(Actor, self).__init__()
        self.action_high = action_high
        self.action_low = action_low
        self.linear1 = layers.Dense(hidden_size/2, activation='relu', input_shape=(None, state_dim))
        self.linear2 = layers.Dense(hidden_size, activation='relu')
        self.linear3 = layers.Dense(hidden_size/2, activation='relu')
        self.mean_linear = layers.Dense(num_actions)
        self.log_std_linear = layers.Dense(num_actions)
        
    def call(self, state):
        x1 = self.linear1(state)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        x = layers.Add()([x1, x3])
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.action_low, self.action_high)
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        z = normal.sample()
        action = tf.tanh(z)
        log_prob = normal.log_prob(z) - tf.math.log(1 - tf.pow(action, 2) + epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):

        try:
            mean, log_std = self.call(state)
        except:
            mean, log_std = self.call(state[0])
      
        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        z = normal.sample()
        action = tf.tanh(z)
        return action.numpy()

class SAC:
    def __init__(self, value_net, value_target, dqn, dqn2, policy_net, replay_buffer, gamma, learning_rate, q_learning_rate, initial_alpha):
        self.dqn = dqn
        self.dqn2 = dqn2
        self.value_net = value_net
        self.value_target = value_target
        self.policy = policy_net
        self.experience_replay = replay_buffer
        self.gamma = gamma
        self.tau = 0.005
        self.log_alpha = tf.Variable(initial_value=tf.math.log(initial_alpha), trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.target_entropy = -tf.reduce_sum(tf.ones(self.policy.log_std_linear.units))
        self.value_net_opt = tf.keras.optimizers.Adam(learning_rate=q_learning_rate)
        self.critic_1_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_2_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.policy_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    #@tf.function
    def train(self):
        # if len(self.experience_replay.buffer) < self.experience_replay.batch_size:
            # return
        
        states, next_states, actions, rewards, dones, idxs, weights = self.experience_replay.sample()

        # Training Critic
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                q_values1 = self.dqn(states, actions)
                q_values2 = self.dqn2(states, actions)
                target_value = self.value_target(next_states)
                target_q_value = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * self.gamma * target_value
                loss1 = tf.reduce_mean(tf.square(q_values1 - target_q_value))
                loss2 = tf.reduce_mean(tf.square(q_values2 - target_q_value))

        grads1 = tape.gradient(loss1, self.dqn.trainable_variables)
        grads2 = tape2.gradient(loss2, self.dqn2.trainable_variables)
        
        grads1, _ = tf.clip_by_global_norm(grads1, clip_norm=1.0)
        grads2, _ = tf.clip_by_global_norm(grads2, clip_norm=1.0)
        
        self.critic_1_opt.apply_gradients(zip(grads1, self.dqn.trainable_variables))
        self.critic_2_opt.apply_gradients(zip(grads2, self.dqn2.trainable_variables))

        # Calculate TD errors for updating priorities
        td_errors = tf.abs(target_q_value - tf.minimum(q_values1, q_values2))
        td_errors = td_errors.numpy().flatten()  # Convert to numpy for updating the buffer

        # Training Value Net
        new_actions, log_probs, z, mean, log_std = self.policy.evaluate(states)
        with tf.GradientTape() as tape:
            prior_values = self.value_net(states)
            q_values1 = self.dqn(states, new_actions)
            q_values2 = self.dqn2(states, new_actions)
            q_value = tf.minimum(q_values1, q_values2)
            target_values = q_value - self.alpha * log_probs
            value_loss = tf.reduce_mean(tf.square(prior_values - target_values))

        grads_value = tape.gradient(value_loss, self.value_net.trainable_variables)
        
        grads_value, _ = tf.clip_by_global_norm(grads_value, clip_norm=1.0) 
        
        self.value_net_opt.apply_gradients(zip(grads_value, self.value_net.trainable_variables))

        # Training Policy
        with tf.GradientTape() as tape:
            new_actions, log_probs, z, mean, log_std = self.policy.evaluate(states)
            q_values1 = self.dqn(states, new_actions)
            q_values2 = self.dqn2(states, new_actions)
            q_value = tf.minimum(q_values1, q_values2)
            policy_loss = tf.reduce_mean(self.alpha * log_probs - q_value)

        grads_policy = tape.gradient(policy_loss, self.policy.trainable_variables)
        grads_policy, _ = tf.clip_by_global_norm(grads_policy, clip_norm=1.0)
        
        self.policy_opt.apply_gradients(zip(grads_policy, self.policy.trainable_variables))

        # Autotune alpha
        with tf.GradientTape() as tape:
            _, log_probs, _, _, _ = self.policy.evaluate(states)
            alpha_loss = tf.reduce_mean(-self.log_alpha * (log_probs + self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        alpha_grads, _ = tf.clip_by_global_norm(alpha_grads, clip_norm=1.0)
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
           
        new_weights = [(1 - self.tau) * old + self.tau * new for old, new in zip(self.value_target.get_weights(), self.value_net.get_weights())]
        self.value_target.set_weights(new_weights)
        
        #Update Experience Replay
        self.experience_replay.update_priorities(idxs, td_errors)
        

def makeEnv(envName):
    return lambda: gym.make(envName)

def record(frames,fName):
    width, height = frames[0].size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(fName, fourcc, 30, (width, height))  # 30 fps

    # Write each image to the video
    for file in frames:
        # Convert Pillow image to NumPy array (compatible with OpenCV)
        frame = np.array(file)
        
        # Ensure the frame has the correct color format (BGR for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame)

    # Release the video writer
    out.release()






if __name__ == "__main__":

    # envs = gym.vector.AsyncVectorEnv([
        # lambda: gym.make(environment),
        # lambda: gym.make(environment),
        # lambda: gym.make(environment),
        # lambda: gym.make(environment)
    # ])
    
    if visualMode or recordMode == True:
        from PIL import Image
        import cv2
        envs = gym.make(environment, render_mode="rgb_array")
    else:
        envs = gym.vector.AsyncVectorEnv([makeEnv(environment) for _ in range(nEnvs)])
        
    # Model Loading/Instantiation
    paramEnv = gym.make(environment)
    env_size = paramEnv.observation_space.shape[0] 
    num_actions = paramEnv.action_space.shape[0]
    del paramEnv

    value_net = ValueNetwork(env_size, hidden_size)
    value_target = ValueNetwork(env_size, hidden_size)

    critic_1 = SoftCritic(env_size, num_actions, hidden_size)
    critic_2 = SoftCritic(env_size, num_actions, hidden_size)

    actor = Actor(env_size, num_actions, hidden_size,actorLogHigh,actorLogLow)

    if loadCheckpoints == True:
        try:
            actor.load_weights(actorWeightPath)
            print("Loaded Actor Successfully")
        except:
            print("ERROR - Actor Model unable to be loaded")
        try:
            critic_1.load_weights(critic1WeightPath)
            print("Loaded Critic1 Successfully")
        except:
            print("ERROR - Critic1 unable to be loaded")
        try:            
            critic_2.load_weights(critic2WeightPath)
            print("Loaded Critic2 Successfully")
        except:
            print("ERROR - Critic2 unable to be loaded")
        try:
            value_net.load_weights(valueWeightPath)
            print("Loaded ValueNet Successfully")
        except:
            print("ERROR - ValueNet unable to be loaded")
        try:
            value_target.load_weights(valueWeightPath)
            print("Loaded ValueTarget Successfully")
        except:
            print("ERROR - ValueTarget unable to be loaded")

    experience_replay = ReplayBuffer(bufferSize, batch_size)

    sac = SAC(value_net, value_target, critic_1, critic_2, actor, experience_replay, gamma, learning_rate, q_learning_rate, alpha)

    returns = []
    avgReturns = []
    hScore = 0
    totalSteps = 0
    
    print(envs)
    
    for episode in range(epochs):
        Return = 0
        done = False
        state = envs.reset()
        state = state[0]
        #print(f"Number of environments in the vector: {len(envs.envs)}")
        #print(f"Shape of the first component (observations): {state[0].shape}")
        steps = 0
        frames = []
        envChs = [False] * nEnvs
        
        fDone = False


        while not fDone:

            action = sac.policy.get_action(state)
            if visualMode or recordMode == True:
                action = action.squeeze()
                frames.append(Image.fromarray(envs.render()))
            next_state, reward, done, info, _ = envs.step(action)

            Return += reward 

            if recordMode == False:
                for itr, (s, ns, a, r, d, i) in enumerate(zip(state, next_state, action, reward, done, info)):
                    experience_replay.add(s, ns, a, r * rewardScale, d, i)
                    if d == False:
                        steps += 1
                    elif envChs[itr] == 0:
                        steps += 1
                        envChs[itr] = 1
                    
            state = next_state
            
            if episode > lStart:  
                if steps % 2 == 0:
                    sac.train()
            if visualMode or recordMode == True:
                if done == True:
                    fDone = True
            
            if sum(envChs) == nEnvs:
                fDone = True      
        
        if (episode % saveInterval == 0) and (episode > lStart):
            x = [i+1 for i in range(episode)]
            exportName = str(episode) + "-" + filename
            plot_learning_curve(x, avgReturns, exportName)   

            sac.dqn.save_weights(f"critic1_{episode}_{int(hScore)}", save_format='tf')
            sac.dqn2.save_weights(f"critic2_{episode}_{int(hScore)}", save_format='tf')
            sac.value_net.save_weights(f"value_{episode}_{int(hScore)}", save_format='tf')
            sac.policy.save_weights(f"actor_{episode}_{int(hScore)}", save_format='tf')
            
            print("hScore Save: ",hScore)
            
            if recordMode == True:
                record(frames,f"Video_{episode}_{nEnvs}.mp4")            

        if visualMode == True:
            print("hScore: ",Return)
            record(frames,fName)
            break
            break
        
        if visualMode == False:

            returns.append(Return)
            avgReturn = np.mean(returns[-10:])
            avgReturns.append(avgReturn)
            avgSteps = int(steps/nEnvs) 
            
            totalSteps += steps
            print(f"Episode: {episode}/{epochs} return: {list(map(int, Return))} av reward: {int(avgReturn)} in {avgSteps} av steps - Total Steps: {totalSteps}")


    # Save weights periodically
    if visualMode == False:
        sac.dqn.save_weights(f"critic1_{episode}", save_format='tf')
        sac.dqn2.save_weights(f"critic2_{episode}", save_format='tf')
        sac.value_net.save_weights(f"value_{episode}", save_format='tf')
        sac.policy.save_weights(f"actor_{episode}", save_format='tf')

        #Save Figure
        x = [i+1 for i in range(epochs)]
        plot_learning_curve(x, avgReturns, filename)
