import tensorflow as tf
import numpy as np
import sonnet as snt
import gym
import itertools

learning_rate = 0.01
decay = 0.99
gae_decay = 0.95
RENDER_TH = 100
render = False

def build_actor(inputs, action_size):
  h = snt.Linear(32)(inputs)
  h = tf.nn.relu(h)
  h = snt.Linear(32)(h)
  h = tf.nn.relu(h)
  logits = snt.Linear(action_size)(h)
  new_action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
  baseline = tf.squeeze(snt.Linear(1)(h), axis=-1)
  return new_action, logits, baseline

def calculate_gae(rewards, values):
  advs = np.zeros_like(rewards)
  gae = 0
  n_steps = len(rewards)
  for t in reversed(range(n_steps)):
    if t == (n_steps - 1):
      value_next = values[n_steps - 1]
    else:
      value_next = values[t + 1]
    delta = rewards[t] + decay * value_next - values[t]
    gae = delta + gae_decay * decay * gae
    advs[t] = gae
  advs = advs - np.mean(advs)
  advs = advs / (np.std(advs) + 1e-8)
  return advs

def calculate_return(rewards):
  returns = np.zeros_like(rewards)
  acc = 0
  n_steps = len(rewards)
  for t in reversed(range(n_steps)):
    acc = returns[t] + decay * acc
    returns[t] = acc
  return returns

class Storage(object):
  def __init__(self):
    self.observtions = []
    self.actions = []
    self.rewards = []
    self.values = []

  def add(self, obs, act, rew, val):
    self.observtions.append(obs)
    self.actions.append(act)
    self.rewards.append(rew)
    self.values.append(val)

  def empty(self):
    self.observtions = []
    self.actions = []
    self.rewards = []
    self.values = []

env = gym.make('CartPole-v0')
storage = Storage()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

states_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
returns_ph = tf.placeholder(dtype=tf.float32, shape=[None, ])
advantages_ph = tf.placeholder(dtype=tf.float32, shape=[None, ])

actions, logits, baselines = build_actor(states_ph, action_size)

log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions,
                logits=logits,
                name="log_policy")

actor_loss = tf.reduce_mean(log_policy * advantages_ph)
critic_loss = .5 * tf.reduce_mean(tf.square(returns_ph - baselines))
loss = actor_loss + critic_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
global_steps = tf.Variable(0, trainable=False)
train_op = optimizer.minimize(loss, global_step=global_steps)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)

  for i in itertools.count():
    next_state = env.reset()
    storage.empty()

    episode_length = 1
    while True:
      action_sample = sess.run(actions, feed_dict={states_ph: [next_state]})[0]
      baseline_sample = sess.run(baselines, feed_dict={states_ph: [next_state]})[0]
      obs, rew, done, info = env.step(action_sample)   

      if render:
        env.render()  
      storage.add(next_state, action_sample, rew, baseline_sample)
      next_state = obs

      if done or episode_length > 1000:
        ep_reward = sum(storage.rewards)
        gae = calculate_gae(storage.rewards, storage.values)
        returns = calculate_return(storage.rewards)
        _, loss_v = sess.run([train_op, loss], 
                 feed_dict={states_ph: storage.observtions,
                            actions: storage.actions,
                            returns_ph: returns,
                            advantages_ph: gae})
        print("episode {}, reward = {}, loss = {}".format(i, ep_reward, loss_v))

        if ep_reward > RENDER_TH: render = True
        break
      episode_length += 1