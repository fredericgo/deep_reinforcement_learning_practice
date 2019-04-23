import tensorflow as tf
import multiprocessing as mp
import numpy as np

from collections import namedtuple
import scipy.signal
import gym

FLAGS = tf.app.flags.FLAGS

# General
tf.app.flags.DEFINE_string("job_name", "ps", "[ps/worker]")
tf.app.flags.DEFINE_integer("task_index", 0, "[task index]")
tf.app.flags.DEFINE_string("log_dir", "./train_logs" , "the log dir")

# training
tf.app.flags.DEFINE_integer("num_workers", 2, "number of workers")
tf.app.flags.DEFINE_integer("num_epochs", 50, "number of epochs")
#tf.app.flags.DEFINE_integer("steps_per_epoch", 50, "number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
tf.app.flags.DEFINE_float("clip_ratio", 0.2, "PPO clip ratio")


AgentOutput = namedtuple("AgentOutput", 
                         "actions, log_policy, baselines")

PlaceHolders = namedtuple("PlaceHolders", 
                         "states, returns, log_policy, advs, num_frames")

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Storage(object):
  def __init__(self, action_size, state_size, 
               lam=0.99, gamma=0.99):
    self.obs_buf = []
    self.act_buf = []
    self.adv_buf = []
    self.rew_buf = []
    self.ret_buf = []
    self.val_buf = []
    self.logp_buf = []
    self.gamma, self.lam = gamma, lam

  def add(self, obs, act, rew, val, logp):
    """
    Append one timestep of agent-environment interaction to the buffer.
    """
    self.obs_buf.append(obs)
    self.act_buf.append(act)
    self.rew_buf.append(rew)
    self.val_buf.append(val)
    self.logp_buf.append(logp)

  def finish_path(self, last_val=0.):
    self.obs_buf = np.array(self.obs_buf)
    self.act_buf = np.squeeze(np.array(self.act_buf))
    self.adv_buf = np.array(self.adv_buf)
    self.ret_buf = np.array(self.ret_buf)
    self.logp_buf = np.squeeze(np.array(self.logp_buf))

    self.rew_buf.append(last_val)
    self.val_buf.append([last_val])
    self.val_buf = np.squeeze(self.val_buf)
    rews = np.array(self.rew_buf)
    vals = np.array(self.val_buf)
    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv_buf = discount_cumsum(deltas, self.gamma * self.lam)
    
    # the next line computes rewards-to-go, to be targets for the value function
    self.ret_buf = discount_cumsum(rews, self.gamma)[:-1]
    
  def get(self):
    #print(self.act_buf)
    self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) 
    self.adv_buf = self.adv_buf / (np.std(self.adv_buf) + 1e-8)
    return [self.obs_buf, self.act_buf, self.adv_buf, 
            self.ret_buf, self.logp_buf]

  def reset(self):
    self.obs_buf = []
    self.act_buf = []
    self.adv_buf = []
    self.rew_buf = []
    self.ret_buf = []
    self.val_buf = []
    self.logp_buf = []


def build_nets(inputs, action_size):
  with tf.variable_scope("actor"):
    h = tf.layers.dense(inputs, 64, activation='relu')
    h = tf.layers.dense(h, 64, activation='relu')
    logits = tf.layers.dense(h, action_size, activation=None)
    new_action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    logp_all = tf.nn.log_softmax(logits)
    log_policy = tf.reduce_sum(tf.one_hot(new_action, depth=action_size) * logp_all, axis=1)

  with tf.variable_scope("critic"):
    h = tf.layers.dense(inputs, 64, activation='relu')
    h = tf.layers.dense(h, 64, activation='relu')
    baseline = tf.squeeze(tf.layers.dense(h, 1, activation=None), axis=-1)

  return AgentOutput(actions=new_action, 
                     log_policy=log_policy, 
                     baselines=baseline)

def create_placeholders(state_size):
  states  = tf.placeholder(dtype=tf.float32, 
                           shape=[None, state_size])
  returns = tf.placeholder(dtype=tf.float32, shape=[None, ])
  advs = tf.placeholder(dtype=tf.float32, shape=[None, ])
  log_policy = tf.placeholder(dtype=tf.float32, shape=[None, ])
  num_frames = tf.placeholder(dtype=tf.int32, shape=[])
  return PlaceHolders(states=states,
                      returns=returns,
                      log_policy=log_policy,
                      advs=advs,
                      num_frames=num_frames)

def train_run_parallel(cluster, job, task, num_workers):
  if FLAGS.job_name == 'ps':
    server = tf.train.Server(cluster, 
                         job_name=job,
                         task_index=task)
    server.join()
  else:
    server = tf.train.Server(cluster, 
                         job_name=job,
                         task_index=task)

    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    storage = Storage(action_size, state_size)

    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
      ph = create_placeholders(state_size)
      actor = build_nets(ph.states, action_size)
      ratio = tf.exp(actor.log_policy - ph.log_policy)
      min_adv = tf.where(ph.advs > 0, 
                         (1 + FLAGS.clip_ratio) * ph.advs, 
                         (1 - FLAGS.clip_ratio) * ph.advs)
      pi_loss = - tf.reduce_mean(tf.minimum(ratio * ph.advs, min_adv))
      v_loss = tf.reduce_mean((ph.returns - actor.baselines)**2)
      loss = pi_loss + v_loss
      # Optimizers
      global_step = tf.Variable(0, 
                                dtype=tf.int32,
                                trainable=False,
                                name='global_step')
      num_env_frames = tf.Variable(0, 
                                dtype=tf.int32,
                                trainable=False,
                                name='global_step')
      adam = tf.train.AdamOptimizer(
                    learning_rate=FLAGS.learning_rate
                  )
      optimizer = tf.train.SyncReplicasOptimizer(adam, 
                            replicas_to_aggregate=num_workers,
                            total_num_replicas=num_workers)
      train_op = optimizer.minimize(loss, global_step)
      # Merge updating the network and environment frames into a single tensor.
      with tf.control_dependencies([train_op]):
        num_env_frames = num_env_frames.assign_add(ph.num_frames)

     
      sync_replicas_hook = optimizer.make_session_run_hook(task == 0)

    with tf.train.MonitoredTrainingSession(
            server.target,
            is_chief=(task == 0),
            hooks=[sync_replicas_hook],
            checkpoint_dir=FLAGS.log_dir,
        ) as sess:
      summary_writer = tf.summary.FileWriterCache.get(FLAGS.log_dir)


      curr_state = env.reset()
      ep_ret, ep_len = 0, 0
      for epoch in range(FLAGS.num_epochs):
        while True:
          act, logp, val = sess.run([actor.actions, 
                                actor.log_policy, 
                                actor.baselines], 
                            feed_dict={ph.states: [curr_state]})
          next_state, rew, done, info = env.step(act[0])   
          storage.add(curr_state, act, rew, val, logp)

          curr_state = next_state
          ep_ret += rew
          ep_len += 1
          if done:
            storage.finish_path()
            data = storage.get()
            obs, act, adv, ret, logp_old = storage.get()
            _, num_env_frames_v = sess.run([train_op, num_env_frames],
                                   feed_dict={
                                    ph.states: obs,
                                    actor.actions: act,
                                    ph.log_policy: logp_old,
                                    ph.advs: adv,
                                    ph.returns: ret,
                                    ph.num_frames: ep_len
                                   })
            storage.reset()
            curr_state = env.reset()
            summary = tf.summary.Summary()
            summary.value.add(tag='episode_return',
                            simple_value=ep_ret)
          
            summary_writer.add_summary(summary, num_env_frames_v)
            ep_ret, ep_len = 0, 0



def main(argv):
    cluster = tf.train.ClusterSpec({
      'ps': ['localhost:2222'], 
      'worker': ['localhost:'+str(2223+w) for w in range(FLAGS.num_workers)],
      })
    train_run_parallel(cluster, 
                       FLAGS.job_name, 
                       FLAGS.task_index,
                       FLAGS.num_workers)
"""
    job_task_index_map = [('ps', 0)]
    for w in range(FLAGS.num_workers): 
      job_task_index_map.append(('worker', w))

    procs = []

    for job, task in job_task_index_map:
        proc = mp.Process(target=train_run_parallel, args=(cluster, job, task, FLAGS.num_workers))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()
"""

if __name__ == "__main__":
  tf.app.run()