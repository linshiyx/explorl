import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds
from baselines.common.runners import AbstractEnvRunner

from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import EpisodeStats
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance

import csv
import os.path as osp

# remove last step
def strip(var, nenvs, nsteps, flat = False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)

def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret

# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef, q_coef, e_vf_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
                 c, trust_region, alpha, delta):
        # config = tf.ConfigProto(# allow_soft_placement=True,
        #                         intra_op_parallelism_threads=num_procs,
        #                         inter_op_parallelism_threads=num_procs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * nsteps


        A = tf.placeholder(tf.int32, [nbatch]) # actions
        D = tf.placeholder(tf.float32, [nbatch]) # dones
        R = tf.placeholder(tf.float32, [nbatch]) # rewards, not returns
        MU = tf.placeholder(tf.float32, [nbatch, nact]) # mu's
        LR = tf.placeholder(tf.float32, [])
        eps = 1e-6

        self.nenvs = nenvs
        self.nsteps = nsteps
        self.A = A
        self.D = D
        self.eps = eps
        self.gamma = gamma
        self.nact = nact
        self.c = c
        self.q_coef = q_coef
        self.delta = delta
        self.trust_region = trust_region

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)

        e_R = tf.placeholder(tf.float32, [nbatch]) # rewards, not returns
        e_MU = tf.placeholder(tf.float32, [nbatch, nact]) # mu's

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/acer')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/explore')

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        ema_apply_op = ema.apply(params + e_params)

        def custom_getter(getter, *args, **kwargs):
            v0 = getter(*args, **kwargs)
            v = ema.average(v0)
            # v = ema.average(getter(*args, **kwargs))
            if v is None:
                return v0
            else:
                print(v.name)
                return v

        with tf.variable_scope("", custom_getter=custom_getter, reuse=True):
            polyak_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)

        grads, loss_policy, loss_q = self.get_gradient(train_model.pi, polyak_model.pi, train_model.q, MU, R, params)
        e_grads, e_loss_policy, e_loss_q = self.get_gradient(train_model.e_pi, polyak_model.e_pi, train_model.e_q, e_MU, e_R, e_params)

        grads = grads + e_grads
        params = params + e_params

        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)

        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        # with tf.control_dependencies([_opt_op]):
        with tf.control_dependencies([_opt_op]):
            _train = tf.group([ema_apply_op])

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        run_ops = [_train, loss_policy, loss_q, e_loss_policy, e_loss_q]
        names_ops = ['loss_policy', 'loss_q', 'e_loss_policy', 'e_loss_q']
        # run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        # names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
        #              'norm_grads']
        # if trust_region:
        #     run_ops = run_ops + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g, avg_norm_k_dot_g,
        #                          avg_norm_adj, e_pg_loss, e_vf_loss]
        #     names_ops = names_ops + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k', 'avg_norm_g',
        #                              'avg_norm_k_dot_g', 'avg_norm_adj', 'e_pg_loss', 'e_vf_loss']


        def train(obs, actions, rewards, dones, mus, states, masks, steps, e_rewards, e_mus):
            cur_lr = lr.value_steps(steps)
            td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr,
                      e_MU: e_mus, e_R: e_rewards}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
                td_map[polyak_model.S] = states
                td_map[polyak_model.M] = masks
            return names_ops, sess.run(run_ops, td_map)[1:]  # strip off _train

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.evaluate_step = step_model.evaluate_step
        self.e_step = step_model.e_step
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

    def get_gradient(self, pi, polyak_pi, q, MU, R, params):
        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i
        v = tf.reduce_sum(pi * q, axis = -1) # shape is [nenvs * (nsteps + 1)]
        # strip off last step
        f, f_pol, q = map(lambda var: strip(var, self.nenvs, self.nsteps), [pi, polyak_pi, q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, self.A)
        q_i = get_by_index(q, self.A)

        # Compute ratios for importance truncation
        rho = f / (MU + self.eps)
        rho_i = get_by_index(rho, self.A)

        # Calculate Q_retrace targets
        qret = q_retrace(R, self.D, q_i, v, rho_i, self.nenvs, self.nsteps, self.gamma)

        # Calculate losses
        # Entropy
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, self.nenvs, self.nsteps, True)
        check_shape([qret, v, rho_i, f_i], [[self.nenvs * self.nsteps]] * 4)
        check_shape([rho, f, q], [[self.nenvs * self.nsteps, self.nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + self.eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(self.c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [self.nenvs * self.nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + self.eps) # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[self.nenvs * self.nsteps, self.nact]]*2)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (self.c / (rho + self.eps))) * f), axis = 1) #IMP: This is sum, as expectation wrt f
        loss_bc= -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[self.nenvs * self.nsteps]]*2)
        ev = q_explained_variance(tf.reshape(q_i, [self.nenvs, self.nsteps]), tf.reshape(qret, [self.nenvs, self.nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i)*0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)
        # loss = loss_policy + q_coef * loss_q - ent_coef * entropy
        loss = loss_policy + self.q_coef * loss_q

        if self.trust_region:
            g = tf.gradients(- loss_policy * self.nsteps * self.nenvs, f) #[nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + self.eps) #[nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            # k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - self.delta) / (tf.reduce_sum(tf.square(k), axis=-1) + self.eps)) #[nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            # avg_norm_k = avg_norm(k)
            # avg_norm_g = avg_norm(g)
            # avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            # avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [self.nenvs * self.nsteps, 1]) * k
            grads_f = -g/(self.nenvs*self.nsteps) # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(loss_q * self.q_coef, params)
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]

            # avg_norm_grads_f = avg_norm(grads_f) * (self.nsteps * self.nenvs)
            # norm_grads_q = tf.global_norm(grads_q)
            # norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, params)

        return grads, loss_policy, loss_q

class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, nstack):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.nstack = nstack
        nh, nw, nc = env.observation_space.shape
        self.nc = nc  # nc = 1 for atari, but just in case
        self.nenv = nenv = env.num_envs
        self.nact = env.action_space.n
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1), nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)

        self.max_progress = [0] * self.nenv
        self.progress = [0] * self.nenv

    def update_obs(self, obs, dones=None):
        if dones is not None:
            self.obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs[:, :, :, :]

    def run(self, explore=False):
        enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        mb_e_mus, mb_e_rewards = [], []
        for _ in range(self.nsteps):
            # actions, mus, states = self.model.step(self.obs, state=self.states, mask=self.dones)
            # if explore:
            # if 6847 > self.max_progress[i] > 6200:
            #     actions, mus, e_mus, states = self.model.e_step(self.obs, state=self.states, mask=self.dones)
            # else:
            #     actions, mus, e_mus, states = self.model.step(self.obs, state=self.states, mask=self.dones)

            actions, e_actions, mus, e_mus, states = self.model.e_step(self.obs, state=self.states, mask=self.dones)
            # print('mus: ', mus)
            # print('e_mus: ', e_mus, '\n')
            print(self.progress[0])
            for i in range(self.nenv):
                if 68.47 > self.progress[i] > 62:
                    actions[i] = e_actions[i]
                    # print('exploring: ', actions[i])
                    # time.sleep(0.02)
            # print(mus)
            # print(e_mus)
            # print(actions)

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_e_mus.append(e_mus)
            mb_dones.append(self.dones)
            obs, rewards, dones, info = self.env.step(actions)
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.update_obs(obs, dones)
            mb_rewards.append(rewards)

            e_rewards = -np.sum(mus*np.log(mus + 1e-5), axis=1)
            e_rewards[dones == True] = 0
            mb_e_rewards.append(e_rewards)
            enc_obs.append(obs)

            for i in range(self.nenv):
                if dones[i]:
                    self.max_progress[i] = 0
                    self.progress[i] = 0
                else:
                    self.progress[i] += rewards[i]
                    if rewards[i] > 0:
                        self.max_progress[i] += rewards[i]

        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        # for explore rewards
        _, last_mus, _, _ = self.model.step(self.obs, state=self.states, mask=self.dones)
        e_rewards = -np.sum(last_mus*np.log(last_mus + 1e-5), axis=1)
        e_rewards[dones == True] = 0
        mb_e_rewards.append(e_rewards)
        mb_e_rewards.pop(0)
        mb_e_rewards = np.asarray(mb_e_rewards, dtype=np.float32).swapaxes(1, 0)

        enc_obs = np.asarray(enc_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_e_mus = np.asarray(mb_e_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks, mb_e_rewards, mb_e_mus

    def evaluate(self, env):
        done = False
        nh, nw, nc = env.observation_space.shape
        e_obs = np.zeros((self.nenv, nh, nw, nc * self.nstack), dtype=np.uint8)
        obs = env.reset()
        obs = np.concatenate([obs]*self.nenv, axis=0)
        e_obs = np.roll(e_obs, shift=-self.nc, axis=3)
        e_obs[:, :, :, -self.nc:] = obs[:, :, :, :]
        reward_episode = 0
        length_episode = 0
        while not done:
            actions, mus, e_mus, states = self.model.evaluate_step(e_obs)
            obs, rew, done, info = env.step(actions)
            obs = np.concatenate([obs]*self.nenv, axis=0)
            e_obs = np.roll(e_obs, shift=-self.nc, axis=3)
            e_obs[:, :, :, -self.nc:] = obs[:, :, :, :]
            reward_episode += rew[0]
            length_episode += 1
        return reward_episode ,length_episode

class Acer():
    def __init__(self, runner, model, log_interval, evaluate_env, evaluate_interval, evaluate_n, logdir, load_info=None):
        self.runner = runner
        self.model = model
        self.buffer = None
        self.log_interval = log_interval
        self.tstart = None
        self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
        self.steps = None

        self.evaluate_env = evaluate_env
        self.evaluate_interval = evaluate_interval
        self.evaluate_n = evaluate_n

        if logdir:
            self.summary_writer = tf.summary.FileWriter(logdir=logdir)
            self.logdir=logdir
            if load_info is not None:
                self.best_mean_reward = load_info['rewards']
            else:
                self.best_mean_reward = 0

            self.evaluation_f = open(logdir+'/evaluation_monitor.csv', "wt")
            self.evaluation_logger = csv.DictWriter(self.evaluation_f, fieldnames=('r', 'l'))
            self.evaluation_logger.writeheader()
        else:
            self.summary_writer = None

    def call(self, on_policy, explore=False):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps
        if on_policy:
            # enc_obs, obs, actions, rewards, mus, dones, masks = runner.run()
            enc_obs, obs, actions, rewards, mus, dones, masks, e_rewards, e_mus = runner.run(explore)
            self.episode_stats.feed(rewards, dones)
            if buffer is not None:
                buffer.put(enc_obs, actions, rewards, mus, dones, masks)
        else:
            # get obs, actions, rewards, mus, dones from buffer.
            obs, actions, rewards, mus, dones, masks = buffer.get()

        # reshape stuff correctly
        obs = obs.reshape(runner.batch_ob_shape)
        actions = actions.reshape([runner.nbatch])
        rewards = rewards.reshape([runner.nbatch])
        mus = mus.reshape([runner.nbatch, runner.nact])
        dones = dones.reshape([runner.nbatch])
        masks = masks.reshape([runner.batch_ob_shape[0]])

        e_rewards = e_rewards.reshape([runner.nbatch])
        e_mus = e_mus.reshape([runner.nbatch, runner.nact])

        names_ops, values_ops = model.train(obs, actions, rewards, dones, mus, model.initial_state, masks, steps, e_rewards, e_mus)

        if on_policy and (int(steps/runner.nbatch) % self.evaluate_interval== 0) and self.summary_writer:
            rewards_mean, length_mean = self.evaluate(self.evaluate_env, self.evaluate_n)
            # logger.record_tabular("mean_episode_length", rewards_mean)
            # logger.record_tabular("mean_episode_reward", length_mean)
            stats = tf.Summary(value=[
                tf.Summary.Value(tag="reward_mean", simple_value=rewards_mean),
                tf.Summary.Value(tag="length_mean", simple_value=length_mean),
            ],)
            self.summary_writer.add_summary(stats, steps)

            self.evaluation_logger.writerow({'r': rewards_mean, 'l': length_mean})
            self.evaluation_f.flush()

            if rewards_mean > self.best_mean_reward:
                self.best_mean_reward = rewards_mean
                self.model.save(self.logdir + '/' + str(int(steps // 1e4)) + '_' + str(int(rewards_mean)))
            elif int(steps/runner.nbatch) % (self.evaluate_interval*10) == 0:
                self.model.save(self.logdir + '/' + str(int(steps // 1e4)) + '_' + str(int(rewards_mean)))


        if on_policy and (int(steps/runner.nbatch) % self.log_interval == 0):
            logger.record_tabular("total_timesteps", steps)
            logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
            # IMP: In EpisodicLife env, during training, we get done=True at each loss of life, not just at the terminal state.
            # Thus, this is mean until end of life, not end of episode.
            # For true episode rewards, see the monitor files in the log folder.
            # logger.record_tabular("mean_episode_length", self.episode_stats.mean_length())
            # logger.record_tabular("mean_episode_reward", self.episode_stats.mean_reward())
            for name, val in zip(names_ops, values_ops):
                logger.record_tabular(name, float(val))
            logger.dump_tabular()

    def evaluate(self, env, n):
        reward_total = 0
        length_total = 0
        for i in range(n):
            reward_episode, length_episode = self.runner.evaluate(env)
            reward_total += reward_episode
            length_total += length_episode

        reward_mean = reward_total / n
        length_mean = length_total / n
        return reward_mean, length_mean


def learn(policy, env, evaluate_env, seed, nsteps=20, nstack=4, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0,
          trust_region=True, alpha=0.99, delta=1, logdir=None, load_info=None):
    print("Running Acer Simple")
    print(locals())
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    # num_procs = len(env.remotes) # HACK
    num_procs = 1 # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, q_coef=q_coef, e_vf_coef=q_coef, gamma=gamma,
                  max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon,
                  total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
                  trust_region=trust_region, alpha=alpha, delta=delta)

    steps_start = 0
    if load_info is not None:
        model.load(load_info['path'])
        steps_start = int(load_info['steps'] * 1e4)

    runner = Runner(env=env, model=model, nsteps=nsteps, nstack=nstack)
    # if replay_ratio > 0:
    #     buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size=buffer_size)
    # else:
    #     buffer = None
    nbatch = nenvs*nsteps
    acer = Acer(runner, model, log_interval, evaluate_env, 1e5//nsteps//nenvs, 5, logdir, load_info)
    acer.tstart = time.time()
    for acer.steps in range(steps_start, total_timesteps, nbatch): #nbatch samples, 1 on_policy call and multiple off-policy calls
        acer.call(on_policy=True, explore=False)
        # if replay_ratio > 0 and buffer.has_atleast(replay_start):
        #     n = np.random.poisson(replay_ratio)
        #     for _ in range(n):
        #         acer.call(on_policy=False)  # no simulation steps in this

    env.close()
