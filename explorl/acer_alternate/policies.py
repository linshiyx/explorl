import numpy as np
import tensorflow as tf
from baselines.ppo2.policies import nature_cnn
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample


class AcerCnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("acer"):
                h = nature_cnn(X)
                pi_logits = fc(h, 'pi', nact, init_scale=0.01)
                pi = tf.nn.softmax(pi_logits)
                q = fc(h, 'q', nact)

            with tf.variable_scope("explore"):
                # for explore
                nogradient_h = tf.stop_gradient(h)
                e_pi_logits = fc(nogradient_h, 'e_pi', nact, init_scale=0.01)
                e_pi = tf.nn.softmax(e_pi_logits)
                # e_v = fc(nogradient_h, 'e_v', 1)[:, 0]
                e_q = fc(nogradient_h, 'e_q', nact)

        a = sample(pi_logits)  # could change this to use self.pi instead
        evaluate_a = sample(pi_logits)

        self.initial_state = []  # not stateful
        self.X = X
        self.pi = pi  # actual policy params now
        self.q = q

        # for explore
        e_a = sample(e_pi_logits)  # could change this to use self.pi instead
        self.e_pi_logits = e_pi_logits
        self.e_pi = e_pi
        # self.e_v = e_v
        self.e_q = e_q

        def step(ob, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0, e_pi0 = sess.run([a, pi, e_pi], {X: ob})
            return a0, pi0, e_pi0, []  # dummy state

        def evaluate_step(ob, *args, **kwargs):
            evaluate_a0, pi0, e_pi0 = sess.run([evaluate_a, pi, e_pi], {X: ob})
            return evaluate_a0, pi0, e_pi0, []  # dummy state
        self.evaluate_step = evaluate_step

        # for explore
        def e_step(ob, *args, **kwargs):
            a0, e_a0, pi0, e_pi0 = sess.run([a, e_a, pi, e_pi], {X: ob})
            return a0, e_a0, pi0, e_pi0, []  # dummy state
        self.e_step = e_step

        def out(ob, *args, **kwargs):
            pi0, q0 = sess.run([pi, q], {X: ob})
            return pi0, q0

        def act(ob, *args, **kwargs):
            return sess.run(a, {X: ob})

        self.step = step
        self.out = out
        self.act = act

class AcerLstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, nlstm=256):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)

            # lstm
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)

            pi_logits = fc(h5, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h5, 'q', nact)

        a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        self.X = X
        self.M = M
        self.S = S
        self.pi = pi  # actual policy params now
        self.q = q

        def step(ob, state, mask, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0, s = sess.run([a, pi, snew], {X: ob, S: state, M: mask})
            return a0, pi0, s

        self.step = step
