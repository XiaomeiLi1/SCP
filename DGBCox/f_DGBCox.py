# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd
import numpy as np

from utils import _check_config
from utils import _check_input_dimension
from utils import _check_surv_data
from utils import _prepare_surv_data
from utils import concordance_index
from utils import baseline_survival_function
from visualization import plot_surv_curve

class DGBCox(object):
    """DGBCox model"""

    def __init__(self, hidden_layers_nodes, input_samples, input_nodes, config={}):
        """
        Deep Global Balancing Cox regression (DGBCox) Class Constructor.

        Parameters
        ----------
        hidden_layers_nodes: list
            Number of nodes in hidden layers of deep auto-encoder neural network.
        input_samples: int
            The number of input samples.
        input_nodes: int
            The number of input nodes (i.e. features/genes).
        config: dict
            Some configurations or hyper-parameters of DGBCox.
        """
        # super(DGBCox, self).__init__()

        # nodes
        self.input_nodes = input_nodes

        # samples
        self.input_samples = input_samples

        # neural nodes
        self.hidden_layers_nodes = hidden_layers_nodes

        # hyper-parameters
        _check_config(config)
        self.config = config

        # reset default computational graph before tf.Session()
        tf.compat.v1.reset_default_graph()
        # graph level random seed
        tf.compat.v1.set_random_seed(config["seed"])

        # some gobal settings
        self.global_step = tf.compat.v1.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.keep_prob = self.config['dropout_keep_prob']

        # It's the best way to use `tf.placeholder` instead of `tf.data.Dataset`.
        # Since style of `batch` is not appropriate in survival analysis.
        self.X = tf.compat.v1.placeholder(tf.float32, [None, input_nodes], name='X-Input')
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='Y-Input')

    def _create_encoder_layer(self, x, output_dim, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable('weights', [x.shape[1], output_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )

            b = tf.compat.v1.get_variable('biases', [output_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )

            # add weights and bias to collections
            tf.compat.v1.add_to_collection("var_weight", w)
            tf.compat.v1.add_to_collection("var_bias", b)

            # Output of encoder hidden layer with dropout
            layer_out = tf.nn.dropout(tf.matmul(x, w) + b, rate=1.0 - self.keep_prob)

            if self.config['activation'] == 'relu':
                layer_out = tf.nn.relu(layer_out)
            elif self.config['activation'] == 'sigmoid':
                layer_out = tf.nn.sigmoid(layer_out)
            elif self.config['activation'] == 'tanh':
                layer_out = tf.nn.tanh(layer_out)
            else:
                raise NotImplementedError('activation not recognized')

            return layer_out

    def _create_decoder_layer(self, x, output_dim, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable('weights', [x.shape[1], output_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )

            b = tf.compat.v1.get_variable('biases', [output_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )

            # add weights and bias to collections
            tf.compat.v1.add_to_collection("var_weight", w)
            tf.compat.v1.add_to_collection("var_bias", b)

            # Output of encoder hidden layer with dropout
            layer_out = tf.nn.dropout(tf.matmul(x, w) + b, rate=1.0 - self.keep_prob)

            if self.config['activation'] == 'relu':
                layer_out = tf.nn.relu(layer_out)
            elif self.config['activation'] == 'sigmoid':
                layer_out = tf.nn.sigmoid(layer_out)
            elif self.config['activation'] == 'tanh':
                layer_out = tf.nn.tanh(layer_out)
            else:
                raise NotImplementedError('activation not recognized')

            return layer_out

    def _create_gb_layer(self, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            self.G = tf.compat.v1.get_variable('weights', [self.input_samples, 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                                    )

    def _cox_layer(self, x, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable('weights', [x.shape[1], 1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )
            # add weights to collections
            tf.compat.v1.add_to_collection("var_weight", w)

            # Output of Cox regression layer
            layer_out = tf.matmul(x, w)

            layer_out = tf.nn.sigmoid(layer_out)

            return layer_out

    def _create_network(self):
        """
         Define the network that includes auto encoder, survival global balancing and cox layers.
        """
        with tf.name_scope("hidden_layers"):
            # Encoder
            cur_x = self.X
            for i, num_nodes in enumerate(self.hidden_layers_nodes):
                cur_x = self._create_encoder_layer(cur_x, num_nodes, "encoder_layer" + str(i + 1))
            # output of network
            self.X_encoder = cur_x

            # Decoder
            decoder_layers_nodes = self.hidden_layers_nodes[::-1]
            del decoder_layers_nodes[0]
            decoder_layers_nodes.append(self.input_nodes)
            for i, num_nodes in enumerate(decoder_layers_nodes):
                cur_x = self._create_decoder_layer(cur_x, num_nodes, "decoder_layer" + str(i + 1))
            self.X_hat = cur_x

            # cox regression
            self.Y_hat = self._cox_layer(self.X_encoder, "cox_layer")

            # global balancing
            self._create_gb_layer("gb_layer")

    def _create_loss(self):
        """
        Define the loss function.

        Notes
        -----
        #The loss function defined here is weighted negative log partial likelihood function.
        """
        with tf.name_scope("loss"):
            # Obtain T and E from self.Y
            # NOTE: negtive value means E = 0
            Y_c = tf.squeeze(self.Y)
            Y_hat_c = tf.squeeze(self.Y_hat)
            Y_label_T = tf.abs(Y_c)
            Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
            Obs = tf.reduce_sum(Y_label_E)

            Y_hat_hr = tf.exp(Y_hat_c)
            Y_hat_cumsum = tf.math.log(tf.cumsum(Y_hat_hr))

            # Start Computation of Loss function
            # no tier
            loss_predictive = -tf.divide(tf.reduce_sum(
                tf.divide(
                    self.G * self.G * (Y_hat_c - Y_hat_cumsum) * Y_label_E,
                    tf.reduce_sum(self.G * self.G))), Obs)

            # Compute Regularization Term Loss, weighted sum of L1 L2 regularization
            reg_item = tf.contrib.layers.l1_l2_regularizer(self.config["L1_reg"], self.config["L2_reg"])
            loss_reg = tf.contrib.layers.apply_regularization(reg_item, tf.compat.v1.get_collection("var_weight"))
            # Regularization over layer biases is less common/useful, but assuming proper data preprocessing/mean
            # subtraction, it usually shouldn't hurt much either.

            # Start Computation of Loss function
            loss_autoencoder = tf.reduce_mean(tf.pow(self.X_hat - self.X, 2))

            # binary the X_encoder used for treatment
            X_m = tf.reduce_mean(self.X_encoder, axis=0)

            # estimate the inverse probability of censoring weight vector for each representation
            self.X_bm = tf.cast(self.X_encoder > X_m, tf.float32)

            # global balancing loss
            loss_balancing = tf.constant(0, tf.float32)
            for j in range(self.hidden_layers_nodes[-1]):
                # column number j you want to be zeroed out
                mask = tf.one_hot(j * tf.ones((self.input_samples,), dtype=tf.int32), self.hidden_layers_nodes[-1])
                mask = tf.reduce_sum(mask, axis=0)
                mask = tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32)
                # element-wise multiplication tf.math.multiply: *
                X_j = self.X_encoder * mask  # X_j[:, j] = 0

                # only binary the I used for treatment, 1: treated, 0: control
                I = tf.slice(self.X_bm, [0, j], [self.input_samples, 1])

                d = tf.math.count_nonzero(I)
                lower_tensor = tf.greater(d, tf.cast(self.input_samples * 0.2, dtype=tf.int64))
                upper_tensor = tf.less(d, tf.cast(self.input_samples * 0.8, dtype=tf.int64))
                in_range = tf.logical_and(lower_tensor, upper_tensor)

                def f1():
                    nj = tf.ones([self.input_samples, 1], dtype=tf.float32)  # shape = (self.input_samples,)
                    treated_id = tf.where(tf.not_equal(tf.squeeze(I), tf.constant(0, dtype=tf.float32)))  # shape=(?, 1)
                    treated_cc = tf.gather(tf.squeeze(Y_label_E), treated_id) # shape=(?, 1)
                    # Kaplan Meier estimator
                    # time has be sorted by DESC (high to low)
                    treated_nj = tf.gather(tf.squeeze(nj), treated_id) # shape=(?, 1)
                    # tf.cumsum([a, b, c], reverse=True)  # [a + b + c, b + c, c]
                    ns = tf.cumsum(treated_nj) # shape=(?, 1)
                    dr = tf.divide(treated_cc, ns)
                    sr = treated_nj - dr
                    sr = tf.where(tf.equal(sr, 0), tf.ones_like(sr), sr)
                    treated_cipcw = tf.divide(treated_cc, tf.math.cumprod(sr, reverse=True)) # shape=(?, 1)

                    control_id = tf.where(tf.equal(tf.squeeze(I), tf.constant(0, dtype=tf.float32)))
                    control_cc = tf.gather(tf.squeeze(Y_label_E), control_id)

                    control_nj = tf.gather(tf.squeeze(nj), control_id)
                    control_sr = control_nj -tf.divide(control_cc, tf.cumsum(control_nj))
                    control_sr = tf.where(tf.equal(control_sr, 0), tf.ones_like(control_sr), control_sr)
                    control_cipcw = tf.divide(control_cc,tf.math.cumprod(control_sr, reverse=True))

                    logits_treated = tf.SparseTensor(treated_id, tf.squeeze(treated_cipcw), tf.shape(Y_label_T, out_type=tf.int64))
                    logits_control = tf.SparseTensor(control_id, tf.squeeze(control_cipcw), tf.shape(Y_label_T, out_type=tf.int64))

                    j_cipcw = tf.reshape(tf.add(
                        tf.sparse.to_dense(logits_treated, default_value=0.),
                        tf.sparse.to_dense(logits_control, default_value=0.)
                    ), [-1, 1])  # -1 means "all"

                    return tf.divide(tf.matmul(tf.transpose(X_j), self.G * self.G * j_cipcw * I), \
                                        tf.reduce_sum(self.G * self.G * j_cipcw * I)) \
                              - tf.divide(tf.matmul(tf.transpose(X_j), self.G * self.G * j_cipcw * (1 - I)), \
                                          tf.reduce_sum(self.G * self.G * j_cipcw * (1 - I)))

                def f2(): return tf.zeros([self.hidden_layers_nodes[-1], 1], dtype=tf.float32)
                balancing_j = tf.cond(in_range, f1, f2)

                loss_balancing += tf.norm(balancing_j, ord=2)

            loss_gb_L2_reg = (tf.reduce_sum(self.G * self.G)) ** 2
            loss_gb_reg = (tf.reduce_sum(self.G * self.G - 1)) ** 2

            # Loss function = Cox loss + DAE loss + GB loss + Regularization Term
            self.loss = loss_predictive + self.config["Lamda_3"] / self.hidden_layers_nodes[-1] * loss_autoencoder +\
                        loss_reg + self.config["Lamda_4"] / self.input_nodes * loss_balancing + \
                        self.config["L2_reg"] * loss_gb_L2_reg + self.config["Lamda_5"] * loss_gb_reg

    def _create_optimizer(self):
        """
        Define optimizer
        """
        # SGD Optimizer
        if self.config["optimizer"] == 'sgd':
            lr = tf.compat.v1.train.exponential_decay(
                self.config["learning_rate"],
                self.global_step,
                1,
                self.config["learning_rate_decay"]
            )
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(self.loss,
                                                                                      global_step=self.global_step)
        # Adam Optimizer
        elif self.config["optimizer"] == 'adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss,
                                                                                                     global_step=self.global_step)
        elif self.config["optimizer"] == 'rms':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.config["learning_rate"]).minimize(self.loss,
                                                                                                        global_step=self.global_step)
        else:
            raise NotImplementedError('Optimizer not recognized')

    def build_graph(self):
        """Build graph of DGBCox
        """
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self.sess = tf.compat.v1.Session()

    def close_session(self):
        self.sess.close()
        print("Current session closed.")

    def train(self, data_X, data_y, num_steps, num_skip_steps=1,
              load_model="", save_model="", silent=False):
        """
        Training DGBCox model.

        Parameters
        ----------
        data_X, data_y: DataFrame
            Covariates and labels of survival data. It's suggested that you utilize
            `tfdeepsurv.datasets.survival_df` to obtain the DataFrame object.
        num_steps: int
            The number of training steps.
        num_skip_steps: int
            The number of skipping training steps. Model would be saved after
            each `num_skip_steps`.
        load_model: string
            Path for loading model.
        save_model: string
            Path for saving model.
        silent: boolean
            Print infos to screen.

        Returns
        -------
        dict
            Values of C-index and loss function during training.
        """
        # check dimension
        _check_input_dimension(self.input_nodes, data_X.shape[1])

        # Prepare the survival data. The surv_data will be sorted by abs(`surv_data_y`) DESC.
        self.indices, self.train_data_X, self.train_data_y = _prepare_surv_data(data_X, data_y)

        # self.G is corresponding with self.indices

        # data to feed
        feed_data = {
            self.X: self.train_data_X.values,
            self.Y: self.train_data_y.values,
        }

        # Session Running
        self.sess.run(tf.compat.v1.global_variables_initializer())
        if load_model != "":
            value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer1')
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer2'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer3'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cox_layer'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer1'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer2'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer3'))
            saver = tf.compat.v1.train.Saver(value_list, max_to_keep=100)
            saver.restore(self.sess, load_model)
            var_to_init = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gb_layer')
            var_to_init.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step'))
            var_to_init.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='beta1_power'))
            var_to_init.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='beta2_power'))
            self.sess.run(tf.variables_initializer(var_to_init))

        # we use this to calculate late average loss in the last SKIP_STEP steps
        total_loss = 0.0
        # Get current global step
        initial_step = self.global_step.eval(session=self.sess)

        for index in range(initial_step, initial_step + num_steps):
            y_hat, loss_value, _ = self.sess.run(
                [self.Y_hat, self.loss, self.optimizer], feed_dict=feed_data)
            total_loss += loss_value
            if (index + 1) % num_skip_steps == 0:
                if (not silent):
                    print('Average loss at step {}: {:.5f}'.format(index + 1, total_loss / num_skip_steps))
                total_loss = 0.0

        # we only save the final trained model
        if save_model != "":
            # saving parameters for prediction
            value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer1')
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer2'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer3'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cox_layer'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer1'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer2'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer3'))
            saver = tf.compat.v1.train.Saver(value_list, max_to_keep=100)
            saver.save(self.sess, save_model)

        # update the baseline survival function after all training ops
        self.HR = self.predict(self.train_data_X, output_margin=False)
        # we estimate the baseline survival function S0(t) using training data
        # which returns a DataFrame
        self.BSF = baseline_survival_function(self.train_data_y.values, self.HR)

    def predict(self, X, output_margin=True, load_model=""):
        """
        Predict log hazard ratio using trained model.

        Parameters
        ----------
        X : DataFrame
            Input data with covariate variables, shape of which is (n, input_nodes).
        output_margin: boolean
            If output_margin is set to True, then output of model is log hazard ratio.
            Otherwise the output is hazard ratio, i.e. exp(beta*x).

        Returns
        -------
        np.array
            Predicted log hazard ratio (or hazard ratio) of samples with shape of (n, 1).

        Examples
        --------
        >>> # "array([[0.3], [1.88], [-0.1], ..., [0.98]])"
        >>> # model.predict(test_X)
        """
        if load_model != "":
            value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer1')
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer2'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_layer3'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cox_layer'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer1'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer2'))
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_layer3'))
            saver = tf.compat.v1.train.Saver(value_list, max_to_keep=100)
            saver.restore(self.sess, load_model)
        # we set dropout to 1.0 when making prediction
        log_hr = self.sess.run([self.Y_hat], feed_dict={self.X: X.values})
        log_hr = log_hr[0]
        if output_margin:
            return log_hr
        return np.exp(log_hr)

    def evals(self, data_X, data_y, load_model=""):
        """
        Evaluate labeled dataset using the CI metrics under current trained model.

        Parameters
        ----------
        data_X, data_y: DataFrame
            Covariates and labels of survival data. It's suggested that you utilize
            `tfdeepsurv.datasets.survival_df` to obtain the DataFrame object.

        Returns
        -------
        float
            CI metrics on your dataset.

        Notes
        -----
        We use negtive hazard ratio as the score. See
        https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
        """
        _check_surv_data(data_X, data_y)
        preds = - self.predict(data_X, load_model=load_model)
        # check for NaNs
        preds = np.nan_to_num(preds)
        return concordance_index(data_y.values, preds)

    def predict_survival_function(self, X, plot=False):
        """
        Predict survival function of samples.

        Parameters
        ----------
        X: DataFrame
            Input data with covariate variables, shape of which is (n, input_nodes).
        plot: boolean
            Plot the estimated survival curve of samples.

        Returns
        -------
        DataFrame
            Predicted survival function of samples, shape of which is (n, #Time_Points).
            `Time_Points` indicates the time point that exists in the training data.
        """
        pred_hr = self.predict(X, output_margin=False)
        survf = pd.DataFrame(self.BSF.iloc[:, 0].values ** pred_hr, columns=self.BSF.index.values)

        # plot survival curve
        if plot:
            plot_surv_curve(survf)

        return survf

