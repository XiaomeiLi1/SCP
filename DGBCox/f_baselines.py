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

class Coxnet(object):
    """Coxnet model"""

    def __init__(self, input_nodes, config={}):
        """
        Cox regression (Cox) Class Constructor.

        Parameters
        ----------
        input_nodes: int
            The number of input nodes. It's also equal to the number of features.
        config: dict
            Some configurations or hyper-parameters of GBCox.
        """
        # super(Coxnet, self).__init__()

        # nodes
        self.input_nodes = input_nodes

        # hyper-parameters
        _check_config(config)
        self.config = config

        # reset computational graph
        tf.compat.v1.reset_default_graph()
        # graph level random seed
        tf.compat.v1.set_random_seed(config["seed"])

        # some gobal settings
        self.global_step = tf.compat.v1.get_variable('global_step', initializer=tf.constant(0), trainable=False)

        # It's the best way to use `tf.placeholder` instead of `tf.data.Dataset`.
        # Since style of `batch` is not appropriate in survival analysis.
        self.X = tf.compat.v1.placeholder(tf.float32, [None, input_nodes], name='X-Input')
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='Y-Input')

    def _cox_layer(self, x, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable('weights', [x.shape[1], 1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )
            # add weights and bias to collections
            tf.compat.v1.add_to_collection("var_weight", w)

            # Output of encoder hidden layer
            layer_out = tf.matmul(x, w)

            #layer_out = tf.nn.sigmoid(layer_out)

            return layer_out

    def _create_network(self):
        """
        Define the network that only includes Cox layers.
        """
        with tf.name_scope("Cox_layer"):
            self.Y_hat = self._cox_layer(self.X, "Cox_layer")

    def _create_loss(self):
        """
        Define the loss function.

        Notes
        -----
        The loss function defined here is negative log partial likelihood function.
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

            # Get Segment from T
            unique_values, segment_ids = tf.unique(Y_label_T)
            # Get Segment_max
            loss_s2_v = tf.math.segment_max(Y_hat_cumsum, segment_ids)
            # Get Segment_count
            loss_s2_count = tf.math.segment_sum(Y_label_E, segment_ids)
            # Compute S2
            loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
            # Compute S1
            loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
            # Compute Breslow Loss
            loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)

            # Compute Regularization Term Loss
            reg_item = tf.contrib.layers.l1_l2_regularizer(self.config["L1_reg"], self.config["L2_reg"])
            loss_reg = tf.contrib.layers.apply_regularization(reg_item, tf.compat.v1.get_collection("var_weight"))

            # Loss function = prediction loss Function + Regularization Term
            self.loss = loss_breslow + loss_reg

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
        """Build graph of Cox
        """
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self.sess = tf.compat.v1.Session(config=tf.ConfigProto(device_count={'gpu': 0}))

    def close_session(self):
        self.sess.close()
        print("Current session closed.")

    def train(self, data_X, data_y, num_steps, num_skip_steps=1,
              load_model="", save_model="", silent=False):
        """
        Training Cox model.

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
        plot: boolean
            Plot the learning curve.
        silent: boolean
            Print infos to screen.

        Returns
        -------
        dict
            Values of C-index and loss function during training.
        """
        # check dimension
        _check_input_dimension(self.input_nodes, data_X.shape[1])

        # dataset pre-processing
        self.indices, self.train_data_X, self.train_data_y = _prepare_surv_data(data_X, data_y)

        # data to feed
        feed_data = {
            self.X: self.train_data_X.values,
            self.Y: self.train_data_y.values,
        }

        # Session Running
        self.sess.run(tf.compat.v1.global_variables_initializer())
        if load_model != "":
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, load_model)

        # we use this to calculate late average loss in the last SKIP_STEP steps
        total_loss = 0.0
        # Get current global step
        initial_step = self.global_step.eval(session=self.sess)

        for index in range(initial_step, initial_step + num_steps):
            y_hat, loss_value, _ = self.sess.run([self.Y_hat, self.loss, self.optimizer], feed_dict=feed_data)
            total_loss += loss_value
            if (index + 1) % num_skip_steps == 0:
                if (not silent):
                    print('Average loss at step {}: {:.5f}'.format(index + 1, total_loss / num_skip_steps))
                total_loss = 0.0

        # we only save the final trained model
        if save_model != "":
            # defaults to saving all variables
            saver = tf.train.Saver()
            saver.save(self.sess, save_model)

        # update the baseline survival function after all training ops
        self.HR = self.predict(self.train_data_X, output_margin=False)
        # we estimate the baseline survival function S0(t) using training data
        # which returns a DataFrame
        self.BSF = baseline_survival_function(self.train_data_y.values, self.HR)


    def predict(self, X, output_margin=True, load_model = ""):
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
        >>> model.predict(test_X)
        """
        if load_model != "":
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, load_model)
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
        We use negtive hazard ratio as the score. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
        """
        _check_surv_data(data_X, data_y)
        preds = - self.predict(data_X, load_model=load_model)
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

class DAECox(object):
    """Deep Auto Encoder Cox model"""

    def __init__(self, input_nodes, hidden_layers_nodes, config={}):
        """
        Deep Auto Encoder Cox Class Constructor.
        Parameters
        ----------
        input_nodes: int
            The number of input nodes. It's also equal to the number of features.
        hidden_layers_nodes: list
            Number of nodes in hidden layers of neural network.
        config: dict
            Some configurations or hyper-parameters of neural network.
        """
        #super(DAECox, self).__init__()

        # neural nodes
        self.input_nodes = input_nodes
        self.hidden_layers_nodes = hidden_layers_nodes

        # network hyper-parameters
        _check_config(config)
        self.config = config

        # reset computational graph
        tf.compat.v1.reset_default_graph()
        # graph level random seed
        tf.compat.v1.set_random_seed(config["seed"])

        # some gobal settings
        self.global_step = tf.compat.v1.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

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

    def _cox_layer(self, x, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable('weights', [x.shape[1], 1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)
                                          )
            # add weights and bias to collections
            tf.compat.v1.add_to_collection("var_weight", w)

            # Output of encoder hidden layer with dropout
            layer_out = tf.nn.dropout(tf.matmul(x, w), rate=1.0 - self.keep_prob)
            #layer_out = tf.matmul(x, w)

            layer_out = tf.nn.sigmoid(layer_out)

            return layer_out

    def _create_network(self):
        """
        Define the auto encoder network that includes encoder, cox and decoder layers.
        """
        with tf.name_scope("hidden_layers"):
            # Encoder
            cur_x = self.X
            for i, num_nodes in enumerate(self.hidden_layers_nodes):
                cur_x = self._create_encoder_layer(cur_x, num_nodes, "encoder_layer"+str(i+1))
            # output of network
            self.X_encoder = cur_x

            # cox regression
            self.Y_hat = self._cox_layer(self.X_encoder, "cox_layer")

            # Decoder
            # cur_x = self.X_encoder
            decoder_layers_nodes = self.hidden_layers_nodes[::-1]
            del decoder_layers_nodes[0]
            decoder_layers_nodes.append(self.input_nodes)
            for i, num_nodes in enumerate(decoder_layers_nodes):
                cur_x = self._create_decoder_layer(cur_x, num_nodes, "decoder_layer"+str(i+1))
            self.X_hat = cur_x

    def _create_loss(self):
        """
        Define the loss function.
        Notes
        -----
        The loss function defined here is negative log of Breslow Approximation partial
        likelihood function. See more in "Breslow N., 'Covariance analysis of censored
        survival data, ' Biometrics 30.1(1974):89-99.".
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
            loss_autoencoder = tf.reduce_mean(tf.pow(self.X_hat - self.X, 2))

            # Get Segment from T
            unique_values, segment_ids = tf.unique(Y_label_T)
            # Get Segment_max
            loss_s2_v = tf.math.segment_max(Y_hat_cumsum, segment_ids)
            # Get Segment_count
            loss_s2_count = tf.math.segment_sum(Y_label_E, segment_ids)
            # Compute S2
            loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
            # Compute S1
            loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
            # Compute Breslow Loss
            loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)

            # Compute Regularization Term Loss
            reg_item = tf.contrib.layers.l1_l2_regularizer(self.config["L1_reg"], self.config["L2_reg"])
            loss_reg = tf.contrib.layers.apply_regularization(reg_item, tf.compat.v1.get_collection("var_weight"))

            # Loss function = prediction loss Function + autoencoder loss function + Regularization Term
            # todo the weights of loss items
            self.loss = loss_breslow + self.config["Lamda"] / self.hidden_layers_nodes[-1] * loss_autoencoder + loss_reg

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
        """Build graph of DeepCox
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
        Training Deep Auto Encoder Cox model.
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

        # dataset pre-processing
        self.indices, self.train_data_X, self.train_data_y = _prepare_surv_data(data_X, data_y)

        # data to feed
        feed_data = {
            self.keep_prob: self.config['dropout_keep_prob'],
            self.X: self.train_data_X.values,
            self.Y: self.train_data_y.values
        }

        # Session Running
        self.sess.run(tf.compat.v1.global_variables_initializer())
        if load_model != "":
            saver = tf.train.Saver() #saver = tf.train.Saver([w1,w2])
            saver.restore(self.sess, load_model)

        # we use this to calculate late average loss in the last SKIP_STEP steps
        total_loss = 0.0
        # Get current global step
        initial_step = self.global_step.eval(session=self.sess)

        for index in range(initial_step, initial_step + num_steps):
            y_hat, loss_value, _ = self.sess.run([self.Y_hat, self.loss, self.optimizer], feed_dict=feed_data)

            total_loss += loss_value
            if (index + 1) % num_skip_steps == 0:
                if (not silent):
                    print('Average loss at step {}: {:.5f}'.format(index + 1, total_loss / num_skip_steps))
                total_loss = 0.0

        # we only save the final trained model
        if save_model != "":
            # defaults to saving all variables
            saver = tf.train.Saver()
            saver.save(self.sess, save_model)

        # update the baseline survival function after all training ops
        self.HR = self.predict(self.train_data_X, output_margin=False)
        # we estimate the baseline survival function S0(t) using training data
        # which returns a DataFrame
        self.BSF = baseline_survival_function(self.train_data_y.values, self.HR)


    def predict(self, X, output_margin=True, load_model = ""):
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
        >>> model.predict(test_X)
        """
        if load_model != "":
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, load_model)
        # we set dropout to 1.0 when making prediction
        log_hr = self.sess.run([self.Y_hat], feed_dict={self.X: X.values, self.keep_prob: 1.0})
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
        We use negtive hazard ratio as the score. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
        """
        _check_surv_data(data_X, data_y)
        preds = - self.predict(data_X, output_margin=False, load_model=load_model)
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
