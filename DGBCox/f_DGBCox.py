import tensorflow as tf
import time
import os

from lifelines.utils import concordance_index
from datasets import *

def f_DGBCox(train_or_test, X_in, Y_in, E_in):
    if train_or_test == 1:
        # First step: learn autoencoder by deep cox regression with non-reweighting samples
        print('train AutoEncoder ...')
        learning_rate = 0.01
        num_steps = 2000
        tol = 1e-8
        tf.reset_default_graph()
        # Training the autoencoder with autoencoder loss function and cox loss function
        _ = f_DeepCox(1, X_in, Y_in, E_in, learning_rate, num_steps, tol)

        n, p = X_in.shape

        # Load the autoencoder model
        print('load the autoencoder model ...', flush=True)
        tf.reset_default_graph()
        X_encoder = f_DeepCox(0, X_in, Y_in, E_in, 0, 0, 0)
        X_all_encoder = X_encoder[0]
        for j in range(p):
            X_j = np.copy(X_in)
            X_j[:, j] = 0
            tf.reset_default_graph()
            X_j_encoder = f_DeepCox(0, X_j, Y_in, E_in, 0, 0, 0)
            X_all_encoder = np.vstack((X_all_encoder, X_j_encoder[0]))

        # binary the X used for treatment
        X_m = np.mean(X_in, axis=0)  # column
        X_b_feed = (X_in > X_m).astype(np.float32)

        # estimate the inverse probability of censoring weight vector for each feature
        X_bm = X_in > X_m
        M_cipcw = np.ones(X_in.shape)
        for j in range(p):
            X_bj = X_bm[:, j]
            M_cipcw[:, j] = conditional_ipcw(Y_in, X_bj, E_in)

        # Second step: global sample weights learning by global balancing on embedded confounders
        print('train GlobalBalancing ...')
        tf.reset_default_graph()
        learning_rate = 0.005
        num_steps = 4000
        tol = 1e-8
        GG = f_globalbalancing(1, X_in, X_all_encoder, X_b_feed, M_cipcw, learning_rate, num_steps, tol)

        # Third step: retaining preditive model by deep logistic regression with reweighted samples
        print('train AutoEncoder by weighting ...')
        learning_rate = 0.005
        num_steps = 4000
        tol = 1e-8
        tf.reset_default_graph()
        ci_index = f_DeepCox_weighted(1, X_in, GG[0], Y_in, E_in, learning_rate, num_steps, tol)

    else:
        tf.reset_default_graph()
        ci_index = f_DeepCox_weighted(0, X_in, np.ones([X_in.shape[0], 1]), Y_in, E_in, 0, 0, 0)

    return ci_index

def f_DeepCox(train_or_test, X_input, Y_input, E_input, learning_rate, num_steps, tol):
    n, num_input = X_input.shape

    num_hidden_1 = int(num_input / 8)  # 1st layer num features
    num_hidden_2 = int(num_input / 64)  # 2nd layer num features (the latent dim)

    display_step = 100

    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, 1])
    E = tf.placeholder(tf.float32, [None, 1])  # Observed events (censor status)

    # class dict
    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([num_input, num_hidden_1],0, 1, tf.float32)),
        'encoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2], 0, 1, tf.float32)),
        'decoder_h1': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_1], 0, 1, tf.float32)),
        'decoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_input], 0, 1, tf.float32)),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1], 0, 1, tf.float32)),
        'encoder_b2': tf.Variable(tf.truncated_normal([num_hidden_2], 0, 1, tf.float32)),
        'decoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1], 0, 1, tf.float32)),
        'decoder_b2': tf.Variable(tf.truncated_normal([num_input], 0, 1, tf.float32)),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # encoder of X
    X_encoder = encoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # prediction
    W = tf.Variable(tf.random_normal([num_hidden_2, 1], 0, 1, tf.float32))
    hypothesis = tf.matmul(X_encoder, W)
    #hypothesis = tf.nn.sigmoid(hypothesis)

    #saver = tf.train.Saver([weights]+[biases])
    saver = tf.train.Saver()
    sess = tf.Session()

    if train_or_test == 1:
        ## sort data
        X_input, Y_input, E_input = get_all_data_origin_sort(X_input, Y_input, E_input)

        loss_autoencoder = tf.reduce_mean(tf.square(y_true - y_pred))
        loss_predictive = -tf.reduce_sum(
            (hypothesis - tf.log(tf.cumsum(tf.minimum(tf.exp(hypothesis), tf.float32.max), reverse=True))) * tf.reshape(E, [-1]))
        loss_l2reg = tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(weights['encoder_h1'])) + tf.reduce_sum(
            tf.square(weights['encoder_h2'])) + tf.reduce_sum(tf.square(weights['decoder_h1'])) + tf.reduce_sum(
            tf.square(weights['decoder_h2']))

        loss = 1.0 * loss_predictive + 10.0 / num_hidden_2 * loss_autoencoder + 0.0001 * loss_l2reg
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l_predictive, l_autoencoder, l_l2reg = sess.run(
                [optimizer, loss, loss_predictive, loss_autoencoder, loss_l2reg],
                feed_dict={X: X_input, Y: Y_input, E: E_input})
            if abs(l - l_pre) <= tol:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (
                i, l, l_predictive, l_autoencoder, l_l2reg))
                break
            l_pre = l
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (i, l, l_predictive, l_autoencoder, l_l2reg))

        if not os.path.isdir('models/autoencoder/'):
            os.makedirs('models/autoencoder/')
        saver.save(sess, 'models/autoencoder/autoencoder.ckpt')
    else:
        # saver.restore(sess, 'model/autoencoder.ckpt')
        # saver = tf.train.import_meta_graph('models/autoencoder/autoencoder.ckpt.meta')
        saver.restore(sess, 'models/autoencoder/autoencoder.ckpt')

    return sess.run([X_encoder], feed_dict={X: X_input})


def f_globalbalancing(train_or_test, X_input, X_encoder_input, X_b_input, M_cipcw, learning_rate, num_steps, tol):
    n, p = X_input.shape
    n_e, p_e = X_encoder_input.shape

    display_step = 100

    X = tf.placeholder(tf.float32, [None, p])
    X_encoder = tf.placeholder(tf.float32, [None, p_e])
    X_b = tf.compat.v1.placeholder(tf.float32, [None, p])
    Y_cipcw = tf.compat.v1.placeholder(tf.float32, [None, p])

    #G = tf.Variable(tf.ones([n, 1]),   tf.float32)
    G = tf.Variable(tf.truncated_normal([n, 1], 0, 1,  tf.float32), name='G')

    loss_balancing = tf.constant(0,  tf.float32)
    for j in range(1, p + 1):
        X_j = tf.slice(X_encoder, [j * n, 0], [n, p_e])
        I = tf.slice(X_b, [0, j - 1], [n, 1])
        j_cipcw = tf.slice(Y_cipcw, [0, j - 1], [n, 1])
        balancing_j = tf.divide(tf.matmul(tf.transpose(X_j), G * G * j_cipcw * I), \
                                tf.maximum(tf.reduce_sum(G * G * j_cipcw * I), tf.constant(0.1,  tf.float32))) \
                      - tf.divide(tf.matmul(tf.transpose(X_j), G * G * j_cipcw * (1 - I)),\
                                  tf.maximum(tf.reduce_sum(G * G * j_cipcw * (1 - I)), tf.constant(0.1,  tf.float32)))
        loss_balancing += tf.norm(balancing_j, ord=2)
    loss_regulizer = (tf.reduce_sum(G * G) - n) ** 2 + 10 * (tf.reduce_sum(G * G - 1)) ** 2

    loss = loss_balancing + 0.0001 * loss_regulizer

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if train_or_test == 1:
        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l_balancing, l_regulizer = sess.run([optimizer, loss, loss_balancing, loss_regulizer],
                                                      feed_dict={X: X_input, X_encoder: X_encoder_input,
                                                                 X_b: X_b_input, Y_cipcw: M_cipcw})
            if abs(l - l_pre) <= tol:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_balancing, l_regulizer))
                break
            l_pre = l
            if l_balancing < 0.05:
                print('Good enough ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_balancing, l_regulizer))
                break
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_balancing, l_regulizer))
                '''
                W_final = sess.run(G)
                fw = open('weight_from_tf_'+str(i)+'.txt', 'wb')
                for items in W_final:
                    fw.write(str(items[0])+'\r\n')
                fw.close()
                '''
        if not os.path.isdir('models/globalancing/'):
            os.makedirs('models/globalancing/')
        saver.save(sess, 'models/globalancing/globalancing.ckpt')
    else:
        # saver = tf.train.import_meta_graph('models/globalancing/globalancing.ckpt.meta')
        saver.restore(sess, 'models/globalancing/globalancing.ckpt')

    #return sess.run([G], feed_dict={X: X_input, X_encoder: X_encoder_input, X_b: X_b_input})
    return sess.run([G])


def f_DeepCox_weighted(train_or_test, X_input, G_input, Y_input, E_input, learning_rate, num_steps, tol):
    n, num_input = X_input.shape

    num_hidden_1 = int(num_input / 8)  # 1st layer num features
    num_hidden_2 = int(num_input / 64)  # 2nd layer num features (the latent dim)

    display_step = 100

    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, 1])
    G = tf.placeholder(tf.float32, [None, 1])
    E = tf.placeholder(tf.float32, [None, 1])  # Observed events (censor status)

    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([num_input, num_hidden_1], 0, 1, tf.float32)),
        'encoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2], 0, 1, tf.float32)),
        'decoder_h1': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_1], 0, 1, tf.float32)),
        'decoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_input], 0, 1, tf.float32)),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1], 0, 1, tf.float32)),
        'encoder_b2': tf.Variable(tf.truncated_normal([num_hidden_2], 0, 1, tf.float32)),
        'decoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1], 0, 1, tf.float32)),
        'decoder_b2': tf.Variable(tf.truncated_normal([num_input], 0, 1, tf.float32)),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # encoder of X
    X_encoder = encoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # prediction
    W = tf.Variable(tf.truncated_normal([num_hidden_2, 1], 0, 1, tf.float32))
    hypothesis = tf.matmul(X_encoder, W)
    ##flattening
    hypothesis = tf.reshape(hypothesis, [-1])

    # Nelson-Aalen estimator estimates h0 (no ties)
    """
    h0_b = (1 / tf.cumsum(tf.exp(hypothesis), reverse=True)) * tf.reshape(E, [-1])
    H0_b = tf.cumsum(h0_b)
    s0_b = tf.exp(-H0_b)
    s_b = tf.pow(s0_b, tf.exp(hypothesis))
    """
    saver = tf.train.Saver()
    sess = tf.Session()

    if train_or_test == 1:
        ## sort data
        X_input, Y_input, E_input, G_input = get_all_data_origin_sort2(X_input, Y_input, E_input, G_input)

        # loss_autoencoder = tf.reduce_sum(tf.divide((G*G*tf.pow((y_true - y_pred),2)),tf.reduce_sum(G*G)))
        loss_autoencoder = tf.reduce_mean(tf.pow((G * G) * (y_true - y_pred), 2))
        # loss_predictive = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
        loss_predictive = -tf.reduce_sum(
            tf.divide(G * G * (hypothesis - tf.log(tf.cumsum(tf.exp(hypothesis), reverse=True))) * tf.reshape(E, [-1]),
                      tf.reduce_sum(G * G)))
        loss_l2reg = tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(weights['encoder_h1'])) + tf.reduce_sum(
            tf.square(weights['encoder_h2'])) + tf.reduce_sum(tf.square(weights['decoder_h1'])) + tf.reduce_sum(
            tf.square(weights['decoder_h2']))

        loss = 1.0 * loss_predictive + 15.0 / num_hidden_2 * loss_autoencoder + 0.0005 * loss_l2reg
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        #print(sess.run([tf.reduce_sum(G * G)], feed_dict={G: G_input}))
        l_pre = 0
        for i in range(1, num_steps + 1):
            _, l, l_predictive, l_autoencoder, l_l2reg = sess.run(
                [optimizer, loss, loss_predictive, loss_autoencoder, loss_l2reg],
                feed_dict={X: X_input, G: G_input, Y: Y_input, E: E_input})
            if abs(l - l_pre) <= tol:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (
                i, l, l_predictive, l_autoencoder, l_l2reg))
                break
            l_pre = l
            if i % display_step == 0 or i == 1:
                ydata, edata, risk = \
                    sess.run([Y, E, hypothesis], feed_dict={X: X_input, Y: Y_input, E: E_input})
                partial_hazards = np.exp(risk.ravel())
                ci_index = concordance_index(ydata, -partial_hazards, edata)
                print('Step %i: Minibatch Loss: %f ... %f ... %f ... %f: %f' % (i, l, l_predictive, l_autoencoder, l_l2reg,ci_index))

        if not os.path.isdir('models/autoencoder_weighted/'):
            os.makedirs('models/autoencoder_weighted/')
        saver.save(sess, 'models/autoencoder_weighted/autoencoder_weighted.ckpt')

    else:
        ## sort data
        X_input, Y_input, E_input = get_all_data_origin_sort(X_input, Y_input, E_input)

        # saver.restore(sess, 'model/autoencoder.ckpt')
        # saver = tf.train.import_meta_graph('models/autoencoder_weighted/autoencoder_weighted.ckpt.meta')
        saver.restore(sess, 'models/autoencoder_weighted/autoencoder_weighted.ckpt')

    ydata, edata, risk = \
        sess.run([Y, E, hypothesis], feed_dict={X: X_input, Y: Y_input, E: E_input})
    partial_hazards = np.exp(risk.ravel())
    ci_index = concordance_index(ydata, -partial_hazards, edata)
    # ibs = integrated_brier_score(ydata, edata, survival)
    # ibs_value = r_integrated_brier_score(ydata, edata, partial_hazards)
    return ci_index

def deepCox_weighted_prediction(X_input, Y_input, E_input):
    n, num_input = X_input.shape

    num_hidden_1 = int(num_input / 8)  # 1st layer num features
    num_hidden_2 = int(num_input / 64)  # 2nd layer num features (the latent dim)

    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, 1])
    G = tf.placeholder(tf.float32, [None, 1])
    E = tf.placeholder(tf.float32, [None, 1])  # Observed events (censor status)

    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([num_input, num_hidden_1], 0, 1, tf.float32)),
        'encoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2], 0, 1, tf.float32)),
        'decoder_h1': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_1], 0, 1, tf.float32)),
        'decoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_input], 0, 1, tf.float32)),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1], 0, 1, tf.float32)),
        'encoder_b2': tf.Variable(tf.truncated_normal([num_hidden_2], 0, 1, tf.float32)),
        'decoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1], 0, 1, tf.float32)),
        'decoder_b2': tf.Variable(tf.truncated_normal([num_input], 0, 1, tf.float32)),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # encoder of X
    X_encoder = encoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # prediction
    W = tf.Variable(tf.truncated_normal([num_hidden_2, 1], 0, 1, tf.float32))
    hypothesis = tf.matmul(X_encoder, W)
    ##flattening
    hypothesis = tf.reshape(hypothesis, [-1])

    # Nelson-Aalen estimator estimates h0 (no ties)
    """
    h0_b = (1 / tf.cumsum(tf.exp(hypothesis), reverse=True)) * tf.reshape(E, [-1])
    H0_b = tf.cumsum(h0_b)
    s0_b = tf.exp(-H0_b)
    s_b = tf.pow(s0_b, tf.exp(hypothesis))
    """
    saver = tf.train.Saver()
    sess = tf.Session()

    ## sort data
    X_input, Y_input, E_input = get_all_data_origin_sort(X_input, Y_input, E_input)
    saver.restore(sess, 'models/autoencoder_weighted/autoencoder_weighted.ckpt')
    ydata, edata, risk = \
        sess.run([Y, E, hypothesis], feed_dict={X: X_input, Y: Y_input, E: E_input})
    partial_hazards = np.exp(risk.ravel())
    return ydata, edata, partial_hazards

if __name__ == '__main__':
    #run the training
    tf.compat.v1.set_random_seed(7)
    start = time.time()

    # read data from file
    train_data = pd.read_csv("data/METABRIC.csv")
    train_data = train_data.to_numpy("float32")
    n, p = train_data.shape
    X_in = train_data[:, :-2]
    Y_in = train_data[:, -2]
    E_in = train_data[:, -1]

    tf.reset_default_graph()
    ci_tr = f_DGBCox(1, X_in, Y_in, E_in)
    print(ci_tr)

    end = time.time()
    print("Training DGBCox takes %f" % (end - start))

    ci_list = np.zeros([12, 1])
    dn = np.array(["transbig", "unt", "upp", "mainz", "nki", "GSE6532", "GEO", "TCGA753", "TCGA500",
                "UK", "HEL", "GSE19783"])
    for i in range(12):
        # read data from file
        test_data = pd.read_csv("data/" + dn[i] + ".csv")
        test_data = test_data.to_numpy("float32")
        n, p = test_data.shape
        X_in = test_data[:, :-2]
        Y_in = test_data[:, -2]
        E_in = test_data[:, -1]
        
        tf.reset_default_graph()
        ci_list[i] = f_DGBCox(0, X_in, Y_in, E_in)

    ci_mean = np.mean(ci_list)
    ci_sd = np.std(ci_list)
    ci_stability = ci_mean - ci_sd

    print(ci_list, ci_mean, ci_sd, ci_stability)