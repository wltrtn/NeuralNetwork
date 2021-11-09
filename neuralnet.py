import sys
import numpy as np

def parse_data(file_in):
    with open(file_in, 'r') as f_in:
        data = np.loadtxt(f_in, delimiter=',')
    
    y = data[:, 0] # extract y labels
    labels = np.zeros((y.size, 4))
    labels[np.arange(y.size), y.astype('int')] = 1    # convert labels to one-hot vectors
    features = data.copy()
    features[:, 0] = 1   # insert bias weight into x
    
    return labels, features

def init_weights(hidden_units, init_flag, num_x, num_y):
    if init_flag == '1': # randomize weights
        alpha = np.random.uniform(-0.1, 0.1, (hidden_units, num_x-1))
        alpha = np.c_[np.zeros(hidden_units), alpha] # add bias parameters
        beta = np.random.uniform(-0.1, 0.1, (num_y, hidden_units))
        beta = np.c_[np.zeros(num_y), beta] # add bias parameters
    elif init_flag == '2': # zero weights
        alpha = np.zeros((hidden_units, num_x))
        beta = np.zeros((num_y, hidden_units+1))
    return alpha, beta

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x): # calculates softmax on a vector
    exps = np.exp(x)
    return exps / np.sum(exps)

def nn_forward(x, alpha, beta):
    a = np.dot(alpha, x)    # a = alpha * x
    z_star = sigmoid(a)
    z = np.insert(z_star, 0, 1) # add bias z_0=1
    b = np.dot(beta, z)     # b = beta * z
    y_hat = softmax(b)
    return y_hat, z

def nn_backward(x, beta, z, y_hat, y):
    # dJ/dB = dJ/db db/dB = [y_hat-y][z]
    dJdb = y_hat - y
    g_beta = np.outer(dJdb, z)
    dJdz = np.dot(beta[:, 1:].T, dJdb)
    dzda = z[1:] * (1 - z[1:])
    dJda = dJdz * dzda
    g_alpha = np.outer(dJda, x)
    return g_alpha, g_beta

def cross_entropy(y, y_hat):
    return -np.dot(y, np.log(y_hat))

def mean_cross_entropy(y, x, alpha, beta):
    s = 0
    for i in range(len(x)):
        y_hat, _ = nn_forward(x[i], alpha, beta)
        s += cross_entropy(y[i], y_hat)

    return s / len(x)

def sgd(y_train, x_train, y_test, x_test, hidden_units, init_flag, num_epoch, learning_rate):
    alpha, beta = init_weights(hidden_units, init_flag, x_train.shape[1], 4)
    s_a = np.zeros(alpha.shape)
    s_b = np.zeros(beta.shape)
    train_entropy = []
    valid_entropy = []
    for e in range(num_epoch):
        for i in range(len(x_train)):
            y_hat, z = nn_forward(x_train[i], alpha, beta)
            #print(cross_entropy(y_train[i], y_hat))
            g_alpha, g_beta = nn_backward(x_train[i], beta, z, y_hat, y_train[i])
            # Adagrad updates
            s_a += g_alpha * g_alpha
            alpha -= (learning_rate / np.sqrt(s_a + 1e-5)) * g_alpha
            s_b += g_beta * g_beta
            beta -= (learning_rate / np.sqrt(s_b + 1e-5)) * g_beta

        train_entropy.append(mean_cross_entropy(y_train, x_train, alpha, beta))
        valid_entropy.append(mean_cross_entropy(y_test, x_test, alpha, beta))
        
    return alpha, beta, train_entropy, valid_entropy

def predict(y, x, alpha, beta):
    predictions = []
    for xi in x:
        y_hat, _ = nn_forward(xi, alpha, beta)
        predictions.append(np.argmax(y_hat))

    return predictions 

def get_error(y, labels):
    y = np.argmax(y, axis=1) # convert one-hot vectors into indexes
    errors = 0
    for i in range(len(y)):
        if y[i] != labels[i]:
            errors += 1

    return errors/len(y)

def write_label(labels, file_out):
    with open(file_out, 'w') as f_out:
        for l in labels:
            f_out.write(str(l) + '\n')

def write_metrics(train_entropy, valid_entropy, train_error, valid_error, metrics_out):
    with open(metrics_out, 'w') as f_out:
        for i in range(len(train_entropy)):
            f_out.write("epoch={} crossentropy(train): {:.11f}\n".format(i+1, train_entropy[i]))
            f_out.write("epoch={} crossentropy(validation): {:.11f}\n".format(i+1, valid_entropy[i]))
        f_out.write("error(train): {}\n".format(train_error))
        f_out.write("error(validation): {}\n".format(valid_error))

if __name__ == '__main__':
    args = sys.argv
    train_in = args[1]
    valid_in = args[2]
    train_out = args[3]
    valid_out = args[4]
    metrics_out = args[5]
    num_epoch = int(args[6])
    hidden_units = int(args[7])
    init_flag = args[8]
    learning_rate = float(args[9])

    y_train, x_train = parse_data(train_in)
    y_valid, x_valid = parse_data(valid_in)
    
    alpha, beta, train_entropy, valid_entropy = sgd(y_train, x_train, y_valid, x_valid, hidden_units, init_flag, num_epoch, learning_rate)
    
    train_labels = predict(y_train, x_train, alpha, beta)
    valid_labels = predict(y_valid, x_valid, alpha, beta)
    
    write_label(train_labels, train_out)
    write_label(valid_labels, valid_out)

    train_error = get_error(y_train, train_labels)
    valid_error = get_error(y_valid, valid_labels)
    
    write_metrics(train_entropy, valid_entropy, train_error, valid_error, metrics_out)