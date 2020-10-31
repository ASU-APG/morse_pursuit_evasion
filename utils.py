from smtplib import SMTPException

from pandas import concat, DataFrame
import numpy as np

import logging


class PursuerCompleteException(Exception):
    pass


class EvaderCompleteException(Exception):
    pass


def setup_my_logger(name, level):
    logger = logging.getLogger(name)
    if level == 0:
        level = logging.DEBUG
    elif level == 1:
        level = logging.INFO
    else:
        level = logging.WARN
    logger.setLevel(level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # print(type(data), data.shape)
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)


def scale_ip2op(data_arr, in_low, in_high, out_low, out_high):
    data_scaled = []
    for value in data_arr:
        output_value = ((value - in_low) / (in_high - in_low)) * (
                out_high - out_low) + out_low
        # print('val:{}, op_val: {}'.format(value, output_value))
        data_scaled.append(output_value)
    return np.asarray(data_scaled)


def scale_op2ip(data_arr, input_low, input_high, output_low, output_high):
    data_re = []
    for value in data_arr:
        output_value = ((value - output_low) / (output_high - output_low)) * (
                input_high - input_low) + input_low
        data_re.append(output_value)
    return np.asarray(data_re)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


def send_mail():
    import smtplib
    sender = 'psychic-alien@asu-apg.com'
    receivers = ['varunjammula@gmail.com']

    message = """ Simulation complete."""

    try:
        smtpObj = smtplib.SMTP('localhost')
        smtpObj.sendmail(sender, receivers, message)
        print("Successfully sent email")
    except SMTPException:
        print("Error: unable to send email")
