import os
from time import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import keras as ks
import keras.backend as K
import tensorflow as tf
import tensorflow.keras.backend as kb
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from bayes_opt import BayesianOptimization
from functools import partial
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def RMSLE(pred,actual):
    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))

def load_data():
    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")
    train['Province_State'].fillna('', inplace=True)
    test['Province_State'].fillna('', inplace=True)
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    train["Country_State"] = train['Country_Region'] + train['Province_State']
    test["Country_State"] = test['Country_Region'] + test['Province_State']
    train = train.sort_values(['Date', 'Country_Region', 'Province_State'])
    test = test.sort_values(['Date', 'Country_Region', 'Province_State'])
   # print(train.head())
 #   print(test.tail())
    return train, test



expcache = {}
def exp_smooth(v, w=50):
    vn = v[:]
    for i in range(len(v)):
        if v[i] > 0:
            vn = v[i:]
            break
    if len(vn) < 2:
        vn =  pad_sequences([v],maxlen=3)[0]
    if vn[-1] == 0:
        return np.zeros((30,))
    key = tuple(vn.tolist())
    if key in expcache:
        return expcache[key]
    model = ExponentialSmoothing(vn, trend='additive').fit()
    ec = model.forecast(30)
    expcache[key] = ec
  #  print(ec)
    return ec


def create_dataset2(df,sh, w, st=0):
    country_state = df["Country_State"].unique()
    dates = df["Date"].unique()

    Xc = np.zeros((dates.shape[0]-st, country_state.shape[0]))
    Xf = np.zeros((dates.shape[0]-st, country_state.shape[0]))
  #  print(Xc.shape)
  #  print(country_state)
  #  print(dates)
    for i, d in enumerate(dates[st:]):
        dayrec = df[df["Date"]==d].sort_values("Country_State")
        Xc[i,:] = dayrec['ConfirmedCases'].values
        Xf[i, :] = dayrec['Fatalities'].values
    xdata = []
    ydatac = []
    ydataf = []
    if w >= Xc.shape[0]:
        w = Xc.shape[0] - sh - 10
    for i in range(w, Xc.shape[0]-1):
        if i+sh >= Xc.shape[0]:
            break
        for j in range(Xc.shape[1]):
            rc = Xc[i - w:i, j].tolist()
            rf = Xf[i - w:i, j].tolist()
            ec = exp_smooth(rc)
        #    ef = exp_smooth(rf)
            xdata.append(rc +[ec[sh]])
        #    xdata.append(rc + rf)
            ydatac.append(Xc[i + sh,j])
            ydataf.append(Xf[i + sh, j])
    X = np.array(xdata, dtype=np.float32)

    yc = np.array(ydatac, dtype=np.float32)
    yf = np.array(ydataf, dtype=np.float32)
    print(X.shape, yc.shape, w)
    return X, yc, yf


def create_dataset(df):
    country_state = df["Country_State"].unique()
    dates = df["Date"].unique()
    Xc = np.zeros((dates.shape[0], country_state.shape[0]))
    Xf = np.zeros((dates.shape[0], country_state.shape[0]))
  #  print(Xc.shape)
  #  print(country_state)
  #  print(dates)
    for i, d in enumerate(dates):
        dayrec = df[df["Date"]==d].sort_values("Country_State")
        Xc[i,:] = dayrec['ConfirmedCases'].values
        Xf[i, :] = dayrec['Fatalities'].values
    #    print(dayrec.shape)

    Sc = np.log1p(np.sum(Xc, axis=1))
    Sf = np.log1p(np.sum(Xf, axis=1))
  #  print(Sc.shape, Xc.shape)
    Xc = np.log1p(Xc)
    Xf = np.log1p(Xf)
    xdata = []
    ydata = []
    for i in range(21, Sc.shape[0]):
      #  r =  Xc[i - 2, :].tolist() + Xf[i - 2, :].tolist() + \
        r = Xc[i - 1, :].tolist() + Xf[i - 1, :].tolist()  #+ \
           # Sc[i - 21:i].tolist() + Sf[i - 21:i].tolist()
        for j in range(0,30):
            if i+j < Xc.shape[0]:
                ydata.append(Xc[i+j,:].tolist() + Xf[i+j,:].tolist())
                pl = [0] * 30
                pl[j] = 1
                xdata.append(r + pl)

    X = np.array(xdata)
    y = np.array(ydata)
    print(X.shape, y.shape)
    return X, y

def root_mean_squared_log_error_0(y_true, y_pred):
    return kb.sqrt(kb.mean(kb.square(kb.log(y_pred + 1) - kb.log(y_true + 1)))+0.00000001)

def root_mean_squared_log_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred ) - (y_true ))))


def create_model(dinput, dout, dm = 10, dp=0.5):
    inp_1 = tf.keras.Input(shape=(dinput,), dtype='float32',sparse=False)
    encoded = tf.keras.layers.Dense(dm, activation='relu')(inp_1)
    encoded = tf.keras.layers.Dropout(dp)(encoded)
    decoded = tf.keras.layers.Dense(dout, activation='relu')(encoded)
    model = tf.keras.Model(inp_1, decoded)
    model.compile(loss=root_mean_squared_log_error_0, optimizer='adam', metrics=[root_mean_squared_log_error_0])
   # model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def create_model_encode_decode(dinput, dout, sparse=False):
    dm = 250
    inp_1 = ks.Input(shape=(dinput,), dtype='float32',sparse=sparse)
    encoded = ks.layers.Dense(dm, activation='relu')(inp_1)
    encoded = ks.layers.Dropout(0.2)(encoded)
  #  encoded = BatchNormalization()(encoded)
  #  encoded = ks.layers.Dense(50, activation='relu')(encoded)
    decoded = ks.layers.Dense(dout, activation='relu')(encoded)
    model = ks.Model(inp_1, decoded)
    model.compile(loss=root_mean_squared_log_error, optimizer='adam', metrics=['mae'])
    return model


def train_nn():
    np.random.seed(21)
    train, test = load_data()
    X, y = create_dataset(train)
    model = create_model_encode_decode(X.shape[1], y.shape[1])
    bs = 10
    model.fit(X, y, batch_size=bs, epochs=20, validation_split=0.1, verbose=1)

def opt_params(target):
    np.random.seed(21)
    trnn = partial(train_nn2, target)
  #  res = trnn(15, 10, 10, 0.5)
    optimizer = BayesianOptimization(trnn,{
        'w': (50,80),
        'bs': (10,40),
        'dm': (10, 160),
        'dp': (0.1,0.4),
        'st': (0,0)
    })
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        optimizer.maximize(init_points=3, n_iter=10)  #, acq='ei', xi=0.0
    print(optimizer.max)
    #print(res)


def train_nn2(target,w, bs, dm , dp,st):
    w = int(w)
    bs = int(bs)
    dm = int(dm)
    st = int(st)
    train, test = load_data()
    X, yc, yf = create_dataset2(train,target,w=w,st=st)
    if X.shape[0] < w:
        return -10
    # print(X[0:10])
    # print(yc[0:20])
    print(X.shape,w, bs, dm , dp)
    model = create_model(X.shape[1], 1, dm = dm, dp=dp)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    history = model.fit(X, yc, batch_size=bs, epochs=100, validation_split=0.05, callbacks=[es], shuffle=False,verbose=0)
  #  print(print(history.history))
    minepoch = np.argmin(np.asarray(history.history['val_loss']))
    mn_loss = history.history['val_loss'][minepoch]
    print(minepoch, mn_loss, w, bs,dm, dp)
    return -mn_loss

def train_expsmooth(n,a, b):
    n = int(n)
    a = int(a)
    b = int(b)
    feature_day = [1, a,  100, 200, 500, 1000, 2000, 5000, b, 20000]
    train_, test_ = load_data()
    test = train_[train_['Date']>'2020-04-10']
    train = train_[train_['Date'] < '2020-04-11']
    print(test.shape, train.shape)
    def CreateInput(data,country,province):
        feature = []
        for day in feature_day:
            # Get information in train data
            data.loc[:, 'Number day from ' + str(day) + ' case'] = 0
            if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (
                    train['ConfirmedCases'] < day)]['Date'].count() > 0):
                fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (
                            train['ConfirmedCases'] < day)]['Date'].max()
            else:
                fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)][
                    'Date'].min()
            for i in range(0, len(data)):
                if (data['Date'].iloc[i] > fromday):
                    day_denta = data['Date'].iloc[i] - fromday
                    data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days
            feature = feature + ['Number day from ' + str(day) + ' case']
        return data[feature]
    pred_data_all = pd.DataFrame()
    predc = None
    actualc = None
    predf = None
    actualf = None
    for country in train['Country_Region'].unique():
        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():
            df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]
            df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
            X_train = CreateInput(df_train,country,province)
            y_train_confirmed = df_train['ConfirmedCases'].ravel()
            y_train_fatalities = df_train['Fatalities'].ravel()
            X_pred = CreateInput(df_test,country,province)

            # Only train above 50 cases
            for day in sorted(feature_day, reverse=True):
                feature_use = 'Number day from ' + str(day) + ' case'
                idx = X_train[X_train[feature_use] == 0].shape[0]
                if (X_train[X_train[feature_use] > 0].shape[0] >= n):
                    break

            adjusted_y_train_confirmed = y_train_confirmed[idx:]
            adjusted_y_train_fatalities = y_train_fatalities[idx:]  # .values.reshape(-1, 1)
        #    print(country, province, idx, day, len(adjusted_y_train_confirmed),adjusted_y_train_confirmed[0])
            idx = X_pred[X_pred[feature_use] == 0].shape[0]
            pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
            max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)][
                'Date'].max()
            min_test_date = pred_data['Date'].min()
            model = ExponentialSmoothing(adjusted_y_train_confirmed, trend='additive').fit()
            y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
            y_valid_confirmed = df_test['ConfirmedCases'].ravel()
        #    print(RMSLE(y_hat_confirmed,y_valid_confirmed))
            if predc is None:
                predc = y_hat_confirmed
                actualc = y_valid_confirmed
            else:
                predc = np.hstack((predc,y_hat_confirmed))
                actualc = np.hstack((actualc, y_valid_confirmed))
            model = ExponentialSmoothing(adjusted_y_train_fatalities, trend='additive').fit()
            y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
            y_valid_fatalities = df_test['Fatalities'].ravel()
        #    print(RMSLE(y_hat_fatalities, y_valid_fatalities))
            if predf is None:
                predf = y_hat_fatalities
                actualf = y_valid_fatalities
            else:
                predf = np.hstack((predf,y_hat_fatalities))
                actualf = np.hstack((actualf, y_valid_fatalities))
            # print(y_hat_confirmed)
            # print(y_valid_confirmed)
    err = (RMSLE(predc, actualc) + RMSLE(predf, actualf))/2
    return -err

def opt_expsmooth():
    np.random.seed(21)
    optimizer = BayesianOptimization(train_expsmooth,{
        'n': (5,60),
        'a': (20,80),
        'b': (7000,15000)
    })
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        optimizer.maximize(init_points=2, n_iter=5)  #, acq='ei', xi=0.0
    print(optimizer.max)

def test_expsmooth(target,w=50):
    w = int(w)
    train, test = load_data()
    country_state = train["Country_State"].unique()
    dates = train["Date"].unique()

    Xc = np.zeros((dates.shape[0], country_state.shape[0]))
    Xf = np.zeros((dates.shape[0], country_state.shape[0]))
    for i, d in enumerate(dates):
        dayrec = train[train["Date"] == d].sort_values("Country_State")
        Xc[i, :] = dayrec['ConfirmedCases'].values
        Xf[i, :] = dayrec['Fatalities'].values
    pred = []
    ytrue = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        for i in range(69,80):
            for j in range(Xc.shape[1]):
                ytrue.append(Xc[i,j])
                ec = exp_smooth(Xc[19:69,j])
                pred.append(ec[0])
            print(i)
    pred = np.asarray(pred)
    ytrue = np.asarray(ytrue)
    err = RMSLE(pred, ytrue)
    print(err)


if __name__ == "__main__":
    start_time = time()
 #   train_expsmooth()
    opt_expsmooth()
  #  print(tf.version)

   # train_nn2(0, 15, 10, 10, 0.1)
   # print(root_mean_squared_log_error_0(np.array([1], dtype=np.float),np.array([2], dtype=np.float)))
   # opt_params(0)
    print("Time :", time() - start_time, "seconds")