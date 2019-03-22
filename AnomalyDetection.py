import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, GRU, SimpleRNN
from keras.optimizers import Adam, SGD, Adamax, RMSprop
from keras.callbacks import Callback
import pickle
# import matplotlib # if you using PuTTY, remember to cancle mark
# matplotlib.use('agg') # if you using PuTTY, remember to cancle mark
import matplotlib.pyplot as plt
# %matplotlib inline # ipython 支援語法

datasets_path = './datasets/'
savePlot_path = './plot/'

# ----- Deserialize the two numpy arrays -----
# UnicodeDecodeError: 'ascii' codec can't decode byte 0xa4 in position 0: ordinal not in range(128)
# 編碼問題，ASCII 只有 128 字元，不夠表示
# Solution：encoding='iso-8859-1'
# ISO-8859-1 為 ASCII 擴充，共可以以表示 256 字元
data_healthy = pickle.load(open(datasets_path + 'watsoniotp.healthy.phase_aligned.pickle', 'rb'), encoding='iso-8859-1')
data_broken = pickle.load(open(datasets_path + 'watsoniotp.broken.phase_aligned.pickle', 'rb'), encoding='iso-8859-1')
print ('data_healthy : \n', data_healthy)
print ('data_broken : \n', data_broken)

# ----- Set the correct shape for the data -----
# (row, col)
data_healthy = data_healthy.reshape(3000, 3)
data_broken = data_broken.reshape(3000, 3)
print ('data_healthy : \n', data_healthy)
print ('data_broken : \n', data_broken)

# ----- draw plot -----
# healthy's plot
subplots(facecolor(標記點內部顏色) = 'w'(白色), edgecolor(標記點邊緣顏色) = 'k'(黑色))
fig, ax = plt.subplots(num = None, figsize = (14, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')
size = len(data_healthy)
# plot(x, y:data_healthy[first_row:last_row, col], '-'(solid line style), animated = True(動畫))
ax.plot(range(0, size), data_healthy[:, 0], '-', color = 'blue', linewidth = 1)
ax.plot(range(0, size), data_healthy[:, 1], '-', color = 'red', linewidth = 1)
ax.plot(range(0, size), data_healthy[:, 2], '-', color = 'green', linewidth = 1)
# use putty cannot show，because has no tkinter
# Solution：save file or use mobaxterm'
plt.savefig(savePlot_path + 'init_healthy')
plt.show()
print ('Save healthy ok!')

# broken's plot
fig, ax = plt.subplots(num = None, figsize = (14, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')
size = len(data_healthy)
ax.plot(range(0, size), data_broken[:, 0], '-', color = 'blue', linewidth = 1)
ax.plot(range(0, size), data_broken[:, 1], '-', color = 'red', linewidth = 1)
ax.plot(range(0, size), data_broken[:, 2], '-', color = 'green', linewidth = 1)
plt.savefig(savePlot_path + 'init_broken')
plt.show()
print ('Save broken ok!')

# FFT(快速傅立葉轉換)，DSP(數位訊號處理)，將 signal 從 time domain(時域) -> frequency domain(頻域)
data_healthy_fft = np.fft.fft(data_healthy)
data_broken_fft = np.fft.fft(data_broken)
print ('data_healthy_fft : \n', data_healthy_fft)
print ('data_broken_fft : \n', data_broken_fft)

# healthy_fft's plot
fig, ax = plt.subplots(num = None, figsize = (14, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')
size = len(data_healthy_fft)
ax.plot(range(0, size), data_healthy_fft[:,0].real, '-', color = 'blue', linewidth = 1)
ax.plot(range(0, size), data_healthy_fft[:,1].imag, '-', color = 'red', linewidth = 1)
ax.plot(range(0, size), data_healthy_fft[:,2].real, '-', color = 'green', linewidth = 1)
plt.savefig(savePlot_path + 'healthy_fft')
plt.show()
print ('Save healthy_fft ok!')

# broken_fft's plot
fig, ax = plt.subplots(num = None, figsize = (14, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')
size = len(data_healthy_fft)
ax.plot(range(0, size), data_broken_fft[:,0].real, '-', color = 'blue', linewidth = 1)
ax.plot(range(0, size), data_broken_fft[:,1].imag, '-', color = 'red', linewidth = 1)
ax.plot(range(0, size), data_broken_fft[:,2].real, '-', color = 'green', linewidth = 1)
plt.savefig(savePlot_path + 'broken_fft')
plt.show()
print ('Save broken_fft ok!')

# ----- scale data to a range between 0 ~ 1 -----
def scaleData(data):
    # normalize features
    scaler = MinMaxScaler(feature_range = (0, 1))
    return scaler.fit_transform(data)

data_healthy_scaled = scaleData(data_healthy)
data_broken_scaled = scaleData(data_broken)
print ('data_healthy_scaled : \n', data_healthy_scaled)
print ('data_broken_scaled : \n', data_broken_scaled)

# ----- LSTMs want their input to contain windows of times -----
timesteps = 10
dim = 3
samples = 3000 # not use
data_healthy_scaled_reshaped = data_healthy_scaled # not used
# reshape to (300,10,3)
# 3000 / 10 = 300.0 -> float
# TypeError: 'float' object cannot be interpreted as an integer
# Solution：int()'
data_healthy_scaled_reshaped.shape = (int(samples / timesteps), timesteps, dim) # not used
print ('data_healthy_scaled_reshaped : \n', data_healthy_scaled_reshaped)

# ----- Create a callback by self -----
losses = []

# use to save all loss
def handleLoss(loss):
    global losses
    losses += [loss]
    print (loss)

# self callback history
class LossHistory(Callback):
    def on_train_begin(self, logs = {}):
        self.losses = [] # new losses array

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss')) # get loss from logs
        handleLoss(logs.get('loss'))

# design network
model = Sequential()
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(LSTM(input_shape = (timesteps, dim), units = 50, return_sequences = True))
model.add(Dense(units = 3))
model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['accuracy'])

# training
def train(data):
    print ('Training--------')
    data.shape = (300, 10, 3)
    # fit(verbose : 進度條 0 -> silent, 1 -> progress, 2 ->  one line per epoch)
    model.fit(data, data, batch_size = 72, epochs = 50, validation_data = (data, data), verbose = 1, shuffle = True, callbacks = [LossHistory()])
    data.shape = (3000, 3)

# predict # not to use
def score(data):
    print ('Score-----------')
    data.shape = (300, 10, 3)
    yhat =  model.predict(data) # return type is numpy array
    yhat.shape = (3000, 3)
    return yhat

trainHealthyTimes = 20

# use data_healthy_scaled will train 20次，start at i = 0
for i in range(trainHealthyTimes):
    print ("----------------healthy", i)
    train(data_healthy_scaled)
    yhat_healthy = score(data_healthy_scaled) # not to use
    yhat_broken = score(data_broken_scaled) # not to use
    data_healthy_scaled.shape = (3000, 3) # not to use
    data_broken_scaled.shape = (3000, 3) # not to use

# use data_broken_scaled only train 1次
print ("----------------broken")
train(data_broken_scaled)
yhat_healthy = score(data_healthy_scaled) # not to use
yhat_broken = score(data_broken_scaled) # not to use
data_healthy_scaled.shape = (3000, 3) # not to use
data_broken_scaled.shape = (3000, 3) # not to use

# result's plot
fig, ax = plt.subplots(num = None, figsize = (14, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')
size = len(losses)
ax.plot(range(0, size), losses, '-', color = 'blue', linewidth = 1)
plt.savefig(savePlot_path + 'resultLSTM_training-' + str(trainHealthyTimes) + '_adam' + '_hidden-10')
plt.show()
print ('Save result ok!')