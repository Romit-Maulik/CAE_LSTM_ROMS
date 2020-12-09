import numpy as np
import tensorflow as tf

# Set seeds
np.random.seed(10)
tf.random.set_seed(10)

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
data = np.load('./snapshot_matrix_pod.npy').T

# Scale the training data
scaler = StandardScaler()
data = scaler.fit_transform(data)
# Transpose (rows are DOF, columns are snapshots)
data = data.T

swe_train_data = np.zeros(shape=(np.shape(data)[1],64,64,3)) # Channels last
for i in range(np.shape(data)[1]):
    temp_1 = data[0:64*64,i].reshape(64,64)
    temp_2 = data[64*64:2*64*64,i].reshape(64,64)
    temp_3 = data[2*64*64:3*64*64,i].reshape(64,64)
    swe_train_data[i,:,:,0] = np.transpose(temp_1[:,:])
    swe_train_data[i,:,:,1] = np.transpose(temp_2[:,:])
    swe_train_data[i,:,:,2] = np.transpose(temp_3[:,:])
    
    
data = np.load('./snapshot_matrix_test.npy').T
data = scaler.transform(data)
# Transpose (rows are DOF, columns are snapshots)
data = data.T

swe_test_data = np.zeros(shape=(np.shape(data)[1],64,64,3)) # Channels last
for i in range(np.shape(data)[1]):
    temp_1 = data[0:64*64,i].reshape(64,64)
    temp_2 = data[64*64:2*64*64,i].reshape(64,64)
    temp_3 = data[2*64*64:3*64*64,i].reshape(64,64)
    swe_test_data[i,:,:,0] = np.transpose(temp_1[:,:])
    swe_test_data[i,:,:,1] = np.transpose(temp_2[:,:])
    swe_test_data[i,:,:,2] = np.transpose(temp_3[:,:])
    
# Randomize train
idx =  np.arange(swe_train_data.shape[0])
np.random.shuffle(idx)
swe_train_data_randomized = swe_train_data[idx[:]]

time_window = 10 # The window size of the LSTM
mode = 'train'

lrate = 0.001
num_latent = 6

# Custom activation (swish)
def my_swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

# Define model architecture
weights_filepath = 'best_weights_cae.h5'
## Encoder
encoder_inputs = Input(shape=(64,64,3),name='Field')
# Encode   
x = Conv2D(30,kernel_size=(3,3),activation=my_swish,padding='same')(encoder_inputs)
enc_l2 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

x = Conv2D(25,kernel_size=(3,3),activation=my_swish,padding='same')(enc_l2)
enc_l3 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

x = Conv2D(20,kernel_size=(3,3),activation=my_swish,padding='same')(enc_l3)
enc_l4 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

x = Conv2D(15,kernel_size=(3,3),activation=my_swish,padding='same')(enc_l4)
enc_l5 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

x = Conv2D(10,kernel_size=(3,3),activation=my_swish,padding='same')(enc_l5)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

x = Flatten()(x)
x = Dense(50, activation=my_swish)(x)
x = Dense(25, activation=my_swish)(x)
x = Dense(10, activation=my_swish)(x)
encoded = Dense(num_latent)(x)
encoder = Model(inputs=encoder_inputs,outputs=encoded)
    
## Decoder
decoder_inputs = Input(shape=(num_latent,),name='decoded')
x = Dense(10,activation=my_swish)(decoder_inputs)
x = Dense(25,activation=my_swish)(x)
x = Dense(50,activation=my_swish)(x)
x = Dense(2*2*3,activation=my_swish)(x)

x = Reshape(target_shape=(2,2,3))(x)

x = Conv2D(10,kernel_size=(3,3),activation=my_swish,padding='same')(x)
dec_l1 = UpSampling2D(size=(2, 2))(x)

x = Conv2D(15,kernel_size=(3,3),activation=my_swish,padding='same')(dec_l1)
dec_l2 = UpSampling2D(size=(2, 2))(x)

x = Conv2D(20,kernel_size=(3,3),activation=my_swish,padding='same')(dec_l2)
dec_l3 = UpSampling2D(size=(2, 2))(x)

x = Conv2D(25,kernel_size=(3,3),activation=my_swish,padding='same')(dec_l3)
dec_l4 = UpSampling2D(size=(2, 2))(x)

x = Conv2D(30,kernel_size=(3,3),activation=my_swish,padding='same')(dec_l4)
dec_l5 = UpSampling2D(size=(2, 2))(x)

decoded = Conv2D(3,kernel_size=(3,3),activation='linear',padding='same')(dec_l5)
    
decoder = Model(inputs=decoder_inputs,outputs=decoded)

## Autoencoder
ae_outputs = decoder(encoder(encoder_inputs))
  
model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')
    
# design network
my_adam = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
callbacks_list = [checkpoint,earlystopping]

# fit network
model.compile(optimizer=my_adam,loss='mean_squared_error')    
model.summary()

num_epochs = 5000
batch_size = 128

if mode == 'train':
    train_history = model.fit(x=swe_train_data_randomized, 
                              y=swe_train_data_randomized, 
                              epochs=num_epochs, batch_size=batch_size, 
                              callbacks=callbacks_list, validation_split=0.1)

# Train LSTM
encoded_list = []
for i in range(90):
    encoded_list.append(K.eval(encoder(swe_train_data[100*i:100*(i+1),:,:,:].astype('float32'))))

encoded = np.asarray(encoded_list)

# Add parameter information
parameters = np.load('./Locations.npy')
parameters_train = parameters[:90]
parameters_test = parameters[90:]

# Prepare training data
lstm_training_data = np.copy(encoded)
num_train_snapshots = 90
total_size = np.shape(lstm_training_data)[0]*np.shape(lstm_training_data)[1]

# Shape the inputs and outputs
input_seq = np.zeros(shape=(total_size-time_window*num_train_snapshots,time_window,num_latent+2))
output_seq = np.zeros(shape=(total_size-time_window*num_train_snapshots,num_latent))

# Setting up inputs
sample = 0
for snapshot in range(num_train_snapshots):
    lstm_snapshot = lstm_training_data[snapshot,:,:]
    for t in range(time_window,100):
        input_seq[sample,:,:num_latent] = lstm_snapshot[t-time_window:t,:]
        input_seq[sample,:,num_latent:] = parameters_train[snapshot,:]
        output_seq[sample,:] = lstm_snapshot[t,:]
        sample = sample + 1

# Saving all the training data (for future use)
parameter_info = np.zeros(shape=(90,100,2),dtype='double')
# Setting up inputs
sample = 0
for snapshot in range(num_train_snapshots):
    parameter_info[snapshot,:,:] = parameters_train[snapshot,:]
        
total_training_data = np.concatenate((lstm_training_data,parameter_info),axis=-1)

# Model architecture
lstm_model = models.Sequential()
lstm_model.add(LSTM(50,input_shape=(time_window, num_latent+2),return_sequences=True))
lstm_model.add(LSTM(50,input_shape=(time_window, num_latent+2),return_sequences=True))
lstm_model.add(LSTM(50,input_shape=(time_window, num_latent+2),return_sequences=False))  #
lstm_model.add(Dense(num_latent, activation=None))

# training parameters
num_epochs = 500
batch_size = 24

# design network
lstm_filepath = 'lstm_weights.h5'
lstm_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
checkpoint = ModelCheckpoint(lstm_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min',save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
lstm_callbacks_list = [checkpoint,earlystopping]

# fit network
lstm_model.compile(optimizer=lstm_adam,loss='mean_squared_error')
lstm_train_history = lstm_model.fit(input_seq, output_seq, 
                                    epochs=num_epochs, batch_size=batch_size, 
                                    callbacks=lstm_callbacks_list, validation_split=0.1)