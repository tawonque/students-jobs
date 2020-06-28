
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor



def make_estimator(mode='autoencoder'):
    def make_autoencoder():

        # set seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # this is the size of our encoded representations
        encoding_dim = 10  # 19 floats -> arbitrary, because there are roughly 20 criteria represented in 86 variables

        # this is our input placeholder
        input_img = Input(shape=(86,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(86, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)


        # create a placeholder for an encoded (20-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error') #mean_squared_error binary_crossentropy

        return autoencoder, encoder

    autoencoder, encoder = make_autoencoder()

    estimator = []
    if mode == 'autoencoder':
        estimator.append(('autoencoder', KerasRegressor(build_fn=autoencoder)))
    elif mode == 'encoder':
        estimator.append('encoder', KerasRegressor(build_fn=encoder))
    
    return estimator