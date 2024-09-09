import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error
from data_formating import get_train_data

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        # Compute the score
        u_t = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        score = tf.tensordot(u_t, self.u, axes=1)

        attention_weights = tf.nn.softmax(score, axis=1) # Compute the attention weights


        weighted_input = inputs * tf.expand_dims(attention_weights, -1) # Apply the attention weights to the inputs
        
        return tf.reduce_sum(weighted_input, axis=1) # Return the context vector (not the attention weights)
    
input_length = 180
prediction_length = 40

if __name__ == "__main__":

    (X_train, X_test, y_train, y_test), scalers = get_train_data(inputL=input_length, outputL=prediction_length)

    model = Sequential([
        Input(shape=(input_length, 18)),
        LSTM(200, return_sequences=True),
        AttentionLayer(),  # Add custom attention layer after LSTM
        Dense(75, activation='relu'),
        Dense(prediction_length)
    ])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', 
                                     factor=0.5, 
                                     patience=5,  
                                     min_lr=1e-6, 
                                     verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error', 
                  metrics=['mae'])

    history = model.fit(X_train, y_train, 
                        batch_size=16,
                        epochs=50, 
                        validation_data=(X_test, y_test), 
                        callbacks=[lr_scheduler, early_stopping]) 

    model.save("LSTM_40.keras")

    loss, mae = model.evaluate(X_test, y_test)
    print(f'Model Loss: {loss}')
    print(f'MAE: {mae}')

    predictions = model.predict(X_test)



    y_test_rescaled = []
    predictions_rescaled = []


    for i in range(len(predictions)):
        stock_idx = i % len(scalers)  
        stock_name = list(scalers.keys())[stock_idx]  
        scaler_close = scalers[stock_name]['Close']  


        y_test_rescaled.append(scaler_close.inverse_transform(y_test[i].reshape(-1, 1)).flatten())
        predictions_rescaled.append(scaler_close.inverse_transform(predictions[i].reshape(-1, 1)).flatten())


    y_test_rescaled = np.array(y_test_rescaled)
    predictions_rescaled = np.array(predictions_rescaled)


    mape_rescaled = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100
    print(f'Rescaled MAPE: {mape_rescaled}%')
