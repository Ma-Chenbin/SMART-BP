import numpy as np
from scipy import signal
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split


def traditional_filter(ppg_signal, filter_type='median', filter_size=3):
    """
    - 'median': Applies a median filter to remove noise.
                You can specify the filter size using the filter_size parameter.
    - 'mean': Applies a moving average filter to smooth the signal.
              You can specify the window size using the filter_size parameter.
    - 'butterworth': Applies a Butterworth low-pass filter to remove high-frequency noise.
                     You can adjust the filter order (order parameter), cutoff frequency (cutoff parameter),
                     and sampling frequency (fs parameter) based on your specific requirements.
        """
    # Apply optional filters to denoise the PPG signal
    if filter_type == 'median':
        filtered_signal = signal.medfilt(ppg_signal, filter_size)
    elif filter_type == 'mean':
        kernel = np.ones(filter_size) / filter_size
        filtered_signal = np.convolve(ppg_signal, kernel, mode='same')
    elif filter_type == 'butterworth':
        order = 4  # Filter order
        cutoff = 2  # Cutoff frequency in Hz (adjust as needed)
        fs = 100  # Sampling frequency in Hz (adjust as needed)
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        filtered_signal = signal.lfilter(b, a, ppg_signal)
    else:
        filtered_signal = ppg_signal

    return filtered_signal



def wavelet_filter(ppg_signal, wavelet_type='db4', threshold_type='soft', threshold_level=3,
                       filter_type='butterworth', filter_order=4, cutoff=0.5, fs=100):
    # Decompose the PPG signal using wavelet transform
    coeffs = pywt.wavedec(ppg_signal, wavelet_type, level=threshold_level)

    # Apply thresholding to remove noise
    thresholded_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            threshold = np.median(np.abs(coeff)) / 0.6745  # Threshold for approximation coefficients
        else:
            threshold = np.median(np.abs(coeff)) / 0.6745 / np.sqrt(
                np.log(len(coeff)))  # Threshold for detail coefficients
        thresholded_coeff = pywt.threshold(coeff, value=threshold, mode=threshold_type)
        thresholded_coeffs.append(thresholded_coeff)

    # Reconstruct the denoised signal using inverse wavelet transform
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_type)

    # Apply optional filters to further denoise the signal
    if filter_type == 'butterworth':
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(filter_order, normalized_cutoff, btype='low', analog=False)
        denoised_signal = signal.lfilter(b, a, denoised_signal)
    elif filter_type != 'none':
        raise ValueError("Unrecognized filter type.")

    return denoised_signal



def create_denoising_model(input_shape, model_type='autoencoder'):
    """
    - 'autoencoder': Creates a fully connected autoencoder model.
    - 'conv_autoencoder': Creates a convolutional autoencoder model.
    - 'cnn_lstm': Creates a combination of convolutional and LSTM layers.
    """

    if model_type == 'autoencoder':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_shape[0])
        ])
    elif model_type == 'conv_autoencoder':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.MaxPooling1D(2, padding='same'),
            tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2, padding='same'),
            tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv1D(1, 3, activation='linear', padding='same')
        ])
    elif model_type == 'cnn_lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(input_shape[0], activation='linear')
        ])
    else:
        raise ValueError("Unrecognized model type.")

    return model



def learning_based_filter(ppg_signal, model_type='autoencoder', model_weights=None, epochs=20, batch_size=32,
                       test_size=0.2):
    # Normalize the PPG signal to range [0, 1]
    ppg_signal = (ppg_signal - np.min(ppg_signal)) / (np.max(ppg_signal) - np.min(ppg_signal))

    # Split the signal into training and testing sets
    X_train, X_test = train_test_split(ppg_signal, test_size=test_size, random_state=42)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Create the denoising model
    input_shape = X_train.shape[1:]
    model = create_denoising_model(input_shape, model_type)

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)

    # Load pre-trained model weights if available
    if model_weights is not None:
        model.load_weights(model_weights)

    # Denoise the testing set
    denoised_test = model.predict(X_test)
    denoised_test = denoised_test.squeeze(axis=-1)

    # Restore the denoised signal to its original range
    denoised_signal = denoised_test * (np.max(ppg_signal) - np.min(ppg_signal)) + np.min(ppg_signal)

    return denoised_signal
