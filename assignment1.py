import tensorflow as tf
import numpy as np
import soundfile as sf
from tensorflow import keras
from tensorflow.keras import layers
import librosa

# Path to the TFRecord dataset
TFRECORD_PATH = 'dataset/datasets_maestro_v3.0.0_maestro-v3.0.0_ns_wav_test.tfrecord-00000-of-00025'

# Example: parse TFRecord for audio (update feature description as needed)
def _parse_function(proto):
    feature_description = {
        'audio': tf.io.FixedLenFeature([], tf.string),
        # Removed 'sample_rate' since it's not present in the TFRecord
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    audio = tf.io.decode_raw(parsed_features['audio'], tf.float32)
    return audio

def load_dataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    # Debug: print a sample
    for audio in parsed_dataset.take(1):
        print('Sample audio shape:', audio.shape)
        print('Sample audio values:', audio[:10].numpy())
    return parsed_dataset

# Dummy model for unconditioned generation (replace with your model)
def generate_waveform(length=16000):
    # Example: white noise as placeholder
    return np.random.randn(length).astype(np.float32)

# If your model generates spectrograms, add a function to convert to waveform
# def spectrogram_to_waveform(spectrogram):
#     # Use Griffin-Lim or a neural vocoder here
#     pass

# Simple generator model: maps noise to waveform
class SimpleGenerator(keras.Model):
    def __init__(self, output_length):
        super().__init__()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(1024, activation='relu')
        self.dense3 = layers.Dense(output_length, activation='tanh')
    def call(self, z):
        x = self.dense1(z)
        x = self.dense2(x)
        return self.dense3(x)

def train_simple_generator(dataset, output_length, epochs=1, batch_size=16):
    # Prepare dataset: sample random segment if possible, skip all-zero
    def preprocess(audio):
        audio = tf.reshape(audio, [-1])
        audio_len = tf.shape(audio)[0]
        def get_segment():
            start = tf.random.uniform([], 0, audio_len - output_length + 1, dtype=tf.int32)
            return audio[start:start+output_length]
        def pad_audio():
            return tf.pad(audio, [[0, output_length - audio_len]])[:output_length]
        segment = tf.cond(audio_len >= output_length, get_segment, pad_audio)
        # Remove NaN/Inf before normalization
        segment = tf.where(tf.math.is_finite(segment), segment, tf.zeros_like(segment))
        max_val = tf.reduce_max(tf.abs(segment))
        segment = tf.cond(max_val > 0, lambda: segment / max_val, lambda: segment)
        # Replace any remaining NaN/Inf with zero
        segment = tf.where(tf.math.is_finite(segment), segment, tf.zeros_like(segment))
        return segment
    ds = dataset.map(preprocess)
    ds = ds.filter(lambda x: tf.reduce_any(tf.not_equal(x, 0.0)))
    ds = ds.batch(batch_size).shuffle(100)
    # Debug: print first batch
    for batch in ds.take(1):
        print('First training batch shape:', batch.shape)
        print('First training batch values:', batch[0][:10].numpy())
    # Model and optimizer
    model = SimpleGenerator(output_length)
    optimizer = keras.optimizers.Adam(1e-3)
    loss_fn = keras.losses.MeanSquaredError()
    # Training loop (very basic)
    for epoch in range(epochs):
        for batch in ds:
            noise = tf.random.normal([tf.shape(batch)[0], 100])
            with tf.GradientTape() as tape:
                generated = model(noise)
                loss = loss_fn(batch, generated)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
    return model

def generate_with_model(model, output_length):
    noise = tf.random.normal([1, 100])
    generated = model(noise, training=False).numpy().reshape(-1)
    return generated

# Improved 1D CNN generator model: deeper, more filters, batch norm, leaky relu
class BetterCNNGenerator(keras.Model):
    def __init__(self, output_length):
        super().__init__()
        self.output_length = output_length
        self.dense = layers.Dense((output_length // 64) * 256, activation='relu')
        self.reshape = layers.Reshape((output_length // 64, 256))
        self.conv1 = layers.Conv1DTranspose(128, 25, strides=4, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv1DTranspose(64, 25, strides=4, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU(0.2)
        self.conv3 = layers.Conv1DTranspose(32, 25, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.LeakyReLU(0.2)
        self.conv4 = layers.Conv1D(1, 25, padding='same', activation='tanh')
        self.flatten = layers.Flatten()
    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        # Ensure output is exactly output_length
        x = x[..., :self.output_length]
        x = tf.cond(tf.shape(x)[-1] < self.output_length,
                    lambda: tf.pad(x, [[0,0],[0, self.output_length - tf.shape(x)[-1]]]),
                    lambda: x)
        return x

def train_better_cnn_generator(dataset, output_length, epochs=30, batch_size=16):
    def preprocess(audio):
        audio = tf.reshape(audio, [-1])
        audio_len = tf.shape(audio)[0]
        def get_segment():
            start = tf.random.uniform([], 0, audio_len - output_length + 1, dtype=tf.int32)
            return audio[start:start+output_length]
        def pad_audio():
            return tf.pad(audio, [[0, output_length - audio_len]])[:output_length]
        segment = tf.cond(audio_len >= output_length, get_segment, pad_audio)
        segment = tf.where(tf.math.is_finite(segment), segment, tf.zeros_like(segment))
        max_val = tf.reduce_max(tf.abs(segment))
        segment = tf.cond(max_val > 0, lambda: segment / max_val, lambda: segment)
        segment = tf.where(tf.math.is_finite(segment), segment, tf.zeros_like(segment))
        return segment
    ds = dataset.map(preprocess)
    ds = ds.filter(lambda x: tf.reduce_any(tf.not_equal(x, 0.0)))
    ds = ds.batch(batch_size).shuffle(200)
    for batch in ds.take(1):
        print('First BetterCNN training batch shape:', batch.shape)
        print('First BetterCNN training batch values:', batch[0][:10].numpy())
    model = BetterCNNGenerator(output_length)
    optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
    loss_fn = keras.losses.MeanSquaredError()
    for epoch in range(epochs):
        for batch in ds:
            noise = tf.random.normal([tf.shape(batch)[0], 100])
            with tf.GradientTape() as tape:
                generated = model(noise)
                loss = loss_fn(batch, generated)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"BetterCNN Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
    return model

def generate_with_better_cnn_model(model, output_length):
    noise = tf.random.normal([1, 100])
    generated = model(noise, training=False).numpy().reshape(-1)
    return generated

# Discriminator model for GAN training: 1D CNN classifier
class Discriminator(keras.Model):
    def __init__(self, input_length):
        super().__init__()
        self.reshape = layers.Reshape((input_length, 1))
        self.conv1 = layers.Conv1D(32, 25, strides=4, padding='same')
        self.act1 = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv1D(64, 25, strides=4, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU(0.2)
        self.conv3 = layers.Conv1D(128, 25, strides=4, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.LeakyReLU(0.2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)
    def call(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.flatten(x)
        return self.fc(x)

# GAN training: train generator and discriminator adversarially
def train_gan(dataset, output_length, epochs=50, batch_size=16):
    # preprocess as before
    def preprocess(audio):
        audio = tf.reshape(audio, [-1])
        audio_len = tf.shape(audio)[0]
        segment = tf.cond(audio_len >= output_length,
                          lambda: audio[tf.random.uniform([],0,audio_len-output_length+1,dtype=tf.int32):][:output_length],
                          lambda: tf.pad(audio, [[0, output_length-audio_len]]))
        segment = tf.where(tf.math.is_finite(segment), segment, tf.zeros_like(segment))
        max_val = tf.reduce_max(tf.abs(segment))
        segment = tf.cond(max_val>0, lambda: segment/max_val, lambda: segment)
        return segment
    ds = dataset.map(preprocess).filter(lambda x: tf.reduce_any(x!=0.0))
    ds = ds.shuffle(500).batch(batch_size)
    # instantiate models
    gen = BetterCNNGenerator(output_length)
    disc = Discriminator(output_length)
    # optimizers and loss
    g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    # labels
    real_labels = tf.ones((batch_size,1))
    fake_labels = tf.zeros((batch_size,1))
    for epoch in range(epochs):
        for real in ds:
            noise = tf.random.normal([tf.shape(real)[0], 100])
            # train discriminator
            with tf.GradientTape() as tape_d:
                fake = gen(noise)
                d_real = disc(real)
                d_fake = disc(fake)
                d_loss = bce(real_labels[:tf.shape(real)[0]], d_real) + bce(fake_labels[:tf.shape(real)[0]], d_fake)
            grads_d = tape_d.gradient(d_loss, disc.trainable_variables)
            d_opt.apply_gradients(zip(grads_d, disc.trainable_variables))
            # train generator
            with tf.GradientTape() as tape_g:
                fake2 = gen(noise)
                d_fake2 = disc(fake2)
                g_loss = bce(real_labels[:tf.shape(real)[0]], d_fake2)
            grads_g = tape_g.gradient(g_loss, gen.trainable_variables)
            g_opt.apply_gradients(zip(grads_g, gen.trainable_variables))
        print(f"Epoch {epoch+1}, D_loss: {d_loss.numpy():.4f}, G_loss: {g_loss.numpy():.4f}")
    return gen

# ----- Spectrogram-based generation functions -----

def compute_average_mel(dataset, sample_rate, n_fft=1024, hop_length=256, n_mels=128, segments=20):
    mel_accum = None
    count = 0
    for audio in dataset.take(segments):
        # Convert tensor to numpy and clean non-finite values
        y = audio.numpy().astype(np.float32)
        # Replace NaN or Inf with zeros
        y = np.where(np.isfinite(y), y, 0.0)
        mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels)
        if mel_accum is None:
            mel_accum = mel
        else:
            mel_accum += mel
        count += 1
    return mel_accum / count


def generate_mel_variation(avg_mel, scale=0.1):
    noise = np.random.randn(*avg_mel.shape) * scale * np.max(avg_mel)
    return np.clip(avg_mel + noise, a_min=0, a_max=None)


def mel_to_waveform(mel, sample_rate, n_fft=1024, hop_length=256, n_iter=60):
    # Invert mel-spectrogram to linear spectrogram
    S = librosa.feature.inverse.mel_to_stft(mel, sr=sample_rate, n_fft=n_fft)
    # Reconstruct waveform via Griffin-Lim
    y = librosa.griffinlim(S, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
    return y

if __name__ == '__main__':
    dataset = load_dataset(TFRECORD_PATH)
    output_length = 16000 * 5  # 5 seconds at 16kHz
    # GAN generation
    model = train_gan(dataset, output_length, epochs=50, batch_size=8)
    generated = generate_with_better_cnn_model(model, output_length)
    sf.write('generated_gan.wav', generated, 16000)
    print('GAN-generated waveform saved to generated_gan.wav')

    # Spectrogram-based generation (average Mel + variation)
    sr = 16000
    avg_mel = compute_average_mel(dataset, sr)
    gen_mel = generate_mel_variation(avg_mel, scale=0.2)
    gen_wave = mel_to_waveform(gen_mel, sr)
    sf.write('generated_mel.wav', gen_wave, sr)
    print('Mel-spectrogram-based waveform saved to generated_mel.wav')