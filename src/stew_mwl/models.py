
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_vae(image_shape=(80,60,3), latent_dim=128):
    encoder_inputs = keras.Input(shape=image_shape, name="encoder_input")
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(encoder_inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    feature_map = layers.MaxPooling2D(pool_size=2, name="vae_feature_map")(x)
    flat = layers.Flatten()(feature_map)
    z_mean = layers.Dense(latent_dim, name="z_mean")(flat)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(flat)

    def sample(args):
        z_mean, z_log_var = args
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    z = layers.Lambda(sample, name="z")([z_mean, z_log_var])

    fm_shape = feature_map.shape[1:]
    decoder_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(int(fm_shape[0] * fm_shape[1] * fm_shape[2]), activation="relu")(decoder_inputs)
    x = layers.Reshape((int(fm_shape[0]), int(fm_shape[1]), int(fm_shape[2])))(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    decoder_outputs = layers.Conv2D(image_shape[-1], 3, padding="same", activation="sigmoid")(x)

    encoder = keras.Model(encoder_inputs, [feature_map, z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                feature_map, z_mean, z_log_var, z = self.encoder(data, training=True)
                reconstruction = self.decoder(z, training=True)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {m.name: m.result() for m in self.metrics}

    vae = VAE(encoder, decoder, name="vae")
    return vae, encoder, decoder

def cbam_block(inputs, reduction_ratio=8, spatial_kernel=7, name_prefix="cbam"):
    ch = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)

    shared_dense_1 = layers.Dense(max(ch // reduction_ratio, 1), activation="relu", name=f"{name_prefix}_mlp1")
    shared_dense_2 = layers.Dense(ch, activation=None, name=f"{name_prefix}_mlp2")

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_attn = layers.Activation("sigmoid", name=f"{name_prefix}_channel_sigmoid")(layers.Add()([avg_out, max_out]))
    channel_attn = layers.Reshape((1,1,ch))(channel_attn)
    x = layers.Multiply(name=f"{name_prefix}_channel_mul")([inputs, channel_attn])

    avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attn = layers.Conv2D(1, kernel_size=spatial_kernel, padding="same", activation="sigmoid", name=f"{name_prefix}_spatial_conv")(spatial)
    x = layers.Multiply(name=f"{name_prefix}_spatial_mul")([x, spatial_attn])
    return x

def build_classifier_from_encoder(
    frame_shape=(80,60,3),
    sequence_length=10,
    latent_dim=128,
    n_classes=4,
    dropout=0.3,
    use_cbam=True,
    use_encoder=True,
    bidirectional=True,
    sequence_model="lstm",
):
    # per-frame feature extractor
    frame_input = keras.Input(shape=frame_shape, name="frame_input")
    x = frame_input
    if use_encoder:
        _, encoder, _ = build_vae(frame_shape, latent_dim)
        feature_map = encoder.get_layer("vae_feature_map").output  # unavailable in a fresh model path
        # safer manual encoder replica
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2, name="shared_feature_map")(x)
    else:
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2, name="shared_feature_map")(x)

    if use_cbam:
        x = cbam_block(x, reduction_ratio=8, spatial_kernel=7, name_prefix="cbam")
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    frame_encoder = keras.Model(frame_input, x, name="frame_encoder")

    seq_input = keras.Input(shape=(sequence_length, *frame_shape), name="sequence_input")
    x = layers.TimeDistributed(frame_encoder, name="td_frame_encoder")(seq_input)

    if sequence_model == "cnn":
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
    else:
        if sequence_model == "lstm":
            rnn = layers.LSTM(20, return_sequences=False)
        else:
            raise ValueError("sequence_model must be one of: 'lstm', 'cnn'")
        if bidirectional:
            x = layers.Bidirectional(rnn)(x)
        else:
            x = rnn(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(n_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(seq_input, output, name="vae_cbam_bilstm_classifier")
    return model

def compile_classifier(model, learning_rate=1e-3):
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
