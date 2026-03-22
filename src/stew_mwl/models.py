
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _spatial_channel_mean_max(t, name_prefix: str):
    """Keras-3-safe pooling over channel axis for CBAM spatial attention."""
    avg = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_ch_mean",
    )(t)
    mx = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_ch_max",
    )(t)
    return avg, mx


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

        def test_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            feature_map, z_mean, z_log_var, z = self.encoder(data, training=False)
            reconstruction = self.decoder(z, training=False)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }

    vae = VAE(encoder, decoder, name="vae")
    return vae, encoder, decoder

def cbam_block(
    inputs,
    reduction_ratio=8,
    spatial_kernel=7,
    name_prefix="cbam",
    order: str = "channel_spatial",
):
    """
    CBAM: `channel_spatial` (default), `spatial_channel` (spatial then channel, PRD sequential_spatial_channel),
    or `parallel` (channel and spatial masks both from the input, then fused by multiply — PRD parallel).
    """
    o = (order or "channel_spatial").lower()
    if o == "spatial_channel":
        return _cbam_spatial_then_channel(inputs, reduction_ratio, spatial_kernel, name_prefix)
    if o == "parallel":
        return _cbam_parallel(inputs, reduction_ratio, spatial_kernel, name_prefix)
    return _cbam_channel_then_spatial(inputs, reduction_ratio, spatial_kernel, name_prefix)


def _cbam_channel_then_spatial(inputs, reduction_ratio, spatial_kernel, name_prefix):
    ch = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)

    shared_dense_1 = layers.Dense(max(ch // reduction_ratio, 1), activation="relu", name=f"{name_prefix}_mlp1")
    shared_dense_2 = layers.Dense(ch, activation=None, name=f"{name_prefix}_mlp2")

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_attn = layers.Activation("sigmoid", name=f"{name_prefix}_channel_sigmoid")(layers.Add()([avg_out, max_out]))
    channel_attn = layers.Reshape((1, 1, ch))(channel_attn)
    x = layers.Multiply(name=f"{name_prefix}_channel_mul")([inputs, channel_attn])

    avg_pool_spatial, max_pool_spatial = _spatial_channel_mean_max(x, f"{name_prefix}_cs")
    spatial = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attn = layers.Conv2D(
        1, kernel_size=spatial_kernel, padding="same", activation="sigmoid", name=f"{name_prefix}_spatial_conv"
    )(spatial)
    x = layers.Multiply(name=f"{name_prefix}_spatial_mul")([x, spatial_attn])
    return x


def _cbam_parallel(inputs, reduction_ratio, spatial_kernel, name_prefix):
    """Channel and spatial attention both computed from `inputs`, then applied together."""
    ch = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    shared_dense_1 = layers.Dense(max(ch // reduction_ratio, 1), activation="relu", name=f"{name_prefix}_par_mlp1")
    shared_dense_2 = layers.Dense(ch, activation=None, name=f"{name_prefix}_par_mlp2")
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_attn = layers.Activation("sigmoid", name=f"{name_prefix}_par_ch_sigmoid")(layers.Add()([avg_out, max_out]))
    channel_attn = layers.Reshape((1, 1, ch))(channel_attn)

    avg_pool_spatial, max_pool_spatial = _spatial_channel_mean_max(inputs, f"{name_prefix}_par")
    spatial = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attn = layers.Conv2D(
        1, kernel_size=spatial_kernel, padding="same", activation="sigmoid", name=f"{name_prefix}_par_sp_conv"
    )(spatial)
    x = layers.Multiply(name=f"{name_prefix}_par_ch")([inputs, channel_attn])
    return layers.Multiply(name=f"{name_prefix}_par_sp")([x, spatial_attn])


def _cbam_spatial_then_channel(inputs, reduction_ratio, spatial_kernel, name_prefix):
    ch = inputs.shape[-1]
    avg_pool_spatial, max_pool_spatial = _spatial_channel_mean_max(inputs, f"{name_prefix}_sc")
    spatial = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attn = layers.Conv2D(
        1, kernel_size=spatial_kernel, padding="same", activation="sigmoid", name=f"{name_prefix}_spatial_first_conv"
    )(spatial)
    x = layers.Multiply(name=f"{name_prefix}_spatial_first_mul")([inputs, spatial_attn])

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    shared_dense_1 = layers.Dense(max(ch // reduction_ratio, 1), activation="relu", name=f"{name_prefix}_mlp1_sc")
    shared_dense_2 = layers.Dense(ch, activation=None, name=f"{name_prefix}_mlp2_sc")
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_attn = layers.Activation("sigmoid", name=f"{name_prefix}_channel_sc_sigmoid")(layers.Add()([avg_out, max_out]))
    channel_attn = layers.Reshape((1, 1, ch))(channel_attn)
    x = layers.Multiply(name=f"{name_prefix}_channel_sc_mul")([x, channel_attn])
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
    blstm_units: int = 20,
    cbam_reduction_ratio: int = 8,
    cbam_spatial_kernel: int = 7,
    cbam_attention_order: str = "channel_spatial",
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
        x = cbam_block(
            x,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel=cbam_spatial_kernel,
            name_prefix="cbam",
            order=cbam_attention_order,
        )
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
            rnn = layers.LSTM(blstm_units, return_sequences=False)
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


def build_proposed_model(cfg, n_channels: int = 3):
    """
    PRD H1 entry point: build the VAE-style encoder + optional CBAM + BiLSTM classifier graph (uncompiled).
    `n_channels` is the number of topomap channels (typically 3 for θ/α/β).
    """
    from .config import CLASS_NAMES, Config

    if not isinstance(cfg, Config):
        raise TypeError("cfg must be a stew_mwl.config.Config instance")
    seq_len = cfg.seq_len
    return build_classifier_from_encoder(
        frame_shape=(cfg.image_h, cfg.image_w, n_channels),
        sequence_length=seq_len,
        latent_dim=cfg.latent_dim,
        n_classes=len(CLASS_NAMES),
        dropout=cfg.dropout,
        use_cbam=cfg.cbam_enabled,
        use_encoder=True,
        bidirectional=True,
        sequence_model="lstm",
        blstm_units=cfg.blstm_units,
        cbam_reduction_ratio=cfg.cbam_reduction_ratio,
        cbam_spatial_kernel=cfg.cbam_spatial_kernel,
        cbam_attention_order=cfg.cbam_attention_order,
    )


def compile_classifier(
    model,
    learning_rate=1e-3,
    use_decay: bool = False,
    decay_steps: int = 1000,
    schedule: str = "exponential",
):
    """
    Learning-rate schedule: `exponential` (manuscript-style decay), `cosine`, or fixed when use_decay is False.
    """
    if not use_decay or (schedule or "none").lower() == "none":
        lr = learning_rate
    elif (schedule or "").lower() == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps)
    else:
        lr = keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=0.96,
            staircase=True,
        )
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _cnn_frame_stack(frame_input, use_encoder: bool):
    x = frame_input
    if use_encoder:
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2, name="shared_feature_map")(x)
    else:
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2, name="shared_feature_map")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    return x


def build_blstm_lstm_classifier(
    frame_shape=(80, 60, 3),
    sequence_length=10,
    n_classes=4,
    dropout=0.3,
    blstm_units=20,
    lstm_units=20,
    use_encoder: bool = False,
):
    frame_input = keras.Input(shape=frame_shape, name="frame_input")
    x = _cnn_frame_stack(frame_input, use_encoder=use_encoder)
    frame_encoder = keras.Model(frame_input, x, name="frame_encoder_blstm")

    seq_input = keras.Input(shape=(sequence_length, *frame_shape), name="sequence_input")
    x = layers.TimeDistributed(frame_encoder, name="td_frame_encoder")(seq_input)
    x = layers.Bidirectional(layers.LSTM(blstm_units, return_sequences=True))(x)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(n_classes, activation="softmax", name="classifier")(x)
    return keras.Model(seq_input, output, name="blstm_lstm_classifier")


def copy_vae_encoder_weights_to_classifier(vae: keras.Model, classifier: keras.Model) -> None:
    """Align early Conv2D / MaxPool weights from the VAE encoder into the classifier frame encoder."""
    td = classifier.get_layer("td_frame_encoder")
    fe = td.layer
    enc = vae.encoder
    enc_layers = [L for L in enc.layers if isinstance(L, (layers.Conv2D, layers.MaxPooling2D))]
    fe_layers = [L for L in fe.layers if isinstance(L, (layers.Conv2D, layers.MaxPooling2D))]
    for e_layer, f_layer in zip(enc_layers[:4], fe_layers[:4]):
        ew, fw = e_layer.get_weights(), f_layer.get_weights()
        if ew and fw and len(ew) == len(fw):
            try:
                if ew[0].shape == fw[0].shape:
                    f_layer.set_weights(ew)
            except Exception:
                pass
