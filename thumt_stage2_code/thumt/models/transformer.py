# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy,code
import numpy as np
import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def _layer_process(x, mode, trainable=True):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x, trainable=trainable)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None, trainable=True):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True, trainable=trainable)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True, trainable=trainable)

        return output

def _load_embedding(word_list, params, uniform_scale = 0.25, dimension_size = 300, embed_file='glove'):

    word2embed = {}
    if embed_file == 'w2v':
        file_path = params.embedding_path
    else:
        file_path = params.embedding_path

    with open(file_path, 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
    word_vectors = []

    c = 0
    for word in word_list:
        if word in word2embed:
            c += 1
            s = np.array(word2embed[word], dtype=np.float32)
            word_vectors.append(s)
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))

    print('glove initializes {}'.format(c))
    print('all words initializes {}'.format(len(word_vectors)))

    return np.array(word_vectors, dtype=np.float32)

def birnn(inputs, sequence_length, params):
    lstm_fw_cell = rnn.BasicLSTMCell(params.hidden_size)
    lstm_bw_cell = rnn.BasicLSTMCell(params.hidden_size)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs,
                                                 sequence_length=sequence_length, dtype=tf.float32)
    states_fw, states_bw = outputs
    return tf.concat([states_fw, states_bw], axis=2)

def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def w_encoder_attention(queries,
                        keys,
                        sequence_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        using_mask=False,
                        mymasks=None,
                        scope="w_encoder_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        # print(queries)
        # print(queries.get_shape().as_list)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        # Linear projections

        Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, use_bias=False)  # (N, T_k, C)

        x = K * Q
        x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],num_heads, int(num_units/num_heads)])
        outputs = tf.transpose(tf.reduce_sum(x, 3),[0,2,1])
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        if using_mask:
            key_masks = mymasks
            key_masks = tf.reshape(tf.tile(key_masks, [1, num_heads]),
                                   [tf.shape(key_masks)[0], num_heads, tf.shape(key_masks)[1]])
        else:
            key_masks = tf.sequence_mask(sequence_length, tf.shape(keys)[1], dtype=tf.float32)
            key_masks = tf.reshape(tf.tile(key_masks,[1, num_heads]),[tf.shape(key_masks)[0],num_heads,tf.shape(key_masks)[1]])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs, 2)
        V_ = tf.reshape(V, [tf.shape(V)[0], tf.shape(V)[1], num_heads, int(num_units / num_heads)])
        V_ = tf.transpose(V_, [0, 2, 1, 3])
        outputs = tf.layers.dense(tf.reshape(tf.reduce_sum(V_ * tf.expand_dims(outputs, -1), 2), [-1, num_units]),
                                  num_units, activation=None, use_bias=False)
        weight = outputs
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs, weight

def transformer_context(inputs, bias, params, dtype=None, scope="ctx_transformer", trainable=True):
    with tf.variable_scope(scope, default_name="context", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_context_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        trainable=trainable
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=trainable
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs

def transformer_encoder(inputs, bias, params, dia_mask=None, dtype=None, scope=None, trainable=True, get_first_layer=False):
    with tf.variable_scope("encoder", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_encoder_layers):
            if layer < params.bottom_block:
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        max_relative_dis = params.max_relative_dis \
                            if params.position_info_type == 'relative' else None
    
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                            trainable=trainable
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                    with tf.variable_scope("feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            trainable=trainable
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)
                first_layer_output = x
                #print("first_layer_output", first_layer_output)
                if get_first_layer and layer == (params.bottom_block - 1):
                    return x, first_layer_output
            else:
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        max_relative_dis = params.max_relative_dis \
                            if params.position_info_type == 'relative' else None

                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                            trainable=trainable,
                            dia_mask=dia_mask
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                    with tf.variable_scope("feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            trainable=trainable
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

#            if params.bottom_block and get_first_layer:
#                return first_layer_output, first_layer_output

        outputs = _layer_process(x, params.layer_preprocess)
        if params.bottom_block == 0:
            first_layer_output = x

        return outputs, first_layer_output


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None, trainable=True):
    with tf.variable_scope("decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias], reuse=tf.AUTO_REUSE):
#    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
#                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                        trainable=trainable
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                        trainable=trainable
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=trainable
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    sample_seq = features["sample"]
    print(features)
    ctx_dia_src_seq = features["context_dia_src"]
    ctx_dia_tgt_seq = features["context_dia_tgt"]
    ctx_sty_src_seq = features["context_sty_src"]
    ctx_sty_tgt_seq = features["context_sty_tgt"]
    ctx_lan_src_seq = features["context_lan_src"]
    ctx_lan_tgt_seq = features["context_lan_tgt"]


    #emotion = features["emotion"]
    src_len = features["source_length"]
    sample_len = features["sample_length"]

    ctx_dia_src_len = features["context_dia_src_length"]
    ctx_dia_tgt_len = features["context_dia_tgt_length"]
    ctx_sty_src_len = features["context_sty_src_length"]
    ctx_sty_tgt_len = features["context_sty_tgt_length"]
    ctx_lan_src_len = features["context_lan_src_length"]
    ctx_lan_tgt_len = features["context_lan_tgt_length"]


    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    top_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=dtype or tf.float32)

    dia_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=dtype or tf.float32)

    true_mask = dia_mask * top_mask

    ctx_dia_src_mask = tf.sequence_mask(ctx_dia_src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=tf.float32)
    ctx_dia_tgt_mask = tf.sequence_mask(ctx_dia_tgt_len,
                                maxlen=tf.shape(features["context_dia_tgt"])[1],
                                dtype=tf.float32)

    ctx_sty_src_mask = tf.sequence_mask(ctx_sty_src_len,
                                maxlen=tf.shape(features["context_sty_src"])[1],
                                dtype=tf.float32)
    ctx_sty_tgt_mask = tf.sequence_mask(ctx_sty_tgt_len,
                                maxlen=tf.shape(features["context_sty_tgt"])[1],
                                dtype=tf.float32)

    ctx_lan_src_mask = tf.sequence_mask(ctx_lan_src_len,
                                maxlen=tf.shape(features["context_lan_src"])[1],
                                dtype=tf.float32)
    ctx_lan_tgt_mask = tf.sequence_mask(ctx_lan_tgt_len,
                                maxlen=tf.shape(features["context_lan_tgt"])[1],
                                dtype=tf.float32)
    sample_mask = tf.sequence_mask(sample_len,
                                maxlen=tf.shape(features["sample"])[1],
                                dtype=dtype or tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)
    #emotion_inputs = tf.gather(src_embedding, emotion)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    with tf.variable_scope("turn_position_embedding"):
        pos_emb = tf.get_variable("turn_pos_embedding", [len(params.vocabulary["position"]), hidden_size], initializer=tf.contrib.layers.xavier_initializer())

    inputs = inputs * tf.expand_dims(src_mask, -1) #src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)
    #segment embeddings
    if params.segment_embeddings:
        seg_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_src"])
        encoder_input += seg_pos_emb

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)
    emo_inputs = dia_mask
    #top_mask = tf.expand_dims(dia_mask, -1)
    encoder_output, first_layer_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    ## context
    # ctx_seq: [batch, max_ctx_length]
    print("building context graph")
    if params.context_representation == "self_attention":
        print('use self attention')
        # dialogue src context
        get_first_layer = True
        dia_mask = None
        turn_dia_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_src"])
        
        ctx_inputs = tf.gather(src_embedding, ctx_dia_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_dia_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_dia_src_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_dia_src_mask, "masking")
#        context_dia_src = transformer_context(context_input, ctx_attn_bias, params)
        context_dia_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)
        
        context_dia_src = first_layer_output

        # dialogue tgt context
        turn_dia_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_dia_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_dia_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_dia_tgt_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_dia_tgt_mask, "masking")
#        context_dia_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_dia_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # style src context
        turn_sty_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_sty_src"])
        ctx_inputs = tf.gather(src_embedding, ctx_sty_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_sty_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_sty_src_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_sty_src_mask, "masking")
#        context_sty_src = transformer_context(context_input, ctx_attn_bias, params)
        context_sty_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # style tgt context
        turn_sty_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_sty_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_sty_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_sty_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_sty_tgt_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_sty_tgt_mask, "masking")
#        context_sty_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_sty_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # language src context
        turn_lan_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_lan_src"])
        ctx_inputs = tf.gather(src_embedding, ctx_lan_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_lan_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_lan_src_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_lan_src_mask, "masking")
#        context_lan_src = transformer_context(context_input, ctx_attn_bias, params)
        context_lan_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # language tgt context
        turn_lan_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_lan_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_lan_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_lan_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_lan_tgt_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_lan_tgt_mask, "masking")
#        context_lan_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_lan_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)
#        context_output = transformer_encoder(context_input, ctx_attn_bias, params)

        # sample
        sa_inputs = tf.gather(tgt_embedding, sample_seq) * (hidden_size ** 0.5)
        sa_inputs = sa_inputs  * tf.expand_dims(sample_mask, -1)
        sa_input = tf.nn.bias_add(sa_inputs, bias)
        sa_input  = layers.attention.add_timing_signal(sa_input)
        sa_input = tf.nn.dropout(sa_input, 1.0 - params.embed_dropout)
        sa_attn_bias = layers.attention.attention_bias(sample_mask, "masking")

        sa_tgt, _ = transformer_encoder(sa_input, sa_attn_bias, params, dia_mask, get_first_layer)

    elif params.context_representation == "embedding":
        print('use embedding')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)
        context_input = tf.nn.bias_add(ctx_inputs, bias)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
        context_output = context_input
    elif params.context_representation == "bilstm":
        print('use bilstm')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)
        context_input = tf.nn.bias_add(ctx_inputs, bias)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
        context_output = birnn(context_input, ctx_len, params)

    return encoder_output, emo_inputs, context_dia_src, context_dia_tgt, context_sty_src, context_sty_tgt, context_lan_src, context_lan_tgt, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_sty_src_pos_emb, turn_sty_tgt_pos_emb, turn_lan_src_pos_emb, turn_lan_tgt_pos_emb, first_layer_output, sa_tgt


def decoding_graph(features, state, mode, params):
    is_training = True
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0
        is_training = False

    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    sample_len = features["sample_length"]    #
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    ctx_dia_src_len = features["context_dia_src_length"]
    ctx_dia_tgt_len = features["context_dia_tgt_length"]
    ctx_lan_src_len = features["context_lan_src_length"]
    ctx_lan_tgt_len = features["context_lan_tgt_length"]

    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    ctx_dia_src_mask = tf.sequence_mask(ctx_dia_src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=tf.float32)
    ctx_dia_tgt_mask = tf.sequence_mask(ctx_dia_tgt_len,
                                maxlen=tf.shape(features["context_dia_tgt"])[1],
                                dtype=tf.float32)

    ctx_lan_src_mask = tf.sequence_mask(ctx_lan_src_len,
                                maxlen=tf.shape(features["context_lan_src"])[1],
                                dtype=tf.float32)
    ctx_lan_tgt_mask = tf.sequence_mask(ctx_lan_tgt_len,
                                maxlen=tf.shape(features["context_lan_tgt"])[1],
                                dtype=tf.float32)

    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)
    sample_mask = tf.sequence_mask(sample_len,
                                maxlen=tf.shape(features["sample"])[1],
                                dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("target_embedding",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer, trainable=False)

    if params.use_mrg:
        mrg_weights = tf.get_variable("mrg_softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer, trainable=True)

    if params.use_crg:
        crg_weights = tf.get_variable("crg_softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer, trainable=True)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)

    ctx_lan_tgt_attn_bias = layers.attention.attention_bias(ctx_lan_tgt_mask, "masking", dtype=dtype)
    ctx_lan_src_attn_bias = layers.attention.attention_bias(ctx_lan_src_mask, "masking", dtype=dtype)

    tgt_dec_attn_bias = layers.attention.attention_bias(tgt_mask,
                                                    "tgt_masking", dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

    tgt_decoder_input = tf.nn.bias_add(targets, bias)

    if params.position_info_type == 'absolute':
        decoder_input = layers.attention.add_timing_signal(decoder_input)
        tgt_decoder_input = layers.attention.add_timing_signal(tgt_decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)
        tgt_decoder_input = tf.nn.dropout(tgt_decoder_input, keep_prob)

    encoder_output = state["encoder"]
    tgt_decoder_output, first_layer_output_dec = transformer_encoder(tgt_decoder_input, tgt_dec_attn_bias, params)

    emo_inputs = state["emotion"]
    turn_dia_src_pos_emb = state["position_dia_src"]
    turn_dia_tgt_pos_emb = state["position_dia_tgt"]
    turn_sty_src_pos_emb = state["position_sty_src"]
    turn_sty_tgt_pos_emb = state["position_sty_tgt"]
    turn_lan_src_pos_emb = state["position_lan_src"]
    turn_lan_tgt_pos_emb = state["position_lan_tgt"]

    context_dia_src_output = state["context_dia_src"]
    context_dia_tgt_output = state["context_dia_tgt"]
    context_sty_src_output = state["context_sty_src"]
    context_sty_tgt_output = state["context_sty_tgt"]
    context_lan_src_output = state["context_lan_src"]
    context_lan_tgt_output = state["context_lan_tgt"]
    first_layer_output = state["first_layer_output"]
    sample_output = state["sample"]

    context_dia_src = context_dia_src_output[:,0,:]
#    context_dia_src = first_layer_output[:,0,:]
    context_dia_tgt = context_dia_tgt_output[:,0,:]
    context_speaker_src = context_sty_src_output[:,0,:]
    context_speaker_tgt = context_sty_tgt_output[:,0,:]
    context_lan_src = context_lan_src_output[:,0,:]
    context_lan_tgt = context_lan_tgt_output[:,0,:]
    
    s_mask = tf.expand_dims(src_mask, -1)
    t_mask = tf.expand_dims(tgt_mask, -1)

    sample_mask = tf.expand_dims(sample_mask, -1)
    sample_rep = tf.reduce_sum(sample_output * sample_mask, -2) / tf.reduce_sum(sample_mask, -2)

    if mode != "infer":
        # for MT
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)

        if params.use_mrg: # monolingual response generation task
            mrg_decoder_output = transformer_decoder(decoder_input, context_lan_tgt_output,
                                                 dec_attn_bias, ctx_lan_tgt_attn_bias,
                                                 params)
        if params.use_crg: # cross-lingual response generation task
            crg_decoder_output = transformer_decoder(decoder_input, context_lan_src_output,
                                                 dec_attn_bias, ctx_lan_src_attn_bias,
                                                 params) 
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)

        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state, "emotion": emo_inputs, "context_dia_src": context_dia_src_output, "context_dia_tgt": context_dia_tgt_output, "context_sty_src": context_sty_src_output, "context_sty_tgt": context_sty_tgt_output, "context_lan_src": context_lan_src_output, "context_lan_tgt": context_lan_tgt_output, "position_dia_src": turn_dia_src_pos_emb, "position_dia_tgt": turn_dia_tgt_pos_emb, "position_sty_src": turn_sty_src_pos_emb, "position_sty_tgt": turn_sty_tgt_pos_emb, "position_lan_src": turn_lan_src_pos_emb, "position_lan_tgt": turn_lan_tgt_pos_emb, "first_layer_output": first_layer_output, "sample": sample_output}
    # for MT
    print(decoder_output, weights)
    sp_loss = 0.0
    coh_loss = 0.0
    if params.use_speaker:
        reference = tf.reduce_sum(tgt_decoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        speaker_weight = tf.get_variable("speaker_weights",
                                        [2, hidden_size * 2],
                                        initializer=initializer, trainable=True)
        binary_1 = tf.matmul(tf.concat([reference, context_speaker_src], -1), speaker_weight, False, True)
        binary_0 = tf.matmul(tf.concat([reference, context_speaker_tgt], -1), speaker_weight, False, True)
        sp1_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_1, labels=tf.ones([tf.shape(reference)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        sp2_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_0, labels=tf.zeros([tf.shape(reference)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        sp_loss = tf.reduce_mean(sp1_ce) + tf.reduce_mean(sp2_ce)
    if params.use_coherence:
        reference = tf.reduce_sum(tgt_decoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        coherence_weight = tf.get_variable("coherence_weights",
                                        [2, hidden_size * 2],
                                        initializer=initializer, trainable=True)
        tf.random.shuffle(sample_rep)
        binary_1 = tf.matmul(tf.concat([context_dia_tgt, reference], -1), coherence_weight, False, True)
        binary_0 = tf.matmul(tf.concat([context_dia_tgt, sample_rep], -1), coherence_weight, False, True)

        coh1_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_1, labels=tf.ones([tf.shape(reference)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        coh2_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_0, labels=tf.zeros([tf.shape(reference)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        coh_loss = tf.reduce_mean(coh1_ce) + tf.reduce_mean(coh2_ce)
#        code.interact(local=locals())
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
#    print(decoder_output)
    logits = tf.matmul(decoder_output, weights, False, True)
#    print(logits)
    labels = features["target"]
    #code.interact(local=locals())
    # label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)
    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if params.use_mrg:
        mrg_decoder_output = tf.reshape(mrg_decoder_output, [-1, hidden_size])
        mrg_logits = tf.matmul(mrg_decoder_output, mrg_weights, False, True)
        mrg_ce = losses.smoothed_softmax_cross_entropy_with_logits(
            logits=mrg_logits,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        mrg_ce = tf.reshape(mrg_ce, tf.shape(tgt_seq))

    if params.use_crg:
        crg_decoder_output = tf.reshape(crg_decoder_output, [-1, hidden_size])
        crg_logits = tf.matmul(crg_decoder_output, crg_weights, False, True)
        crg_ce = losses.smoothed_softmax_cross_entropy_with_logits(
            logits=crg_logits,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        crg_ce = tf.reshape(crg_ce, tf.shape(tgt_seq))

    if mode == "eval":
        loss = -tf.reduce_sum(ce * tgt_mask, axis=1)
        if params.use_mrg:
            loss += -tf.reduce_sum(mrg_ce * tgt_mask, axis=1)
        if params.use_crg:
            loss += -tf.reduce_sum(crg_ce * tgt_mask, axis=1)
        return loss #-tf.reduce_sum(ce * tgt_mask, axis=1)

    ce_loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    mrg_ce_loss = 0.0
    crg_ce_loss = 0.0
    if params.use_mrg:
        mrg_ce_loss = tf.reduce_sum(mrg_ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    if params.use_crg:
        crg_ce_loss = tf.reduce_sum(crg_ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    kl_loss = mrg_ce_loss

    avg_bow_loss = crg_ce_loss

    return ce_loss, kl_loss, avg_bow_loss, sp_loss, coh_loss


def model_graph(features, mode, params):
    encoder_output, emo_inputs, context_dia_src, context_dia_tgt, context_sty_src, context_sty_tgt, context_lan_src, context_lan_tgt, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_sty_src_pos_emb, turn_sty_tgt_pos_emb, turn_lan_src_pos_emb, turn_lan_tgt_pos_emb,first_layer_output, sample_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output,
        "emotion": emo_inputs,
        "context_dia_src": context_dia_src,
        "context_dia_tgt": context_dia_tgt,
        "context_sty_src": context_sty_src,
        "context_sty_tgt": context_sty_tgt,
        "context_lan_src": context_lan_src,
        "context_lan_tgt": context_lan_tgt,
        "position_dia_src": turn_dia_src_pos_emb,
        "position_dia_tgt": turn_dia_tgt_pos_emb,
        "position_sty_src": turn_sty_src_pos_emb,
        "position_sty_tgt": turn_sty_tgt_pos_emb,
        "position_lan_src": turn_lan_src_pos_emb,
        "position_lan_tgt": turn_lan_tgt_pos_emb,
        "first_layer_output": first_layer_output,
        "sample": sample_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
                return loss#, kl_loss, bow_loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score, _ = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output, emo_inputs, context_dia_src, context_dia_tgt, context_sty_src, context_sty_tgt, context_lan_src, context_lan_tgt, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_sty_src_pos_emb, turn_sty_tgt_pos_emb, turn_lan_src_pos_emb, turn_lan_tgt_pos_emb, first_layer_output, sample_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "emotion": emo_inputs,
                    "context_dia_src": context_dia_src,
                    "context_dia_tgt": context_dia_tgt,
                    "context_sty_src": context_sty_src,
                    "context_sty_tgt": context_sty_tgt,
                    "context_lan_src": context_lan_src,
                    "context_lan_tgt": context_lan_tgt,
                    "position_dia_src": turn_dia_src_pos_emb,
                    "position_dia_tgt": turn_dia_tgt_pos_emb,
                    "position_sty_src": turn_sty_src_pos_emb,
                    "position_sty_tgt": turn_sty_tgt_pos_emb,
                    "position_lan_src": turn_lan_src_pos_emb,
                    "position_lan_tgt": turn_lan_tgt_pos_emb,
                    "first_layer_output": first_layer_output,
                    "sample": sample_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.attention_key_channels or params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.attention_value_channels or params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            num_units=512,
            use_bowloss=False,
            use_srcctx=True,
            mrg_alpha=0.0,
            crg_alpha=0.0,
            sp_alpha=0.0,
            coh_alpha=0.0,
            use_dialog_latent=False,
            use_language_latent=False,
            use_mtstyle_latent=False,
            use_mrg=False,
            use_crg=False,
            use_speaker=False,
            use_coherence=False,
            use_emovec=True,
            segment_embeddings=False,
            hidden_size=512,
            latent_dim=32,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.5,
            relu_dropout=0.0,
            embed_dropout=0.1,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            context_representation="self_attention",
            num_context_layers=1,
            bottom_block=1,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # "absolute" or "relative"
            position_info_type="relative",
            # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            max_relative_dis=16
        )

        return params
