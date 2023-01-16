
# По материалам https://www.tensorflow.org/text/tutorials/transformer

import numpy as np
import tensorflow as tf
import os

from mlm_shared import MAX_LEN
from mlm_shared import TOTAL_DIMS
from embeddings_reader import EmbeddingsReader

# for-debug
#MAX_LEN=4



# positional encoding funcs
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


# генератор позиционных эмбеддингов размерности d_model для входа длиной в position
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# masking
def create_padding_mask(seq):
    SP_PAD_ID = 3  # PAD кодируется индексом 3 в нашей sentencepiece-модели
    seq = tf.cast(tf.math.equal(seq, SP_PAD_ID), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# attention-block
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

class MultiHeadAttention_tr(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention_tr, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')
        self.dense = tf.keras.layers.Dense(d_model, name='dns')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

class MultiHeadAttention_tr2(tf.keras.layers.Layer):
    def __init__(self, d_model, enc_dims, **kwargs):
        super(MultiHeadAttention_tr2, self).__init__(**kwargs)
        self.enc_dims = enc_dims
        self.d_model = d_model
        self.wq = tf.keras.layers.Dense(enc_dims, name='wq')
        self.wk = tf.keras.layers.Dense(enc_dims, name='wk')
        self.wv = tf.keras.layers.Dense(enc_dims, name='wv')
        self.dense = tf.keras.layers.Dense(d_model, name='dns')

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, enc_dims)
        k = self.wk(k)  # (batch_size, seq_len, enc_dims)
        v = self.wv(v)  # (batch_size, seq_len, enc_dims)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, d_model)
        return output


# decoder layer
class DecoderLayer_tr(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(DecoderLayer_tr, self).__init__(**kwargs)
        
        self.mha1 = MultiHeadAttention_tr(d_model=d_model, num_heads=num_heads, name='mha_1')
        self.mha2 = MultiHeadAttention_tr2(d_model=d_model, enc_dims=TOTAL_DIMS+30, name='mha_2')
        
        self.ffn_d1 = tf.keras.layers.Dense(dff, activation='relu', name='ffn_dff')  # (batch_size, seq_len, dff)
        self.ffn_d2 = tf.keras.layers.Dense(d_model, name='ffn_o')  # (batch_size, seq_len, d_model)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln_1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln_2')
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln_3')
        
        self.dropout1 = tf.keras.layers.Dropout(rate, name='drp_1')
        self.dropout2 = tf.keras.layers.Dropout(rate, name='drp_2')
        self.dropout3 = tf.keras.layers.Dropout(rate, name='drp_3')

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        
        attn1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn_d1(out2)
        ffn_output = self.ffn_d2(ffn_output) # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        return out3


# decoder
class Decoder_tr(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1, **kwargs):
        super(Decoder_tr, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, name='dec_emb')
        self.pos_encoding = positional_encoding(2*MAX_LEN, d_model)
        
        self.dec_layers = [
            DecoderLayer_tr(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, name=f'decl_{i}')
            for i in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(rate, name='drp_dec_tr')

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x


# подобие трансформера с нашим энкодером
class Transformer_tr(tf.keras.Model):
    def __init__(self, lle, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.lle = lle
        #self.mode = mode
        self.train_transfomer = True
        self.pos_encoding = positional_encoding(MAX_LEN, 30)
        self.decoder = Decoder_tr( num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                   target_vocab_size=target_vocab_size, rate=rate, trainable=self.train_transfomer, name='decoder' )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, trainable=self.train_transfomer, name='final_dense')
        SP_PAD_ID = 3  # PAD кодируется индексом 3 в нашей sentencepiece-модели
        self.pad_alt = tf.zeros([1,1,target_vocab_size], dtype=tf.float32)
        self.pad_alt = tf.tensor_scatter_nd_update(self.pad_alt, [[0,0,SP_PAD_ID]], [1.0])

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        batch_size = tf.shape(inp)[0]
        #seq_len = tf.shape(inp)[1]

        input_padding_mask = tf.cast( tf.math.equal(inp[:,:,0], EmbeddingsReader.PAD_IDX), dtype=tf.float32 ) # (batch_size, inp_seq_len)
        input_padding_mask = input_padding_mask[:, tf.newaxis, :] # (batch_size, 1, inp_seq_len)
        look_ahead_mask, dec_target_padding_mask = self.create_masks(tar)
#         print(input_padding_mask)
#         print(look_ahead_mask)
#         print(dec_target_padding_mask)
        
        enc_output = self.lle(inp)
        enc_output_p = tf.concat([enc_output, tf.tile(self.pos_encoding, [batch_size,1,1])], -1)  # (batch_size, inp_seq_len, TOTAL_DIMS+30)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(tar, enc_output_p, self.train_transfomer, look_ahead_mask, input_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        # masking pad, for loss simplicity
        if training:
            SP_PAD_ID = 3  # PAD кодируется индексом 3 в нашей sentencepiece-модели
            pm = tf.math.equal(tar, SP_PAD_ID) # (batch_size, tar_seq_len)
            pm = tf.concat([pm[:, 1:], tf.tile(tf.constant([[True]]),[batch_size,1])], -1)
            pm = pm[:, :, tf.newaxis]
            fo = tf.where(pm, self.pad_alt, final_output)
            return fo
        else:
            return final_output

    def create_masks(self, tar):
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return look_ahead_mask, dec_target_padding_mask

    def save_dec_weights(self, dir = 'dec_model'):
        own_layers = [self.decoder, self.final_layer]
        names = [[weight.name for weight in layer.weights] for layer in own_layers]
        wlist = []
        for layer in own_layers:
            wlist.append(layer.get_weights())
        for i, layer in enumerate(own_layers):
            assert len(names[i]) == len(wlist[i])
            wdict = {}
            for n, w in zip(names[i], wlist[i]):
                wdict[n] = w
            filename = os.path.join(dir, layer.name)
            np.savez(filename, **wdict)

    def load_dec_weights(self, dir = 'dec_model'):
        own_layers = [self.decoder, self.final_layer]
        names = [[weight.name for weight in layer.weights] for layer in own_layers]
        for i, layer in enumerate(own_layers):
            wdict = np.load(os.path.join(dir, layer.name) +'.npz')
            wlist = [wdict[n] for n in names[i]]
            layer.set_weights(wlist)

#     def summary(self):
#         x = tf.keras.layers.Input(shape=(MAX_LEN, 2,))
#         x_sp = tf.keras.layers.Input(shape=(2*MAX_LEN,))
#         model = tf.keras.Model(inputs=[x,x_sp], outputs=self.call([x,x_sp],False))
#         return model.summary()





# # код для отладки и самодиагностики
# from low_level_encoder import LowLevelEncoder
# input_data = [
#              [
#                [8,4],
#                [6,9],
#                [2,3],
#                [0,0],
#              ],
#              [
#                [4,12],
#                [30,2],
#                [0,0],
#                [0,0],
#              ]
#              ]
#                                                 
#                                                 
# tinpd = tf.convert_to_tensor(input_data, dtype=tf.int32)
# print()
# print('Входные данные:')
# print(tinpd.numpy())
#  
# target_data = [
#                 [1,8,0,6,5,4,2,3],
#                 [1,12,11,10,2,3,3,3]
#               ]
#  
# ttard = tf.convert_to_tensor(target_data, dtype=tf.int32)
# print()
# print('Ожидаемые выходные данные:')
# print(ttard.numpy())
#  
# vm = EmbeddingsReader("vectors.bin")
# common_lle_layer = LowLevelEncoder(60, 40, 20, vm.stems_count, vm.sfx_count, ctxer_training=True, balance_training=False, name='lle')
# ed_model = Transformer_tr(common_lle_layer, 2, 256, 8, 512, 100)
# #ed_model.build(input_shape = [(None, MAX_LEN, 2), (None, 2*MAX_LEN)])
# #ed_model.summary()
#  
# # lam, dpm = ed_model.create_masks(ttard)
# # print(lam)
# # print(dpm)
#  
# r = ed_model([tinpd, ttard], True)
# print()
# print('Результат (shape):')
# print(r.shape)




