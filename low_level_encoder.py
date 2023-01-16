
from embeddings_reader import EmbeddingsReader

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np



# Вспомогательный класс контекстуализатора
class CtxNetHelper(tf.keras.layers.Layer):
    def __init__(self, output_dims, input_drop_rate, lstm_units, l2_rate, algo='rctx_to_left_lstm+theme', grad_theme=True, name='CtxNetHelper', **kwargs):
        super(CtxNetHelper, self).__init__(name=name, **kwargs)
        self.algo = algo
        self.grad_theme = grad_theme
        self.drp_li = tf.keras.layers.Dropout(input_drop_rate)
        self.drp_ri = tf.keras.layers.Dropout(input_drop_rate)
        self.lctx_lstm = tf.keras.layers.LSTM(lstm_units)
        self.rctx_lstm = tf.keras.layers.LSTM(lstm_units)
        self.drp_pf = tf.keras.layers.Dropout(0.05)
        self.dns_final = tf.keras.layers.Dense(output_dims)
#        self.dns_final = tf.keras.layers.Dense(output_dims, kernel_regularizer=tf.keras.regularizers.l2(l2_rate), bias_regularizer=tf.keras.regularizers.l2(l2_rate))
    def build(self, input_shape):
        self.input_dims = input_shape[-1]
        if self.algo == 'rctx_to_left_lstm+theme':
            self.rescale_dns = tf.keras.layers.Dense(self.input_dims)
        elif self.algo == 'gramm_algo':
            self.drp_th = tf.keras.layers.Dropout(0.75)
    def call(self, lctx, rctx, theme, training):
        lctx = self.drp_li(lctx, training=training)
        rctx = self.drp_ri(rctx, training=training)
        if self.algo == 'rctx_to_left_lstm+theme':
            r_inf = self.rctx_lstm( rctx, training=training )
            r_inf = self.rescale_dns(r_inf)
            r_inf = tf.expand_dims(r_inf, -2)
            lctx = tf.concat([lctx, r_inf], -2)
            x = self.lctx_lstm( lctx, training=training )
            if theme is not None:
                if not self.grad_theme:
                    theme = tf.stop_gradient(theme)
                x = tf.concat([x, theme], -1)
            x = self.drp_pf(x, training=training)
            return self.dns_final(x)
        if self.algo == 'gramm_algo':
            if not self.grad_theme:
                theme = tf.stop_gradient(theme)
            t_inf = self.drp_th(theme, training=training) # интенсивно маскируем категориальную информацию
            tsh = tf.shape(theme)
            t_inf = tf.concat([t_inf, tf.zeros([tsh[0], tf.shape(lctx)[-1] - tsh[1]], tf.float32)], -1) # дописываем нулей до размерности входных векторов
            t_inf = tf.expand_dims(t_inf, -2)
            lctx = tf.concat([lctx, t_inf], -2)
            rctx = tf.concat([rctx, t_inf], -2)
            l_inf = self.lctx_lstm( lctx, training=training )
            r_inf = self.rctx_lstm( rctx, training=training )
            x = tf.concat([l_inf, r_inf], -1)
            x = self.drp_pf(x, training=training)
            return self.dns_final(x)
            
            
    def get_name2sublayer_dict(self):
        if self.algo == 'rctx_to_left_lstm+theme':
            return {
                    'lctx_lstm': self.lctx_lstm,
                    'rctx_lstm': self.rctx_lstm,
                    'dns_r': self.rescale_dns,
                    'dns_f': self.dns_final,
                   }
        elif self.algo == 'gramm_algo':
            return {
                    'lctx_lstm': self.lctx_lstm,
                    'rctx_lstm': self.rctx_lstm,
                    'dns_f': self.dns_final,
                   }
            
    def save_own_weights(self, filename):
        n2sl = self.get_name2sublayer_dict()
        wdict = {}
        for k,v in n2sl.items():
            w = v.get_weights()
            for i in range(len(w)):
                wdict[k+'_w{}'.format(i)] = w[i]
        np.savez(filename, **wdict)
    def load_own_weights(self, filename):
        wdict = np.load(filename+'.npz')
        n2sl = self.get_name2sublayer_dict()
        for k,v in n2sl.items():
            w_list = []
            for i in range(len(v.get_weights())):
                w_list.append( wdict[k+'_w{}'.format(i)] )
            v.set_weights( w_list )
  
# Слой контекстуализации
class Ctxer(tf.keras.layers.Layer):
    def __init__(self, cat_dims, ass_dims, gra_dims, stems_count, sfx_count, name="Ctxer", **kwargs):
        super(Ctxer, self).__init__(name=name, **kwargs)
        self.cat_dims = cat_dims
        self.ass_dims = ass_dims
        self.stems_dims = cat_dims + ass_dims
        self.gra_dims = gra_dims
        self.sfx_dims = gra_dims
        self.dims = cat_dims + ass_dims + gra_dims
        self.stems_count = stems_count
        self.sfx_count = sfx_count

    def build(self, input_shape):
        # контекстуализирующие слои
        self.cat_ctxer = CtxNetHelper(self.cat_dims, 0.15, 768, 0.01, grad_theme=False, name = 'cat_ctxer') # 500
        #self.cat_ctxer = CtxNetHelper(self.cat_dims, 0.15, 500, 0.01, name = 'cat_ctxer')
        self.ass_ctxer = CtxNetHelper(self.ass_dims, 0.15, 256, 0.01, name = 'ass_ctxer')  # 200
        self.gra_ctxer = CtxNetHelper(self.gra_dims, 0.2, 256, 0.01, grad_theme=False, algo = 'gramm_algo', name = 'gra_ctxer') # 128
        #self.gra_ctxer = CtxNetHelper(self.gra_dims, 0.1, 100, 0.001, grad_theme=False, name = 'gra_ctxer')
        # подсеть для вычисления коэф. внимания, используемых при построении тематического вектора
        self.make_att_subnet()
        #self.make_att_subnet_for_cat()
        super(Ctxer, self).build(input_shape)
    
    def make_att_subnet(self):
        self.att_drp_1 = tf.keras.layers.Dropout(0.1)
        self.att_dns_1 = tf.keras.layers.Dense(256, activation='tanh') # 160
        self.att_drp_2 = tf.keras.layers.Dropout(0.05)
        self.att_dns_2 = tf.keras.layers.Dense(128, activation='tanh') # 80
        self.att_drp_3 = tf.keras.layers.Dropout(0.05)
#        self.att_dns_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001)) 
        self.att_dns_final = tf.keras.layers.Dense(self.ass_dims, activation='sigmoid') 
    def apply_att_subnet(self, input, training):
        x = self.att_drp_1(input, training=training)
        x = self.att_dns_1(x)
        x = self.att_drp_2(x, training=training)
        x = self.att_dns_2(x)
        x = self.att_drp_3(x, training=training)
        return self.att_dns_final(x)
        
    def get_name2sublayer_dict(self):
        return {
                'cat_ctxer': self.cat_ctxer,
                'ass_ctxer': self.ass_ctxer,
                'gra_ctxer': self.gra_ctxer
               }
    def get_name2ksublayer_dict(self):
        return {
                'theme_att_dns_1': self.att_dns_1,
                'theme_att_dns_2': self.att_dns_2,
                'theme_att_dns_f': self.att_dns_final
               }
    def save_own_weights(self, dir = 'lle_weights'):
        n2sl = self.get_name2sublayer_dict()
        for k,v in n2sl.items():
            v.save_own_weights( os.path.join(dir, k) )        
        n2ksl = self.get_name2ksublayer_dict()
        wdict = {}
        for k,v in n2ksl.items():
            w = v.get_weights()
            for i in range(len(w)):
                wdict[k+'_w{}'.format(i)] = w[i]
        np.savez(os.path.join(dir, 'lle_theme_att'), **wdict)
    def load_own_weights(self, dir = 'lle_weights'):
        n2sl = self.get_name2sublayer_dict()
        for k,v in n2sl.items():
            v.load_own_weights( os.path.join(dir, k) )        
        wdict = np.load(os.path.join(dir, 'lle_theme_att') +'.npz')
        n2ksl = self.get_name2ksublayer_dict()
        for k,v in n2ksl.items():
            w_list = []
            for i in range(len(v.get_weights())):
                w_list.append( wdict[k+'_w{}'.format(i)] )
            v.set_weights( w_list )

    def call(self, input, training):
        x_joined = input[0]
        line_msk = input[1]

        xj_shape = tf.shape(x_joined)
        batch_size, seq_len = xj_shape[0], xj_shape[1]

        # найдем координаты слов в тензоре
        w_indices = tf.where( line_msk == 1 )
        

        window_size = 5
        
        # преобразуем входную последовательность в наборы левых и правых контекстов (длиной window_size) для каждого слова
        padding_block = tf.fill([batch_size, window_size, self.dims], EmbeddingsReader.PAD_FILLER)
        padded_input = tf.concat([padding_block, x_joined, padding_block], -2) 
        padded_input = tf.expand_dims(padded_input, -2)
        sequences = tf.image.extract_patches(images=padded_input, sizes=[1, window_size, 1, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='VALID')
        l_sequences = tf.reshape( sequences[:, :-(window_size+1), :, :], [batch_size, seq_len, window_size*self.dims] )
        r_sequences = tf.reshape( sequences[:, (window_size+1):, :, :], [batch_size, seq_len, window_size*self.dims] )
        l_sequences = tf.gather_nd(l_sequences, w_indices)
        r_sequences = tf.gather_nd(r_sequences, w_indices)
        l_sequences = tf.reshape(l_sequences, [-1, window_size, self.dims])
        r_sequences = tf.reshape(r_sequences, [-1, window_size, self.dims])
        r_sequences = tf.reverse(r_sequences, [-2])  

        # дополним контекст общим тематическим вектором
        ctxs_counts = tf.reduce_sum(line_msk, -1)                # кол-во слов в каждом предложении
        x_selected = tf.gather_nd(x_joined, w_indices)           # список всех значимых слов (без pad)
        att_w = self.apply_att_subnet(x_selected, training)      # для каждого слова определяем его значимость в качестве источника тематической информации
        t = x_selected[:, self.cat_dims:self.stems_dims]         # теперь возьмем ассоц.части всех значимых слов
        t = t * att_w                                            # взвесим их
        t = tf.RaggedTensor.from_row_lengths(t, ctxs_counts)     # сгруппируем в предложения
        t = tf.reduce_mean(t, axis=-2)                           # найдем тематический вектор для предложения путем усреднения взвешенных ассоц.частей всех слов
        t = tf.repeat(t, ctxs_counts, -2)                        # растиражируем по числу слов в предложении

        # вычисляем контекстуализации для каждого вида связности
        cat_x = self.cat_ctxer(l_sequences, r_sequences, t, training)
        cat_x = tf.scatter_nd(w_indices, cat_x, [batch_size, seq_len, self.cat_dims])
        
        ass_x = self.ass_ctxer(l_sequences, r_sequences, t, training)
        ass_x = tf.scatter_nd(w_indices, ass_x, [batch_size, seq_len, self.ass_dims])

        gra_x = self.gra_ctxer(l_sequences, r_sequences, x_selected[:, :self.cat_dims], training) # в качестве темы передаем категориальную компоненту текущего слова, которая попадет под сильный dropout
#        gra_x = self.gra_ctxer(l_sequences, r_sequences, None, training) #игнорируем тему
        gra_x = tf.scatter_nd(w_indices, gra_x, [batch_size, seq_len, self.gra_dims])

#        cat_x = tf.zeros([tf.reduce_sum(ctxs_counts), self.cat_dims], tf.float32)
#        ass_x = tf.zeros([tf.reduce_sum(ctxs_counts), self.ass_dims], tf.float32)
#        gra_x = tf.zeros([tf.reduce_sum(ctxs_counts), self.gra_dims], tf.float32)
        
        return [cat_x, ass_x, gra_x]
        



# Модель "низкоуровневого энкодера"
# Включает в себя слои статических эбмеддингов, низкоуровневой контекстуализации и смешивания
class LowLevelEncoder(tf.keras.Model):
    def __init__(self, cat_dims, ass_dims, gra_dims, stems_count, sfx_count, ctxer_training=False, balance_training=False, name="LowLevelEncoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cat_dims, self.ass_dims, self.gra_dims = cat_dims, ass_dims, gra_dims
        self.stems_dims, self.sfx_dims = cat_dims + ass_dims, gra_dims
        self.dims = cat_dims + ass_dims + gra_dims
        self.stems_count = stems_count
        self.sfx_count = sfx_count
        self.bypass = False   # режим отключения контекстуализатора: на выход пробрасываются статические представления
        self.ctxer_training = ctxer_training   # режим обучения контекстуализатора (выключает смешивание со статическими эмбеддингами)
        self.balance_training = balance_training   # режим обучения балансов
        self.stems_emb_layer = tf.keras.layers.Embedding(self.stems_count, self.stems_dims, trainable=False, name='stems_embeddings')
        self.sfx_emb_layer   = tf.keras.layers.Embedding(self.sfx_count, self.sfx_dims, trainable=False, name='sfx_embeddings')
        self.cat_balance     = tf.keras.layers.Embedding(self.stems_count, 1, 
                                                         embeddings_initializer=tf.keras.initializers.Constant(0.9), 
                                                         #embeddings_constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99), 
                                                         trainable=balance_training, name='cat_balances')
        self.ass_balance     = tf.keras.layers.Embedding(self.stems_count, 1, 
                                                         embeddings_initializer=tf.keras.initializers.Constant(0.9), 
                                                         #embeddings_constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99), 
                                                         trainable=balance_training, name='ass_balances')
        self.gra_balance     = tf.keras.layers.Embedding(self.sfx_count, 1, 
                                                         embeddings_initializer=tf.keras.initializers.Constant(0.9), 
                                                         #embeddings_constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99), 
                                                         trainable=balance_training, name='gra_balances')
        self.ctxer = Ctxer(cat_dims, ass_dims, gra_dims, stems_count, sfx_count, trainable=ctxer_training, name='ctxer')
    
    def load_embeddings(self, vm):
        assert vm.stems_size == self.stems_dims
        assert vm.sfx_size == self.sfx_dims
        self.stems_emb_layer.set_weights([vm.stems_embs])
        self.sfx_emb_layer.set_weights([vm.sfx_embs])
    def get_balances_dict(self):
        return {
                'cat_balances': self.cat_balance,
                'ass_balances': self.ass_balance,
                'gra_balances': self.gra_balance
                # спец.обработка с clip для балансов... нельзя сюда добавлять другие веса
               }
    def save_own_weights(self, dir = 'lle_weights'):
        self.ctxer.save_own_weights(dir)
        wdict = {}
        bdict = self.get_balances_dict()
        for k,v in bdict.items():
            w = v.get_weights()
            if not w:
                wdict[k] = np.full( (v.input_dim, 1), 0.9 )
            else:
                wdict[k] = np.clip( v.get_weights()[0], 0.01, 0.99 )
        np.savez(os.path.join(dir, 'balances'), **wdict)
    def load_own_weights(self, dir = 'lle_weights'):
        self.ctxer.load_own_weights(dir)
        wdict = np.load(os.path.join(dir, 'balances') +'.npz')
        bdict = self.get_balances_dict()
        for k,v in bdict.items():
            v.set_weights( [wdict[k]] )
        
    def switch_bypass(self, value):
        self.bypass = value
    def switch_ctxer_trainable(self, value):
        self.ctxer.trainable = value
    def switch_balancer_trainable(self, value):
        self.cat_balance.trainable = value
        self.ass_balance.trainable = value
        self.gra_balance.trainable = value
    

    def call(self, input, training):
        
        if self.ctxer_training:
            # в тренировочном режиме в input приходит пара (результат_токенизации, маска)
            input_toks = input[0]
            line_msk = tf.cast( input[1], dtype=tf.int32 )  # (batch, seq_len)
        else:
            # в режиме вывода в input приходит только результат_токенизации
            input_toks = input
            # построим pad-маску
            line_msk = tf.cast( tf.math.not_equal(input_toks[:,:,0], EmbeddingsReader.PAD_IDX), dtype=tf.int32 )  # (batch, seq_len)
        
        input_shape = tf.shape(input_toks)
        batch_size, seq_len = input_shape[0], input_shape[1]
        
        # oov-маскирование входных данных при обучении слоя контекстуализации
        if self.ctxer_training:
            rnd = tf.random.uniform([batch_size, seq_len], minval=0.0, maxval=1.0, dtype=tf.float32)
            rnd = tf.tile( tf.expand_dims(rnd, -1), [1,1,2] )
            input_toks = tf.where(rnd < 0.05, tf.ones_like(input_toks)*EmbeddingsReader.OOV_IDX, input_toks)

        # перекодируем индексы в эмбеддинги
        x_stems = self.stems_emb_layer(input_toks[:,:,0])
        x_sfxs = self.sfx_emb_layer(input_toks[:,:,1])
        x_joined = tf.concat([x_stems, x_sfxs], -1)
        
        if self.bypass:
            return x_joined
        
        cat_x, ass_x, gra_x = self.ctxer([x_joined, line_msk], training=self.ctxer_training)

        # смешивание с исходными статическими эмбеддингами        
        if not self.ctxer_training:
            # получаем текущие коэффициенты баланса
            cat_ratio = self.cat_balance(input_toks[:,:,0])
            ass_ratio = self.ass_balance(input_toks[:,:,0])
            gra_ratio = self.gra_balance(input_toks[:,:,1])
            # обход в связи с ошибкой constraint у слоя embeddings
            if self.balance_training:
                cat_ratio = tf.clip_by_value(cat_ratio, 0.01, 0.99)
                ass_ratio = tf.clip_by_value(ass_ratio, 0.01, 0.99)
                gra_ratio = tf.clip_by_value(gra_ratio, 0.01, 0.99)
            # смешивание со статическими представлениями
            cat_x = tf.reduce_sum( tf.stack([cat_x*cat_ratio, x_joined[:,:,:self.cat_dims]*(1.0-cat_ratio)], -2), -2 )  # weighted average
            ass_x = tf.reduce_sum( tf.stack([ass_x*ass_ratio, x_joined[:,:,self.cat_dims:self.stems_dims]*(1.0-ass_ratio)], -2), -2 )
            gra_x = tf.reduce_sum( tf.stack([gra_x*gra_ratio, x_joined[:,:,self.stems_dims:]*(1.0-gra_ratio)], -2), -2 )
        # собираем единый вектор
        result = tf.concat([cat_x, ass_x, gra_x], -1)
        # маскировка отступов
        result = tf.where( tf.equal(tf.expand_dims(line_msk,-1), 0), tf.ones_like(result)*EmbeddingsReader.PAD_FILLER, result )
        # выдача результата
        return result
        #return result, tf.scatter_nd(w_indices, tf.reshape(att_w, [-1]), [batch_size, seq_len])
        #return result, tf.scatter_nd(w_indices, tf.reshape(tf.reduce_mean(att_w, axis=-1), [-1]), [batch_size, seq_len])



# Заглушка энкодера (просто проброс статических эмбеддингов)
class LowLevelEncoderStub(tf.keras.Model):
    def __init__(self, cat_dims, ass_dims, gra_dims, stems_count, sfx_count, name="LowLevelEncoderStub", **kwargs):
        super().__init__(name=name, **kwargs)
        self.stems_dims, self.sfx_dims = cat_dims + ass_dims, gra_dims
        self.stems_count = stems_count
        self.sfx_count = sfx_count
        self.stems_emb_layer = tf.keras.layers.Embedding(self.stems_count, self.stems_dims, trainable=False, name='stems_embeddings')
        self.sfx_emb_layer   = tf.keras.layers.Embedding(self.sfx_count, self.sfx_dims, trainable=False, name='sfx_embeddings')
    
    def load_embeddings(self, vm):
        assert vm.stems_size == self.stems_dims
        assert vm.sfx_size == self.sfx_dims
        self.stems_emb_layer.set_weights([vm.stems_embs])
        self.sfx_emb_layer.set_weights([vm.sfx_embs])
    def save_own_weights(self, dir = 'lle_weights'):
        pass
    def load_own_weights(self, dir = 'lle_weights'):
        pass
    def setup_balance(self, balance):
        pass

    def call(self, input, training):
        x_stems = self.stems_emb_layer(input[:,:,0])
        x_sfxs = self.sfx_emb_layer(input[:,:,1])
        return tf.concat([x_stems, x_sfxs], -1)
        




# # код для отладки и самодиагностики
# print('Tensorflow version: {}'.format(tf.version.VERSION))
# print()
#  
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
# print()
# print('СОЗДАНИЕ И ИНИЦИАЛИЗАЦИЯ СЛОЯ')                    
# att_layer = LowLevelEncoder(2, 2, 2, 100, 100, ctxer_training=True, balance_training=False)
#                            
# print()
# print('FORWARD PASS')
# loss_fn = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam()
# with tf.GradientTape() as tape:
#     ctx_vecs = att_layer( tinpd, training=True )
#     print('Результат:')
#     print('Вектора контекстов:')
#     print(ctx_vecs.numpy())
#     loss_value = loss_fn([ [[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[10,10,10,10,10,10]],
#                            [[-2,-2,-2,-2,-2,-2],[-3,-3,-3,-3,-3,-3],[10,10,10,10,10,10],[10,10,10,10,10,10]] ], ctx_vecs) # Compute the loss value
#     print('loss tensor')
#     print(loss_value)



