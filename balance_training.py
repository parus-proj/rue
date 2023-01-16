# Скрипт для обучения балансировочных коэффициентов модели

# Использовать ли все GPU на машине
USE_ALL_GPU = True

SP_PAD_ID = 3 # иднтификатор pad в sentencepiece-модели

from mlm_shared import MAX_LEN
from mlm_shared import BATCH_SIZE
from mlm_shared import CACHE_DIR

from mlm_shared import CATEG_DIMS
from mlm_shared import ASSOC_DIMS
from mlm_shared import GRAMM_DIMS

# from mlm_shared import ED_DELAY_STEPS
# from mlm_shared import ED_WARMUP_STEPS
# from mlm_shared import ED_RESTORE_STEPS


from embeddings_reader import EmbeddingsReader

from low_level_encoder import LowLevelEncoder
from low_level_encoder import LowLevelEncoderStub

from mlm_learning_net import Transformer_tr

import sentencepiece as spm

import argparse
import sys
import os
import time
import numpy as np
import tensorflow as tf


# # контроллер скорости обучения для сети Энкодер-Декодер
# class CustomScheduleEncDec(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, model_dims, warmup_steps=2000):
#         super(CustomScheduleEncDec, self).__init__()
#         self.model_dims = tf.cast(model_dims, tf.float32)
#         self.warmup_steps = warmup_steps
#     
#     def __call__(self, step):
#         #tf.print(step)
#         step = tf.cast(step, tf.float32)
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#         return tf.math.rsqrt(self.model_dims) * tf.math.minimum(arg1, arg2)

# temp_learning_rate_schedule = CustomScheduleEncDec(2048)
# import matplotlib.pyplot as plt
# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel('Learning Rate')
# plt.xlabel('Train Step')
# plt.savefig('schedule-lr-ed.pdf')
# import sys
# sys.exit(0)


#  мэппинг записи из файла данных в тензор, аналогичный по структуре тому, что выдает для предложения токенизатор
def record_as_tokenized(x):
    l0 = tf.strings.split(x, sep=tf.constant('/'))
    l1    = tf.strings.split(l0[0], sep=tf.constant(' '))
    l1_sp = tf.strings.split(l0[1], sep=tf.constant(' '))
    l2    = tf.strings.to_number(l1, out_type=tf.int32)
    l2_sp = tf.strings.to_number(l1_sp, out_type=tf.int32)
    l3 = tf.reshape(l2, [-1, 2])
    sp_begin_item = tf.constant([1])
    sp_end_item = tf.constant([2])
    l2_sp = tf.concat([sp_begin_item, l2_sp, sp_end_item], -1)
    return l3, l2_sp

# паддинг записи данных
def record_padding(x, x_sp):
    pad_item = tf.constant([[EmbeddingsReader.PAD_IDX, EmbeddingsReader.PAD_IDX]])
    pad_seq = tf.tile(pad_item, [MAX_LEN-tf.shape(x)[0], 1])
    sp_pad_item = tf.constant([3]) # None in vocab or <pad> in manual: https://github.com/google/sentencepiece
    x_sp = x_sp[:(2*MAX_LEN)]
    sp_pad_seq = tf.repeat(sp_pad_item, [2*MAX_LEN-tf.shape(x_sp)[0]]) 
    return tf.concat([x, pad_seq], 0), tf.concat([x_sp, sp_pad_seq], 0)


 
 

class BalanceTrainingController(object):
    def __init__(self, mode, time, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.time = time
        
        # Создадим объект-стратегию
        if USE_ALL_GPU:
            self.strategy = tf.distribute.MirroredStrategy() # стратегия single host multi gpu
        else:
            self.strategy = tf.distribute.get_strategy() # умолчательная стратегия
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        
        # Загрузим векторную модель (одновременно добавив в нее служебные векторы)
        print('Static model loading')
        self.vm = EmbeddingsReader("vectors.c2v")
        
        # Загрузим sentencepiece-модель (только чтобы узнать размер словаря)
        print('SentencePiece model loading')
        sp = spm.SentencePieceProcessor(model_file='sp_m.model')
        self.target_vocab_size = sp.get_piece_size()
        
        # Создадим датасет для доступа к данным
        print('Prepare dataset')
        self.GLOBAL_BATCH_SIZE = BATCH_SIZE * self.strategy.num_replicas_in_sync
        files_list = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.el') and os.path.isfile(os.path.join(CACHE_DIR, f))]
        self.dataset = tf.data.TextLineDataset(files_list) \
                        .map(record_as_tokenized, num_parallel_calls=tf.data.AUTOTUNE) \
                        .map(record_padding, num_parallel_calls=tf.data.AUTOTUNE) \
                        .batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
                        .prefetch(tf.data.AUTOTUNE)
        if USE_ALL_GPU:
            # донастраиваем датасет под стратегию
            ds_options = tf.data.Options()
            ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            self.dataset = self.dataset.with_options(ds_options)
            self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        
        # Создадим модель
        print('Create model & optimizer')
        with self.strategy.scope():
            decoder_layers = 2
            decoder_dims = 128 # 256
            decoder_heads = 8
            decoder_ff = 256   # 512
            if self.mode == 'prepare':
                # режим "без контекстуализации" (для первичной настройки весов декодера)
                # в качестве энкодера используются статические представления
                self.common_lle_layer = LowLevelEncoderStub(CATEG_DIMS, ASSOC_DIMS, GRAMM_DIMS, self.vm.stems_count, self.vm.sfx_count, name='lle')
                self.ed_model = Transformer_tr(self.common_lle_layer, decoder_layers, decoder_dims, decoder_heads, decoder_ff, self.target_vocab_size, mode=self.mode, name='EDM')
                print("PREPARE MODE")
            elif self.mode == 'balance':
                # режим тренировки баланса (с замороженной контекстуализацией)
                self.common_lle_layer = LowLevelEncoder(CATEG_DIMS, ASSOC_DIMS, GRAMM_DIMS, self.vm.stems_count, self.vm.sfx_count, ctxer_training=False, balance_training=True, name='lle')
                self.ed_model = Transformer_tr(self.common_lle_layer, decoder_layers, decoder_dims, decoder_heads, decoder_ff, self.target_vocab_size, mode=self.mode, name='EDM')
                print("BALANCE MODE")
            else:
                print("UNKNOWN MODE")
            self.ed_model.build(input_shape = [(None, MAX_LEN, 2), (None, 2*MAX_LEN)])
            #ed_lr_schedule = CustomScheduleEncDec(1000, warmup_steps=ED_WARMUP_STEPS)
            #self.ed_optimizer = tf.keras.optimizers.SGD(learning_rate=ed_lr_schedule)
            self.ed_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999999)
            
        # Загружаем веса согласно режиму обучения
        print('Weights loading')
        self.common_lle_layer.load_embeddings(self.vm)
        if self.mode == 'balance':
            self.common_lle_layer.load_own_weights('lle_weights')
            self.ed_model.load_dec_weights('dec_model')
        self.ed_model.summary()
        
        # Создадим loss-объект  
        print('Loss creation')
        self.ed_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)  # from_logits=True, т.к. выход декодера без softmax
        
    def __del__(self):
        ## fix: https://github.com/tensorflow/tensorflow/issues/50487
        #import atexit
        #atexit.register(self.strategy._extended._collective_ops._pool.close) # type: ignore
        pass

    def ed_compute_loss(self, true_vals, predictions, mlm_true_vals=None, mlm_predictions=None):
        per_example_loss = self.ed_loss_object(true_vals, predictions)  # (batch, seq_len) = (batch, 2*MAX_LEN-1)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.GLOBAL_BATCH_SIZE)
       
    @tf.function
    def ed_train_step(self, x, x_sp):
        tar_inp  = x_sp[:, :-1]
        tar_real = x_sp[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self.ed_model([x, tar_inp], training = True)
            loss_value = self.ed_compute_loss(tar_real, predictions)
        grads = tape.gradient(loss_value, self.ed_model.trainable_variables)
        self.ed_optimizer.apply_gradients(zip(grads, self.ed_model.trainable_variables))
        return loss_value
    
    @tf.function
    def ed_distributed_train_step(self, x, x_sp):
        per_replica_losses = self.strategy.run(self.ed_train_step, args=(x, x_sp,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        
    def run(self):
        # цикл обучения
        start_time = time.perf_counter()
        timeout_flag = False
        gstep = -1
        # цикл по эпохам
        for epoch in range(3):
            print('\nStart of epoch {}'.format(epoch))
            start_epoch_time = time.perf_counter()
            ed_epoch_loss, ed_loss_value = 0.0, 0.0
            ed_step = 0
            # цикл по батчам датасета
            for step, (x, x_sp) in enumerate(self.dataset):
                gstep += 1
                
#                 # выход из прогрева
#                 if gstep == 3:
#                     self.ed_optimizer.lr.assign(5e-4)
                    
                # выполняем шаг обучения
                ed_loss_value = self.ed_distributed_train_step(x, x_sp)
                ed_epoch_loss += ed_loss_value
                ed_step += 1
        
                # выводим проткол через каждые N батчей.
                if step % 500 == 0:
                    print('Step No:   {}'.format(step))
                    #ed_opt_current_step = strategy.reduce(tf.distribute.ReduceOp.MEAN, ed_optimizer.iterations, axis=None).numpy()
                    #print('Current ED learning rate:           {:.5f}'.format(ed_lr_schedule(ed_opt_current_step)))
                    print('ED training loss:                   {:.4f}'.format(ed_epoch_loss/ed_step if ed_step > 0 else 0.0))
                    print('ED. training loss for last batch:   {:.4f}'.format(ed_loss_value))
                    epoch_samples_count = (step+1)*self.GLOBAL_BATCH_SIZE
                    print('Seen so far:                        {} samples'.format(epoch_samples_count))
                    eptime = time.perf_counter() - start_epoch_time
                    print('Speed:                              {:.1f} samples per second'.format(epoch_samples_count/eptime))
                    etime = time.perf_counter() - start_time
                    print('Execution time:                     {:.1f} sec == {:.1f} hours'.format(etime, etime/3600))
                    # критерий остановки по времени обучения
                    if etime > self.time: #60 * 60 * 24 * 3:
                        print('Stop training by timeout')
                        timeout_flag = True
                        break
            if timeout_flag:
                break
        # сохраняем модель декодера
        dec_model_dir = 'dec_model'
        if not os.path.exists(dec_model_dir):
            os.makedirs(dec_model_dir, exist_ok=True)
        self.ed_model.save_dec_weights(dec_model_dir)
        if self.mode == 'balance':
            # сохраняем все веса LowLevelEncoder
            self.common_lle_layer.save_own_weights('lle_weights')
        


def main():
    parser = argparse.ArgumentParser(description='balance_training')
    parser.add_argument('mode', type=str, choices=['prepare', 'balance'], help='training mode')
    parser.add_argument('time', type=int, help='time to train')
    args = parser.parse_args()
    if not args.mode or not args.time:
        parser.print_help()
        sys.exit(-1)
    btc = BalanceTrainingController(args.mode, args.time)
    btc.run()


if __name__ == "__main__":
    main()


