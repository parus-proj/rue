# Скрипт для обучения модели

# Использовать ли все GPU на машине
USE_ALL_GPU = True

from mlm_shared import MAX_LEN
from mlm_shared import BATCH_SIZE
from mlm_shared import CACHE_DIR

from mlm_shared import CATEG_DIMS
from mlm_shared import ASSOC_DIMS
from mlm_shared import GRAMM_DIMS

# from mlm_shared import ENC_WARMUP_STEPS
# from mlm_shared import ENC_STEPS_TO_CHANGE
# from mlm_shared import ENC_MIN_LR
# from mlm_shared import ENC_MAX_LR
# from mlm_shared import ENC_CHANGE_VALUE



from embeddings_reader import EmbeddingsReader

from low_level_encoder import LowLevelEncoder


import os
import time
#import numpy as np
import tensorflow as tf


# # контроллер скорости обучения для сети Энкодер
# class CustomScheduleEncOnly(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, warmup_steps=2000, min_lr = 1e-5, max_lr=1e-3, steps_to_change=300, change_val=5e-6):
#         super(CustomScheduleEncOnly, self).__init__()
#         self.warmup_steps = warmup_steps
#         self.min_lr = min_lr
#         self.max_lr = max_lr
#         self.steps_to_change = steps_to_change
#         self.change_val = change_val
#      
#     def __call__(self, step):
#         return tf.where( tf.less_equal( step,  self.warmup_steps ), self.incr(step), self.decr(step) )
#     def incr(self, step):
#         return (self.max_lr-self.min_lr)/self.warmup_steps * step + self.min_lr
#     def decr(self, step):
#         current = self.max_lr - (step-self.warmup_steps)/self.steps_to_change * self.change_val
#         return tf.where(current<self.min_lr, self.min_lr, current)




# мэппинг записи из файла данных в тензор, аналогичный по структуре тому, что выдает для предложения токенизатор
def record_as_tokenized(x):
    l0 = tf.strings.split(x, sep=tf.constant('/'))
    l1 = tf.strings.split(l0[0], sep=tf.constant(' '))
    l2 = tf.strings.to_number(l1, out_type=tf.int32)
    l3 = tf.reshape(l2, [-1, 2])                                # (sentence_len, 2) -- входное предложение после токенизации
    return l3

# паддинг записи данных
def record_padding_and_subsampling(x):
    seq_len = tf.shape(x)[0]
    pad_len = MAX_LEN - seq_len
    
    sentence_pad_seq = tf.tile( [[EmbeddingsReader.PAD_IDX, EmbeddingsReader.PAD_IDX]], [pad_len, 1] )
    
    sentence_msk = tf.concat( [tf.tile([1],[seq_len]), tf.tile([0],[pad_len])] , 0 ) # маска входной последовательности (единицы -- токены, нули -- PAD)
    
    return tf.concat([x, sentence_pad_seq], 0), sentence_msk



# функция создания модели
def create_model_for_mlm(vm):
    input = tf.keras.layers.Input(shape=(MAX_LEN,2), name='data')
    smsk_input = tf.keras.layers.Input(shape=(MAX_LEN), name='smsk')
    output = LowLevelEncoder(CATEG_DIMS, ASSOC_DIMS, GRAMM_DIMS, vm.stems_count, vm.sfx_count, ctxer_training=True, balance_training=False, name='lle') ([input, smsk_input])
    return tf.keras.Model(inputs=[input, smsk_input], outputs=output, name='MLM')





class MlmTrainingController(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Создадим объект-стратегию
        #print('GPU physical devices: {}'.format(tf.config.list_physical_devices('GPU')))
        #print('GPU logical devices: {}'.format(tf.config.list_logical_devices('GPU')))
        if USE_ALL_GPU:
            self.strategy = tf.distribute.MirroredStrategy() # стратегия single host multi gpu
        else:
            self.strategy = tf.distribute.get_strategy() # умолчательная стратегия
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        # Загрузим векторную модель (одновременно добавив в нее служебные векторы)
        print('Static model loading')
        self.vm = EmbeddingsReader("vectors.c2v")

        # Создадим датасет для доступа к данным
        print('Prepare dataset')
        self.GLOBAL_BATCH_SIZE = BATCH_SIZE * self.strategy.num_replicas_in_sync
        files_list = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.el') and os.path.isfile(os.path.join(CACHE_DIR, f))]
        self.dataset = tf.data.TextLineDataset(files_list) \
                        .map(record_as_tokenized, num_parallel_calls=tf.data.AUTOTUNE) \
                        .map(record_padding_and_subsampling, num_parallel_calls=tf.data.AUTOTUNE) \
                        .batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
                        .prefetch(tf.data.AUTOTUNE)
        if USE_ALL_GPU:
            # донастраиваем датасет под стратегию
            ds_options = tf.data.Options()
            ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            self.dataset = self.dataset.with_options(ds_options)
            self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        
        # Создадим loss-объекты  
        print('Loss creation')
        self.loss_object = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.secondary_loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        
    def prepare(self, lr=1e-3, lr2=1e-3):
        # Создадим модель и оптимизатор
        with self.strategy.scope():
            print('Create model')
            self.mlm_model = create_model_for_mlm(self.vm)
            self.enc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.enc_optimizer_fast = tf.keras.optimizers.Adam(learning_rate=lr2, beta_1=0.9, beta_2=0.99999)
            self.optimizer_selector = tf.Variable(0.0, trainable=False)
        # Загрузим веса
        print('Weights loading')
        self.mlm_model.get_layer('lle').load_embeddings(self.vm)
        # получим доступ к слоям эмбеддингов -- они используются как перекодировщики при вычислении loss
        self.stems_embedder = self.mlm_model.get_layer('lle').stems_emb_layer
        self.sfx_embedder   = self.mlm_model.get_layer('lle').sfx_emb_layer
        # выведем данные о модели
        self.mlm_model.summary()

    
    @tf.function
    def enc_compute_loss(self, true_vals, predictions, smsk):
        # Т.к. вычисление loss без редукции (Reduction.NONE), результатом применения и той, и другой loss-функции будут тензоры (batch, seq_len).
        # Надо свести всё к (batch, 1), чтобы потом корректно считать distributed-усреднение.
        # Среднюю косинусную меру для предложения (по всей seq_len) считаем с учетом маски.
        # Среднюю mse тоже с учетом маски, т.к. среднее, а не сумму считать будем.
        # На выходе (с учетом distributed-усреднения) будем иметь величину ошибки в рассчете на слово.
        msk_counts = tf.cast( tf.reduce_sum(smsk, -1), dtype=tf.float32 )
        msk_counts = tf.where( msk_counts == 0, tf.ones_like(msk_counts), msk_counts ) # защита от деления на 0
        # косинусная ошибка
        loss_1 = self.loss_object(true_vals, predictions)
        loss_1 = loss_1 + 1                                                            # для удобства интерпретации будем стремить loss к нулю
        loss_1 = tf.where(smsk == 1, loss_1, tf.zeros_like(loss_1))
        loss_1 = tf.reduce_sum(loss_1, -1) / msk_counts
        # среднеквадратичная ошибка
        loss_2 = self.secondary_loss_object(true_vals, predictions)
        loss_2 = tf.reduce_sum(loss_2, -1) / msk_counts
        
        per_example_loss = loss_1 + 0.1 * loss_2
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.GLOBAL_BATCH_SIZE)
    
       
    @tf.function
    def enc_get_loss_and_grads(self, x):
        with tf.GradientTape() as tape:
            x_real = x[0]  # x_real = (batch, seq_len, 2)  -- обучающие примеры
            x_smsk = x[1]  # x_tmsk = (batch, seq_len)     -- маска реальных токенов (без PAD)
            # выполняем шаг обучения
            # в замаскированных согласно x_smsk позициях модель возвращает PAD !
            y = self.mlm_model( [x_real,x_smsk], training=True )  # y = (batch, seq_len, dims)
            # получим входные данные в виде эмбеддингов
            embedded_x = tf.concat([self.stems_embedder(x_real[:,:,0]), self.sfx_embedder(x_real[:,:,1])], -1)  # embedded_x = (batch, seq_len, dims)
            loss_value = self.enc_compute_loss(embedded_x, y, x_smsk)
            #loss_value += tf.math.add_n(mlm_model.losses) # прибавим loss от L2-регуляризаторов
        grads = tape.gradient(loss_value, self.mlm_model.trainable_weights)
        return loss_value, grads


    @tf.function
    def enc_train_step_1(self, x):
        loss_value, grads = self.enc_get_loss_and_grads(x)
        self.enc_optimizer.apply_gradients(zip(grads, self.mlm_model.trainable_weights))
        return loss_value

    @tf.function
    def enc_train_step_2(self, x):
        loss_value, grads = self.enc_get_loss_and_grads(x)
        self.enc_optimizer_fast.apply_gradients(zip(grads, self.mlm_model.trainable_weights))
        return loss_value

    @tf.function
    def enc_distributed_train_step(self, x):
        if self.optimizer_selector == 0.0:
            per_replica_losses = self.strategy.run(self.enc_train_step_1, args=(x,))
        else:
            per_replica_losses = self.strategy.run(self.enc_train_step_2, args=(x,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    def run(self, timeout):
        # цикл обучения
        start_time = time.perf_counter()
        timeout_flag = False
        gstep = -1
        # цикл по эпохам
        for epoch in range(4):
            print('\nStart of epoch {}'.format(epoch))
            start_epoch_time = time.perf_counter()
            enc_epoch_loss = 0.0
            enc_step = 0
            # цикл по батчам датасета
            for step, x in enumerate(self.dataset):
                gstep += 1
                
                # выполняем шаг обучения
                enc_loss_value = self.enc_distributed_train_step( x )
                enc_epoch_loss += enc_loss_value
                enc_step += 1
                
                # выводим проткол через каждые N батчей.
                if step % 500 == 0:
                    print('Step No:   {}'.format(step))
                    #enc_opt_current_step = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, self.enc_optimizer.iterations, axis=None).numpy()
                    #print('Current encoder learning rate:      {:.5f}'.format(self.enc_lr_schedule(enc_opt_current_step)))
                    print('Enc. training loss:                 {:.4f}'.format(enc_epoch_loss/enc_step if enc_step > 0 else 0.0))
                    print('Enc. training loss for last batch:  {:.4f}'.format(enc_loss_value))
                    epoch_samples_count = (step+1)*self.GLOBAL_BATCH_SIZE
                    print('Seen so far:                        {} samples'.format(epoch_samples_count))
                    eptime = time.perf_counter() - start_epoch_time
                    print('Speed:                              {:.1f} samples per second'.format(epoch_samples_count/eptime))
                    etime = time.perf_counter() - start_time
                    print('Execution time:                     {:.1f} sec == {:.1f} hours'.format(etime, etime/3600))
                    # критерий остановки по времени обучения
                    if etime > timeout:
                        print('Stop training by timeout')
                        timeout_flag = True
                        break

                if gstep == 4000:
                    self.optimizer_selector.assign(1.0)
                                    
            if timeout_flag:
                break
           
        # сохраняем все веса LowLevelEncoder
        self.mlm_model.get_layer('lle').save_own_weights('lle_weights')




def main():
    mtc = MlmTrainingController()
    mtc.prepare(lr=8e-3, lr2=9e-3)
    mtc.run(3*60*60)


if __name__ == "__main__":
    main()


