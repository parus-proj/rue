
from gensim.models import KeyedVectors
import numpy as np

# Класс для загрузки векторных представлений в память + добаление специальных эмбеддингов
class EmbeddingsReader(object):
    # Константы
    PAD_NAME = '_PAD_'
    OOV_NAME = '_OOV_'
    PAD_IDX = 0
    OOV_IDX = 1
    PAD_FILLER = 10.0
    OOV_FILLER = 0.0
    
    # Конструктор
    # на выходе имеем:
    #   self.stems_count -- размер векторной модели псевдооснов
    #   self.sfx_count   -- размер векторной модели суффиксов
    #   self.stems_size  -- размерность векторов псевдооснов
    #   self.sfx_size    -- размерность векторов суффиксов
    #   self.stems_embs  -- матрица векторных представлений псевдооснов (двумерный numpy-массив)
    #   self.sfx_embs    -- матрица векторных представлений суффиксов (двумерный numpy-массив)
    #   self.forms_dict, self.stems_dict, self.sfx_dict -- для использования внутри
    def __init__(self, embeddings_filename, force_oov_with_sfx=False):
        self.force_oov_with_sfx = force_oov_with_sfx
        
        # Загрузка векторных представлений
        em_forms = KeyedVectors.load_word2vec_format(embeddings_filename+'.forms', binary=True)
        em_stems = KeyedVectors.load_word2vec_format(embeddings_filename+'.stems', binary=True)
        em_sfx   = KeyedVectors.load_word2vec_format(embeddings_filename+'.sfx',   binary=True)
        
        # Вычисляем размеры матриц псевдооснов и суффиксов
        self.stems_size = em_stems.vectors.shape[1]
        self.sfx_size = em_sfx.vectors.shape[1]
        assert self.stems_size+self.sfx_size == em_forms.vectors.shape[1]
        
        SPECIAL_EMBEDDINGS_COUNT = 2
        self.stems_count = em_forms.vectors.shape[0] + em_stems.vectors.shape[0] + SPECIAL_EMBEDDINGS_COUNT
        self.sfx_count = em_forms.vectors.shape[0] + em_sfx.vectors.shape[0] + SPECIAL_EMBEDDINGS_COUNT

        # Создаем матрицы        
        self.stems_embs = np.zeros((self.stems_count, self.stems_size), dtype=np.float32)
        self.sfx_embs = np.zeros((self.sfx_count, self.sfx_size), dtype=np.float32)
        
        # Наполняем матрицы
        self.stems_embs[EmbeddingsReader.PAD_IDX, :] = np.full((self.stems_size,), EmbeddingsReader.PAD_FILLER)
        self.sfx_embs[EmbeddingsReader.PAD_IDX, :]   = np.full((self.sfx_size,), EmbeddingsReader.PAD_FILLER)
        self.stems_embs[EmbeddingsReader.OOV_IDX, :] = np.full((self.stems_size,), EmbeddingsReader.OOV_FILLER)
        self.sfx_embs[EmbeddingsReader.OOV_IDX, :]   = np.full((self.sfx_size,), EmbeddingsReader.OOV_FILLER)
        
        offset = SPECIAL_EMBEDDINGS_COUNT
        for i in range(em_forms.vectors.shape[0]):
            self.stems_embs[offset+i, :] = em_forms.vectors[i, :self.stems_size]
            self.sfx_embs[offset+i, :] = em_forms.vectors[i, self.stems_size:]
        offset += em_forms.vectors.shape[0]
        for i in range(em_stems.vectors.shape[0]):
            self.stems_embs[offset+i, :] = em_stems.vectors[i, :]
        for i in range(em_sfx.vectors.shape[0]):
            self.sfx_embs[offset+i, :] = em_sfx.vectors[i, :]
            
        # Создадим словари для последующих преобразований текста в индексы эмбеддингов
        self.forms_dict = {}
        for idx, word in enumerate(list(em_forms.key_to_index.keys())):
            self.forms_dict[word] = idx+SPECIAL_EMBEDDINGS_COUNT
        self.stems_dict = {}
        for idx, word in enumerate(list(em_stems.key_to_index.keys())):
            self.stems_dict[word] = idx + SPECIAL_EMBEDDINGS_COUNT + em_forms.vectors.shape[0]
        self.sfx_dict = {}
        self.sfx_dict[''] = EmbeddingsReader.OOV_IDX # добавим пустой суффикс с отсылкой к _OOV_
        for idx, word in enumerate(list(em_sfx.key_to_index.keys())):
            self.sfx_dict[word[5:]] = idx + SPECIAL_EMBEDDINGS_COUNT + em_forms.vectors.shape[0]  # сразу отсекаем префикс _OOV_
        
        em_forms, em_stems, em_sfx = None, None, None
        

    def token2ids(self, token_form, oov_info = None):
        # простейшее решение для цифровых последовательностей
        if token_form.isnumeric():
            if oov_info:
                oov_info["numeric"] += 1
            idx = self.forms_dict['@num@']
            return [idx, idx]
        
        # прежде всего ищем в словаре полных форм
        if token_form in self.forms_dict:
            idx = self.forms_dict[token_form]
            return [idx, idx]
        
        # если не нашли в словаре полных форм, пытаемся разрезать на псевдооснову и суффикс (от нулевого суффикса к наиболее длинному)
        tl = len(token_form)
        MIN_STEM_SIZE = 3 # псевдооснова должна быть длиной хотя бы 3 символа
        MAX_SFX_SIZE = 5  # суффиксы длиной от 0 до 5
        if tl >= MIN_STEM_SIZE:
            # проверим вариант с нулевым суффиксом
            
            max_sfx_pos = tl - 1 - MAX_SFX_SIZE
            if max_sfx_pos < MIN_STEM_SIZE-1:
                max_sfx_pos = MIN_STEM_SIZE-1
            for i in range(tl, max_sfx_pos, -1):
                stem = token_form[:i]
                suffix = token_form[i:]
                #print('see {}~{}'.format(stem, suffix))
                if stem in self.stems_dict and suffix in self.sfx_dict:
                    # попробуем найти суффикс подлиннее (он информативнее)
                    for j in range(i, max_sfx_pos, -1):
                        sfx_j = token_form[j:]
                        if sfx_j in self.sfx_dict:
                            suffix = sfx_j
                    # фиксируем сборную конструкцию из псевдоосновы и суффикса
                    #print('final {}+{}'.format(stem, suffix))
                    return [self.stems_dict[stem], self.sfx_dict[suffix]]
                
        # если разрезать не удалось, то пытаемся найти хотя бы суффикс
        if tl > MAX_SFX_SIZE:
            for i in range(-MAX_SFX_SIZE, 0, 1):
                suffix = token_form[i:]
                if suffix in self.sfx_dict:
                    if oov_info:
                        oov_info["oov_sfx"] += 1
                    stem_idx = self.stems_dict[EmbeddingsReader.OOV_NAME+suffix] if not self.force_oov_with_sfx else EmbeddingsReader.OOV_IDX
                    return [stem_idx, self.sfx_dict[suffix]]  # the most long suffix
        
        # не нашли вообще ничего -- полный _OOV_
        if oov_info:
            oov_info["oov"] += 1
        return [EmbeddingsReader.OOV_IDX, EmbeddingsReader.OOV_IDX]





# # код для самодиагностики
# print('Loading embeddings...')
# vm = EmbeddingsReader("vectors.c2v")
# print('  stems emb size = {}'.format(vm.stems_size))
# print('  sfx emb size = {}'.format(vm.sfx_size))
# print('  stems emb count = {}'.format(vm.stems_count))
# print('  sfx emb count = {}'.format(vm.sfx_count))
#       
#       
# for token in ['в', 'ясными', 'гравитационная', '100%-ного', '3d-принтере', '3d-принтер', '1920-1930-', 'jhdfjkahdf', 'оылврзация', ';', '45']:
#     print('{} - {}'.format(token, vm.token2ids(token)))

