
import re


class Tokenizer(object):
    
    def __init__(self, embeddings_reader):
        self.embeddings_reader = embeddings_reader
        
    def process_text(self, text):
        tlist = [] # сами токены
        for i in re.split(r'\s+', text):
            ws = [j for j in re.split(r'([«»‛‚’‘‟„”“ˮʼ‒–—…,.:;"!?()[\]{}⟨⟩<>=≈≠+±/%°§])', i) if j]
            tlist.extend(ws)
        plist = [] # смещения токенов в тексте
        curpos = 0
        for t in tlist:
            curpos = text.find(t, curpos)
            plist.append(curpos)
            curpos += len(t)
        return self.tokens2subtokens([t.lower() for t in tlist]), plist
    
    def tokens2subtokens(self, tokens_list, oov_info = None):
        return [self.token2subtokens(t, oov_info) for t in tokens_list]
    
    def token2subtokens(self, token, oov_info = None):
        return self.embeddings_reader.token2ids(token, oov_info)
    
    
    
# код для самодиагностики

