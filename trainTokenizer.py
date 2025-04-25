from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

import glob


path1 = glob.glob("./data/Malicious/*")
path2 = glob.glob("./data/Legitimate/*")
partPath2= path2[0:int(len(path2)/30)]
paths = path1 + partPath2



tokenizer = Tokenizer(BPE())


train = True 

if train:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    #trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=8192,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(paths, trainer)

    tokenizer.save('./tokenizer.json')


#fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/bpe_tokenizer/tokenizer.json")
trained_tokenizer = Tokenizer.from_file("./tokenizer.json")
print("Vocab size %d" %(trained_tokenizer.get_vocab_size()))
