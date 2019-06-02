from io import open
import re
import torch
import os
import torch.utils.data as Data


dir_path = os.path.dirname(os.path.abspath(__file__))
SOS_token = 1
EOS_token = 2
PAD = 0

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1="en", lang2="cn", reverse=False):
    print("Read lines...")
    # Read the file and split into lines
    with open(os.path.join(dir_path, "data", "%s-%s.txt" % (lang1, lang2)), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Split every line into pairs and normalize
    line = [l.split('\t') for l in lines]
    en_data = [normalizeString(word[0]) for word in line]
    cn_data = [" ".join(s for s in word[1]) for word in line]
    pairs = [[en_sentence, cn_sentence]for en_sentence, cn_sentence in zip(en_data, cn_data)]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def prepareData(lang1="en", lang2="cn", reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read {} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        if not reverse:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def compute_max_length(sequences):
    length_list = []
    for seq in sequences:
        length = len(seq)
        length_list.append(length)
    return max(length_list)


def input_collate_func(inputs, max_seq_length):
    return inputs + [PAD]*(max_seq_length-len(inputs))


def output_collate_func(outputs, max_seq_length):
    return [SOS_token] + outputs + [EOS_token] + [PAD] * (max_seq_length - len(outputs))


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_indexes(indexes, device=device):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def train_data():
    input_lang, output_lang, pairs = prepareData()
    inputs_indexes = []
    outputs_indexes = []
    for input_sentence, output_sentence in pairs:
        input_indexes = indexes_from_sentence(input_lang, input_sentence)
        output_indexes = indexes_from_sentence(output_lang, output_sentence)
        inputs_indexes.append(input_indexes)
        outputs_indexes.append(output_indexes)
    input_max_length = compute_max_length(inputs_indexes)
    output_max_length = compute_max_length(outputs_indexes)
    padded_inputs_indexes = [input_collate_func(input_indexes, input_max_length) for input_indexes in inputs_indexes]
    padded_outputs_indexes = [output_collate_func(output_indexes, output_max_length) for output_indexes in outputs_indexes]
    inputs_tensor = [tensor_from_indexes(padded_input_indexes) for padded_input_indexes in padded_inputs_indexes]
    outputs_tensor = [tensor_from_indexes(padded_output_indexes) for padded_output_indexes in padded_outputs_indexes]
    return inputs_tensor, outputs_tensor, input_lang, output_lang


class PairsDataSet(Data.Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def __getitem__(self, index):
        return self.src_sequences[index], self.tgt_sequences[index]

    def __len__(self):
        return len(self.src_sequences)


def batch_train_data(batch_size=16):
    inputstensor, outputstensor, input_lang, output_lang = train_data()
    dataset = PairsDataSet(inputstensor, outputstensor)
    data_generator = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_generator, input_lang, output_lang


# if __name__ == "__main__":
#     data_generator, inputmaxlength, outputmaxlen, input_lang, output_lang = batch_train_data()
#     print(inputmaxlength, outputmaxlen)
#     for x_batch, y_batch in data_generator:
#         print("x_train_len:{}, y_train_len:{}".format(x_batch.shape, y_batch.shape))
