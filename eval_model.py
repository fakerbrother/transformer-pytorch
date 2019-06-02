from Harvard_Transformer import *
from DataSet import *


dir_path = os.path.dirname(os.path.abspath(__file__))
SOS_token = 1
EOS_token = 2


def greedy_decode(model, src, max_len):
    tgt = torch.ones(1, 1).fill_(SOS_token).type_as(src.data)
    for i in range(max_len-1):
        output = model(src, tgt, src_mask=None, tgt_mask=None)
        output = output.view(-1, output.size(-1))
        next_word = torch.argmax(output, dim=1)
        next_word = next_word[-1].item()
        tgt = torch.cat([tgt, torch.zeros(1, 1).fill_(next_word).type_as(src.data)], dim=1)
        if next_word == EOS_token:
            break

    return tgt


def eval_model(save_dir=os.path.join(dir_path, "model", "16_checkpoint.tar"), max_len=50):
    checkpoint = torch.load(save_dir, map_location='cpu')

    input_lang = Lang("en")
    output_lang = Lang("cn")

    input_lang.__dict__ = checkpoint['input_lang']
    output_lang.__dict__ = checkpoint['output_lang']

    tf_model_sd = checkpoint['model']
    tf_model = make_model(input_lang.n_words, output_lang.n_words)
    tf_model.load_state_dict(tf_model_sd)

    tf_model.eval()
    test_sentence = "i like playing basketball very much !"
    test_indexes = indexes_from_sentence(input_lang, test_sentence)
    test_tensors = tensor_from_indexes(test_indexes, device=torch.device('cpu'))
    with torch.no_grad():
        tgt = list(greedy_decode(tf_model, test_tensors, max_len).squeeze().numpy())
    print(test_sentence)
    print(''.join(output_lang.index2word[index] for index in tgt[1:-1]))


if __name__ == "__main__":
    eval_model()