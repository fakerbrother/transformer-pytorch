import torch.optim as optim
from Harvard_Transformer import *
from DataSet import *
import time


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, output_size, padding_idx=PAD, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = output_size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def init(batch_szie):
    data_generator, input_lang, output_lang = batch_train_data(batch_size=batch_szie)
    src_vocab_size, tgt_voacb_szie = input_lang.n_words, output_lang.n_words
    transformer_model = make_model(src_vocab_size, tgt_voacb_szie)
    optimizer = get_std_opt(transformer_model)
    criterion = LabelSmoothing(output_lang.n_words)
    return transformer_model, optimizer, data_generator, input_lang, output_lang, criterion


def train(save_dir=dir_path, epochs=16, batch_size=32):
    tf_model, optimizer_class, data_gen, input_language, output_language, criterion_class = init(batch_szie=batch_size)
    tf_model = tf_model.to(device)
    for epoch in range(epochs):
        start = time.time()
        print("\n开始第{}epoch的训练！\n".format(epoch+1))
        total_tokens = 0
        total_loss = 0.
        tokens = 0
        for src_x, target_y in data_gen:
            src_x, target_y = src_x.squeeze(-1), target_y.squeeze(-1)
            target_input, target_true = target_y[:, :-1], target_y[:, 1:].contiguous().view(-1)
            src_mask = (src_x != PAD).unsqueeze(-2)
            tgt_mask = make_std_mask(target_input, PAD)
            ntokens = (target_true != PAD).data.sum()
            out = tf_model(src_x, target_input, src_mask, tgt_mask)
            out = out.contiguous().view(-1, output_language.n_words)
            loss = criterion_class(out, target_true) / ntokens
            loss.backward()
            total_loss += loss.item() * ntokens
            total_tokens += ntokens
            tokens += ntokens
            optimizer_class.step()
            optimizer_class.optimizer.zero_grad()
            if optimizer_class._step % 20 == 0:
                elapsed = time.time() - start
                print("Epoch Step: {} Loss: {:.6f} Tokens per Sec: {:.6f}".format(
                    optimizer_class._step, loss, tokens / elapsed
                ))
                start = time.time()
                tokens = 0
        print("This Epoch's Total Loss is: {}".format(total_loss / total_tokens))
        torch.save({
            "model": tf_model.state_dict(),
            "input_lang": input_language.__dict__,
            "output_lang": output_language.__dict__
        }, os.path.join(save_dir, "model", "{}_checkpoint.tar".format(epoch+1)))


if __name__ == "__main__":
    train()