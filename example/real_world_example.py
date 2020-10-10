import torch
import torch.nn as nn
from exported import Batch, LabelSmoothing, NoamOpt
from exported import make_model, batch_size_fn, run_epoch
from .first_example import greedy_decode, SimpleLossCompute
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

# For data loading.
from torchtext import data, datasets

# REAL WORLD EXAMPLE =====================================

# !pip install torchtext spacy
# !python -m spacy download en
# !python -m spacy download de


# Batching matters a ton for speed. We want to have very evenly divided
# batches, with absolutely minimal padding. To do this we have to hack a bit
# around the default torchtext batching. This code patches their default
# batching to make sure we search over enough sentences to find tight batches.
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:

            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key), self.batch_size,
                        self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# Finally to really target fast training, we will use multi-gpu. This code
# implements multi-gpu word generation. It is not specific to transformer
# so I won't go into too much detail. The idea is to split up word generation
# at training time into chunks to be processed in parallel across many
# different gpus. We do this using pytorch parallel primitives:


# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[
                Variable(
                    o[:, i:i + chunk_size].data,
                    requires_grad=self.opt is not None)
            ] for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_value = nn.parallel.gather(loss, target_device=self.devices[0])
            l_value = l_value.sum()[0] / normalize
            total += l_value.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_value.backward()
                for j, l_value in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


def run(**kwargs):
    if True:
        import spacy
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(
            tokenize=tokenize_en,
            init_token=BOS_WORD,
            eos_token=EOS_WORD,
            pad_token=BLANK_WORD)

        MAX_LEN = 100
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'),
            fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
        )
        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    # Now we create our model, criterion, optimizer, data iterators, and
    # paralelization

    # GPUs to use
    # devices = [0, 1, 2, 3]
    devices = None
    train_model = True

    if True:
        cuda_enabled = (devices is not None) and len(devices) > 0

        pad_idx = TGT.vocab.stoi["<blank>"]
        model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
        if cuda_enabled:
            model.cuda()
        criterion = LabelSmoothing(
            size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        if cuda_enabled:
            criterion.cuda()

        BATCH_SIZE = 12000
        train_iter = MyIterator(
            train,
            batch_size=BATCH_SIZE,
            device=0,
            repeat=False,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            batch_size_fn=batch_size_fn,
            train=True)
        valid_iter = MyIterator(
            val,
            batch_size=BATCH_SIZE,
            device=0,
            repeat=False,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            batch_size_fn=batch_size_fn,
            train=False)
        model_par = nn.DataParallel(model, device_ids=devices)
    None

    # !wget https://s3.amazonaws.com/opennmt-models/iwslt.pt

    if train_model:
        model_opt = NoamOpt(
            model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model_par.train()

            def get_loss_compute(opt):
                if cuda_enabled:
                    return MultiGPULossCompute(
                        model.generator, criterion, devices=devices, opt=opt)
                else:
                    return SimpleLossCompute(model.generator, criterion, opt)

            run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                      get_loss_compute(model_opt))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             get_loss_compute(None))
            print(loss)
    else:
        model = torch.load("iwslt.pt")

    # Once trained we can decode the model to produce a set of translations. Here
    # we simply translate the first sentence in the validation set. This dataset
    # is pretty small so the translations with greedy search are reasonably
    # accurate.
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(
            model,
            src,
            src_mask,
            max_len=60,
            start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        break

    model.eval()
    sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split(
    )
    src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
    src = Variable(src)
    src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(
        model, src, src_mask, max_len=60, start_symbol=TGT.stoi["<s>"])
    print("Translation:", end="\t")
    trans = "<s> "
    for i in range(1, out.size(1)):
        sym = TGT.itos[out[0, i]]
        if sym == "</s>":
            break
        trans += sym + " "
    print(trans)

    # Attention Visualization
    # Even with a greedy decoder the translation looks pretty good. We can further
    # visualize it to see what is happening at each layer of the attention

    tgt_sent = trans.split()

    def draw(data, x, y, ax):
        seaborn.heatmap(
            data,
            xticklabels=x,
            square=True,
            yticklabels=y,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=ax)

    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(4):
            draw(
                model.encoder.layers[layer].self_attn.attn[0, h].data,
                sent,
                sent if h == 0 else [],
                ax=axs[h])
        plt.show()

    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Decoder Self Layer", layer + 1)
        for h in range(4):
            draw(
                model.decoder.layers[layer].self_attn.attn[0, h].
                data[:len(tgt_sent), :len(tgt_sent)],
                tgt_sent,
                tgt_sent if h == 0 else [],
                ax=axs[h])
        plt.show()
        print("Decoder Src Layer", layer + 1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(
                model.decoder.layers[layer].self_attn.attn[0, h].
                data[:len(tgt_sent), :len(sent)],
                sent,
                tgt_sent if h == 0 else [],
                ax=axs[h])
        plt.show()
