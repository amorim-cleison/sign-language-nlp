# """
# Module to represents whole models
# """

# import model.util as util
# import torch.nn.functional as F
# from joeynmt.decoders import Decoder, RecurrentDecoder
# from joeynmt.embeddings import Embeddings
# from joeynmt.encoders import Encoder, RecurrentEncoder
# from joeynmt.initialization import initialize_model
# from joeynmt.model import Model
# from joeynmt.vocabulary import Vocabulary
# from torch import Tensor, nn
# import torch


# class CustomModel(Model):
#     def __init__(self, encoder: Encoder, decoder: Decoder,
#                  src_embed: Embeddings, trg_embed: Embeddings,
#                  src_vocab: Vocabulary, trg_vocab: Vocabulary) -> None:
#         super(CustomModel, self).__init__(encoder, decoder, src_embed,
#                                           trg_embed, src_vocab, trg_vocab)

#     def forward(self, return_type: str = None, **kwargs) \
#             -> (Tensor, Tensor, Tensor, Tensor):
#         """ Interface for multi-gpu

#         For DataParallel, We need to encapsulate all model call: model.encode(),
#         model.decode(), and model.encode_decode() by model.__call__().
#         model.__call__() triggers model.forward() together with pre hooks
#         and post hooks, which take care of multi-gpu distribution.

#         :param return_type: one of {"softmax", "loss", "encode", "decode"}
#         """
#         if return_type is None:
#             raise ValueError("Please specify return_type: "
#                              "{`softmax`, `loss`, `encode`, `decode`}.")

#         return_tuple = (None, None, None, None)

#         if return_type == "softmax":
#             out, x, _, _ = self._encode_decode(**kwargs)

#             # compute log probs
#             log_probs = F.log_softmax(out, dim=-1)
#             return_tuple = (log_probs, None, None, None)
#         else:
#             return_tuple = super().forward(return_type, **kwargs)

#         return return_tuple


# class EncoderDecoderAttnBase(nn.Module):
#     def __init__(self,
#                  src_vocab,
#                  tgt_vocab,
#                  batch_first,
#                  rnn_type,
#                  embedding_size=256,
#                  hidden_size=512,
#                  num_layers=1,
#                  dropout=0.1,
#                  **kwargs):
#         super(EncoderDecoderAttnBase, self).__init__()

#         assert (rnn_type in {"gru", "lstm", "transformer"}), "Invalid RNN type"

#         self.src_vocab = src_vocab
#         self.tgt_vocab = tgt_vocab

#         # Pad index:
#         # src_padding_idx = src_vocab.stoi[PAD_TOKEN]
#         # trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]
#         src_pad_idx = util.get_pad_idx(src_vocab)
#         tgt_pad_idx = util.get_pad_idx(tgt_vocab)

#         # Vocab size:
#         src_vocab_size = len(src_vocab)
#         tgt_vocab_size = len(tgt_vocab)

#         # src_embed = Embeddings(
#         #     **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
#         #     padding_idx=src_padding_idx)
#         src_embed = Embeddings(embedding_dim=embedding_size,
#                                vocab_size=src_vocab_size,
#                                padding_idx=src_pad_idx,
#                                scale=False,
#                                freeze=False)

#         # this ties source and target embeddings
#         # for softmax layer tying, see further below
#         # if cfg.get("tied_embeddings", False):
#         #     if src_vocab.itos == trg_vocab.itos:
#         #         # share embeddings for src and trg
#         #         trg_embed = src_embed
#         #     else:
#         #         raise ConfigurationError(
#         #             "Embedding cannot be tied since vocabularies differ.")
#         # else:
#         #     trg_embed = Embeddings(
#         #         **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
#         #         padding_idx=trg_padding_idx)

#         trg_embed = Embeddings(embedding_dim=embedding_size,
#                                vocab_size=tgt_vocab_size,
#                                padding_idx=tgt_pad_idx,
#                                scale=False,
#                                freeze=False)

#         # build encoder
#         # enc_dropout = cfg["encoder"].get("dropout", 0.)
#         # enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
#         # if cfg["encoder"].get("type", "recurrent") == "transformer":
#         #     assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
#         #         cfg["encoder"]["hidden_size"], \
#         #         "for transformer, emb_size must be hidden_size"

#         #     encoder = TransformerEncoder(**cfg["encoder"],
#         #                                 emb_size=src_embed.embedding_dim,
#         #                                 emb_dropout=enc_emb_dropout)
#         # else:
#         #     encoder = RecurrentEncoder(**cfg["encoder"],
#         #                             emb_size=src_embed.embedding_dim,
#         #                             emb_dropout=enc_emb_dropout)

#         encoder = RecurrentEncoder(
#             rnn_type=rnn_type,
#             hidden_size=hidden_size,
#             emb_size=src_embed.embedding_dim,
#             num_layers=num_layers,
#             dropout=dropout,
#             #    emb_dropout=dropout,
#             bidirectional=True,
#             freeze=False)

#         # build decoder
#         # dec_dropout = cfg["decoder"].get("dropout", 0.)
#         # dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
#         # if cfg["decoder"].get("type", "recurrent") == "transformer":
#         #     decoder = TransformerDecoder(
#         #         **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
#         #         emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
#         # else:
#         #     decoder = RecurrentDecoder(
#         #         **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
#         #         emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

#         decoder = RecurrentDecoder(
#             rnn_type=rnn_type,
#             emb_size=trg_embed.embedding_dim,
#             hidden_size=hidden_size,
#             encoder=encoder,
#             num_layers=num_layers,
#             vocab_size=tgt_vocab_size,
#             dropout=dropout,
#             #    emb_dropout=dropout,
#             hidden_dropout=dropout,
#             input_feeding=True,
#             attention="bahdanau",
#             init_hidden="bridge",
#             freeze=False)

#         model = CustomModel(encoder=encoder,
#                             decoder=decoder,
#                             src_embed=src_embed,
#                             trg_embed=trg_embed,
#                             src_vocab=src_vocab,
#                             trg_vocab=tgt_vocab)

#         # tie softmax layer with trg embeddings
#         # if cfg.get("tied_softmax", False):
#         #     if trg_embed.lut.weight.shape == \
#         #             model.decoder.output_layer.weight.shape:
#         #         # (also) share trg embeddings and softmax layer:
#         #         model.decoder.output_layer.weight = trg_embed.lut.weight
#         #     else:
#         #         raise ConfigurationError(
#         #             "For tied_softmax, the decoder embedding_dim and decoder "
#         #             "hidden_size must be the same."
#         #             "The decoder must be a Transformer.")

#         # custom initialization of model parameters
#         initialize_model(model, {}, src_pad_idx, tgt_pad_idx)

#         # initialize embeddings from file
#         # pretrained_enc_embed_path = cfg["encoder"]["embeddings"].get(
#         #     "load_pretrained", None)
#         # pretrained_dec_embed_path = cfg["decoder"]["embeddings"].get(
#         #     "load_pretrained", None)
#         # if pretrained_enc_embed_path:
#         #     logger.info("Loading pretraind src embeddings...")
#         #     model.src_embed.load_from_file(pretrained_enc_embed_path, src_vocab)
#         # if pretrained_dec_embed_path and not cfg.get("tied_embeddings", False):
#         #     logger.info("Loading pretraind trg embeddings...")
#         #     model.trg_embed.load_from_file(pretrained_dec_embed_path, trg_vocab)

#         # logger.info("Enc-dec model built.")

#         self.model = model

#     def forward(self, X, y, lengths, **kwargs):
#         # def forward(self, return_type: str = None, **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
#         # def _encode_decode(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
#         #                    src_length: Tensor, trg_mask: Tensor = None, **kwargs) -> (Tensor, Tensor, Tensor, Tensor):

#         X, y, lengths = self.sort_by_lengths(X, y, lengths)

#         src = X
#         tgt = y
#         # tgt = self.create_bos_tensor_like(tgt, self.tgt_vocab)

#         if (tgt.ndim < 2):
#             tgt = tgt.unsqueeze(dim=-1)
#         # tgt = self.prepend_bos(tgt, self.tgt_vocab)

#         # Masks:
#         src_mask = self.generate_mask(src, self.src_vocab)
#         tgt_mask = self.generate_mask(tgt, self.tgt_vocab)

#         # Lengths:
#         src_lengths = lengths
#         # tgt_lengths = util.resolve_lengths(tgt, self.tgt_vocab)

#         output, _, _, _ = self.model(
#             return_type='softmax',
#             src=src,
#             trg_input=tgt,
#             src_mask=src_mask,
#             src_length=src_lengths,
#             trg_mask=tgt_mask,
#         )
#         return output[:, -1]

#     def generate_mask(self, data, vocab):
#         pad_idx = util.get_pad_idx(vocab)
#         return (data != pad_idx).unsqueeze(1)

#     def sort_by_lengths(self, X, y, lengths):
#         def tensor_like(data, like):
#             if (data[0].ndim == 0):
#                 return torch.tensor(data, dtype=like.dtype, device=like.device)
#             else:
#                 return torch.cat(data).view(like.size())

#         _lengths_dim = -1
#         _sorted = sorted(zip(X, y, lengths),
#                          key=lambda item: item[_lengths_dim],
#                          reverse=True)
#         _X, _y, _lengths = zip(*_sorted)
#         return (tensor_like(_X,
#                             X), tensor_like(_y,
#                                             y), tensor_like(_lengths, lengths))

#     def prepend_bos(self, data, vocab):
#         bos_data = self.create_bos_tensor_like(data, vocab)
#         return torch.cat([bos_data, data], dim=1).to(data.device)

#     def create_bos_tensor_like(self, data, vocab):
#         batch_size = data.size(0)
#         bos_idx = util.get_bos_idx(vocab)
#         return torch.full((batch_size, 1), bos_idx).to(data.device)
