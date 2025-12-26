import torch
from sentencepiece import SentencePieceProcessor
from torch import Tensor

from .decoder import CTCHead, RNNTHead


class Tokenizer:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """

    def __init__(self, vocab: list[str], model_path: str | None = None):
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            self.model = SentencePieceProcessor()
            self.model.load(model_path)  # type: ignore[attr-defined]

    def decode(self, tokens: list[int]) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)  # type: ignore[attr-defined]

    def __len__(self):
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)


class CTCGreedyDecoding:
    """
    Class for performing greedy decoding of CTC outputs.
    """

    def __init__(self, vocabulary: list[str], model_path: str | None = None):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)

    @torch.inference_mode()
    def decode(self, head: CTCHead, encoded: Tensor, lengths: Tensor) -> list[str]:
        """
        Decode the output of a CTC model into a list of hypotheses.
        """
        log_probs = head(encoder_output=encoded)
        assert len(log_probs.shape) == 3, f"Expected log_probs shape {log_probs.shape} == [B, T, C]"
        b, _, c = log_probs.shape
        assert c == len(self.tokenizer) + 1, f"Num classes {c} != len(vocab) + 1 {len(self.tokenizer) + 1}"
        labels = log_probs.argmax(dim=-1, keepdim=False)

        skip_mask = labels != self.blank_id
        skip_mask[:, 1:] = torch.logical_and(skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1])
        for i, length in enumerate(lengths):
            skip_mask[i, length:] = 0

        pred_texts: list[str] = []
        masked_labels = [labels[i][skip_mask[i]] for i in range(b)]

        for i in range(b):
            decoded_tokens = self.tokenizer.decode(masked_labels[i].cpu().tolist())
            pred_texts.append("".join(decoded_tokens))

        return pred_texts


class RNNTGreedyDecoding:
    """
    Class for performing greedy decoding of RNN-T outputs.
    """

    def __init__(
        self,
        vocabulary: list[str],
        model_path: str | None = None,
        max_symbols_per_step: int = 10,
    ):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    def _greedy_decode(self, head: RNNTHead, x: Tensor, seqlen: Tensor) -> str:
        """
        Internal helper function for performing greedy decoding on a single sequence.
        """
        hyp: list[int] = []
        dec_state: Tensor | None = None
        last_label: Tensor | None = None

        for t in range(seqlen):
            f = x[t, :, :].unsqueeze(1)
            not_blank = True
            new_symbols = 0
            while not_blank and new_symbols < self.max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
                if k == self.blank_id:
                    not_blank = False
                else:
                    hyp.append(int(k))
                    dec_state = hidden
                    if last_label is None:
                        last_label = torch.zeros((1, 1), dtype=torch.long, device=x.device)
                    last_label[0, 0] = hyp[-1]
                    new_symbols += 1

        return self.tokenizer.decode(hyp)

    @torch.inference_mode()
    def decode(self, head: RNNTHead, encoded: Tensor, enc_len: Tensor) -> list[str]:
        """
        Decode the output of an RNN-T model into a list of hypotheses.
        """
        b = encoded.shape[0]
        assert enc_len.shape[0] == b, f"Batch size mismatch: {enc_len.shape[0]} != {b}"
        pred_texts = []
        encoded = encoded.transpose(1, 2)
        for i in range(b):
            inseq = encoded[i, :, :].unsqueeze(1)
            pred_texts.append(self._greedy_decode(head, inseq, enc_len[i]))
        return pred_texts
