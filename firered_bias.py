"""FireRedASR2-AED 的熱詞加權（contextual biasing），不改上游 clone。

FireRed AED 只吃聲音，沒有文字輸入可放熱詞。這裡在 beam search 每一步替「正在拼出
字典詞」的 token 加分，拼到一半斷掉就把已加的分數扣回去，拼完整個詞才保留加分。
做法同 WeNet 的 context graph。

用法：
    import firered_bias
    firered_bias.install()                       # 載入 fireredasr2s 之後呼叫一次
    firered_bias.set_hotwords(tokenizer, words, weight)   # 每次請求前
    ... system.process(...) ...
    firered_bias.set_hotwords(tokenizer, [], 0)  # 清掉

_batch_beam_search 改寫自 FireRedASR2S 的 TransformerDecoder.batch_beam_search
（Copyright 2026 Xiaohongshu，Apache-2.0）。

notes: 整個 batch_beam_search 是從上游（FireRedASR2S main，2026-10-21 clone）複製過來再加三段，
       標了 [bias]。上游改了這個函式，這份不會跟著變——升級 clone 後要重新比對。
       不加熱詞時（_current 為 None）行為與上游相同。
notes: 狀態轉移在 Python 迴圈裡做，每步 O(beam × batch × 子節點數)。ZeroType 一次一句、
       beam 3、字典幾十個詞，成本可忽略；字典到上千詞或 batch 變大時要改成張量化。
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

_current: Optional["ContextBias"] = None


class ContextBias:
    """狀態是 (節點, 已加但還沒落袋的分數, 這條 hypothesis 已拼完的詞)。

    notes: 每個詞在一條 hypothesis 裡只加分一次。實測不設這條限制時，權重 4.5 起就會
           無限重複字典詞（「Session Session Session…」）賺加分，100 筆裡 13 筆出事，
           權重 6 幾乎每筆都壞。代價是同一句講兩次同一個詞，第二次只能靠聲音本身。
    """

    def __init__(self, token_seqs: List[List[int]], weight: float):
        self.weight = weight
        self.children: List[Dict[int, int]] = [{}]
        self.end_word: List[int] = [-1]
        for wid, seq in enumerate(token_seqs):
            node = 0
            for tok in seq:
                nxt = self.children[node].get(tok)
                if nxt is None:
                    nxt = len(self.children)
                    self.children[node][tok] = nxt
                    self.children.append({})
                    self.end_word.append(-1)
                node = nxt
            self.end_word[node] = wid
        # 每個節點底下還拼得出哪些詞；已拼完的詞從這裡排除，就不會再加分
        self.reach: List[frozenset] = [frozenset()] * len(self.children)
        for node in reversed(range(len(self.children))):
            ids = {self.end_word[node]} if self.end_word[node] >= 0 else set()
            for c in self.children[node].values():
                ids |= self.reach[c]
            self.reach[node] = frozenset(ids)

    def initial(self):
        return (0, 0.0, frozenset())

    def _open(self, node: int, done: frozenset) -> Dict[int, int]:
        return {t: c for t, c in self.children[node].items() if self.reach[c] - done}

    def _enter(self, node: int, acc: float, done: frozenset):
        """進入一個節點後的狀態。拼完一個詞：加分落袋（acc 歸零）、記成已拼完；沒有更長的詞就回根。"""
        wid = self.end_word[node]
        if wid >= 0:
            acc = 0.0
            done = done | {wid}
            if not self._open(node, done):
                node = 0
        return node, acc, done

    def step(self, state, tok: int):
        """(新狀態, 這一步的分數變化)。三種情況：接著拼、斷掉重新開始、斷掉。"""
        node, acc, done = state
        w = self.weight
        nxt = self._open(node, done).get(tok)
        if nxt is not None:
            return self._enter(nxt, acc + w, done), w
        if node != 0:
            nxt = self._open(0, done).get(tok)
            if nxt is not None:
                return self._enter(nxt, w, done), w - acc
        return (0, 0.0, done), -acc

    def bonus(self, states, vocab: int, device, dtype) -> Tensor:
        rows = torch.zeros(len(states), vocab, dtype=dtype)
        for r, (node, acc, done) in enumerate(states):
            if acc:
                rows[r] -= acc
            if node != 0:
                for tok in self._open(0, done):
                    rows[r, tok] = self.weight - acc
            for tok in self._open(node, done):
                rows[r, tok] = self.weight
        return rows.to(device)

    def advance(self, states, prev_rows: List[int], tokens: List[int]):
        return [self.step(states[p], t)[0] for p, t in zip(prev_rows, tokens)]


def set_hotwords(tokenizer, words: List[str], weight: float) -> int:
    """設定這次請求的熱詞，回傳實際用上的詞數。含 <unk> 的詞（字典外的字）直接略過。"""
    global _current
    seqs = []
    unk = tokenizer.dict.unkid
    for w in words:
        w = w.strip()
        if not w:
            continue
        _, ids = tokenizer.tokenize(w)
        if ids and unk not in ids:
            seqs.append(ids)
    _current = ContextBias(seqs, weight) if seqs and weight > 0 else None
    return len(seqs)


def install() -> None:
    from fireredasr2s.fireredasr2.models.module.transformer_decoder import TransformerDecoder
    TransformerDecoder.batch_beam_search = _batch_beam_search


def _batch_beam_search(self, encoder_outputs, src_masks,
               beam_size=1, nbest=1, decode_max_len=0,
               softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
               elm=None, elm_weight=0.0):
    B = beam_size
    N, Ti, H = encoder_outputs.size()
    device = encoder_outputs.device
    maxlen = decode_max_len if decode_max_len > 0 else Ti
    assert eos_penalty > 0.0

    # Init
    encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
    src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
    ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
    t_ys = ys.clone()
    confidences = torch.zeros(N*B, 1).float().to(device)
    caches: List[Optional[Tensor]] = []
    for _ in range(self.n_layers):
        caches.append(None)
    scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
    scores = scores.repeat(N).view(N*B, 1)
    is_finished = torch.zeros_like(scores)
    if elm is not None:
        elm_cache = elm.init_hidden(encoder_outputs, N*B)
    bias = _current  # [bias]
    bias_states = [bias.initial()] * (N*B) if bias is not None else None

    # Autoregressive Prediction
    for t in range(maxlen):
        tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)

        dec_output = self.dropout(
            self.tgt_word_emb(ys) * self.scale +
            self.positional_encoding(ys))
        i = 0
        for dec_layer in self.layer_stack:
            dec_output = dec_layer.forward(
                dec_output, encoder_outputs,
                tgt_mask, src_mask,
                cache=caches[i])
            caches[i] = dec_output
            i += 1

        dec_output = self.layer_norm_out(dec_output)

        t_logit = self.tgt_word_prj(dec_output[:, -1])
        t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

        if elm is not None and elm_weight > 0.0:
            elm_logit, elm_cache = elm.forward_model(t_ys, hidden=elm_cache)
            t_lm_scores = torch.log_softmax(elm_logit[:, -1], dim=-1) * (1 - is_finished.float())
            t_lm_scores[:, elm.eos_id] *= 3
            t_scores = t_scores + elm_weight * t_lm_scores

        if eos_penalty != 1.0:
            t_scores[:, self.eos_id] *= eos_penalty

        # [bias] 加在 eos_penalty 之後，信心值用扣掉加分的原始分數算（加分會讓 exp 超過 1）。
        t_bias = None
        if bias is not None:
            t_bias = bias.bonus(bias_states, t_scores.size(1), t_scores.device, t_scores.dtype)
            t_scores = t_scores + t_bias

        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
        t_topB_raw = t_topB_scores if t_bias is None else t_topB_scores - torch.gather(t_bias, 1, t_topB_ys)
        t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
        t_topB_raw = self.set_finished_beam_score_to_zero(t_topB_raw, is_finished)
        t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

        # Accumulated
        scores = scores + t_topB_scores

        # Pruning
        scores = scores.view(N, B*B)
        scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
        scores = scores.view(-1, 1)

        topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
        stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B).to(device)
        topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

        # Update ys
        ys = ys[topB_row_number_in_ys]
        t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
        ys = torch.cat((ys, t_ys), dim=1)

        # Update confidences
        confidences = confidences[topB_row_number_in_ys]
        t_confidences = torch.gather(t_topB_raw.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
        t_confidences = torch.exp(t_confidences)
        assert torch.all(t_confidences <= 1.0)
        assert torch.all(t_confidences >= 0.0)
        confidences = torch.cat((confidences, t_confidences), dim=1)

        # [bias] 狀態跟著存活的 beam 重排
        if bias is not None:
            bias_states = bias.advance(bias_states, topB_row_number_in_ys.tolist(), t_ys.view(-1).tolist())

        # Update caches
        new_caches: List[Optional[Tensor]] = []
        for cache in caches:
            if cache is not None:
                new_caches.append(cache[topB_row_number_in_ys])
        caches = new_caches
        if elm and elm_weight > 0.0:
            elm_cache = (elm_cache[0][:, topB_row_number_in_ys], elm_cache[1][:, topB_row_number_in_ys])

        # Update finished state
        is_finished = t_ys.eq(self.eos_id)
        if is_finished.sum().item() == N*B:
            break

    # Length penalty (follow GNMT)
    scores = scores.view(N, B)
    ys = ys.view(N, B, -1)
    ys_lengths = self.get_ys_lengths(ys)
    if length_penalty > 0.0:
        penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
        scores /= penalty
    nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
    nbest_scores = -1.0 * nbest_scores
    index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
    nbest_ys = ys.view(N*B, -1)[index.view(-1)]
    nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
    nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)
    nbest_confidences = confidences.view(N*B, -1)[index.view(-1)].view(N, nbest_ids.size(1), -1)

    nbest_hyps: List[List[Dict[str, Tensor]]] = []
    for n in range(N):
        n_nbest_hyps: List[Dict[str, Tensor]] = []
        for i, score in enumerate(nbest_scores[n]):
            confidence = nbest_confidences[n, i, 1:nbest_ys_lengths[n, i]]
            confidence = confidence.mean()
            new_hyp = {
                "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]],
                "confidence": confidence
            }
            n_nbest_hyps.append(new_hyp)
        nbest_hyps.append(n_nbest_hyps)
    return nbest_hyps


def restore_case(text: str, words: List[str]) -> str:
    """FireRed 把輸出全轉小寫（asr.py 的 text.lower()），英文熱詞照字典的大小寫寫回去。

    notes: 只處理含英文字母的詞，比對忽略大小寫與詞內空白（"to do"、"todo" 都對到 "TO DO"）。
           詞前後要不是英數字才算，避免把 "session" 換進 "sessions"。
    """
    import re
    for w in words:
        w = w.strip()
        if not re.search(r"[A-Za-z]", w):
            continue
        letters = [re.escape(c) for c in w if not c.isspace()]
        pat = r"(?<![A-Za-z0-9])" + r"\s*".join(letters) + r"(?![A-Za-z0-9])"
        text = re.sub(pat, w, text, flags=re.IGNORECASE)
    return text
