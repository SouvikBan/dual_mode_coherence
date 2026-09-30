#!/usr/bin/env python3
"""Branch-safe seeded inference for CorPipe26 two-stage models.

GOLD_SILVER_SENTENCE_SPECS_FIX = 2026-08-12

CorPipe's released inference loop already compares each current mention with
earlier mention positions, but it discards input Entity annotations at test
time. This wrapper changes only that inference path:

1. annotate a context once with ordinary CorPipe;
2. preload those silver mention spans and entity ids as antecedent candidates;
3. decode each mutually exclusive continuation independently;
4. inherit a context entity id when a focus mention links to any seeded mention,
   otherwise allocate a branch-local new id.

No POS/NP pre-filter is applied, so verbal, clausal and other non-nominal spans
that CorPipe detects remain eligible (including discourse-deictic chains).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

from textproc import TextProcessor


def decode_entity_ids(scores: Sequence[Sequence[float]], previous_ids: Sequence[int],
                      current_positions: Sequence[Sequence[int]],
                      next_entity_id: int) -> tuple[list[int], int]:
    """Decode CorPipe antecedent rows while preserving seeded cluster ids.

    This pure helper mirrors the released inference rule and is unit-testable
    without torch/transformers. Each row contains previous candidates, earlier
    current mentions, and the current mention's self/new candidate.
    """
    refs = list(previous_ids)
    assigned: list[int] = []
    n_prev = len(previous_ids)
    for i, row_values in enumerate(scores):
        values = list(row_values[:n_prev + i + 1])
        # CorPipe prevents same-start nested mentions from anteceding one another.
        j = i - 1
        while j >= 0 and current_positions[j][0] == current_positions[i][0]:
            values[n_prev + j] = values[n_prev + i] - 1.0
            j -= 1
        antecedent = max(range(len(values)), key=values.__getitem__)
        if antecedent == n_prev + i:
            refs.append(next_entity_id)
            assigned.append(next_entity_id)
            next_entity_id += 1
        else:
            refs.append(refs[antecedent])
            assigned.append(refs[antecedent])
    return assigned, next_entity_id


def antecedent_link_costs(scores, previous_ids, current_positions):
    """Per-mention link cost from CorPipe's own antecedent scores.

    Each row of `scores` scores the candidates for one mention of the current
    utterance: the seeded context mentions, then the earlier mentions of this
    utterance, then the mention itself (the "start a new entity" option). The
    decoder takes the argmax; softmaxing the same row turns it into a
    distribution over where the mention could attach, so

        link_cost = -log2 P(the antecedent the decoder chose)

    is how unsure CorPipe was about that attachment, in bits. A pronoun with one
    obvious antecedent costs near 0; one the model cannot place costs more. The
    entropy of the whole row is reported too, as it does not depend on which
    candidate won.

    Masking follows the decoder exactly, so the distribution is over the same
    candidate set the decision was made on.
    """
    import math
    n_prev = len(previous_ids)
    out = []
    for i, row_values in enumerate(scores):
        values = [float(v) for v in row_values[:n_prev + i + 1]]
        j = i - 1
        while j >= 0 and current_positions[j][0] == current_positions[i][0]:
            values[n_prev + j] = values[n_prev + i] - 1.0
            j -= 1
        top = max(values)
        weights = [math.exp(v - top) for v in values]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        chosen = max(range(len(values)), key=values.__getitem__)
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        out.append({
            "link_cost_bits": -math.log2(max(probabilities[chosen], 1e-12)),
            "link_entropy_bits": entropy,
            "link_probability": probabilities[chosen],
            "n_candidates": len(values),
            "is_new_entity": int(chosen == n_prev + i),
            "antecedent_in_context": int(chosen < n_prev),
        })
    return out


def gold_link_costs(scores, previous_ids, current_ids, current_positions):
    """How much probability CorPipe puts on the GOLD antecedent of each mention.

    For a mention whose gold cluster is c, the correct candidates are every
    earlier mention of cluster c -- in the frozen context or earlier in this
    same utterance -- and, if c has never been mentioned before, the "start a
    new entity" option. Summing the softmax over that set gives P(gold link),
    and

        gold_link_cost = -log2 P(gold link)

    is the model's surprisal at the coreference decision the annotation
    actually makes, the same shape as token surprisal. It is low when the model
    would have resolved the mention the way the gold does and high when it
    would not, whatever it happened to choose itself.

    decoder_agrees records whether CorPipe's own argmax landed inside the gold
    set, so the measure can be split into the cases where the model was right
    and the cases where it was not.
    """
    import math
    n_prev = len(previous_ids)
    out = []
    for i, row_values in enumerate(scores):
        values = [float(v) for v in row_values[:n_prev + i + 1]]
        j = i - 1
        while j >= 0 and current_positions[j][0] == current_positions[i][0]:
            values[n_prev + j] = values[n_prev + i] - 1.0
            j -= 1
        top = max(values)
        weights = [math.exp(v - top) for v in values]
        total = sum(weights)
        probabilities = [w / total for w in weights]

        cluster = current_ids[i]
        gold = [k for k in range(n_prev) if previous_ids[k] == cluster]
        gold += [n_prev + k for k in range(i) if current_ids[k] == cluster]
        gold_is_new = 0
        if not gold:                       # first mention of this entity anywhere
            gold = [n_prev + i]
            gold_is_new = 1
        mass = sum(probabilities[k] for k in gold)
        chosen = max(range(len(values)), key=values.__getitem__)
        out.append({
            "gold_link_cost_bits": -math.log2(max(mass, 1e-12)),
            "gold_link_probability": mass,
            "gold_is_new": gold_is_new,
            "n_gold_candidates": len(gold),
            "n_candidates": len(values),
            "decoder_agrees": int(chosen in gold),
            "decoder_cost_bits": -math.log2(max(probabilities[chosen], 1e-12)),
        })
    return out


def _load_upstream(path: str | Path) -> ModuleType:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CorPipe source not found: {path}. Clone ufal/crac2026-corpipe and "
            "pass --corpipe-source path/to/corpipe26_twostage.py."
        )
    spec = importlib.util.spec_from_file_location("corpipe26_twostage_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CorPipe source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class StartedCorPipeRuntime:
    """An upstream CorPipe module whose minnt runtime is already initialized."""

    module: ModuleType
    seed: int
    threads: int


def start_corpipe_runtime(corpipe_source: str | Path, seed: int = 42,
                          threads: int = 2) -> StartedCorPipeRuntime:
    """Start minnt before libraries such as Stanza choose an MP context.

    minnt deliberately selects Python's ``forkserver`` start method. Python
    permits that choice only once per process, so this must happen before a
    Stanza pipeline (or any other multiprocessing user) is constructed.
    """
    module = _load_upstream(corpipe_source)
    try:
        module.minnt.startup(seed, threads)
    except RuntimeError as exc:
        if "context has already been set" not in str(exc):
            raise
        raise RuntimeError(
            "minnt could not select the forkserver multiprocessing context. "
            "Call start_corpipe_runtime() before constructing Stanza or other "
            "Torch/multiprocessing pipelines; do not force-reset a live CUDA "
            "process. smoke_test.py does this automatically."
        ) from exc
    return StartedCorPipeRuntime(module, seed, threads)


def _conllu_escape(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _mention_head(tokens: Sequence[dict], start: int, end: int) -> int:
    """Choose the span token whose dependency governor lies outside the span."""
    token_ids = {int(tokens[i].get("id", i + 1)) for i in range(start, end + 1)}
    candidates = [
        i for i in range(start, end + 1)
        if int(tokens[i].get("head", 0)) not in token_ids
    ]
    if not candidates:
        candidates = list(range(start, end + 1))
    non_punctuation = [
        i for i in candidates
        if tokens[i].get("upos") != "PUNCT"
        and tokens[i].get("xpos") not in {".", ",", ":", "-LRB-", "-RRB-"}
    ]
    return (non_punctuation or candidates)[0]


def _mention_form(token: dict) -> str:
    upos, xpos = str(token.get("upos", "")), str(token.get("xpos", ""))
    if upos == "PRON" or xpos.startswith("PRP") or xpos in {"WP", "WP$"}:
        return "pronoun"
    if upos == "PROPN" or xpos in {"NNP", "NNPS"}:
        return "proper"
    if upos == "NOUN" or xpos in {"NN", "NNS"}:
        return "common"
    return "other"


class SeededCorPipe:
    """Load CorPipe once and expose context + independent-branch annotation."""

    def __init__(self, corpipe_source: str | Path, model_name: str,
                 text_processor: TextProcessor | None = None,
                 segment: int = 2560, device: str = "auto",
                 threads: int = 2, seed: int = 42,
                 runtime: StartedCorPipeRuntime | None = None) -> None:
        self.text = text_processor or TextProcessor()
        runtime = runtime or start_corpipe_runtime(corpipe_source, seed, threads)
        if runtime.seed != seed or runtime.threads != threads:
            raise ValueError(
                "the started CorPipe runtime seed/threads differ from the "
                "SeededCorPipe constructor arguments"
            )
        self.cp = runtime.module

        resolved = (model_name if os.path.isdir(model_name)
                    else self.cp.huggingface_hub.snapshot_download(model_name))
        with open(os.path.join(resolved, "options.json"), encoding="utf-8") as stream:
            options = json.load(stream)
        options.update({
            "load": [resolved],
            "segment": segment,
            # Upstream treats any non-None value (including 0) as permission to
            # append right context, so None is required for strict prefix-only
            # features and no future-text leakage.
            "right": None,
            "batch_size": 1,
            "seed": seed,
            "threads": threads,
            "label_smoothing": options.get("label_smoothing", 0.0),
        })
        self.args = argparse.Namespace(**options)

        encoder = self.args.encoder
        if "t5gemma" in encoder:
            tokenizer_name = "google/t5gemma-l-l-ul2"
        elif "umt5" in encoder:
            tokenizer_name = "google/umt5-xl"
        elif "mt5" in encoder:
            tokenizer_name = "google/mt5-xl"
        else:
            tokenizer_name = encoder
        self.tokenizer = self.cp.transformers.AutoTokenizer.from_pretrained(
            tokenizer_name, legacy=False
        )
        special = [self.cp.Dataset.TOKEN_EMPTY]
        if self.tokenizer.cls_token_id is None:
            special.append(self.cp.Dataset.TOKEN_CLS)
        self.tokenizer.add_special_tokens({"additional_special_tokens": special})

        with open(os.path.join(resolved, "tags.txt"), encoding="utf-8") as stream:
            self.tags = [line.rstrip("\r\n") for line in stream]
        self.tags_map = {tag: i for i, tag in enumerate(self.tags)}
        self.model = self.cp.Model(self.tokenizer, self.tags, self.args)
        self.model.load_weights(os.path.join(resolved, "model.pt"))

        torch = self.cp.torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        # opt-in: keep the antecedent distribution of the last decoded utterance
        self.collect_link_costs = False
        self.last_link_costs: list[dict] = []

    def _sentence_specs(self, sentences: Sequence[str | dict]) -> list[dict]:
        specs = []
        for sentence_i, sentence in enumerate(sentences):
            if isinstance(sentence, dict):
                spec = {
                    **sentence,
                    "tokens": [dict(token) for token in sentence["tokens"]],
                }
            else:
                tokens = self.text.tokens(sentence)
                spec = {
                    "text": sentence,
                    "grammar_source": self.text.backend,
                    "tokens": [
                        {
                            "id": i + 1,
                            "text": token.text,
                            "lemma": token.lemma,
                            "upos": token.upos,
                            "xpos": token.xpos,
                            "feats": "_",
                            "head": token.head,
                            "deprel": token.deprel,
                            "deps": "_",
                            "misc": "_",
                            "char_start": token.start,
                            "char_end": token.end,
                            "grammar_source": self.text.backend,
                        }
                        for i, token in enumerate(tokens)
                    ],
                }
            if not spec["tokens"]:
                raise ValueError(f"sentence {sentence_i} tokenized to zero tokens")
            specs.append(spec)
        return specs

    @staticmethod
    def _write_conllu(path: Path, sentences: Sequence[dict], doc_id: str) -> None:
        with path.open("w", encoding="utf-8") as stream:
            stream.write(f"# newdoc id = {_conllu_escape(doc_id)}\n")
            for sent_i, sentence in enumerate(sentences, 1):
                tokens = sentence["tokens"]
                stream.write(f"# sent_id = {doc_id}-{sent_i}\n")
                stream.write(f"# text = {_conllu_escape(sentence['text'])}\n")
                fallback_tree = all(int(token.get("head", 0)) == 0 for token in tokens)
                for i, token in enumerate(tokens, 1):
                    misc = []
                    if (i < len(tokens) and
                            int(token["char_end"]) == int(tokens[i]["char_start"])):
                        misc.append("SpaceAfter=No")
                    misc.extend((f"StartChar={token['char_start']}",
                                 f"EndChar={token['char_end']}"))
                    head = (0 if i == 1 else 1) if fallback_tree else int(token["head"])
                    deprel = (("root" if i == 1 else "dep") if fallback_tree
                              else str(token["deprel"]).lower())
                    fields = [
                        str(i), _conllu_escape(token["text"]),
                        _conllu_escape(token.get("lemma", "_")),
                        token.get("upos", "X"), token.get("xpos", "_"),
                        token.get("feats", "_"), str(head), deprel,
                        "_", "|".join(misc),
                    ]
                    stream.write("\t".join(fields) + "\n")
                stream.write("\n")

    def _dataset(self, sentences: Sequence[str | dict], doc_id: str):
        specs = self._sentence_specs(sentences)
        tmp = tempfile.TemporaryDirectory(prefix="corpipe-seeded-")
        path = Path(tmp.name) / f"{doc_id}.conllu"
        self._write_conllu(path, specs, doc_id)
        dataset = self.cp.Dataset(str(path), self.tokenizer)
        examples = dataset.dataset(self.tags_map, False, self.args)
        if len(dataset.docs) != 1 or len(examples) != len(specs):
            tmp.cleanup()
            raise RuntimeError("unexpected CorPipe document/sentence alignment")
        return tmp, dataset, examples, specs

    @staticmethod
    def _mention_json(sentence: str, tokens: Sequence[dict],
                      mentions: Sequence[Sequence[int]],
                      context_ids: set[int] | None = None) -> list[dict]:
        output = []
        for start, end, entity_id in mentions:
            if not (0 <= start <= end < len(tokens)):
                raise ValueError(f"invalid mention token span {(start, end)}")
            char_start = int(tokens[start]["char_start"])
            char_end = int(tokens[end]["char_end"])
            head_index = _mention_head(tokens, start, end)
            head = tokens[head_index]
            mention = {
                "token_start": int(start),
                "token_end": int(end),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "text": sentence[char_start:char_end],
                "cluster_id": int(entity_id),
                "head_token_index": int(head_index),
                "head_text": head["text"],
                "head_lemma": head.get("lemma", "_"),
                "head_upos": head.get("upos", "X"),
                "head_xpos": head.get("xpos", "_"),
                "deprel": head.get("deprel", "dep"),
                "grammatical_role": head.get("grammatical_role", "other"),
                "mention_form": _mention_form(head),
                "grammar_source": head.get("grammar_source"),
            }
            if context_ids is not None:
                mention["information_status"] = (
                    "given" if int(entity_id) in context_ids else "new"
                )
            output.append(mention)
        return output

    @staticmethod
    def _sentence_json(index: int, sentence: dict,
                       mentions: Sequence[Sequence[int]],
                       context_ids: set[int] | None = None) -> dict:
        output = {
            "sentence_index": index,
            "text": sentence["text"],
            "grammar_source": sentence.get("grammar_source"),
            "tokens": [dict(token) for token in sentence["tokens"]],
            "mentions": SeededCorPipe._mention_json(
                sentence["text"], sentence["tokens"], mentions, context_ids
            ),
        }
        for key in ("source_story_id", "source_sentence_index"):
            if key in sentence:
                output[key] = sentence[key]
        return output

    def annotate_document(self, sentences: Sequence[str | dict],
                          doc_id: str = "context") -> list[dict]:
        """Ordinary CorPipe pass used once to obtain the silver context state."""
        if not sentences:
            return []
        tmp, dataset, examples, specs = self._dataset(sentences, doc_id)
        try:
            dataloader = self.cp.torch.utils.data.DataLoader(
                examples, batch_size=1, collate_fn=self.cp.Dataset.padded_batch(False)
            )
            predicted = self.model.predict(dataset, dataloader)
            if len(predicted) != len(sentences):
                raise RuntimeError("CorPipe returned a different sentence count")
            return [
                self._sentence_json(i, sentence, mentions)
                for i, (sentence, mentions) in enumerate(zip(specs, predicted))
            ]
        finally:
            tmp.cleanup()

    @staticmethod
    def _seed_absolute(dataset, context: Sequence[dict]) -> tuple[list[list[int]], int]:
        seeds: list[list[int]] = []
        absolute = 0
        for doc_sentence, annotation in zip(dataset.docs[0], context):
            subwords, word_indices, _, _ = doc_sentence
            for mention in annotation.get("mentions", []):
                start_word = int(mention["token_start"])
                end_word = int(mention["token_end"])
                start = absolute + word_indices[start_word]
                # CorPipe represents a mention by the first subword of its
                # first and last token (matching upstream Dataset/predict).
                end = absolute + word_indices[end_word]
                seeds.append([start, end, int(mention["cluster_id"])])
            absolute += len(subwords)
        return seeds, absolute

    def _predict_focus(self, example, seed_mentions: list[list[int]],
                       doc_subwords: int, next_entity_id: int
                       ) -> tuple[list[list[int]], int, int]:
        torch = self.cp.torch
        batch = self.cp.Dataset.padded_batch(False)([example])
        b_subwords, b_word_indices = (tensor.to(self.device) for tensor in batch)

        with torch.inference_mode():
            embeddings, logits = self.model(b_subwords, b_word_indices)
            tags = self.model.decode_mentions(logits, b_word_indices[:, :-1] >= 0)
            del logits

            word_indices = b_word_indices[0].numpy(force=True)
            word_indices = word_indices[word_indices >= 0]
            tag_ids = tags[0].numpy(force=True)
            tag_ids = tag_ids[b_word_indices[0, 1:].numpy(force=True) >= 0]

            mentions, stack = [], []
            for word_i, tag in enumerate(
                    self.tags[tag % len(self.tags)] for tag in tag_ids):
                for command in tag.split(","):
                    if command == "PUSH":
                        stack.append(word_i)
                    elif command.startswith("POP:"):
                        depth = int(command.removeprefix("POP:"))
                        if stack:
                            depth = len(stack) - (depth if depth <= len(stack) else 1)
                            mentions.append((stack.pop(depth), word_i))
                    elif command:
                        raise ValueError(f"unknown CorPipe stack command: {command}")
            while stack:
                mentions.append((stack.pop(), len(tag_ids) - 1))
            mentions = sorted(set(mentions), key=lambda span: (span[0], -span[1]))
            if not mentions:
                return [], next_entity_id, int(word_indices[-1] - word_indices[0])

            offset = doc_subwords - (int(word_indices[0]) - 2)
            visible = [seed for seed in seed_mentions if seed[0] >= offset]
            previous_positions = [
                [start - offset + 1, end - offset + 1]
                for start, end, _ in visible
            ]
            current_positions = [
                [int(word_indices[start]), int(word_indices[end])]
                for start, end in mentions
            ]
            all_positions = previous_positions + current_positions
            scores = self.model.compute_antecedents(
                embeddings,
                torch.as_tensor(all_positions, dtype=torch.int64, device=self.device)
                    .view(1, -1, 2),
                torch.as_tensor(current_positions, dtype=torch.int64, device=self.device)
                    .view(1, -1, 2),
            )[0].numpy(force=True)

        previous_ids = [seed[2] for seed in visible]
        if getattr(self, "collect_link_costs", False):
            self.last_link_costs = antecedent_link_costs(
                scores, previous_ids, current_positions)
        entity_ids, next_entity_id = decode_entity_ids(
            scores, previous_ids, current_positions, next_entity_id
        )
        output = [
            [int(start), int(end), int(entity_id)]
            for (start, end), entity_id in zip(mentions, entity_ids)
        ]
        consumed = int(word_indices[-1] - word_indices[0])
        return output, next_entity_id, consumed

    def score_gold_links(self, context: Sequence[dict], focus: Sequence[dict],
                         job_id: str = "gold") -> list[list[dict]]:
        """Per-mention gold link cost for each focus sentence.

        The mention spans scored are the GOLD ones, not CorPipe's own: they are
        passed to compute_antecedents directly. Seeds advance with the gold
        mentions too, so each sentence is scored against the true prior
        coreference state rather than against the model's running guess.
        """
        torch = self.cp.torch
        sentences = list(context) + list(focus)
        tmp, dataset, examples, specs = self._dataset(sentences, job_id)
        try:
            seed_mentions, doc_subwords = self._seed_absolute(dataset, context)
            per_sentence = []
            for local_i, example in enumerate(examples[len(context):]):
                gold = sorted(focus[local_i].get("mentions", []),
                              key=lambda m: (int(m["token_start"]), -int(m["token_end"])))
                batch = self.cp.Dataset.padded_batch(False)([example])
                b_subwords, b_word_indices = (t.to(self.device) for t in batch)
                with torch.inference_mode():
                    embeddings, logits = self.model(b_subwords, b_word_indices)
                    del logits
                    word_indices = b_word_indices[0].numpy(force=True)
                    word_indices = word_indices[word_indices >= 0]
                    rows = []
                    if gold:
                        offset = doc_subwords - (int(word_indices[0]) - 2)
                        visible = [seed for seed in seed_mentions if seed[0] >= offset]
                        previous_positions = [[s[0] - offset + 1, s[1] - offset + 1]
                                              for s in visible]
                        current_positions = [
                            [int(word_indices[int(m["token_start"])]),
                             int(word_indices[int(m["token_end"])])] for m in gold]
                        all_positions = previous_positions + current_positions
                        scores = self.model.compute_antecedents(
                            embeddings,
                            torch.as_tensor(all_positions, dtype=torch.int64,
                                            device=self.device).view(1, -1, 2),
                            torch.as_tensor(current_positions, dtype=torch.int64,
                                            device=self.device).view(1, -1, 2),
                        )[0].numpy(force=True)
                        rows = gold_link_costs(scores, [s[2] for s in visible],
                                               [int(m["cluster_id"]) for m in gold],
                                               current_positions)
                per_sentence.append([{**m, **row} for m, row in zip(gold, rows)])
                # advance the state with the GOLD mentions
                doc_sentence = dataset.docs[0][len(context) + local_i]
                _, sentence_word_indices, _, _ = doc_sentence
                for m in gold:
                    seed_mentions.append([
                        doc_subwords + sentence_word_indices[int(m["token_start"])],
                        doc_subwords + sentence_word_indices[int(m["token_end"])],
                        int(m["cluster_id"])])
                doc_subwords += int(word_indices[-1] - word_indices[0])
            return per_sentence
        finally:
            tmp.cleanup()

    def annotate_branches(self, context: Sequence[dict], branches: Sequence[dict],
                          job_id: str = "job") -> list[dict]:
        """Decode branches independently against exactly the same context state."""
        context_sentences = list(context)
        context_ids = {
            int(mention["cluster_id"])
            for sentence in context for mention in sentence.get("mentions", [])
        }
        base_next_id = max(context_ids, default=0) + 1
        outputs = []
        for branch_i, branch in enumerate(branches):
            focus_sentences = branch.get("grammatical_sentences")
            if not focus_sentences:
                focus_texts = self.text.sentences(branch["text"]) or [branch["text"]]
                focus_sentences = self._sentence_specs(focus_texts)
            sentences = context_sentences + focus_sentences
            tmp, dataset, examples, specs = self._dataset(
                sentences, f"{job_id}-branch-{branch_i}"
            )
            try:
                for expected, actual in zip(context, specs[:len(context)]):
                    if ([t["text"] for t in expected["tokens"]] !=
                            [t["text"] for t in actual["tokens"]]):
                        raise RuntimeError("context tokenization changed between silver and branch pass")
                seed_mentions, doc_subwords = self._seed_absolute(dataset, context)
                next_entity_id = base_next_id
                focus_annotations = []
                for local_i, example in enumerate(examples[len(context):]):
                    mentions, next_entity_id, consumed = self._predict_focus(
                        example, seed_mentions, doc_subwords, next_entity_id
                    )
                    doc_sentence = dataset.docs[0][len(context) + local_i]
                    subwords, word_indices, _, _ = doc_sentence
                    for start, end, entity_id in mentions:
                        abs_start = doc_subwords + word_indices[start]
                        abs_end = doc_subwords + word_indices[end]
                        seed_mentions.append([abs_start, abs_end, entity_id])
                    focus_annotations.append(self._sentence_json(
                        local_i,
                        specs[len(context) + local_i],
                        mentions,
                        context_ids,
                    ))
                    doc_subwords += consumed

                branch_output = dict(branch)
                branch_output.pop("grammatical_sentences", None)
                branch_output["sentences"] = focus_annotations
                branch_output["context_cluster_ids"] = sorted(context_ids)
                branch_output["n_given_mentions"] = sum(
                    m["information_status"] == "given"
                    for sentence in focus_annotations for m in sentence["mentions"]
                )
                branch_output["n_new_mentions"] = sum(
                    m["information_status"] == "new"
                    for sentence in focus_annotations for m in sentence["mentions"]
                )
                outputs.append(branch_output)
            finally:
                tmp.cleanup()
        return outputs

    def annotate_job(self, job: dict) -> dict:
        context_sentences = job.get("context_sentences") or self.text.sentences(job["context"])
        context = self.annotate_document(context_sentences, f"{job['id']}-context")
        output = dict(job)
        output["context"] = context
        output["branches"] = self.annotate_branches(context, job["branches"], job["id"])
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpipe-source", required=True,
                        help="path to upstream corpipe26_twostage.py")
    parser.add_argument("--model", default="ufal/corpipe26-twostage-corefud1.4-large-260702")
    parser.add_argument("--input", required=True,
                        help="JSON manifest: {jobs:[{id,context,branches:[...]}]}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment", type=int, default=2560)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    processor = TextProcessor(args.spacy_model)
    annotator = SeededCorPipe(
        args.corpipe_source, args.model, processor, args.segment,
        args.device, args.threads,
    )
    output = {
        "schema_version": 1,
        "model": args.model,
        "sentence_backend": processor.backend,
        "jobs": [annotator.annotate_job(job) for job in manifest["jobs"]],
    }
    with Path(args.output).open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(f"wrote {len(output['jobs'])} seeded jobs to {args.output}")


if __name__ == "__main__":
    main()