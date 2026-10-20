#!/usr/bin/env python3
"""Natural Stories word-level surprisal for the word-level localisation of Appendix A.

Pythia-70m-deduped at training step 1,000 (the paper's reading-time surprisal model), each story
scored as one running text with the full left context inside the story. The surprisal of a zone
(whitespace word) is the sum of the surprisals of its sub-tokens, in bits, with leading-space
tokenisation (" word") for every word but the first.

Usage: word_surprisal.py <all_stories.tok (word, zone, item)> <out_csv>
"""

import csv
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-70m-deduped"
REVISION = "step1000"


def load_words(path):
    words = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            words.setdefault(int(row["item"]), []).append((int(row["zone"]), row["word"]))
    return {item: sorted(zones) for item, zones in words.items()}


def score_story(tokenizer, model, device, zones):
    ids = [tokenizer.bos_token_id]
    spans = []
    for position, (_, word) in enumerate(zones):
        pieces = tokenizer.encode((" " if position else "") + word, add_special_tokens=False)
        spans.append((len(ids), len(ids) + len(pieces)))
        ids += pieces
    inputs = torch.tensor([ids], device=device)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs).logits[0, :-1].float(), dim=-1)
    targets = inputs[0, 1:]
    token_log_prob = log_probs[torch.arange(len(targets)), targets]   # log p(token_t | tokens_<t)
    for (zone, word), (start, end) in zip(zones, spans):
        bits = float(-token_log_prob[start - 1:end - 1].sum() / math.log(2))
        yield {"zone": zone, "word": word, "n_subtokens": end - start, "surprisal_bits": bits}


def main(words_tsv, out_csv):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    rows = []
    for item, zones in sorted(load_words(words_tsv).items()):
        for row in score_story(tokenizer, model, device, zones):
            rows.append({"item": item, **row})
        print("story", item, flush=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out_csv, len(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
