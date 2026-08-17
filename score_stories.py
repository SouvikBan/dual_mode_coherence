from argparse import ArgumentParser
from minicons import scorer
import numpy as np
import csv
from collections import defaultdict
from itertools import chain
import torch
from tqdm import tqdm

from typing import List, Tuple


device = "mps" if torch.mps.is_available() else "cpu"
print(f"Using device: {device}")

NATURAL_STORIES_TOK_FILE = "naturalstories/naturalstories_RTS/all_stories.tok"


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--model-name-or-path", type=str, default="openai-community/gpt2")
	parser.add_argument("--window-size", type=int, default=128)
	parser.add_argument("--stride", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--out-file", type=str, default="naturalstories/naturalstories_RTS/stories_surprisal.tsv")
	return parser.parse_args()


def _compile_stimuli(story_words: List[str], window_size: int, stride: int, tokenizer) -> List[str]:
	stimuli = []
	for current_idx, current_word in enumerate(story_words):
		current_word_tokens = tokenizer.encode(f" {current_word}")
		stimulus = [current_word_tokens]
		num_context_tokens = 0
		for context_idx in range(current_idx-1, np.max(current_idx-window_size,0), -stride):
			# first token needs no additional context
			if context_idx < 0: break
			# add whitespace for proper tokenization
			word_tokens = tokenizer.encode(f" {story_words[context_idx]}")
			stimulus.insert(0, word_tokens)
			num_context_tokens += len(word_tokens)
			if num_context_tokens >= window_size:
				break
		stimulus.insert(0, [tokenizer.bos_token_id])
		stimuli.append(list(chain.from_iterable(stimulus)))
	return stimuli


def _score(stimuli: List[str], scorer: scorer.IncrementalLMScorer, batch_size: int) -> List[Tuple[str, float]]:
	scores = []
	for idx in range(batch_size, len(stimuli)+batch_size, batch_size):
		batch_stimuli = stimuli[idx-batch_size:idx]
		batch_scores = scorer.token_score(
			batch=batch_stimuli,
			base_two=True,
			bow_correction=True
		)
		scores.extend(batch_scores)
	return scores


def _get_word_score(token_scores: List[Tuple[str, float]]) -> float:
	"""Assumes a GPT-2-style tokenizer, i.e., 'Ġ' is expected to mark the beginning of words"""
	word_scores = []
	current_word_score = 0.
	for idx in range(len(token_scores)):
		current_word_score += token_scores[idx][-1]
		if idx == len(token_scores)-1 or token_scores[idx+1][0].startswith("Ġ"):
			word_scores.append(current_word_score)
			current_word_score = 0.0
	# we are only interested int the close surprisal
	return -1*word_scores[-1]


def main():

	args = parse_args()

	ilm_scorer = scorer.IncrementalLMScorer(model=args.model_name_or_path, device=device)
	tokenizer = ilm_scorer.tokenizer

	csv_header = ["word", "story_id", "story_position", "sentence_id", "sentence_position", "surprisal"]
	f_out = open(args.out_file, "w")
	csvwriter = csv.writer(f_out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
	csvwriter.writerow(csv_header)
		
	with open(NATURAL_STORIES_TOK_FILE, "r") as f:
		stories = f.read().split("\n")[1:-1]

	words_by_story = defaultdict(list)
	for line in stories:
		word, _, story_id = line.split("\t")
		words_by_story[story_id].append(word)

	for story_id, story_words in tqdm(words_by_story.items()):
		# tokenize and compile stimuli with at least window_size tokens in the context
		stimuli = _compile_stimuli(story_words, args.window_size, args.stride, tokenizer)
		# detokenize
		stimuli = [tokenizer.decode(stimulus) for stimulus in stimuli]
		token_scores = _score(stimuli, scorer=ilm_scorer, batch_size=args.batch_size)
		word_scores = [_get_word_score(stimulus_score[1:]) for stimulus_score in token_scores]
		# make sure no token was lost on the way
		assert len(word_scores) == len(story_words), len(story_words)-len(word_scores)

		# save with sentence ids and sentence positions
		sentence_id = 1
		sentence_position = 1
		for story_position, (word, score) in enumerate(zip(story_words, word_scores), start=1):
			csvwriter.writerow([word, story_id, story_position, sentence_id, sentence_position, score])
			sentence_position += 1
			if word.endswith("."):
				sentence_id += 1
				sentence_position = 1


if __name__ == "__main__":
	main()