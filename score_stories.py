# from argparse import ArgumentParser
from minicons import scorer
import numpy as np
import csv
from collections import defaultdict
# from itertools import chain
import torch
from tqdm import tqdm

from typing import List, Tuple

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
elif torch.mps.is_available():
	device = "mps"
print(f"Using device: {device}")

NATURAL_STORIES_TOK_FILE = "naturalstories/naturalstories_RTS/all_stories.tok"
OUTPUT_FILE = "naturalstories/naturalstories_RTS/stories_surprisa_pythia_70m_step1000.tsv"


# def parse_args():
# 	parser = ArgumentParser()
# 	parser.add_argument("--window-size", type=int, default=2048)
# 	parser.add_argument("--stride", type=int, default=1)
# 	parser.add_argument("--batch-size", type=int, default=16)
# 	parser.add_argument("--out-file", type=str, default="naturalstories/naturalstories_RTS/stories_surprisal.tsv")
# 	return parser.parse_args()


# def _compile_stimuli(story_words: List[str], window_size: int, stride: int, tokenizer) -> List[str]:
# 	stimuli = []
# 	for current_idx, current_word in enumerate(story_words):
# 		current_word_tokens = tokenizer.encode(f" {current_word}")
# 		stimulus = [current_word_tokens]
# 		num_context_tokens = 0
# 		for context_idx in range(current_idx-1, np.max(current_idx-window_size,0), -stride):
# 			# first token needs no additional context
# 			if context_idx < 0: break
# 			# add whitespace for proper tokenization
# 			word_tokens = tokenizer.encode(f" {story_words[context_idx]}")
# 			stimulus.insert(0, word_tokens)
# 			num_context_tokens += len(word_tokens)
# 			# stop sampling once window size is exceeded
# 			if num_context_tokens >= window_size:
# 				break
# 		stimulus.insert(0, [tokenizer.bos_token_id])
# 		stimuli.append(list(chain.from_iterable(stimulus)))
# 	return stimuli


# def _score(stimuli: List[str], scorer: scorer.IncrementalLMScorer, batch_size: int) -> List[Tuple[str, float]]:
# 	scores = []
# 	for idx in range(batch_size, len(stimuli)+batch_size, batch_size):
# 		batch_stimuli = stimuli[idx-batch_size:idx]
# 		batch_scores = scorer.token_score(
# 			batch=batch_stimuli,
# 			base_two=True,
# 			bow_correction=True
# 		)
# 		scores.extend(batch_scores)
# 	return scores


def _get_word_scores(token_scores: List[Tuple[str, float]]) -> List[float]:
	"""Assumes a GPT-2-style tokenizer, i.e., 'Ġ' is expected to mark the beginning of words"""
	word_scores = []
	current_word_score = 0.
	for idx in range(len(token_scores)):
		current_word_score += token_scores[idx][-1]
		if idx == len(token_scores)-1 or token_scores[idx+1][0].startswith("Ġ"):
			word_scores.append(current_word_score)
			current_word_score = 0.0
	return [-1*s for s in word_scores]


def main():

	# args = parse_args()
	from transformers import AutoTokenizer
	tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
	ilm_scorer = scorer.IncrementalLMScorer(
		model="EleutherAI/pythia-70m-deduped", tokenizer=tokenizer, device=device, revision="step1000")
	tokenizer = ilm_scorer.tokenizer

	csv_header = ["word", "item", "zone", "sentence_id", "sentence_position", "surprisal"]
	f_out = open(OUTPUT_FILE, "w")
	csvwriter = csv.writer(f_out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
	csvwriter.writerow(csv_header)
		
	with open(NATURAL_STORIES_TOK_FILE, "r") as f:
		stories = f.read().split("\n")[1:-1]

	words_by_story = defaultdict(list)
	for line in stories:
		word, _, story_id = line.split("\t")
		words_by_story[story_id].append(word)

	for story_id, story_words in tqdm(words_by_story.items()):

		story_words.insert(0, tokenizer.bos_token)
		token_scores = ilm_scorer.token_score(" ".join(story_words))
		word_scores = _get_word_scores(token_scores[0]) # skip bos token
		# make sure no token was lost on the way
		assert len(word_scores) == len(story_words), len(story_words) -len(word_scores)

		# # tokenize and compile stimuli with at least window_size tokens in the context
		# stimuli = _compile_stimuli(story_words, args.window_size, args.stride, tokenizer)
		# # detokenize
		# stimuli = [tokenizer.decode(stimulus) for stimulus in stimuli]
		# token_scores = _score(stimuli, scorer=ilm_scorer, batch_size=args.batch_size)
		# word_scores = [_get_word_score(stimulus_score[1:]) for stimulus_score in token_scores]

		# save with sentence ids and sentence positions
		sentence_id = 1
		sentence_position = 1
		for story_position, (word, score) in enumerate(zip(story_words[1:], word_scores[1:]), start=1):
			csvwriter.writerow([word, story_id, story_position, sentence_id, sentence_position, score])
			sentence_position += 1
			if (word.endswith(".") or word.endswith(".'")) and not word in ["Mr.", "Dr."]:
				sentence_id += 1
				sentence_position = 1


if __name__ == "__main__":
	main()