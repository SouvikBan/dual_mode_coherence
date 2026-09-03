# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "minicons>=0.3.39",
# ]
# ///
from argparse import ArgumentParser
from minicons import scorer
import csv
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Tuple

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
elif torch.mps.is_available():
	device = "mps"
print(f"Using device: {device}")

# NATURAL_STORIES_TOK_FILE = "naturalstories/naturalstories_RTS/all_stories.tok"
NATURAL_STORIES_CONLLX_FILE = "naturalstories/parses/ud/stories-aligned.conllx"

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--model-name-or-path", type=str, default="EleutherAI/pythia-70m-deduped")
	parser.add_argument("--revision", type=str, default=None)
	parser.add_argument("--window-size", type=int, default=None)
	parser.add_argument("--output-dir", type=str, default="annotated/surprisal/naturalstories")
	# parser.add_argument("--stride", type=int, default=1)
	# parser.add_argument("--batch-size", type=int, default=8)
	return parser.parse_args()


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


# def _score_batched(stimuli: List[str], scorer: scorer.IncrementalLMScorer, batch_size: int) -> List[Tuple[str, float]]:
# 	scores = []
# 	for idx in range(batch_size, len(stimuli)+batch_size, batch_size):
# 		batch_stimuli = stimuli[idx-batch_size:idx]
# 		batch_scores = scorer.token_score(
# 			batch=batch_stimuli,
# 			base_two=True,
# 			bow_correction=True
# 		)s
# 		scores.extend(batch_scores)
# 	return scores

def _load_naturalstories_conllx(conllx_path):
	"""Load naturalstories with sentence splitting info from ud parse."""
	grouped_sents = []
	with open(conllx_path, "r", encoding="utf-8") as f_conllx:
		current_sentence_id = 1
		last_item = 1
		current_tokens, current_items, current_zones = [], [], []
		for line in f_conllx.read().split("\n")[:-1]:
			# empty lines signal sentence boundaries
			if not line:
				last_item = current_items[-1]
				for t, z in zip(current_tokens, current_zones):
					grouped_sents.append([
						int(last_item), t, int(z), current_sentence_id
					])
				current_sentence_id += 1
				current_tokens, current_items, current_zones = [], [], []
			else:
				line_content = line.split("\t")
				token = line_content[1]
				misc = line_content[-1].split(".")
				item, zone = misc[:2]
				item = item[8:]
				if last_item != item:
					current_sentence_id = 1
				current_tokens.append(token)
				current_items.append(item)
				current_zones.append(zone)

	# reassemble punctuation by grouping uniquely by item, zone and sentence, 
	# then aggregating partial token strings and sorting
	df = pd.DataFrame(data=grouped_sents, columns=["item", "token", "zone", "sentence_id"]) \
			.groupby(["item", "zone", "sentence_id"])["token"] \
			.agg("".join).reset_index() \
			.sort_values(by=["item", "sentence_id", "zone"], axis=0)

	return df

def _get_out_file_name(args):
	out_file_prefix = f"{args.model_name_or_path.split("/")[-1]}"
	if args.revision:
		out_file_prefix += f"-{args.revision}"
	if args.window_size:
		out_file_prefix += f"-{args.window_size}"
	else:
		out_file_prefix += "-full"
	out_file_name = out_file_prefix + ".tsv"
	return out_file_name


def _get_word_scores(token_scores: List[Tuple[str, float]]) -> List[float]:
	"""Assumes a GPT-2-style tokenizer, i.e., 'Ġ' is expected to mark the beginning of words."""
	word_scores = []
	current_word_score = 0.
	for idx in range(len(token_scores)):
		current_word_score += token_scores[idx][-1]
		if idx == len(token_scores)-1 or token_scores[idx+1][0].startswith("Ġ"):
			word_scores.append(current_word_score)
			current_word_score = 0.0
	return [-1*s for s in word_scores]


def main():

	args = parse_args()
	# One should use revision="step1000" for pythia models (equals 2B training tokens)
	ilm_scorer = scorer.IncrementalLMScorer(
		model=args.model_name_or_path, device=device, revision=args.revision)
	tokenizer = ilm_scorer.tokenizer

	stories_df = _load_naturalstories_conllx(NATURAL_STORIES_CONLLX_FILE)

	if not Path(args.output_dir).exists():
		Path(args.output_dir).mkdir(parents=True)

	out_file_name = _get_out_file_name(args)
	out_file_path = Path(args.output_dir) / out_file_name
	f_out = open(out_file_path, "w")

	csv_header = ["word", "item", "zone", "sentence_id", "surprisal"]
	csvwriter = csv.writer(f_out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
	csvwriter.writerow(csv_header)

	for story_id in tqdm(stories_df["item"].unique(), desc="Scording naturalstories"):
		words, zones, items, sent_ids = \
			stories_df[stories_df.item == story_id].token.tolist(), \
			stories_df[stories_df.item == story_id].zone.tolist(), \
			stories_df[stories_df.item == story_id].item.tolist(), \
			stories_df[stories_df.item == story_id].sentence_id.tolist()

		# no window size provided -> assume that stories fit in the context of the model
		if not args.window_size:
			token_scores = ilm_scorer.token_score(" ".join([tokenizer.bos_token] + words))
			word_scores = _get_word_scores(token_scores[0])[1:] # skip bos token
			# make sure no token was lost on the way
			assert len(word_scores) == len(words), len(words) - len(word_scores)

		else:
			# for now complain
			raise Exception
			
			# # tokenize and compile stimuli with at least window_size tokens in the context
			# stimuli = _compile_stimuli(story_words, args.window_size, args.stride, tokenizer)
			# # detokenize
			# stimuli = [tokenizer.decode(stimulus) for stimulus in stimuli]
			# token_scores = _score(stimuli, scorer=ilm_scorer, batch_size=args.batch_size)
			# word_scores = [_get_word_score(stimulus_score[1:]) for stimulus_score in token_scores]

		for word, zone, item, sent_id, score in zip(words, zones, items, sent_ids, word_scores):
			csvwriter.writerow([word, item, zone, sent_id, score])


if __name__ == "__main__":
	main()