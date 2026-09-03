# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "minicons>=0.3.39",
# ]
# ///
from argparse import ArgumentParser
from minicons import scorer
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
elif torch.mps.is_available():
	device = "mps"
print(f"Using device: {device}")


CLASP_FILE = "BLL2018/data/processed_ratings.csv"
CLASP_WITH_CONTEXT_PREPROCESSED = "BLL2018/data/clasp_by_worker_and_sent.csv"

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--model-name-or-path", type=str, default="EleutherAI/pythia-70m-deduped")
	parser.add_argument("--revision", type=str, default=None)
	parser.add_argument("--window-size", type=int, default=None)
	parser.add_argument("--output-dir", type=str, default="annotated/surprisal/clasp")
	# parser.add_argument("--stride", type=int, default=1)
	# parser.add_argument("--batch-size", type=int, default=8)
	return parser.parse_args()


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


# def _get_word_scores(token_scores: List[Tuple[str, float]]) -> List[float]:
# 	"""Assumes a GPT-2-style tokenizer, i.e., 'Ġ' is expected to mark the beginning of words."""
# 	word_scores = []
# 	current_word_score = 0.
# 	for idx in range(len(token_scores)):
# 		current_word_score += token_scores[idx][-1]
# 		if idx == len(token_scores)-1 or token_scores[idx+1][0].startswith("Ġ"):
# 			word_scores.append(current_word_score)
# 			current_word_score = 0.0
# 	return [-1*s for s in word_scores]


def main():

	args = parse_args()
	# One should use revision="step1000" for pythia models (equals 2B training tokens)
	ilm_scorer = scorer.IncrementalLMScorer(
		model=args.model_name_or_path, device=device, revision=args.revision)
	tokenizer = ilm_scorer.tokenizer

	clasp_df = pd.read_csv(CLASP_WITH_CONTEXT_PREPROCESSED) \
		.drop(["suffix", "suffixextra"], axis=1)

	sentence_scores = []
	for i, row in tqdm(clasp_df.iterrows()):
		stimulus = row["sent"]
		prefix = f"{tokenizer.bos_token} {row["prefix"]}"
		# print(f"{prefix} {stimulus}")

		# no window size provided -> assume that stories fit in the context of the model
		if not args.window_size:
			sentence_score = ilm_scorer.conditional_score(
				prefix, stimulus, 
				# we want summed sentence surprisals
				reduction=lambda x: x.sum().item(),
				# thus base two logarithm 
				base_two=True, 
				# and Tiago's bow correction
				bow_correction=True
			)
			sentence_scores.append(-1*sentence_score[0])
		else:
			# for now complain
			raise Exception

	if not Path(args.output_dir).exists():
		Path(args.output_dir).mkdir(parents=True)
	out_file_name = _get_out_file_name(args)	

	clasp_df = clasp_df.assign(sentence_surprisal = sentence_scores)
	clasp_df.to_csv(Path(args.output_dir) / out_file_name, sep="\t")

if __name__ == "__main__":
	main()