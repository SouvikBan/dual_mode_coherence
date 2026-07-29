from pathlib import Path
import csv

from conllu import parse_incr

csv_header = ["story_id", "sent_id", "token_id", "deprel", "form", "GRP", "etype", "infstat", "minspan", "link", "identity"]


if not Path("to_annotate").exists():
    Path("to_annotate").mkdir()

sent_id_decr = 0
current_story_id = "1"

f_story = open(f"to_annotate/story_{current_story_id}_coref.csv", "w")
csvwriter = csv.writer(f_story, delimiter=";", quoting=csv.QUOTE_MINIMAL)

with open("naturalstories/parses/ud/stories-aligned.conllx", "r") as f:
    for sent_id, sentence in enumerate(parse_incr(f)):
        for token in sentence:
            # split only once as there could be subtokens
            story_id, token_id = token["misc"]["TokenId"].split(".", 1)
            # save to file if end of story encountered
            if story_id != current_story_id:
                f_story = open(f"to_annotate/story_{story_id}_coref.csv", "w")
                csvwriter = csv.writer(f_story, delimiter=";", quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(csv_header)

                sent_id_decr = sent_id
                current_story_id = story_id

            csvwriter.writerow([story_id, sent_id-sent_id_decr, token_id, token["deprel"], token["form"]])
    