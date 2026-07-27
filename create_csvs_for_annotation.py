from pathlib import Path

from conllu import parse_incr

if not Path("to_annotate").exists():
    Path("to_annotate").mkdir()

current_story_id = "1"
sent_id_decr = 0
story_file = None

with open("naturalstories/parses/ud/stories-aligned.conllx", "r") as f:
    current_story_str = "story_id,sent_id,token_id,deprel,form\n"
    for sent_id, sentence in enumerate(parse_incr(f)):
        for token in sentence:
            # split only once as there could be subtokens
            story_id, token_id = token["misc"]["TokenId"].split(".", 1)
            # save to file if end of story encountered
            if story_id != current_story_id:
                with open(f"to_annotate/story_{current_story_id}_coref.csv", "w") as f_story:
                    f_story.write(current_story_str)
                sent_id_decr = sent_id
                current_story_id = story_id
                current_story_str = "story_id;sent_id;token_id;deprel;form\n"

            current_story_str += f"{story_id};{sent_id-sent_id_decr};{token_id};{token['deprel']};{token['form']}\n"