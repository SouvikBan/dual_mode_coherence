from pathlib import Path
import csv
import pandas as pd
import numpy as np

annotated_dir = Path("annotated")
csv_file_name = "story_6_coref_new.csv"


with open(annotated_dir / csv_file_name, "r") as f:
    df_story = pd.read_csv(f, sep=";", dtype={"token_id": str, "GRP": str})
    entity_ids = df_story["GRP"][~df_story["GRP"].isna()].unique()
    # create mapping of old entity ids to new entity ids (in numerical order)
    mapping = {
        id_old: str(id_new) for id_new, id_old in enumerate(entity_ids, start=1)
    }
    print(mapping)
    df_story["GRP"] = df_story["GRP"].apply(lambda x: mapping[x] if isinstance(x, str) else x)

    df_story.to_csv(annotated_dir / f"{csv_file_name.split(".")[0]}_cleaned.csv", sep=";")
