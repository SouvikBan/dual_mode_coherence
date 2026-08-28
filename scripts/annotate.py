import argparse
import json

from dual_mode_coherence.annotation import (
	load_spacy, 
	iter_jsonl, 
	annotate_document, 
	build_coref
)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--coref", default="maverick",
                    choices=["maverick", "fastcoref", "coreferee", "none"])
    ap.add_argument("--coref-model", default=None,
                    help="maverick checkpoint, e.g. sapienzanlp/maverick-mes-ontonotes "
                         "(default) or sapienzanlp/maverick-mes-litbank")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0 (maverick/fastcoref)")
    ap.add_argument("--spacy-sm", action="store_true",
                    help="prefer the small spaCy model over the transformer one")
    ap.add_argument("--annotate-alternatives", action="store_true",
                    help="also annotate each string in an 'alternatives' list (each vs context)")
    ap.add_argument("--turn-sep", default=None,
                    help="dialogue turn separator in 'context' (e.g. '</s> <s>'); replaced by a space for parsing")
    args = ap.parse_args()

    nlp = load_spacy(prefer_trf=not args.spacy_sm)
    coref = build_coref(args.coref, nlp, args.device, model=args.coref_model)

    n = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for item in iter_jsonl(args.inp):
            context = item.get("context", "") or ""
            target = item.get("target", "") or ""
            post = item.get("post_context") or ""
            if args.turn_sep:
                context = context.replace(args.turn_sep, " ")

            segments = {"context": context, "target": target}
            if post:
                segments["post_context"] = post

            seg_ann, clusters, grid = annotate_document(nlp, coref, segments)
            item["segments"] = seg_ann
            item["coref_clusters"] = clusters
            item["entity_grid"] = grid

            if args.annotate_alternatives and isinstance(item.get("alternatives"), list):
                alt_out = []
                for alt in item["alternatives"]:
                    a_seg, a_cl, a_grid = annotate_document(
                        nlp, coref, {"context": context, "target": str(alt)})
                    alt_out.append({"text": alt, "segments": a_seg,
                                    "coref_clusters": a_cl, "entity_grid": a_grid})
                item["alternatives_annotated"] = alt_out

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  ...{n} items")
    print(f"[done] annotated {n} items -> {args.out}")


if __name__ == "__main__":
    main()