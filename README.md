
## Natural stories coreference annotation pipeline

**Prepare Natural Stories Corpus**

``uv run scripts/prepare_ns.py --repo naturalstories/ --out ns.jsonl``

**Annotate with Maverick Coref**

``uv run scripts/annotate_ns.py --in ns.jsonl --out ns.sentences.annotated.litbank.jsonl --coref-model sapienzanlp/maverick-mes-litbank --device cuda``

## Running the visual tool for easier manual annotation 
``uv run scripts/viz_ns_annotations.py --in ns.sentences.annotated.litbank.jsonl --out audit.html`` 

## Distance Function
Check Distance.MD 







