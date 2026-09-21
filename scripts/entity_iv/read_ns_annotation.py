#!/usr/bin/env python3
"""Reading the manual Natural Stories annotation CSVs.

This is the only loader the counting scripts need. It was extracted from the
research checkout's count_gold_entities.py and build_gum_inf_transition_cost.py,
neither of which is shipped here: the first is a superseded measure whose output
correlates r = 0.92 with position in the story, the second fits the information
status cost model, which none of the released columns use.

The CSV is semicolon separated with columns

    story_id;sent_id;token_id;deprel;form;head;pos;entity_layer

where `entity_layer` carries GUM-style bracket spans, for example
`Entity=(3-person-giv:act` opening and `3)` closing. Sentence ids are 0-based
and must be contiguous.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict

# `(payload` opens a mention, `)` or `id)` closes one
EVENT = re.compile(r'\(([^()]*)|(\d+)?\)')
INFSTATS = ('new', 'giv:act', 'giv:inact', 'acc:inf', 'acc:com', 'acc:aggr')
ALIASES = {'given:active': 'giv:act', 'given:inactive': 'giv:inact',
           'accessible:inferrable': 'acc:inf', 'accessible:inferable': 'acc:inf',
           'accessible:commonground': 'acc:com', 'accessible:aggregate': 'acc:aggr'}
# GUM marks automatically assigned, rather than manually annotated, information
# status as 'auto'. It is a provenance marker, not a seventh state.
NOT_A_STATUS = {'auto'}


def normalise_infstat(value):
    if not isinstance(value, str):
        raise ValueError('missing information status: use the original NS CSV Entity annotations')
    label = ALIASES.get(value.strip().lower(), value.strip().lower())
    if label in NOT_A_STATUS:
        return None
    if label not in INFSTATS:
        raise ValueError(f'unknown/missing information status {value!r}; expected {INFSTATS}')
    return label


def entity_mentions(layers, infstat_index=2, minspan_index=5):
    """Parse GUM bracket spans, keeping the status at each mention's opening.

    Handles nested and overlapping spans, and either bare or explicit-id
    closings. The Entity field order is entity id, entity type, information
    status.
    """
    stack, mentions = [], []
    for token_index, value in enumerate(layers):
        layer = (value or '_').removeprefix('Entity=')
        for event in EVENT.finditer(layer):
            payload, closing_id = event.groups()
            if payload is not None:
                fields = payload.split('-')
                if not fields[0].isdigit():
                    raise ValueError(f'invalid entity opening: {payload!r}')
                if len(fields) <= infstat_index:
                    raise ValueError('Entity annotations have lost infstat; '
                                     'use the original rich annotation CSV')
                minspan = fields[minspan_index] if len(fields) > minspan_index else ''
                stack.append((int(fields[0]), token_index,
                              normalise_infstat(fields[infstat_index]), minspan))
            else:
                if not stack:
                    raise ValueError(f'closing unopened entity at token {token_index}')
                index = len(stack) - 1
                if closing_id is not None:
                    matches = [i for i, item in enumerate(stack) if item[0] == int(closing_id)]
                    if not matches:
                        raise ValueError(f'closing unopened entity {closing_id}')
                    index = matches[-1]
                entity, start, status, minspan = stack.pop(index)
                mentions.append({'cluster_id': entity, 'token_start': start,
                                 'token_end': token_index, 'infstat': status,
                                 'minspan': minspan})
    if stack:
        raise ValueError(f'unclosed entities: {stack}')
    return sorted(mentions,
                  key=lambda m: (m['token_start'], -m['token_end'], m['cluster_id']))


def load_ns_csvs(directory):
    """Yield one document per story, with its sentences, tokens and mentions."""
    paths = sorted(directory.glob('story_*_entity_annotated.csv'),
                   key=lambda p: int(p.name.split('_')[1]))
    if not paths:
        raise ValueError(f'no Natural Stories annotation CSVs in {directory}')
    for path in paths:
        story_id = int(path.name.split('_')[1])
        by_sentence = defaultdict(list)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, delimiter=';')
            required = {'story_id', 'sent_id', 'token_id', 'form', 'head', 'pos',
                        'deprel', 'entity_layer'}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'{path}: missing CSV columns {sorted(missing)}')
            for row in reader:
                if int(row['story_id']) != story_id:
                    raise ValueError(f'{path}: inconsistent story ID')
                by_sentence[int(row['sent_id'])].append(row)
        if sorted(by_sentence) != list(range(len(by_sentence))):
            raise ValueError(f'{path}: sentence IDs must be contiguous from zero')
        sentences = []
        for index, rows in sorted(by_sentence.items()):
            rows.sort(key=lambda row: tuple(int(part) if part.isdigit() else part
                                            for part in re.split(r'([0-9]+)', row['token_id'])))
            tokens = [{'id': i + 1, 'text': row['form'], 'head': int(row['head']),
                       'upos': row['pos'], 'xpos': row['pos'], 'deprel': row['deprel'],
                       'entity_layer': row['entity_layer']}
                      for i, row in enumerate(rows)]
            mentions = entity_mentions([row['entity_layer'] for row in rows])
            sentences.append({'sentence_index': index,
                              'text': ' '.join(t['text'] for t in tokens),
                              'tokens': tokens, 'mentions': mentions})
        yield {'dataset': 'naturalstories', 'story_id': f'naturalstories_{story_id}',
               'context': sentences}
