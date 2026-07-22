# Entity Information Value: Distance Functions for GUM

This note gives a simple formulation of distance functions for **entity information value** using the GUM corpus.

The basic setup is:

- We have a full preceding discourse context:  
  $$C_i = S_1, \ldots, S_i$$

- The gold next sentence is:  
  $$S_{i+1}^{gold}$$

- The language model generates alternatives:  
  $$S_{i+1}^{1}, \ldots, S_{i+1}^{K}$$

- A distance function compares the gold next sentence with each alternative.

The entity information value for sentence $S_{i+1}$ can then be estimated as:

$$
IIV_d(i) =
\frac{1}{K}
\sum_{k=1}^{K}
d\left(S_{i+1}^{gold}, S_{i+1}^{k} \mid C_i\right)
$$

where $d$ is an entity distance function.

The important point is that $d$ should compare **entity behaviour**, not just surface text.



## Sentence as a Set of Entity Mentions

For each sentence $S$, define:

$$
M(S) = \{m_1, m_2, \ldots, m_n\}
$$

where each $m$ is an entity mention.

Each mention has:

$$
m = (e, g, f, r, t)
$$

where:

- $e$: cluster ID
- $g$: givenness
- $f$: referring-expression form
- $r$: grammatical role
- $t$: mention text or head text

The context $C_i$ tells us which entities are already known.

So we define:

$$
E(C_i) = \text{set of entity clusters already introduced in the context}
$$

This lets us distinguish:

```text
old entity recurrence
new entity introduction
```

---

##  Distance Function 1: Simple Entity-Set Distance

This is the simplest baseline.

For the gold sentence and an alternative sentence, collect the set of old entities they mention:

$$
Old(S \mid C) =
\{e : e \in M(S), e \in E(C)\}
$$

Then compare the two sets using Jaccard distance:

$$
d_{entity\text{-}set}(S_g, S_a \mid C)
=
1 -
\frac{
|Old(S_g \mid C) \cap Old(S_a \mid C)|
}{
|Old(S_g \mid C) \cup Old(S_a \mid C)|
}
$$

If both mention the same old entities, distance is low.

If the gold continues one entity and the alternative continues a different entity, distance is high.

### What it Captures

This captures basic entity recurrence.

Example:

```text
Context: John bought a bike. Mary liked it.

Gold: He rode it home.
Alternative: She rode it home.
```

Gold continues `John` and `bike`.

Alternative continues `Mary` and `bike`.




##  Distance Function 2: Feature-Vector Distance

This is the main practical baseline.

For each sentence, build a feature vector from entity mentions.

A feature can be:

```text
cluster_id + givenness + form + role
```

For example:

```text
entity=17 | given | pronoun       | subject
entity=22 | given | pronoun       | object
NEW-person | new  | indefinite_np | subject
```

So a sentence becomes a vector of counts:

$$
\phi(S \mid C)
$$

Example:

```text
He saw the dog.
```

might become:

```text
entity=John | given | pronoun     | subject = 1
entity=dog  | given | definite_np | object  = 1
```

Then compare the gold vector and alternative vector.

A simple option is normalized L1 distance:

$$
d_{feat\text{-}L1}(S_g, S_a \mid C)
=
\frac{1}{2}
\left\|
\hat{\phi}(S_g \mid C)
-
\hat{\phi}(S_a \mid C)
\right\|_1
$$

where $\hat{\phi}$ is the normalized feature vector.

Another option is Jensen-Shannon distance:

$$
d_{feat\text{-}JS}(S_g, S_a \mid C)
=
\sqrt{
JS\left(
\hat{\phi}(S_g \mid C),
\hat{\phi}(S_a \mid C)
\right)
}
$$

### What it Captures

This captures:

- whether the same old entities recur
- whether new entities are introduced
- whether old entities appear as pronouns, definite NPs, proper names, etc.
- whether entities appear as subjects, objects, or peripheral mentions
- whether the gold and alternative have similar entity structure

This should be the first serious distance function.

### Why it is Useful

It is interpretable.

If a generated alternative has a high distance, we can inspect which feature bins changed.

For example:

```text
gold:
entity=17 | given | pronoun     | subject

alternative:
entity=17 | given | proper_name | subject
```

The entity is the same, but the form changed.

That is a smaller difference than changing the entity entirely.

---

## 5. Distance Function 3: Weighted Feature Distance

The feature-vector distance can be made more linguistically sensitive by weighting features.

For example, grammatical roles can have weights:

```text
subject     = 3.0
object      = 2.0
oblique     = 1.0
possessive  = 1.0
other       = 0.5
```

Then the feature vector becomes weighted:

$$
\phi_j(S \mid C)
=
\sum_{m \in S}
w(role(m))
\cdot
\mathbf{1}[m \text{ has feature } j]
$$

This makes subject continuations count more than peripheral mentions.

### Example

```text
Gold: He won the race.
Alternative: The race was won by him.
```

Both mention the same entity.

But in the gold sentence, the entity is subject.

In the alternative, the entity is oblique or passive agent.

The distance should not be zero, because the entity's discourse role changed.

---

## 6. Distance Function 4: Mention-Level Optimal Transport

Feature vectors are useful, but they lose individual mention alignment.

Optimal transport compares mentions directly.

Let:

$$
M_g = M(S_g)
$$

and:

$$
M_a = M(S_a)
$$

be the mentions in the gold and alternative sentence.

We define a cost between one gold mention $m$ and one alternative mention $m'$:

$$
c(m,m')
$$

A simple cost is:

$$
c(m,m')
=
\lambda_e d_e(m,m')
+
\lambda_g d_g(m,m')
+
\lambda_f d_f(m,m')
+
\lambda_r d_r(m,m')
+
\lambda_t d_t(m,m')
$$

where:

- $d_e$: entity-cluster mismatch
- $d_g$: givenness mismatch
- $d_f$: referring-expression form mismatch
- $d_r$: grammatical-role mismatch
- $d_t$: text/head semantic distance

For example:

```text
same cluster      -> d_e = 0
different cluster -> d_e = 1

same form         -> d_f = 0
different form    -> d_f = 1
```

Then OT finds the cheapest global alignment between mentions:

$$
d_{OT}(S_g, S_a)
=
\min_{\pi}
\sum_{m,m'} \pi(m,m') c(m,m')
$$

where $\pi$ says how much each gold mention is matched to each alternative mention.

### Why OT is Useful

A max-cosine or nearest-neighbour method can cheat.

One alternative mention can become the best match for many gold mentions.

OT avoids this by forcing a more global matching.

### Practical Version

Use unbalanced OT, because the gold and alternative may have different numbers of mentions.

This allows insertions and deletions of entity mentions.

In practice:

```text
gold:        John / he / the bike
alternative: Mary / the car
```

OT must pay for:

- John versus Mary
- bike versus car
- missing mention mass

This gives a better entity distance than pairwise max similarity.

---

## 7. Distance Function 5: Entity-Transition Entropy

This is the more explicitly information-theoretic version.

Instead of directly comparing sentence vectors, classify each sentence into an entity-transition type.

Examples of transition types:

```text
old_subject_continues
old_object_becomes_subject
previous_subject_dropped
only_new_entities
new_person_introduced
given_pronoun_subject
given_definite_subject
accessible_definite_np
```

Let:

$$
z(S \mid C)
$$

be the entity-transition class of sentence $S$ given context $C$.

From generated alternatives, estimate:

$$
q_i(z) = P(z \mid C_i)
$$

using counts:

$$
q_i(z)
=
\frac{
\#\{k : z(S_{i+1}^{k} \mid C_i) = z\} + \alpha
}{
K + \alpha |\mathcal{Z}|
}
$$

The gold sentence has transition type:

$$
z^* = z(S_{i+1}^{gold} \mid C_i)
$$

Then entity information value is:

$$
IV_{entropy}(i) = -\log q_i(z^*)
$$

### Intuition

If the gold entity transition is common among the alternatives, it has low information value.

If the gold entity transition is rare among the alternatives, it has high information value.

### Example

Suppose the context strongly predicts that the previous subject will continue.

Most generated alternatives do this:

```text
He ...
He ...
The man ...
He ...
```

But the gold sentence shifts to a new entity:

```text
Meanwhile, a woman entered the room.
```

Then the gold entity transition is surprising.

So:

$$
-\log q_i(z^*)
$$

is high.

### Difference from Distance Functions

This is not exactly a distance between two sentences.

It is a surprisal of the gold entity-transition type under the generated alternative distribution.

That makes it closer to the original information-value idea.

---

## 8. Distance Function 6: Graph Distance

This is inspired by the Guinaudeau and Strube entity graph idea.

Build a bipartite graph:

```text
sentence nodes  <->  entity nodes
```

A sentence is connected to an entity if that entity is mentioned in the sentence.

Example:

```text
S1 -- John
S2 -- John
S2 -- bike
S3 -- bike
```

The edge can be weighted by grammatical role:

```text
subject = 3
object  = 2
other   = 1
```

Now compare how the gold sentence and the alternative sentence attach to the previous discourse graph.

Let $B_C$ be the sentence-entity matrix for the context.

Let $b_g$ be the entity vector for the gold next sentence.

Let $b_a$ be the entity vector for the alternative sentence.

The connection profile of the gold sentence to previous sentences is:

$$
g_g = B_C b_g^\top
$$

The connection profile of the alternative is:

$$
g_a = B_C b_a^\top
$$

Then define:

$$
d_{graph}(S_g,S_a \mid C)
=
\frac{
\|g_g - g_a\|_1
}{
\|g_g\|_1 + \|g_a\|_1 + \epsilon
}
$$

### Intuition

This distance asks:

> Does the candidate sentence connect to the same earlier parts of the discourse as the gold sentence?

This is different from just asking whether the same entity appears.

An entity that appeared many times across the discourse creates a stronger graph connection than an entity that appeared only once.

### Why this is Useful

It captures structural coherence.

Example:

```text
Context:
S1: John bought a bike.
S2: He repaired it.
S3: The bike was expensive.
S4: Mary called him.

Gold:
The bike broke again.

Alternative:
Mary laughed.
```

Both alternatives mention an old entity.

But the gold sentence reconnects to the central `bike` chain.

The alternative reconnects to a less central entity.

The graph distance captures this.

---

## 9. Combined Distance

Eventually, we can combine distances:

$$
d_{combined}
=
\alpha d_{feat}
+
\beta d_{OT}
+
\gamma d_{graph}
$$

But this should not be the first step.

First, keep the distances separate.

Each distance answers a different question:

| Distance | Question |
|---|---|
| Entity-set distance | Did the same old entities recur? |
| Feature-vector distance | Did entities recur with similar form, givenness, and role? |
| Weighted feature distance | Did important roles like subject continuation match? |
| OT distance | Can gold and alternative mentions be globally aligned? |
| Entropy / transition surprisal | Was the gold entity transition expected under alternatives? |
| Graph distance | Does the sentence connect to the same previous discourse structure? |

---





---

