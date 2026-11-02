# A small language model, entirely inside SQL Server

A working neural language model — training and inference — implemented in
Transact-SQL. No CLR, no Python, no external services. The weights are rows in
tables, the forward pass is a join, and backpropagation is a sequence of
set-based `UPDATE` statements.

The training corpus is **examples of SQL queries available on the internet**,
so the model generates SQL. A language model written in SQL, trained on SQL,
that emits SQL.

## Run order

| File | What it does |
|---|---|
| `01_schema_and_corpus.sql` | Database, hyperparameters, 122-query corpus, normaliser, tokeniser, training-set construction |
| `02_model_and_forward_pass.sql` | Weight tables, initialisation, and the forward pass as inline table-valued functions |
| `03_training.sql` | Backpropagation and the SGD-with-momentum training loop |
| `04_example_queries.sql` | Sampling procedure and eleven example queries against the trained model |

Execute them in order. Files 01–03 are self-running: each ends by invoking the
procedure it just defined.

Requires SQL Server 2016 (compatibility level 130) or later, or Azure SQL.
`OPENJSON` is the only version-sensitive dependency — it's used for tokenising
because it yields an ordinal on every supported version, whereas
`STRING_SPLIT` only does from 2022.

## The architecture

A Bengio-style neural language model — the direct ancestor of the transformer,
and small enough that every tensor fits comfortably in a table:

```
slot k of C holds token t_k
x[k*D + d] = Emb[t_k, d]                     concatenated embeddings
z[h]       = SUM_i x[i] * W1[i,h] + B1[h]
a[h]       = MAX(z[h], 0)                    ReLU
logit[o]   = SUM_h a[h] * W2[h,o] + B2[o]
p[o]       = EXP(logit[o] - max) / SUM(...)  softmax
```

Defaults: context `C=3`, embedding `D=16`, hidden `H=32`, vocabulary ~213,
which is about 12,000 parameters. Every one of those lines is a `GROUP BY`
over a join — that's the entire trick, and it's why the batched version
(`fn_Score`, which scores every position of a query at once) is the same code
with `pos` carried through the `GROUP BY` clauses.

Dimensions live in `dbo.ModelConfig` rather than being hard-coded, so the model
is resizable without editing the functions.

## The four things most likely to go wrong if you adapt this

1. **Ordering in the backward pass.** `dZ` needs the *pre-update* `W2` and `dX`
   needs the *pre-update* `W1`. Both are materialised into temp tables before
   either weight matrix is written. Update the weights first and the model
   still trains, just slowly and silently wrongly.

2. **`RAND()` is constant per statement**, so it can't fill a table with noise.
   `CHECKSUM(NEWID())` is evaluated per row. Note also that `ABS(CHECKSUM(...))`
   overflows on the single value `-2^31`; taking the modulus first avoids it,
   since `%` preserves sign.

3. **Softmax needs two scopes.** SQL Server won't allow a windowed function
   inside the argument of another windowed function, so
   `SUM(EXP(x - MAX(x) OVER (...))) OVER (...)` — the obvious one-liner — is
   rejected. The max-subtraction and the normalising sum are split across two
   CTE levels wherever softmax appears.

4. **Multi-character operators.** A naive single-character padding pass turns
   `<>` into two tokens. The normaliser parks `<>`, `>=`, `<=`, `!=` behind
   placeholders first, then restores them.

## Expected training behaviour

| Epoch | Loss | Perplexity |
|---|---|---|
| 1 | ~5.3 | ~200 |
| 15 | ~2.1 | ~8 |
| 30 | ~1.5 | ~4.4 |
| 60 | ~1.05 | ~2.9 |

Cost is dominated by the `batch × hidden × vocabulary` joins — roughly two
billion row operations for the default 60 epochs. On a modest instance that's
several minutes. `@Epochs = 120` reaches perplexity ~2.0 and noticeably cleaner
generations for double the time.

Speed knobs, in order of bluntness: raise `@BatchSize`; set `@GradEps` to
around `1e-6` to skip the large number of near-zero output gradients (this
makes training approximate, which is why it's off by default); or move the
parameter tables to memory-optimised tables, where the repeated small
aggregations are considerably cheaper.

## Why do this in SQL at all

Two things genuinely come out nicer than in an imperative implementation, and
the example queries in file 04 lean on both:

- **The model is a relation.** `fn_NextToken` returns a distribution as a
  rowset, so `CROSS APPLY` runs a batch of prompts and returns one tidy result
  set (query 3), and constrained decoding — restricting generation to a grammar
  — is a `WHERE` clause plus a renormalising window function (query 9). No
  special decoder is required.

- **The model is inspectable.** Nothing is hidden in a binary blob. Cosine
  similarity between token embeddings is a `GROUP BY` (query 7), and the
  trained weights can be backed up, diffed, version-controlled, or shipped by
  `INSERT`.

Autoregressive sampling is the one part that stays procedural: each token
depends on the last, and a recursive CTE may not call a table-valued function
on its recursive member.

## Honest limitations

- A 3-token context window means no long-range structure. It will happily open
  a `WITH` clause and never close it, or emit `WHERE order_id = 'london'`. It
  has learned local SQL syntax, not SQL semantics.
- 122 training queries is a very small corpus, so the model memorises templates
  as much as it generalises. That's visible in query 6, where the least
  predictable corpus entries are the ones whose patterns appear only once.
- Embedding neighbours are meaningful for frequent tokens — table names cluster
  with table names, aggregates with aggregates — and close to noise for rare
  ones, whose vectors have barely moved from initialisation. Query 7 filters on
  `freq >= 8` for that reason. The vector arithmetic in query 8 is included
  because the mechanic is interesting, not because the results are reliable at
  this scale.
- ReLU rather than tanh, because SQL Server has no `TANH` and
  `(EXP(2z)-1)/(EXP(2z)+1)` needs clamping to avoid overflow. `CASE WHEN z > 0`
  is cheaper and clearer.
- Momentum is not decayed for parameters absent from a batch's gradient. For
  embeddings that's the standard sparse treatment; for the dense layers every
  parameter receives a gradient each batch, so it doesn't arise.

## Making it bigger

The obvious next step is a single attention head over the context instead of
concatenation: score each slot with `q · k_j`, softmax over slots, take the
weighted sum of values. The forward pass is three more joins. The backward pass
is the fiddly part — the softmax Jacobian over slots adds a self-join that the
current MLP doesn't need. Widening the context and the corpus will buy more
than attention will at this scale, though.
