/*==============================================================================
  A SMALL LANGUAGE MODEL, ENTIRELY INSIDE SQL SERVER
  File 04 of 04 : Sampling procedure and example queries against the model

  Everything below treats the language model as what it now is: a set of
  tables, and a function that returns a probability distribution as a rowset.
  That means the usual toolkit -- TOP, WHERE, GROUP BY, CROSS APPLY, window
  functions -- applies to the model itself, not just to its training data.
==============================================================================*/

USE SqlLlm;
GO
SET NOCOUNT ON;
GO

/*==============================================================================
  SAMPLER

  Autoregressive decoding needs a loop, because each step depends on the
  previous one, and a recursive CTE may not call a table-valued function on
  its recursive member. So this is the one part of the system that is
  procedural rather than set-based.

  The context is recovered by re-tokenising the accumulated text each step.
  That is O(n) per token rather than O(1), but n is a few dozen tokens, and
  it keeps the state in one place.
==============================================================================*/
CREATE OR ALTER PROCEDURE dbo.usp_Generate
    @Prompt      nvarchar(max) = N'select',
    @MaxTokens   int   = 40,
    @Temperature float = 0.8,   -- < 1 sharpens, > 1 flattens
    @TopK        int   = 8,     -- sample only from the k most likely tokens
    @Greedy      bit   = 0,     -- 1 = always take the argmax (deterministic)
    @Samples     int   = 1
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @out TABLE (sample_no int, generated nvarchar(max), tokens int);
    DECLARE @s   int = 1,
            @txt nvarchar(max),
            @i   int,
            @tok nvarchar(100),
            @r   float;

    WHILE @s <= @Samples
    BEGIN
        SET @txt = dbo.fn_Normalise(@Prompt);
        SET @i   = 0;

        WHILE @i < @MaxTokens
        BEGIN
            SET @tok = NULL;

            IF @Greedy = 1 OR @Temperature < 0.05
            BEGIN
                SELECT TOP (1) @tok = token
                FROM   dbo.fn_NextToken(@txt)
                ORDER BY p DESC;
            END
            ELSE
            BEGIN
                SET @r = RAND(CHECKSUM(NEWID()));

                ;WITH topk AS (
                    SELECT TOP (@TopK) token, p
                    FROM   dbo.fn_NextToken(@txt)
                    ORDER BY p DESC
                ),
                tempered AS (
                    SELECT token, q = POWER(p, 1.0 / @Temperature) FROM topk
                ),
                normed AS (
                    SELECT token, q = q / NULLIF(SUM(q) OVER (), 0) FROM tempered
                ),
                cumulative AS (
                    SELECT token,
                           hi = SUM(q) OVER (ORDER BY q DESC ROWS UNBOUNDED PRECEDING)
                    FROM   normed
                )
                SELECT TOP (1) @tok = token
                FROM   cumulative
                WHERE  hi >= @r
                ORDER BY hi;

                -- floating-point tail guard
                IF @tok IS NULL
                    SELECT TOP (1) @tok = token FROM dbo.fn_NextToken(@txt) ORDER BY p DESC;
            END;

            IF @tok IS NULL OR @tok = N'</s>' BREAK;

            SET @txt = @txt + N' ' + @tok;
            SET @i += 1;
        END;

        INSERT INTO @out (sample_no, generated, tokens)
        VALUES (@s, dbo.fn_Detokenise(@txt), @i);

        SET @s += 1;
    END;

    SELECT sample_no, generated, tokens FROM @out ORDER BY sample_no;
END;
GO


/*==============================================================================
  QUERY 1 -- The model card.
  How big is this thing, and what did it read?
==============================================================================*/
SELECT  context_tokens = C,
        embed_dim      = D,
        hidden_units   = H,
        vocab_size     = V,
        total_parameters = V*D + (C*D)*H + H + H*V + V
FROM    dbo.ModelConfig;

SELECT  tensor = 'Emb (token embeddings)', parameters = COUNT(*) FROM dbo.Emb
UNION ALL SELECT 'W1 (input -> hidden)',  COUNT(*) FROM dbo.W1
UNION ALL SELECT 'B1',                    COUNT(*) FROM dbo.B1
UNION ALL SELECT 'W2 (hidden -> vocab)',  COUNT(*) FROM dbo.W2
UNION ALL SELECT 'B2',                    COUNT(*) FROM dbo.B2
ORDER BY parameters DESC;

SELECT  source_site, queries = COUNT(*)
FROM    dbo.Corpus
GROUP BY source_site
ORDER BY queries DESC;
GO


/*==============================================================================
  QUERY 2 -- Next-token distribution.
  The core inference call. Ask the model what comes after a prefix.
==============================================================================*/
SELECT TOP (10)
        token,
        probability = FORMAT(p, 'P2'),
        logit       = ROUND(logit, 3)
FROM    dbo.fn_NextToken(N'SELECT * FROM')
ORDER BY p DESC;

-- How confident is it? Shannon entropy of the same distribution, in bits.
SELECT  entropy_bits = -SUM(p * LOG(p) / LOG(2.0)),
        uniform_bits =  LOG((SELECT V FROM dbo.ModelConfig) * 1.0) / LOG(2.0)
FROM    dbo.fn_NextToken(N'SELECT * FROM');
GO


/*==============================================================================
  QUERY 3 -- Many prompts at once.

  This is the part that is genuinely nicer in SQL than in an imperative
  language: the model is a table-valued function, so CROSS APPLY runs a
  whole batch of prompts and the result is one tidy relation.
==============================================================================*/
SELECT  pr.prompt,
        rank_pos = ROW_NUMBER() OVER (PARTITION BY pr.prompt ORDER BY nt.p DESC),
        nt.token,
        probability = FORMAT(nt.p, 'P1')
FROM    (VALUES
            (N'SELECT'),
            (N'SELECT * FROM'),
            (N'SELECT customer_id,'),
            (N'SELECT COUNT(*) FROM orders GROUP BY'),
            (N'SELECT * FROM customers WHERE'),
            (N'INNER JOIN customers c ON'),
            (N'WITH ranked AS ('),
            (N'UPDATE employees SET')
        ) AS pr(prompt)
CROSS APPLY (
        SELECT TOP (4) token, p
        FROM   dbo.fn_NextToken(pr.prompt)
        ORDER BY p DESC
) AS nt
ORDER BY pr.prompt, nt.p DESC;
GO


/*==============================================================================
  QUERY 4 -- Generation.
  Deterministic first, then sampled at three temperatures.
==============================================================================*/
EXEC dbo.usp_Generate @Prompt = N'SELECT * FROM', @Greedy = 1, @MaxTokens = 40;

EXEC dbo.usp_Generate @Prompt = N'SELECT',        @Temperature = 0.3, @Samples = 3;
EXEC dbo.usp_Generate @Prompt = N'SELECT',        @Temperature = 0.8, @Samples = 3;
EXEC dbo.usp_Generate @Prompt = N'SELECT',        @Temperature = 1.4, @Samples = 3;

EXEC dbo.usp_Generate @Prompt = N'WITH monthly AS (',            @Temperature = 0.7, @Samples = 3;
EXEC dbo.usp_Generate @Prompt = N'SELECT c.customer_name, COUNT(', @Temperature = 0.7, @Samples = 3;
GO


/*==============================================================================
  QUERY 5 -- Scoring.

  fn_Score returns, per token, the probability the model assigned to the
  token that actually appeared. High surprisal marks the spots where the
  query departs from what internet SQL usually looks like.
==============================================================================*/
SELECT  position, token,
        probability   = FORMAT(p, 'P2'),
        surprisal_bits= ROUND(surprisal, 2)
FROM    dbo.fn_Score(N'SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;')
ORDER BY position;

-- Typical query versus deliberately odd query, side by side.
SELECT  q.label,
        tokens      = COUNT(*),
        avg_bits    = ROUND(AVG(s.surprisal), 3),
        perplexity  = ROUND(EXP(-AVG(s.log_p)), 2)
FROM    (VALUES
            (N'idiomatic',  N'SELECT * FROM customers WHERE country = ''germany'';'),
            (N'idiomatic',  N'SELECT category_id, AVG(price) FROM products GROUP BY category_id;'),
            (N'unusual',    N'SELECT GROUP BY customers price WHERE FROM ;'),
            (N'unusual',    N'DELETE products SET orders JOIN AVG ( 3 ;')
        ) AS q(label, sql_text)
CROSS APPLY dbo.fn_Score(q.sql_text) AS s
GROUP BY q.label, q.sql_text
ORDER BY perplexity;
GO


/*==============================================================================
  QUERY 6 -- Perplexity leaderboard over the training corpus.

  Which published queries does the model find most and least predictable?
  The worst-scoring rows are the ones whose patterns appear only once, which
  makes this a serviceable outlier detector for a query log.
==============================================================================*/
WITH scored AS (
    SELECT  c.doc_id, c.source_site, c.query_text,
            tokens     = COUNT(*),
            perplexity = EXP(-AVG(s.log_p))
    FROM    dbo.Corpus AS c
    CROSS APPLY dbo.fn_Score(c.query_text) AS s
    GROUP BY c.doc_id, c.source_site, c.query_text
)
SELECT  band = CASE WHEN rn <= 8 THEN 'most predictable' ELSE 'least predictable' END,
        query_text, tokens, perplexity = ROUND(perplexity, 2)
FROM (
    SELECT *, rn = ROW_NUMBER() OVER (ORDER BY perplexity),
              rd = ROW_NUMBER() OVER (ORDER BY perplexity DESC)
    FROM scored
) AS x
WHERE rn <= 8 OR rd <= 8
ORDER BY perplexity;
GO


/*==============================================================================
  QUERY 7 -- What did the embeddings learn?

  Cosine similarity between token vectors. Restricted to reasonably frequent
  tokens, because a token seen twice has an embedding that barely moved from
  its random initialisation.

  Expect table names to cluster with table names, and aggregate functions to
  cluster with aggregate functions -- the model was never told that these
  categories exist.
==============================================================================*/
DECLARE @probe nvarchar(100) = N'customers';

WITH probe AS (
    SELECT e.dim, e.w
    FROM   dbo.Emb AS e
    JOIN   dbo.Vocab AS v ON v.token_id = e.token_id
    WHERE  v.token = @probe
),
pnorm AS (SELECT nrm = SQRT(SUM(w * w)) FROM probe)
SELECT TOP (10)
        v.token, v.freq,
        cosine = ROUND(SUM(e.w * probe.w)
                       / (SQRT(SUM(e.w * e.w)) * MAX(pnorm.nrm)), 3)
FROM    dbo.Emb AS e
JOIN    probe ON probe.dim = e.dim
JOIN    dbo.Vocab AS v ON v.token_id = e.token_id
CROSS JOIN pnorm
WHERE   v.freq >= 8
  AND   v.token <> @probe
GROUP BY v.token, v.freq
ORDER BY cosine DESC;
GO

-- The same probe, run over several tokens at once.
SELECT  probe.token AS probe_token, nn.token AS neighbour, nn.cosine
FROM    (VALUES (N'customers'), (N'sum'), (N'where'), (N'inner'), (N'group'))
            AS probe(token)
CROSS APPLY (
    SELECT TOP (5)
           v.token,
           cosine = ROUND(SUM(e.w * pe.w)
                          / (SQRT(SUM(e.w * e.w)) * MAX(pn.nrm)), 3)
    FROM   dbo.Emb AS e
    JOIN   dbo.Vocab AS v ON v.token_id = e.token_id
    JOIN  (SELECT e2.dim, e2.w
           FROM   dbo.Emb AS e2
           JOIN   dbo.Vocab AS v2 ON v2.token_id = e2.token_id
           WHERE  v2.token = probe.token) AS pe ON pe.dim = e.dim
    CROSS JOIN (SELECT nrm = SQRT(SUM(e3.w * e3.w))
                FROM   dbo.Emb AS e3
                JOIN   dbo.Vocab AS v3 ON v3.token_id = e3.token_id
                WHERE  v3.token = probe.token) AS pn
    WHERE  v.freq >= 8 AND v.token <> probe.token
    GROUP BY v.token
    ORDER BY cosine DESC
) AS nn
ORDER BY probe.token, nn.cosine DESC;
GO


/*==============================================================================
  QUERY 8 -- Vector arithmetic on the embedding table.

      vec('orders') - vec('order_id') + vec('customer_id')  ~=  ?

  On a corpus this small the result is noisy, and it is honest to say so;
  the point is that the arithmetic is a GROUP BY over a 16-column table.
==============================================================================*/
DECLARE @a nvarchar(100) = N'orders',
        @b nvarchar(100) = N'order_id',
        @c nvarchar(100) = N'customer_id';

WITH target AS (
    SELECT  e.dim,
            w = SUM(CASE v.token WHEN @a THEN e.w WHEN @b THEN -e.w WHEN @c THEN e.w END)
    FROM    dbo.Emb AS e
    JOIN    dbo.Vocab AS v ON v.token_id = e.token_id
    WHERE   v.token IN (@a, @b, @c)
    GROUP BY e.dim
),
tnorm AS (SELECT nrm = SQRT(SUM(w * w)) FROM target)
SELECT TOP (8)
        v.token,
        cosine = ROUND(SUM(e.w * target.w)
                       / (SQRT(SUM(e.w * e.w)) * MAX(tnorm.nrm)), 3)
FROM    dbo.Emb AS e
JOIN    target ON target.dim = e.dim
JOIN    dbo.Vocab AS v ON v.token_id = e.token_id
CROSS JOIN tnorm
WHERE   v.freq >= 5 AND v.token NOT IN (@a, @b, @c)
GROUP BY v.token
ORDER BY cosine DESC;
GO


/*==============================================================================
  QUERY 9 -- Constrained decoding.

  Because the distribution arrives as a rowset, constraining generation to a
  grammar is a WHERE clause plus a renormalisation window. No special decoder
  is needed. Here: after FROM, only allow known table names.
==============================================================================*/
SELECT  token,
        p_raw        = FORMAT(p, 'P3'),
        p_constrained= FORMAT(p / SUM(p) OVER (), 'P1')
FROM    dbo.fn_NextToken(N'SELECT * FROM')
WHERE   token IN (N'customers', N'orders', N'products', N'employees',
                  N'payments', N'departments', N'categories')
ORDER BY p DESC;
GO


/*==============================================================================
  QUERY 10 -- Training diagnostics.
==============================================================================*/
SELECT  epoch, avg_loss = ROUND(avg_loss, 4),
        perplexity = ROUND(perplexity, 3),
        lr = ROUND(lr, 5), elapsed_ms
FROM    dbo.TrainLog
ORDER BY epoch;

-- Loss reduction per epoch, as a window function over the log.
SELECT  epoch,
        perplexity  = ROUND(perplexity, 3),
        delta       = ROUND(perplexity - LAG(perplexity) OVER (ORDER BY epoch), 4)
FROM    dbo.TrainLog
ORDER BY epoch DESC;

-- Weight magnitudes. A hidden unit whose W2 row is near zero has stopped
-- contributing; a very negative B1 usually means a dead ReLU.
SELECT  tensor = 'W1', mean_abs = ROUND(AVG(ABS(w)), 5), max_abs = ROUND(MAX(ABS(w)), 4) FROM dbo.W1
UNION ALL SELECT 'W2',  ROUND(AVG(ABS(w)), 5), ROUND(MAX(ABS(w)), 4) FROM dbo.W2
UNION ALL SELECT 'Emb', ROUND(AVG(ABS(w)), 5), ROUND(MAX(ABS(w)), 4) FROM dbo.Emb;

SELECT  dead_ish_hidden_units = COUNT(*)
FROM    dbo.B1
WHERE   w < -0.5;
GO


/*==============================================================================
  QUERY 11 -- The whole model, as data.

  Nothing here is hidden in a binary blob. This is the entire trained model,
  and it can be backed up, diffed, version-controlled, or shipped by INSERT.
==============================================================================*/
SELECT TOP (20) v.token, e.dim, weight = ROUND(e.w, 4)
FROM    dbo.Emb AS e
JOIN    dbo.Vocab AS v ON v.token_id = e.token_id
WHERE   v.token = N'select'
ORDER BY e.dim;
GO
