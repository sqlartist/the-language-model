/*==============================================================================
  A SMALL LANGUAGE MODEL, ENTIRELY INSIDE SQL SERVER
  File 02 of 04 : Weight tables, initialisation, forward pass

  The model in full:

      slot k in 0..C-1 holds token t_k
      x[k*D + d]  = Emb[t_k, d]                        -- concatenated embeddings
      z[h]        = SUM_i x[i] * W1[i,h] + B1[h]
      a[h]        = MAX(z[h], 0)                       -- ReLU
      logit[o]    = SUM_h a[h] * W2[h,o] + B2[o]
      p[o]        = EXP(logit[o] - max) / SUM(...)     -- softmax

  Every one of those lines is a GROUP BY over a join. That is the whole trick.
==============================================================================*/

USE SqlLlm;
GO
SET NOCOUNT ON;
GO

/*------------------------------------------------------------------
  1. The parameter tables. Each carries a companion column `v`
     holding the SGD momentum velocity for that parameter.
------------------------------------------------------------------*/
DROP TABLE IF EXISTS dbo.Emb;
DROP TABLE IF EXISTS dbo.W1;
DROP TABLE IF EXISTS dbo.B1;
DROP TABLE IF EXISTS dbo.W2;
DROP TABLE IF EXISTS dbo.B2;
DROP TABLE IF EXISTS dbo.TrainLog;

CREATE TABLE dbo.Emb                     -- token embeddings          [V x D]
(
    token_id int   NOT NULL,
    dim      int   NOT NULL,
    w        float NOT NULL,
    v        float NOT NULL CONSTRAINT DF_Emb_v DEFAULT (0),
    CONSTRAINT PK_Emb PRIMARY KEY (token_id, dim)
);

CREATE TABLE dbo.W1                      -- input -> hidden           [C*D x H]
(
    inp int   NOT NULL,
    hid int   NOT NULL,
    w   float NOT NULL,
    v   float NOT NULL CONSTRAINT DF_W1_v DEFAULT (0),
    CONSTRAINT PK_W1 PRIMARY KEY (inp, hid)
);
CREATE INDEX IX_W1_hid ON dbo.W1 (hid) INCLUDE (inp, w);

CREATE TABLE dbo.B1
(
    hid int   NOT NULL CONSTRAINT PK_B1 PRIMARY KEY,
    w   float NOT NULL,
    v   float NOT NULL CONSTRAINT DF_B1_v DEFAULT (0)
);

CREATE TABLE dbo.W2                      -- hidden -> vocabulary      [H x V]
(
    hid    int   NOT NULL,
    out_id int   NOT NULL,
    w      float NOT NULL,
    v      float NOT NULL CONSTRAINT DF_W2_v DEFAULT (0),
    CONSTRAINT PK_W2 PRIMARY KEY (hid, out_id)
);
CREATE INDEX IX_W2_out ON dbo.W2 (out_id) INCLUDE (hid, w);

CREATE TABLE dbo.B2
(
    out_id int   NOT NULL CONSTRAINT PK_B2 PRIMARY KEY,
    w      float NOT NULL,
    v      float NOT NULL CONSTRAINT DF_B2_v DEFAULT (0)
);

CREATE TABLE dbo.TrainLog
(
    log_id     int IDENTITY(1,1) NOT NULL CONSTRAINT PK_TrainLog PRIMARY KEY,
    epoch      int      NOT NULL,
    avg_loss   float    NOT NULL,
    perplexity AS (EXP(avg_loss)),
    lr         float    NOT NULL,
    elapsed_ms int      NOT NULL,
    logged_at  datetime2(0) NOT NULL CONSTRAINT DF_TrainLog_At DEFAULT (SYSUTCDATETIME())
);
GO

/*------------------------------------------------------------------
  2. Initialisation.

     RAND() is constant for the whole statement, so it is useless for
     filling a table. CHECKSUM(NEWID()) is evaluated per row, which is
     exactly what is needed.

     He initialisation is used for the ReLU layers: scale = sqrt(2/fan_in).
------------------------------------------------------------------*/
CREATE OR ALTER PROCEDURE dbo.usp_InitParams
    @EmbScale float = 0.5
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @C int, @D int, @H int, @V int;
    SELECT @C = C, @D = D, @H = H, @V = V FROM dbo.ModelConfig;

    IF @V = 0
        THROW 50001, 'Vocabulary is empty. Run dbo.usp_BuildDataset first (file 01).', 1;

    DELETE FROM dbo.Emb;  DELETE FROM dbo.W1;  DELETE FROM dbo.B1;
    DELETE FROM dbo.W2;   DELETE FROM dbo.B2;

    -- uniform(-1,1), evaluated per row:  (CHECKSUM(NEWID()) % 1000001) / 1000000.0
    -- The modulus keeps the sign of the operand, so no ABS() is needed --
    -- and ABS() would overflow on the one CHECKSUM value equal to -2^31.

    INSERT INTO dbo.Emb (token_id, dim, w)
    SELECT v.token_id, d.n,
           (CHECKSUM(NEWID()) % 1000001) / 1000000.0 * @EmbScale
    FROM   dbo.Vocab AS v
    CROSS JOIN (SELECT n FROM dbo.Nums WHERE n < @D) AS d;

    INSERT INTO dbo.W1 (inp, hid, w)
    SELECT i.n, h.n,
           (CHECKSUM(NEWID()) % 1000001) / 1000000.0
             * SQRT(2.0 / (@C * @D))
    FROM   (SELECT n FROM dbo.Nums WHERE n < @C * @D) AS i
    CROSS JOIN (SELECT n FROM dbo.Nums WHERE n < @H) AS h;

    INSERT INTO dbo.B1 (hid, w) SELECT n, 0 FROM dbo.Nums WHERE n < @H;

    INSERT INTO dbo.W2 (hid, out_id, w)
    SELECT h.n, v.token_id,
           (CHECKSUM(NEWID()) % 1000001) / 1000000.0
             * SQRT(2.0 / @H)
    FROM   (SELECT n FROM dbo.Nums WHERE n < @H) AS h
    CROSS JOIN dbo.Vocab AS v;

    INSERT INTO dbo.B2 (out_id, w) SELECT token_id, 0 FROM dbo.Vocab;

    DELETE FROM dbo.TrainLog;

    /* B1 and B2 are vectors, so their "shape" is a single number. CONCAT
       requires at least two arguments, hence CAST rather than CONCAT here. */
    SELECT tensor = 'Emb', shape = CONCAT(@V, ' x ', @D), parameters = COUNT(*) FROM dbo.Emb
    UNION ALL SELECT 'W1', CONCAT(@C * @D, ' x ', @H),  COUNT(*) FROM dbo.W1
    UNION ALL SELECT 'B1', CAST(@H AS varchar(20)),     COUNT(*) FROM dbo.B1
    UNION ALL SELECT 'W2', CONCAT(@H, ' x ', @V),       COUNT(*) FROM dbo.W2
    UNION ALL SELECT 'B2', CAST(@V AS varchar(20)),     COUNT(*) FROM dbo.B2;
END;
GO

EXEC dbo.usp_InitParams;
GO

/*==============================================================================
  3. THE FORWARD PASS

  fn_ContextIds  : text -> the last C token ids, left-padded with <s>
  fn_NextToken   : text -> probability distribution over the whole vocabulary
  fn_Score       : text -> per-token log probability (for perplexity/scoring)

  All three are inline table-valued functions, so they compose into ordinary
  queries and the optimiser can inline them into the calling plan.
==============================================================================*/

CREATE OR ALTER FUNCTION dbo.fn_ContextIds (@txt nvarchar(max))
RETURNS TABLE
AS
RETURN
    WITH cfg AS (SELECT C FROM dbo.ModelConfig),
         tok AS (SELECT pos, token FROM dbo.fn_Tokenize(@txt)),
         cnt AS (SELECT c = ISNULL((SELECT MAX(pos) FROM tok), 0))
    SELECT  slot     = n.n,
            token_id = CASE
                          WHEN t.token IS NULL                        -- before the start
                               THEN (SELECT token_id FROM dbo.Vocab WHERE token = N'<s>')
                          ELSE COALESCE(v.token_id,
                               (SELECT token_id FROM dbo.Vocab WHERE token = N'<unk>'))
                       END
    FROM    cfg
    CROSS JOIN cnt
    JOIN    dbo.Nums AS n ON n.n < cfg.C
    LEFT JOIN tok       AS t ON t.pos   = cnt.c - cfg.C + n.n + 1
    LEFT JOIN dbo.Vocab AS v ON v.token = t.token;
GO

/*------------------------------------------------------------------
  fn_NextToken: the complete forward pass in one statement.
------------------------------------------------------------------*/
CREATE OR ALTER FUNCTION dbo.fn_NextToken (@txt nvarchar(max))
RETURNS TABLE
AS
RETURN
    WITH cfg AS (SELECT D FROM dbo.ModelConfig),

         /* x : the concatenated embedding vector, length C*D */
         x AS (
            SELECT  i = c.slot * cfg.D + e.dim,
                    val = e.w
            FROM    dbo.fn_ContextIds(@txt) AS c
            CROSS JOIN cfg
            JOIN    dbo.Emb AS e ON e.token_id = c.token_id
         ),

         /* z, a : hidden pre-activation and ReLU activation */
         a AS (
            SELECT  s.hid,
                    act = CASE WHEN s.z + b.w > 0 THEN s.z + b.w ELSE 0 END
            FROM   (SELECT w1.hid, z = SUM(x.val * w1.w)
                    FROM   x JOIN dbo.W1 AS w1 ON w1.inp = x.i
                    GROUP BY w1.hid) AS s
            JOIN    dbo.B1 AS b ON b.hid = s.hid
         ),

         /* logits : one per vocabulary entry. Driven from B2 with a LEFT JOIN
            so that a fully dead hidden layer still yields a complete row set. */
         lg AS (
            SELECT  b.out_id,
                    logit = b.w + ISNULL(s.s, 0)
            FROM    dbo.B2 AS b
            LEFT JOIN (
                    SELECT  w2.out_id, s = SUM(a.act * w2.w)
                    FROM    a JOIN dbo.W2 AS w2 ON w2.hid = a.hid
                    WHERE   a.act > 0
                    GROUP BY w2.out_id
            ) AS s ON s.out_id = b.out_id
         ),

         /* softmax, with the usual max-subtraction for numerical safety */
         ex AS (
            SELECT  out_id, logit,
                    e = EXP(logit - MAX(logit) OVER ())
            FROM    lg
         )
    SELECT  token_id = ex.out_id,
            token    = v.token,
            logit    = ex.logit,
            p        = ex.e / SUM(ex.e) OVER ()
    FROM    ex
    JOIN    dbo.Vocab AS v ON v.token_id = ex.out_id;
GO

/*------------------------------------------------------------------
  fn_Score: runs the forward pass at EVERY position of a query at once
  and returns the log probability the model assigned to the token that
  actually appeared. Sum them for total log-likelihood; negate and
  average, then exponentiate, for perplexity.

  This is the batched version of the same algebra -- note that adding a
  batch dimension means nothing more than carrying `pos` through the
  GROUP BY clauses.
------------------------------------------------------------------*/
CREATE OR ALTER FUNCTION dbo.fn_Score (@txt nvarchar(max))
RETURNS TABLE
AS
RETURN
    WITH cfg AS (SELECT C, D FROM dbo.ModelConfig),
         tok AS (SELECT pos, token FROM dbo.fn_Tokenize(@txt)),

         /* the token sequence, with an explicit end-of-query target */
         seq AS (
            SELECT pos, token FROM tok
            UNION ALL
            SELECT ISNULL((SELECT MAX(pos) FROM tok), 0) + 1, N'</s>'
         ),
         ids AS (
            SELECT  s.pos, s.token,
                    token_id = COALESCE(v.token_id,
                               (SELECT token_id FROM dbo.Vocab WHERE token = N'<unk>'))
            FROM    seq AS s
            LEFT JOIN dbo.Vocab AS v ON v.token = s.token
         ),

         /* every target position gets its own C context slots */
         ctx AS (
            SELECT  tgt.pos,
                    slot = n.n,
                    token_id = CASE WHEN prev.token_id IS NULL
                                    THEN (SELECT token_id FROM dbo.Vocab WHERE token = N'<s>')
                                    ELSE prev.token_id END
            FROM    ids AS tgt
            CROSS JOIN cfg
            JOIN    dbo.Nums AS n ON n.n < cfg.C
            LEFT JOIN ids AS prev ON prev.pos = tgt.pos - cfg.C + n.n
         ),

         x AS (
            SELECT  c.pos, i = c.slot * cfg.D + e.dim, val = e.w
            FROM    ctx AS c
            CROSS JOIN cfg
            JOIN    dbo.Emb AS e ON e.token_id = c.token_id
         ),
         a AS (
            SELECT  s.pos, s.hid,
                    act = CASE WHEN s.z + b.w > 0 THEN s.z + b.w ELSE 0 END
            FROM   (SELECT x.pos, w1.hid, z = SUM(x.val * w1.w)
                    FROM   x JOIN dbo.W1 AS w1 ON w1.inp = x.i
                    GROUP BY x.pos, w1.hid) AS s
            JOIN    dbo.B1 AS b ON b.hid = s.hid
         ),
         lg AS (
            SELECT  r.pos, b.out_id, logit = b.w + ISNULL(s.s, 0)
            FROM    (SELECT DISTINCT pos FROM ctx) AS r
            CROSS JOIN dbo.B2 AS b
            LEFT JOIN (
                    SELECT  a.pos, w2.out_id, s = SUM(a.act * w2.w)
                    FROM    a JOIN dbo.W2 AS w2 ON w2.hid = a.hid
                    WHERE   a.act > 0
                    GROUP BY a.pos, w2.out_id
            ) AS s ON s.pos = r.pos AND s.out_id = b.out_id
         ),
         /* Softmax in two steps. SQL Server will not allow a windowed
            function inside the argument of another windowed function, so
            the max-subtraction and the normalising sum need separate scopes. */
         ex AS (
            SELECT  pos, out_id,
                    e = EXP(logit - MAX(logit) OVER (PARTITION BY pos))
            FROM    lg
         ),
         pr AS (
            SELECT  pos, out_id,
                    p = e / SUM(e) OVER (PARTITION BY pos)
            FROM    ex
         )
    SELECT  position    = i.pos,
            token       = i.token,
            p           = pr.p,
            log_p       = LOG(CASE WHEN pr.p < 1e-300 THEN 1e-300 ELSE pr.p END),
            surprisal   = -LOG(CASE WHEN pr.p < 1e-300 THEN 1e-300 ELSE pr.p END)
                          / LOG(2.0)                       -- bits
    FROM    ids AS i
    JOIN    pr  ON pr.pos = i.pos AND pr.out_id = i.token_id;
GO

/*------------------------------------------------------------------
  Cosmetic detokeniser: turns "count ( * ) ;" back into "count(*);"
  It is presentation only -- the model never sees this.
------------------------------------------------------------------*/
CREATE OR ALTER FUNCTION dbo.fn_Detokenise (@txt nvarchar(max))
RETURNS nvarchar(max)
AS
BEGIN
    DECLARE @s nvarchar(max) = @txt;
    SET @s = REPLACE(@s, NCHAR(39) + N' ', NCHAR(39));      -- quote then space
    SET @s = REPLACE(@s, N' ' + NCHAR(39), NCHAR(39));      -- space then quote
    SET @s = REPLACE(@s, N' ,', N',');
    SET @s = REPLACE(@s, N' ;', N';');
    SET @s = REPLACE(@s, N' )', N')');
    SET @s = REPLACE(@s, N'( ', N'(');
    SET @s = REPLACE(@s, N' .', N'.');
    SET @s = REPLACE(@s, N'. ', N'.');
    SET @s = REPLACE(@s, N' <s>', N'');
    SET @s = REPLACE(@s, N'</s>', N'');
    WHILE CHARINDEX(N'  ', @s) > 0 SET @s = REPLACE(@s, N'  ', N' ');
    RETURN LTRIM(RTRIM(@s));
END;
GO

/* Smoke test on the untrained model: the distribution should be roughly flat,
   i.e. every token near 1/V. This confirms the plumbing before any training. */
SELECT TOP 5 token, p FROM dbo.fn_NextToken(N'select * from') ORDER BY p DESC;
SELECT uniform_baseline = 1.0 / (SELECT V FROM dbo.ModelConfig);
GO
