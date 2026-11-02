/*==============================================================================
  A SMALL LANGUAGE MODEL, ENTIRELY INSIDE SQL SERVER
  File 03 of 04 : Backpropagation and the training loop

  Gradients, in the order they must be computed:

      dLogit[s,o] = (p[s,o] - 1{o = target[s]}) / B
      dW2[h,o]    = SUM_s a[s,h] * dLogit[s,o]
      dB2[o]      = SUM_s dLogit[s,o]
      dZ[s,h]     = (SUM_o dLogit[s,o] * W2[h,o]) * 1{z[s,h] > 0}
      dW1[i,h]    = SUM_s x[s,i] * dZ[s,h]
      dB1[h]      = SUM_s dZ[s,h]
      dX[s,i]     = SUM_h dZ[s,h] * W1[i,h]
      dEmb[t,d]  += SUM over every (s,slot) where context token = t, of dX

  ORDER MATTERS. dZ needs the OLD W2 and dX needs the OLD W1, so both are
  materialised before either weight matrix is updated. Getting this wrong
  produces a model that trains slowly and silently wrongly.

  Updates use SGD with momentum:  v <- m*v + g ;  w <- w - lr*v
  Both assignments in a single UPDATE read pre-update values, which is
  exactly the semantics required.
==============================================================================*/

USE SqlLlm;
GO
SET NOCOUNT ON;
GO

CREATE OR ALTER PROCEDURE dbo.usp_Train
    @Epochs     int   = 60,
    @BatchSize  int   = 256,
    @Lr         float = 0.15,
    @Momentum   float = 0.9,
    @HalfLife   float = 30,     -- epochs until the learning rate halves
    @GradEps    float = 0,      -- optional: prune |dLogit| below this. 0 = exact.
    @Verbose    bit   = 1
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    DECLARE @C int, @D int, @H int, @V int;
    SELECT @C = C, @D = D, @H = H, @V = V FROM dbo.ModelConfig;

    IF NOT EXISTS (SELECT 1 FROM dbo.Emb)
        THROW 50002, 'Parameters are not initialised. Run dbo.usp_InitParams (file 02).', 1;

    /*---------------------------------------------------------------
      Working tables. Created once, truncated per batch, so the plans
      are compiled once and the indexes survive the loop.
    ---------------------------------------------------------------*/
    DROP TABLE IF EXISTS #Ctx;  DROP TABLE IF EXISTS #X;   DROP TABLE IF EXISTS #A;
    DROP TABLE IF EXISTS #P;    DROP TABLE IF EXISTS #DL;  DROP TABLE IF EXISTS #DZ;
    DROP TABLE IF EXISTS #DX;

    CREATE TABLE #Ctx (seq_id int NOT NULL, slot smallint NOT NULL, token_id int NOT NULL,
                       PRIMARY KEY (seq_id, slot));

    CREATE TABLE #X   (seq_id int NOT NULL, i int NOT NULL, val float NOT NULL,
                       PRIMARY KEY (seq_id, i));
    CREATE INDEX IX_X_i ON #X (i) INCLUDE (seq_id, val);

    CREATE TABLE #A   (seq_id int NOT NULL, hid int NOT NULL, act float NOT NULL,
                       PRIMARY KEY (seq_id, hid));

    CREATE TABLE #P   (seq_id int NOT NULL, out_id int NOT NULL, p float NOT NULL,
                       PRIMARY KEY (seq_id, out_id));

    CREATE TABLE #DL  (seq_id int NOT NULL, out_id int NOT NULL, g float NOT NULL,
                       PRIMARY KEY (seq_id, out_id));
    CREATE INDEX IX_DL_out ON #DL (out_id) INCLUDE (seq_id, g);

    CREATE TABLE #DZ  (seq_id int NOT NULL, hid int NOT NULL, dz float NOT NULL,
                       PRIMARY KEY (seq_id, hid));
    CREATE INDEX IX_DZ_hid ON #DZ (hid) INCLUDE (seq_id, dz);

    CREATE TABLE #DX  (seq_id int NOT NULL, i int NOT NULL, grad float NOT NULL,
                       PRIMARY KEY (seq_id, i));

    /* Note: T-SQL variable names are case-insensitive, so @batchNo and @batchRows would be
       the same variable. Batch index and batch size get distinct names. */
    /* NOTE: T-SQL compares variable names case-insensitively, regardless of
       database collation. So @b and @B are the same variable, and a local
       @lr would collide with the @Lr parameter above. Every name here is
       distinct in more than just case. */
    DECLARE @batchNo    int,
            @maxBatchNo int,
            @batchRows  int,
            @epoch      int = 0,
            @curLr      float,
            @epochLoss  float,
            @rows       int,
            @t0         datetime2(3),
            @msg        nvarchar(200);

    DECLARE @startEpoch int = ISNULL((SELECT MAX(epoch) FROM dbo.TrainLog), 0);

    WHILE @epoch < @Epochs
    BEGIN
        SET @t0 = SYSDATETIME();
        SET @curLr = @Lr * POWER(0.5, (@epoch + @startEpoch) * 1.0 / @HalfLife);
        SET @epochLoss = 0;

        /*-----------------------------------------------------------
          Reshuffle the training rows into fresh mini-batches.
        -----------------------------------------------------------*/
        ;WITH shuffled AS (
            SELECT batch_id,
                   nb = (ROW_NUMBER() OVER (ORDER BY NEWID()) - 1) / @BatchSize
            FROM   dbo.TrainSeq
        )
        UPDATE shuffled SET batch_id = nb;

        SELECT @maxBatchNo = MAX(batch_id), @rows = COUNT(*) FROM dbo.TrainSeq;
        SET @batchNo = 0;

        WHILE @batchNo <= @maxBatchNo
        BEGIN
            TRUNCATE TABLE #Ctx;  TRUNCATE TABLE #X;   TRUNCATE TABLE #A;
            TRUNCATE TABLE #P;    TRUNCATE TABLE #DL;  TRUNCATE TABLE #DZ;
            TRUNCATE TABLE #DX;

            /*=======================  FORWARD  =======================*/

            INSERT INTO #Ctx (seq_id, slot, token_id)
            SELECT c.seq_id, c.slot, c.token_id
            FROM   dbo.TrainSeq AS t
            JOIN   dbo.TrainCtx AS c ON c.seq_id = t.seq_id
            WHERE  t.batch_id = @batchNo;

            SELECT @batchRows = COUNT(*) FROM dbo.TrainSeq WHERE batch_id = @batchNo;
            IF @batchRows = 0 BEGIN SET @batchNo += 1; CONTINUE; END;

            -- x : concatenated embeddings, one row per (example, input index)
            INSERT INTO #X (seq_id, i, val)
            SELECT c.seq_id, c.slot * @D + e.dim, e.w
            FROM   #Ctx AS c
            JOIN   dbo.Emb AS e ON e.token_id = c.token_id;

            -- a : hidden activations after ReLU
            INSERT INTO #A (seq_id, hid, act)
            SELECT s.seq_id, s.hid,
                   CASE WHEN s.z + b1.w > 0 THEN s.z + b1.w ELSE 0 END
            FROM  (SELECT x.seq_id, w1.hid, z = SUM(x.val * w1.w)
                   FROM   #X AS x
                   JOIN   dbo.W1 AS w1 ON w1.inp = x.i
                   GROUP BY x.seq_id, w1.hid) AS s
            JOIN   dbo.B1 AS b1 ON b1.hid = s.hid;

            -- p : softmax over the vocabulary, per example.
            -- Two scopes are needed: a windowed function may not appear
            -- inside the argument of another windowed function.
            INSERT INTO #P (seq_id, out_id, p)
            SELECT  ex.seq_id, ex.out_id,
                    ex.e / SUM(ex.e) OVER (PARTITION BY ex.seq_id)
            FROM (
            SELECT  z.seq_id, z.out_id,
                    e = EXP(z.logit - MAX(z.logit) OVER (PARTITION BY z.seq_id))
            FROM (
                SELECT  r.seq_id, b2.out_id, logit = b2.w + ISNULL(s.s, 0)
                FROM    dbo.TrainSeq AS r
                CROSS JOIN dbo.B2 AS b2
                LEFT JOIN (
                        SELECT a.seq_id, w2.out_id, s = SUM(a.act * w2.w)
                        FROM   #A AS a
                        JOIN   dbo.W2 AS w2 ON w2.hid = a.hid
                        WHERE  a.act > 0
                        GROUP BY a.seq_id, w2.out_id
                ) AS s ON s.seq_id = r.seq_id AND s.out_id = b2.out_id
                WHERE   r.batch_id = @batchNo
            ) AS z
            ) AS ex;

            -- cross-entropy on this batch
            SELECT @epochLoss = @epochLoss
                 + SUM(-LOG(CASE WHEN p.p < 1e-300 THEN 1e-300 ELSE p.p END))
            FROM   #P AS p
            JOIN   dbo.TrainSeq AS t ON t.seq_id = p.seq_id AND t.target = p.out_id;

            /*=======================  BACKWARD  ======================*/

            -- dLogit = (p - onehot) / B
            INSERT INTO #DL (seq_id, out_id, g)
            SELECT p.seq_id, p.out_id,
                   (p.p - CASE WHEN t.target = p.out_id THEN 1.0 ELSE 0.0 END) / @batchRows
            FROM   #P AS p
            JOIN   dbo.TrainSeq AS t ON t.seq_id = p.seq_id
            WHERE  @GradEps = 0
               OR  ABS(p.p - CASE WHEN t.target = p.out_id THEN 1.0 ELSE 0.0 END) / @batchRows > @GradEps;

            -- dZ, using the CURRENT (not yet updated) W2, masked by the ReLU
            INSERT INTO #DZ (seq_id, hid, dz)
            SELECT s.seq_id, s.hid, s.da
            FROM  (SELECT d.seq_id, w2.hid, da = SUM(d.g * w2.w)
                   FROM   #DL AS d
                   JOIN   dbo.W2 AS w2 ON w2.out_id = d.out_id
                   GROUP BY d.seq_id, w2.hid) AS s
            JOIN   #A AS a ON a.seq_id = s.seq_id AND a.hid = s.hid
            WHERE  a.act > 0;

            -- dX, using the CURRENT (not yet updated) W1
            INSERT INTO #DX (seq_id, i, grad)
            SELECT d.seq_id, w1.inp, SUM(d.dz * w1.w)
            FROM   #DZ AS d
            JOIN   dbo.W1 AS w1 ON w1.hid = d.hid
            GROUP BY d.seq_id, w1.inp;

            /*=======================  UPDATE  ========================*/

            ;WITH g AS (
                SELECT a.hid, d.out_id, gr = SUM(a.act * d.g)
                FROM   #A AS a
                JOIN   #DL AS d ON d.seq_id = a.seq_id
                GROUP BY a.hid, d.out_id
            )
            UPDATE t
            SET    v = @Momentum * t.v + g.gr,
                   w = t.w - @curLr * (@Momentum * t.v + g.gr)
            FROM   dbo.W2 AS t
            JOIN   g ON g.hid = t.hid AND g.out_id = t.out_id;

            ;WITH g AS (SELECT out_id, gr = SUM(g) FROM #DL GROUP BY out_id)
            UPDATE t
            SET    v = @Momentum * t.v + g.gr,
                   w = t.w - @curLr * (@Momentum * t.v + g.gr)
            FROM   dbo.B2 AS t
            JOIN   g ON g.out_id = t.out_id;

            ;WITH g AS (
                SELECT x.i, d.hid, gr = SUM(x.val * d.dz)
                FROM   #X AS x
                JOIN   #DZ AS d ON d.seq_id = x.seq_id
                GROUP BY x.i, d.hid
            )
            UPDATE t
            SET    v = @Momentum * t.v + g.gr,
                   w = t.w - @curLr * (@Momentum * t.v + g.gr)
            FROM   dbo.W1 AS t
            JOIN   g ON g.i = t.inp AND g.hid = t.hid;

            ;WITH g AS (SELECT hid, gr = SUM(dz) FROM #DZ GROUP BY hid)
            UPDATE t
            SET    v = @Momentum * t.v + g.gr,
                   w = t.w - @curLr * (@Momentum * t.v + g.gr)
            FROM   dbo.B1 AS t
            JOIN   g ON g.hid = t.hid;

            -- Scatter dX back onto the embedding rows it came from.
            -- i / D recovers the context slot; i % D recovers the dimension.
            ;WITH g AS (
                SELECT c.token_id, dim = dx.i % @D, gr = SUM(dx.grad)
                FROM   #DX AS dx
                JOIN   #Ctx AS c ON c.seq_id = dx.seq_id AND c.slot = dx.i / @D
                GROUP BY c.token_id, dx.i % @D
            )
            UPDATE t
            SET    v = @Momentum * t.v + g.gr,
                   w = t.w - @curLr * (@Momentum * t.v + g.gr)
            FROM   dbo.Emb AS t
            JOIN   g ON g.token_id = t.token_id AND g.dim = t.dim;

            SET @batchNo += 1;
        END;

        INSERT INTO dbo.TrainLog (epoch, avg_loss, lr, elapsed_ms)
        VALUES (@epoch + @startEpoch + 1, @epochLoss / @rows, @curLr,
                DATEDIFF(millisecond, @t0, SYSDATETIME()));

        IF @Verbose = 1
        BEGIN
            SET @msg = CONCAT('epoch ', @epoch + @startEpoch + 1,
                              '  loss ', FORMAT(@epochLoss / @rows, 'N4'),
                              '  ppl ',  FORMAT(EXP(@epochLoss / @rows), 'N2'),
                              '  lr ',   FORMAT(@curLr, 'N4'),
                              '  ',      DATEDIFF(millisecond, @t0, SYSDATETIME()), ' ms');
            RAISERROR (@msg, 0, 1) WITH NOWAIT;
        END;

        SET @epoch += 1;
    END;

    SELECT epoch, avg_loss, perplexity, lr, elapsed_ms FROM dbo.TrainLog ORDER BY epoch;
END;
GO

/*==============================================================================
  Run it.

  Expect roughly:   epoch  1  loss 5.3  ppl 200
                    epoch 15  loss 2.1  ppl   8
                    epoch 30  loss 1.5  ppl   4.4
                    epoch 60  loss 1.05 ppl   2.9

  Cost is dominated by the batch x hidden x vocabulary joins, about 2 billion
  row operations in total for the defaults below. On a modest instance that is
  a handful of minutes. Doubling @Epochs to 120 gets perplexity to roughly 2.0
  and noticeably cleaner generations.

  If you want it faster: raise @BatchSize, drop @Epochs, or set @GradEps to
  something like 1e-6 to skip the vast number of near-zero output gradients.
==============================================================================*/

EXEC dbo.usp_Train @Epochs = 60, @BatchSize = 256, @Lr = 0.15, @Momentum = 0.9, @HalfLife = 30;
GO

/* The model has now stopped being random. Compare against file 02's smoke test. */
SELECT TOP 8 token, p FROM dbo.fn_NextToken(N'select * from') ORDER BY p DESC;
GO
