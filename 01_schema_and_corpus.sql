/*==============================================================================
  A SMALL LANGUAGE MODEL, ENTIRELY INSIDE SQL SERVER
  File 01 of 04 : Schema, corpus, tokeniser, training-set construction
  Subject       : "examples of SQL queries available on the internet"

  Architecture  : Bengio-style neural language model
                    context of C tokens -> embedding lookup -> concatenate
                    -> dense layer (ReLU) -> output layer -> softmax over vocab
  No CLR, no Python, no external services. Weights live in tables; the forward
  pass is a join; training is a sequence of set-based UPDATE statements.

  Tested design targets: SQL Server 2016 (compat 130) or later, incl. Azure SQL.
==============================================================================*/

IF DB_ID(N'SqlLlm') IS NULL
    EXEC (N'CREATE DATABASE SqlLlm;');
GO
USE SqlLlm;
GO
SET NOCOUNT ON;
GO

/*------------------------------------------------------------------
  1. Hyperparameters. One row. Every function joins to this instead
     of hard-coding dimensions, so the model is resizable.
------------------------------------------------------------------*/
DROP TABLE IF EXISTS dbo.ModelConfig;
CREATE TABLE dbo.ModelConfig
(
    lock_col  char(1)  NOT NULL CONSTRAINT PK_ModelConfig PRIMARY KEY
                       CONSTRAINT CK_ModelConfig_OneRow CHECK (lock_col = 'X'),
    C         int      NOT NULL,   -- context length, in tokens
    D         int      NOT NULL,   -- embedding dimension per token
    H         int      NOT NULL,   -- hidden units
    V         int      NOT NULL    -- vocabulary size (filled in by usp_BuildDataset)
);
INSERT INTO dbo.ModelConfig (lock_col, C, D, H, V) VALUES ('X', 3, 16, 32, 0);
GO

/*------------------------------------------------------------------
  2. A numbers table. Used to fan single rows out into dimensions.
------------------------------------------------------------------*/
DROP TABLE IF EXISTS dbo.Nums;
CREATE TABLE dbo.Nums (n int NOT NULL CONSTRAINT PK_Nums PRIMARY KEY);

WITH e1(x) AS (SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 1
               UNION ALL SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 1),
     e2(x) AS (SELECT 1 FROM e1 a CROSS JOIN e1 b),
     e3(x) AS (SELECT 1 FROM e2 a CROSS JOIN e2 b)
INSERT INTO dbo.Nums (n)
SELECT TOP (8192) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1 FROM e3;
GO

/*------------------------------------------------------------------
  3. The corpus: SQL queries of the kind published on the internet.
     This is the training data. Add your own rows and rebuild to
     retrain the model on a different flavour of SQL.
------------------------------------------------------------------*/
DROP TABLE IF EXISTS dbo.Corpus;
CREATE TABLE dbo.Corpus
(
    doc_id      int IDENTITY(1,1) NOT NULL CONSTRAINT PK_Corpus PRIMARY KEY,
    source_site nvarchar(100)  NOT NULL,
    query_text  nvarchar(1000) NOT NULL
);
GO

INSERT INTO dbo.Corpus (source_site, query_text) VALUES
(N'w3schools.com/sql', N'select * from customers;'),
(N'stackoverflow.com', N'select * from employees;'),
(N'sqlservertutorial.net', N'select * from orders;'),
(N'learn.microsoft.com', N'select top 10 * from products;'),
(N'geeksforgeeks.org', N'select first_name, last_name from employees;'),
(N'sqlshack.com', N'select customer_id, customer_name from customers;'),
(N'w3schools.com/sql', N'select product_name, price from products;'),
(N'stackoverflow.com', N'select distinct country from customers;'),
(N'sqlservertutorial.net', N'select distinct city from suppliers;'),
(N'learn.microsoft.com', N'select count(*) from orders;'),
(N'geeksforgeeks.org', N'select count(*) from customers;'),
(N'sqlshack.com', N'select count(distinct customer_id) from orders;'),
(N'w3schools.com/sql', N'select sum(amount) from payments;'),
(N'stackoverflow.com', N'select avg(price) from products;'),
(N'sqlservertutorial.net', N'select min(price) from products;'),
(N'learn.microsoft.com', N'select max(salary) from employees;'),
(N'geeksforgeeks.org', N'select * from customers where country = ''germany'';'),
(N'sqlshack.com', N'select * from customers where city = ''london'';'),
(N'w3schools.com/sql', N'select * from products where price > 100;'),
(N'stackoverflow.com', N'select * from products where price < 50;'),
(N'sqlservertutorial.net', N'select * from employees where salary >= 50000;'),
(N'learn.microsoft.com', N'select * from orders where order_date > ''2023-01-01'';'),
(N'geeksforgeeks.org', N'select * from customers where country = ''usa'' and city = ''seattle'';'),
(N'sqlshack.com', N'select * from products where price between 10 and 20;'),
(N'w3schools.com/sql', N'select * from employees where department_id in(1, 2, 3);'),
(N'stackoverflow.com', N'select * from customers where customer_name like ''a%'';'),
(N'sqlservertutorial.net', N'select * from customers where postal_code is null;'),
(N'learn.microsoft.com', N'select * from employees where manager_id is not null;'),
(N'geeksforgeeks.org', N'select * from orders where status <> ''cancelled'';'),
(N'sqlshack.com', N'select * from products where category_id = 1 or category_id = 2;'),
(N'w3schools.com/sql', N'select * from products order by price;'),
(N'stackoverflow.com', N'select * from products order by price desc;'),
(N'sqlservertutorial.net', N'select * from employees order by last_name asc;'),
(N'learn.microsoft.com', N'select * from orders order by order_date desc;'),
(N'geeksforgeeks.org', N'select top 5 * from products order by price desc;'),
(N'sqlshack.com', N'select customer_id, count(*) from orders group by customer_id;'),
(N'w3schools.com/sql', N'select country, count(*) from customers group by country;'),
(N'stackoverflow.com', N'select category_id, avg(price) from products group by category_id;'),
(N'sqlservertutorial.net', N'select department_id, sum(salary) from employees group by department_id;'),
(N'learn.microsoft.com', N'select customer_id, sum(amount) from payments group by customer_id;'),
(N'geeksforgeeks.org', N'select country, count(*) from customers group by country order by count(*) desc;'),
(N'sqlshack.com', N'select customer_id, count(*) from orders group by customer_id having count(*) > 5;'),
(N'w3schools.com/sql', N'select category_id, avg(price) from products group by category_id having avg(price) > 20;'),
(N'stackoverflow.com', N'select department_id, count(*) from employees group by department_id having count(*) >= 10;'),
(N'sqlservertutorial.net', N'select o.order_id, c.customer_name from orders o inner join customers c on o.customer_id = c.customer_id;'),
(N'learn.microsoft.com', N'select c.customer_name, o.order_id from customers c left join orders o on c.customer_id = o.customer_id;'),
(N'geeksforgeeks.org', N'select e.first_name, d.department_name from employees e inner join departments d on e.department_id = d.department_id;'),
(N'sqlshack.com', N'select p.product_name, c.category_name from products p inner join categories c on p.category_id = c.category_id;'),
(N'w3schools.com/sql', N'select o.order_id, p.product_name from orders o inner join order_items i on o.order_id = i.order_id inner join products p on i.product_id = p.product_id;'),
(N'stackoverflow.com', N'select c.customer_name, count(o.order_id) from customers c left join orders o on c.customer_id = o.customer_id group by c.customer_name;'),
(N'sqlservertutorial.net', N'select e.first_name, m.first_name from employees e left join employees m on e.manager_id = m.employee_id;'),
(N'learn.microsoft.com', N'select * from orders o right join customers c on o.customer_id = c.customer_id;'),
(N'geeksforgeeks.org', N'select * from products p full outer join categories c on p.category_id = c.category_id;'),
(N'sqlshack.com', N'select * from customers where customer_id in(select customer_id from orders);'),
(N'w3schools.com/sql', N'select * from products where price >(select avg(price) from products);'),
(N'stackoverflow.com', N'select * from employees where salary >(select avg(salary) from employees);'),
(N'sqlservertutorial.net', N'select * from customers where exists(select 1 from orders where orders.customer_id = customers.customer_id);'),
(N'learn.microsoft.com', N'select * from products where category_id in(select category_id from categories where category_name = ''beverages'');'),
(N'geeksforgeeks.org', N'select customer_name,(select count(*) from orders where orders.customer_id = customers.customer_id) from customers;'),
(N'sqlshack.com', N'with recent_orders as(select * from orders where order_date > ''2023-01-01'') select * from recent_orders;'),
(N'w3schools.com/sql', N'with sales as(select customer_id, sum(amount) as total from payments group by customer_id) select * from sales where total > 1000;'),
(N'stackoverflow.com', N'with ranked as(select employee_id, salary, row_number() over(order by salary desc) as rn from employees) select * from ranked where rn <= 10;'),
(N'sqlservertutorial.net', N'select employee_id, salary, rank() over(order by salary desc) from employees;'),
(N'learn.microsoft.com', N'select employee_id, department_id, dense_rank() over(partition by department_id order by salary desc) from employees;'),
(N'geeksforgeeks.org', N'select order_id, amount, sum(amount) over(order by order_date) from payments;'),
(N'sqlshack.com', N'select customer_id, amount, lag(amount) over(partition by customer_id order by order_date) from payments;'),
(N'w3schools.com/sql', N'select customer_id, amount, lead(amount) over(partition by customer_id order by order_date) from payments;'),
(N'stackoverflow.com', N'select department_id, avg(salary) over(partition by department_id) from employees;'),
(N'sqlservertutorial.net', N'select product_name, price, ntile(4) over(order by price) from products;'),
(N'learn.microsoft.com', N'select order_id, row_number() over(partition by customer_id order by order_date desc) as rn from orders;'),
(N'geeksforgeeks.org', N'insert into customers(customer_name, city, country) values(''acme'', ''london'', ''uk'');'),
(N'sqlshack.com', N'insert into products(product_name, price, category_id) values(''widget'', 9.99, 1);'),
(N'w3schools.com/sql', N'insert into orders(customer_id, order_date) values(1, ''2023-05-01'');'),
(N'stackoverflow.com', N'insert into employees(first_name, last_name, salary) values(''jane'', ''smith'', 60000);'),
(N'sqlservertutorial.net', N'update products set price = 19.99 where product_id = 1;'),
(N'learn.microsoft.com', N'update employees set salary = salary * 1.1 where department_id = 3;'),
(N'geeksforgeeks.org', N'update orders set status = ''shipped'' where order_id = 100;'),
(N'sqlshack.com', N'update customers set city = ''paris'' where customer_id = 7;'),
(N'w3schools.com/sql', N'delete from orders where order_date < ''2020-01-01'';'),
(N'stackoverflow.com', N'delete from customers where customer_id = 42;'),
(N'sqlservertutorial.net', N'delete from products where price is null;'),
(N'learn.microsoft.com', N'create table employees(employee_id int primary key, first_name varchar(50), salary decimal(10, 2));'),
(N'geeksforgeeks.org', N'create table orders(order_id int primary key, customer_id int, order_date date);'),
(N'sqlshack.com', N'create index ix_orders_customer_id on orders(customer_id);'),
(N'w3schools.com/sql', N'create view active_customers as select * from customers where status = ''active'';'),
(N'stackoverflow.com', N'alter table employees add hire_date date;'),
(N'sqlservertutorial.net', N'drop table temp_data;'),
(N'learn.microsoft.com', N'truncate table staging_orders;'),
(N'geeksforgeeks.org', N'select case when price > 100 then ''high'' else ''low'' end from products;'),
(N'sqlshack.com', N'select customer_name, case when country = ''uk'' then ''domestic'' else ''export'' end from customers;'),
(N'w3schools.com/sql', N'select coalesce(phone, email, ''unknown'') from customers;'),
(N'stackoverflow.com', N'select isnull(discount, 0) from order_items;'),
(N'sqlservertutorial.net', N'select nullif(quantity, 0) from order_items;'),
(N'learn.microsoft.com', N'select cast(price as int) from products;'),
(N'geeksforgeeks.org', N'select convert(varchar(10), order_date, 120) from orders;'),
(N'sqlshack.com', N'select upper(customer_name) from customers;'),
(N'w3schools.com/sql', N'select lower(email) from customers;'),
(N'stackoverflow.com', N'select len(customer_name) from customers;'),
(N'sqlservertutorial.net', N'select substring(customer_name, 1, 3) from customers;'),
(N'learn.microsoft.com', N'select replace(phone, ''-'', '''') from customers;'),
(N'geeksforgeeks.org', N'select trim(customer_name) from customers;'),
(N'sqlshack.com', N'select concat(first_name, '' '', last_name) from employees;'),
(N'w3schools.com/sql', N'select first_name + '' '' + last_name from employees;'),
(N'stackoverflow.com', N'select year(order_date), count(*) from orders group by year(order_date);'),
(N'sqlservertutorial.net', N'select month(order_date), sum(amount) from payments group by month(order_date);'),
(N'learn.microsoft.com', N'select datediff(day, order_date, ship_date) from orders;'),
(N'geeksforgeeks.org', N'select dateadd(day, 30, order_date) from orders;'),
(N'sqlshack.com', N'select getdate();'),
(N'w3schools.com/sql', N'select round(price, 2) from products;'),
(N'stackoverflow.com', N'select abs(balance) from accounts;'),
(N'sqlservertutorial.net', N'select customer_id from orders union select customer_id from returns;'),
(N'learn.microsoft.com', N'select city from customers union all select city from suppliers;'),
(N'geeksforgeeks.org', N'select customer_id from orders intersect select customer_id from payments;'),
(N'sqlshack.com', N'select customer_id from customers except select customer_id from orders;'),
(N'w3schools.com/sql', N'select * from employees where department_id = 1 order by salary desc offset 0 rows fetch next 10 rows only;'),
(N'stackoverflow.com', N'select * from products order by product_id offset 20 rows fetch next 20 rows only;'),
(N'sqlservertutorial.net', N'select o.customer_id, sum(i.quantity * p.price) from order_items i inner join orders o on i.order_id = o.order_id inner join products p on i.product_id = p.product_id group by o.customer_id;'),
(N'learn.microsoft.com', N'select c.country, count(distinct c.customer_id), sum(p.amount) from customers c inner join payments p on c.customer_id = p.customer_id group by c.country having sum(p.amount) > 10000 order by sum(p.amount) desc;'),
(N'geeksforgeeks.org', N'select d.department_name, avg(e.salary) from departments d left join employees e on d.department_id = e.department_id group by d.department_name order by avg(e.salary) desc;'),
(N'sqlshack.com', N'select top 10 p.product_name, sum(i.quantity) from products p inner join order_items i on p.product_id = i.product_id group by p.product_name order by sum(i.quantity) desc;'),
(N'w3schools.com/sql', N'select customer_id, min(order_date), max(order_date), count(*) from orders group by customer_id;'),
(N'stackoverflow.com', N'select category_id, count(*), avg(price), max(price) from products group by category_id order by count(*) desc;');
GO

/*------------------------------------------------------------------
  4. Normalisation. Lower-cases and separates punctuation so that
     "COUNT(*)" becomes "count ( * )" -- three tokens, not one blob.

     Multi-character operators (<>, >=, <=, !=) are parked behind
     placeholders first, otherwise the single-character pass would
     shred them into two tokens.
------------------------------------------------------------------*/
CREATE OR ALTER FUNCTION dbo.fn_Normalise (@txt nvarchar(max))
RETURNS nvarchar(max)
AS
BEGIN
    DECLARE @s nvarchar(max) = LOWER(@txt);

    -- JSON-hostile characters are not expected in SQL text, but be safe.
    SET @s = REPLACE(REPLACE(REPLACE(@s, NCHAR(92), N' '), NCHAR(34), N' '), NCHAR(9), N' ');
    SET @s = REPLACE(REPLACE(@s, NCHAR(13), N' '), NCHAR(10), N' ');

    -- Park two-character operators.
    SET @s = REPLACE(REPLACE(REPLACE(REPLACE(@s,
                N'<>', N'~1~'), N'>=', N'~2~'), N'<=', N'~3~'), N'!=', N'~4~');

    -- Pad single-character punctuation with spaces.
    SET @s = REPLACE(@s, N'(',  N' ( ');
    SET @s = REPLACE(@s, N')',  N' ) ');
    SET @s = REPLACE(@s, N',',  N' , ');
    SET @s = REPLACE(@s, N';',  N' ; ');
    SET @s = REPLACE(@s, N'*',  N' * ');
    SET @s = REPLACE(@s, N'=',  N' = ');
    SET @s = REPLACE(@s, N'<',  N' < ');
    SET @s = REPLACE(@s, N'>',  N' > ');
    SET @s = REPLACE(@s, N'+',  N' + ');
    SET @s = REPLACE(@s, N'.',  N' . ');
    SET @s = REPLACE(@s, N'[',  N' [ ');
    SET @s = REPLACE(@s, N']',  N' ] ');
    SET @s = REPLACE(@s, NCHAR(39), N' ' + NCHAR(39) + N' ');   -- single quote

    -- Restore the two-character operators, now safely spaced.
    SET @s = REPLACE(REPLACE(REPLACE(REPLACE(@s,
                N'~1~', N' <> '), N'~2~', N' >= '), N'~3~', N' <= '), N'~4~', N' != ');

    -- Collapse runs of whitespace.
    WHILE CHARINDEX(N'  ', @s) > 0
        SET @s = REPLACE(@s, N'  ', N' ');

    RETURN LTRIM(RTRIM(@s));
END;
GO

/*------------------------------------------------------------------
  5. Tokeniser. Splits on spaces and returns 1-based positions.
     OPENJSON is used rather than STRING_SPLIT because it yields an
     ordinal on every supported version, not just 2022+.
------------------------------------------------------------------*/
CREATE OR ALTER FUNCTION dbo.fn_Tokenize (@txt nvarchar(max))
RETURNS TABLE
AS
RETURN
    SELECT  pos   = ROW_NUMBER() OVER (ORDER BY CAST(j.[key] AS int)),
            token = j.value
    FROM    OPENJSON(N'["' + REPLACE(dbo.fn_Normalise(@txt), N' ', N'","') + N'"]') AS j
    WHERE   j.value <> N'';
GO

/*------------------------------------------------------------------
  6. Vocabulary, token stream, and training rows.

     TrainSeq  : one row per prediction target
     TrainCtx  : C rows per target, one per context slot
                 (kept as rows rather than columns so that changing C
                  needs no schema change)
------------------------------------------------------------------*/
DROP TABLE IF EXISTS dbo.TrainCtx;
DROP TABLE IF EXISTS dbo.TrainSeq;
DROP TABLE IF EXISTS dbo.TokenStream;
DROP TABLE IF EXISTS dbo.Vocab;

CREATE TABLE dbo.Vocab
(
    token_id int           NOT NULL CONSTRAINT PK_Vocab PRIMARY KEY,
    token    nvarchar(100) NOT NULL CONSTRAINT UQ_Vocab UNIQUE,
    freq     int           NOT NULL
);

CREATE TABLE dbo.TokenStream
(
    doc_id   int NOT NULL,
    pos      int NOT NULL,
    token_id int NOT NULL,
    CONSTRAINT PK_TokenStream PRIMARY KEY (doc_id, pos)
);

CREATE TABLE dbo.TrainSeq
(
    seq_id   int NOT NULL CONSTRAINT PK_TrainSeq PRIMARY KEY,
    doc_id   int NOT NULL,
    pos      int NOT NULL,
    target   int NOT NULL,
    batch_id int NOT NULL CONSTRAINT DF_TrainSeq_Batch DEFAULT (0)
);
CREATE INDEX IX_TrainSeq_Batch ON dbo.TrainSeq (batch_id) INCLUDE (target);

CREATE TABLE dbo.TrainCtx
(
    seq_id   int      NOT NULL,
    slot     smallint NOT NULL,
    token_id int      NOT NULL,
    CONSTRAINT PK_TrainCtx PRIMARY KEY (seq_id, slot)
);
GO

CREATE OR ALTER PROCEDURE dbo.usp_BuildDataset
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @C int = (SELECT C FROM dbo.ModelConfig);

    DELETE FROM dbo.TrainCtx;
    DELETE FROM dbo.TrainSeq;
    DELETE FROM dbo.TokenStream;
    DELETE FROM dbo.Vocab;

    /* Every document is wrapped as  <s> <s> <s> ... tokens ... </s>
       so the model learns how a query starts and how it ends. */
    DROP TABLE IF EXISTS #Raw;
    CREATE TABLE #Raw (doc_id int NOT NULL, pos int NOT NULL, token nvarchar(100) NOT NULL,
                       PRIMARY KEY (doc_id, pos));

    INSERT INTO #Raw (doc_id, pos, token)
    SELECT c.doc_id, n.n + 1, N'<s>'
    FROM   dbo.Corpus c
    JOIN   dbo.Nums   n ON n.n < @C;

    INSERT INTO #Raw (doc_id, pos, token)
    SELECT c.doc_id, t.pos + @C, t.token
    FROM   dbo.Corpus c
    CROSS APPLY dbo.fn_Tokenize(c.query_text) AS t;

    /* Compute the end positions into their own table first. Inserting into
       #Raw while aggregating over #Raw in the same statement is the kind of
       thing that works until the day the optimiser picks a different plan. */
    DROP TABLE IF EXISTS #Ends;
    CREATE TABLE #Ends (doc_id int NOT NULL PRIMARY KEY, end_pos int NOT NULL);

    INSERT INTO #Ends (doc_id, end_pos)
    SELECT doc_id, MAX(pos) + 1 FROM #Raw GROUP BY doc_id;

    INSERT INTO #Raw (doc_id, pos, token)
    SELECT doc_id, end_pos, N'</s>' FROM #Ends;

    /* Vocabulary. Reserved tokens are given the low ids on purpose so
       they are easy to spot when eyeballing the weight tables. */
    INSERT INTO dbo.Vocab (token_id, token, freq)
    SELECT 1, N'<unk>', 0
    UNION ALL SELECT 2, N'<s>',  (SELECT COUNT(*) FROM #Raw WHERE token = N'<s>')
    UNION ALL SELECT 3, N'</s>', (SELECT COUNT(*) FROM #Raw WHERE token = N'</s>');

    INSERT INTO dbo.Vocab (token_id, token, freq)
    SELECT  3 + ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC, r.token),
            r.token,
            COUNT(*)
    FROM    #Raw AS r
    WHERE   r.token NOT IN (N'<s>', N'</s>', N'<unk>')
    GROUP BY r.token;

    UPDATE dbo.ModelConfig SET V = (SELECT COUNT(*) FROM dbo.Vocab);

    INSERT INTO dbo.TokenStream (doc_id, pos, token_id)
    SELECT r.doc_id, r.pos, v.token_id
    FROM   #Raw AS r
    JOIN   dbo.Vocab AS v ON v.token = r.token;

    /* One training row per position that has a full C-token history. */
    INSERT INTO dbo.TrainSeq (seq_id, doc_id, pos, target)
    SELECT  ROW_NUMBER() OVER (ORDER BY s.doc_id, s.pos),
            s.doc_id, s.pos, s.token_id
    FROM    dbo.TokenStream AS s
    WHERE   s.pos > @C;

    INSERT INTO dbo.TrainCtx (seq_id, slot, token_id)
    SELECT  t.seq_id, n.n, h.token_id
    FROM    dbo.TrainSeq AS t
    JOIN    dbo.Nums     AS n ON n.n < @C
    JOIN    dbo.TokenStream AS h
              ON  h.doc_id = t.doc_id
              AND h.pos    = t.pos - @C + n.n;

    SELECT  documents    = (SELECT COUNT(*) FROM dbo.Corpus),
            vocab_size   = (SELECT COUNT(*) FROM dbo.Vocab),
            tokens       = (SELECT COUNT(*) FROM dbo.TokenStream),
            training_rows= (SELECT COUNT(*) FROM dbo.TrainSeq);
END;
GO

EXEC dbo.usp_BuildDataset;
GO

/* Sanity check: see how a query gets shredded. */
SELECT * FROM dbo.fn_Tokenize(N'SELECT c.Name, COUNT(*) FROM Cust c WHERE c.Age>=18 AND s<>''x'';');
GO
