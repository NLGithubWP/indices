CREATE TABLE census_int_train_left (
    id INT PRIMARY KEY,
    label INT,
    col1 INT, col2 INT, col3 INT, col4 INT, col5 INT,
    col6 INT, col7 INT, col8 INT, col9 INT, col10 INT,
    col11 INT, col12 INT, col13 INT, col14 INT, col15 INT,
    col16 INT, col17 INT, col18 INT, col19 INT, col20 INT
);

CREATE TABLE census_int_train_right (
    right_id SERIAL PRIMARY KEY,  -- Auto-generated primary key
    id INT REFERENCES census_int_train_left(id),  -- Foreign key reference to left table's ID
    col21 INT, col22 INT, col23 INT, col24 INT, col25 INT,
    col26 INT, col27 INT, col28 INT, col29 INT, col30 INT,
    col31 INT, col32 INT, col33 INT, col34 INT, col35 INT,
    col36 INT, col37 INT, col38 INT, col39 INT, col40 INT,
    col41 INT
);


INSERT INTO census_int_train_left (id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20)
SELECT id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20
FROM census_int_train;


INSERT INTO census_int_train_right (id, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, col39, col40, col41)
SELECT id, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, col39, col40, col41
FROM census_int_train;


SELECT
    l.id,
    l.label,
    l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10,
    l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20,
    r.col21, r.col22, r.col23, r.col24, r.col25, r.col26, r.col27, r.col28, r.col29, r.col30,
    r.col31, r.col32, r.col33, r.col34, r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41
FROM
    census_int_train_left l
JOIN
    census_int_train_right r ON l.id = r.id limit 10;

-- Split credit_int_train into Two Tables
CREATE TABLE credit_int_train_left (
    id INT PRIMARY KEY,
    label INT,
    col1 INT, col2 INT, col3 INT, col4 INT, col5 INT, col6 INT, col7 INT, col8 INT, col9 INT, col10 INT,
    col11 INT, col12 INT
);

CREATE TABLE credit_int_train_right (
    right_id SERIAL PRIMARY KEY,
    id INT REFERENCES credit_int_train_left(id),
    col13 INT, col14 INT, col15 INT, col16 INT, col17 INT, col18 INT, col19 INT, col20 INT, col21 INT, col22 INT, col23 INT
);

INSERT INTO credit_int_train_left (id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12)
SELECT id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12
FROM credit_int_train;

INSERT INTO credit_int_train_right (id, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23)
SELECT id, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23
FROM credit_int_train;

SELECT
    l.id,
    l.label,
    l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12,
    r.col13, r.col14, r.col15, r.col16, r.col17, r.col18, r.col19, r.col20, r.col21, r.col22, r.col23
FROM
    credit_int_train_left l
JOIN
    credit_int_train_right r ON l.id = r.id limit 10;

-- Split diabetes_int_train into Two Tables
CREATE TABLE diabetes_int_train_left (
    id INT PRIMARY KEY,
    label INT,
    col1 INT, col2 INT, col3 INT, col4 INT, col5 INT, col6 INT, col7 INT, col8 INT, col9 INT, col10 INT,
    col11 INT, col12 INT, col13 INT, col14 INT, col15 INT, col16 INT, col17 INT, col18 INT, col19 INT, col20 INT,
    col21 INT, col22 INT, col23 INT, col24 INT
);

CREATE TABLE diabetes_int_train_right (
    right_id SERIAL PRIMARY KEY,
    id INT REFERENCES diabetes_int_train_left(id),
    col25 INT, col26 INT, col27 INT, col28 INT, col29 INT, col30 INT, col31 INT, col32 INT, col33 INT, col34 INT,
    col35 INT, col36 INT, col37 INT, col38 INT, col39 INT, col40 INT, col41 INT, col42 INT, col43 INT, col44 INT,
    col45 INT, col46 INT, col47 INT, col48 INT
);

INSERT INTO diabetes_int_train_left (id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24)
SELECT id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24
FROM diabetes_int_train;

INSERT INTO diabetes_int_train_right (id, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, col39, col40, col41, col42, col43, col44, col45, col46, col47, col48)
SELECT id, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, col39, col40, col41, col42, col43, col44, col45, col46, col47, col48
FROM diabetes_int_train;

SELECT
    l.id,
    l.label,
    l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20, l.col21, l.col22, l.col23, l.col24,
    r.col25, r.col26, r.col27, r.col28, r.col29, r.col30, r.col31, r.col32, r.col33, r.col34, r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41, r.col42, r.col43, r.col44, r.col45, r.col46, r.col47, r.col48
FROM
    diabetes_int_train_left l
JOIN
    diabetes_int_train_right r ON l.id = r.id limit 10;

-- Split hcdr_int_train into Two Tables
CREATE TABLE hcdr_int_train_left (
    id INT PRIMARY KEY,
    label INT,
    col1 INT, col2 INT, col3 INT, col4 INT, col5 INT, col6 INT, col7 INT, col8 INT, col9 INT, col10 INT,
    col11 INT, col12 INT, col13 INT, col14 INT, col15 INT, col16 INT, col17 INT, col18 INT, col19 INT, col20 INT,
    col21 INT, col22 INT, col23 INT, col24 INT, col25 INT, col26 INT, col27 INT, col28 INT, col29 INT, col30 INT,
    col31 INT, col32 INT, col33 INT, col34 INT
);

CREATE TABLE hcdr_int_train_right (
    right_id SERIAL PRIMARY KEY,
    id INT REFERENCES hcdr_int_train_left(id),
    col35 INT, col36 INT, col37 INT, col38 INT, col39 INT, col40 INT, col41 INT, col42 INT, col43 INT, col44 INT,
    col45 INT, col46 INT, col47 INT, col48 INT, col49 INT, col50 INT, col51 INT, col52 INT, col53 INT, col54 INT,
    col55 INT, col56 INT, col57 INT, col58 INT, col59 INT, col60 INT, col61 INT, col62 INT, col63 INT, col64 INT,
    col65 INT, col66 INT, col67 INT, col68 INT, col69 INT
);

INSERT INTO hcdr_int_train_left (id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34)
SELECT id, label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34
FROM hcdr_int_train;

INSERT INTO hcdr_int_train_right (id, col35, col36, col37, col38, col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49, col50, col51, col52, col53, col54, col55, col56, col57, col58, col59, col60, col61, col62, col63, col64, col65, col66, col67, col68, col69)
SELECT id, col35, col36, col37, col38, col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49, col50, col51, col52, col53, col54, col55, col56, col57, col58, col59, col60, col61, col62, col63, col64, col65, col66, col67, col68, col69
FROM hcdr_int_train;

SELECT
    l.id,
    l.label,
    l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20, l.col21, l.col22, l.col23, l.col24, l.col25, l.col26, l.col27, l.col28, l.col29, l.col30, l.col31, l.col32, l.col33, l.col34,
    r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41, r.col42, r.col43, r.col44, r.col45, r.col46, r.col47, r.col48, r.col49, r.col50, r.col51, r.col52, r.col53, r.col54, r.col55, r.col56, r.col57, r.col58, r.col59, r.col60, r.col61, r.col62, r.col63, r.col64, r.col65, r.col66, r.col67, r.col68, r.col69
FROM
    hcdr_int_train_left l
JOIN
    hcdr_int_train_right r ON l.id = r.id limit 10;

