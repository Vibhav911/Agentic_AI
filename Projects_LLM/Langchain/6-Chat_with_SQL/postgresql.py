# This progrma is used to create a table in PostgreSQL and insert records into it.

import postgres as pg
import psycopg2 as pc


# Connect to sqlite database
connection = pc.connect(
    dbname="student",
    user="postgres",
    password="localhost",
    host="localhost",
    port="5432"
)

# Create a cursor object to insert records and create tables
cursor = connection.cursor()

# Create a table
table_info = """
create table STUDENT (
    NAME VARCHAR(25),
    CLASS VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS INT 
)
"""

# Execute the create table statement
cursor.execute(table_info)

# Insert records into the table
cursor.execute("Insert Into STUDENT values ('Krish', 'Data Science', 'A', 90)")
cursor.execute("Insert Into STUDENT values ('John', 'Data Science', 'B', 100)")
cursor.execute("Insert Into STUDENT values ('Mukesh', 'Data Science', 'A', 86)")
cursor.execute("Insert Into STUDENT values ('Jacob', 'DEVOPS', 'A', 50)")
cursor.execute("Insert Into STUDENT values ('Dipesh', 'DEVOPS', 'A', 35)")

# Display all records from the table
print("Inserted recorsds are:")
data = cursor.execute("Select * from STUDENT")
#for row in data:
   # print(row)

# Commit the changes to the database
connection.commit()
connection.close()
