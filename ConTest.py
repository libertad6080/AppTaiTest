import sqlite3

con = sqlite3.connect(r"file:Z:/Opo/AppTest/Gemini/tests.db?mode=ro", uri=True)

print(con.execute("SELECT name FROM sqlite_master").fetchall())

for fila in con.execute("SELECT numero, enunciado FROM preguntas"):
    print(fila)

con.close()

