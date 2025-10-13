from langchain_core.tools import tool
import mysql.connector

@tool
def query_notices(query: str) -> str:
    """Execute SQL query on the web-notice database and return results."""
    conn = mysql.connector.connect(
        user='root',
        password='1234',
        host='localhost',
        database='web-notice'
    )
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return str(rows)


print(query_notices.invoke("select * from notice_notice"))