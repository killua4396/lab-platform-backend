import mysql.connector
from config import DB_CONFIG
from fastapi import HTTPException

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"数据库连接失败: {err}")