from werkzeug.security import generate_password_hash
import psycopg2

# Database connection details
DB_HOST = "localhost"
DB_NAME = "face_recognition_attendance"
DB_USER = "postgres"
DB_PASS = "kai23"

def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    return conn

def register(name, username, password):
    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO admin (name, username, hashed_password) VALUES (%s, %s, %s)",
        (name, username, hashed_password)
    )
    conn.commit()

    cursor.close()
    conn.close()

register('Aleena', 'admin', '1234')