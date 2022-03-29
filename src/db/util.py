import psycopg2
import dotenv
import os
import pickle

env_path = os.path.join(os.path.dirname(__file__), '../../.env')
dotenv.read_dotenv(env_path)


def connect():
    """ Connect to the PostgreSQL database server """
    db_name = os.getenv("DATABASE_NAME")
    db_user = os.getenv("DATABASE_USER")
    db_password = os.getenv("DATABASE_PASSWORD")
    return psycopg2.connect(database=db_name, user=db_user, password=db_password)


def exec_commit(sql, args={}):
    conn = connect()
    cur = conn.cursor()
    result = cur.execute(sql, args)
    conn.commit()
    conn.close()
    return result


def bulk_insert(table, tuple_lst):
    conn = connect()
    cur = conn.cursor()
    args_str = ','.join(cur.mogrify(cur.mogrify('('+','.join(['%s' for s in range(len(tuple_lst[0]))])+')', x).decode('utf-8'), x).decode('utf-8') for x in tuple_lst)
    cur.execute(f"INSERT INTO {table} VALUES {args_str}")
    conn.commit()
    conn.close()


def exec_sql_file(path):
    full_path = os.path.join(os.path.dirname(__file__), path)
    conn = connect()
    cur = conn.cursor()
    with open(full_path, 'r') as file:
        cur.execute(file.read())
    conn.commit()
    conn.close()


def insert_user(user_id, arr):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO genre_distributions(user_id, user_data)
        VALUES (%s, %s)
        """,
        (user_id, pickle.dumps(arr))
    )
    conn.commit()
    conn.close()


def insert_track_analysis_array(track_id, arr):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO track_audio_analysis(track_id, analysis_data)
        VALUES (%s, %s)
        """,
        (track_id, pickle.dumps(arr))
    )
    conn.commit()
    conn.close()


def query_user(user_id):
    conn = connect()
    cur = conn.cursor()
    cur.itersize = 10000  # how many records to buffer on a client
    cur.execute(
        """
        SELECT user_data
        FROM genre_distributions
        WHERE user_id=%s
        """,
        (user_id,)
    )
    res = cur.fetchone()
    if res is not None:
        return pickle.loads(res[0])
    return None


def exec_get_one(sql, args={}):
    conn = connect()
    cur = conn.cursor()
    cur.execute(sql, args)
    one = cur.fetchone()
    conn.close()
    return one


def exec_get_all(sql, args={}):
    conn = connect()
    cur = conn.cursor()
    cur.execute(sql, args)
    # https://www.psycopg.org/docs/cursor.html#cursor.fetchall
    list_of_tuples = cur.fetchall()
    conn.close()
    return list_of_tuples


exec_sql_file('init.sql')
