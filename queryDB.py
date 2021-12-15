import sqlite3
from sqlite3 import Error
import os

myfiles = ['1013.mp3', '2597.mp3', '3205.mp3', '22818.mp3', '24325.mp3', '184640.mp3', '184751.mp3']

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def getSegments(conn, fileid):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM segmentlist WHERE fileid = ?", (fileid,))

    rows = cur.fetchall()
    print('segments for %s' % fileid)
    for row in rows:
        print('%f %f %f %s'% (row[1], row[2], row[3], row[4]))
    print()

def getFileId(conn, filename):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM filelist WHERE filename=?", (filename,))

    row = cur.fetchall()
    for r in row:
        print(r)
    fileid = row[0][0]

    print('file id for ', filename, ' is ', fileid)

    return fileid

def getfilesmatch(conn):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    for item in myfiles:
        cur.execute("SELECT * FROM filelist WHERE filename=?", (item,))
        row = cur.fetchall()
        if(len(row)>=1):
            print('found file: ', row[0])

    return 

# dir = 'NewAudioFiles/NewAudioFiles'
dir = 'TestSounds'

def main():
    database = r'aume_Freesound_Currated.sqlite'

    # create a database connection
    conn = create_connection(database)

    with conn:
        ids = {}
        # for file in myfiles:
        #     fileid = getFileId(conn, file)
        #     ids[file] = fileid
        
        for file in os.listdir(dir):
            try:
                print(file)
                fileid = getFileId(conn, file)
                ids[file] = fileid
            except:
                print('no file %s in db'% file)

        print('get segemnts...')
        for file, id in ids.items():
            print(file)
            getSegments(conn, id)

        # getfilesmatch(conn)


if __name__ == '__main__':
    main()