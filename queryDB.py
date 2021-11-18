import sqlite3
from sqlite3 import Error

myfiles = ['1013.mp3', '1014.mp3', '1145.mp3', '1344.mp3', '1366.mp3', '1379.mp3', '1464.mp3', '1481.mp3', '1490.mp3', '1585.mp3', '1665.mp3', '1699.mp3', '1700.mp3', '1706.mp3', '1715.mp3', '1734.mp3', '1735.mp3', '1743.mp3', '1773.mp3', '1774.mp3', '1778.mp3', '1841.mp3', '184640.mp3', '184751.mp3', '184796.mp3', '184818.mp3', '1861.mp3', 
'1865.mp3', '1867.mp3', '1879.mp3', '194357.mp3', '1945.mp3', '1950.mp3', '1966.mp3', '1988.mp3', '2000.mp3', '2057.mp3', '2104.mp3', '2109.mp3', '2126.mp3', '2132.mp3', '2151.mp3', '2185.mp3', '2189.mp3', '2193.mp3', '2204.mp3', '2213.mp3', '2214.mp3', '2215.mp3', '2216.mp3', '2253.mp3', '22818.mp3', '22828.mp3', '22862.mp3', '22863.mp3', '2347.mp3', '2356.mp3', '2394.mp3', '2401.mp3', '24325.mp3', '24326.mp3', '24327.mp3', '24329.mp3', '2451.mp3', '2465.mp3', '2504.mp3', '2526.mp3', '2593.mp3', '2597.mp3', '2736.mp3', '2740.mp3', '2792.mp3', '2817.mp3', '2826.mp3', '2853.mp3', '3040.mp3', '3183.mp3', '3205.mp3', '3217.mp3', '3370.mp3', '3379.mp3', '3384.mp3', '3436.mp3', '3450.mp3', '3458.mp3', '3470.mp3', '3515.mp3', '3615.mp3', '3629.mp3', '3631.mp3', '3632.mp3', '3715.mp3', '3723.mp3', '3801.mp3', '3802.mp3', '3811.mp3', '3834.mp3', '837.mp3', '933.mp3', '992.mp3']

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

    for row in rows:
        print('%f %f %f %s'% (row[1], row[2], row[3], row[4]))

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

def main():
    database = r'aume_Freesound_Currated.sqlite'

    # create a database connection
    conn = create_connection(database)

    with conn:
        fileid = getFileId(conn, '3205.mp3')

        print('get segemnts...')
        getSegments(conn, fileid)

        # getfilesmatch(conn)


if __name__ == '__main__':
    main()