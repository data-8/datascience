from datascience import Table
from datascience.util import make_array

"""
print(Table.read_table("https://raw.githubusercontent.com/data-8/textbook/gh-pages/data/deflategate.csv"))

t = Table.read_table("https://raw.githubusercontent.com/data-8/textbook/gh-pages/data/deflategate.csv")

print(t.show())

def table():
    """Setup Scrabble table"""
    return Table().with_columns([
        'letter', ['a', 'b', 'c', 'z'],
        'count', [9, 3, 3, 1],
        'points', [1, 2, 2, 10],
        ])
    
def stack(self, key, labels=None):
    """Takes k original columns and returns two columns, with col. 1 of
    all column names and col. 2 of all associated data.
    """
    rows, labels = [], labels or self.labels
    print('rows: ', rows, ' labels: ', labels)
    for row in self.rows:
        [rows.append((getattr(row, key), k, v)) for k, v in row.asdict().items()
         if k != key and k in labels]
    return type(self)([key, 'column', 'value']).with_rows(rows)

t = table()
a = t.stack(key='letter')

print(a)
"""
players = Table().with_columns('player_id', \
               make_array(110234, 110235), 'wOBA', make_array(.354, .236))
players.stack(key="wOBA", labels="abc")

jobs = Table().with_columns( \
        'job',  make_array('a', 'b', 'c', 'd'),
        'wage', make_array(10, 20, 15, 8))
jobs.stack(key='job')

table = Table().with_columns( \
        'days',  make_array(0, 1, 2, 3, 4, 5), \
        'price', make_array(90.5, 90.00, 83.00, 95.50, 82.00, 82.00), \
        'projection', make_array(90.75, 82.00, 82.50, 82.50, 83.00, 82.50))
table.stack(key='price', labels="days")