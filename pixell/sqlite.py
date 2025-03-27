"""Module for making it a bit more convenient to work with sqlite files.
Similar to using a Connection directly, but also provides easy access
to a list of which tables and columns are available, which I always find
cumbersome to get at otherwise"""
import sqlite3, pprint

class SQL:
	def __init__(self, fname):
		self.fname= fname
		self.conn = sqlite3.connect(fname)
	def execute(self, command, args=[]):
		return self.conn.execute(command, args)
	def close(self):
		self.conn.close()
	def tables(self): return tables(self)
	def columns(self, tname): return columns(self, tname)
	def show(self, table, limit=10): return show(self, table, limit=limit)
	def __repr__(self): return info(self, "SQL")
	def __enter__(self): return self
	def __exit__(self, *args, **kwargs):
		self.close()

def tables(conn):
	return [entry[0] for entry in conn.execute("SELECT name from sqlite_master WHERE type='table';")]
def columns(conn, tname):
	return [col[1] for col in conn.execute("PRAGMA table_info('%s');" % tname)]
def info(conn, name="Connection"):
	tnames   = tables(conn)
	coldescs = [str(columns(conn, tname)) for tname in tnames]
	tabdescs = ", ".join(["%s:%s" % (tname, coldesc) for tname, coldesc in zip(tnames, coldescs)])
	return "%s(tables=[%s])" % (name, tabdescs)
def show(conn, table, limit=10):
	query = table # either table name or a full query
	toks  = query.split()
	if len(toks) == 1:
		# Someone just gave us a table name. Rewrite to a full format
		query = "select * from " + toks[0]
	limit_included = "limit" in toks or "LIMIT" in toks
	if not limit_included:
		# Add our limit. We add one more so we can use ... to indicate
		# there's more
		query += " limit %d" % (limit+1)
	result = list(conn.execute(query))
	if limit_included: limit = len(result)
	print(format_result(result, limit=limit))

def format_result(result, limit=None):
	if limit is None: limit = len(result)
	if len(result) == 0: return "<empty>"
	trunc = len(result) > limit
	if trunc: result = result[:limit]
	# Find the max-width of each column
	widths = [0 for field in result[0]]
	for row in result:
		for fi, field in enumerate(row):
			widths[fi] = max(widths[fi], len(str(field)))
	fmt   = " ".join(["%%%ds" % w for w in widths])
	lines = [fmt % row for row in result]
	if trunc: lines.append("...")
	return "\n".join(lines)

# Make sqlite.open an alias for sqlite.SQL
open = SQL
