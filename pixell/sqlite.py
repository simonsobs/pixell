"""Module for making it a bit more convenient to work with sqlite files.
Similar to using a Connection directly, but also provides easy access
to a list of which tables and columns are available, which I always find
cumbersome to get at otherwise"""
import sqlite3, pprint, contextlib, tempfile, os

class SQL:
	def __init__(self, fname=":memory:"):
		# Is it a file name?
		if isinstance(fname, str):
			self.conn = sqlite3.connect(fname)
			self.fname= fname
			self.own  = True
		# Is it an SQL or similar?
		elif hasattr(fname, "conn"):
			self.conn = fname.conn
			self.fname= get_fname(self.conn)
			self.own  = False
		# Is it a plain sqlite3 connection?
		elif hasattr(fname, "execute"):
			self.conn  = fname
			self.fname = get_fname(self.conn)
			self.own   = False
		else: raise ValuError("SQL.__init__ needs either a file name, an SQL object or an sqlite3 connection")
	def execute(self, command, args=[]):
		return self.conn.execute(command, args)
	def executemany(self, command, args=[]):
		return self.conn.executemany(command, args)
	def derive(self, query, tname="result", aname="_src"):
		return derive(self.conn, query, tname=tname, aname=aname)
	def close(self):
		if self.own: self.conn.close()
	def backup(self, other): backup(self, other)
	def attach(self, other, name="other", mode="r"):
		"""Context manager. Temporarily attaches other to us.
		To do it permanently, just use a manual execute."""
		return attach(self, other, name=name, mode=mode)
	def tables(self): return tables(self)
	def columns(self, tname): return columns(self, tname)
	def show(self, table, limit=10): return show(self, table, limit=limit)
	def __repr__(self): return info(self, "SQL", extra=["fname='%s'" % self.fname, "own=%d" % self.own])
	def __enter__(self): return self
	def __exit__(self, *args, **kwargs):
		self.close()

def tables(conn):
	return [entry[0] for entry in conn.execute("SELECT name from sqlite_master WHERE type='table';")]
def columns(conn, tname):
	return [col[1] for col in conn.execute("PRAGMA table_info('%s');" % tname)]
def rows(conn, tname):
	return list(conn.execute("select count(*) from %s" % tname))[0][0]
def info(conn, name="Connection", extra=[]):
	tnames   = tables(conn)
	coldescs = ["%s*%d" % (str(columns(conn, tname)),rows(conn,tname)) for tname in tnames]
	tabdescs = ", ".join(["%s:%s" % (tname, coldesc) for tname, coldesc in zip(tnames, coldescs)])
	fields = ["tables=[%s]" % tabdescs]+extra
	return "%s(%s)" % (name, ", ".join(fields))
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

def get_fname(conn):
	row = next(conn.execute("pragma database_list"))
	return row[2]

def backup(source, target):
	"""Simple wrapper that supports both raw connections
	and SQL objects"""
	if hasattr(source, "conn"): source = source.conn
	if hasattr(target, "conn"): target = target.conn
	source.backup(target)

@contextlib.contextmanager
def attach(conn_base, conn_other, name="other", mode="r"):
	"""Context manager to temporarily attach one database to another.
	Useful when transfering data between two databases.

	Works around sqlite's lack of support for attaching memory-only
	databases to each other by dumping other into an anonymous temporary
	file, and then restoring it afterwards if mode contains "w".
	Yuck.
	"""
	fname = get_fname(conn_other)
	if fname:
		conn_base.execute("attach database '%s' as %s" % (fname, name))
		try:
			yield
		finally:
			conn_base.execute("detach database %s" % name)
	else:
		with tempfile.NamedTemporaryFile(suffix=".sqlite") as tfile:
			with SQL(tfile.name) as tmp_db: # this ensures it's auto-closed
				if "r" in mode: backup(conn_other, tmp_db)
				conn_base.execute("attach database '%s' as %s" % (tfile.name, name))
				try:
					yield
				finally:
					conn_base.execute("detach database %s" % name)
					if "w" in mode: backup(tmp_db, conn_other)

def derive(conn, query, tname="result", out_conn=None, aname="_src"):
	"""Return a new temporary database containing the result of
	running query on the provided connection."""
	if out_conn:
		# This may seem backwards, but doing it from the old db
		# has two advantages: 1. It works if the old one is in-memory,
		# and 2. one doesn't need to modify the query
		with attach(conn, out_conn, aname, mode="rw"):
			conn.execute("create table %s.%s as %s" % (aname, tname, query))
		return out_conn
	else:
		derived = SQL()
		try:
			return derive(conn, query, tname=tname, out_conn=derived)
		except:
			# Yes, this will only happen on error. The normal close will happen
			# as part of the user using a with with our result.
			# We do it this way to support both with and manual open/close
			if derived: derived.close()
			raise

# Make sqlite.open an alias for sqlite.SQL
open = SQL

# Full list of SQLite keywords
keywords = set([
	"abort", "action", "add", "after", "all", "alter", "always", "analyze", "and", "as",
	"asc", "attach", "autoincrement", "before", "begin", "between", "by", "cascade", "case",
	"cast", "check", "collate", "column", "commit", "conflict", "constraint", "create",
	"cross", "current", "current_date", "current_time", "current_timestamp", "database",
	"default", "deferrable", "deferred", "delete", "desc", "detach", "distinct", "do",
	"drop", "each", "else", "end", "escape", "except", "exclude", "exclusive", "exists",
	"explain", "fail", "filter", "first", "following", "for", "foreign", "from", "full",
	"generated", "glob", "group", "groups", "having", "if", "ignore", "immediate", "in",
	"index", "indexed", "initially", "inner", "insert", "instead", "intersect", "into",
	"is", "isnull", "join", "key", "last", "left", "like", "limit", "match", "materialized",
	"natural", "no", "not", "nothing", "notnull", "null", "nulls", "of", "offset", "on",
	"or", "order", "others", "outer", "over", "partition", "plan", "pragma", "preceding",
	"primary", "query", "raise", "range", "recursive", "references", "regexp", "reindex",
	"release", "rename", "replace", "restrict", "returning", "right", "rollback", "row",
	"rows", "savepoint", "select", "set", "table", "temp", "temporary", "then", "ties",
	"to", "transaction", "trigger", "unbounded", "union", "unique", "update", "using",
	"vacuum", "values", "view", "virtual", "when", "where", "window", "with", "without",
])
