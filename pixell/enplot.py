from __future__ import division, print_function
import numpy as np, argparse, time, sys, warnings, os, shlex, glob, PIL.Image, PIL.ImageDraw
from scipy import ndimage
from . import enmap, colorize, mpi, cgrid, utils, memory, bunch, wcsutils
# Optional dependency array_ops needed for contour drawing
try: from . import array_ops
except ImportError: pass

# Python 3 compatibility
try: basestring
except NameError: basestring = str

class Printer:
	def __init__(self, level=1, prefix=""):
		self.level  = level
		self.prefix = prefix
	def write(self, desc, level, exact=False, newline=True, prepend=""):
		if level == self.level or not exact and level <= self.level:
			prepend = "%6.2f " % (memory.max()/1024.**3) + prepend
			sys.stderr.write(prepend + self.prefix + desc + ("\n" if newline else ""))
	def push(self, desc):
		return Printer(self.level, self.prefix + desc)
	def time(self, desc, level, exact=False, newline=True):
		class PrintTimer:
			def __init__(self, printer): self.printer = printer
			def __enter__(self): self.time = time.time()
			def __exit__(self, type, value, traceback):
				self.printer.write(desc, level, exact=exact, newline=newline, prepend="%6.2f " % (time.time()-self.time))
		return PrintTimer(self)
noprint = Printer(0)

def plot(*arglist, **args):
	"""The main plotting function in this module. Plots the given maps/files,
	returning them as a list of plots, one for each separate image. This
	function has two equivalent interfaces:
	1. A command-line like interface, where plotting options are specified with
	strings like "-r 500:50 -m 0 -t 2".
	2. A python-like interface, where plotting options are specified with
	keyword arguments, like range="500:50", mask=0, ticks=2.
	These interfaces can be mixed and matched.
	
	Input maps are specified either as part of the option strings, as separate
	file names, or as enmaps passed to the function. Here are some examples:
	
	plot(mapobj):
	  Plots the given enmap object mapobj. If mapobj is a scalar map,
	  the a length-1 list containing a single plot will be returned.
	  If mapobj is e.g. a 3-component TQU map, then a length-3 list
	  of plots will be returned.
	plot((mapobj,"foo")):
	  If a tuple is given, the second element specifies the name tag to
	  use. This tag will be used to populate the plot.name attribute
	  for each output plot, which can be useful when plotting and writing
	  the maps.
	 plot("file.fits"):
	  Reads a map from file.fits, and plots it. This sets the tag to "file",
	  so that the result can easily be written to "file.png" (or "file_0.png" etc).
	 plot(["a.fits","b.fits",(mapobj,"c"),(mapobj,"d.fits")])
	  Reads a.fits and plots it to a.png, reads b.fits and plots it to b.png,
	  plots the first mapobj to c.png and the second one to d.png (yes, the
	  extension in the filename specified in the tuple is ignored. This is
	  because that filename actually supplies the *input* filename that the
	  output filename should be computed from).
	 plot("foo*.fits")
	  Reads and plots every file matching the glob foo*.fits.
	 plot("a.fits -r 500:50 -m 0 -d 4 -t 4")
	  Reads and plots the file a.fits to a.png, using a color range of +-500
	  for the first field in the map (typically the total intensity), and
	  +-50 for the remaining fields (typically Q and U). The maps are downsampled
	  by a factor of 4, and plotted with a grid spacing of 4.
	
	Here is a list of the individual options plot recognizes. The short and long
	ones are recognized when used in the argument string syntax, while the long
	ones (with - replaced by _) also work as keyword arguments.

	See plot_iterator for an iterator version of this function.
	"""
	return list(plot_iterator(*arglist, **args))

def pshow(*arglist, method="auto", **args):
	"""Convenience function to both build plots and show them.
	pshow(...) is equivalent to show(plot(...))."""
	show(plot(*arglist, **args), method=method)

# Compatibility function
def get_plots(*arglist, **args):
	"""This function is identical to enplot.plot"""
	return plot(*arglist, **args)

def plot_iterator(*arglist, **kwargs):
	"""Iterator that yields the plots for each input map/file. Each yield
	will be a plot object, with fields
	.type: The type of the plot. Can be "pil" or "mpl". Usually "pil".
	.img:  The plot image object, of the given .type.
	.name: Suggested file name
	These plots objects can be written to disk using enplot.write.
	See the plot function documentation for a description of the arguments"""
	# This allows us to pass in both command-line style plot specifications
	# as well as python-style lists of enmaps and keyword arguments. The maps
	# to be plotted are collected in maps, which is a list that can contain
	# either enmaps or filenames. Filenames will later be read in as enmaps
	imaps = []
	comm   = extract_arg(kwargs, "comm",   None)
	noglob = extract_arg(kwargs, "noglob", False)

	# Set up defaults
	args = parse_args([])
	# Then process all the args
	for arg in arglist:
		if isinstance(arg, basestring):
			parsed = parse_args(arg, noglob=noglob)
			imaps += parsed.ifiles
			args.update(parsed)
		else:
			imaps.append(arg)
	del args["ifiles"]
	# Override with kwargs
	args.update(kwargs)
	args = bunch.Bunch(**args)
	# Check for invalid kwargs
	check_args(args)
	if comm is None:
		comm = mpi.COMM_SELF
	# Set up verbose output
	printer = Printer(args.verbosity)
	cache = {}
	# Plot each map
	for fi in range(comm.rank,len(imaps),comm.size):
		imap  = imaps[fi]
		if isinstance(imap, basestring):
			iname = imap
		elif isinstance(imap, tuple):
			imap, iname = imap
		else:
			iname = ""
			#if len(imaps) == 1: iname = "map.fits"
			#else: iname = "map%0*d.fits" % (get_num_digits(len(imaps)), fi)
		with printer.time("read %s" % iname, 3):
			map, minfo = get_map(imap, args, return_info=True, name=iname)
		with printer.time("ranges", 3):
			crange= get_color_range(map, args)
		for ci, cr in enumerate(crange.T):
			printer.write("color range %d: %12.5e to %15.7e" % (ci, cr[0], cr[1]), 4)
		# Loop over map fields
		ncomp  = map.shape[0]
		ngroup = 3 if args.rgb else 1
		crange_ind = 0
		# Collect output images for this map in a list
		for i in range(0, ncomp, ngroup):
			# The unflattened index of the current field
			N = minfo.ishape[:-2]
			I = np.unravel_index(i, N) if len(N) > 0 else []
			if args.symmetric and np.any(np.sort(I) != I):
				continue
			# Construct default out format
			ndigit   = get_num_digits(ncomp)
			ndigits  = [get_num_digits(n) for n in N]
			subprint = printer.push(("%%0%dd/%%d " % ndigit) % (i+1,ncomp))
			dir, base, ext = split_file_name(minfo.fname)
			if args.odir is not None: dir = args.odir
			map_field = map[i:i+ngroup]
			if minfo.wcslist:
				# HACK: If stamp extraction messed with the wcs, fix it here
				map_field.wcs = minfo.wcslist[I[0]]
			# Build output file name
			oinfo = {"dir":"" if dir == "." else dir + "/", "base":base, "iext":ext,
					"fi":fi, "fn":len(imaps), "ci":i, "cn":ncomp, "pi":comm.rank, "pn":comm.size,
					"pre":args.prefix, "suf":args.suffix,
					"comp": "_"+"_".join(["%0*d" % (ndig,ind) for ndig,ind in zip(ndigits,I)]) if len(N) > 0 else "",
					"fcomp": "_%0*d" % (ndigit,i) if len(minfo.ishape) > 2 else "",
					"ext":args.ext, "layer":""}
			oname = args.oname.format(**oinfo)
			# Draw the map
			if args.driver.lower() == "pil":
				img, info = draw_map_field(map_field, args, crange[:,crange_ind:crange_ind+ngroup], return_info=True, return_layers=args.layers, printer=subprint, cache=cache)
				padding = np.array([-info.bounds[0,::-1],info.bounds[1,::-1]-map_field.shape[-2:]],dtype=int)
				printer.write("padded by %d %d %d %d" % tuple(padding.reshape(-1)), 4)
				if args.layers:
					for layer, name in zip(img, info.names):
						oinfo["layer"] = "_" + name
						oname = args.oname.format(**oinfo)
						yield bunch.Bunch(img=layer, name=oname, type="pil", info=info, printer=subprint, **oinfo)
				else:
					yield bunch.Bunch(img=img, name=oname, type="pil", info=info, printer=subprint, **oinfo)
			elif args.driver.lower() in ["matplotlib","mpl"]:
				figure = draw_map_field_mpl(map_field, args, crange[:,crange_ind:crange_ind+ngroup], printer=subprint)
				yield bunch.Bunch(img=figure, name=oname, type="mpl", dpi=args.mpl_dpi, printer=subprint, **oinfo)
			# Progress report
			printer.write("\r%s %5d/%d" % (iname, i+1,ncomp), 2, exact=True, newline=False)
			crange_ind += 1
		printer.write("",    2, exact=True)
		printer.write(iname, 1, exact=True)

def write(fname, plot):
	"""Write the given plot or plots to file. If plot is a single plot, then
	it will simply be written to the specified filename. If plot is a list of
	plots, then it fname will be interpreted as a prefix, and the plots will
	be written to prefix + plot.name for each individual plot. If name was
	specified during plotting, then plot.name will either be ".png" for scalar
	maps or "_0.png", "_1.png", etc. for vector maps. It's also possible to pass in
	plain images (either PIL or matplotlib), which will be written to the given
	filename."""
	"""Write the given plot to the specified file."""
	if isinstance(plot, list):
		# Allow writing a whole list of plots at once. In this case the fname is
		# interpreted as a prefix
		for p in plot:
			write(fname + p.name, p)
	else:
		try:
			printer = plot.printer if "printer" in plot else noprint
			if plot.type == "pil":
				with printer.time("write to %s" % fname, 3):
					plot.img.save(fname)
			elif plot.type == "mpl":
				with printer.time("write to %s" % fname, 3):
					plot.img.savefig(fname,bbox_inches="tight",dpi=plot.dpi)
			else:
				raise ValueError("Unknown plot type '%s'" % plot.type)
		except (AttributeError, TypeError):
			# Apparently we don't have a plot object. Check if it's a plain image
			try: plot.save(fname)
			except AttributeError:
				try: plot.savefig(fname,bbox_inches="tight",dpi=75)
				except AttributeError:
					raise ValueError("Error writing plot: The plot is not a recognized type")

def define_arg_parser():
	argdefs = []
	def add_argument(*args, **kwargs):
		short = [arg[1:] for arg in args if len(arg) >= 2 and arg[0] == "-" and arg[1] != "-"]
		long_ = [arg[2:] for arg in args if len(arg) >= 3 and arg[:2] == "--" and arg[2] != "-"]
		if len(long_) > 0: name = long_[0]
		else:              name = short[0]
		name = name.replace("-","_")
		argdefs.append([name, [args, kwargs]])
	add_argument("-o", "--oname", default="{dir}{pre}{base}{suf}{comp}{layer}.{ext}", help="The format to use for the output name. Default is {dir}{pre}{base}{suf}{comp}{layer}.{ext}")
	add_argument("-c", "--color", default="planck", help="The color scheme to use, e.g. planck, wmap, gray, hotcold, etc., or a colors pecification in the form val:rrggbb,val:rrggbb,.... Se enlib.colorize for details.")
	add_argument("-r", "--range", type=str, help="The symmetric color bar range to use. If specified, colors in the map will be truncated to [-range,range]. To give each component in a multidimensional map different color ranges, use a colon-separated list, for example -r 250:100:50 would plot the first component with a range of 250, the second with a range of 100 and the third and any subsequent component with a range of 50.")
	add_argument("--min", type=str, help="The value at which the color bar starts. See --range.")
	add_argument("--max", type=str, help="The value at which the color bar ends. See --range.")
	add_argument("-q", "--quantile", type=float, default=0.01, help="Which quantile to use when automatically determining the color range. If specified, the color bar will go from [quant(q),quant(1-q)].")
	add_argument("-v", dest="verbosity", action="count", default=0, help="Verbose output. Specify multiple times to increase verbosity further.")
	add_argument("-u", "-s", "--upgrade", "--scale", type=str, default="1", help="Upscale the image using nearest neighbor interpolation by this amount before plotting. For example, 2 would make the map twice as large in each direction, while 4,1 would make it 4 times as tall and leave the width unchanged.")
	add_argument("--verbosity", dest="verbosity", type=int, help="Specify verbosity directly as an integer.")
	add_argument("--method", default="auto", help="Which colorization implementation to use: auto, fortran or python.")
	add_argument("--slice", type=str, help="Apply this numpy slice to the map before plotting.")
	add_argument("--sub",   type=str, help="Slice a map based on dec1:dec2,ra1:ra2.")
	add_argument("-H", "--hdu",  type=int, default=0, help="Header unit of the fits file to use")
	add_argument("--op", type=str, help="Apply this general operation to the map before plotting. For example, 'log(abs(m))' would give you a lograithmic plot.")
	add_argument("-d", "--downgrade", type=str, default="1", help="Downsacale the map by this factor before plotting. This is done by averaging nearby pixels. See --upgrade for syntax.")
	add_argument("--prefix", type=str, default="", help="Specify a prefix for the output file. See --oname.")
	add_argument("--suffix", type=str, default="", help="Specify a suffix for the output file. See --oname.")
	add_argument("--odir",   type=str, default=None, help="Override the output directory. See --oname.")
	add_argument("--ext", type=str, default="png", help="Specify an extension for the output file. This will determine the file type of the resulting image. Can be anything PIL recognizes. The default is png.")
	add_argument("-m", "--mask", type=float, help="Mask this value, making it transparent in the output image. For example -m 0 would mark all values exactly equal to zero as missing.")
	add_argument("--mask-tol", type=float, default=1e-14, help="The tolerance to use with --mask.")
	add_argument("-g", "--grid", action="count", default=1, help="Toggle the coordinate grid. Disabling it can make plotting much faster when plotting many small maps.")
	add_argument("--grid-color", type=str, default="00000020", help="The RGBA color to use for the grid.")
	add_argument("-t", "--ticks", type=str, default="1", help="The grid spacing in degrees. Either a single number to be used for both axis, or ty,tx.")
	add_argument("--tick-unit", "--tu", type=str, default=None, help="Units for tick axis. Can be the unit size in degrees, or the word 'degree', 'arcmin' or 'arcsec' or the shorter 'd','m','s'.")
	add_argument("--nolabels", action="store_true", help="Disable the generation of coordinate labels outside the map when using the grid.")
	add_argument("--nstep", type=int, default=200, help="The number of steps to use when drawing grid lines. Higher numbers result in smoother curves.")
	add_argument("--subticks", type=float, default=0, help="Subtick spacing. Only supported by matplotlib driver.")
	add_argument("-b", "--colorbar", default=0, action="count", help="Whether to draw the color bar or not")
	add_argument("--font", type=str, default="arial.ttf", help="The font to use for text.")
	add_argument("--font-size", type=int, default=20, help="Font size to use for text.")
	add_argument("--font-color", type=str, default="000000", help="Font color to use for text.")
	add_argument("-D", "--driver", type=str, default="pil", help="The driver to use for plotting. Can be pil (the default) or mpl. pil cleanly maps input pixels to output pixels, and has better coordiante system support, but doesn't have as pretty grid lines or axis labels.")
	add_argument("--mpl-dpi", type=float, default=75, help="The resolution to use for the mpl driver.")
	add_argument("--mpl-pad", type=float, default=1.6, help="The padding to use for the mpl driver.")
	add_argument("--rgb", action="store_true", help="Enable RGB mode. The input maps must have 3 components, which will be interpreted as red, green and blue channels of a single image instead of 3 separate images as would be the case without this option. The color scheme is overriden in this case.")
	add_argument("--reverse-color",  action="store_true", help="Reverse the color scale. For example, a black-to-white scale will become a white-to-black sacle.")
	add_argument("-a", "--autocrop", action="store_true", help="Automatically crop the image by removing expanses of uniform color around the edges. This is done jointly for all components in a map, making them directly comparable, but is done independently for each input file.")
	add_argument("-A", "--autocrop-each", action="store_true", help="As --autocrop, but done individually for each component in each map.")
	add_argument("-L", "--layers", action="store_true", help="Output the individual layers that make up the final plot (such as the map itself, the coordinate grid, the axis labels, any contours and lables) as individual files instead of compositing them into a final image.")
	add_argument(      "--no-image", action="store_true", help="Skip the main image plotting. Useful for getting a pure contour plot, for example.")
	add_argument("-C", "--contours", type=str, default=None, help="Enable contour lines. For example -C 10 to place a contour at every 10 units in the map, -C 5:10 to place it at every 10 units, but starting at 5, and 1,2,4,8 or similar to place contours at manually chosen locations.")
	add_argument("--contour-type",  type=str, default="uniform", help="The type of the contour specification. Only used when the contours specification is a list of numbers rather than a string (so not from the command line interface). 'uniform': the list is [interval] or [base, interval]. 'list': the list is an explicit list of the values the contours should be at.")
	add_argument("--contour-color", type=str, default="000000", help="The color scheme to use for contour lines. Either a single rrggbb, a val:rrggbb,val:rrggbb,... specification or a color scheme name, such as planck, wmap or gray.")
	add_argument("--contour-width", type=int, default=1, help="The width of each contour line, in pixels.")
	add_argument("--annotate",      type=str, default=None, help="""Annotate the map with text, lines or circles. Should be a text file with one entry per line, where an entry can be:
		c[ircle] lat lon dy dx [rad [width [color]]]
		t[ext]   lat lon dy dx text [size [color]]
		l[ine]   lat lon dy dx lat lon dy dx [width [color]]
	dy and dx are pixel-unit offsets from the specified lat/lon.""")
	add_argument("--annotate-maxrad", type=int, default=0, help="Assume that annotations do not extend further than this from their center, in pixels. This is used to prune which annotations to attempt to draw, as they can be a bit slow. The special value 0 disables this.")
	add_argument("--stamps", type=str, default=None, help="Plot stamps instead of the whole map. Format is srcfile:size:nmax, where the last two are optional. srcfile is a file with [ra dec] in degrees, size is the size in pixels of each stamp, and nmax is the max number of stamps to produce.")
	add_argument("--tile",  type=str, default=None, help="Stack components vertically and horizontally. --tile 5,4 stacks into 5 rows and 4 columns. --tile 5 or --tile 5,-1 stacks into 5 rows and however many columns are needed. --tile -1,5 stacks into 5 columns and as many rows are needed. --tile -1 allocates both rows and columns to make the result as square as possible. The result is treated as a single enmap, so the wcs will only be right for one of the tiles.")
	add_argument("--tile-transpose", action="store_true", help="Transpose the ordering of the fields when tacking. Normally row-major stacking is used. This sets column-major order instead.")
	add_argument("-S", "--symmetric", action="store_true", help="Treat the non-pixel axes as being asymmetric matrix, and only plot a non-redundant triangle of this matrix.")
	add_argument("-z", "--zenith",    action="store_true", help="Plot the zenith angle instead of the declination.")
	add_argument("-F", "--fix-wcs",   action="store_true", help="Fix the wcs for maps in cylindrical projections where the reference point was placed too far away from the map center.")

	# Define the argument parser
	parser   = argparse.ArgumentParser()
	optnames = ["ifiles"]
	parser.add_argument("ifiles", nargs="*", help="The map files to plot. Each file will be processed independently and output as an image file with a name derived from that of the input file (see --oname). For each file a color range will be determined, and each component of the map (if multidimensional) will be written to a separate image file. If the file has more than 1 non-pixel dimension, these will be flattened first.")
	for name, (args, kwargs) in argdefs:
		parser.add_argument(*args, **kwargs)
		optnames.append(name)
	optnames = set(optnames)
	return parser, optnames

arg_parser, optnames = define_arg_parser()
# Hack: Update the plot docstring. I suspect that this will confuse automated tools
help_short= "\n\t".join(arg_parser.format_help().split("positional arguments:")[0].split("\n")).rstrip()
help_long = "\n\t".join(arg_parser.format_help().split("optional arguments:")[1].split("\n"))
plot.__doc__ += "\n\t" + help_long

def parse_args(args=sys.argv[1:], noglob=False):
	if isinstance(args, basestring):
		args = shlex.split(args)
	res = arg_parser.parse_args(args)
	res = bunch.Bunch(**res.__dict__)
	# Glob expansion
	if not noglob:
		ifiles = []
		for pattern in res.ifiles:
			matches = glob.glob(pattern)
			if len(matches) > 0:
				ifiles += matches
			else:
				ifiles.append(pattern)
		res.ifiles = ifiles
	return res

def extract_arg(args, name, default):
	if name not in args: return default
	res = args[name]
	del args[name]
	return res

def check_args(kwargs):
	for key in kwargs:
		if not key in optnames:
			msg = "Unrecognized argument '%s'\n\n%s" % (key, help_short)
			raise ValueError(msg)

def get_map(ifile, args, return_info=False, name=None):
	"""Read the specified map, and massage it according to the options
	in args. Relevant ones are sub, autocrop, slice, op, downgrade, scale,
	mask. Retuns with shape [:,ny,nx], where any extra dimensions have been
	flattened into a single one."""
	# TODO: this should be reorganized so that slicing can happen earlier.
	# Currently the whole file needs to be read.
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		if isinstance(ifile, basestring):
			toks  = ifile.split(":")
			ifile, slice = toks[0], ":".join(toks[1:])
			m0    = enmap.read_map(ifile, hdu=args.hdu)
			if name is None: name = ifile
		else:
			m0    = ifile
			slice = ""
			if name is None: name = ".fits"
		# This fills in a dummy, plain wcs if one does not exist
		m0 = enmap.enmap(m0, copy=False)
		if args.fix_wcs:
			m0.wcs = wcsutils.fix_wcs(m0.wcs)
		# Save the original map, so we can compare its wcs later
		m  = m0
		# Submap slicing currently has wrapping issues
		if args.sub is not None:
			default = [[-90,-180],[90,180]]
			sub  = np.array([[(default[j][i] if q == '' else float(q))*np.pi/180 for j,q in enumerate(w.split(":"))]for i,w in enumerate(args.sub.split(","))]).T
			m = m.submap(sub)
		# Perform a common autocrop across all fields
		if args.autocrop:
			m = enmap.autocrop(m)
		# If necessary, split into stamps. If no stamp splitting occurs,
		# a list containing only the original map is returned
		mlist = extract_stamps(m, args)
		# The stamp stuff is a bit of an ugly hack. This loop and wcslist
		# are parts of that hack.
		for i, m in enumerate(mlist):
			# Downgrade
			downgrade = parse_list(args.downgrade, int)
			m = enmap.downgrade(m, downgrade)
			# Slicing, either at the file name level or though the slice option
			m = eval("m"+slice)
			if args.slice is not None:
				m = eval("m"+args.slice)
			flip = (m.wcs.wcs.cdelt*m0.wcs.wcs.cdelt)[::-1]<0
			assert m.ndim >= 2, "Image must have at least 2 dimensions"
			# Apply arbitrary map operations
			m1 = m
			if args.op is not None:
				m = eval(args.op, {"m":m,"enmap":enmap,"utils":utils},np.__dict__)
			# Scale if requested
			scale = parse_list(args.upgrade, int)
			if np.any(np.array(scale)>1):
				m = enmap.upgrade(m, scale)
			# Flip such that pixels are in PIL or matplotlib convention,
			# such that RA increases towards the left and dec upwards in
			# the final image. Unless a slicing operation on the image
			# overrrode this.
			if m.wcs.wcs.cdelt[1] > 0: m = m[...,::-1,:]
			if m.wcs.wcs.cdelt[0] > 0: m = m[...,:,::-1]
			if flip[0]: m = m[...,::-1,:]
			if flip[1]: m = m[...,:,::-1]
			# Update stamp list
			mlist[i] = m
		wcslist = [m.wcs for m in mlist]
		m = enmap.samewcs(np.asarray(mlist),mlist[0])
		if args.stamps is None:
			m, wcslist = m[0], None
		# Flatten pre-dimensions
		mf = m.reshape((-1,)+m.shape[-2:])
		# Stack
		if args.tile is not None:
			toks = parse_list(args.tile, int)
			nrow = toks[0] if len(toks) > 0 else -1
			ncol = toks[1] if len(toks) > 1 else -1
			mf = hwstack(hwexpand(mf, nrow, ncol, args.tile_transpose))[None]
		# Mask bad data
		if args.mask is not None:
			if not np.isfinite(args.mask): mf[np.abs(mf)==args.mask] = np.nan
			else: mf[np.abs(mf-args.mask)<=args.mask_tol] = np.nan
		# Done
		if not return_info: return mf
		else:
			info = bunch.Bunch(fname=name, ishape=m.shape, wcslist=wcslist)
			return mf, info

def extract_stamps(map, args):
	"""Given a map, extract a set of identically sized postage stamps based on
	args.stamps. Returns a new map consisting of a stack of these stamps, along
	with a list of each of these' wcs object."""
	if args.stamps is None: return [map]
	# Stamps specified by format srcfile[:size[:nmax]], where the srcfile has
	# lines of [ra, dec] in degrees
	toks = args.stamps.split(":")
	# Read in our positions, optionally truncating the list
	srcs = np.loadtxt(toks[0], usecols=[0,1]).T[1::-1]*utils.degree
	size = int(toks[1]) if len(toks) > 1 else 16
	nsrc = int(toks[2]) if len(toks) > 2 else len(srcs.T)
	srcs = srcs[:,:nsrc]
	# Convert to pixel coordinates of corners
	pix  = np.round(map.sky2pix(srcs)-0.5*size).astype(int)
	# Extract stamps
	return map.stamps(pix.T, size, aslist=True)

def get_cache(cache, key, fun):
	if cache is None: return fun()
	if key not in cache: cache[key] = fun()
	return cache[key]

def draw_map_field(map, args, crange=None, return_layers=False, return_info=False, printer=noprint, cache=None):
	"""Draw a single map field, resulting in a single image. Adds a coordinate grid
	and lables as specified by args. If return_layers is True, an array will be
	returned instead of an image, wich each entry being a component of the image,
	such as the base image, the coordinate grid, the labels, etc. If return_bounds
	is True, then the """
	map, color = prepare_map_field(map, args, crange, printer=printer)
	tag    = (tuple(map.shape), map.wcs.to_header_string(), repr(args))
	layers = []
	names  = []
	yoff   = map.shape[-2]
	# Image layer
	if not args.no_image:
		with printer.time("to image", 3):
			img = PIL.Image.fromarray(utils.moveaxis(color,0,2)).convert('RGBA')
		layers.append((img,[[0,0],img.size]))
		names.append("img")
	# Contours
	if args.contours:
		with printer.time("draw contours", 3):
			contour_levels = calc_contours(crange, args)
			cimg = draw_contours(map, contour_levels, args)
			layers.append((cimg, [[0,0],cimg.size]))
			names.append("cont")
	# Annotations
	if args.annotate:
		with printer.time("draw annotations", 3):
			def get_aimg():
				annots = parse_annotations(args.annotate)
				return draw_annotations(map, annots, args)
			aimg = get_cache(cache, ("annotate",tag), get_aimg)
			layers.append((aimg, [[0,0],aimg.size]))
			names.append("annot")
	# Coordinate grid
	if args.grid % 2:
		with printer.time("draw grid", 3):
			ginfo = get_cache(cache, ("ginfo",tag), lambda: calc_gridinfo(map.shape, map.wcs, args))
			grid  = get_cache(cache, ("grid", tag), lambda: draw_grid(ginfo, args))
			layers.append(grid)
			names.append("grid")
		if not args.nolabels:
			with printer.time("draw labels", 3):
				labels, bounds = get_cache(cache, ("labels",tag), lambda: draw_grid_labels(ginfo, args))
				yoff = bounds[1][1]
				layers.append((labels,bounds))
				names.append("tics")
	if args.colorbar % 2:
		with printer.time("draw colorbar", 3):
			bimg, bounds = draw_colorbar(crange, map.shape[-1], args)
			bounds[:,1] += yoff
			yoff = bounds[1,1]
			layers.append((bimg,bounds))
			names.append("colorbar")
	 # Possibly other stuff too, like point source circles
	 # or contours
	with printer.time("stack layers", 3):
		layers, bounds = standardize_images(layers)
		if not return_layers: layers = merge_images(layers)
	info = bunch.Bunch(bounds=bounds, names=names)
	if return_info: return layers, info
	else: return layers

def draw_colorbar(crange, width, args):
	col  = tuple([int(args.font_color[i:i+2],16) for i in range(0,len(args.font_color),2)])
	font = cgrid.get_font(args.font_size)
	fmt  = "%g"
	labels, boxes = [], []
	for val in crange:
		labels.append(fmt % val)
		boxes.append(font.getsize(labels[-1]))
	boxes = np.array(boxes,int)
	lw, lh = np.max(boxes,0)
	img    = PIL.Image.new("RGBA", (width, lh))
	draw   = PIL.ImageDraw.Draw(img)
	# Draw the labels on the image
	draw.text((lw-boxes[0,0], 0), labels[0], col, font=font)
	draw.text((width-lw, 0),      labels[1], col, font=font)
	# Draw the color bar itself
	bar_data    = np.zeros((lh,width-2*lw))
	bar_data[:] = np.linspace(0,1,bar_data.shape[-1])
	bar_col     = map_to_color(bar_data[None], [0,1], args)
	bar_img     = PIL.Image.fromarray(utils.moveaxis(bar_col,0,2)).convert('RGBA')
	# Overlay it on the output image
	img.paste(bar_img, (lw,0))
	bounds = np.array([[0,0],[width,lh]])
	return img, bounds

def draw_map_field_mpl(map, args, crange=None, printer=noprint):
	"""Render a map field using matplotlib. Less tested and
	maintained than draw_map_field, and supports fewer features.
	Returns an object one can call savefig on to draw."""
	map, color = prepare_map_field(map, args, crange, printer=printer)
	# Set up matplotlib. We do it locally here to
	# avoid having it as a dependency in general
	with printer.time("matplotplib", 3):
		import matplotlib
		matplotlib.use("Agg")
		from matplotlib import pyplot, ticker
		matplotlib.rcParams.update({'font.size': 10})
		dpi, pad = args.mpl_dpi, args.mpl_pad
		winch, hinch = map.shape[2]/dpi, map.shape[1]/dpi
		fig  = pyplot.figure(figsize=(winch+pad,hinch+pad))
		box  = map.box()*180/np.pi
		pyplot.imshow(utils.moveaxis(color,0,2), extent=[box[0,1],box[1,1],box[1,0],box[0,0]])
		# Make conformal in center of image
		pyplot.axes().set_aspect(1/np.cos(np.mean(map.box()[:,0])))
		if args.grid % 2:
			ax = pyplot.axes()
			ticks = np.full(2,1.0)
			ticks[:] = parse_list(args.ticks)
			ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[1]))
			ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[0]))
			if args.subticks:
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(args.sub))
				ax.yaxis.set_minor_locator(ticker.MultipleLocator(args.sub))
				pyplot.minorticks_on()
				pyplot.grid(True, which="major", linewidth=2)
				pyplot.grid(True, which="minor", linewidth=1)
			else:
				pyplot.grid(True)
		pyplot.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
	return pyplot
	#pyplot.savefig(oname,bbox_inches="tight",dpi=dpi)

def parse_range(desc,n):
	res = parse_list(desc, sep=":")[:n]
	return np.concatenate([res,np.repeat([res[-1]],n-len(res))])

def parse_list(desc, dtype=float, sep=","):
	if isinstance(desc, basestring):
		# Support strings of type "1.2,3.4,5,6"
		return [dtype(w) for w in desc.split(sep)]
	elif isinstance(desc, list) or isinstance(desc, tuple):
		# Support lists of numbers
		return [dtype(w) for w in desc]
	else:
		# Perhaps it's just a single plain number? If so,
		# convert it to the right dtype and return it as a 1-element list
		return [dtype(desc)]

def get_color_range(map, args):
	"""Compute an appropriate color bare range from map[:,ny,nx]
	given the args. Relevant members are range, min, max, quantile."""
	# Construct color ranges
	ncomp  = map.shape[0]
	crange = np.zeros((2,ncomp))+np.nan
	# Try explicit limits if given
	if args.range is not None:
		crange[1] = parse_range(args.range,ncomp)
		crange[0] = -crange[1]
	if args.min is not None: crange[0] = parse_range(args.min,ncomp)
	if args.max is not None: crange[1] = parse_range(args.max,ncomp)
	# Fall back on quantile otherwise
	if np.any(np.isnan(crange)):
		vals = np.sort(map[np.isfinite(map)])
		n    = len(vals)
		if n == 0: return np.repeat(np.array([-1,1])[:,None], ncomp, -1)
		i    = min(n-1,int(round(n*args.quantile)))
		v1, v2 = vals[i], vals[n-1-i]
		# Avoid division by zero later, in case min and max are the same
		if v2 == v1: (v1,v2) = (v1-1,v2+1)
		crange[0,np.isnan(crange[0])] = v1
		crange[1,np.isnan(crange[1])] = v2
	return crange

def get_num_digits(n): return int(np.log10(n))+1
def split_file_name(fname):
	"""Split a file name into directory, base name and extension,
	such that fname = dirname + "/" + basename + "." + ext."""
	dirname  = os.path.dirname(fname)
	if len(dirname) == 0: dirname = "."
	base_ext = os.path.basename(fname)
	# Find the extension. Using the last dot does not work for .fits.gz.
	# Using the first dot in basename does not work for foo2.5_bar.fits.
	# Will therefore handle .gz as a special case.
	if base_ext.endswith(".gz"):
		dot = base_ext[:-3].rfind(".")
	else:
		dot  = base_ext.rfind(".")
	if dot < 0: dot = len(base_ext)
	base = base_ext[:dot]
	ext  = base_ext[dot+1:]
	return dirname, base, ext

def map_to_color(map, crange, args):
	"""Compute an [{R,G,B},ny,nx] color map based on a map[1 or 3, ny,nx]
	map and a corresponding color range crange[{min,max}]. Relevant args
	fields: color, method, rgb. If rgb is not true, only the first element
	of the input map will be used. Otherwise 3 will be used."""
	map = ((map.T-crange[0])/(crange[1]-crange[0])).T # .T ensures broadcasting for rgb case
	if args.reverse_color: map = 1-map
	if args.rgb: m_color = colorize.colorize(map,    desc=args.color, driver=args.method, mode="direct")
	else:        m_color = colorize.colorize(map[0], desc=args.color, driver=args.method)
	m_color = enmap.samewcs(np.rollaxis(m_color,2), map)
	return m_color

def calc_gridinfo(shape, wcs, args):
	"""Compute the points making up the grid lines for the given map.
	Depends on args.ticks and args.nstep."""
	ticks = np.full(2,1.0)
	ticks[:] = parse_list(args.ticks)
	try:               unit = float(args.tick_unit)
	except TypeError:  unit = 1.0
	except ValueError: unit = args.tick_unit
	return cgrid.calc_gridinfo(shape, wcs, steps=ticks, nstep=args.nstep, zenith=args.zenith, unit=unit)

def draw_grid(ginfo, args):
	"""Return a grid based on gridinfo. args.grid_color controls the color
	the grid will be drawn with."""
	grid = cgrid.draw_grid(ginfo, color=args.grid_color)
	bounds = np.array([[0,0],ginfo.shape[::-1]])
	return grid, bounds

def draw_grid_labels(ginfo, args):
	"""Return an image with a coordinate grid, along with abounds of this
	image relative to the coordinate shape stored in ginfo. Depends
	on the following args members: args.font, args.font_size, args.font_color"""
	linfo = []
	for gi in [ginfo.lat, ginfo.lon]:
		linfo += cgrid.calc_label_pos(gi, ginfo.shape[::-1])
	canvas = PIL.Image.new("RGBA", ginfo.shape[::-1])
	labels, bounds = cgrid.draw_labels(canvas, linfo, fname=args.font, fsize=args.font_size, color=args.font_color, return_bounds=True)
	return labels, bounds

def calc_contours(crange, args):
	"""Returns a list of values at which to place contours based on
	the valure range of the map crange[{from,to}] and the contour
	specification in args.

	Contour specifications:
		base:step or val,val,val...
	base: number
	step: number (explicit), -number (relative)
	"""
	if args.contours is None: return None
	def setup_uniform(vals):
		if len(vals) == 1:
			base, step = 0, vals[0]
		else:
			base, step = vals[:2]
		if step < 0:
			step = (crange[1]-crange[0])/(-step)
		# expand to fill crange
		a = int(np.ceil ((crange[0]-base)/step))
		b = int(np.floor((crange[1]-base)/step))+1
		return np.arange(a,b)*step + base
	# The input can either be a string or a list of numbers.
	# If it's a string, then it's interpreted based on whether
	# it contains commas or colons. If it's a list, then the
	# type is conrolled by args.contour_type
	if isinstance(args.contours, basestring):
		vals = args.contours.split(",")
		if len(vals) > 1:
			return np.array([float(v) for v in vals if len(v) > 0])
		else:
			return setup_uniform([float(tok) for tok in args.contours.split(":")])
	else:
		vals = parse_list(args.contours, float)
		if   args.contour_type == "list":    return np.array(vals)
		elif args.contour_type == "uniform":
			return setup_uniform(vals)
		else: raise ValueError("Unknown contour type '%s'" % args.contour_type)

def draw_contours(map, contours, args):
	img   = PIL.Image.new("RGBA", map.shape[-2:][::-1])
	inds  = np.argsort(contours)
	cmap  = array_ops.find_contours(map[0], contours[inds])
	cmap  = contour_widen(cmap, args.contour_width)
	cmap -= 1
	# Undo sorting if we sorted
	if not np.allclose(inds, np.arange(len(inds))):
		mask = cmap>=0
		cmap[mask] = inds[cmap[mask]]
	cmap  = cmap.astype(float)
	# Make non-contour areas transparent
	cmap[cmap<0] = np.nan
	# Rescale to 0:1
	if len(contours) > 1:
		cmap /= len(contours)-1
	color = colorize.colorize(cmap, desc=args.contour_color, driver=args.method)
	return PIL.Image.fromarray(color).convert('RGBA')

def parse_annotations(afile):
	with open(afile,"r") as f:
		return [shlex.split(line) for line in f]

def draw_annotations(map, annots, args):
	"""Draw a set of annotations on the map. These are specified
	as a list of ["type",param,param,...]. The recognized formats
	are:
		c[ircle] lat lon dy dx [rad [width [color]]]
		t[ext]   lat lon dy dx text [size [color]]
		l[ine]   lat lon dy dx lat lon dy dx [width [color]]
		r[ect]   lat lon dy dx lat lon dy dx [width [color]]
	dy and dx are pixel-unit offsets from the specified lat/lon.
	This is useful for e.g. placing text next to circles."""
	img  = PIL.Image.new("RGBA", map.shape[-2:][::-1])
	draw = PIL.ImageDraw.Draw(img, "RGBA")
	font = None
	font_size_prev = 0
	def topix(pos_off):
		unit = utils.degree if not wcsutils.is_plain(map.wcs) else 1.0
		pix = map.sky2pix(np.array([float(w) for w in pos_off[:2]])*unit)
		pix += np.array([float(w) for w in pos_off[2:]])
		return pix[::-1].astype(int)
	def skippable(x,y):
		rmax = args.annotate_maxrad
		if rmax == 0: return False
		return x <= -rmax or y <= -rmax or x >= map.shape[-1]-1+rmax or y >= map.shape[-2]-1+rmax
	for annot in annots:
		atype = annot[0].lower()
		color = "black"
		width = 2
		if atype in ["c","circle"]:
			x,y = topix(annot[1:5])
			if skippable(x,y): continue
			rad = 8
			if len(annot) > 5: rad   = int(annot[5])
			if len(annot) > 6: width = int(annot[6])
			if len(annot) > 7: color = annot[7]
			antialias = 1 if width < 1 else 4
			draw_ellipse(img,
					(x-rad,y-rad,x+rad,y+rad),
					outline=color,width=width, antialias=antialias)
		elif atype in ["l","line"] or atype in ["r","rect"]:
			x1,y1 = topix(annot[1:5])
			x2,y2 = topix(annot[5:9])
			nphi   = utils.nint(abs(360/map.wcs.wcs.cdelt[0]))
			x1, x2 = utils.unwind([x1,x2], nphi, ref=nphi//2)
			if skippable(x1,y1) and skippable(x2,y2): continue
			if len(annot) >  9: width = int(annot[9])
			if len(annot) > 10: color = annot[10]
			if atype[0] == "l":
				draw.line((x1,y1,x2,y2), fill=color, width=width)
			else:
				if x2 < x1: x1,x2 = x2,x1
				if y2 < y1: y1,y2 = y2,y1
				for i in range(width):
					draw.rectangle((x1+i,y1+i,x2-i,y2-i), outline=color)
		elif atype in ["t", "text"]:
			x,y  = topix(annot[1:5])
			if skippable(x,y): continue
			text = annot[5]
			size = 16
			if len(annot) > 6: size  = int(annot[6])
			if len(annot) > 7: color = annot[7]
			if font is None or size != font_size_prev:
				font = cgrid.get_font(size)
				font_size_prev = size
			tbox = font.getsize(text)
			draw.text((x-tbox[0]/2, y-tbox[1]/2), text, color, font=font)
		else:
			raise NotImplementedError
	return img

def standardize_images(tuples):
	"""Given a list of (img,bounds), composite them on top of each other
	(first at the bottom), and return the total image and its new bounds."""
	bounds_all = np.array([bounds for img, bounds in tuples])
	bounds_full= cgrid.calc_bounds(bounds_all, tuples[0][1][1])
	# Build canvas
	totsize = bounds_full[1]-bounds_full[0]
	res = []
	for img, bounds in tuples:
		# Expand to full size
		img_big = PIL.Image.new("RGBA", tuple(totsize))
		img_big.paste(img, tuple(bounds[0]-bounds_full[0]))
		res.append(img_big)
	return res, bounds_full

def merge_images(images):
	"""Stack all images into an alpha composite. Images must all have consistent
	extent before this. Use standardize_images to achieve this."""
	res = images[0]
	for img in images[1:]:
		res = PIL.Image.alpha_composite(res, img)
	return res

def merge_plots(plots):
	res = plots[0].copy()
	imgs, bounds = standardize_images([(plot.img, plot.info.bounds) for plot in plots])
	res.img         = imgs[0]
	res.info.bounds = bounds
	for img in imgs[1:]:
		res.img = PIL.Image.alpha_composite(res.img, img)
	return res

def prepare_map_field(map, args, crange=None, printer=noprint):
	if crange is None:
		with printer.time("ranges", 3):
			crange = get_color_range(map, args)
	if map.ndim == 2:
		map = map[None]
	if args.autocrop_each:
		map = enmap.autocrop(map)
	with printer.time("colorize", 3):
		color = map_to_color(map, crange, args)
	return map, color

def makefoot(n):
	b = np.full((2*n+1,2*n+1),1)
	b[n,n] = 0
	b = ndimage.distance_transform_edt(b)
	return b[1::2,1::2] < n

def contour_widen(cmap, width):
	if width <= 1: return cmap
	foot = makefoot(width)
	return ndimage.grey_dilation(cmap, footprint=foot)

def draw_ellipse(image, bounds, width=1, outline='white', antialias=1):
	"""Improved ellipse drawing function, based on PIL.ImageDraw.
	Improved from
	http://stackoverflow.com/questions/32504246/draw-ellipse-in-python-pil-with-line-thickness"""
	bounds = np.asarray(bounds)
	# Create small coordinate system around ellipse, with a
	# margin of width on each side
	esize  = bounds[2:]-bounds[:2] + 2*width
	ebounds= bounds - bounds[[0,1,0,1]] + width
	# Use a single channel image (mode='L') as mask.
	# The size of the mask can be increased relative to the imput image
	# to get smoother looking results. 
	mask = PIL.Image.new(size=tuple(esize*antialias), mode='L', color='black')
	draw = PIL.ImageDraw.Draw(mask)
	# draw outer shape in white (color) and inner shape in black (transparent)
	for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
		a = (ebounds[:2] + offset)*antialias
		b = (ebounds[2:] - offset)*antialias
		draw.ellipse([a[0],a[1],b[0],b[1]], fill=fill)
	# downsample the mask using PIL.Image.LANCZOS 
	# (a high-quality downsampling filter).
	mask = mask.resize(esize, PIL.Image.LANCZOS)
	# paste outline color to input image through the mask
	image.paste(outline, tuple(bounds[:2]-width), mask=mask)

def hwexpand(mflat, nrow=-1, ncol=-1, transpose=False):
	"""Stack the maps in mflat[n,ny,nx] into a single flat map mflat[nrow,ncol,ny,nx]"""
	n, ny, nx = mflat.shape
	if nrow < 0 and ncol < 0:
		ncol = int(np.ceil(n**0.5))
	if nrow < 0: nrow = (n+ncol-1)//ncol
	if ncol < 0: ncol = (n+nrow-1)//nrow
	if not transpose:
		omap = enmap.zeros([nrow,ncol,ny,nx],mflat.wcs,mflat.dtype)
		omap.reshape(-1,ny,nx)[:n] = mflat
	else:
		omap = enmap.zeros([ncol,nrow,ny,nx],mflat.wcs,mflat.dtype)
		omap.reshape(-1,ny,nx)[:n] = mflat
		omap = np.transpose(omap,(1,0,2,3))
	return omap

def hwstack(mexp):
	nr,nc,ny,nx = mexp.shape
	return np.transpose(mexp,(0,2,1,3)).reshape(nr*ny,nc*nx)

class BackendError(BaseException): pass

def show(img, title=None, method="auto"):
	if   method == "tk":      show_tk(img, title=title)
	elif method == "qt":      show_qt(img, title=title)
	elif method == "wx":      show_wx(img, title=title)
	elif method == "ipython": show_ipython(img, title=title)
	elif method == "auto":
		# If matplotlib exists, use its perferences
		#try:
		#	import matplotlib
		#	backend = matplotlib.get_backend().lower()
		#	if    backend.startswith("tk"): return show_tk(img, title)
		#	elif  backend.startswith("qt"): return show_qt(img, title)
		#	elif  backend.startswith("wx"): return show_wx(img, title)
		#	elif "ipykernel" in backend: return show_ipython(img, title)
		#except ImportError: pass
		# If we get to this point, matplotlib couldn't tell us what to
		# do. Try them one by one in priority order
		try:
			# Only use ipython for graphical notebooks
			if "ZMQ" in get_ipython().__class__.__name__:
				return show_ipython(img, title=title)
		except (ImportError, NameError): pass
		try: return show_tk(img, title=title)
		except (ImportError, BackendError): pass
		try: return show_wx(img, title=title)
		except ImportError: pass
		try: return show_qt(img, title=title)
		except ImportError: pass
		# If we got here, nothing worked
		raise ImportError("Could not find any working display backends for enplot.show")

def show_ipython(img, title=None):
	from IPython.core.display import display
	for img_, title_ in _show_helper(img, title):
		display(img_)

def show_tk(img, title=None):
	from six.moves import tkinter
	from PIL import ImageTk
	class Displayer:
		def __init__(self):
			self.root    = tkinter.Tk()
			self.root.withdraw()
			self.windows = []
			self.nclosed = 0
		def add_window(self, img, title):
			window   = tkinter.Toplevel()
			window.minsize(img.width, img.height)
			window.title(title)
			canvas = tkinter.Canvas(window, width=img.width, height=img.height)
			canvas.pack()
			canvas.configure(background="white")
			photo  = ImageTk.PhotoImage(img)
			sprite = canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
			self.windows.append([window,canvas,photo,sprite])
			def closer():
				self.nclosed += 1
				window.destroy()
				if self.nclosed >= len(self.windows): self.root.destroy()
			window.protocol("WM_DELETE_WINDOW", closer)
	try:
		app = Displayer()
		for img_, title_ in _show_helper(img, title):
			app.add_window(img_, title_)
		app.root.mainloop()
	except tkinter.TclError: raise BackendError

def show_wx(img, title=None):
	import wx
	from PIL import Image
	class Panel(wx.Panel):
		def __init__(self, parent, id, img):
			wx.Panel.__init__(self, parent, id)
			self.SetBackgroundColour("white")
			# Make a non-transparend image with white background
			background = Image.new("RGB", img.size, (255,255,255))
			background.paste(img, mask=img.split()[3])
			wximg = wx.EmptyImage(*background.size)
			wximg.SetData(background.convert('RGB').tobytes())
			wxbmp = wx.BitmapFromImage(wximg)

			sizer = wx.BoxSizer()
			self.img_control = wx.StaticBitmap(self, -1, wxbmp, (0,0))
			sizer.Add(self.img_control, 1, wx.EXPAND)
			self.SetSizer(sizer)
			sizer.Fit(parent)
	app = wx.App(False)
	frames = []
	for img_, title_ in _show_helper(img, title):
		frame = wx.Frame(None, -1, title_, size=img_.size)
		Panel(frame,-1, img_)
		frame.Show(1)
		frames.append(frame)
	app.MainLoop()

def show_qt(img, title=None):
	from matplotlib.backends.backend_qt5 import QtCore, QtGui, QtWidgets
	from PIL.ImageQt import ImageQt
	import sys
	# Set up window
	class ImageWindow(QtWidgets.QMainWindow):
		def __init__(self, img, title):
			QtWidgets.QMainWindow.__init__(self)
			self.setWindowTitle(title)
			widget = QtWidgets.QWidget()
			self.setCentralWidget(widget)
			layout = QtWidgets.QVBoxLayout(widget)
			qimg   = QtGui.QImage(ImageQt(img))
			pixmap = QtGui.QPixmap(qimg)
			label = QtWidgets.QLabel()
			label.setPixmap(pixmap)
			label.adjustSize()
			layout.addWidget(label)
			self.resize(label.width(), label.height())
	app    = QtWidgets.QApplication([])
	windows= []
	for img_, title_ in _show_helper(img, title):
		window = ImageWindow(img_, title_)
		window.show()
		windows.append(window)
	app.exec_()

def _show_helper(img, title=None):
	res = []
	if isinstance(img, list):
		for i, im in enumerate(img):
			if isinstance(title, list): tit = title[i]
			else: tit = title
			res += _show_helper(im, tit)
		return res
	else:
		try:
			return [(img.img, (title or img.name))]
		except AttributeError:
			return [(img, (title or "plot"))]
