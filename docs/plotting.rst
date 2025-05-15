Plotting
========

:py:mod:`pixell.enplot` can be used to plot maps and the results of any analysis performed with pixell.

The module is a wrapper around different map plotting mechanisms. It provides the basic plotting functionality, such as plot and show, as well as the ability to read and write maps. `enplot` module can also be used as an executable.
Since this module can be called as an executable, let's first see all the arguments one can pass:

:param oname: The format to use for the output name. Default is {dir}{pre}{base}{suf}{comp}{layer}.{ext}
:param color: The color scheme to use, e.g. planck, wmap, gray, hotcold, etc., or a colors pecification in the form val:rrggbb,val:rrggbb. The default value is planck. More in :py:mod:`pixell.enlib.colorize`
:param range: The symmetric color bar range to use. If specified, colors in the map will be truncated to [-range,range]. To give each component in a multidimensional map different color ranges, use a colon-separated list, for example -r 250:100:50 would plot the first component with a range of 250, the second with a range of 100 and the third and any subsequent component with a range of 50.
:param min, type=str, help="The value at which the color bar starts. See --range."
:param max, type=str, help="The value at which the color bar ends. See --range."
:param quantile, type=float, default=0.01, help="Which quantile to use when automatically determining the color range. If specified, the color bar will go from [quant(q),quant(1-q)]."
:param -v, dest="verbosity", action="count", default=0, help="Verbose output. Specify multiple times to increase verbosity further."
:param upgrade:
:param scale:", type=str, default="1", help="Upscale the image using nearest neighbor interpolation by this amount before plotting. For example, 2 would make the map twice as large in each direction, while 4,1 would make it 4 times as tall and leave the width unchanged."
:param "--verbosity", dest="verbosity", type=int, help="Specify verbosity directly as an integer."
:param "--method", default="auto", help="Which colorization implementation to use: auto, fortran or python."
:param "--slice", type=str, help="Apply this numpy slice to the map before plotting."
:param "--sub",   type=str, help="Slice a map based on dec1:dec2,ra1:ra2."
:param "-H", "--hdu",  type=int, default=0, help="Header unit of the fits file to use"
:param "--op", type=str, help="Apply this general operation to the map before plotting. For example, 'log(abs(m))' would give you a lograithmic plot."
:param "--op2", type=str, help="Like op, but allows multiple statements"
:param "-d", "--downgrade", type=str, default="1", help="Downsacale the map by this factor before plotting. This is done by averaging nearby pixels. See --upgrade for syntax."
:param "--prefix", type=str, default="", help="Specify a prefix for the output file. See --oname."
:param "--suffix", type=str, default="", help="Specify a suffix for the output file. See --oname."
:param "--odir",   type=str, default=None, help="Override the output directory. See --oname."
:param "--ext", type=str, default="png", help="Specify an extension for the output file. This will determine the file type of the resulting image. Can be anything PIL recognizes. The default is png."
:param "-m", "--mask", type=float, help="Mask this value, making it transparent in the output image. For example -m 0 would mark all values exactly equal to zero as missing."
:param "--mask-tol", type=float, default=1e-14, help="The tolerance to use with --mask."
:param "-g", "--grid", action="count", default=1, help="Toggle the coordinate grid. Disabling it can make plotting much faster when plotting many small maps."
:param "--grid-color", type=str, default="00000020", help="The RGBA color to use for the grid."
:param "--grid-width", type=int, default=1, help="The line width to use for the grid."
:param "-t", "--ticks", type=str, default="1", help="The grid spacing in degrees. Either a single number to be used for both axis, or ty,tx."
:param "--tick-unit", "--tu", type=str, default=None, help="Units for tick axis. Can be the unit size in degrees, or the word 'degree', 'arcmin' or 'arcsec' or the shorter 'd','m','s'."
:param "--nolabels", action="store_true", help="Disable the generation of coordinate labels outside the map when using the grid."
:param "--nstep", type=int, default=200, help="The number of steps to use when drawing grid lines. Higher numbers result in smoother curves."
:param "--subticks", type=float, default=0, help="Subtick spacing. Only supported by matplotlib driver."
:param "-b", "--colorbar", default=0, action="count", help="Whether to draw the color bar or not"
:param "--font", type=str, default="arial.ttf", help="The font to use for text."
:param "--font-size", type=int, default=20, help="Font size to use for text."
:param "--font-color", type=str, default="000000", help="Font color to use for text."
:param "-D", "--driver", type=str, default="pil", help="The driver to use for plotting. Can be pil (the default) or mpl. pil cleanly maps input pixels to output pixels, and has better coordiante system support, but doesn't have as pretty grid lines or axis labels."
:param "--mpl-dpi", type=float, default=75, help="The resolution to use for the mpl driver."
:param "--mpl-pad", type=float, default=1.6, help="The padding to use for the mpl driver."
:param "--rgb", action="store_true", help="Enable RGB mode. The input maps must have 3 components, which will be interpreted as red, green and blue channels of a single image instead of 3 separate images as would be the case without this option. The color scheme is overriden in this case."
:param "--rgb-mode", type=str, default="direct", help="The rgb mode to use. Can be direct or direct_colorcap. These only differ in whether colors are preserved when too high or low colors are capped. direct_colorcap preserves colors, at the cost of noise from one noisy component leaking into others during capping."
:param "--reverse-color",  action="store_true", help="Reverse the color scale. For example, a black-to-white scale will become a white-to-black sacle."
:param "-a", "--autocrop", action="store_true", help="Automatically crop the image by removing expanses of uniform color around the edges. This is done jointly for all components in a map, making them directly comparable, but is done independently for each input file."
:param "-A", "--autocrop-each", action="store_true", help="As --autocrop, but done individually for each component in each map."
:param "-L", "--layers", action="store_true", help="Output the individual layers that make up the final plot (such as the map itself, the coordinate grid, the axis labels, any contours and lables) as individual files instead of compositing them into a final image."
:param       "--no-image", action="store_true", help="Skip the main image plotting. Useful for getting a pure contour plot, for example."
:param "-C", "--contours", type=str, default=None, help="Enable contour lines. For example -C 10 to place a contour at every 10 units in the map, -C 5:10 to place it at every 10 units, but starting at 5, and 1,2,4,8 or similar to place contours at manually chosen locations."
:param "--contour-type",  type=str, default="uniform", help="The type of the contour specification. Only used when the contours specification is a list of numbers rather than a string (so not from the command line interface). 'uniform': the list is [interval] or [base, interval]. 'list': the list is an explicit list of the values the contours should be at."
:param "--contour-color", type=str, default="000000", help="The color scheme to use for contour lines. Either a single rrggbb, a val:rrggbb,val:rrggbb,... specification or a color scheme name, such as planck, wmap or gray."
:param "--contour-width", type=int, default=1, help="The width of each contour line, in pixels."
:param "--annotate",      type=str, default=None, help="""Annotate the map with text, lines or circles. Should be a text file with one entry per line, where an entry can be: c[ircle] lat lon dy dx [rad [width [color]] t[ext]   lat lon dy dx text [size [color]] l[ine]   lat lon dy dx lat lon dy dx [width [color]] dy and dx are pixel-unit offsets from the specified lat/lon. Alternatively, from python one can pass in a list of lists containig the same information, e.g. [["circle", 5.10,222.3,0,0,32,3,"black"]]"""
:param "--annotate-maxrad", type=int, default=0, help="Assume that annotations do not extend further than this from their center, in pixels. This is used to prune which annotations to attempt to draw, as they can be a bit slow. The special value 0 disables this."
:param "--stamps", type=str, default=None, help="Plot stamps instead of the whole map. Format is srcfile:size:nmax, where the last two are optional. srcfile is a file with [ra dec] in degrees, size is the size in pixels of each stamp, and nmax is the max number of stamps to produce."
:param "--tile",  type=str, default=None, help="Stack components vertically and horizontally. --tile 5,4 stacks into 5 rows and 4 columns. --tile 5 or --tile 5,-1 stacks into 5 rows and however many columns are needed. --tile -1,5 stacks into 5 columns and as many rows are needed. --tile -1 allocates both rows and columns to make the result as square as possible. The result is treated as a single enmap, so the wcs will only be right for one of the tiles."
:param "--tile-transpose", action="store_true", help="Transpose the ordering of the fields when tacking. Normally row-major stacking is used. This sets column-major order instead."
:param "--tile-dims", type=str, default=None
:param "-S", "--symmetric", action="store_true", help="Treat the non-pixel axes as being asymmetric matrix, and only plot a non-redundant triangle of this matrix."
:param "-z", "--zenith",    action="store_true", help="Plot the zenith angle instead of the declination."
:param "-F", "--fix-wcs",   action="store_true", help="Fix the wcs for maps in cylindrical projections where the reference point was placed too far away from the map center."
:param       "--pos-ra",    action="store_true", help="RA goes from 0 to 360 instead of -180 to 180



Plotting maps
----------------

:py:func:`pixell.enplot.plot` is the main function for plotting maps. It takes a map and a set of options and produces a plot. The options can be used to control the appearance of the plot, such as the color map, the title, and the axis labels.
:py:func:`pixell.enplot.plot_iterator`
:py:func:`pixell.enplot.get_plots`
:py:func:`pixell.enplot.merge_plots`

Show maps
----------------
:py:func:`pixell.enplot.show`
:py:func:`pixell.enplot.pshow`


Plots I/O
----------------
:py:func:`pixell.enplot.write`
:py:func:`pixell.enplot.get_map`