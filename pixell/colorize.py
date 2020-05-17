# Transform from real numbers to RGB colors.
import numpy as np, time
has_fortran = True
try: from . import _colorize
except ImportError: has_fortran = False
try: basestring
except NameError: basestring = str

# Predefined schemes
schemes = {}

class Colorscheme:
	def __init__(self, desc):
		"""Parses a color description string of the form "v1:c1,v2:c2,...,vn,vn"
		into a numpy array of values [v1,v2,..,vn] and a numpy array of colors,
		[[r,g,b,a],[r,g,b,a],[r,g,b,a],...]."""
		try:
			desc = schemes[desc]
		except KeyError:
			pass
		try:
			self.vals, self.cols, self.desc = desc.vals, desc.cols, desc.desc
			return
		except AttributeError:
			pass
		toks = desc.split(",")
		if len(toks) == 1:
			# Constant color mode
			desc = "0:%s,1:%s" % (desc,desc)
			toks = desc.split(",")
		# Construct the output arrays
		vals = np.zeros((len(toks)))
		cols = np.zeros((len(toks),4))
		# And populate them
		for i, tok in enumerate(toks):
			val, code = tok.split(":")
			vals[i] = float(val)
			color = np.array((0,0,0,0xff),dtype=np.uint8)
			m = len(code)//2
			for j in range(m):
				color[j] = int(code[2*j:2*(j+1)],16)
			cols[i,:] = color
		# Sort result
		order = np.argsort(vals)
		self.vals, self.cols = vals[order], cols[order]
		self.desc = desc
	def reverse(self):
		res = Colorscheme(self)
		res.vals = 1-self.vals[::-1]
		res.cols = self.cols[::-1]
		return res

def colorize(arr, desc="planck", mode="scalar", driver="auto"):
	"""Transform a set of values into RGB tuples. Two modes are supported.
	"scalar": arr constains independent scalar values, each of which is
	expanded into an RGB tuple using the color scheme descriptor "desc",
	which can be a Colorscheme or a string that can be converted into a
	Colorscheme. arr[...] -> res[...,{rgba}]. The outputs will be between
	0 and 255 inclusive. This is the default.

	"direct": arr contains RGB or RGBA tuples, which should simply be
	carried over into the output. "desc" has no effect in this case.
	If A is missing, it will be set to 255. arr[{rgb(a)},...] -> res[...,{rgba}].

	In both cases, invalid values are set to fully transparent (A=0).

	There are two drivers. "fortran" is fastest, and uses less memory,
	but requires a helper module to be compiled with f2py. "python"
	can be run directly. The default is "auto", which uses fortran
	if available, and python otherwise."""
	if driver == "auto":
		driver = "fortran" if has_fortran else "python"
	desc = Colorscheme(desc)
	if len(desc.vals) == 0:
		return np.zeros(arr.shape+(4,),dtype=np.uint8)
	elif len(desc.vals) == 1:
		return np.tile(desc.cols[0],arr.shape+(1,)).T
	else:
		if mode == "scalar":
			a = arr.reshape(-1)
			if   driver == "python":  res = colorize_scalar_python(a, desc)
			elif driver == "fortran": res = colorize_scalar_fortran(a, desc)
			else: raise ValueError("Invalid colorize driver '%s' for type '%s'" % (driver, type))
			return res.reshape(arr.shape + (4,))
		elif mode == "direct":
			a = arr.reshape(arr.shape[0],-1)
			if   driver == "python":  res = colorize_direct_python(a, desc)
			elif driver == "fortran": res = colorize_direct_fortran(a, desc)
			else: raise ValueError("Invalid colorize driver '%s' for type '%s'" % (driver, type))
			return res.reshape(arr.shape[1:] + (4,))

schemes["gray"]    = Colorscheme("0:000000,1:ffffff")
schemes["wmap"]    = Colorscheme("0:000080,0.15:0000ff,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")
schemes["planck_old"]= Colorscheme("0:0000ff,0.33:ffedd9,0.83:ff4b00,1:640000")
schemes["planck"]    = Colorscheme("0:0000ff,0.332:00d7ff,0.5:ffedd9,0.664:ffb400,0.828:ff4b00,1:640000")
schemes["pwhite"]    = Colorscheme("0:0000ff,0.332:00d7ff,0.5:ffffff,0.55:ffedd9,0.664:ffb400,0.828:ff4b00,1:640000")

schemes["hotcold"] = Colorscheme("0:0000ff,0.5:000000,1:ff0000")
schemes["hotcold2"] = Colorscheme("0:0000ff,0.5:ffffff,1:ff0000")
schemes["cooltowarm"] = Colorscheme("0.00000:3b4cc0,0.03125:445acc,0.06250:4d68d7,0.09375:5775e1,0.12500:6282ea,0.15625:6c8ef1,0.18750:779af7,0.21875:82a5fb,0.25000:8db0fe,0.28125:98b9ff,0.31250:a3c2ff,0.34375:aec9fd,0.37500:b8d0f9,0.40625:c2d5f4,0.43750:ccd9ee,0.46875:d5dbe6,0.50000:dddddd,0.53125:e5d8d1,0.56250:ecd3c5,0.59375:f1ccb9,0.62500:f5c4ad,0.65625:f7bba0,0.68750:f7b194,0.71875:f7a687,0.75000:f49a7b,0.78125:f18d6f,0.81250:ec7f63,0.84375:e57058,0.87500:de604d,0.90625:d55042,0.93750:cb3e38,0.96875:c0282f,1.00000:b40426")
schemes["cubehelix"] = Colorscheme("0.000:000000,0.004:020002,0.008:040103,0.012:060105,0.016:080207,0.020:0a0209,0.024:0b020b,0.027:0d030d,0.031:0e030f,0.035:100412,0.039:110514,0.043:120516,0.047:140618,0.051:15071b,0.055:16071d,0.059:16081f,0.063:170922,0.067:180a24,0.071:190b26,0.075:190c29,0.078:1a0d2b,0.082:1a0e2d,0.086:1a0f2f,0.090:1a1032,0.094:1a1234,0.098:1a1336,0.102:1a1438,0.106:1a163a,0.110:1a173c,0.114:1a193e,0.118:191a40,0.122:191c42,0.125:181d44,0.129:181f46,0.133:172147,0.137:172249,0.141:16244b,0.145:15264c,0.149:14284d,0.153:142a4f,0.157:132b50,0.161:122d51,0.165:112f52,0.169:103153,0.173:0f3354,0.176:0e3554,0.180:0d3755,0.184:0c3955,0.188:0c3c56,0.192:0b3e56,0.196:0a4056,0.200:094256,0.204:084457,0.208:074656,0.212:074856,0.216:064a56,0.220:054c56,0.224:054e55,0.227:045055,0.231:045254,0.235:035453,0.239:035652,0.243:035852,0.247:035a51,0.251:025c50,0.255:025e4e,0.259:03604d,0.263:03624c,0.267:03644b,0.271:036549,0.275:046748,0.278:046947,0.282:056a45,0.286:066c44,0.290:076d42,0.294:086f40,0.298:09703f,0.302:0a723d,0.306:0b733c,0.310:0d743a,0.314:0e7538,0.318:107737,0.322:127835,0.325:147933,0.329:167a32,0.333:187b30,0.337:1a7b2f,0.341:1d7c2d,0.345:1f7d2c,0.349:227e2a,0.353:247e29,0.357:277f27,0.361:2a7f26,0.365:2d8025,0.369:308024,0.373:338023,0.376:368122,0.380:398121,0.384:3d8120,0.388:40811f,0.392:44811e,0.396:47811e,0.400:4b811d,0.404:4f811d,0.408:53811d,0.412:56801c,0.416:5a801c,0.420:5e801c,0.424:62801d,0.427:667f1d,0.431:6a7f1d,0.435:6e7e1e,0.439:727e1e,0.443:767d1f,0.447:7a7d20,0.451:7e7c21,0.455:827c22,0.459:867b23,0.463:8a7b25,0.467:8e7a26,0.471:927928,0.475:96792a,0.478:9a782b,0.482:9e782d,0.486:a27730,0.490:a57632,0.494:a97634,0.498:ad7536,0.502:b07539,0.506:b4743c,0.510:b7743e,0.514:ba7341,0.518:bd7344,0.522:c17247,0.525:c4724a,0.529:c7714e,0.533:c97151,0.537:cc7154,0.541:cf7058,0.545:d1705b,0.549:d4705f,0.553:d67062,0.557:d87066,0.561:da706a,0.565:dc706d,0.569:de7071,0.573:e07075,0.576:e27079,0.580:e3707d,0.584:e47181,0.588:e67185,0.592:e77189,0.596:e8728d,0.600:e97291,0.604:ea7395,0.608:ea7399,0.612:eb749c,0.616:eb75a0,0.620:eb76a4,0.624:ec77a8,0.627:ec78ac,0.631:ec79b0,0.635:ec7ab3,0.639:eb7bb7,0.643:eb7cbb,0.647:eb7dbe,0.651:ea7fc2,0.655:e980c5,0.659:e981c8,0.663:e883cc,0.667:e784cf,0.671:e686d2,0.675:e588d5,0.678:e489d8,0.682:e38bdb,0.686:e28ddd,0.690:e08fe0,0.694:df91e3,0.698:de93e5,0.702:dd94e7,0.706:db96ea,0.710:da98ec,0.714:d89aee,0.718:d79df0,0.722:d59ff1,0.725:d4a1f3,0.729:d2a3f5,0.733:d1a5f6,0.737:cfa7f8,0.741:cea9f9,0.745:cdacfa,0.749:cbaefb,0.753:cab0fc,0.757:c8b2fd,0.761:c7b5fe,0.765:c6b7fe,0.769:c5b9ff,0.773:c4bbff,0.776:c2bdff,0.780:c1bfff,0.784:c0c2ff,0.788:bfc4ff,0.792:bfc6ff,0.796:bec8ff,0.800:bdcaff,0.804:bcccff,0.808:bcceff,0.812:bbd0ff,0.816:bbd2ff,0.820:bbd4fe,0.824:bbd6fe,0.827:bad8fd,0.831:bad9fd,0.835:bbdbfc,0.839:bbddfb,0.843:bbdffb,0.847:bbe0fa,0.851:bce2f9,0.855:bce3f9,0.859:bde5f8,0.863:bee6f7,0.867:bfe8f7,0.871:c0e9f6,0.875:c1eaf5,0.878:c2ecf5,0.882:c3edf4,0.886:c4eef3,0.890:c6eff3,0.894:c7f0f2,0.898:c9f1f2,0.902:caf2f1,0.906:ccf3f1,0.910:cef4f1,0.914:cff5f0,0.918:d1f5f0,0.922:d3f6f0,0.925:d5f7f0,0.929:d7f7f0,0.933:d9f8f0,0.937:dbf9f0,0.941:def9f1,0.945:e0faf1,0.949:e2faf1,0.953:e4fbf2,0.957:e7fbf2,0.961:e9fcf3,0.965:ebfcf4,0.969:edfcf5,0.973:f0fdf6,0.976:f2fdf7,0.980:f4fdf8,0.984:f6fef9,0.988:f9fefa,0.992:fbfefc,0.996:fdfffd,1.000:ffffff")
schemes["nozero"]    = Colorscheme("0:000080,0.15:0000ff,0.499998:55ffaa,0.499999:55ffaa00,0.500001:55ffaa00,0.500002:55ffaa,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")
schemes["iron"]      = Colorscheme("0.00000:000000,0.00840:000024,0.01681:000033,0.02521:000042,0.03361:000051,0.04202:02005a,0.05042:040063,0.05882:07006a,0.06723:0b0073,0.07563:0e0077,0.08403:14007b,0.09244:1b0080,0.10084:210085,0.10924:290089,0.11765:30008c,0.12605:37008f,0.13445:3d0092,0.14286:420095,0.15126:480096,0.15966:4e0097,0.16807:540098,0.17647:5b0099,0.18487:61009b,0.19328:68009b,0.20168:6e009c,0.21008:73009d,0.21849:7a009d,0.22689:80009d,0.23529:86009d,0.24370:8b009d,0.25210:92009c,0.26050:98009b,0.26891:9d009b,0.27731:a2009b,0.28571:a7009a,0.29412:ab0099,0.30252:af0198,0.31092:b20197,0.31933:b60295,0.32773:b90495,0.33613:bc0593,0.34454:bf0692,0.35294:c10890,0.36134:c30b8e,0.36975:c60d8b,0.37815:c91187,0.38655:cb1484,0.39496:ce177f,0.40336:d01a79,0.41176:d21d74,0.42017:d4216f,0.42857:d62567,0.43697:d92961,0.44538:db2e59,0.45378:dd314e,0.46218:df3542,0.47059:e03836,0.47899:e23c2a,0.48739:e4401e,0.49580:e54419,0.50420:e74814,0.51261:e84c10,0.52101:ea4e0c,0.52941:eb520a,0.53782:ec5608,0.54622:ed5a07,0.55462:ee5d05,0.56303:ef6004,0.57143:f06403,0.57983:f16703,0.58824:f16a02,0.59664:f26d01,0.60504:f37101,0.61345:f47400,0.62185:f47800,0.63025:f57d00,0.63866:f68100,0.64706:f78500,0.65546:f88800,0.66387:f88b00,0.67227:f98e00,0.68067:f99100,0.68908:fa9500,0.69748:fb9a00,0.70588:fc9f00,0.71429:fda300,0.72269:fda800,0.73109:fdac00,0.73950:feb000,0.74790:feb300,0.75630:feb800,0.76471:febb00,0.77311:febf00,0.78151:fec300,0.78992:fec700,0.79832:feca01,0.80672:fecd02,0.81513:fed005,0.82353:fed409,0.83193:fed80c,0.84034:ffdb0f,0.84874:ffdd17,0.85714:ffe020,0.86555:ffe327,0.87395:ffe532,0.88235:ffe83f,0.89076:ffeb4b,0.89916:ffee58,0.90756:ffef66,0.91597:fff174,0.92437:fff286,0.93277:fff495,0.94118:fff5a4,0.94958:fff7b3,0.95798:fff8c0,0.96639:fff9cb,0.97479:fffbd8,0.98319:fffde4,0.99160:fffeef,1.00000:fffff9")

def colorize_scalar_fortran(a, desc):
	res = np.empty((len(a),4),dtype=np.uint16)
	_colorize.remap(a, res.T, desc.vals, desc.cols.astype(np.int16).T)
	return res.astype(np.uint8)

def colorize_scalar_python(a, desc):
	res = np.empty((len(a),4),dtype=np.uint8)
	ok  = np.where(~np.isnan(a))
	bad = np.where( np.isnan(a))
	# Bad values are transparent
	res[bad,:] = np.array((0,0,0,0),np.uint8)
	# Good ones get proper treatment
	i = np.searchsorted(desc.vals, a[ok])
	# We always want a point to our left and right
	i = np.minimum(np.maximum(i,1),len(desc.vals)-1)
	# Fractional distance to next point
	x = (a[ok] - desc.vals[i-1])/(desc.vals[i]-desc.vals[i-1])
	# Cap this value too
	x = np.minimum(np.maximum(x,0),1)
	# The result is the linear combination of the two
	# end points
	col = np.round(desc.cols[i-1]*(1-x)[:,None] + desc.cols[i]*x[:,None])
	res[ok] = np.array(np.minimum(np.maximum(col,0),0xff),dtype=np.uint8)
	return res

def colorize_direct_python(a, desc):
	nc  = a.shape[0]
	res = np.empty((a.shape[1],4),dtype=np.uint16)
	good = np.isfinite(a[0])
	res[~good] = 0
	res[good,:nc] = np.maximum(0,np.minimum(255,a[:nc,good]*256))
	res[good,nc:] = 255
	return res

def colorize_direct_fortran(a, desc):
	res = np.empty((a.shape[1],4),dtype=np.uint16)
	_colorize.direct(a.T, res.T)
	return res.astype(np.uint8)

def to_mpl_colormap(name, data=None):
	import matplotlib.colors
	if data is None: data = schemes[name]
	return matplotlib.colors.LinearSegmentedColormap.from_list(name,
			[(val,"#%02x%02x%02x%02x"%tuple(col)) for val,col in zip(data.vals, data.cols.astype(int))])

def mpl_register(names=None):
	import matplotlib.cm
	if names is None: names = schemes.keys()
	if isinstance(names, basestring): names = [names]
	for name in names:
		cmap = to_mpl_colormap(name, schemes[name])
		matplotlib.cm.register_cmap(name, cmap)

def mpl_setdefault(name):
	import matplotlib.pyplot
	mpl_register(name)
	matplotlib.pyplot.rcParams['image.cmap'] = name
