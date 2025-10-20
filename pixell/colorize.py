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
	arr  = np.asarray(arr)
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
		elif mode == "direct_colorcap":
			a = arr.reshape(arr.shape[0],-1)
			if   driver == "python":  raise NotImplementedError("direct_colorcap not implemented in python")
			elif driver == "fortran": res = colorize_direct_colorcap_fortran(a, desc)
			else: raise ValueError("Invalid colorize driver '%s' for type '%s'" % (driver, type))
			return res.reshape(arr.shape[1:] + (4,))

schemes["gray"]    = Colorscheme("0:000000,1:ffffff")
schemes["wmap"]    = Colorscheme("0:000080,0.15:0000ff,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")
schemes["planck_old"]= Colorscheme("0:0000ff,0.33:ffedd9,0.83:ff4b00,1:640000")
schemes["planck"]    = Colorscheme("0:0000ff,0.332:00d7ff,0.5:ffedd9,0.664:ffb400,0.828:ff4b00,1:640000")
schemes["pcont"]    = Colorscheme("0:0000ff,0.332:00d7ff,0.5:00cc00,0.664:ffb400,0.828:ff4b00,1:640000")
schemes["pwhite"]    = Colorscheme("0:0000ff,0.332:00d7ff,0.5:ffffff,0.55:ffedd9,0.664:ffb400,0.828:ff4b00,1:640000")

schemes["hotcold"] = Colorscheme("0:0000ff,0.5:000000,1:ff0000")
schemes["hotcold2"] = Colorscheme("0:0000ff,0.5:ffffff,1:ff0000")
schemes["cooltowarm"] = Colorscheme("0.00000:3b4cc0,0.03125:445acc,0.06250:4d68d7,0.09375:5775e1,0.12500:6282ea,0.15625:6c8ef1,0.18750:779af7,0.21875:82a5fb,0.25000:8db0fe,0.28125:98b9ff,0.31250:a3c2ff,0.34375:aec9fd,0.37500:b8d0f9,0.40625:c2d5f4,0.43750:ccd9ee,0.46875:d5dbe6,0.50000:dddddd,0.53125:e5d8d1,0.56250:ecd3c5,0.59375:f1ccb9,0.62500:f5c4ad,0.65625:f7bba0,0.68750:f7b194,0.71875:f7a687,0.75000:f49a7b,0.78125:f18d6f,0.81250:ec7f63,0.84375:e57058,0.87500:de604d,0.90625:d55042,0.93750:cb3e38,0.96875:c0282f,1.00000:b40426")
schemes["cubehelix"] = Colorscheme("0.000:000000,0.004:020002,0.008:040103,0.012:060105,0.016:080207,0.020:0a0209,0.024:0b020b,0.027:0d030d,0.031:0e030f,0.035:100412,0.039:110514,0.043:120516,0.047:140618,0.051:15071b,0.055:16071d,0.059:16081f,0.063:170922,0.067:180a24,0.071:190b26,0.075:190c29,0.078:1a0d2b,0.082:1a0e2d,0.086:1a0f2f,0.090:1a1032,0.094:1a1234,0.098:1a1336,0.102:1a1438,0.106:1a163a,0.110:1a173c,0.114:1a193e,0.118:191a40,0.122:191c42,0.125:181d44,0.129:181f46,0.133:172147,0.137:172249,0.141:16244b,0.145:15264c,0.149:14284d,0.153:142a4f,0.157:132b50,0.161:122d51,0.165:112f52,0.169:103153,0.173:0f3354,0.176:0e3554,0.180:0d3755,0.184:0c3955,0.188:0c3c56,0.192:0b3e56,0.196:0a4056,0.200:094256,0.204:084457,0.208:074656,0.212:074856,0.216:064a56,0.220:054c56,0.224:054e55,0.227:045055,0.231:045254,0.235:035453,0.239:035652,0.243:035852,0.247:035a51,0.251:025c50,0.255:025e4e,0.259:03604d,0.263:03624c,0.267:03644b,0.271:036549,0.275:046748,0.278:046947,0.282:056a45,0.286:066c44,0.290:076d42,0.294:086f40,0.298:09703f,0.302:0a723d,0.306:0b733c,0.310:0d743a,0.314:0e7538,0.318:107737,0.322:127835,0.325:147933,0.329:167a32,0.333:187b30,0.337:1a7b2f,0.341:1d7c2d,0.345:1f7d2c,0.349:227e2a,0.353:247e29,0.357:277f27,0.361:2a7f26,0.365:2d8025,0.369:308024,0.373:338023,0.376:368122,0.380:398121,0.384:3d8120,0.388:40811f,0.392:44811e,0.396:47811e,0.400:4b811d,0.404:4f811d,0.408:53811d,0.412:56801c,0.416:5a801c,0.420:5e801c,0.424:62801d,0.427:667f1d,0.431:6a7f1d,0.435:6e7e1e,0.439:727e1e,0.443:767d1f,0.447:7a7d20,0.451:7e7c21,0.455:827c22,0.459:867b23,0.463:8a7b25,0.467:8e7a26,0.471:927928,0.475:96792a,0.478:9a782b,0.482:9e782d,0.486:a27730,0.490:a57632,0.494:a97634,0.498:ad7536,0.502:b07539,0.506:b4743c,0.510:b7743e,0.514:ba7341,0.518:bd7344,0.522:c17247,0.525:c4724a,0.529:c7714e,0.533:c97151,0.537:cc7154,0.541:cf7058,0.545:d1705b,0.549:d4705f,0.553:d67062,0.557:d87066,0.561:da706a,0.565:dc706d,0.569:de7071,0.573:e07075,0.576:e27079,0.580:e3707d,0.584:e47181,0.588:e67185,0.592:e77189,0.596:e8728d,0.600:e97291,0.604:ea7395,0.608:ea7399,0.612:eb749c,0.616:eb75a0,0.620:eb76a4,0.624:ec77a8,0.627:ec78ac,0.631:ec79b0,0.635:ec7ab3,0.639:eb7bb7,0.643:eb7cbb,0.647:eb7dbe,0.651:ea7fc2,0.655:e980c5,0.659:e981c8,0.663:e883cc,0.667:e784cf,0.671:e686d2,0.675:e588d5,0.678:e489d8,0.682:e38bdb,0.686:e28ddd,0.690:e08fe0,0.694:df91e3,0.698:de93e5,0.702:dd94e7,0.706:db96ea,0.710:da98ec,0.714:d89aee,0.718:d79df0,0.722:d59ff1,0.725:d4a1f3,0.729:d2a3f5,0.733:d1a5f6,0.737:cfa7f8,0.741:cea9f9,0.745:cdacfa,0.749:cbaefb,0.753:cab0fc,0.757:c8b2fd,0.761:c7b5fe,0.765:c6b7fe,0.769:c5b9ff,0.773:c4bbff,0.776:c2bdff,0.780:c1bfff,0.784:c0c2ff,0.788:bfc4ff,0.792:bfc6ff,0.796:bec8ff,0.800:bdcaff,0.804:bcccff,0.808:bcceff,0.812:bbd0ff,0.816:bbd2ff,0.820:bbd4fe,0.824:bbd6fe,0.827:bad8fd,0.831:bad9fd,0.835:bbdbfc,0.839:bbddfb,0.843:bbdffb,0.847:bbe0fa,0.851:bce2f9,0.855:bce3f9,0.859:bde5f8,0.863:bee6f7,0.867:bfe8f7,0.871:c0e9f6,0.875:c1eaf5,0.878:c2ecf5,0.882:c3edf4,0.886:c4eef3,0.890:c6eff3,0.894:c7f0f2,0.898:c9f1f2,0.902:caf2f1,0.906:ccf3f1,0.910:cef4f1,0.914:cff5f0,0.918:d1f5f0,0.922:d3f6f0,0.925:d5f7f0,0.929:d7f7f0,0.933:d9f8f0,0.937:dbf9f0,0.941:def9f1,0.945:e0faf1,0.949:e2faf1,0.953:e4fbf2,0.957:e7fbf2,0.961:e9fcf3,0.965:ebfcf4,0.969:edfcf5,0.973:f0fdf6,0.976:f2fdf7,0.980:f4fdf8,0.984:f6fef9,0.988:f9fefa,0.992:fbfefc,0.996:fdfffd,1.000:ffffff")
schemes["nozero"]    = Colorscheme("0:000080,0.15:0000ff,0.499998:55ffaa,0.499999:55ffaa00,0.500001:55ffaa00,0.500002:55ffaa,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")
schemes["iron"]      = Colorscheme("0.00000:000000,0.00840:000024,0.01681:000033,0.02521:000042,0.03361:000051,0.04202:02005a,0.05042:040063,0.05882:07006a,0.06723:0b0073,0.07563:0e0077,0.08403:14007b,0.09244:1b0080,0.10084:210085,0.10924:290089,0.11765:30008c,0.12605:37008f,0.13445:3d0092,0.14286:420095,0.15126:480096,0.15966:4e0097,0.16807:540098,0.17647:5b0099,0.18487:61009b,0.19328:68009b,0.20168:6e009c,0.21008:73009d,0.21849:7a009d,0.22689:80009d,0.23529:86009d,0.24370:8b009d,0.25210:92009c,0.26050:98009b,0.26891:9d009b,0.27731:a2009b,0.28571:a7009a,0.29412:ab0099,0.30252:af0198,0.31092:b20197,0.31933:b60295,0.32773:b90495,0.33613:bc0593,0.34454:bf0692,0.35294:c10890,0.36134:c30b8e,0.36975:c60d8b,0.37815:c91187,0.38655:cb1484,0.39496:ce177f,0.40336:d01a79,0.41176:d21d74,0.42017:d4216f,0.42857:d62567,0.43697:d92961,0.44538:db2e59,0.45378:dd314e,0.46218:df3542,0.47059:e03836,0.47899:e23c2a,0.48739:e4401e,0.49580:e54419,0.50420:e74814,0.51261:e84c10,0.52101:ea4e0c,0.52941:eb520a,0.53782:ec5608,0.54622:ed5a07,0.55462:ee5d05,0.56303:ef6004,0.57143:f06403,0.57983:f16703,0.58824:f16a02,0.59664:f26d01,0.60504:f37101,0.61345:f47400,0.62185:f47800,0.63025:f57d00,0.63866:f68100,0.64706:f78500,0.65546:f88800,0.66387:f88b00,0.67227:f98e00,0.68067:f99100,0.68908:fa9500,0.69748:fb9a00,0.70588:fc9f00,0.71429:fda300,0.72269:fda800,0.73109:fdac00,0.73950:feb000,0.74790:feb300,0.75630:feb800,0.76471:febb00,0.77311:febf00,0.78151:fec300,0.78992:fec700,0.79832:feca01,0.80672:fecd02,0.81513:fed005,0.82353:fed409,0.83193:fed80c,0.84034:ffdb0f,0.84874:ffdd17,0.85714:ffe020,0.86555:ffe327,0.87395:ffe532,0.88235:ffe83f,0.89076:ffeb4b,0.89916:ffee58,0.90756:ffef66,0.91597:fff174,0.92437:fff286,0.93277:fff495,0.94118:fff5a4,0.94958:fff7b3,0.95798:fff8c0,0.96639:fff9cb,0.97479:fffbd8,0.98319:fffde4,0.99160:fffeef,1.00000:fffff9")
schemes["comap"] = Colorscheme("0.00000:723959,0.00508:713a5c,0.01015:703a5d,0.01523:6f3b5f,0.02030:6f3b60,0.02538:6d3d64,0.03046:6c3d65,0.03553:6c3e67,0.04061:6a3f6b,0.04569:69406c,0.05076:68416e,0.05584:674372,0.06091:664474,0.06599:654576,0.07107:644677,0.07614:63487b,0.08122:62497d,0.08629:614b7f,0.09137:5f4d83,0.09645:5e4f85,0.10152:5d5087,0.10660:5c5289,0.11168:5b558d,0.11675:5a568f,0.12183:595891,0.12690:575b94,0.13198:575d96,0.13706:565f98,0.14213:54629c,0.14721:54649e,0.15228:53669f,0.15736:5267a1,0.16244:516ba4,0.16751:516da6,0.17259:506fa8,0.17766:4f73ab,0.18274:4f75ac,0.18782:4f76ae,0.19289:4e7ab0,0.19797:4e7cb2,0.20305:4e7eb3,0.20812:4e80b4,0.21320:4f84b7,0.21827:4f86b8,0.22335:4f88b9,0.22843:508bbb,0.23350:518dbc,0.23858:528fbd,0.24365:5291be,0.24873:5495c0,0.25381:5597c1,0.25888:5699c2,0.26396:589cc4,0.26904:599ec4,0.27411:5aa0c5,0.27919:5da4c7,0.28426:5fa6c7,0.28934:60a8c8,0.29442:62a9c9,0.29949:65adca,0.30457:67afca,0.30964:68b1cb,0.31472:6cb4cc,0.31980:6eb6cc,0.32487:70b8cd,0.32995:72b9cd,0.33503:76bdce,0.34010:78bece,0.34518:7bc0ce,0.35025:7fc3cf,0.35533:81c5cf,0.36041:84c6cf,0.36548:88c9cf,0.37056:8bcbcf,0.37563:8dcccf,0.38071:8fcecf,0.38579:94d0ce,0.39086:96d1ce,0.39594:99d3ce,0.40102:9dd5cd,0.40609:a0d6cc,0.41117:a2d7cc,0.41624:a7d9cb,0.42132:a9daca,0.42640:abdbc9,0.43147:addcc8,0.43655:b1ddc7,0.44162:b4dec6,0.44670:b6dec5,0.45178:b9dfc3,0.45685:bbe0c2,0.46193:bde0c0,0.46701:bfe1bf,0.47208:c2e1bc,0.47716:c4e1bb,0.48223:c5e1b9,0.48731:c8e1b6,0.49239:cae1b5,0.49746:cbe1b3,0.50254:cde1af,0.50761:cee0ad,0.51269:cfe0ab,0.51777:d0e0a9,0.52284:d2dfa5,0.52792:d3dea3,0.53299:d3dda1,0.53807:d4dc9c,0.54315:d5db9a,0.54822:d5da98,0.55330:d5d995,0.55838:d6d790,0.56345:d6d58e,0.56853:d6d48b,0.57360:d5d186,0.57868:d5d083,0.58376:d5ce81,0.58883:d4cb7b,0.59391:d4c978,0.59898:d3c876,0.60406:d2c673,0.60914:d1c26e,0.61421:d0c06b,0.61929:cfbe68,0.62437:ceba63,0.62944:cdb861,0.63452:ccb55e,0.63959:cab159,0.64467:c9af57,0.64975:c8ac54,0.65482:c6aa52,0.65990:c4a54e,0.66497:c3a34b,0.67005:c2a149,0.67513:bf9c46,0.68020:be9a44,0.68528:bd9842,0.69036:bc9540,0.69543:b9913d,0.70051:b88e3b,0.70558:b78c3a,0.71066:b58837,0.71574:b48636,0.72081:b28335,0.72589:b07f33,0.73096:af7d32,0.73604:ae7b31,0.74112:ad7930,0.74619:aa752f,0.75127:a9732e,0.75635:a8712e,0.76142:a66d2d,0.76650:a56b2d,0.77157:a4692c,0.77665:a3672c,0.78173:a0642c,0.78680:9f622b,0.79188:9e602b,0.79695:9c5d2b,0.80203:9b5b2c,0.80711:9a5a2c,0.81218:98572c,0.81726:97552c,0.82234:96542d,0.82741:95522d,0.83249:94502e,0.83756:934e2e,0.84264:924d2f,0.84772:904a30,0.85279:8f4930,0.85787:8e4831,0.86294:8c4632,0.86802:8c4533,0.87310:8b4433,0.87817:8a4334,0.88325:884136,0.88832:874037,0.89340:874037,0.89848:853e39,0.90355:843d3a,0.90863:843d3b,0.91371:833c3c,0.91878:813b3e,0.92386:813b3f,0.92893:803a40,0.93401:7e3942,0.93909:7e3943,0.94416:7d3945,0.94924:7c3847,0.95431:7b3848,0.95939:7a3849,0.96447:79384b,0.96954:78384d,0.97462:77384f,0.97970:773850,0.98477:753853,0.98985:753954,0.99492:743956,1.00000:733957")
schemes["reddish"] = Colorscheme("0:000000,0.5:b60000,0.7:ff6500,0.75:ff7f00,1:ffffff")
schemes["viridis"] = Colorscheme("0.000:440154,0.004:440255,0.008:440357,0.012:450558,0.016:45065a,0.020:45085b,0.024:46095c,0.027:460b5e,0.031:460c5f,0.035:460e61,0.039:470f62,0.043:471163,0.047:471265,0.051:471466,0.055:471567,0.059:471669,0.063:47186a,0.067:48196b,0.071:481a6c,0.075:481c6e,0.078:481d6f,0.082:481e70,0.086:482071,0.090:482172,0.094:482273,0.098:482374,0.102:472575,0.106:472676,0.110:472777,0.114:472878,0.118:472a79,0.122:472b7a,0.125:472c7b,0.129:462d7c,0.133:462f7c,0.137:46307d,0.141:46317e,0.145:45327f,0.149:45347f,0.153:453580,0.157:453681,0.161:443781,0.165:443982,0.169:433a83,0.173:433b83,0.176:433c84,0.180:423d84,0.184:423e85,0.188:424085,0.192:414186,0.196:414286,0.200:404387,0.204:404487,0.208:3f4587,0.212:3f4788,0.216:3e4888,0.220:3e4989,0.224:3d4a89,0.227:3d4b89,0.231:3d4c89,0.235:3c4d8a,0.239:3c4e8a,0.243:3b508a,0.247:3b518a,0.251:3a528b,0.255:3a538b,0.259:39548b,0.263:39558b,0.267:38568b,0.271:38578c,0.275:37588c,0.278:37598c,0.282:365a8c,0.286:365b8c,0.290:355c8c,0.294:355d8c,0.298:345e8d,0.302:345f8d,0.306:33608d,0.310:33618d,0.314:32628d,0.318:32638d,0.322:31648d,0.325:31658d,0.329:31668d,0.333:30678d,0.337:30688d,0.341:2f698d,0.345:2f6a8d,0.349:2e6b8e,0.353:2e6c8e,0.357:2e6d8e,0.361:2d6e8e,0.365:2d6f8e,0.369:2c708e,0.373:2c718e,0.376:2c728e,0.380:2b738e,0.384:2b748e,0.388:2a758e,0.392:2a768e,0.396:2a778e,0.400:29788e,0.404:29798e,0.408:287a8e,0.412:287a8e,0.416:287b8e,0.420:277c8e,0.424:277d8e,0.427:277e8e,0.431:267f8e,0.435:26808e,0.439:26818e,0.443:25828e,0.447:25838d,0.451:24848d,0.455:24858d,0.459:24868d,0.463:23878d,0.467:23888d,0.471:23898d,0.475:22898d,0.478:228a8d,0.482:228b8d,0.486:218c8d,0.490:218d8c,0.494:218e8c,0.498:208f8c,0.502:20908c,0.506:20918c,0.510:1f928c,0.514:1f938b,0.518:1f948b,0.522:1f958b,0.525:1f968b,0.529:1e978a,0.533:1e988a,0.537:1e998a,0.541:1e998a,0.545:1e9a89,0.549:1e9b89,0.553:1e9c89,0.557:1e9d88,0.561:1e9e88,0.565:1e9f88,0.569:1ea087,0.573:1fa187,0.576:1fa286,0.580:1fa386,0.584:20a485,0.588:20a585,0.592:21a685,0.596:21a784,0.600:22a784,0.604:23a883,0.608:23a982,0.612:24aa82,0.616:25ab81,0.620:26ac81,0.624:27ad80,0.627:28ae7f,0.631:29af7f,0.635:2ab07e,0.639:2bb17d,0.643:2cb17d,0.647:2eb27c,0.651:2fb37b,0.655:30b47a,0.659:32b57a,0.663:33b679,0.667:35b778,0.671:36b877,0.675:38b976,0.678:39b976,0.682:3bba75,0.686:3dbb74,0.690:3ebc73,0.694:40bd72,0.698:42be71,0.702:44be70,0.706:45bf6f,0.710:47c06e,0.714:49c16d,0.718:4bc26c,0.722:4dc26b,0.725:4fc369,0.729:51c468,0.733:53c567,0.737:55c666,0.741:57c665,0.745:59c764,0.749:5bc862,0.753:5ec961,0.757:60c960,0.761:62ca5f,0.765:64cb5d,0.769:67cc5c,0.773:69cc5b,0.776:6bcd59,0.780:6dce58,0.784:70ce56,0.788:72cf55,0.792:74d054,0.796:77d052,0.800:79d151,0.804:7cd24f,0.808:7ed24e,0.812:81d34c,0.816:83d34b,0.820:86d449,0.824:88d547,0.827:8bd546,0.831:8dd644,0.835:90d643,0.839:92d741,0.843:95d73f,0.847:97d83e,0.851:9ad83c,0.855:9dd93a,0.859:9fd938,0.863:a2da37,0.867:a5da35,0.871:a7db33,0.875:aadb32,0.878:addc30,0.882:afdc2e,0.886:b2dd2c,0.890:b5dd2b,0.894:b7dd29,0.898:bade27,0.902:bdde26,0.906:bfdf24,0.910:c2df22,0.914:c5df21,0.918:c7e01f,0.922:cae01e,0.925:cde01d,0.929:cfe11c,0.933:d2e11b,0.937:d4e11a,0.941:d7e219,0.945:dae218,0.949:dce218,0.953:dfe318,0.957:e1e318,0.961:e4e318,0.965:e7e419,0.969:e9e419,0.973:ece41a,0.976:eee51b,0.980:f1e51c,0.984:f3e51e,0.988:f6e61f,0.992:f8e621,0.996:fae622,1.000:fde724")
schemes["plasma"] = Colorscheme("0.000:0c0786,0.004:100787,0.008:130689,0.012:15068a,0.016:18068b,0.020:1b068c,0.024:1d068d,0.027:1f058e,0.031:21058f,0.035:230590,0.039:250591,0.043:270592,0.047:290593,0.051:2b0594,0.055:2d0494,0.059:2f0495,0.063:310496,0.067:330497,0.071:340498,0.075:360498,0.078:380499,0.082:3a049a,0.086:3b039a,0.090:3d039b,0.094:3f039c,0.098:40039c,0.102:42039d,0.106:44039e,0.110:45039e,0.114:47029f,0.118:49029f,0.122:4a02a0,0.125:4c02a1,0.129:4e02a1,0.133:4f02a2,0.137:5101a2,0.141:5201a3,0.145:5401a3,0.149:5601a3,0.153:5701a4,0.157:5901a4,0.161:5a00a5,0.165:5c00a5,0.169:5e00a5,0.173:5f00a6,0.176:6100a6,0.180:6200a6,0.184:6400a7,0.188:6500a7,0.192:6700a7,0.196:6800a7,0.200:6a00a7,0.204:6c00a8,0.208:6d00a8,0.212:6f00a8,0.216:7000a8,0.220:7200a8,0.224:7300a8,0.227:7500a8,0.231:7601a8,0.235:7801a8,0.239:7901a8,0.243:7b02a8,0.247:7c02a7,0.251:7e03a7,0.255:7f03a7,0.259:8104a7,0.263:8204a7,0.267:8405a6,0.271:8506a6,0.275:8607a6,0.278:8807a5,0.282:8908a5,0.286:8b09a4,0.290:8c0aa4,0.294:8e0ca4,0.298:8f0da3,0.302:900ea3,0.306:920fa2,0.310:9310a1,0.314:9511a1,0.318:9612a0,0.322:9713a0,0.325:99149f,0.329:9a159e,0.333:9b179e,0.337:9d189d,0.341:9e199c,0.345:9f1a9b,0.349:a01b9b,0.353:a21c9a,0.357:a31d99,0.361:a41e98,0.365:a51f97,0.369:a72197,0.373:a82296,0.376:a92395,0.380:aa2494,0.384:ac2593,0.388:ad2692,0.392:ae2791,0.396:af2890,0.400:b02a8f,0.404:b12b8f,0.408:b22c8e,0.412:b42d8d,0.416:b52e8c,0.420:b62f8b,0.424:b7308a,0.427:b83289,0.431:b93388,0.435:ba3487,0.439:bb3586,0.443:bc3685,0.447:bd3784,0.451:be3883,0.455:bf3982,0.459:c03b81,0.463:c13c80,0.467:c23d80,0.471:c33e7f,0.475:c43f7e,0.478:c5407d,0.482:c6417c,0.486:c7427b,0.490:c8447a,0.494:c94579,0.498:ca4678,0.502:cb4777,0.506:cc4876,0.510:cd4975,0.514:ce4a75,0.518:cf4b74,0.522:d04d73,0.525:d14e72,0.529:d14f71,0.533:d25070,0.537:d3516f,0.541:d4526e,0.545:d5536d,0.549:d6556d,0.553:d7566c,0.557:d7576b,0.561:d8586a,0.565:d95969,0.569:da5a68,0.573:db5b67,0.576:dc5d66,0.580:dc5e66,0.584:dd5f65,0.588:de6064,0.592:df6163,0.596:df6262,0.600:e06461,0.604:e16560,0.608:e26660,0.612:e3675f,0.616:e3685e,0.620:e46a5d,0.624:e56b5c,0.627:e56c5b,0.631:e66d5a,0.635:e76e5a,0.639:e87059,0.643:e87158,0.647:e97257,0.651:ea7356,0.655:ea7455,0.659:eb7654,0.663:ec7754,0.667:ec7853,0.671:ed7952,0.675:ed7b51,0.678:ee7c50,0.682:ef7d4f,0.686:ef7e4e,0.690:f0804d,0.694:f0814d,0.698:f1824c,0.702:f2844b,0.706:f2854a,0.710:f38649,0.714:f38748,0.718:f48947,0.722:f48a47,0.725:f58b46,0.729:f58d45,0.733:f68e44,0.737:f68f43,0.741:f69142,0.745:f79241,0.749:f79341,0.753:f89540,0.757:f8963f,0.761:f8983e,0.765:f9993d,0.769:f99a3c,0.773:fa9c3b,0.776:fa9d3a,0.780:fa9f3a,0.784:faa039,0.788:fba238,0.792:fba337,0.796:fba436,0.800:fca635,0.804:fca735,0.808:fca934,0.812:fcaa33,0.816:fcac32,0.820:fcad31,0.824:fdaf31,0.827:fdb030,0.831:fdb22f,0.835:fdb32e,0.839:fdb52d,0.843:fdb62d,0.847:fdb82c,0.851:fdb92b,0.855:fdbb2b,0.859:fdbc2a,0.863:fdbe29,0.867:fdc029,0.871:fdc128,0.875:fdc328,0.878:fdc427,0.882:fdc626,0.886:fcc726,0.890:fcc926,0.894:fccb25,0.898:fccc25,0.902:fcce25,0.906:fbd024,0.910:fbd124,0.914:fbd324,0.918:fad524,0.922:fad624,0.925:fad824,0.929:f9d924,0.933:f9db24,0.937:f8dd24,0.941:f8df24,0.945:f7e024,0.949:f7e225,0.953:f6e425,0.957:f6e525,0.961:f5e726,0.965:f5e926,0.969:f4ea26,0.973:f3ec26,0.976:f3ee26,0.980:f2f026,0.984:f2f126,0.988:f1f326,0.992:f0f525,0.996:f0f623,1.000:eff821")

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

def colorize_direct_colorcap_fortran(a, desc):
	res = np.empty((a.shape[1],4),dtype=np.uint16)
	_colorize.direct_colorcap(a.T, res.T)
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
		matplotlib.colormaps.register(cmap)

def mpl_setdefault(name):
	import matplotlib.pyplot
	mpl_register(name)
	matplotlib.pyplot.rcParams['image.cmap'] = name
