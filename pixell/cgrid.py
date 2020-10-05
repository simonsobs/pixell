"""This module implements functions for drawing a coordinate grid and
coordinate axes on an image, for example for use with enmap."""
import numpy as np, time, os
from PIL import Image, ImageDraw, ImageFont
from . import utils, enmap, wcsutils

def calc_line_segs(pixs, steplim=10.0, extrapolate=2.0):
	"""Given a sequence of points, split into subsequences
	where huge jumps in values occur. Extrapolate from each
	edge of the cut to avoid ----|   |---- effects."""
	lens = np.sum((pixs[1:]-pixs[:-1])**2,1)**0.5
	typical = np.median(lens)
	jump = np.where(lens > typical*steplim)[0]
	segs = np.split(pixs, jump+1)
	# Extrapolate into the gaps left from splitting
	def extrap(seg):
		if len(seg) < 2: return seg
		return np.concatenate([seg,[seg[-1]+(seg[-1]-seg[-2])*extrapolate]])
	nseg = len(segs)
	for i in range(nseg-1): segs[i] = extrap(segs[i])
	for i in range(1,nseg): segs[i] = extrap(segs[i][::-1])[::-1]
	return segs

#def calc_line_segs(pixs, steplim=10.0):
#	# Split on huge jumps
#	lens = np.sum((pixs[1:]-pixs[:-1])**2,1)**0.5
#	typical = np.median(lens)
#	jump = np.where(lens > typical*steplim)[0]
#	return np.split(pixs, jump+1)

class Gridinfo: pass

def calc_gridinfo(shape, wcs, steps=[2,2], nstep=[200,200], zenith=False, unit=1):
	"""Return an array of line segments representing a coordinate grid
	for the given shape and wcs. the steps argument describes the
	number of points to use along each meridian."""
	if   unit in ["d","degree"]: unit = 1.0
	elif unit in ["m","arcmin"]: unit = 1.0/60
	elif unit in ["s","arcsec"]: unit = 1.0/3600
	steps = (np.zeros([2])+steps)*unit
	nstep = np.zeros([2],dtype=int)+nstep

	gridinfo = Gridinfo()
	if wcsutils.is_plain(wcs):
		box = np.sort(enmap.box(shape, wcs),0)/utils.degree
		start = np.floor(box[0]/steps)*steps
		nline = np.floor(box[1]/steps)-np.floor(box[0]/steps)+1
	else:
		box   = np.array([[-90.,0.],[90.,360.]])
		start = np.array([-90.,0.])
		nline = np.array([180.,360.])/steps+1

	gridinfo.lon = []
	gridinfo.lat = []
	gridinfo.shape = shape[-2:]
	gridinfo.wcs = wcs
	# Draw lines of longitude
	for phi in start[1] + np.arange(nline[1])*steps[1]:
		# Loop over theta
		pixs = np.array(wcsutils.nobcheck(wcs).wcs_world2pix(phi, np.linspace(box[0,0],box[1,0],nstep[0],endpoint=True), 0)).T
		if not wcsutils.is_plain(wcs): phi = utils.rewind(phi, 0, 360)
		gridinfo.lon.append((phi/unit,calc_line_segs(pixs)))
	# Draw lines of latitude
	for theta in start[0] + np.arange(nline[0])*steps[0]:
		# Loop over phi
		pixs = np.array(wcsutils.nobcheck(wcs).wcs_world2pix(np.linspace(box[0,1],box[1,1]+0.9,nstep[1],endpoint=True), theta, 0)).T
		if zenith: theta = 90-theta
		gridinfo.lat.append((theta/unit,calc_line_segs(pixs)))
	return gridinfo

def draw_grid(gridinfo, color="00000020", background=None):
	col = tuple([int(color[i:i+2],16) for i in range(0,len(color),2)])
	grid = Image.new("RGBA", gridinfo.shape[-2:][::-1])
	draw = ImageDraw.Draw(grid, "RGBA")
	for cval, segs in gridinfo.lon:
		for seg in segs:
				draw.line([tuple(i) for i in seg], fill=col)
	for cval, segs in gridinfo.lat:
		for seg in segs:
			draw.line([tuple(i) for i in seg], fill=col)
	if background is not None:
		grid = Image.alpha_composite(background, grid)
	return grid

def calc_label_pos(linesegs, shape):
	"""Given linegegs = [values, curves], where values[ncurve]
	is a float array containing a coordinate label for each curve,
	and pixels[ncurve][nsub][npoint,{x,y}] contains the pixel coordinates
	for the points makeing up each curve, computes
	[nlabel,{label,x,y}] for each label to place on the image.
	Labels are placed where curves cross the edge of the image."""
	lables = []
	shape  = np.array(shape)
	for label_value, curveset in linesegs:
		# Loop over subsegments, which are generated due to angle wrapping
		for curve in curveset:
			# Check if we cross one of the edges of the image. We want
			# the crossing to happen between the selected position and the next
			# Used to have curve + 0.5 for ldist and shape - 0.5 - curve for rdist
			# here. Not sure why. It messed up polar coordinates, so I removed it
			# for now.
			ldist  = curve
			rdist  = shape - curve
			cross1 = np.sign(ldist[1:]) != np.sign(ldist[:-1])
			cross2 = np.sign(rdist[1:]) != np.sign(rdist[:-1])
			cands  = np.array(np.where(cross1 | cross2))
			# Cands is [2,ncand] indices into curve. The second index indicates
			# whether the crossing happened in the x or y coordinate.
			# Remove crossings that happen entirely outside the image. We
			# approximate this as happening when the non-crossing coordinate
			# of each candidate is outside the image.
			other = curve[cands[0],1-cands[1]]
			outside = (other<0)|(other>shape[1-cands[1]])
			matches = cands[:,~outside]
			# Do we have any matches? If so, compute the exact crossing point
			# and place the label there
			if matches.size > 0:
				for ind, dim in matches.T:
					# Here "a" indicates the pixel coordiante that crossed the
					# edge, and "b" is the other one. For example, if we crosses
					# in the horizontal direction, a=x and b=y.
					a = curve[[ind,ind+1],[dim,dim]]
					b = curve[[ind,ind+1],[1-dim,1-dim]]
					# Crossing point
					slope = (b[1]-b[0])/(a[1]-a[0])
					across = float(0 if a[0]*a[1] <= 0 else shape[dim])
					bcross = b[0] + slope*(across-a[0])
					label  = [label_value,0,0]
					# Unshuffle from a,b to x,y
					label[1+dim] = across
					label[2-dim] = bcross
					lables.append(label)
			else:
				# No edge crossing. But perhaps that's because everything is
				# happening inside the image. If so, the first point should
				# be inside.
				if np.all(curve[0]>=0) and np.all(curve[0]<shape):
					lables.append([label_value,curve[0,0],curve[0,1]])
	return lables

#def calc_label_pos(linesegs, shape):
#	# For each grid line, identify where we enter and exit the
#	# image. If these points exist, draw coorinates there. If they
#	# do not, check if the 0 coordinate is in the image. If it is,
#	# draw the coordinate there. Otherwise there is no point in
#	# drawing.
#	shape = np.array(shape)
#	label_pos = []
#	for cval, segs in linesegs:
#		for seg in segs:
#			seg = np.array(seg)
#			edges = np.array(np.where((seg[1:]*seg[:-1] < 0)|((seg[1:]-shape)*(seg[:-1]-shape) < 0)))
#			# Mask those outside the image
#			ocoord = edges.copy(); ocoord[1] = 1-ocoord[1]
#			other = seg[tuple(ocoord)]
#			outside = (other<0)|(other>shape[1-edges[1]])
#			edges = edges[:,~outside]
#
#			## Also look for cases where we're right on top of an image edge
#			#onedge = (seg == 0) | (seg == shape)
#			## Some lines run along an edge in parallel to it. Disqualify these
#			#good = ~np.any(np.all(onedge,0))
#			#onedge &= good
#			#onedge = np.array(np.where(onedge))
#			#edges = np.concatenate([edges,onedge],1)
#
#			if edges.size > 0:
#				# Ok, we cross an edge. Interpolate the position for each
#				for ei,ec in edges.T:
#					x = seg[([ei,ei+1],[ec,ec])]
#					y = seg[([ei,ei+1],[1-ec,1-ec])]
#					xcross = float(0 if x[0]*x[1] <= 0 else shape[ec])
#					ycross = y[0] + (y[1]-y[0])*(xcross-x[0])/(x[1]-x[0])
#					entry  = [cval,0,0]
#					entry[2-ec] = ycross
#					entry[1+ec] = xcross
#					label_pos.append(entry)
#			else:
#				# No edge crossing. But perhaps that's because everything is
#				# happening inside the image. If so, the first point should
#				# be inside.
#				if np.all(seg[0]>=0) and np.all(seg[0]<shape):
#					label_pos.append([cval,seg[0,0],seg[0,1]])
#	return label_pos

def calc_bounds(boxes, size):
	"""Compute bounding box for a set of boxes [:,{from,to},{x,y}].
	The result will no less than ((0,0),size)."""
	return np.array([np.minimum((0,0),np.min(boxes[:,0],0)),np.maximum(size,np.max(boxes[:,1],0))])

def expand_image(img, bounds):
	res = Image.new("RGBA", tuple(bounds[1]-bounds[0]))
	res.paste(img, tuple(-bounds[0]))
	return res

def get_font(fsize=16, fname="arial.ttf"):
	try:
		font = ImageFont.truetype(fname, size=fsize)
	except IOError:
		# Load fallback font
		font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "arial.ttf"), size=fsize)
	return font

def draw_labels(img, label_pos, fname="arial.ttf", fsize=16, fmt="%g", color="000000", return_bounds=False):
	# For each label, determine the size the text would be, and
	# displace it left, right, up or down depending on which edge
	# of the image it is at
	col = tuple([int(color[i:i+2],16) for i in range(0,len(color),2)])
	font = get_font(fsize, fname)
	labels = []
	boxes  = []
	for cval, x, y in label_pos:
		pos   = np.array([x,y])
		label = fmt % cval
		lsize = np.array(font.getsize(label))
		if   x == 0:           box = np.array([pos-[lsize[0],lsize[1]/2],pos+[0,lsize[1]/2]])
		elif x == img.size[0]: box = np.array([pos-[0,lsize[1]/2],pos+[lsize[0],lsize[1]/2]])
		elif y == 0:           box = np.array([pos-[lsize[0]/2,lsize[1]],pos+[lsize[0]/2,0]])
		elif y == img.size[1]: box = np.array([pos-[lsize[0]/2,0],pos+[lsize[0]/2,lsize[1]]])
		else:                  box = np.array([pos-lsize/2,pos+lsize/2])
		labels.append(label)
		boxes.append(box)
	boxes  = np.array(boxes).astype(int)
	# Image might contain no lines at all.
	if boxes.size == 0: boxes = np.array([[0,0],[0,0]])
	# Pad image to be large enough to hold the displaced labels
	bounds = calc_bounds(boxes, img.size)
	img    = expand_image(img, bounds)
	boxes -= bounds[0]
	# And draw the text
	draw = ImageDraw.Draw(img)
	for label, box in zip(labels, boxes):
		draw.text(box[0], label, col, font=font)
	if return_bounds:
		return img, bounds
	else:
		return img
