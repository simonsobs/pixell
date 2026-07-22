import numpy as np
from . import bunch

sites = bunch.Bunch(
	# The distances between these amount to up to 0.06 arcmin, so
	# small, but not as small as I thought.
	act = bunch.Bunch(
		lat       = -22.9585,
		lon       = -67.7876,
		alt       = 5188.,
		weather   = "toco",
	),
	lat = bunch.Bunch(
		lat       = -22.96096,
		lon       = -67.78769,
		alt       = 5188.,
		weather   = "toco",
	),
	sat1 = bunch.Bunch(
		lat       = -22.96011,
		lon       = -67.78836,
		alt       = 5188.,
		weather   = "toco",
	),
	sat2 = bunch.Bunch(
		lat       = -22.96010,
		lon       = -67.78813,
		alt       = 5188.,
		weather   = "toco",
	),
	sat3 = bunch.Bunch(
		lat       = -22.95999,
		lon       = -67.78793,
		alt       = 5188.,
		weather   = "toco",
	),
)
sites.so      = sites.lat
sites.toco    = sites.lat
sites.default = sites.toco

default_site  = sites.default

weathers = bunch.Bunch(
	toco = bunch.Bunch(
		temperature =   0,
		humidity    = 0.2,
		pressure    = 550,
	),
)
weathers.default = weathers.toco

default_weather = weathers.default

def expand_site(site):
	if isinstance(site, str):
		if site in sites: return sites[site]
		else:
			raise ValueError("Unknown site '%s'" % str(site))
	return site

def expand_weather(weather, site=None):
	if weather is None or weather == "typical":
		weather = site.weather
	if isinstance(weather, str):
		if weather in weathers: return weathers[weather]
		else:
			raise ValueError("Unknown weather '%s'" % str(weather))
	else:
		return weather
