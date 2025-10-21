import numpy as np
from . import bunch

default_site = bunch.Bunch(
	lat       = -22.9585,
	lon       = -67.7876,
	alt       = 5188.,
	weather   = "toco",
)

default_weather = bunch.Bunch(
	temperature =   0,
	humidity    = 0.2,
	pressure    = 550,
)

def expand_site(site):
	if isinstance(site, str):
		if site in ["default", "toco", "act", "so", "sa"]:
			return default_site
		else:
			raise ValueError("Unknown site '%s'" % str(site))
	return site

def expand_weather(weather, site=None):
	if weather is None or weather == "typical":
		weather = site.weather
	if isinstance(weather, str):
		if weather in ["default", "toco", "act", "so", "sa"]:
			return default_weather
		else:
			raise ValueError("Unknown weather '%s'" % str(weather))
	else:
		return weather
