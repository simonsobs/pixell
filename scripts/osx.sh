#!/usr/bin/env bash

if [ "$(uname)" == "Darwin" ]; then
	if (gcc --version | grep clang); then

		brew --version
		if [ $? -eq 0 ]; then
			echo
		else
			echo "NOTE: You might need to enter your user password since we are going to attempt to install homebrew."
			ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
		fi
		
		# We need to install gcc and gfortran
		brew reinstall gcc
		CC=/usr/local/bin/gcc-8
		FC=/usr/local/bin/gfortran-8
	fi

fi
