#!/usr/bin/env bash

if [ "$(uname)" == "Darwin" ]; then
	if (gfortran --version); then
		echo "gfortran found. Assuming gcc (not just clang) is installed."
	else
		brew --version
		if [ $? -eq 0 ]; then
			echo
		else
			echo "NOTE: You might need to enter your user password since we are going to attempt to install homebrew."
			ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
		fi
		
		brew reinstall gcc

	fi

fi
