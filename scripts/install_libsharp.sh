#!/usr/bin/env bash
DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
cd $DEPDIR
[ -e libsharp ] || git clone https://github.com/Libsharp/libsharp # do we want a frozen version?
cd libsharp
aclocal
if [ $? -eq 0 ]; then
    echo Found automake.
else
	if [ "$(uname)" == "Darwin" ]; then
		echo WARNING: automake not found. Since this looks like Mac OS, attempting to install it.
		brew install autoconf
		aclocal
	elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
		echo WARNING: automake not found. Since this looks like Linux, attempting to load its module.
		module load autotools
		aclocal
	fi
	if [ $? -eq 0 ]; then
		echo Found automake.
	else
		echo WARNING: automake not found. Please install this or libsharp will not be installed correctly.
		exit 127
	fi
	
autoconf
./configure --enable-pic
make
rm -rf python/
