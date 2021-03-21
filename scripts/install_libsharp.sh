#!/usr/bin/env bash

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
cd $DEPDIR
[ -e libsharp2 ] || git clone https://gitlab.mpcdf.mpg.de/mtr/libsharp.git libsharp2 # do we want a frozen version?
cd libsharp2
mkdir -p build

aclocal
if [ $? -eq 0 ]; then
    echo Found automake.
else
	if [ "$(uname)" == "Darwin" ]; then
		echo WARNING: automake not found. Since this looks like Mac OS, attempting to install it.
		brew install autoconf automake
		if [ $? -eq 0 ]; then
			echo
		else
			echo "NOTE: You might need to enter your user password since we are going to attempt to install homebrew."
			ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
			brew install autoconf automake
		fi
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
fi

autoreconf -i
cat configure.ac

echo $CIBW_PLATFORM
echo $CIBUILDWHEEL
# Only the last dockerenv check actually works for cibuildwheel
if [[ $CIBUILDWHEEL ]] ; then
	CFLAGS="-DMULTIARCH -std=c99 -O3 -ffast-math"
else
	echo "Using -march=native. Binary will not be portable."
	CFLAGS="-march=native -std=c99 -O3 -ffast-math"
fi

CFLAGS=$CFLAGS ./configure --prefix=${PWD}/build --enable-shared=no --with-pic=yes

cat config.log
make
make install
if [ $? -eq 0 ]; then
    echo Successfully installed libsharp.
	touch success.txt
else
	echo ERROR: Libsharp did not install correctly.
	exit 127
fi
rm -rf python/
