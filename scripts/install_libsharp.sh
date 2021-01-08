#!/usr/bin/env bash

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
cd $DEPDIR
[ -e libsharp ] || git clone https://github.com/Libsharp/libsharp # do we want a frozen version?
cd libsharp
echo $CIBW_PLATFORM
echo $CIBUILDWHEEL
# Only the last dockerenv check actually works for cibuildwheel
if [[ $CIBUILDWHEEL ]] ; then
	sed -i 's/march=native/march=x86-64/g' configure.ac
else
	echo "Not replacing native with x86-64. Binary will not be portable."
fi
cat configure.ac
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
autoconf
./configure --enable-pic
cat config.log
make
if [ $? -eq 0 ]; then
    echo Successfully installed libsharp.
	touch success.txt
else
	echo ERROR: Libsharp did not install correctly.
	exit 127
fi
rm -rf python/
