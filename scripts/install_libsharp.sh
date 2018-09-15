#!/bin/bash
DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
cd $DEPDIR
[ -e libsharp ] || git clone https://github.com/Libsharp/libsharp # do we want a frozen version?
cd libsharp  
aclocal && autoconf
./configure --enable-pic
make
rm -rf python/
