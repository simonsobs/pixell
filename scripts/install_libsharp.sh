#!/bin/bash
mkdir _build
cd _build 
git clone https://github.com/Libsharp/libsharp # do we want a frozen version?
cd libsharp  
aclocal && autoconf
./configure --enable-pic
make
