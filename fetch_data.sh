#!/usr/bin/env bash
mkdir -p data

while read p; do
	parr=($p)
	name=${parr[0]}
	url=${parr[1]}
	if [ ! -d "data/$name" ]; then
       		wget "$url" -O "data/$name.zip"
		unzip "data/$name.zip" -d "data/$name"
		rm -f "data/$name.zip"
	else
		echo "Dataset '$name' seems to be downloaded"
	fi
done < datasets.txt
