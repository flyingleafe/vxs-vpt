#!/usr/bin/env bash
mkdir -p data

while read p; do
	parr=($p)
	name=${parr[0]}
	url=${parr[1]}
        archive_type=${parr[2]}
	if [ ! -d "data/$name" ]; then
		archive_name="data/$name.$archive_type"
       		wget "$url" -O "$archive_name"
		if [ $archive_type == "zip" ]; then
			unzip "$archive_name" -d "data/$name"
			rm -f "$archive_name"
		elif [ $archive_type == "rar" ]; then
			mkdir -p "data/$name"
			mv "$archive_name" "data/$name/data.rar"
			pushd "data/$name"
			unrar x data.rar
			rm -f data.rar
			popd
		else
			echo "Unknown archive type '$archive_type' for dataset '$name', skipping"
		fi	
	else
		echo "Dataset '$name' seems to be downloaded"
	fi
done < datasets.txt
