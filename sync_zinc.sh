#!/bin/bash
set -e

kalki="boris@golab.zapto.org:/home/boris/Data/Zinc"
bb="jack@bb.zapto.org:/home/jack/Data/Zinc"
pitt="mixarcid@cluster.csb.pitt.edu:~/Data/Zinc"
home="$HOME/Data/Zinc"

if [ $1 = "kalki" ]; then
    echo "grabbing from kalki"
    #scp $kalki/files.txt $home/
    #scp $kalki/files_filtered_mol.txt $home/
    #scp $home/files.txt $bb/
    #scp $home/files_filtered_mol.txt $bb/
    scp $home/files_filtered*.txt $bb/
elif [ $1 = "bb" ]; then
    echo "grabbing from bb"
    #scp $bb/files.txt $home/
    #scp $bb/files_filtered*.txt $home/
    #scp $home/files.txt $kalki/
    scp $home/files_filtered*.txt $kalki/
else
    echo "sending to all"
    #scp $home/files.txt $bb/
    scp $home/files_filtered_$1.txt $bb/
    #scp $home/files.txt $kalki/
    scp $home/files_filtered_$1.txt $kalki/
    scp $home/files_filtered_$1.txt $pitt/
fi
