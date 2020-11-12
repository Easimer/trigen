#!/bin/sh

 if [ -z ${1+x} ]; then
     echo "Usage: $0 path_to_data_dir"
     exit 1
 fi

 DATA_DIR=$1

 echo "Data dir: $DATA_DIR"

 for BACKEND in `seq 1 3`; do
     for SIZE in `seq 4 115`; do
         cat $DATA_DIR/raw.$BACKEND.$SIZE.txt | python ../../tools/parse_benchmark.py > $DATA_DIR/stats.$BACKEND.$SIZE.csv &
     done
done

wait
echo "All done"

