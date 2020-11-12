#!/bin/bash

if [ -z ${CROSSCHECK+x} ]; then
    CROSSCHECK=../softbody_crosscheck/sb_crosscheck
fi

echo "Cross-check executable: $CROSSCHECK"

RUN_ID=`date +%s`
DATA_DIR="data_$RUN_ID"
mkdir $DATA_DIR

for BACKEND in `seq 1 3`; do
    echo "==========================="
    echo "Benching backend #$BACKEND"
    for SIZE in `seq 4 115`; do
        echo "Size=$SIZE"
        $CROSSCHECK -B $BACKEND -s $SIZE > $DATA_DIR/raw.$BACKEND.$SIZE.txt
    done
done
