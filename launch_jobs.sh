#!/bin/bash

N=2

for i in {1..2}
do
    echo "Starting job number $i out of $N"
    qsub job
done
