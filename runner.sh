#!/bin/bash

alphas=( 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 )  

dev=0


pids=()

for alpha in ${alphas[@]}; do
    cmd="python3 train.py --alpha $alpha --device $dev $@ > /dev/null 2> /dev/null &"
    dev=$((dev+1))

    echo $cmd
    eval $cmd
    ##sleep 3 &

    pids[${i}]=$!

    if [ $dev -eq 8 ]; then
        dev=0
        wait
    fi
done

wait

echo "All processes done"
