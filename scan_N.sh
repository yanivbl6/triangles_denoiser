#!/bin/bash
#!/bin/bash

Ns=( 1 2 4 8 16 64 256 )

dev=0


pids=()

step=5

start=0
final=360
seeds=1

for N in ${Ns[@]}; do
    
    E=$((Es/N))

    alpha=$start



    while [ $alpha -le $final ]; do
        for seed in $(seq 1 $seeds); do
            cmd="python3 train.py --alpha $alpha --seed $seed --device $dev --num-samples $N $@ > /dev/null 2> /dev/null &"
            dev=$((dev+1))

            echo $cmd
            eval $cmd
            ##sleep 0 &

            if [ $dev -eq 8 ]; then
                dev=0
                wait
            fi
        done
        alpha=$((alpha+step))
    done



done 

wait

echo "All processes done"
