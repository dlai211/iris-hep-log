#!/bin/bash

# pts=(10 20 30 40 50 60 70 80 90 100)
pts=(5 15 25 35 45 55 65 75 85 95)
session_count=0

for ((i=0; i<${#pts[@]}; i+=2)); do
    pt1=${pts[i]}
    pt2=${pts[i+1]}

    session_name="pt_${pt1}_${pt2}"
    echo "Starting screen session: $session_name"

    screen -dmS "$session_name" bash -c "
        cd /data/jlai/iris-hep && \
        source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh && \
        lsetup \"views LCG_106a x86_64-el9-gcc13-opt\" && \
        source acts-build/python/setup.sh && \

        echo 'Running for pT = $pt1' && \
        rm -rf OutputPT/output_pt_${pt1} && \
        mkdir OutputPT/output_pt_${pt1} && \
        python acts/Examples/Scripts/Python/full_chain_odd.py \
            --events 100000 \
            --gun-particles 1 \
            --gun-multiplicity 1 \
            --gun-eta-range 0 0 \
            --gun-pt-range $pt1 $pt1 \
            --output OutputPT/output_pt_${pt1}

        echo 'Running for pT = $pt2' && \
        rm -rf OutputPT/output_pt_${pt2} && \
        mkdir OutputPT/output_pt_${pt2} && \
        python acts/Examples/Scripts/Python/full_chain_odd.py \
            --events 100000 \
            --gun-particles 1 \
            --gun-multiplicity 1 \
            --gun-eta-range 0 0 \
            --gun-pt-range $pt2 $pt2 \
            --output OutputPT/output_pt_${pt2}

        echo 'Done for session $session_name'
        exec bash
    "

    ((session_count++))
    if [ "$session_count" -ge 5 ]; then
        break
    fi
done

