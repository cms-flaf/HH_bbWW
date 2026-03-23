#!/bin/bash

#for MASS in 300 400 500 550 600 650 700 800 900
for MASS in 600
do
	combine -M AsymptoticLimits Run3card_recovery.txt --rMax 1 -t -1 -n m${MASS}_recovery -m ${MASS}
	combine -M AsymptoticLimits Run3card_res2b.txt --rMax 1 -t -1 -n m${MASS}_res2b -m ${MASS}
	combine -M AsymptoticLimits Run3card_boosted.txt --rMax 1 -t -1 -n m${MASS}_boosted -m ${MASS}
done
