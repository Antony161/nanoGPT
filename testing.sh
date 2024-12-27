#!/bin/bash
for n in {1..3}
do
time python sample.py --out_dir=out-shakespeare-char --device=cpu &> ouput.txt
done
