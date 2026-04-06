#!/bin/bash

datasets=('barbell' 'stair' 'fork' 'backdoor')
exogenous=('n' 'nf' 'gmm')
mappings=('dnme' 'tnme' 'cmsm' 'tvsm')

for d in "${datasets[@]}"; do
  for e in "${exogenous[@]}"; do
    for s in "${mappings[@]}"; do
      python tmscm_sym.py -n exogenous -d $d -e $e -s $s
    done
  done
done