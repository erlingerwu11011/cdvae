#!/bin/bash

datasets=('barbell' 'stair' 'fork' 'backdoor')
mappings=('dnme' 'tnme' 'cmsm' 'tvsm')

for d in "${datasets[@]}"; do
  for s in "${mappings[@]}"; do
    python tmscm_sym.py -n ablation -d $d -e nf -s $s
    python tmscm_sym.py -n ablation -d $d -e nf -s $s -nm
    python tmscm_sym.py -n ablation -d $d -e nf -s $s -nc
    if [[ $s == 'cmsm' || $s == 'tvsm' ]]; then
      python tmscm_sym.py -n ablation -d $d -e nf -s $s -nt
    fi
  done
done