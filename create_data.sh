#!/bin/bash
directory="data/rootba/final"
directory_1="data/rootba/trafalgar"
directory_2="data/rootba/dubrovnik"
directory_3="data/rootba/ladybug"
directory_4="data/rootba/venice"
for file in "$directory"/*; do
	./bin/bal --input "$file" --solver-type SCHUR_COMPLEMENT --max-num-iterations 15 --power-sc-iterations 20 --residual-robust-norm NONE --varpro EXPOSE --num-threads 1 --alpha 10 --max-num-iterations-inner 10 --joint
done
for file in "$directory_1"/*; do
	./bin/bal --input "$file" --solver-type SCHUR_COMPLEMENT --max-num-iterations 15 --power-sc-iterations 20 --residual-robust-norm NONE --varpro EXPOSE --num-threads 1 --alpha 10 --max-num-iterations-inner 10 --joint
done
for file in "$directory_2"/*; do
	./bin/bal --input "$file" --solver-type SCHUR_COMPLEMENT --max-num-iterations 15 --power-sc-iterations 20 --residual-robust-norm NONE --varpro EXPOSE --num-threads 1 --alpha 10 --max-num-iterations-inner 10 --joint
done
for file in "$directory_3"/*; do
	./bin/bal --input "$file" --solver-type SCHUR_COMPLEMENT --max-num-iterations 15 --power-sc-iterations 20 --residual-robust-norm NONE --varpro EXPOSE --num-threads 1 --alpha 10 --max-num-iterations-inner 10 --joint
done
for file in "$directory_4"/*; do
	./bin/bal --input "$file" --solver-type SCHUR_COMPLEMENT --max-num-iterations 15 --power-sc-iterations 20 --residual-robust-norm NONE --varpro EXPOSE --num-threads 1 --alpha 10 --max-num-iterations-inner 10 --joint
done