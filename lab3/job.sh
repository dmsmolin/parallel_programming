#!/bin/bash
#SBATCH --job-name=lab_mpi
#SBATCH --output=matrix_report.txt
#SBATCH --error=matrix_errors.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:25:00

set -u

module purge
module load intel/icc18
module load intel/mpi4

if [ ! -x "./multmatrix_mpi" ]; then
    echo "ERROR: ./multmatrix_mpi not found or not executable" >&2
    exit 1
fi

for N in 200 400 800 1600 2000; do
    for f in "matrix_a${N}.txt" "matrix_b${N}.txt"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: missing input file $f" >&2
            exit 1
        fi
    done
done

rm -f timing.txt errors.txt matrix_c*.txt

FAILED=0
for P in 1 2 4 8; do
    for N in 200 400 800 1600 2000; do
        echo "=== P=$P, n=$N ==="
        if mpirun -np $P -bootstrap slurm ./multmatrix_mpi $N; then
            echo "  -> OK"
        else
            echo "  -> FAILED (exit $?)" >&2
            FAILED=$((FAILED+1))
        fi
    done
done

if [ $FAILED -ne 0 ]; then
    echo "WARNING: $FAILED run(s) failed, see matrix_errors.err" >&2
else
    echo "All runs OK"
fi

echo "Done!"