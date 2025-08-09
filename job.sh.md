```
#!/bin/bash
```
```
# List your data files here
data_files=("data1.data" "data2.data" "data3.data" "data4.data" "data5.data")
```
数据文件列表，要跑5个体系。
```
for data_file in "${data_files[@]}"; do
    # Get base name for job name
    base=$(basename "$data_file" .data)

    # Create a new input file for this job by replacing the read_data line
    sed "s|read_data .*|read_data ${data_file}|" in.lmp > in.${base}.lmp

    # Create a job script for this data file
    cat > job_${base}.sh <<EOF
```
逐个遍历数据文件
```
#!/bin/bash
#SBATCH --job-name=${base}
#SBATCH --output=output.%j.%N
#SBATCH --error=error.%j.%N
#SBATCH --partition=compute
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=128
#SBATCH --mem=240G
#SBATCH --account=TG-MAT250010
#SBATCH --export=ALL
#SBATCH -t 20:00:00
#SBATCH --mail-user=YOUREMAIL
#SBATCH --mail-type=ALL
```
资源配置。
```
module purge
module load slurm
module load cpu/0.15.4
module load gcc/10.2.0
module load intel-mpi/2019.8.254
```
配置运行环境。
```
INPUT=in.${base}.lmp
mpirun -np \$SLURM_NTASKS ./lmp_mpi -in \$INPUT
EOF

    # Submit the job
    sbatch job_${base}.sh
done
```
每个数据文件对应的脚本结尾与提交。
