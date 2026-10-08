# HPC & Supercomputer Deployment Readiness Checklist for FANNS

## 1. Workload & Memory Scaling Estimation

| Dataset Scale | Vectors | Dimensions | Precision | Raw Vector RAM | HNSW Graph RAM (M=32) | Total Expected RAM | Feasibility (Local Laptop) | Supercomputer Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small (Current)** | 1,000 | 4,096 | Float32 | ~16 MB | ~1 MB | **~25 MB** | Feasible (Verified) | Single Core / Debug |
| **Medium (100K)** | 100,000 | 4,096 | Float32 | ~1.6 GB | ~250 MB | **~3.5 GB** | Feasible on 16GB+ RAM | 1 Node (4-8 Cores) |
| **Full (1M)** | 1,000,000 | 4,096 | Float32 | ~16.4 GB | ~2.5 GB | **~32-40 GB** | **OOM on Laptop** | **HPC Compute Node (64GB+ RAM)** |

---

## 2. Supercomputer Cluster Checklist

### Hardware Specifications
- [ ] **Node Architecture**: x86_64 (e.g., AMD EPYC 7763 or Intel Xeon Platinum 8358)
- [ ] **RAM Allocation**: Minimum 64 GB per task for 1M vectors (or 16 GB for 100K vectors)
- [ ] **CPU Cores**: 16 to 32 dedicated physical cores per job
- [ ] **Local Scratch Storage**: High-speed NVMe `/tmp` or Lustre parallel file system (>50 GB free)

### Software & Environment
- [ ] **Operating System**: Linux (CentOS/RHEL/Ubuntu LTS)
- [ ] **Scheduler**: Slurm Workload Manager (`sbatch`, `srun`)
- [ ] **Python Version**: Python 3.10, 3.11, or 3.12 (with GCC toolchain available for native extensions)
- [ ] **C++ Compiler**: GCC >= 9.3 or Clang >= 11 for OpenMP multithreading
- [ ] **Environment Isolation**: Python virtual environment or Conda module (`module load python/3.11`)

---

## 3. Recommended Slurm Job Parameters

```bash
#SBATCH --job-name=fanns_benchmark_100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --output=logs/fanns_%j.out
#SBATCH --error=logs/fanns_%j.err
```
