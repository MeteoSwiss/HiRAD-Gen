run:
```
export EDF_PATH=`pwd`/.edf
```
This adds the repository path to the EDF search path.

run:
```
srun -A a-a122 --environment=ubuntu2 cat /etc/os-release
```