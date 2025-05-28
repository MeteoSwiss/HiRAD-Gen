run:
```
export EDF_PATH=`pwd`/.edf
```
This adds the repository path to the EDF search path.

run:
```
srun -A a-a122 --environment=ubuntu2 cat /etc/os-release
``` 

# local development
srun --environment $PWD/.edf/hirad-ci.toml -A a-a122 -p debug --pty bash 
 


# list current images
podman images

# build according to the dockerfile into an image with tag tmpv1, from current directory.
podman build -f ci/docker/Dockerfile -t tmpv1 .

# 
podman run -it localhost/tmpv1

mkdir /capstor/scratch/cscs/mmcgloho/images

# export the image into a sqsh file so it is availabe outside the interactive shell
enroot import -x mount -o /capstor/scratch/cscs/mmcgloho/images/hirad-pytorch-25.01-py3.sqsh podman://localhost/tmpv1

ls /capstor/scratch/cscs/mmcgloho/images