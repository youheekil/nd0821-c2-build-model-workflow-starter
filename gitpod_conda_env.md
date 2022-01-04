# Install Miniconda - Anaconda on Gitpod workspace

## Instructions
1. create ``get.sh``

```sh
# get.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. and run 
```bash
cat get.sh # for checking contents of get.sh file
bash get.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
type `yes` and then set the path workspace as 
```
{pwd}/miniconda3
```

**make sure to set the path of workplace after checking current working directory in another terminal** 

```bash
pwd # print current working directory
```

3. modify ``.gitgnore`` file

```gitgnore
# conda for gitpod env
anaconda3
miniconda3
```

4. after pushing it to git remove ``get.sh``, ``Miniconda3-lat
est-Linux-x86_64.sh`` files. 


