Docker:

- Build docker image: `docker build -t pyspark:latest -f Dockerfile .`
- To start the docker in interactive mode:
  Run `docker run -p 8888:8888 -it pyspark /bin/bash`
- To run notebook within docker: run `jupyter notebook`.

- More details can be found at: https://github.com/jupyter/docker-stacks

Info about Flint:
https://github.com/twosigma/flint/tree/master/python
Follow the instructions to install

Info about MNE:
https://mne.tools/stable/overview/cookbook.html

