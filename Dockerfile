FROM jupyter/pyspark-notebook


RUN \
  git clone https://github.com/twosigma/flint.git && \
  cd ~/flint/python && \
  python setup.py install

CMD ["bash"]

