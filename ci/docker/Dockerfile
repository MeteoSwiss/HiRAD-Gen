#FROM physicsnemo-cscs as builder
FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-physicsnemo:25.11-alps4-dev

# setup
RUN pip install --upgrade \
        pip

# install dependencies
RUN pip install \
        anemoi-datasets \
        cartopy \
        mlflow \
        xskillscore
