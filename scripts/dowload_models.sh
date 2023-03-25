#!/bin/bash

wget "https://onedrive.live.com/download?cid=7AA4E17800AB44C8&resid=7AA4E17800AB44C8%21510310&authkey=ABxsQgnGdJk7wWI&download=1" -O infosol_models.tar.zst
zstd -dc infosol_models.tar.zst | tar -xvf - ./models
