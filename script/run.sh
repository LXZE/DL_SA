# assest that current pwd is main dir
docker run  -v $(pwd):/file -v $(pwd)/../model:/model --name keras_dlsa -dit dl_sa:latest