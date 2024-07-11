1. Create a super user profile for cvat if not alread complete
    a. Run the following command in your terminal
        --  docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

2. Once complete create a .env file with the following variables:
    CVAT_USERNAME='usr_name_here'
    CVAT_PASSWORD='insert_pw_here'

3. Place video in media folder

4. run command from folder that the main.py file lives:
    python3 main.py input/dir -kf -p --cvat

    -kf,--keyframes specifies whether or not you want keyframes from provided media

    -ds, --downsample specifies whether or not you want downsampled frames (can't do both key frames and downsample)

    -p, --predict if you want to run prediction on keyframed or downsampled images

    --cvat creates a task in cvat, converts annotations to xml and uploads both images and xml into cvat

   NOTES:
    Currently, when uploading projects you can't upload two projects with the same data. The current fix is to export the project, remove the data from cvat and rereun the pipeline. 