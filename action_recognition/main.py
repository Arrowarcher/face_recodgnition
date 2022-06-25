# _*_ encoding: utf-8 _*_
# __author__ = 'lx'
from action_recognition.headPostEstimation import HeadPostEstimation
from rtmp.rtmp import rtmp_to_image, rtmp_to_img_list

camera_path = 'rtmp://192.168.1.213:1935/live/test'
head_post = HeadPostEstimation()

is_pass = False
while not is_pass:
    img_list = rtmp_to_img_list(camera_path)
    is_pass, act = head_post.detect_pose_by_img_list(img_list=img_list, poses=HeadPostEstimation.SHAKE_ACTION)

    print(act)