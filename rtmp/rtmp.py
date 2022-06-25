# _*_ encoding: utf-8 _*_
# __author__ = 'lx'
import cv2
import queue

# frame_queue = queue.Queue()
# camera_path = 'rtmp://58.200.131.2:1935/livetv/hunantv'
camera_path = 'rtmp://192.168.1.213:1935/live/test'

#获取摄像头参数

#读流函数
def Video(camera_path):
    vc = cv2.VideoCapture(camera_path) # vc.read()每次调用按顺序读取视频帧
    # fps = int(cap.get(cv2.CAP_PROP_FPS))                # 20fps
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(fps, width, height)
    if not vc.isOpened():
        raise IOError("could't open webcamera or video")
    while(vc.isOpened()):
        ret,frame = vc.read()
        #下面注释的代码是为了防止摄像头打不开而造成断流
        #if not ret:
            #vid = cv2.VideoCapture(camera_path）
            #if not vid.isOpened():
                #raise IOError("couldn't open webcamera or video")
            #continue
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('result', gray)
            # 如果读取速度低于视频流的输出速度，窗口显示的图片是好几秒钟前的内容。一段时间过后，缓存区将会爆满，程序报错
            if cv2.waitKey(30) & 0xFF == 27:  # 27是退出键
                break
    vc.release()
    cv2.destroyAllWindows()     # 关闭窗口的话关闭所有


frame_interval = 3
def rtmp_to_image(camera_path):
    capture = cv2.VideoCapture(camera_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))                # 20fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps,width,height)
    if not capture.isOpened():
        raise IOError("could't open webcamera or video")
    frame_count = 0         # 统计当前帧

    while (capture.isOpened()):
        ret, frame_rgb = capture.read()
        frame_count += 1
        if frame_count % frame_interval != 0:       # 间隔多少帧
            continue
        if cv2.waitKey(1) & 0xFF == 27:  # 27是退出键
            break
        if frame_rgb is None:
            break
        # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow('result', frame_rgb)
        yield frame_rgb
    capture.release()
    cv2.destroyAllWindows()

def rtmp_to_img_list(camera_path):
    capture = cv2.VideoCapture(camera_path)
    if not capture.isOpened():
        raise IOError("could't open webcamera or video")
    img_list= []
    while (capture.isOpened()):
        ret, frame_rgb = capture.read()
        print(ret)
        if cv2.waitKey(1) & 0xFF == 27:  # 27是退出键
            break
        if frame_rgb is not None:
            cv2.imshow('result', frame_rgb)
            img_list.append(frame_rgb)
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    return img_list


if __name__ == '__main__':
    camera_path = 'http://xxx:xxx@192.168.9.218:8081/video'
    camera_path = 0
    for i in rtmp_to_image(camera_path):
        pass



