from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import socket
import cv2
import numpy as np
import math
import time
import torch


class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + \
                (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):

        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time


def unet(pretrained_weights=None, input_size=(160, 320, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(8, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    # model.summary()

    return model


model = unet()
model.load_weights('viam')

# --------------------------LOAD YOLO-----------------------------------
model_yolo = torch.hub.load('ultralytics/yolov5', 'custom',
                            path='best.pt')

# --------------------------LOAD Classify-------------------------------


def classify(filter_size=(3, 3), input_shape=(64, 64, 3), pool_size=2, num_classes=3):
    model = Sequential()
    model.add(Conv2D(16, filter_size, activation='relu',
              input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model


model_classify = classify(num_classes=7)
model_classify.load_weights('final2')

# ---------------------------------------------------------------------
global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))
# s.connect(('host.docker.internal', PORT))


# Thuat toan dieu khien -------------------------------
def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


def turnLeft():
    message_getState = bytes("0", "utf-8")
    s.sendall(message_getState)
    state_date = s.recv(100)

    try:
        current_speed, current_angle = state_date.decode(
            "utf-8"
        ).split(' ')
    except Exception as er:
        print(er)
        pass

    # for i in range(100000):
    #     message = bytes(f"1 {-25} {1}", "utf-8")
    #     s.sendall(message)
    message = bytes(f"1 {-13} {1}", "utf-8")
    s.sendall(message)
    time.sleep(4)
    data = s.recv(100000)


def turnRight():
    message_getState = bytes("0", "utf-8")
    s.sendall(message_getState)
    state_date = s.recv(100)

    try:
        current_speed, current_angle = state_date.decode(
            "utf-8"
        ).split(' ')
    except Exception as er:
        print(er)
        pass
    message = bytes(f"1 {18} {1}", "utf-8")
    s.sendall(message)
    time.sleep(1.75)
    data = s.recv(100000)


def goStraight():
    message_getState = bytes("0", "utf-8")
    s.sendall(message_getState)
    state_date = s.recv(100)

    try:
        current_speed, current_angle = state_date.decode(
            "utf-8"
        ).split(' ')
    except Exception as er:
        print(er)
        pass
    message = bytes(f"1 {-1} {20}", "utf-8")
    s.sendall(message)
    time.sleep(2.75)
    data = s.recv(100000)

#['left', 'none_object', 'no_left', 'no_right', 'no_straight', 'right', 'straight']


def check_Sign_Traffic(list_Sign):
    print("Success")
    if list_Sign == 'straight':
        print('gap bien di thang... di thang')
        goStraight()

    elif list_Sign == 'right':
        print('gap bien re phai di phai')
        turnRight()

    elif list_Sign == 'left':
        print('gap bien re trai... di ben trai')
        turnLeft()

    elif list_Sign == 'no_straight' and angle_noStraight > 0:
        print('gap bien no_straight va di ben phai')
        turnRight()

    elif list_Sign == 'no_straight' and angle_noStraight < -20:

        print('gap bien no_straight va di ben trai')
        turnLeft()

    elif list_Sign == 'no_right' and check_noLeft_or_noRight == 1:
        print('gap bien no_right va di thang')
        goStraight()

    elif list_Sign == 'no_right' and check_noLeft_or_noRight == 0:
        print('gap bien no_right va di ben trai')
        turnLeft()

    elif list_Sign == 'no_left' and check_noLeft_or_noRight == 1:
        print('gap bien no_left va di thang')
        goStraight()

    elif list_Sign == 'no_left' and check_noLeft_or_noRight == 0:
        print('gap bien no_left va di ben phai')
        turnRight()
    else:
        pass


# findCenter ------------------------------------------
def cal_center(line):
    arr = []
    for x, y in enumerate(line):
        if y == 255:
            arr.append(x)
    arrmax = max(arr)
    arrmin = min(arr)
    center = int((arrmax+arrmin)/2)
    return center


def cal_centerRight(line):
    arr = []
    for x, y in enumerate(line):
        if y == 255:
            arr.append(x)
    arr_1 = []
    arr_2 = []
    for i in range(len(arr)-1):
        if arr[i]+1 == arr[i+1]:
            arr_1.append(arr[i])
        else:
            break
    for i in range(len(arr), 1):
        if arr[i]-1 == arr[i-1]:
            arr_2.append(arr[i])
        else:
            break
    if len(arr_1) > len(arr_2):
        arr = arr_1
    else:
        arr = arr_2
    arrmax = max(arr)
    arrmin = min(arr)
    centerRight = int((75*arrmax + 25*arrmin) / 100)
    return centerRight


angle_PID = [0, 0, 0]
steer_PID = PID(1, 0.35, 0.4)
steer_PID.SetPoint = 0

if __name__ == "__main__":
    try:
        while True:

            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )
                print('---------')
                # print(sendBack_Speed, sendBack_angle)
                # print(current_speed, current_angle)
                # your process here

                # Cắt ảnh và predict
                img = image[136:, :]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img = cv2.resize(img, dsize=(320, 160))
                img = img[None, :, :, :]
                y_pre = model.predict(img)[0]
                # print(y_pre.shape)

                # Lọc nhiễu
                y_pre = np.where(y_pre < 0.8, 0.0, 1.0) * 255
                y_pre = cv2.resize(y_pre, dsize=(640, 224))
                # y_pre = cv2.GaussianBlur(y_pre,(7,7),cv2.BORDER_DEFAULT)
                # y_pre = cv2.erode(y_pre, (9, 9))
                # y_pre = cv2.erode(y_pre, (9, 9))
                y_pre = np.where(y_pre < 240, 0.0, 255.0)
                # print("Up size = " + str(y_pre.shape))

# ---------------Tính góc và vẽ line-------------------------------------------------------
                lineImg = y_pre[40, :]
                centerLine = cal_centerRight(lineImg)
                angle = math.degrees(
                    math.atan((centerLine - y_pre.shape[1]/2) / (y_pre.shape[0] - 40)))
                cv2.line(y_pre, (centerLine, 40),
                         (int(y_pre.shape[1]/2), y_pre.shape[0]), (0, 0, 0), 5)
# ---------------Check cấm đi thẳng -->
                checkLine = y_pre[30, :]
                check_noStraight = cal_center(checkLine)
                angle_noStraight = math.degrees(
                    math.atan((check_noStraight - y_pre.shape[1]/2) / (y_pre.shape[0] - 30)))
                cv2.line(y_pre, (check_noStraight, 30),
                         (int(y_pre.shape[1]/2), y_pre.shape[0]), (0, 0, 255), 5)
                # print("=====")
                # print(angle_noStraight)
# ---------------Check cấm quẹo trái hoặc phải -->
                checkMid = y_pre[5, :]
                for pixel in checkMid:
                    if pixel == 255:
                        check_noLeft_or_noRight = 1
                        break
                    else:
                        check_noLeft_or_noRight = 0


# ---------------Scale góc và update PID------------------------------------------------------------
                sendBack_angle = angle
                steer_PID.update(sendBack_angle)
                sendBack_angle = -steer_PID.output
                sendBack_angle = (sendBack_angle * 25) / 90
                angle_PID.append(sendBack_angle)

                # Điều khiển
                if -2 < sendBack_angle < 2:
                    sendBack_angle = 0
                    sendBack_Speed = 20

                elif -5 < sendBack_angle < 5:
                    if sendBack_angle < 0:
                        sendBack_angle = -1
                    else:
                        sendBack_angle = 1
                    sendBack_Speed = 20

                elif -10 < sendBack_angle < 10:
                    if sendBack_angle < 0:
                        sendBack_angle = -4
                    else:
                        sendBack_angle = 4
                    sendBack_Speed = 15

                elif -20 < sendBack_angle < 20:
                    sendBack_angle = sendBack_angle*1.2
                    sendBack_Speed = 15

                else:
                    sendBack_Speed = 15
                # Control(sendBack_angle, sendBack_Speed)

                # cv2.imshow("IMG", y_pre)

# --------------------------------------------------------------------------------------
                mid = image[:, 280:]
                detections = model_yolo(mid)
                results = detections.pandas().xyxy[0].to_dict(orient="records")

                for result in results:
                    confidence = result['confidence']
                    name = result['name']
                    clas = result['class']
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    #  nay la cho y1 < 25
                    if y1 >= 0 and y1 < 25:
                        crop = mid[y1: y2, x1:x2]
                        crop = cv2.resize(crop, dsize=(64, 64))
                        crop = np.expand_dims(crop, axis=0)
                        y_classify = model_classify.predict(crop)
                        signTraffic = np.argmax(y_classify, axis=1)
                        if signTraffic == [0]:
                            signTraffic = 'left'
                        if signTraffic == [1]:
                            signTraffic = 'none_object'
                        if signTraffic == [2]:
                            signTraffic = 'no_left'
                        if signTraffic == [3]:
                            signTraffic = 'no_right'
                        if signTraffic == [4]:
                            signTraffic = 'no_straight'
                        if signTraffic == [5]:
                            signTraffic = 'right'
                        if signTraffic == [6]:
                            signTraffic = 'straight'

                        # signTraffic = signTraffic + '_return'
                        cv2.rectangle(mid, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # cv2.putText(mid, signTraffic, (x1, y1),
                        #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                        # if y1 < 20:
                        #     # print(signTraffic + ' da lai')
                        #     signTraffic = signTraffic + 'returnnn'
                        #     cv2.putText(mid, signTraffic, (x2, y2),
                        #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

                        check_Sign_Traffic(signTraffic)

                # cv2.imshow("mid", mid)
                # cv2.waitKey(1)

            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()
