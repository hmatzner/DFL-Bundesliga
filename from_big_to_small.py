import os
import glob
import numpy as np
import cv2
from scipy import spatial
from scipy.spatial.distance import cdist
from google.colab import drive
drive.mount('/content/drive')

CLASS_INDEX = 0
X_INDEX = 1
Y_INDEX = 2
PERSON_CLASS = '0'
BALL_CLASS = '32'
X_SIZE = 1928
Y_SIZE = 1024
CROP_SIZE = 224
radius = int(CROP_SIZE / 2)
ANY_INDEX = 0
CONFIDENCE_INDEX = 5

USE_PERSONS = True
USE_PERSONS_THRESH = 15
FIRST_FRAME_USED = 18
FRAMES_PER_EVENT = 35
NUM_CLASSES = 3

UNIQUE_INDEX = 0
ball_close_to_person = True
ball_person_threshold = 0.065
y_camera_threshold = 0.75

def cropping():
    os.chdir('/content/drive/MyDrive')

    ! mkdir Bundesliga
    ! mkdir Bundesliga/PositiveFrames

    POSITIVES_FOLDER = '/content/drive/MyDrive/Bundesliga/PositiveFrames/'
    FOLDER_LABELS = POSITIVES_FOLDER + 'detect/'
    FOLDER_FRAMES = POSITIVES_FOLDER + 'frames_ms2/'

    os.chdir(POSITIVES_FOLDER)

    EVENTS_PER_VIDEO = {
        '1606b0e6_0': 396,
        '1606b0e6_1': 507,
        'cfbe2e94_0': 305,
        'cfbe2e94_1': 285,
        'ecf251d4_0': 386,
        '3c993bd2_0': 414,
        '3c993bd2_1': 376,
        '4ffd5986_0': 302,
        '9a97dae4_1': 291,
        '35bd9041_0': 411,
        '35bd9041_1': 359,
        '407c5a9e_1': 350,
        }

    videos = [video for video in EVENTS_PER_VIDEO.keys()]

    ! mkdir new_crops

    for video in videos:
        ! mkdir new_crops/$video

        for event in range(EVENTS_PER_VIDEO[video]):
            ball = False
            current_event_balls = list()

            substring = f'{FOLDER_LABELS}{video}/labels/{video}_{event}_'
            file = glob.glob(f'{substring}*.txt')[ANY_INDEX]

            len_subs = len(substring)
            class_ = int(file[len_subs])

            event_id = f'{video}_{event}_{class_}'

            for frame in range(FIRST_FRAME_USED, FRAMES_PER_EVENT - 2):
                frame_id = f'{event_id}_{frame}'
                print(frame_id)
                label_txt = f'{frame_id}.txt'
                print(f'{FOLDER_LABELS}{video}/labels/{label_txt}')
                with open(f'{FOLDER_LABELS}{video}/labels/{label_txt}') as f:
                    reader = f.read()
                detections = [i.split(' ') for i in reader.split('\n')][:-1]

                for detection in detections:

                    if detection[CLASS_INDEX] == BALL_CLASS:
                        ball = True

                        x = int(float(detection[X_INDEX]) * X_SIZE)
                        y = int(float(detection[Y_INDEX]) * Y_SIZE)
                        ymin = max(y - radius, 0)
                        ymax = min(y + radius, Y_SIZE)
                        xmin = max(x - radius, 0)
                        xmax = min(x + radius, X_SIZE)

                        if ymin == 0 or y > Y_SIZE * y_camera_threshold or xmin == 0 or xmax == X_SIZE:
                            ball = False

                        elif float(detection[CONFIDENCE_INDEX]) < 0.38:
                            ball = False

                        elif ball_close_to_person:
                            arr = np.array(detections)
                            persons = arr[arr[:, 0] == PERSON_CLASS]
                            xy_persons = persons[:, X_INDEX:Y_INDEX + 1].astype(float)
                            xy_ball = np.array(detection[X_INDEX:Y_INDEX + 1]).astype(float)
                            xy_closest_person = xy_persons[
                                spatial.KDTree(xy_persons).query(xy_ball)[1]]
                            distance, index = spatial.KDTree(xy_persons).query(xy_ball)
                            if distance > ball_person_threshold:
                                ball = False

                        if ball:
                            distance_middle_screen = Y_SIZE / 2 - y
                            current_event_balls.append([frame_id, ymin, ymax, xmin, xmax])

                            for r in range(frame - 3, frame):
                                img = cv2.imread(f'{FOLDER_FRAMES}{video}/{event_id}_{r}.jpg')
                                cropped_image = img[ymin:ymax, xmin:xmax]
                                cv2.imwrite(f'{POSITIVES_FOLDER}new_crops/{video}/{frame_id}_{r}cropped.jpg', cropped_image)
                            break

                if ball:
                    break

                elif frame == FRAMES_PER_EVENT - 3:
                    frame_id = f'{event_id}_{FIRST_FRAME_USED - 1}'
                    label_txt = f'{frame_id}.txt'
                    with open(f'{FOLDER_LABELS}{video}/labels/{label_txt}') as f:
                        reader = f.read()
                    detections = [i.split(' ') for i in reader.split('\n')][:-1]
                    arr = np.array(detections)
                    persons = arr[arr[:, 0] == '0']
                    xy_persons = persons[:, X_INDEX:Y_INDEX + 1].astype(float)
                    dists = cdist(xy_persons, xy_persons)
                    dists[dists == 0] = dists.max()
                    arg1, arg2 = np.unravel_index(dists.argmin(), dists.shape)
                    two_people = xy_persons[[arg1, arg2]]
                    frame_center = np.mean(two_people, axis=0)
                    x, y = frame_center
                    x = int(x * X_SIZE)
                    y = int(y * Y_SIZE)
                    ymin = max(y - radius, 0)
                    ymax = min(y + radius, Y_SIZE)
                    xmin = max(x - radius, 0)
                    xmax = min(x + radius, X_SIZE)

                    if ymin == 0:
                        ymax = CROP_SIZE
                    elif ymax == Y_SIZE:
                        ymin = Y_SIZE - CROP_SIZE
                    if xmin == 0:
                        xmax = CROP_SIZE
                    elif xmax == X_SIZE:
                        xmin = X_SIZE - CROP_SIZE

                    for r in range(FIRST_FRAME_USED - 1,
                                   FIRST_FRAME_USED + 2):
                        img = cv2.imread(f'{FOLDER_FRAMES}{video}/{event_id}_{r}.jpg')
                        cropped_image = img[ymin:ymax, xmin:xmax]
                        cv2.imwrite(f'{POSITIVES_FOLDER}new_crops/{video}/{frame_id}_{r}cropped.jpg', cropped_image)
                    break

if __name__ == '__main__':
    cropping()
