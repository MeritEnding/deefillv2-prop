import cv2
import numpy as np
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

from utils import *
from net import *
from config import Config

import tensorflow as tf
import glob

# 스케치할 이미지와 마우스 이벤트를 위한 변수 초기화
drawing = False
last_point = (-1, -1)
filename = None
mask_filename="../FINAL_TEST/mask_position.txt"

if os.path.exists(mask_filename):
    os.remove(mask_filename)

def save_point(point):
    with open(mask_filename,"a") as file:
        file.write(f"{point[0]},{point[1]}\n")

# 마우스 이벤트 콜백 함수
def draw(event, x, y, flags, param):
    global drawing, last_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
        save_point(last_point)


        update_coordinates(x*0.115,y*0.115)
        print(f'Current Conates: ({x},{y})')
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(image, last_point, (x, y), (128, 128, 128), 5)  # 회색으로 변경
            last_point = (x, y)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, last_point, (x, y), (128, 128, 128), 5)  # 회색으로 변경


# 파일 이름 카운터 초기화
counter = 1

# 저장 폴더 설정
save_folder = '../FINAL_TEST'
os.makedirs(save_folder, exist_ok=True)  # 폴더가 없으면 성

# 카메라 설정
cap = cv2.VideoCapture(0)

# 이미지 캡처 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 복사하여 스케치할 이미지 생성
    image = frame.copy()

    # 마우스 콜백 함수 등록
    cv2.namedWindow('Sketch')
    cv2.setMouseCallback('Sketch', draw)

    # 스케치 보여주기
    while True:
        cv2.imshow('Sketch', image)

        # 마우스 위치 변경



        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF

        # 'a' 키를 누르면 스케치 저장
        if key == ord('s'):  # 64개까지만 저장
            for i in range(1, 33):
                filename = os.path.join(save_folder, f"sketch{counter}.png")  # FINAL_TEST 폴더에 저장
                cv2.imwrite(filename, image)
                saved_image = cv2.imread(filename)  # 저장된 이미지 읽기
                cv2.imshow('Saved Sketch', saved_image)  # 저장된 이미지 보여주기
                print(f"Sketch saved as '{filename}'.")
                counter += 1  # 카운터 증가

        # 'i' 키를 누르면 인페인팅 수행
        if key == ord('i'):
            past_pattern= './images_examples/Test_Result/infer_test_example*.png'
            past_file_list = glob.glob(past_pattern)

            if not past_file_list:
                print("해당 패턴에 맞는 파일을 찾을 수 없습니다.")
            else:
                past_image_path = past_file_list[0]
                if os.path.exists(past_image_path):
                    os.remove(past_image_path)
                    print("이전 실험 파일이 삭제되었습니다.")
                else:
                    print("파일이 존재하지 않습니다.")

            # 여기 부터 인페인팅 동작
            generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
            discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

            FLAGS = Config('./inpaint.yml')

            generator = GeneratorMultiColumn()
            discriminator = Discriminator()

            test_dataset = tf.data.Dataset.list_files("../FINAL_TEST/*.png")
            test_dataset = test_dataset.map(load_image_train)
            test_dataset = test_dataset.batch(FLAGS.batch_size)
            test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            checkpoint_dir = "./training_checkpoints"
            checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                             generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=generator,
                                             discriminator=discriminator)
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
            checkpoint.restore(checkpoint_dir + '/' + 'ckpt-20')
            step = np.int(checkpoint.step)
            print("Continue Training from epoch ", step)

            # restore CSV
            df_load = pd.read_csv(f'./CSV_loss/loss_{step}.csv', delimiter=',')
            g_total = df_load['g_total'].values.tolist()
            g_total = CSV_reader(g_total)
            g_hinge = df_load['g_hinge'].values.tolist()
            g_hinge = CSV_reader(g_hinge)
            g_l1 = df_load['g_l1'].values.tolist()
            g_l1 = CSV_reader(g_l1)
            d = df_load['d'].values.tolist()
            d = CSV_reader(d)
            print(f'Loaded CSV for step: {step}')

            for data in test_dataset.take(50):
                generate_images(data, generator, training=False, num_epoch=step)

            plot_history(g_total, g_hinge, g_l1, d, step, training=False)

        if key ==ord('o'):
            pattern = './images_examples/Test_Result/infer_test_example*.png'
            file_list = glob.glob(pattern)

            if not file_list:
                print("해당 패턴에 맞는 파일을 찾을 수 없습니다.")
            else:
                image_path = file_list[0]
                final_image = cv2.imread(image_path)

                if final_image is None:
                    print("이미지를 불러올 수 없습니다. 파일경로를 확인해주세요.")
                else:
                    image_path = file_list[0]
                    final_image = cv2.imread(image_path)

                    if final_image is None:
                        print("이미지를 불러올 수 없습니다. 파일경로를 확인해주세요.")
                    else:
                        original_height, original_width = final_image.shape[:2]
                        target_size = 3000

                        if original_width > original_height:
                            new_width = target_size
                            new_height = int(original_height * (target_size / original_width))

                        else:
                            new_height = target_size
                            new_width = int(original_width * (target_size / original_height))

                        resized_image = cv2.resize(final_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                        start_input_x = (new_width - 2300) // 2
                        start_input_y = (new_height - 300) // 2
                        end_input_x = start_input_x + 300
                        end_input_y = start_input_y + 300

                        start_output_x = (new_width + 665) // 2
                        start_output_y = (new_height - 300) // 2
                        end_output_x = start_output_x + 300
                        end_output_y = start_output_y + 300

                        start_mask_x = (new_width - 1700) // 2
                        start_mask_y = (new_height - 300) // 2
                        end_mask_x = start_mask_x + 300
                        end_mask_y = start_mask_y + 300

                        cropped_input_image = resized_image[start_input_y:end_input_y, start_input_x:end_input_x]
                        cropped_output_image = resized_image[start_output_y:end_output_y, start_output_x:end_output_x]
                        cropped_mask_image = resized_image[start_mask_y:end_mask_y, start_mask_x:end_mask_x]

                        combined_image=np.hstack((cropped_input_image,cropped_mask_image,cropped_output_image))

                        cv2.imshow("Prop Result", combined_image)
                        cv2.imwrite("../FINAL_TEST/Result_Path/Prop_Result.png",combined_image)
                        # cv2.imshow("Prop Input Image", cropped_input_image)
                        # cv2.imshow("Prop Mask Image", cropped_mask_image)
                        # cv2.imshow("Prop InPainted OutPut Image", cropped_output_image)

                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

        # if key == ord('t'):
        #     image_path= './images_examples/Test_Result/*.png'
        #     image = cv2.imread(image_path)
        #
        #     if image is None:
        #         print("이미지를 불러올수없다")
        #     else:
        #         cv2.imshow('Load', image)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()


        # 'q' 키를 누르면 종료
        if key == ord('q'):
            break

    # 모든 창 닫기
    cv2.destroyAllWindows()
    break  # 프로그램 종료

# 카메라 해제
cap.release()
cv2.destroyAllWindows()
