
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import glob

from utils import *
from net import *
from config import Config

pattern = './images_examples/Test_Result/infer_test_example*.png'
file_list = glob.glob(pattern)

if file_list:
    for file_path in file_list:
        os.remove(file_path)
    print("기존 TEST 사진을 삭제했습니다.")

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

FLAGS = Config('./inpaint.yml')

generator = GeneratorMultiColumn()
discriminator = Discriminator()

test_dataset = tf.data.Dataset.list_files("../FINAL_TEST/*.png")
test_dataset = test_dataset.map(load_image_train)
test_dataset = test_dataset.batch(FLAGS.batch_size)
test_dataset = test_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

checkpoint_dir = 'C:\\Users\\user\\PycharmProjects\\DeepFill\\DeepFillv2-TF2-org\\training_checkpoints'
checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
checkpoint.restore(checkpoint_dir+'/'+'ckpt-20')
step = np.int(checkpoint.step)
print("Continue Training from epoch ", step)

#restore CSV
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

for data in test_dataset.take(15):
  generate_images(data, generator, training=False, num_epoch=step)

plot_history(g_total, g_hinge, g_l1, d, step, training=False)




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
            target_size = 2300

            if original_width > original_height:
                new_width = target_size
                new_height = int(original_height * (target_size / original_width))

            else:
                new_height = target_size
                new_width = int(original_width * (target_size / original_height))

            resized_image = cv2.resize(final_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            start_input_x = (new_width - 1765) // 2
            start_input_y = (new_height - 300) // 2
            end_input_x = start_input_x + 300
            end_input_y = start_input_y + 300

            start_output_x = (new_width + 670) // 2
            start_output_y = (new_height - 300) // 2
            end_output_x = start_output_x + 300
            end_output_y = start_output_y + 300

            start_mask_x = (new_width - 1150) // 2
            start_mask_y = (new_height - 300) // 2
            end_mask_x = start_mask_x + 300
            end_mask_y = start_mask_y + 300

            cropped_input_image = resized_image[start_input_y:end_input_y, start_input_x:end_input_x]
            cropped_output_image = resized_image[start_output_y:end_output_y, start_output_x:end_output_x]
            cropped_mask_image = resized_image[start_mask_y:end_mask_y, start_mask_x:end_mask_x]

            combined_image = np.hstack((cropped_input_image,cropped_mask_image,cropped_output_image))

            # cv2.imshow("ORG InPut Image", cropped_input_image)
            # cv2.imshow("ORG Mask Image", cropped_mask_image)
            # cv2.imshow("ORG InPainted OutPut Image", cropped_output_image)
            cv2.imshow("ORG Result", combined_image)
            cv2.imwrite("../FINAL_TEST/Result_Path/Org_Result.png",combined_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()