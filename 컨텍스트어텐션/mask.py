def brush_stroke_mask(FLAGS, name='mask'):
    test_num = 0;
    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """

    #Εδώ έβαλα μικρότερα τα max_width και min_width γιατί οι εικόνες 
    #όταν το τρέχω με 64X64Χ3 είναι πολύ μικρές για μία τέτοια μάσκα.

    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 5                     #Original 12
    max_width = 18                    #Original 40
    import numpy as np
    import random

    def generate_mask_circle(H, W, T):
        """
        바운딩 박스로부터 동그라미 모양의 마스크를 생성합니다.

        Args:
            H (int): 마스크의 높이
            W (int): 마스크의 너비

        Returns:
            np.array: 형상이 (1, H, W, 1)인 마스크 어레이
        """



        mask = np.zeros((1, H, W, 1), dtype=np.float32)  # 빈 마스크 생성


        # 원의 중심 좌표와 반지름 설정
        num = 10
        cx, cy = W // 2, H // 2
        if num > 0:
            if T % 5 == 0:
                num -=1
        else:
            num=8


        radius= min(W,H) // num


        # 원 그리기
        for y in range(H):
            for x in range(W):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    mask[:, y, x, :] = 1

        # 마스크의 형태 수정
        return mask


    # 입력 이미지의 크기를 가져옵니다.
    img_shape = FLAGS.img_shapes
    height = img_shape[0]  # 이미지의 높이
    width = img_shape[1]  # 이미지의 너비

    # 마스크 생성

    mask = generate_mask_circle(height, width, test_num)
    test_num +=1 # test 증가 횟수 증가할때 1 올려줌

    return mask  # 마스크 반환
