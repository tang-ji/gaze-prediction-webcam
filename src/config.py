img_size = (64, 64)
eye_size = (64, 32)
face_size = (64, 64)
mask_size = (64, 64)
n_channel = 1

# CNN structure for eyes
conv1_eye_size = 7
conv1_eye_out = 64
conv1_eye_strides = 2
pool1_eye_size = 2
pool1_eye_stride = 2

conv2_eye_size = 5
conv2_eye_out = 64
conv2_eye_strides = 1
pool2_eye_size = 2
pool2_eye_stride = 2

# CNN structure for face
conv1_face_size = 7
conv1_face_out = 64
conv1_face_strides = 1
pool1_face_size = 2
pool1_face_stride = 2

conv2_face_size = 5
conv2_face_out = 64
conv2_face_strides = 2
pool2_face_size = 2
pool2_face_stride = 2

conv3_face_size = 3
conv3_face_out = 64
pool3_face_size = 2
pool3_face_stride = 2

# CNN structure for mask
conv1_mask_size = 5
conv1_mask_out = 32
conv1_mask_strides = 2
pool1_mask_size = 2
pool1_mask_stride = 2

conv2_mask_size = 5
conv2_mask_out = 32
conv2_mask_strides = 1
pool2_mask_size = 2
pool2_mask_stride = 2

conv3_mask_size = 3
conv3_mask_out = 32
pool3_mask_size = 2
pool3_mask_stride = 2

fc_eye_size = 128
fc_face_size = 128
fc2_face_size = 64
fc_mask_size = 128
fc_size = 128
fc2_size = 2