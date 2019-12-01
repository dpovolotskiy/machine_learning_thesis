from feature_extract_model import VGG16

vgg = VGG16(r"F:\My Documents\SSU\diplom\vgg16_weights_tf_dim_ordering_tf_kernels.h5")
model = vgg.get_model()
model.summary()