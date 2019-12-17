import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from training import start_fit_model, start_lite_train
from captions_generator import get_predict
from text_data_preparation import prepare_text_data
from feture_extraction import extracting_features_from_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Enter mode training or predict",
                        type=str)
    parser.add_argument("--filepath", help="Enter path to image", default=None,
                        type=str)
    parser.add_argument("--lite", help="Using lite fit method", default=False,
                        type=bool)
    parser.add_argument("--force", help="Rewrite old file with trained "
                                        "parameters", default=False,
                        type=bool)
    parser.add_argument("--modelpath", help="Enter path to model", default=None,
                        type=str)
    args = parser.parse_args()
    if args.mode == "training":
        if args.force:
            prepare_text_data()
            extracting_features_from_image("Flickr8k_Dataset/Flicker8k_Dataset")
            if args.lite:
                start_lite_train(20)
            else:
                start_fit_model(20)
        else:
            if not os.path.exists("captions.txt"):
                prepare_text_data()
            if not os.path.exists("features.pkl"):
                extracting_features_from_image(
                    "Flickr8k_Dataset/Flicker8k_Dataset")
            if args.lite:
                start_lite_train(20)
            else:
                start_fit_model(20)
    elif args.mode == "predict":
        if args.filepath is not None and os.path.exists(args.filepath):
            get_predict(args.filepath, args.modelpath)
        else:
            print("Specified file was not found")
