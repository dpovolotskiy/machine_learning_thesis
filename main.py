import argparse
import os
import warnings

from training import start_fit_model, start_lite_train
from captions_generator import get_predict
from text_data_preparation import prepare_text_data
from feture_extraction import extracting_features_from_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """
    функция используется для извлечения параметров и флагов, который были
    переданны программе в коммандной строке при старте, и в соответствии с
    данными параметрами выполняет определенный набор действий:
    запускает обучение (обычное или облегчённое, с перезаписью файлов, от
    выполненых ранее обучений, или их сохранением);
    выполняет описание изображения
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        help="Укажите режим работы: тренировка (training) "
                             "или создание описания (predict)",
                        type=str)
    parser.add_argument("--filepath",
                        help="Укажите путь до изображения",
                        default=None,
                        type=str)
    parser.add_argument("--lite",
                        help="Укажите True для использования облегченного "
                             "режима обучения",
                        default=False,
                        type=bool)
    parser.add_argument("--force",
                        help="Укажите True, чтобы удалить все файлы обучения "
                             "созданные ранее",
                        default=False,
                        type=bool)
    parser.add_argument("--modelpath",
                        help="Укажите путь к предобученной модели",
                        default=None,
                        type=str)
    args = parser.parse_args()
    if args.mode == "training":
        if args.force:
            prepare_text_data()
            extracting_features_from_image("Flickr8k_Dataset/Flicker8k_Dataset")
            if args.lite:
                start_lite_train(10)
            else:
                start_fit_model(10)
        else:
            if not os.path.exists("captions.txt"):
                prepare_text_data()
            if not os.path.exists("features.pkl"):
                extracting_features_from_image(
                    "Flickr8k_Dataset/Flicker8k_Dataset")
            if args.lite:
                start_lite_train(10)
            else:
                start_fit_model(10)
    elif args.mode == "predict":
        if (args.filepath is not None and os.path.exists(args.filepath)) and \
                (args.modelpath is not None and os.path.exists(args.modelpath)):
            get_predict(args.filepath, args.modelpath)
        else:
            print("Указанные пути не найдены!")
