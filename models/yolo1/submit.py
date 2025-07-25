import time
from torch.nn import Module
from config import Config, parse_args
from yolo1 import Yolo1
from utils import Utils


def main():
    print('Start generating csv submission file..')
    csv_file_name: str = 'submission.csv'
    start_time: float = time.time()
    # 1) 인자 파싱
    config_file: str = 'config/yolo_1.yaml'
    args = parse_args(config_file)
    # 2) Config 로드 + 덮어쓰기
    Config.load(config_file, vars(args))
    # 3) Train을 시작했을 때 파라미터를 가져오기
    Config.load(config_file, Utils.load_args())

    model: Module = Yolo1(Config.IMAGE_CH_SIZE, Config.S, Config.B, Config.C,
                          Config.CONF_THRESH, Config.NMS_IOU_THRESH).to(Config.DEVICE)
    Utils.load_model(model, 'model')


if __name__ == '__main__':
    main()
