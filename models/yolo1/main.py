from yolo1 import Yolo1
from config import Config, parse_args


def main():
    # 1) 인자 파싱
    config_file: str = 'config/yolo_1.yaml'
    args = parse_args(config_file)

    # 2) Config 로드 + 덮어쓰기
    Config.load(config_file, vars(args))


    model = Yolo1()
    # print(model)


if __name__ == '__main__':
    main()
