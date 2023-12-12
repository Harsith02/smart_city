from ultralytics import YOLO

model = YOLO('yolov8m.pt')

def main():
    model.train(data=r"C:\Users\Sanjay\PycharmProjects\smart_city\Dataset\mydata\mydata128.yaml", epochs=10)

if __name__ == '__main__':
    main()
