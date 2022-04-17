from ironacer import find, stream
import time


def bench_yolo():

    yolo = find.Detector(weights='./best.pt', imgsz=640)
    with stream.LoadCamera() as frames:
        for frame in frames:
            t = time.time()
            is_squirrel, results = yolo.inference(frame)  # results = [[x, y, x, y, motion],.. ]
            if results is None:
                continue
            print(time.time() - t)


if __name__ == '__main__':
    bench_yolo()