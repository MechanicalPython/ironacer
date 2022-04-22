from ironacer import find, stream
import time
import sys


def bench_yolo(weights, imgsz):
    # todo - give average and sd inference time,
    yolo = find.Detector(weights=weights, imgsz=int(imgsz))
    with stream.LoadCamera() as frames:
        for frame in frames:
            t = time.time()
            is_squirrel, results = yolo.inference(frame)  # results = [[x, y, x, y, motion],.. ]
            if results is None:
                continue
            print(time.time() - t)


if __name__ == '__main__':
    # takes exactly 2 args.
    filename, weight, imgsz = sys.argv
    bench_yolo(weight, imgsz)

