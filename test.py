import cityflow
import json

def test():
    # a = json.load(open("../examples/config_1x1.json"))
    # print(a)
    eng = cityflow.Engine("examples/config_1x1.json", thread_num=1)
    a = eng.get_lane_vehicle_count()
    b = a

if __name__ == '__main__':
    test()
