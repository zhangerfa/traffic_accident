import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_trace_to_csv(trace_dict, output):
    """
    从视频中提取轨迹并保存到csv文件
    frame_count: 提取的帧数，如果为None则提取整个视频
    """
    fieldnames = ['obj_id', 'trace_time', 'class_id', 'x', 'y', 'w', 'h', 'angle']

    with open(output, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 写入列名（表头）
        writer.writeheader()
        # 写入数据
        for obj_id, trace_ls in trace_dict.items():
            for trace in trace_ls:
                box = trace[0]
                row = {
                    'obj_id': obj_id,
                    'trace_time': trace[1],
                    'class_id': trace[2],
                    'x': box[0],
                    'y': box[1],
                    'w': box[2],
                    'h': box[3],
                    'angle': box[4]
                }
                writer.writerow(row)
    logger.info(f"轨迹已保存到{output}")

def load_trace_from_csv(csv_path):
    """
    从csv文件中加载轨迹
    """
    trace_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = row['obj_id']
            trace_time = row['trace_time']
            class_id = row['class_id']
            box = [float(row['x']), float(row['y']), float(row['w']), float(row['h']), float(row['angle'])]
            if obj_id not in trace_dict:
                trace_dict[obj_id] = []
            trace_dict[obj_id].append([box, trace_time, class_id])
    logger.info(f"轨迹已从{csv_path}加载")
    return trace_dict