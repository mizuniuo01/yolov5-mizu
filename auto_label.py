import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
import detect

if __name__ == '__main__':
    # 自动标注配置
    detect.run(
        # 小模型路径
        weights=Path('runs/train/exp/weights/best.pt'),                                     # 每次训练完成后都要修改这个路径，指向最新的best.pt
        
        # 还没打标签的图片路径
        source=Path('D:/Study/Datacollect/classified_dataset/all_picturedata/images/train'),                             # 每次新建一个数据集都要修改这个路径，指向需要自动标注的图片文件夹
        
        # 配置文件路径
        data=Path('D:/Study/Python/YOLOv5/yolov5/data/myconfigs/auto_aimdevice.yaml'),      # 每次新建一个数据集都要修改这个路径
        
        imgsz=(320, 320),       # 保持和训练时一样的分辨率
        conf_thres=0.5,         # 置信度阈值
        iou_thres=0.45,         # NMS重叠过滤阈值
        
        save_txt=True,          # 开启保存 TXT 标签文件（YOLO格式）
        save_conf=False,        # 标签中不要包含置信度
        nosave=True,           # False表示同时保存画好框的图片（方便你肉眼快速检查），True表示只存txt不存图
        
        # 输出路径设置
        project=Path('runs/auto_label'), # 自动标注的结果会保存在 runs/auto_label/exp 下
        name='exp',
        exist_ok=True           # 允许覆盖同名文件夹
    )
    
    print("\n 自动标注完成！")
    print("生成的标签文件在 runs/auto_label/exp/labels")