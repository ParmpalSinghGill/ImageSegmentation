import random,os,cv2
from preapereData import get_balloon_dicts
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

dataset_dicts = get_balloon_dicts("balloon/train")
balloon_metadata = MetadataCatalog.get("balloon_train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Img",out.get_image()[:, :, ::-1])
    cv2.waitKey(5000)