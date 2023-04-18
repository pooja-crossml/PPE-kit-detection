# Object Detecion 
import cv2
from ultralytics import YOLO

#basics
import pandas as pd
import numpy as np
import os
# import subprocess
import time
from tqdm.notebook import tqdm

# Display image and videos
import IPython
from IPython.display import Video, display
# import threading
# from threading import Thread
# from multiprocessing import Queue
# %matplotlib inline

# '/home/crossml/Work/cross-ml/Pooja_research/video_dataset/vid_data/Pexels Videos 2048246.mp4'
# '/home/crossml/Work/cross-ml/Pooja_research/video_dataset/construction/pexels-akin-victor-10294768.mp4'
path = '/home/pooja/projects/inhouse/projects/construction_videos/production ID_4271760.mp4'
# path = '/home/pooja/projects/inhouse/projects/construction_videos/production ID_4271760.mp4'
#loading a YOLO model
# model = YOLO("/home/pooja/projects/inhouse/projects/tracking/deep_sort/yolov8n.pt")
model = YOLO("/home/pooja/projects/inhouse/projects/tracking/deep_sort/yolov8s.pt")
custom_model = YOLO("/home/pooja/projects/inhouse/projects/tracking/deep_sort/weights/best.pt")

#geting names from classes
dict_classes = model.model.names
# dict_classes = {0:'person'}


# Auxiliary functions
def risize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    dim = (640, 480)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized


def filter_tracks(centers, patience):
    """Function to filter track history"""
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-patience:])

    return filter_dict


def update_tracking(centers_old,obj_center, thr_centers, lastKey, frame, frame_max):
    """Function to update track of objects"""
    is_new = 0
    # breakpoint()
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
    # Calculating distance from existing centers points
    previous_pos = [(k,obj_center) for k,centers in lastpos if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
    # if distance less than a threshold, it will update its positions
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center
    
    # Else a new ID will be set to the given object
    else:
        if lastKey:
            last = lastKey.split('D')[1]
            id_obj = 'ID' + str(int(last)+1)
        else:
            id_obj = 'ID1'
            
        is_new = 1
        centers_old[id_obj] = {frame:obj_center}
        lastKey = list(centers_old.keys())[-1]

    
    return centers_old, id_obj, is_new, lastKey


def check_safety(labels):
    text = ''
    if labels:
        if 4.0 and 0.0 in labels:
            text = "Safe"
            return text
        elif 4.0 or 0.0 in labels:
            text = "Partial safe"
        else:
            text = "Unsafe"
    return text


# detect ppe kit
# def detect_ppe(image, bbox_list):
def detect_ppe(image, orig_img, bounding_box, person_id):
    # print(person_id)
    alpha = 0.2
    overlay = orig_img.copy()
    # cv2.imshow("image", image)
    # # cv2.waitKey(0)
    image = np.ascontiguousarray(image)
    model2_result = custom_model.predict(source=image, classes=[0,4], line_thickness=2)
    # breakpoint()    
    labels=model2_result[0].boxes.cls.tolist()
    safety_value=''
    if labels:
        safety_value = check_safety(labels)
    # print(model2_result[0].boxes.cls, model2_result[0].boxes.xyxy)
    res=[]
    # l = {0:'HARDHAT', 1:'NOSAFETYJACKET', 2:'NOHARDHAT', 4:'SAFETYJACKET'}
    lbl = {0:'HARDHAT', 4:'SAFETYJACKET'}
    # for id, r in enumerate(model2_result):
    boxes=model2_result[0].boxes
    # print(person_id, len(bounding_box), len(boxes))
    for box_id,box in enumerate(boxes):
        x1 = int(boxes[box_id].xyxy[0][0])
        y1 = int(boxes[box_id].xyxy[0][1])
        x2 = int(boxes[box_id].xyxy[0][2])
        y2 = int(boxes[box_id].xyxy[0][3])
        label = int(boxes[box_id].cls)
        # area=[x1,y1,x2,y2]
        
        # breakpoint()
        res.append([x1,y1,x2,y2, label])
        # for b in bounding_box:
        bx1,by1,bx2,by2 = bounding_box.astype('int')
        text = lbl.get(label)
        text_color = (255,255,255)
        background_color = (153, 153, 0) if text=='SAFETYJACKET' else (255, 204, 204)

        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        box_coords = ((x1+bx1,  y1+by1), (x1+bx1+text_width-10, y1+by1-text_height))
        # cv2.addWeighted(orig_img, 0.1, frame_copy , 1 - alpha, 0)
        #cv2.rectangle(orig_img, box_coords[0], box_coords[1], background_color, cv2.FILLED) #box for text background
        # cv2.putText(orig_img, f'person {person_id}',(x1+10, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
        if safety_value == ("Partial safe" or "Unsafe"):
            
            # orig_img = cv2.addWeighted(overlay, alpha,frame_copy , 1 - alpha, 0)
            # print(safety_value)
            # cv2.putText(orig_img, f'{person_id} is {safety_value}', cord, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(overlay,(bx1,by1),(bx2,by2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0, orig_img)
        else:
            # 194, 222, 146   74, 115, 5
            cv2.rectangle(overlay,(bx1,by1),(bx2,by2), (12, 185, 43), -1)
            cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0, orig_img)
        
        # cv2.circle(orig_img, (542, 45), 5, (0, 0, 255), -1)
        # cv2.circle(orig_img, (542, 75), 5, (0, 255, 0), -1)
        # cv2.putText(orig_img, "Safe", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(orig_img, "Unsafe", (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        # cv2.putText(orig_img, text, (x1+bx1+5, y1+by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
        # cv2.rectangle(orig_img, (x1+bx1, y1+by1), (x2+bx1, y2+by1), background_color, 2)
        # cv2.rectangle(overlay, (bx1,by1),(bx2,by2), (0, 155, 100), cv2.FILLED)
 
    return orig_img

class MLThread:
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self


    def update(self):
		# keep looping infinitely
        while True:
			# if the thread indicator variable is set, stop the
			# thread
            if self.stopped:
                return
			# otherwise, ensure the queue has room in it
            if not self.Q.full():
				# read the next frame from the file
                (grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
				# add the frame to the queue
                self.Q.put(frame)
    
    def read(self):
		# return next frame in the queue
        return self.Q.get()

    def more(self):
		# return True if there are still frames in the queue
        return self.Q.qsize() > 0




### Configurations
#Verbose during prediction
verbose = False
# Scaling percentage of original frame
scale_percent = 100
# model confidence level
# conf_level = 0.4
# Threshold of centers ( old\new)
thr_centers = 20
#Number of max frames to consider a object lost 
frame_max = 5
# Number of max tracked centers stored 
patience = 100
# ROI area color transparency
alpha = 0.1
#-------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(path)
stopped = False
# Q = Queue(maxsize=queueSize)

# Objects to detect Yolo
class_IDS = [0] 
# Auxiliary variables
centers_old = {}

obj_id = 0 
end = []
frames_list = []
count_p = 0
lastKey = ''
print(f'[INFO] - Verbose during Prediction: {verbose}')


# Original informations of video
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)
print('[INFO] - Original Dim: ', (640, 480))

# setting frames per seconds

# time_length = 30.0
# # fps=25
# frame_seq = 749
# frame_no = (frame_seq /(time_length*fps))

# Scaling Video for better performance 
if scale_percent != 100:
    print('[INFO] - Scaling change may cause errors in pixels lines ')
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    print('[INFO] - Dim Scaled: ', (width, height))
    

#-------------------------------------------------------
### Video output ####
video_name = 'result.mp4'
output_path = "rep_" + video_name
tmp_output_path = "tmp_" + output_path
# VIDEO_CODEC = "MP4"

output_video = cv2.VideoWriter(tmp_output_path, 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               fps, (width, height))




prev_timestamp = 0

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# print("Total frames in video:", total_frames)
# breakpoint()
ptime = 0
#-------------------------------------------------------
# Executing Recognition 
for i in tqdm(range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 3)):
    # video.set(2,frame_no)


    # reading frame from video
    _, frame = video.read()
    # print(cv2.CAP_PROP_FRAME_COUNT)


     # check if frame is at least 1 second apart from previous frame
    # curr_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
    # if curr_timestamp - prev_timestamp < 1:
    #     continue
    # prev_timestamp = curr_timestamp

    ctime = time.time()
    nfps = 1/(ctime-ptime)
    ptime = ctime

    #Applying resizing of read frame
    frame  = risize_frame(frame, scale_percent)
    frame_copy = frame.copy()
#     frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # area_roi = [np.array([ (1250, 300),(450,300),(400,900) ,(1200,900)], np.int32)]
    # ROI = frame[200:1000, 400:1400]
    # cv2.imshow("ROI", ROI)
    # cv2.imshow("ROI2", frame_copy)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  
    # if verbose:
    #     print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

    # Getting predictions
    y_hat = model.predict(frame_copy, classes = class_IDS)
    
    # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
    boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
    conf    = y_hat[0].boxes.conf.cpu().numpy()
    classes = y_hat[0].boxes.cls.cpu().numpy() 

    # Storing the above information in a dataframe
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    
    #Translating the numeric class labels to text
    labels = [dict_classes[i] for i in classes]
    

    # For each people, draw the bounding-box and counting each one the pass thought the ROI area
    for ix, row in enumerate(positions_frame.iterrows()):
        # Getting the coordinates of each vehicle (row)
        xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')
        
        # Calculating the center of the bounding-box
        center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
        #Updating the tracking for each object
        centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, (center_x, center_y), thr_centers, lastKey, i, frame_max)
        # breakpoint()

        #Updating people in roi
        count_p+=is_new

        # drawing center and bounding-box in the given frame
        # cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0,0,255), 2) # box
        # for center_x,center_y in centers_old[id_obj].values():
        #     cv2.circle(ROI, (center_x,center_y), 5,(0,0,255),-1) # center of box
        

        #Drawing above the bounding-box the name of class recognized.
        # cv2.putText(img=frame_copy, text='person '+id_obj,
        #         org= (xmin,ymin-20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),thickness=1)
        


                 # detect ppe kit for every person
        # breakpoint()
        frame_copy2 = frame.copy()

        for value_id, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox.astype(int)

            person_image = frame[int(y1):int(y2), int(x1):int(x2)]
            frame_copy= detect_ppe(person_image, frame_copy, bbox, id_obj)
        
           
    # Filtering tracks history
    centers_old = filter_tracks(centers_old, patience)
    if verbose:
        print(contador_in, contador_out)
    
    # #Drawing the ROI area
    # overlay = frame_copy.copy()
  
    # cv2.polylines(overlay, pts = area_roi, isClosed = True, color=(255, 0, 0),thickness=2)
    # cv2.fillPoly(overlay, area_roi, (255,0,0))
    # frame = cv2.addWeighted(overlay, alpha,frame_copy , 1 - alpha, 0)

    #Saving frames in a list 
    frames_list.append(frame_copy)
    cv2.circle(frame_copy, (540, 45), 5, (0, 0, 255), -1)
    cv2.circle(frame_copy, (540, 75), 5, (0, 255, 0), -1)
    cv2.putText(frame_copy, "Unsafe", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_copy, "Safe", (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.putText(frame_copy, f"{str(int(nfps))}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("hgvhasdf", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #saving transformed frames in a output video formaat
    output_video.write(frame_copy)


        
#Releasing the video    
output_video.release()
cv2.destroyAllWindows()

####  pos processing
# Fixing video output codec to run in the notebook\browser
if os.path.exists(output_path):
    os.remove(output_path)
    
# subprocess.run(
#     ["ffmpeg",  "-i", tmp_output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])
# os.remove(tmp_output_path)









# # from ultralytics import YOLO

# # # Load a model
# # model = YOLO('/home/pooja/projects/inhouse/projects/tracking/deep_sort/yolov8n.onnx')  # load a pretrained model (recommended for training) # build from YAML and transfer weights

# # # Predict with the model
# # results = model('/home/pooja/projects/inhouse/projects/construction_videos/production ID_4271760.mp4', show=True)  # predict on an image
# # breakpoint()
# # print(results)

