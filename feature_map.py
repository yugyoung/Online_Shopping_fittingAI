'''import torch
import cv2

tmp_model.eval()
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(tmp_model, dummy_input, "crnn.onnx", verbose=True)


net= cv2.dnn.readNet("crnn.onnx")'''

import cv2
from PIL import Image

def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    image_height = 368
    image_width = 368

    
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    
    net.setInput(input_blob)

    
    out = net.forward()
    out_height = out.shape[2]
    out_width = out.shape[3]

    frame_height, frame_width = frame.shape[:2]

    points = []

    print(f"\n============================== {model_name} Model ==============================")
    for i in range(len(BODY_PARTS)):

        prob_map = out[0, i, :, :]

        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    cv2.imshow("Output_Keypoints", frame)
    #cv2.waitKey(0)
    return frame
    
def output_keypoints_with_lines(frame, POSE_PAIRS):
    print()
    for pair in POSE_PAIRS:
        part_a = pair[0]  
        part_b = pair[1]  
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    return frame
    #cv2.imshow("output_keypoints_with_lines", frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}

POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                  [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]


protoFile_mpi = "/data0/yugyoung/VTON/pose_deploy_linevec.prototxt"
protoFile_mpi_faster = "/data0/yugyoung/VTON/pose_deploy_linevec_faster_4_stages.prototxt"
protoFile_coco = "/data0/yugyoung/VTON/pose_deploy_linevec.prototxt"
protoFile_body_25 = "/data0/yugyoung/VTON/pose_deploy.prototxt"


weightsFile_mpi = "/data0/yugyoung/VTON/pose_iter_160000.caffemodel"
weightsFile_coco = "/data0/yugyoung/VTON/pose_iter_440000.caffemodel"
weightsFile_body_25 = "/data0/yugyoung/VTON/pose_iter_584000.caffemodel"


man = "/data0/yugyoung/VTON/02_4_full.jpg"


points = []


frame_mpii = cv2.imread(man)
frame_coco = frame_mpii.copy()
frame_body_25 = frame_mpii.copy()


frame_MPII = output_keypoints(frame=frame_mpii, proto_file=protoFile_mpi_faster, weights_file=weightsFile_mpi,
                             threshold=0.2, model_name="MPII", BODY_PARTS=BODY_PARTS_MPI)
cv2.imwrite('02_4_full_pose_MPII.jpg', frame_MPII)
#frame.save('02_4_full_pose.jpg')
lines = output_keypoints_with_lines(frame=frame_MPII, POSE_PAIRS=POSE_PAIRS_MPI)
cv2.imwrite('02_4_full_pose_MPII_lines.jpg', lines)


frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                             threshold=0.2, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)
output_keypoints_with_lines(frame=frame_COCO, POSE_PAIRS=POSE_PAIRS_COCO)


frame_BODY_25 = output_keypoints(frame=frame_body_25, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                             threshold=0.2, model_name="BODY_25", BODY_PARTS=BODY_PARTS_BODY_25)
output_keypoints_with_lines(frame=frame_BODY_25, POSE_PAIRS=POSE_PAIRS_BODY_25)