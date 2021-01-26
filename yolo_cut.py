def draw_boxes(image, boxes, scores, labels, classes, detection_size, img):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    if boxes is None: 
        #print "NONE"
        return image, None
    answ_scor = []
    cord = []
    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" %(label, score)
        
        coord = [abs(int(x)) for x in bbox]
        
        o0 = coord[0]*(img.shape[1] / 416.0)
        o1 = coord[1]*(img.shape[0] / 416.0)
        o2 = coord[2]*(img.shape[1] / 416.0)
        o3 = coord[3]*(img.shape[0] / 416.0)
        #print (box.shape, s, ratio, coord, bbox,o0, o1, o2, o3)
        if "person" == classes[labels[i]].split("\n")[0]:# or "bus" == classes[labels[i]]:
                img_t = np.array(img[int(o1):int(o3), int(o0):int(o2), :])
                #print (img_t.shape)
                nameFile = str(uuid.uuid4())[:5]
                cv2.imwrite(f'temp_people/{nameFile}.jpg', img_t)
                #img_resized0 = cv2.resize(img_t, (128, 64))
def gg(img):
                img_resized0 = cv2.resize(img, (size, size))
                img_resized = np.reshape(img_resized0, [1, 416, 416, 3])
                I, B, C = sess.run([inputs, boxes, cl_de], feed_dict={inputs: img_resized})
                boxeS, scores, labels = cpu_nms(B, C, len(classes), max_boxes=3000, score_thresh=0.1, iou_thresh=0.2) #Bike
                return draw_boxes(img_resized0, boxeS, scores, labels, classes, 416, img)




if __name__ == "__main__":
    cap = cv2.VideoCapture('IMG_3065.MOV')
    IDX = 0
    while cap.isOpened():
            ret, frame = cap.read()
            P = gg(frame)
    cap.release()
    cv2.destroyAllWindows()
