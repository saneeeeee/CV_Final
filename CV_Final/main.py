import math
import cv2

# 이미지를 넘겨주면 이미지 가로(image.shape[1]), 세로(shape[0])중 큰것을 300으로 맞추는 비율 찾아 사이즈 줄이고, 그 이미지 반환




def image_resize(img):
    percent = 1
    if (img.shape[1] > img.shape[0]):  # 이미지의 가로가 세보다 크면 가로를 300으로 맞추고 세로를 비율에 맞춰서
        percent = 300 / img.shape[1]
    else:
        percent = 300 / img.shape[0]

    img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)
    return img


def getAngle(a, b, c):
    Angle1 = c[0] - b[0]
    Angle2 = a[0] - b[0]

    if Angle1 == 0:
        Angle1 = 1

    if Angle2 == 0:
        Angle2 = 1

    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

    if ang < 0:
        ang = ang + 360
        if ang > 180:
            ang = 180 - (ang % 180)
            return int(ang)
        else:
            return int(ang)

    else:
        if ang > 180:
            ang = 180 - (ang % 180)
            return int(ang)
        else:
            return int(ang)


def output_keypoints(frame, net, threshold, BODY_PARTS, now_frame, total_frame):
    global points

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(image_resize(frame), 1.0 / 255, (image_width, image_height), (0, 0, 0),
                                       swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"============================== frame: {now_frame:.0f} / {total_frame:.0f} ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    return frame


def output_keypoints_with_lines(frame, POSE_PAIRS):
    global countNum
    pointL = 10, 40
    pointR = 10, 80
    pointS = 10, 120
    pointC = 10, 160

    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
    try:
        al = getAngle(points[8], points[9], points[10])
        ar = getAngle(points[11], points[12], points[13])
        angleL = 'angle L = ' + str(al)
        angleR = 'angle R = ' + str(ar)
        st = 'state = ' + str(state(al, ar))
        cnt = 'count = ' + str(countNum)

        prevL = al
        prevR = ar
        prevS = st
        prevC = cnt
    except Exception as e:
        cv2.putText(frame, prevL, pointL, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(frame, prevR, pointR, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(frame, prevS, pointS, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(frame, prevC, pointC, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
    else:

        cv2.putText(frame, angleL, pointL, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(frame, angleR, pointR, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(frame, st, pointS, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(frame, cnt, pointC, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_8)

    finally:
        return frame


def state(aL, aR):
    global status
    global countNum

    if aL < 100 and aR < 100 and status == 'Up':
        status = 'Sit'
    elif aL > 160 and aR > 160 and status == 'Sit':
        status = 'Up'
        countNum = countNum + 1

    return status


def output_keypoints_with_lines_video(proto_file, weights_file, video_path, threshold, BODY_PARTS, POSE_PAIRS):
    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(video_path)
    print(capture.get(cv2.CAP_PROP_FPS))
    while True:
        now_frame_boy = capture.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if now_frame_boy == total_frame_boy:
            break

        ret, frame_boy = capture.read()
        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold, BODY_PARTS=BODY_PARTS,
                                     now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        cv2.imshow("Output_Keypoints", frame_boy)

        if cv2.waitKey(10) == 27:  # esc 입력시 종료
            break

    capture.release()
    cv2.destroyAllWindows()



countNum = 0
status = 'Up'

if __name__ == "__main__":

    BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                      10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                      15: "Background"}

    POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                      [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

    # 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
    protoFile_mpi_faster = "fashion_data\\pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile_mpi = "fashion_data\\pose_iter_160000.caffemodel"
    # 비디오 경로
    man = "fashion_data/openpose07.mp4"

    # 키포인트를 저장할 빈 리스트
    points = []

    output_keypoints_with_lines_video(proto_file=protoFile_mpi_faster, weights_file=weightsFile_mpi, video_path=man,
                                      threshold=0.1, BODY_PARTS=BODY_PARTS_MPI, POSE_PAIRS=POSE_PAIRS_MPI)