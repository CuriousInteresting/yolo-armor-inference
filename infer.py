from ultralytics import YOLO
import cv2
import os



# ==========================
# 路径配置
# ==========================

MODEL_PATH = "weights/best.pt"

VIDEO_PATH = "videos/original.mp4"

SAVE_PATH = "results/output.mp4"

CONF_THRESHOLD = 0.7


# ==========================
# 加载模型
# ==========================

model = YOLO(MODEL_PATH)


print("模型加载成功")

print("类别:")
print(model.names)



# ==========================
# 视频读取
# ==========================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    print("视频打开失败")

    exit()



fps = int(
    cap.get(cv2.CAP_PROP_FPS)
)


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)


height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)



print("================")
print("视频信息")

print("FPS:",fps)

print("尺寸:",width,height)



# ==========================
# 输出视频
# ==========================

os.makedirs(
    "results",
    exist_ok=True
)


fourcc=cv2.VideoWriter_fourcc(
    *"mp4v"
)


out=cv2.VideoWriter(

    SAVE_PATH,

    fourcc,

    fps,

    (width,height)

)



# ==========================
# 关键点名称
# ==========================

keypoint_names=[

    "LT",
    "LB",
    "RB",
    "RT"

]



# ==========================
# 主循环
# ==========================

while True:


    ret,frame=cap.read()


    if not ret:

        break



    # YOLO推理

    results=model(

        frame,

        imgsz=640,

        conf=CONF_THRESHOLD,

        verbose=False

    )


    result=results[0]



    # ==========================
    # 装甲板检测
    # ==========================

    if result.boxes is not None:


        boxes=result.boxes


        for i in range(len(boxes)):



            cls_id=int(
                boxes.cls[i]
            )


            conf=float(
                boxes.conf[i]
            )


            label=model.names[cls_id]



            # ======================
            # 颜色规则
            # ======================

            if label=="B3":

                # B3蓝色

                box_color=(255,0,0)

                text_color=(255,0,0)


            else:

                # 其他装甲板浅蓝色

                box_color=(255,200,100)

                text_color=(255,200,100)



            # 坐标

            xyxy=boxes.xyxy[i].cpu().numpy()


            x1,y1,x2,y2=map(

                int,

                xyxy

            )



            # 绘制框

            cv2.rectangle(

                frame,

                (x1,y1),

                (x2,y2),

                box_color,

                2

            )



            # 绘制文字

            cv2.putText(

                frame,

                f"{label} {conf:.2f}",

                (x1,y1-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                text_color,

                2

            )





    # ==========================
    # Pose关键点
    # ==========================


    if result.keypoints is not None:



        kpts=result.keypoints.xy.cpu().numpy()



        for pts in kpts:



            # 防止异常

            if len(pts)!=4:

                continue



            # 四个点

            for idx,p in enumerate(pts):


                x=int(p[0])

                y=int(p[1])



                # 黄色点

                cv2.circle(

                    frame,

                    (x,y),

                    5,

                    (0,255,255),

                    -1

                )



                # 黄色文字

                cv2.putText(

                    frame,

                    keypoint_names[idx],

                    (x+5,y-5),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0,255,255),

                    2

                )




            # ======================
            # 连接四个关键点
            # ======================

            lines=[

                (0,1),

                (1,2),

                (2,3),

                (3,0)

            ]


            for a,b in lines:


                cv2.line(

                    frame,

                    tuple(
                        pts[a].astype(int)
                    ),

                    tuple(
                        pts[b].astype(int)
                    ),

                    (0,255,255),

                    2

                )




    # 保存

    out.write(frame)



    cv2.imshow(

        "YOLO Armor",

        frame

    )



    if cv2.waitKey(1)&0xff==27:

        break





# ==========================
# 结束
# ==========================

cap.release()

out.release()

cv2.destroyAllWindows()


print("================")

print("推理完成")

print("保存位置:")

print(SAVE_PATH)