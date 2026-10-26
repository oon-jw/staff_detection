
import cv2
import os
import csv
from collections import defaultdict, deque
from ultralytics import YOLO

# ================= 配置参数 =================
VIDEO_PATH = "sample.mp4"
OUTPUT_VIDEO = "outputs/result.mp4"
OUTPUT_CSV = "outputs/roi_coordinates.csv"
MATCH_CSV = "outputs/badge_candidates.csv"

ROI_DIR = "outputs/chest_roi"
ROI_RESIZED_DIR = "outputs/chest_roi_resized"

# 工牌模板
BADGE_TEMPLATES = [
    "templates/badge_white.jpg",
    "templates/badge_black.jpg",
]

IMGSZ = 1280
YOLO_CONF = 0.15
TRACKER_CONFIG = "bytetrack.yaml"

# 胸前 ROI 范围
ROI_X_MIN = 0.20
ROI_X_MAX = 0.80
ROI_Y_MIN = 0.20
ROI_Y_MAX = 0.55

# ROI 保存尺寸
RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128

# 模板缩放比例
MATCH_SCALES = [0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35]

# 橙色工牌候选框显示门槛
# 0.0 = 显示所有最佳候选，不代表确认是工牌
DISPLAY_THRESHOLD = 0.0

# ================= 目标确认参数 =================
BADGE_THRESHOLD = 0.75

# 最近 n 次有效检测中，至少 m 次达到门槛
VOTE_WINDOW = 5
VOTE_REQUIRED = 2

# 人工确认是目标人物的 track_id
TARGET_TRACK_IDS = {195, 321}

# 已知误判的 track_id
IGNORE_TRACK_IDS = {5}

# 每个 track_id 保存最近的匹配结果
track_votes = defaultdict(
    lambda: deque(maxlen=VOTE_WINDOW)
)

# 已经确认的目标 track_id
confirmed_track_ids = set()

os.makedirs("outputs", exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(ROI_RESIZED_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")


# ================= 胸前 ROI =================
def crop_chest_roi(frame, box):
    x1, y1, x2, y2 = map(int, box)

    h, w = frame.shape[:2]
    person_w = x2 - x1
    person_h = y2 - y1

    if person_w <= 0 or person_h <= 0:
        return None, None

    rx1 = max(0, int(x1 + ROI_X_MIN * person_w))
    rx2 = min(w, int(x1 + ROI_X_MAX * person_w))
    ry1 = max(0, int(y1 + ROI_Y_MIN * person_h))
    ry2 = min(h, int(y1 + ROI_Y_MAX * person_h))

    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    roi = frame[ry1:ry2, rx1:rx2].copy()

    return roi, (rx1, ry1, rx2, ry2)


# ================= 加载工牌模板 =================
def load_badge_templates():
    templates = []

    for path in BADGE_TEMPLATES:
        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(
                f"无法读取工牌模板: {path}"
            )

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        templates.append({
            "name": os.path.basename(path),
            "gray": gray
        })

        print(
            f"已加载模板: {path}, "
            f"尺寸={gray.shape[1]}x{gray.shape[0]}"
        )

    return templates


# ================= 工牌相似度检测 =================
def detect_badge_similarity(roi, templates):
    """
    在胸前 ROI 内进行多尺度模板匹配。

    返回：
        score: 匹配分数，不是概率
        candidate_box: ROI 内候选框 (x, y, w, h)
        template_name: 最匹配的模板名称
    """
    if roi is None or roi.size == 0:
        return None, None, None

    roi_gray = cv2.cvtColor(
        roi, cv2.COLOR_BGR2GRAY
    )

    roi_h, roi_w = roi_gray.shape

    best_score = -2.0
    best_box = None
    best_name = None

    for template in templates:
        base_gray = template["gray"]

        for scale in MATCH_SCALES:
            tw = max(
                3, round(base_gray.shape[1] * scale)
            )
            th = max(
                3, round(base_gray.shape[0] * scale)
            )

            if tw > roi_w or th > roi_h:
                continue

            temp_gray = cv2.resize(
                base_gray,
                (tw, th),
                interpolation=cv2.INTER_CUBIC
            )

            result = cv2.matchTemplate(
                roi_gray,
                temp_gray,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(
                result
            )

            if max_val > best_score:
                best_score = float(max_val)
                best_box = (
                    max_loc[0],
                    max_loc[1],
                    tw,
                    th
                )
                best_name = template["name"]

    if best_box is None:
        return None, None, None

    return best_score, best_box, best_name


# ================= 更新目标确认状态 =================
def update_target_status(track_id, score):
    """
    判断当前人物是否为目标。

    优先级：
    1. 无效 ID：不确认
    2. 黑名单：不确认
    3. 人工确认白名单：直接确认
    4. 其他 ID：按模板匹配分数投票
    """
    if track_id is None or track_id < 0:
        return 0, False

    # 黑名单优先，防止误判 ID 被标红
    if track_id in IGNORE_TRACK_IDS:
        return (
            sum(track_votes[track_id]),
            False
        )

    # 人工确认的目标 ID，不依赖工牌分数
    if track_id in TARGET_TRACK_IDS:
        confirmed_track_ids.add(track_id)

    # 已经确认的 ID 保持确认状态
    elif track_id in confirmed_track_ids:
        pass

    else:
        # 其他 ID 才使用工牌分数进行投票
        if score is not None:
            track_votes[track_id].append(
                score >= BADGE_THRESHOLD
            )

        vote_count = sum(track_votes[track_id])

        if vote_count >= VOTE_REQUIRED:
            confirmed_track_ids.add(track_id)

    vote_count = sum(track_votes[track_id])

    is_confirmed = (
        track_id in confirmed_track_ids
        and track_id not in IGNORE_TRACK_IDS
    )

    return vote_count, is_confirmed


# ================= 主程序 =================
def main():
    templates = load_badge_templates()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"无法创建输出视频: {OUTPUT_VIDEO}"
        )

    frame_id = 0

    try:
        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as roi_file, open(
            MATCH_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as match_file:

            roi_writer = csv.writer(roi_file)
            match_writer = csv.writer(match_file)

            # ROI 坐标 CSV
            roi_writer.writerow([
                "frame", "track_id",
                "person_x1", "person_y1",
                "person_x2", "person_y2",
                "person_center_x", "person_center_y",
                "roi_x1", "roi_y1", "roi_x2", "roi_y2",
                "roi_width", "roi_height",
                "roi_original", "roi_resized"
            ])

            # 工牌候选 CSV
            match_writer.writerow([
                "frame", "track_id",
                "template", "score",
                "badge_x1", "badge_y1",
                "badge_x2", "badge_y2",
                "badge_center_x", "badge_center_y",
                "votes_in_window", "confirmed"
            ])

            print(f"开始处理: {VIDEO_PATH}")
            print(f"工牌分数门槛: {BADGE_THRESHOLD}")
            print(
                f"自动确认规则：最近 {VOTE_WINDOW} 次检测中，"
                f"至少 {VOTE_REQUIRED} 次达到门槛"
            )
            print(f"人工确认目标 ID: {TARGET_TRACK_IDS}")
            print(f"排除误判 ID: {IGNORE_TRACK_IDS}")

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    imgsz=IMGSZ,
                    conf=YOLO_CONF,
                    tracker=TRACKER_CONFIG,
                    verbose=False
                )

                result = results[0]

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):
                    boxes = result.boxes.xyxy.cpu().numpy()

                    if result.boxes.id is not None:
                        ids = (
                            result.boxes.id.cpu()
                            .numpy()
                            .astype(int)
                        )
                    else:
                        ids = [-1] * len(boxes)

                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)

                        # 限制人物框在视频范围内
                        x1 = max(0, min(x1, width - 1))
                        x2 = max(0, min(x2, width))
                        y1 = max(0, min(y1, height - 1))
                        y2 = max(0, min(y2, height))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        person_box = (x1, y1, x2, y2)

                        # 人物框中心坐标
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # 截取胸前 ROI
                        roi, roi_box = crop_chest_roi(
                            frame, person_box
                        )

                        # 即使 ROI 无效，也保留人物框显示
                        score = None
                        candidate_box = None
                        template_name = None
                        vote_count = 0

                        if roi is not None:
                            rx1, ry1, rx2, ry2 = roi_box
                            roi_h, roi_w = roi.shape[:2]

                            # 保存 ROI 图片
                            base_name = (
                                f"f{frame_id:06d}_id{track_id}"
                            )

                            original_path = os.path.join(
                                ROI_DIR,
                                base_name + ".jpg"
                            )

                            resized_path = os.path.join(
                                ROI_RESIZED_DIR,
                                base_name + ".jpg"
                            )

                            cv2.imwrite(original_path, roi)

                            resized = cv2.resize(
                                roi,
                                (RESIZE_WIDTH, RESIZE_HEIGHT),
                                interpolation=cv2.INTER_CUBIC
                            )

                            cv2.imwrite(resized_path, resized)

                            # 写入 ROI 坐标 CSV
                            roi_writer.writerow([
                                frame_id, int(track_id),
                                x1, y1, x2, y2,
                                center_x, center_y,
                                rx1, ry1, rx2, ry2,
                                roi_w, roi_h,
                                original_path, resized_path
                            ])

                            # 工牌模板匹配
                            score, candidate_box, template_name = (
                                detect_badge_similarity(
                                    roi, templates
                                )
                            )

                            # 写入工牌候选 CSV
                            if candidate_box is not None:
                                bx, by, bw, bh = candidate_box

                                # 转换为原始视频坐标
                                badge_x1 = rx1 + bx
                                badge_y1 = ry1 + by
                                badge_x2 = badge_x1 + bw
                                badge_y2 = badge_y1 + bh

                                badge_cx = (
                                    badge_x1 + badge_x2
                                ) // 2

                                badge_cy = (
                                    badge_y1 + badge_y2
                                ) // 2

                                # 橙色框：工牌最佳匹配候选
                                if score >= DISPLAY_THRESHOLD:
                                    cv2.rectangle(
                                        frame,
                                        (badge_x1, badge_y1),
                                        (badge_x2, badge_y2),
                                        (0, 165, 255),
                                        2
                                    )

                                    cv2.putText(
                                        frame,
                                        f"Match:{score:.2f}",
                                        (
                                            badge_x1,
                                            max(badge_y1 - 5, 15)
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (0, 165, 255),
                                        1
                                    )

                        # 更新目标确认状态
                        vote_count, is_confirmed = (
                            update_target_status(
                                int(track_id), score
                            )
                        )

                        if candidate_box is not None:
                            match_writer.writerow([
                                frame_id,
                                int(track_id),
                                template_name,
                                round(score, 4),
                                badge_x1, badge_y1,
                                badge_x2, badge_y2,
                                badge_cx, badge_cy,
                                vote_count,
                                is_confirmed
                            ])

                        # ================= 人物框显示 =================
                        if is_confirmed:
                            # 目标人物：红框 + ID + 中心坐标
                            person_color = (0, 0, 255)

                            label = (
                                f"TARGET ID:{track_id}, "
                                f"X:{center_x}, "
                                f"Y:{center_y}"
                            )

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                person_color,
                                2
                            )

                            cv2.putText(
                                frame,
                                label,
                                (
                                    x1,
                                    max(y1 - 10, 20)
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                person_color,
                                2
                            )

                        else:
                            # 其他人物：只画绿色框，显示ID
                            person_color = (0, 255, 0)

                            label = (
                                f"ID:{track_id}, "
                            )
                            
                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                person_color,
                                2
                            )

                            cv2.putText(
                                frame,
                                label,
                                (
                                    x1,
                                    max(y1 - 10, 20)
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                person_color,
                                2
                            )

                out.write(frame)
                frame_id += 1

                if frame_id % 30 == 0:
                    print(
                        f"已处理 {frame_id} 帧，"
                        f"已确认 ID: "
                        f"{sorted(confirmed_track_ids)}"
                    )

    finally:
        cap.release()
        out.release()

    print("处理完成！")
    print(f"输出视频：{OUTPUT_VIDEO}")
    print(f"ROI 坐标：{OUTPUT_CSV}")
    print(f"工牌候选：{MATCH_CSV}")
    print(f"原始 ROI：{ROI_DIR}")
    print(
        "本次确认的 track_id："
        f"{sorted(confirmed_track_ids)}"
    )


if __name__ == "__main__":
    main()