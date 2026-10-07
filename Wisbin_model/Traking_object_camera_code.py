import cv2
import depthai as dai
import numpy as np
import warnings
import time
import json
import os
from datetime import datetime


# ============================================================
# WISBIN - SIMPLE OBJECT ISOLATION TEST
#
# IMPORTANT:
# - RGB + Depth are used ONLY to detect that an object exists.
# - The FINAL object segmentation uses STABLE RGB CHANGE ONLY.
# - NO GrabCut.
# - NO Depth inside the final object mask.
# - NO Edge mask.
#
# FLOW:
#
# Empty chute -> press E
#       ↓
# RGB + Depth presence detection
#       ↓
# Object confirmed
#       ↓
# Wait until object settles
#       ↓
# Capture several fresh RGB frames
#       ↓
# Stable RGB change (temporal voting)
#       ↓
# Minimal cleanup
#       ↓
# Largest RGB component
#       ↓
# Exact RGB mask
#       ↓
# Remove chute/background
#       ↓
# Object-only image
# ============================================================


# ============================================================
# FILES
# ============================================================

CHUTE_ROI_FILE = "chute_roi.json"

# ============================================================
# OLD ECOVISION MODEL
# ============================================================

BLOB_PATH = "wisbin_experiments/D_weights_u7/wisbin_mobilenetv3_openvino_2022.1_8shave.blob"
CLASSES = ["metal", "other", "paper", "plastic"]


MODEL_SIZE = 224

CLASSIFICATION_FRAMES = 5


# ============================================================
# MANUAL DATA COLLECTION FOR FUTURE IMPROVEMENT
#
# While an object is on the chute, press:
# 1 = METAL
# 2 = PLASTIC
# 3 = PAPER
# 4 = OTHER
#
# The EXACT isolated image sent to EcoVision will be saved.
# ============================================================

WISBIN_DATASET_DIR = "wisbin_dataset"

MANUAL_LABEL_KEYS = {
    ord("1"): "metal",
    ord("2"): "plastic",
    ord("3"): "paper",
    ord("4"): "other",
}

def get_dataset_class_dir(
    class_name
):

    # If the existing dataset is already split into
    # train / val / test, manually collected images belong
    # ONLY in train. We never modify val or test.
    train_root = os.path.join(
        WISBIN_DATASET_DIR,
        "train"
    )


    if os.path.isdir(
        train_root
    ):

        class_dir = os.path.join(
            train_root,
            class_name
        )

    else:

        class_dir = os.path.join(
            WISBIN_DATASET_DIR,
            class_name
        )


    os.makedirs(
        class_dir,
        exist_ok=True
    )


    return class_dir


for _class_name in MANUAL_LABEL_KEYS.values():

    get_dataset_class_dir(
        _class_name
    )


# ============================================================
# CAMERA
# ============================================================

FPS = 10

RGB_WIDTH = 640
RGB_HEIGHT = 400


# ============================================================
# EMPTY BASELINE
# ============================================================

BASELINE_FRAMES = 25


# ============================================================
# PRESENCE DETECTION
#
# Presence can use RGB OR Depth.
# This has nothing to do with the final segmentation mask.
# ============================================================

RGB_PIXEL_THRESHOLD = 18

# Small objects should still trigger the test.
RGB_CHANGED_PERCENT_THRESHOLD = 3.0

DEPTH_DELTA_MM = 12

DEPTH_CHANGED_PERCENT_THRESHOLD = 2.0

CONFIRM_FRAMES = 3

RELEASE_FRAMES = 5

# ============================================================
# ROBUST OBJECT REMOVAL DETECTION
#
# بعد ما نخلص عزل وتصنيف القطعة، ما نعتمد على Presence كامل
# الـchute لمسح المربع الأخضر، لأن Depth ممكن يعطي Noise ويبقى
# شايف YES حتى بعد إزالة القطعة.
#
# بدل ذلك نراقب فقط آخر Bounding Box للجسم.
# إذا رجعت هذه المنطقة قريبة من الـEmpty Baseline لعدة Frames،
# نعتبر أن القطعة انشالت ونمسح المربع والنتيجة فوراً.
# ============================================================

REMOVE_CONFIRM_FRAMES = 3

# IMPORTANT:
# We now check ONLY pixels that belonged to the actual FINAL OBJECT MASK,
# not the whole bounding box.
#
# The previous 2% / 3% thresholds were too sensitive:
# normal RGB/Depth noise could keep the object marked as present forever.
#
# On the exact object pixels, a real object should produce a much stronger
# signal, so these thresholds can safely be higher.
REMOVE_RGB_MASK_PERCENT_THRESHOLD = 15.0
REMOVE_DEPTH_MASK_PERCENT_THRESHOLD = 15.0


# ============================================================
# OBJECT TRACKING / SETTLE
# ============================================================

# بعد تأكيد وجود جسم، نتابعه لمدة ثانيتين لأنه ممكن
# يكون دائري أو يتدحرج قبل أن يستقر.
TRACK_OBJECT_SEC = 2.0

# Threshold خاص بالتتبع أثناء الحركة.
TRACK_RGB_THRESHOLD = 24

# أقل مساحة Component كنسبة من مساحة الـ chute أثناء التتبع.
TRACK_MIN_COMPONENT_PERCENT = 0.08


# ============================================================
# STABLE RGB + CONSTRAINED DEPTH SEGMENTATION
# ============================================================

# Number of NEW frames used after tracking finishes.
SEGMENTATION_FRAMES = 7

# Pixel must appear changed in at least this many frames.
SEGMENTATION_REQUIRED_FRAMES = 3

# Strong RGB difference used for the final segmentation.
RGB_SEGMENT_THRESHOLD = 28

# Depth threshold for black / dark objects that RGB may miss.
HYBRID_DEPTH_DELTA_MM = 12

# Depth is allowed ONLY around the final tracked object area.
TRACKED_DEPTH_REGION_EXPAND = 0.25

# If RGB area is much smaller than depth area, use depth to
# complete the missing parts of the object.
RGB_TO_DEPTH_AREA_RATIO_THRESHOLD = 0.60

# Ignore tiny noise blobs.
MIN_COMPONENT_PERCENT = 0.10

# Minimal morphology only.
SEGMENT_CLOSE_KERNEL = 3
SEGMENT_CLOSE_ITERATIONS = 1

# Small neutral border around the final object crop.
FINAL_NEUTRAL_MARGIN_PERCENT = 0.04

# Neutral BGR background for the preview / future model input.
OUTPUT_BACKGROUND_BGR = (
    127,
    127,
    127
)


# ============================================================
# STATE
# ============================================================

baseline_rgb_color = None
baseline_gray = None
baseline_depth = None

chute_points = None
chute_mask = None

object_frames = 0
empty_frames = 0

object_confirmed = False
object_already_segmented = False

last_bbox = None

# Exact final mask of the last classified object.
# Used only to detect when that SAME object has been removed.
last_object_mask = None

# عدد Frames المتتالية التي تؤكد أن آخر جسم تم إزالته.
remove_confirm_count = 0


# ============================================================
# ROI SAVE / LOAD
# ============================================================

def save_chute_roi(
    points,
    frame_shape
):

    h, w = frame_shape[:2]

    normalized = []

    for x, y in points:

        normalized.append(
            [
                x / w,
                y / h
            ]
        )

    data = {
        "points": normalized
    }

    with open(
        CHUTE_ROI_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )

    print()
    print(
        "BLACK CHUTE ROI SAVED:",
        CHUTE_ROI_FILE
    )
    print()


def load_chute_roi(
    frame_shape
):

    if not os.path.exists(
        CHUTE_ROI_FILE
    ):

        return None

    try:

        with open(
            CHUTE_ROI_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        h, w = frame_shape[:2]

        points = []

        for nx, ny in data["points"]:

            points.append(
                (
                    int(nx * w),
                    int(ny * h)
                )
            )

        if len(points) != 4:

            return None

        return points

    except Exception:

        return None


# ============================================================
# SELECT BLACK CHUTE
# ============================================================

def select_chute_polygon(
    frame
):

    window_name = (
        "SELECT BLACK CHUTE ROI"
    )

    points = []


    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
        ):

            if len(points) < 4:

                points.append(
                    (
                        x,
                        y
                    )
                )

                print(
                    f"Point "
                    f"{len(points)}: "
                    f"{x}, {y}"
                )


    cv2.namedWindow(
        window_name
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    print()
    print(
        "===================================="
    )
    print(
        "SELECT BLACK CHUTE"
    )
    print(
        "===================================="
    )
    print(
        "1. Top-left"
    )
    print(
        "2. Top-right"
    )
    print(
        "3. Bottom-right"
    )
    print(
        "4. Bottom-left"
    )
    print()
    print(
        "ENTER / S = Save"
    )
    print(
        "R = Reset"
    )
    print(
        "Q = Quit"
    )
    print(
        "===================================="
    )
    print()


    while True:

        display = frame.copy()


        for i, point in enumerate(
            points
        ):

            cv2.circle(
                display,
                point,
                7,
                (
                    0,
                    255,
                    0
                ),
                -1
            )

            cv2.putText(
                display,
                str(
                    i + 1
                ),
                (
                    point[0] + 8,
                    point[1] - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    0
                ),
                2
            )


        if len(points) >= 2:

            for i in range(
                len(points) - 1
            ):

                cv2.line(
                    display,
                    points[i],
                    points[i + 1],
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


        if len(points) == 4:

            cv2.line(
                display,
                points[3],
                points[0],
                (
                    0,
                    255,
                    0
                ),
                2
            )


        cv2.putText(
            display,
            (
                f"Points: "
                f"{len(points)}/4"
            ),
            (
                15,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255
            ),
            2
        )


        cv2.imshow(
            window_name,
            display
        )


        key = (
            cv2.waitKey(20)
            &
            0xFF
        )


        if (
            key == 13
            or
            key == ord("s")
        ):

            if len(points) == 4:

                save_chute_roi(
                    points,
                    frame.shape
                )

                cv2.destroyWindow(
                    window_name
                )

                return points


        elif key == ord("r"):

            points.clear()


        elif key == ord("q"):

            cv2.destroyWindow(
                window_name
            )

            raise SystemExit


# ============================================================
# CHUTE MASK
# ============================================================

def create_chute_mask(
    frame_shape,
    points
):

    h, w = frame_shape[:2]

    mask = np.zeros(
        (
            h,
            w
        ),
        dtype=np.uint8
    )

    polygon = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [
            polygon
        ],
        255
    )

    return mask


def draw_chute_polygon(
    frame,
    points
):

    polygon = np.array(
        points,
        dtype=np.int32
    )

    cv2.polylines(
        frame,
        [
            polygon
        ],
        True,
        (
            255,
            255,
            255
        ),
        2
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def prepare_gray(
    frame
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (
            5,
            5
        ),
        0
    )

    return gray


def flush_queue(
    queue
):

    while True:

        packet = queue.tryGet()

        if packet is None:

            break


# ============================================================
# BUILD EMPTY BASELINE
# ============================================================

def build_baseline(
    rgb_queue,
    depth_queue
):

    print()
    print(
        "===================================="
    )
    print(
        "BUILDING EMPTY CHUTE BASELINE"
    )
    print(
        "KEEP CHUTE COMPLETELY EMPTY"
    )
    print(
        "===================================="
    )


    flush_queue(
        rgb_queue
    )

    flush_queue(
        depth_queue
    )


    rgb_samples = []

    gray_samples = []

    depth_samples = []


    for i in range(
        BASELINE_FRAMES
    ):

        frame = (
            rgb_queue
            .get()
            .getCvFrame()
        )

        depth = (
            depth_queue
            .get()
            .getFrame()
            .astype(
                np.float32
            )
        )


        rgb_samples.append(
            frame.astype(
                np.float32
            )
        )


        gray_samples.append(
            prepare_gray(
                frame
            )
        )


        depth[
            depth <= 0
        ] = np.nan


        depth_samples.append(
            depth
        )


        print(
            f"Baseline "
            f"{i + 1}/"
            f"{BASELINE_FRAMES}",
            end="\r"
        )


    baseline_color = np.median(
        np.stack(
            rgb_samples,
            axis=0
        ),
        axis=0
    ).astype(
        np.uint8
    )


    baseline_gray_local = np.median(
        np.stack(
            gray_samples,
            axis=0
        ),
        axis=0
    ).astype(
        np.uint8
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            RuntimeWarning
        )

        baseline_depth_local = np.nanmedian(
            np.stack(
                depth_samples,
                axis=0
            ),
            axis=0
        ).astype(
            np.float32
        )


    baseline_depth_local[
        chute_mask == 0
    ] = np.nan


    print()
    print()
    print(
        "EMPTY BASELINE READY"
    )
    print()


    return (
        baseline_color,
        baseline_gray_local,
        baseline_depth_local
    )


# ============================================================
# RGB DIFFERENCE
#
# Same idea that produced the good Stable RGB Change result.
# ============================================================

def get_rgb_difference(
    current_frame
):

    current_gray = prepare_gray(
        current_frame
    )


    chute_pixels = (
        chute_mask > 0
    )


    base_values = baseline_gray[
        chute_pixels
    ].astype(
        np.float32
    )


    current_values = current_gray[
        chute_pixels
    ].astype(
        np.float32
    )


    if (
        base_values.size == 0
        or
        current_values.size == 0
    ):

        return np.zeros_like(
            baseline_gray
        )


    # --------------------------------------------------------
    # Compensate global brightness change
    # --------------------------------------------------------

    brightness_shift = (
        np.median(
            base_values
        )
        -
        np.median(
            current_values
        )
    )


    adjusted_gray = np.clip(
        current_gray.astype(
            np.float32
        )
        +
        brightness_shift,
        0,
        255
    ).astype(
        np.uint8
    )


    gray_diff = cv2.absdiff(
        baseline_gray,
        adjusted_gray
    )


    # --------------------------------------------------------
    # Per-pixel color difference
    # --------------------------------------------------------

    current_float = current_frame.astype(
        np.int16
    )

    baseline_float = baseline_rgb_color.astype(
        np.int16
    )


    color_diff = np.max(
        np.abs(
            current_float
            -
            baseline_float
        ),
        axis=2
    ).astype(
        np.uint8
    )


    difference = np.maximum(
        gray_diff,
        color_diff
    )


    difference[
        chute_mask == 0
    ] = 0


    return difference


# ============================================================
# RGB PRESENCE
# ============================================================

def detect_rgb_presence(
    current_frame
):

    difference = get_rgb_difference(
        current_frame
    )


    mask = (
        difference
        >
        RGB_PIXEL_THRESHOLD
    ).astype(
        np.uint8
    ) * 255


    mask = cv2.bitwise_and(
        mask,
        chute_mask
    )


    kernel = np.ones(
        (
            3,
            3
        ),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    mask = cv2.bitwise_and(
        mask,
        chute_mask
    )


    changed_pixels = cv2.countNonZero(
        mask
    )


    chute_area = cv2.countNonZero(
        chute_mask
    )


    percent = (
        changed_pixels
        /
        max(
            chute_area,
            1
        )
        *
        100.0
    )


    detected = (
        percent
        >=
        RGB_CHANGED_PERCENT_THRESHOLD
    )


    return (
        detected,
        percent,
        mask
    )


# ============================================================
# DEPTH PRESENCE
# ============================================================

def detect_depth_presence(
    current_depth
):

    current_depth = current_depth.astype(
        np.float32
    )


    valid = (
        (chute_mask > 0)
        &
        np.isfinite(
            baseline_depth
        )
        &
        (
            baseline_depth > 0
        )
        &
        (
            current_depth > 0
        )
    )


    delta = np.zeros_like(
        current_depth,
        dtype=np.float32
    )


    delta[
        valid
    ] = (
        baseline_depth[
            valid
        ]
        -
        current_depth[
            valid
        ]
    )


    mask = (
        delta
        >
        DEPTH_DELTA_MM
    ).astype(
        np.uint8
    ) * 255


    mask = cv2.bitwise_and(
        mask,
        chute_mask
    )


    kernel = np.ones(
        (
            3,
            3
        ),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    mask = cv2.bitwise_and(
        mask,
        chute_mask
    )


    valid_count = int(
        np.count_nonzero(
            valid
        )
    )


    changed_count = int(
        cv2.countNonZero(
            mask
        )
    )


    percent = (
        changed_count
        /
        max(
            valid_count,
            1
        )
        *
        100.0
    )


    detected = (
        percent
        >=
        DEPTH_CHANGED_PERCENT_THRESHOLD
    )


    return (
        detected,
        percent,
        mask
    )


# ============================================================
# TEMPORAL RGB VOTING
# ============================================================

def temporal_vote_rgb(
    masks
):

    votes = np.zeros_like(
        chute_mask,
        dtype=np.uint16
    )


    for mask in masks:

        votes += (
            mask > 0
        ).astype(
            np.uint16
        )


    stable = (
        votes
        >=
        SEGMENTATION_REQUIRED_FRAMES
    ).astype(
        np.uint8
    ) * 255


    stable = cv2.bitwise_and(
        stable,
        chute_mask
    )


    return stable


# ============================================================
# BUILD TRACKED DEPTH REGION
#
# Depth is NEVER allowed over the entire chute.
# It is restricted to the last tracked object location with
# a small expansion around it.
# ============================================================

def build_tracked_depth_region(
    bbox
):

    region = np.zeros_like(
        chute_mask,
        dtype=np.uint8
    )

    if bbox is None:
        return region

    x1, y1, x2, y2 = bbox

    w = max(
        1,
        x2 - x1
    )

    h = max(
        1,
        y2 - y1
    )

    margin_x = int(
        w * TRACKED_DEPTH_REGION_EXPAND
    )

    margin_y = int(
        h * TRACKED_DEPTH_REGION_EXPAND
    )

    rx1 = max(
        0,
        x1 - margin_x
    )

    ry1 = max(
        0,
        y1 - margin_y
    )

    rx2 = min(
        region.shape[1],
        x2 + margin_x
    )

    ry2 = min(
        region.shape[0],
        y2 + margin_y
    )

    cv2.rectangle(
        region,
        (
            rx1,
            ry1
        ),
        (
            rx2,
            ry2
        ),
        255,
        -1
    )

    region = cv2.bitwise_and(
        region,
        chute_mask
    )

    return region


# ============================================================
# STABLE DEPTH MASK FOR FINAL SEGMENTATION
# ============================================================

def build_stable_depth_mask(
    depth_frames,
    tracked_bbox
):

    tracked_region = build_tracked_depth_region(
        tracked_bbox
    )

    if cv2.countNonZero(
        tracked_region
    ) == 0:

        return (
            np.zeros_like(
                chute_mask
            ),
            tracked_region
        )

    masks = []

    for depth in depth_frames:

        current_depth = depth.astype(
            np.float32
        )

        valid = (
            (tracked_region > 0)
            &
            np.isfinite(
                baseline_depth
            )
            &
            (
                baseline_depth > 0
            )
            &
            (
                current_depth > 0
            )
        )

        delta = np.zeros_like(
            current_depth,
            dtype=np.float32
        )

        delta[
            valid
        ] = (
            baseline_depth[
                valid
            ]
            -
            current_depth[
                valid
            ]
        )

        depth_mask = (
            delta
            >
            HYBRID_DEPTH_DELTA_MM
        ).astype(
            np.uint8
        ) * 255

        depth_mask = cv2.bitwise_and(
            depth_mask,
            tracked_region
        )

        depth_mask = cv2.morphologyEx(
            depth_mask,
            cv2.MORPH_OPEN,
            np.ones(
                (
                    3,
                    3
                ),
                np.uint8
            )
        )

        depth_mask = cv2.morphologyEx(
            depth_mask,
            cv2.MORPH_CLOSE,
            np.ones(
                (
                    3,
                    3
                ),
                np.uint8
            ),
            iterations=1
        )

        masks.append(
            depth_mask
        )

    stable_depth = temporal_vote_rgb(
        masks
    )

    stable_depth = cv2.bitwise_and(
        stable_depth,
        tracked_region
    )

    return (
        stable_depth,
        tracked_region
    )


# ============================================================
# BUILD FINAL HYBRID MASK
#
# RGB is always primary.
# Depth is used ONLY if RGB appears incomplete compared with
# the restricted depth mask.
# ============================================================

def build_final_hybrid_mask(
    stable_rgb,
    stable_depth,
    tracked_region
):

    rgb_area = cv2.countNonZero(
        stable_rgb
    )

    depth_area = cv2.countNonZero(
        stable_depth
    )

    use_depth_fallback = False

    if depth_area > 0:

        ratio = (
            rgb_area
            /
            max(
                depth_area,
                1
            )
        )

        if (
            rgb_area == 0
            or
            ratio
            <
            RGB_TO_DEPTH_AREA_RATIO_THRESHOLD
        ):

            use_depth_fallback = True

    if use_depth_fallback:

        print()
        print(
            "HYBRID SEGMENTATION:"
        )
        print(
            "RGB appears incomplete -> "
            "using constrained Depth fallback."
        )
        print(
            f"RGB area   : {rgb_area}"
        )
        print(
            f"Depth area : {depth_area}"
        )
        print()

        final_seed = cv2.bitwise_or(
            stable_rgb,
            stable_depth
        )

    else:

        print()
        print(
            "HYBRID SEGMENTATION:"
        )
        print(
            "RGB mask is sufficient -> "
            "Depth fallback NOT used."
        )
        print(
            f"RGB area   : {rgb_area}"
        )
        print(
            f"Depth area : {depth_area}"
        )
        print()

        final_seed = stable_rgb.copy()

    final_seed = cv2.bitwise_and(
        final_seed,
        chute_mask
    )

    kernel = np.ones(
        (
            SEGMENT_CLOSE_KERNEL,
            SEGMENT_CLOSE_KERNEL
        ),
        np.uint8
    )

    final_seed = cv2.morphologyEx(
        final_seed,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=SEGMENT_CLOSE_ITERATIONS
    )

    contours, _ = cv2.findContours(
        final_seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return (
            None,
            use_depth_fallback
        )

    chute_area = cv2.countNonZero(
        chute_mask
    )

    minimum_area = (
        chute_area
        *
        MIN_COMPONENT_PERCENT
        /
        100.0
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(
            contour
        ) >= minimum_area
    ]

    if not valid_contours:
        return (
            None,
            use_depth_fallback
        )

    # Prefer a component overlapping the tracked region.
    best_contour = None
    best_score = -1

    for contour in valid_contours:

        component = np.zeros_like(
            final_seed
        )

        cv2.drawContours(
            component,
            [
                contour
            ],
            -1,
            255,
            -1
        )

        overlap = cv2.countNonZero(
            cv2.bitwise_and(
                component,
                tracked_region
            )
        )

        score = (
            overlap
            *
            10
            +
            cv2.contourArea(
                contour
            )
        )

        if score > best_score:

            best_score = score
            best_contour = contour

    final_mask = np.zeros_like(
        final_seed
    )

    cv2.drawContours(
        final_mask,
        [
            best_contour
        ],
        -1,
        255,
        -1
    )

    final_mask = cv2.bitwise_and(
        final_mask,
        chute_mask
    )

    return (
        final_mask,
        use_depth_fallback
    )


# ============================================================
# OBJECT-ONLY IMAGE FROM RGB MASK
# ============================================================

def extract_object_only(
    frame,
    final_mask
):

    nonzero = cv2.findNonZero(
        final_mask
    )


    if nonzero is None:

        return (
            None,
            None,
            None,
            None
        )


    x, y, w, h = cv2.boundingRect(
        nonzero
    )


    # --------------------------------------------------------
    # Tiny crop margin.
    # This adds only neutral background, NOT chute pixels.
    # --------------------------------------------------------

    margin_x = int(
        w
        *
        FINAL_NEUTRAL_MARGIN_PERCENT
    )

    margin_y = int(
        h
        *
        FINAL_NEUTRAL_MARGIN_PERCENT
    )


    x1 = max(
        0,
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        frame.shape[1],
        x + w + margin_x
    )

    y2 = min(
        frame.shape[0],
        y + h + margin_y
    )


    crop = frame[
        y1:y2,
        x1:x2
    ]


    local_mask = final_mask[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        return (
            None,
            None,
            None,
            None
        )


    # --------------------------------------------------------
    # Object only.
    # Every pixel outside Stable RGB mask is replaced.
    # --------------------------------------------------------

    isolated = np.empty_like(
        crop
    )


    isolated[:] = (
        OUTPUT_BACKGROUND_BGR
    )


    object_pixels = (
        local_mask > 0
    )


    isolated[
        object_pixels
    ] = crop[
        object_pixels
    ]


    # --------------------------------------------------------
    # Square future-model input.
    # Again: padding is neutral, not chute.
    # --------------------------------------------------------

    ch, cw = isolated.shape[:2]


    side = max(
        ch,
        cw
    )


    square = np.empty(
        (
            side,
            side,
            3
        ),
        dtype=np.uint8
    )


    square[:] = (
        OUTPUT_BACKGROUND_BGR
    )


    offset_x = (
        side - cw
    ) // 2


    offset_y = (
        side - ch
    ) // 2


    square[
        offset_y:
        offset_y + ch,

        offset_x:
        offset_x + cw
    ] = isolated


    # --------------------------------------------------------
    # Transparent PNG version
    # --------------------------------------------------------

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2BGRA
    )


    rgba[
        :,
        :,
        3
    ] = local_mask


    bbox = (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )


    return (
        isolated,
        square,
        rgba,
        bbox
    )


# ============================================================
# TRACK OBJECT FOR A FIXED DURATION
#
# بعد تأكيد سقوط جسم، نتابع الجسم المتحرك لمدة ثانيتين.
# الـ Bounding Box يتحدث كل Frame، لذلك لو الجسم تدحرج أو تحرك
# نتابع آخر مكان له. بعد انتهاء التتبع نرمي Frames التتبع ونأخذ
# Fresh Frames جديدة للـ Stable RGB segmentation النهائي.
# ============================================================

def track_object_for_duration(
    rgb_queue,
    depth_queue
):

    print()
    print("====================================")
    print("TRACKING FALLING / MOVING OBJECT")
    print(f"Tracking for {TRACK_OBJECT_SEC:.2f} seconds...")
    print("====================================")

    flush_queue(rgb_queue)
    flush_queue(depth_queue)

    start_time = time.time()
    last_bbox = None
    last_frame = None

    chute_area = cv2.countNonZero(chute_mask)
    min_area = (
        chute_area
        * TRACK_MIN_COMPONENT_PERCENT
        / 100.0
    )

    while True:

        elapsed = time.time() - start_time

        if elapsed >= TRACK_OBJECT_SEC:
            break

        frame = rgb_queue.get().getCvFrame()

        # نستهلك Depth frame للحفاظ على تزامن الـ queues فقط.
        _ = depth_queue.get().getFrame()

        last_frame = frame.copy()

        difference = get_rgb_difference(frame)

        track_mask = (
            difference > TRACK_RGB_THRESHOLD
        ).astype(np.uint8) * 255

        track_mask = cv2.bitwise_and(
            track_mask,
            chute_mask
        )

        # تنظيف خفيف فقط حتى لا نكبر الجسم على الخلفية.
        kernel = np.ones((3, 3), np.uint8)

        track_mask = cv2.morphologyEx(
            track_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        track_mask = cv2.morphologyEx(
            track_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            track_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]

        if valid_contours:

            largest = max(
                valid_contours,
                key=cv2.contourArea
            )

            x, y, w, h = cv2.boundingRect(largest)

            margin_x = int(w * 0.08)
            margin_y = int(h * 0.08)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)

            last_bbox = (x1, y1, x2, y2)

        tracking_display = frame.copy()
        draw_chute_polygon(
            tracking_display,
            chute_points
        )

        if last_bbox is not None:

            x1, y1, x2, y2 = last_bbox

            cv2.rectangle(
                tracking_display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                tracking_display,
                "TRACKING OBJECT",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        remaining = max(
            0.0,
            TRACK_OBJECT_SEC - elapsed
        )

        cv2.putText(
            tracking_display,
            f"Tracking: {remaining:.1f}s",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "OBJECT TRACKING - MOVING ITEM",
            tracking_display
        )

        cv2.imshow(
            "TRACKING RGB MASK",
            track_mask
        )

        cv2.waitKey(1)

    # Close the temporary tracking windows so the last green
    # tracking rectangle does not remain frozen on screen.
    try:
        cv2.destroyWindow(
            "OBJECT TRACKING - MOVING ITEM"
        )
    except cv2.error:
        pass

    try:
        cv2.destroyWindow(
            "TRACKING RGB MASK"
        )
    except cv2.error:
        pass

    print()
    print("Tracking finished.")

    if last_bbox is not None:
        print("Last tracked bbox:", last_bbox)

    print(
        "Now capturing fresh frames for final "
        "Stable RGB segmentation..."
    )
    print()

    return last_frame, last_bbox


# ============================================================
# SEGMENT CURRENT OBJECT
# ============================================================

def segment_current_object(
    rgb_queue,
    depth_queue
):

    print()
    print(
        "===================================="
    )
    print(
        "OBJECT CONFIRMED"
    )
    print(
        "Starting 2-second live tracking..."
    )
    print(
        "===================================="
    )

    (
        tracked_frame,
        tracked_bbox
    ) = track_object_for_duration(
        rgb_queue,
        depth_queue
    )

    flush_queue(
        rgb_queue
    )

    flush_queue(
        depth_queue
    )

    rgb_segmentation_masks = []

    depth_frames = []

    last_frame = None

    # ========================================================
    # CAPTURE FRESH RGB + DEPTH FRAMES
    # ========================================================

    for i in range(
        SEGMENTATION_FRAMES
    ):

        frame = (
            rgb_queue
            .get()
            .getCvFrame()
        )

        depth = (
            depth_queue
            .get()
            .getFrame()
        )

        last_frame = (
            frame.copy()
        )

        depth_frames.append(
            depth.copy()
        )

        difference = get_rgb_difference(
            frame
        )

        rgb_mask = (
            difference
            >
            RGB_SEGMENT_THRESHOLD
        ).astype(
            np.uint8
        ) * 255

        rgb_mask = cv2.bitwise_and(
            rgb_mask,
            chute_mask
        )

        rgb_mask = cv2.morphologyEx(
            rgb_mask,
            cv2.MORPH_OPEN,
            np.ones(
                (
                    3,
                    3
                ),
                np.uint8
            )
        )

        rgb_segmentation_masks.append(
            rgb_mask
        )

    # ========================================================
    # STABLE RGB
    # ========================================================

    stable_rgb = temporal_vote_rgb(
        rgb_segmentation_masks
    )

    # ========================================================
    # STABLE DEPTH - ONLY AROUND FINAL TRACKED AREA
    # ========================================================

    (
        stable_depth,
        tracked_region
    ) = build_stable_depth_mask(
        depth_frames,
        tracked_bbox
    )

    # ========================================================
    # RGB PRIMARY + DEPTH FALLBACK
    # ========================================================

    (
        final_mask,
        used_depth_fallback
    ) = build_final_hybrid_mask(
        stable_rgb,
        stable_depth,
        tracked_region
    )

    if final_mask is None:

        print()
        print(
            "HYBRID SEGMENTATION FAILED."
        )
        print()

        return (
            last_frame,
            stable_rgb,
            stable_depth,
            tracked_region,
            None,
            None,
            None,
            None,
            False
        )

    (
        isolated,
        square,
        transparent,
        bbox
    ) = extract_object_only(
        last_frame,
        final_mask
    )

    return (
        last_frame,
        stable_rgb,
        stable_depth,
        tracked_region,
        final_mask,
        square,
        transparent,
        bbox,
        used_depth_fallback
    )


# ============================================================
# WISBIN CLASS MAPPING
# ============================================================

def to_wisbin(label):

    if label == "metal":
        return "METAL"

    if label == "plastic":
        return "PLASTIC"

    if label in [
        "paper",
        "cardboard"
    ]:
        return "PAPER"

    return "OTHER"


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=np.float32
    )

    logits = (
        logits
        -
        np.max(logits)
    )

    exp_values = np.exp(
        logits
    )

    return (
        exp_values
        /
        np.sum(exp_values)
    )


# ============================================================
# PREPARE IMAGE FOR OLD ECOVISION BLOB
# ============================================================

def prepare_nn_frame(
    crop
):

    resized = cv2.resize(
        crop,
        (
            MODEL_SIZE,
            MODEL_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # OpenCV BGR -> RGB
    rgb_crop = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # HWC -> CHW
    planar = np.transpose(
        rgb_crop,
        (
            2,
            0,
            1
        )
    )

    planar = np.ascontiguousarray(
        planar,
        dtype=np.uint8
    )

    img = dai.ImgFrame()

    img.setType(
        dai.ImgFrame.Type.RGB888p
    )

    img.setWidth(
        MODEL_SIZE
    )

    img.setHeight(
        MODEL_SIZE
    )

    img.setData(
        planar.flatten()
    )

    return img


# ============================================================
# APPLY EXISTING RGB MASK TO A NEW FRAME
#
# Important:
# We keep exactly the same final RGB-only object mask that
# worked well during isolation, and replace everything else
# with neutral gray before sending it to the model.
# ============================================================

def isolate_with_existing_mask(
    frame,
    final_mask,
    bbox
):

    if (
        final_mask is None
        or
        bbox is None
    ):

        return None

    x1, y1, x2, y2 = bbox

    crop = frame[
        y1:y2,
        x1:x2
    ]

    local_mask = final_mask[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None

    isolated = np.empty_like(
        crop
    )

    isolated[:] = (
        OUTPUT_BACKGROUND_BGR
    )

    object_pixels = (
        local_mask > 0
    )

    isolated[
        object_pixels
    ] = crop[
        object_pixels
    ]

    # Square pad with NEUTRAL background only.
    h, w = isolated.shape[:2]

    side = max(
        h,
        w
    )

    square = np.empty(
        (
            side,
            side,
            3
        ),
        dtype=np.uint8
    )

    square[:] = (
        OUTPUT_BACKGROUND_BGR
    )

    ox = (
        side - w
    ) // 2

    oy = (
        side - h
    ) // 2

    square[
        oy:
        oy + h,
        ox:
        ox + w
    ] = isolated

    return square


# ============================================================
# CLASSIFY ISOLATED OBJECT
#
# 5 fresh RGB frames:
# - same good RGB-only mask
# - no chute pixels
# - average probabilities
# ============================================================

def classify_isolated_object(
    rgb_queue,
    nn_input_queue,
    nn_output_queue,
    final_mask,
    bbox
):

    print()
    print(
        "===================================="
    )
    print(
        "CLASSIFYING ISOLATED OBJECT"
    )
    print(
        "Background removed by Stable RGB mask"
    )
    print(
        "===================================="
    )

    flush_queue(
        rgb_queue
    )

    flush_queue(
        nn_output_queue
    )

    probability_samples = []

    last_square = None

    for i in range(
        CLASSIFICATION_FRAMES
    ):

        frame = (
            rgb_queue
            .get()
            .getCvFrame()
        )

        square = isolate_with_existing_mask(
            frame,
            final_mask,
            bbox
        )

        if square is None:

            continue

        last_square = (
            square.copy()
        )

        preview = cv2.resize(
            square,
            (
                400,
                400
            ),
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow(
            "6 - EXACT ISOLATED IMAGE SENT TO ECOVISION",
            preview
        )

        cv2.waitKey(1)

        nn_frame = prepare_nn_frame(
            square
        )

        nn_input_queue.send(
            nn_frame
        )

        result = (
            nn_output_queue.get()
        )

        logits = result.getLayerFp16(
            "logits"
        )

        probabilities = softmax(
            logits
        )

        probability_samples.append(
            probabilities
        )

        top3 = np.argsort(
            probabilities
        )[::-1][:3]

        print()
        print(
            f"AI Frame {i + 1}:"
        )

        for rank, class_id in enumerate(
            top3,
            start=1
        ):

            print(
                f"  {rank}. "
                f"{CLASSES[class_id]:12s} "
                f"{probabilities[class_id] * 100:.2f}%"
            )

    if not probability_samples:

        return (
            None,
            0.0,
            None,
            None
        )

    average = np.mean(
        np.stack(
            probability_samples,
            axis=0
        ),
        axis=0
    )

    best_id = int(
        np.argmax(
            average
        )
    )

    raw_label = CLASSES[
        best_id
    ]

    confidence = float(
        average[
            best_id
        ]
    )

    wisbin_class = to_wisbin(
        raw_label
    )

    order = np.argsort(
        average
    )[::-1][:5]

    print()
    print(
        "===================================="
    )
    print(
        "FINAL ECOVISION RESULT"
    )
    print(
        "===================================="
    )

    for rank, class_id in enumerate(
        order,
        start=1
    ):

        print(
            f"{rank}. "
            f"{CLASSES[class_id]:12s} "
            f"{average[class_id] * 100:.2f}%"
        )

    print(
        "------------------------------------"
    )
    print(
        "Raw class :",
        raw_label
    )
    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )
    print(
        "WISBIN    :",
        wisbin_class
    )
    print(
        "Input     : STABLE RGB OBJECT ONLY"
    )
    print(
        "===================================="
    )
    print()

    return (
        raw_label,
        confidence,
        wisbin_class,
        last_square
    )



# ============================================================
# SAVE MANUALLY LABELED IMAGE INTO wisbin_dataset
#
# Saves ONLY when the user presses 1 / 2 / 3 / 4.
#
# The saved image is the EXACT isolated square image that was
# sent to EcoVision.
#
# Nothing is saved into wisbin_dataset.
# Nothing is automatically saved just because an object was
# detected.
#
# If wisbin_dataset/train exists, images are saved into train
# only. val and test are never modified.
# ============================================================

def save_manual_training_image(
    image,
    class_name,
    predicted_label=None,
    confidence=None
):

    if image is None:

        print()
        print(
            "No current isolated object image to save."
        )
        print()

        return None


    class_dir = get_dataset_class_dir(
        class_name
    )


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )


    predicted_text = (
        predicted_label
        if predicted_label is not None
        else "unknown"
    )


    confidence_text = (
        f"{confidence * 100:.1f}"
        if confidence is not None
        else "na"
    )


    filename = (
        f"{timestamp}"
        f"_pred-{predicted_text}"
        f"_conf-{confidence_text}.png"
    )


    path = os.path.join(
        class_dir,
        filename
    )


    success = cv2.imwrite(
        path,
        image
    )


    if success:

        print()
        print(
            "===================================="
        )

        print(
            "IMAGE SAVED TO WISBIN DATASET"
        )

        print(
            "===================================="
        )

        print(
            "Correct label:",
            class_name.upper()
        )

        print(
            "Model predicted:",
            predicted_text.upper()
        )

        print(
            "Saved:",
            path
        )

        print(
            "===================================="
        )

        print()

        return path


    print()
    print(
        "ERROR: image could not be saved."
    )
    print()

    return None



# ============================================================
# CHECK IF THE PREVIOUS OBJECT IS STILL INSIDE ITS LAST BBOX
#
# هذا الفحص يُستخدم فقط بعد أن يتم عزل وتصنيف الجسم.
# الفكرة: بدل فحص كامل الـchute، نفحص آخر مكان كان الجسم فيه.
# هذا يمنع Depth noise في أماكن ثانية من إبقاء المربع الأخضر.
# ============================================================

def check_last_object_mask(
    frame,
    current_depth,
    object_mask
):

    # ========================================================
    # We check ONLY the pixels that were actually part of the
    # final object segmentation.
    #
    # This is much more reliable than checking the whole bbox:
    # the bbox contains black chute pixels and Depth noise.
    # ========================================================

    if (
        object_mask is None
        or
        baseline_gray is None
        or
        baseline_depth is None
    ):

        return (
            False,
            0.0,
            0.0
        )


    object_pixels = (
        object_mask > 0
    )


    object_pixel_count = int(
        np.count_nonzero(
            object_pixels
        )
    )


    if object_pixel_count <= 0:

        return (
            False,
            0.0,
            0.0
        )


    # ========================================================
    # RGB EVIDENCE
    # ========================================================

    difference = get_rgb_difference(
        frame
    )


    rgb_changed = (
        (difference > RGB_PIXEL_THRESHOLD)
        &
        object_pixels
    )


    rgb_changed_count = int(
        np.count_nonzero(
            rgb_changed
        )
    )


    rgb_percent = (
        rgb_changed_count
        /
        max(
            object_pixel_count,
            1
        )
        *
        100.0
    )


    # ========================================================
    # DEPTH EVIDENCE
    # ========================================================

    depth = current_depth.astype(
        np.float32
    )


    depth_valid = (
        object_pixels
        &
        np.isfinite(
            baseline_depth
        )
        &
        (
            baseline_depth > 0
        )
        &
        (
            depth > 0
        )
    )


    depth_delta = np.zeros_like(
        depth,
        dtype=np.float32
    )


    depth_delta[
        depth_valid
    ] = (
        baseline_depth[
            depth_valid
        ]
        -
        depth[
            depth_valid
        ]
    )


    depth_changed = (
        (depth_delta > DEPTH_DELTA_MM)
        &
        depth_valid
    )


    depth_valid_count = int(
        np.count_nonzero(
            depth_valid
        )
    )


    depth_changed_count = int(
        np.count_nonzero(
            depth_changed
        )
    )


    depth_percent = (
        depth_changed_count
        /
        max(
            depth_valid_count,
            1
        )
        *
        100.0
    )


    # ========================================================
    # STILL PRESENT?
    #
    # A real object should affect a meaningful part of its own
    # saved mask. Tiny noise is no longer enough.
    # ========================================================

    rgb_present = (
        rgb_percent
        >=
        REMOVE_RGB_MASK_PERCENT_THRESHOLD
    )


    depth_present = (
        depth_valid_count > 20
        and
        depth_percent
        >=
        REMOVE_DEPTH_MASK_PERCENT_THRESHOLD
    )


    still_present = (
        rgb_present
        or
        depth_present
    )


    return (
        still_present,
        rgb_percent,
        depth_percent
    )


# ============================================================
# DEPTHAI PIPELINE
# ============================================================

pipeline = dai.Pipeline()


# ============================================================
# RGB CAMERA
# ============================================================

rgb = pipeline.create(
    dai.node.ColorCamera
)

rgb.setPreviewSize(
    RGB_WIDTH,
    RGB_HEIGHT
)

rgb.setPreviewKeepAspectRatio(
    False
)

rgb.setInterleaved(
    False
)

rgb.setColorOrder(
    dai.ColorCameraProperties.ColorOrder.BGR
)

rgb.setFps(
    FPS
)


# ============================================================
# LEFT MONO
# ============================================================

left = pipeline.create(
    dai.node.MonoCamera
)

left.setResolution(
    dai.MonoCameraProperties
    .SensorResolution
    .THE_400_P
)

left.setBoardSocket(
    dai.CameraBoardSocket.LEFT
)

left.setFps(
    FPS
)


# ============================================================
# RIGHT MONO
# ============================================================

right = pipeline.create(
    dai.node.MonoCamera
)

right.setResolution(
    dai.MonoCameraProperties
    .SensorResolution
    .THE_400_P
)

right.setBoardSocket(
    dai.CameraBoardSocket.RIGHT
)

right.setFps(
    FPS
)


# ============================================================
# STEREO DEPTH
# ============================================================

stereo = pipeline.create(
    dai.node.StereoDepth
)

stereo.setDefaultProfilePreset(
    dai.node.StereoDepth
    .PresetMode
    .HIGH_DENSITY
)

stereo.setExtendedDisparity(
    True
)

stereo.setLeftRightCheck(
    True
)

stereo.setSubpixel(
    False
)

stereo.setDepthAlign(
    dai.CameraBoardSocket.RGB
)

stereo.setOutputSize(
    RGB_WIDTH,
    RGB_HEIGHT
)


left.out.link(
    stereo.left
)

right.out.link(
    stereo.right
)


# ============================================================
# OUTPUTS
# ============================================================

xout_rgb = pipeline.create(
    dai.node.XLinkOut
)

xout_rgb.setStreamName(
    "rgb"
)

rgb.preview.link(
    xout_rgb.input
)


xout_depth = pipeline.create(
    dai.node.XLinkOut
)

xout_depth.setStreamName(
    "depth"
)

stereo.depth.link(
    xout_depth.input
)



# ============================================================
# NN INPUT FROM HOST
# ============================================================

xin_nn = pipeline.create(
    dai.node.XLinkIn
)

xin_nn.setStreamName(
    "nn_input"
)


# ============================================================
# OLD ECOVISION NETWORK
# ============================================================

nn = pipeline.create(
    dai.node.NeuralNetwork
)

nn.setBlobPath(
    BLOB_PATH
)

nn.input.setBlocking(
    False
)

nn.input.setQueueSize(
    2
)

xin_nn.out.link(
    nn.input
)


# ============================================================
# NN OUTPUT
# ============================================================

xout_nn = pipeline.create(
    dai.node.XLinkOut
)

xout_nn.setStreamName(
    "nn"
)

nn.out.link(
    xout_nn.input
)


# ============================================================
# FIND DEVICE
# ============================================================

devices = (
    dai.Device.getAllAvailableDevices()
)


if not devices:

    raise RuntimeError(
        "No OAK-D detected"
    )


device_info = devices[0]


print()
print(
    "===================================="
)
print(
    "WISBIN - STABLE RGB + OLD ECOVISION"
)
print(
    "===================================="
)
print(
    "OAK-D:",
    device_info.getMxId()
)
print()
print(
    "Presence:"
)
print(
    "  RGB + Depth"
)
print()
print(
    "Final segmentation:"
)
print(
    "  Stable RGB primary"
)
print(
    "  Constrained Depth fallback for dark/black objects"
)
print()
print(
    "Moving-object tracking:"
)
print(
    f"  {TRACK_OBJECT_SEC:.1f} seconds before final segmentation"
)
print(
    "  Live RGB bbox follows the object while moving"
)
print()
print(
    "GrabCut: OFF"
)
print(
    "Depth segmentation:"
)
print(
    "  Restricted to final tracked area ONLY"
)
print(
    "Edge segmentation: OFF"
)
print()
print(
    "Classifier:"
)
print(
    "  OLD EcoVision RVC2 blob"
)
print(
    "  Input = isolated RGB object only"
)
print(
    f"  AI frames = {CLASSIFICATION_FRAMES}"
)
print()
print(
    "E = Capture EMPTY baseline"
)
print(
    "P = Re-select chute"
)
print(
    "R = Reset baseline"
)
print(
    "Q = Quit"
)
print()
print(
    "MANUAL LABEL / SAVE TO wisbin_dataset:"
)
print(
    "1 = METAL"
)
print(
    "2 = PLASTIC"
)
print(
    "3 = PAPER"
)
print(
    "4 = OTHER"
)
print(
    "===================================="
)
print()


# ============================================================
# START OAK-D
# ============================================================

with dai.Device(
    pipeline,
    device_info,
    dai.UsbSpeed.HIGH
) as device:


    print(
        "USB:",
        device.getUsbSpeed()
    )


    rgb_queue = device.getOutputQueue(
        name="rgb",
        maxSize=4,
        blocking=False
    )


    depth_queue = device.getOutputQueue(
        name="depth",
        maxSize=4,
        blocking=False
    )


    nn_input_queue = device.getInputQueue(
        name="nn_input"
    )


    nn_output_queue = device.getOutputQueue(
        name="nn",
        maxSize=4,
        blocking=False
    )


    # ========================================================
    # FIRST FRAME + ROI
    # ========================================================

    first_frame = (
        rgb_queue
        .get()
        .getCvFrame()
    )


    chute_points = load_chute_roi(
        first_frame.shape
    )


    if chute_points is None:

        chute_points = select_chute_polygon(
            first_frame
        )

    else:

        print()
        print(
            "Loaded saved BLACK CHUTE ROI."
        )
        print(
            "Press P if camera moved."
        )
        print()


    chute_mask = create_chute_mask(
        first_frame.shape,
        chute_points
    )


    last_raw_label = None
    last_confidence = 0.0
    last_wisbin_class = None
    last_ai_square = None


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:


        frame = (
            rgb_queue
            .get()
            .getCvFrame()
        )


        depth = (
            depth_queue
            .get()
            .getFrame()
        )


        display = (
            frame.copy()
        )


        draw_chute_polygon(
            display,
            chute_points
        )


        # ====================================================
        # BASELINE NOT READY
        # ====================================================

        if (
            baseline_rgb_color is None
            or
            baseline_gray is None
            or
            baseline_depth is None
        ):

            cv2.putText(
                display,
                "EMPTY BASELINE NOT SET",
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.putText(
                display,
                "Empty chute -> Press E",
                (
                    15,
                    60
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255
                ),
                2
            )


        # ====================================================
        # PRESENCE
        # ====================================================

        else:

            (
                rgb_detected,
                rgb_percent,
                rgb_presence_mask
            ) = detect_rgb_presence(
                frame
            )


            (
                depth_detected,
                depth_percent,
                depth_presence_mask
            ) = detect_depth_presence(
                depth
            )


            raw_object = (
                rgb_detected
                or
                depth_detected
            )


            # =================================================
            # BEFORE CLASSIFICATION:
            # normal full-chute presence detection
            # =================================================

            if not object_already_segmented:

                if raw_object:

                    object_frames += 1
                    empty_frames = 0

                else:

                    empty_frames += 1
                    object_frames = 0


                if (
                    object_frames
                    >=
                    CONFIRM_FRAMES
                ):

                    object_confirmed = True


            # =================================================
            # AFTER CLASSIFICATION:
            # monitor ONLY the previous object's exact final mask.
            #
            # This exact-mask check is the important fix for the green rectangle.
            # =================================================

            else:

                (
                    still_in_last_mask,
                    remove_rgb_percent,
                    remove_depth_percent
                ) = check_last_object_mask(
                    frame,
                    depth,
                    last_object_mask
                )


                if still_in_last_mask:

                    remove_confirm_count = 0

                else:

                    remove_confirm_count += 1


                cv2.putText(
                    display,
                    (
                        f"RemoveMask RGB "
                        f"{remove_rgb_percent:.1f}% | "
                        f"D {remove_depth_percent:.1f}%"
                    ),
                    (
                        15,
                        185
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (
                        255,
                        255,
                        255
                    ),
                    1,
                    cv2.LINE_AA
                )


            # =================================================
            # OBJECT REMOVED
            # =================================================

            normal_empty_confirmed = (
                not object_already_segmented
                and
                empty_frames
                >=
                RELEASE_FRAMES
            )


            classified_object_removed = (
                object_already_segmented
                and
                remove_confirm_count
                >=
                REMOVE_CONFIRM_FRAMES
            )


            if (
                normal_empty_confirmed
                or
                classified_object_removed
            ):

                if (
                    object_confirmed
                    or
                    object_already_segmented
                ):

                    print()
                    print(
                        "OBJECT REMOVED - READY FOR NEXT ITEM."
                    )
                    print()


                object_confirmed = False

                object_already_segmented = False

                object_frames = 0
                empty_frames = 0
                remove_confirm_count = 0


                # Clear the green box immediately.
                last_bbox = None

                # Forget previous object's exact mask.
                last_object_mask = None


                # Clear previous AI result.
                last_raw_label = None
                last_confidence = 0.0
                last_wisbin_class = None
                last_ai_square = None


                # Close tracking windows if still open.
                try:

                    cv2.destroyWindow(
                        "OBJECT TRACKING - MOVING ITEM"
                    )

                except cv2.error:

                    pass


                try:

                    cv2.destroyWindow(
                        "TRACKING RGB MASK"
                    )

                except cv2.error:

                    pass


            # =================================================
            # SEGMENT ONCE
            # =================================================

            if (
                object_confirmed
                and
                not object_already_segmented
            ):

                (
                    segmentation_frame,
                    stable_rgb,
                    stable_depth,
                    tracked_depth_region,
                    final_mask,
                    square,
                    transparent,
                    bbox,
                    used_depth_fallback
                ) = segment_current_object(
                    rgb_queue,
                    depth_queue
                )


                # ---------------------------------------------
                # Show the mask that already worked well
                # ---------------------------------------------

                if stable_rgb is not None:

                    cv2.imshow(
                        "1 - STABLE RGB CHANGE",
                        stable_rgb
                    )


                if stable_depth is not None:

                    cv2.imshow(
                        "2 - CONSTRAINED STABLE DEPTH",
                        stable_depth
                    )


                if tracked_depth_region is not None:

                    cv2.imshow(
                        "3 - DEPTH ALLOWED REGION",
                        tracked_depth_region
                    )


                if final_mask is not None:

                    cv2.imshow(
                        "4 - FINAL HYBRID MASK",
                        final_mask
                    )


                if square is not None:

                    preview = cv2.resize(
                        square,
                        (
                            400,
                            400
                        ),
                        interpolation=cv2.INTER_LINEAR
                    )

                    cv2.imshow(
                        "5 - OBJECT ONLY - FUTURE MODEL INPUT",
                        preview
                    )


                    last_bbox = bbox

                    # Save the EXACT final mask of this object.
                    # Removal detection will monitor only these pixels.
                    last_object_mask = final_mask.copy()

                    # New classified object starts with fresh removal state.
                    remove_confirm_count = 0

                    (
                        last_raw_label,
                        last_confidence,
                        last_wisbin_class,
                        last_ai_square
                    ) = classify_isolated_object(
                        rgb_queue,
                        nn_input_queue,
                        nn_output_queue,
                        final_mask,
                        bbox
                    )

                    object_already_segmented = True


                else:

                    print()
                    print(
                        "Could not isolate object."
                    )
                    print(
                        "Will retry while object remains."
                    )
                    print()


            # =================================================
            # STATUS
            # =================================================

            cv2.putText(
                display,
                (
                    f"RGB: "
                    f"{rgb_percent:.2f}% "
                    f"{'YES' if rgb_detected else 'NO'}"
                ),
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            cv2.putText(
                display,
                (
                    f"Depth: "
                    f"{depth_percent:.2f}% "
                    f"{'YES' if depth_detected else 'NO'}"
                ),
                (
                    15,
                    58
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            if object_confirmed:

                status = (
                    "OBJECT DETECTED"
                )

            else:

                status = (
                    "EMPTY"
                )


            cv2.putText(
                display,
                status,
                (
                    15,
                    90
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            # =================================================
            # AI RESULT
            # =================================================

            if last_wisbin_class is not None:

                cv2.putText(
                    display,
                    (
                        f"AI: "
                        f"{last_raw_label.upper()} "
                        f"{last_confidence * 100:.1f}%"
                    ),
                    (
                        15,
                        125
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )

                cv2.putText(
                    display,
                    (
                        f"WISBIN: "
                        f"{last_wisbin_class}"
                    ),
                    (
                        15,
                        155
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


            # =================================================
            # DRAW FINAL RGB BBOX
            # =================================================

            if last_bbox is not None:

                x1, y1, x2, y2 = (
                    last_bbox
                )


                cv2.rectangle(
                    display,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    (
                        0,
                        255,
                        0
                    ),
                    3
                )


                cv2.putText(
                    display,
                    "RGB OBJECT ONLY",
                    (
                        x1,
                        max(
                            20,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


        # ====================================================
        # MANUAL LABEL HINT
        # ====================================================

        cv2.putText(
            display,
            "SAVE: 1 METAL | 2 PLASTIC | 3 PAPER | 4 OTHER",
            (
                15,
                RGB_HEIGHT - 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # MAIN WINDOW
        # ====================================================

        cv2.imshow(
            "WISBIN - RGB OBJECT ISOLATION",
            display
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # ====================================================
        # E - EMPTY BASELINE
        # ====================================================

        if key == ord("e"):

            (
                baseline_rgb_color,
                baseline_gray,
                baseline_depth
            ) = build_baseline(
                rgb_queue,
                depth_queue
            )


            object_frames = 0

            empty_frames = 0

            object_confirmed = False

            object_already_segmented = False

            remove_confirm_count = 0

            last_bbox = None
            last_object_mask = None

            last_raw_label = None
            last_confidence = 0.0
            last_wisbin_class = None
            last_ai_square = None


        # ====================================================
        # P - NEW CHUTE ROI
        # ====================================================

        elif key == ord("p"):

            baseline_rgb_color = None

            baseline_gray = None

            baseline_depth = None


            chute_points = select_chute_polygon(
                frame
            )


            chute_mask = create_chute_mask(
                frame.shape,
                chute_points
            )


            object_frames = 0

            empty_frames = 0

            object_confirmed = False

            object_already_segmented = False

            remove_confirm_count = 0

            last_bbox = None
            last_object_mask = None

            last_raw_label = None
            last_confidence = 0.0
            last_wisbin_class = None
            last_ai_square = None


            print()
            print(
                "ROI UPDATED."
            )
            print(
                "Empty chute and press E."
            )
            print()


        # ====================================================
        # R - RESET
        # ====================================================

        elif key == ord("r"):

            baseline_rgb_color = None

            baseline_gray = None

            baseline_depth = None


            object_frames = 0

            empty_frames = 0

            object_confirmed = False

            object_already_segmented = False

            remove_confirm_count = 0

            last_bbox = None
            last_object_mask = None

            last_raw_label = None
            last_confidence = 0.0
            last_wisbin_class = None
            last_ai_square = None


            print()
            print(
                "BASELINE RESET."
            )
            print(
                "Empty chute and press E."
            )
            print()


        # ====================================================
        # 1 / 2 / 3 / 4 - MANUAL CORRECT LABEL + SAVE
        # ====================================================

        elif key in MANUAL_LABEL_KEYS:

            correct_class = MANUAL_LABEL_KEYS[
                key
            ]

            save_manual_training_image(
                last_ai_square,
                correct_class,
                predicted_label=last_raw_label,
                confidence=last_confidence
            )


        # ====================================================
        # Q - QUIT
        # ====================================================

        elif key == ord("q"):

            break


cv2.destroyAllWindows()
