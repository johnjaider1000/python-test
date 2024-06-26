import cv2


def get_grouped_lanes(lanes):
    grouped_lanes = []

    for lane in lanes:
        print("Group Name:", lane["title"])
        laneGroup = {"id": lane["id"], "title": lane["title"], "lines": []}
        for line in lane["lines"]:
            laneGroup["lines"].append(
                {
                    "x1": line["coors"]["x1"],
                    "y1": line["coors"]["y1"],
                    "x2": line["coors"]["x2"],
                    "y2": line["coors"]["y2"],
                    "color": line["coors"]["color"],
                    "type": line["type"],
                }
            )

        grouped_lanes.append(laneGroup)


def list_connected_cameras():
    # Número máximo de dispositivos de video permitidos
    max_devices = 10

    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        print(
            f"Cámara {i}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        cap.release()


list_connected_cameras()
