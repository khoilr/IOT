import cv2


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []

    # if there are more than 5 non working ports stop the testing.
    while len(non_working_ports) < 6:
        camera = cv2.VideoCapture(dev_port)

        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


def draw_boundary(img, boundaries, color, thickness):
    # Draw a rectangle on the image
    pt1 = (int(boundaries[0]), int(boundaries[1]))  # Top-left corner
    pt2 = (int(boundaries[2]), int(boundaries[3]))  # Bottom-right corner
    cv2.rectangle(img, pt1, pt2, color, thickness)


def draw_text(img, text, org, font_scale, color, thickness):
    # Draw text on the image
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_points(img, points, color, radius):
    # Draw points on the image
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), radius, color, -1)


if __name__ == "__main__":
    ports = list_ports()

    print("Available ports: ", ports[0])
    print("Working ports: ", ports[1])
    print("Non working ports: ", ports[2])
