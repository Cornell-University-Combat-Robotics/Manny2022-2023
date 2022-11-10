

def headshot_ratio(x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
    hs_area = (((x1 * y2) - (x2 * y1)) + ((x2 * y3) - (x3 * y2)) + ((x3*y4) - (x4 * y3)) + ((x4 * y1) - (x1 * y4))) / 2

    img_area = img_width * img_height

    return hs_area/img_area
