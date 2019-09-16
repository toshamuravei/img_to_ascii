import sys
import numpy as np
from PIL import Image


BETA = 2.0
EPSILON = 1e-6
N_ITERATIONS = 15
THRESHOLD = 0.00
A_STR = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
DEFAULT_ASCII = "ascii_monospace.png"
DEFAULT_IMG = "Mickey-Mouse-2.jpg"
GREY_IGNORE_RATE = 60
GLYPH_HEIGHT = 24
GLYPH_WIDTH = 11

def zero_reminder(numerator, denominator):
    """
    Returns numerator if it could be clearly divided with
    given denominator. If not, func increases numerator &
    returns increased (clearly divisible) number.
    """
    modulo = numerator % denominator
    if modulo != 0:
        numerator += denominator - modulo
    return numerator


def process_image(img_addr=None,
                  grey_ignore_rate=GREY_IGNORE_RATE, show_processed=False):
    img_addr = img_addr or DEFAULT_IMG
    img = Image.open(img_addr).convert("L")
    pixels = np.array(img)

    with np.nditer(pixels, op_flags=["readwrite"]) as nd_iter:
        for x in nd_iter:
            x[...] = 0 if x >= grey_ignore_rate else 255

    height, width = pixels.shape

    if show_processed:
        Image.fromarray(pixels).show()

    return pixels


def get_ascii_array():
    separate_ascii = []
    low = 10
    ascii_np = process_image(DEFAULT_ASCII, grey_ignore_rate=200)

    for i in range(0, 95):
        high = low + 11
        separate_ascii.append(ascii_np[:, low:high])
        low = high

    separate_ascii.insert(0, separate_ascii[-1])
    del separate_ascii[-1]

    return separate_ascii


def fill_imgarr_with_zeros(img_arr, axis_name=None):

    axis_map = {
        "width": {
            "append_axis": 1,
            "filler_arr": np.array([[0.0] for x in range(0, img_arr.shape[0])])
        },
        "height": {
            "append_axis": 0,
            "filler_arr": np.array([[0.0 for x in range(0, img_arr.shape[1])]])
        }}

    axis = axis_map.get(axis_name)

    if axis is None:
        err_msg = "Can't recognise axis! Please specify it from this: {}!"
        raise AttributeError(err_msg.format(tuple(axis_map.keys())))

    extended = np.append(img_arr, axis["filler_arr"], axis=axis["append_axis"])

    return extended


def resize_imgarr_if_needed(img_arr):
    img_height, img_width = img_arr.shape

    width = zero_reminder(img_width, GLYPH_WIDTH)
    height = zero_reminder(img_height, GLYPH_HEIGHT)

    if img_width != width:
        for i in range(0, width - img_width):
            img_arr = fill_imgarr_with_zeros(img_arr, "width")
    if img_height != height:
        for i in range(0, height - img_height):
            img_arr = fill_imgarr_with_zeros(img_arr, "height")

    return img_arr


def split_imgarr_into_glyphblocks(img_arr):
    img_arr = resize_imgarr_if_needed(img_arr)
    h, w = img_arr.shape
    nrows, ncols = GLYPH_HEIGHT, GLYPH_WIDTH

    splitted = img_arr.reshape(h//nrows, nrows, -1, ncols)\
                      .swapaxes(1, 2).reshape(-1, nrows, ncols)

    return splitted


def flat_and_normalize(splitted_arr):
    flatted = splitted_arr.flatten()

    norm = np.linalg.norm(flatted)
    normalized = flatted / norm

    return normalized


def get_matrix(mtx, glyph_arr, img_splitted):
    if mtx not in ("V", "W", "H"):
        raise AttributeError("Please, specify matrix: V, W ot H")

    m_lmb = lambda s: np.column_stack(tuple([flat_and_normalize(x) for x in s]))

    base_to_operate = {
        "V": m_lmb(img_splitted),
        "W": m_lmb(glyph_arr),
        "H": np.random.rand(len(glyph_arr), len(img_splitted))}

    return base_to_operate[mtx]


def update_H_mtx(v_mtx, w_mtx, h_mtx):
    v_aprx = w_mtx @ h_mtx

    for h_row in range(0, h_mtx.shape[0]):
        for h_col in range(0, h_mtx.shape[1]):
            nmrt = 0.0
            dnmrt = 0.0

            for w_row in range(0, w_mtx.shape[0]):
                if abs(v_aprx[w_row, h_col]) > EPSILON:

                    nmrt += w_mtx[w_row, h_row] * v_mtx[w_row, h_col] \
                        / v_aprx[w_row, h_col] ** (2.0 - BETA)

                    dnmrt += w_mtx[w_row, h_row] * v_aprx[w_row, h_col] \
                        ** (BETA - 1.0)
                else:
                    nmrt += w_mtx[w_row, h_row] * v_mtx[w_row, h_col]
                    if BETA - 1.0 > 0.0:
                        dnmrt += w_mtx[w_row, h_row] * v_aprx[w_row, h_col] \
                            ** (BETA - 1.0)
                    else:
                        dnmrt += w_mtx[w_row, h_row]

            if abs(dnmrt) > EPSILON:
                h_mtx[h_row, h_col] *= nmrt / dnmrt
            else:
                h_mtx[h_row, h_col] *= nmrt


def escaping_map(char):
    charmap = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"}

    return charmap.get(char, char)


def represent_as_ascii(h_mtx, vert_chars, hor_chars):
    result = np.zeros(shape=(vert_chars, hor_chars), dtype=str)
    p_holder = " "
    maximums = np.argmax(h_mtx, axis=0)

    for h_col in range(0, h_mtx.shape[1]):
        max_row = maximums[h_col]
        glyph = A_STR[max_row]
        to_paste = glyph if h_mtx[max_row, h_col] > THRESHOLD else p_holder

        result[int(h_col / hor_chars), h_col % hor_chars] = to_paste

    return result


def write_to_html(result):

    html_file.write("<html>")
    html_file.write("<font face=\"courier\"><pre>")
    html_file.write("<body>")
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            to_paste = escaping_map(str(result[i, j]))
            html_file.write(to_paste)
        html_file.write("\n")

    html_file.write("</body>")
    html_file.write("</html>")


if __name__ == "__main__":
    glyph_arr = get_ascii_array()
    img_arr = process_image()
    splitted_imgarr = split_imgarr_into_glyphblocks(img_arr)
    v_mtx = get_matrix("V", glyph_arr, splitted_imgarr)
    w_mtx = get_matrix("W", glyph_arr, splitted_imgarr)
    h_mtx = get_matrix("H", glyph_arr, splitted_imgarr)
    n_glyphs_hor = int(zero_reminder(img_arr.shape[1], GLYPH_WIDTH) / GLYPH_WIDTH)
    n_glyphs_vert = int(zero_reminder(img_arr.shape[0], GLYPH_HEIGHT) / GLYPH_HEIGHT)
    for i in range(0, N_ITERATIONS):
        if i % 5 == 0:
            print("%r iterations are made!" % i)
        update_H_mtx(v_mtx, w_mtx, h_mtx)

    result = represent_as_ascii(h_mtx, n_glyphs_vert, n_glyphs_hor)
    write_to_html(result)
