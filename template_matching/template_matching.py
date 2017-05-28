import cv2
from matplotlib import pyplot as plt

# http://docs.opencv.org/3.2.0/d4/dc6/tutorial_py_template_matching.html

img_filename = "../testdata/lebron_template.jpg"
template_filename = "../testdata/lebron_face.jpg"

img = cv2.imread(img_filename,0)
img2 = img.copy()

template = cv2.imread(template_filename,0)
w, h = template.shape[::-1]

# 手法をあとで文字列としてプロットしたいので文字列として持っておく
methods = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR",
            "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]

for m in methods:
    # 下でこの画像にrectangleを描画するのでコピーしておく必要がある
    img = img2.copy()

    # 手法をただの文字列から評価して，数値にする
    method = eval(m)

    # テンプレートマッチングする
    res = cv2.matchTemplate(img, template, method)

    # res配列の中から最大最小の値と位置を求める
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 左上の座標は，TM_SQDIFF, TM_SQDIFF_NORMED のときは min_loc, それ以外は max_locとなる
    top_left = min_loc if (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]) else max_loc

    # 右下の座標は, 左上の座標に画像の幅と高さを足したもの
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # imgに四角を描画する
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # 画像をプロットする
    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap="gray")
    plt.title("Detected Point"), plt.xticks([]), plt.yticks([])
    plt.suptitle(m)
    plt.show()
